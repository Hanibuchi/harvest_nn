# ============================================================================
# このファイルは何をするものか
# ----------------------------------------------------------------------------
# 【(C) 計測用スクリプト（このプロジェクトのメイン）】です。
#
# Xie et al. (2024) の Figure 6 上段にある「集中型計算方式
# (Centralized Computing Scheme)」を再現し、次の5つの処理ステップを
# 1台のマシンの上で並列化せずに順番に実行して、それぞれの時間を測ります。
#
#   ステップ1  画像取得        : カラー画像と深度データ（点群）をファイルから読む
#   ステップ2  深度アライメント: 点群をカラー画像の座標系に投影して深度画像を作る
#   ステップ3  推論前処理      : タイル分割・リサイズ・色変換・正規化・テンソル化
#   ステップ4a 画像推論        : ニューラルネットワークの順伝播（NMSは含まない）
#   ステップ4b 後処理(NMS)     : 重なった検出枠を1つにまとめる処理
#   ステップ5  3D空間位置推定  : 検出枠と深度からりんごの3次元座標を求める
#
# ステップ4は「推論そのもの」と「後処理(NMS)」を分けて測れるようにしてあります。
# 合計時間はCSVの total_ms 列で確認できます。
#
# 【タイル分割について】
#   学習には 548x373 の切り出し画像を使っています。
#   もし 1920x1080 の生画像をそのまま 640 に縮めて推論すると、
#   りんごの見かけの大きさが学習時の約3分の1になってしまい、
#   検出精度が大きく落ちます。
#   そこで生画像を学習時と同じ 548x373 の大きさのタイル9枚に切り分け、
#   1枚ずつ推論します（バッチサイズは1固定）。
#   タイルの位置や枚数は camera_params.yaml の tiling で変更できます。
#
# 【正しく時間を測るための工夫】
#   ・時刻は time.perf_counter() で取る（精度が高く、経過時間の測定に適している）
#   ・GPUを使う処理の前後で torch.cuda.synchronize() を呼ぶ
#     （GPUは非同期に動くため、これをしないと実際より短い時間が出てしまう）
#   ・最初の数回はウォームアップとして計測から除外する
#     （1回目はCUDAの初期化やメモリ確保が入り、極端に遅くなるため）
#   ・同じ画像を複数回処理し、平均・中央値・標準偏差を出す
#   ・バッチサイズは1に固定する（ロボットのリアルタイム処理を想定しているため）
#
# 【実行例】
#   python src/measure_centralized.py \
#       --weights outputs/weights/yolov8n_apple_best.pt \
#       --machine-name workstation
#
#   python src/measure_centralized.py \
#       --weights outputs/weights/yolov8n_apple_best.pt \
#       --machine-name jetson_orin --repeats 20 --warmup 5
# ============================================================================

# コマンドライン引数を扱うための標準ライブラリ
import argparse
# ファイルやフォルダを操作するための標準ライブラリ
import os
# CSVファイルを書き出すための標準ライブラリ
import csv
# JSONファイルを書き出すための標準ライブラリ
import json

# 数値計算ライブラリ
import numpy as np
# 画像処理ライブラリ
import cv2
# ディープラーニングのライブラリ
import torch

# YOLO の後処理(NMS)の関数を読み込む。
# ultralytics のバージョンによって置き場所が違うため、両方を試す。
# （関数の読み込みは時間計測の外で済ませておく。計測したいのは処理そのものであって、
#   ライブラリの読み込み時間ではないため）
try:
    # ultralytics 8.4 以降はこちら
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    # それより古いバージョンはこちら
    from ultralytics.utils.ops import non_max_suppression
# タイルをまたいだ枠の統合に使う NMS を読み込む
from torchvision.ops import nms as torchvision_nms

# 自作のモジュールを読み込む
import common_paths
import kinect_io
import depth_alignment
import localization_3d
import timing_utils
import machine_info


# --------------------------------------------------------------------------
# タイル分割の準備
# --------------------------------------------------------------------------

def build_tile_list(tiling_settings, image_width, image_height):
    """
    生画像を切り分けるタイルの位置を計算して、リストにして返す。

    返り値は (左端x, 上端y, 幅, 高さ) の組のリストです。
    タイルが画像の外にはみ出す場合は、画像の内側に収まるように位置をずらします。
    """
    # 設定からタイル1枚の幅を取り出す
    tile_width = int(tiling_settings["tile_width"])
    # 設定からタイル1枚の高さを取り出す
    tile_height = int(tiling_settings["tile_height"])
    # 設定から横に何枚並べるかを取り出す
    number_of_columns = int(tiling_settings["columns"])
    # 設定から縦に何枚並べるかを取り出す
    number_of_rows = int(tiling_settings["rows"])
    # 設定から横方向の移動量を取り出す
    stride_x = int(tiling_settings["stride_x"])
    # 設定から縦方向の移動量を取り出す
    stride_y = int(tiling_settings["stride_y"])
    # 設定からタイルを並べ始める左上の位置を取り出す
    origin_x = int(tiling_settings["origin_x"])
    origin_y = int(tiling_settings["origin_y"])

    # 結果を入れるための空のリストを用意する
    tile_list = []
    # 縦方向に1行ずつ見ていく
    for row_index in range(number_of_rows):
        # 横方向に1列ずつ見ていく
        for column_index in range(number_of_columns):
            # このタイルの左端の位置を計算する
            tile_x = origin_x + column_index * stride_x
            # このタイルの上端の位置を計算する
            tile_y = origin_y + row_index * stride_y
            # 右にはみ出す場合は、画像の右端に合わせて左に寄せる
            if tile_x + tile_width > image_width:
                tile_x = image_width - tile_width
            # 下にはみ出す場合は、画像の下端に合わせて上に寄せる
            if tile_y + tile_height > image_height:
                tile_y = image_height - tile_height
            # 左にはみ出す場合は、0に合わせる
            if tile_x < 0:
                tile_x = 0
            # 上にはみ出す場合は、0に合わせる
            if tile_y < 0:
                tile_y = 0
            # 計算した位置と大きさをリストに追加する
            tile_list.append((tile_x, tile_y, tile_width, tile_height))
    # 完成したリストを返す
    return tile_list


def compute_letterbox_parameters(source_width, source_height, model_input_size):
    """
    タイルをモデルの入力サイズに合わせるときの「縮小率」と「余白の量」を計算する。

    YOLO は正方形の画像を入力に取りますが、タイルは横長です。
    そのまま正方形に引き伸ばすと形が歪んでしまうので、
    縦横比を保ったまま縮小し、余った部分を灰色で埋めます。
    この方法を「レターボックス」と呼びます。

    返り値は (縮小率, 左の余白, 上の余白, 縮小後の幅, 縮小後の高さ) です。
    """
    # 横方向の縮小率を計算する
    scale_for_width = model_input_size / source_width
    # 縦方向の縮小率を計算する
    scale_for_height = model_input_size / source_height
    # 縦横比を保つため、小さい方の縮小率を採用する
    if scale_for_width < scale_for_height:
        scale_ratio = scale_for_width
    else:
        scale_ratio = scale_for_height
    # 縮小後の幅を計算する（小数点以下は四捨五入する）
    resized_width = int(round(source_width * scale_ratio))
    # 縮小後の高さを計算する
    resized_height = int(round(source_height * scale_ratio))
    # 左右に入れる余白の合計を計算し、その半分を左の余白とする
    padding_left = (model_input_size - resized_width) // 2
    # 上下に入れる余白の合計を計算し、その半分を上の余白とする
    padding_top = (model_input_size - resized_height) // 2
    # 計算した値をまとめて返す
    return scale_ratio, padding_left, padding_top, resized_width, resized_height


# --------------------------------------------------------------------------
# ステップ1〜5の本体。1つのステップが1つの関数になっています。
# --------------------------------------------------------------------------

def step1_image_acquisition(color_image_path, point_cloud_npy_path):
    """
    【ステップ1：画像取得】カラー画像と深度データをファイルから読み込む。

    実際のロボットではカメラから受け取る部分にあたります。
    ここではデータセットのファイルから読み込む時間を測ります。
    """
    # カラー画像を読み込む
    color_image_bgr = kinect_io.load_color_image(color_image_path)
    # 点群（深度データ）を読み込む
    point_cloud_array = kinect_io.load_point_cloud_from_npy(point_cloud_npy_path)
    # 2つをまとめて返す
    return color_image_bgr, point_cloud_array


def step2_depth_alignment(point_cloud_array, alignment_transform):
    """
    【ステップ2：深度アライメント】点群をカラー画像の座標系に投影する。

    中身の計算は depth_alignment.py に書いてあります。
    """
    # 点群を投影して深度画像を作る
    depth_image_meters = depth_alignment.align_depth_to_color(
        point_cloud_array, alignment_transform
    )
    # できた深度画像を返す
    return depth_image_meters


def step3_inference_preprocessing(color_image_bgr, tile_list, model_input_size,
                                  device, use_half_precision):
    """
    【ステップ3：推論前処理】画像をモデルに入力できる形に整える。

    1枚のタイルごとに、次の処理を行います。
        1. 生画像からタイルの範囲を切り出す
        2. 縦横比を保ったまま縮小し、余白を足して正方形にする（レターボックス）
        3. 色の並びを BGR から RGB に変える
        4. 配列の並びを (高さ,幅,色) から (色,高さ,幅) に変える
        5. 0〜255 の整数を 0.0〜1.0 の小数に変える（正規化）
        6. PyTorch のテンソルに変換し、GPU（使う場合）に転送する
    """
    # 結果のテンソルを入れるための空のリストを用意する
    input_tensor_list = []
    # タイルを1枚ずつ順番に処理する
    for tile_index in range(len(tile_list)):
        # このタイルの位置と大きさを取り出す
        tile_x, tile_y, tile_width, tile_height = tile_list[tile_index]
        # 生画像からタイルの範囲を切り出す
        tile_image_bgr = color_image_bgr[
            tile_y:tile_y + tile_height, tile_x:tile_x + tile_width
        ]
        # レターボックスに必要な縮小率と余白を計算する
        scale_ratio, padding_left, padding_top, resized_width, resized_height = (
            compute_letterbox_parameters(tile_width, tile_height, model_input_size)
        )
        # タイルを縮小する（INTER_LINEAR は速度と品質のバランスが良い補間方法）
        resized_tile_bgr = cv2.resize(
            tile_image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        # モデル入力と同じ大きさの、灰色(値114)で埋めた入れ物を用意する
        # （114 は YOLO が標準で使っている余白の色）
        padded_image_bgr = np.full(
            (model_input_size, model_input_size, 3), 114, dtype=np.uint8
        )
        # 縮小したタイルを、余白の内側に貼り付ける
        padded_image_bgr[
            padding_top:padding_top + resized_height,
            padding_left:padding_left + resized_width,
        ] = resized_tile_bgr
        # 色の並びを BGR から RGB に変える
        padded_image_rgb = cv2.cvtColor(padded_image_bgr, cv2.COLOR_BGR2RGB)
        # 配列の並びを (高さ,幅,色) から (色,高さ,幅) に入れ替える
        transposed_image = np.transpose(padded_image_rgb, (2, 0, 1))
        # メモリ上の並びを連続にする（このあとのテンソル変換が速くなる）
        contiguous_image = np.ascontiguousarray(transposed_image)
        # numpy の配列を PyTorch のテンソルに変換する
        image_tensor = torch.from_numpy(contiguous_image)
        # 指定した計算装置（GPUまたはCPU）に転送する
        image_tensor = image_tensor.to(device)
        # 整数から小数に変換する（半精度を使う設定ならfloat16、そうでなければfloat32）
        if use_half_precision:
            image_tensor = image_tensor.half()
        else:
            image_tensor = image_tensor.float()
        # 0〜255 の値を 255 で割って 0.0〜1.0 にする（正規化）
        image_tensor = image_tensor / 255.0
        # バッチの次元を先頭に足して (1, 色, 高さ, 幅) の形にする
        # （バッチサイズは1に固定。ロボットは画像を1枚ずつ処理するため）
        image_tensor = image_tensor.unsqueeze(0)
        # できたテンソルをリストに追加する
        input_tensor_list.append(image_tensor)
    # 全タイル分のテンソルのリストを返す
    return input_tensor_list


def step4a_image_inference(detection_model, input_tensor_list):
    """
    【ステップ4a：画像推論】ニューラルネットワークの順伝播だけを行う。

    ここでは NMS（重なった枠をまとめる後処理）は行いません。
    タイル1枚ずつ、バッチサイズ1で推論します。
    """
    # 結果を入れるための空のリストを用意する
    raw_prediction_list = []
    # 勾配の計算を行わないモードにする（推論では不要で、その分速くメモリも節約できる）
    # torch.inference_mode() というさらに速いモードもありますが、
    # そちらで作ったテンソルは後段のNMSが書き換えられないため、no_grad を使います
    with torch.no_grad():
        # タイルのテンソルを1つずつ順番に推論する
        for tensor_index in range(len(input_tensor_list)):
            # 1枚分のテンソルをモデルに通す
            model_output = detection_model(input_tensor_list[tensor_index])
            # モデルの出力が複数個の組で返る場合は、先頭の要素が検出結果になる
            if isinstance(model_output, (list, tuple)):
                raw_prediction = model_output[0]
            else:
                raw_prediction = model_output
            # 結果をリストに追加する
            raw_prediction_list.append(raw_prediction)
    # 全タイル分の結果のリストを返す
    return raw_prediction_list


def step4b_postprocess_nms(raw_prediction_list, tile_list, model_input_size,
                           confidence_threshold, iou_threshold, merge_iou_threshold):
    """
    【ステップ4b：後処理(NMS)】重なった検出枠を1つにまとめ、元の画像の座標に戻す。

    NMS は Non-Maximum Suppression（非最大値抑制）の略で、
    同じりんごに対して何個も出てしまった枠のうち、
    一番自信度の高いものだけを残す処理です。

    このプロジェクトではタイルに分けて推論しているので、
    次の2段階の処理を行います。
        1. タイルごとに NMS を行い、枠を元のタイルの座標に戻す
        2. タイルの位置を足して生画像の座標にし、
           タイルの重なり部分で二重に検出された枠をもう一度 NMS でまとめる
    """
    # 生画像の座標に直した枠を入れるためのリストを用意する
    all_boxes_list = []
    # 枠ごとの自信度を入れるためのリストを用意する
    all_scores_list = []

    # タイルを1枚ずつ順番に処理する
    for tile_index in range(len(raw_prediction_list)):
        # このタイルの推論結果を取り出す
        raw_prediction = raw_prediction_list[tile_index]
        # NMS を行い、残った枠だけを取り出す
        # 返り値は画像ごとのリストで、今回はバッチサイズ1なので先頭だけを使う
        nms_result_list = non_max_suppression(
            raw_prediction, confidence_threshold, iou_threshold
        )
        # 1枚分の結果を取り出す。形は (枠の数, 6) で、
        # 6つの値は [左端x, 上端y, 右端x, 下端y, 自信度, クラス番号]
        detections_for_this_tile = nms_result_list[0]
        # 枠が1つも無ければ、次のタイルに進む
        if detections_for_this_tile.shape[0] == 0:
            continue

        # このタイルの位置と大きさを取り出す
        tile_x, tile_y, tile_width, tile_height = tile_list[tile_index]
        # レターボックスの縮小率と余白を計算する（前処理と同じ計算）
        scale_ratio, padding_left, padding_top, resized_width, resized_height = (
            compute_letterbox_parameters(tile_width, tile_height, model_input_size)
        )

        # 枠の座標部分（最初の4列）だけを取り出す
        box_coordinates = detections_for_this_tile[:, 0:4]
        # 自信度（5列目）を取り出す
        confidence_scores = detections_for_this_tile[:, 4]

        # 余白の分を引いて、縮小後の画像での座標に戻す
        box_coordinates[:, 0] = box_coordinates[:, 0] - padding_left
        box_coordinates[:, 2] = box_coordinates[:, 2] - padding_left
        box_coordinates[:, 1] = box_coordinates[:, 1] - padding_top
        box_coordinates[:, 3] = box_coordinates[:, 3] - padding_top
        # 縮小率で割って、元のタイルの大きさでの座標に戻す
        box_coordinates = box_coordinates / scale_ratio
        # タイルの左上の位置を足して、生画像全体での座標にする
        box_coordinates[:, 0] = box_coordinates[:, 0] + tile_x
        box_coordinates[:, 2] = box_coordinates[:, 2] + tile_x
        box_coordinates[:, 1] = box_coordinates[:, 1] + tile_y
        box_coordinates[:, 3] = box_coordinates[:, 3] + tile_y

        # このタイルの結果を、全体のリストに追加する
        all_boxes_list.append(box_coordinates)
        all_scores_list.append(confidence_scores)

    # どのタイルでも何も検出されなかった場合は、空の配列を返す
    if len(all_boxes_list) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # 全タイルの枠を1つの大きなテンソルにつなげる
    merged_boxes = torch.cat(all_boxes_list, dim=0)
    # 全タイルの自信度も1つにつなげる
    merged_scores = torch.cat(all_scores_list, dim=0)
    # タイルの重なり部分で二重に検出された枠を、もう一度NMSでまとめる
    kept_indices = torchvision_nms(merged_boxes, merged_scores, merge_iou_threshold)
    # 残った枠だけを取り出す
    final_boxes = merged_boxes[kept_indices]
    # 残った枠の自信度だけを取り出す
    final_scores = merged_scores[kept_indices]
    # GPU上にある場合はCPUに移し、numpy の配列に変換して返す
    return final_boxes.detach().cpu().numpy(), final_scores.detach().cpu().numpy()


def step5_spatial_localization(depth_image_meters, detection_boxes, localization_settings):
    """
    【ステップ5：3D空間位置推定】検出枠と深度画像からりんごの3次元座標を求める。

    中身の計算は localization_3d.py に書いてあります。
    """
    # りんごの3次元座標を計算する
    apple_position_list = localization_3d.estimate_apple_positions_3d(
        depth_image_meters, detection_boxes, localization_settings
    )
    # 結果のリストを返す
    return apple_position_list


# --------------------------------------------------------------------------
# 1枚の画像を最初から最後まで処理して、各ステップの時間を測る
# --------------------------------------------------------------------------

def measure_one_image(color_image_path, point_cloud_npy_path, alignment_transform,
                      localization_settings, tile_list, detection_model,
                      model_input_size, device, use_half_precision,
                      confidence_threshold, iou_threshold, merge_iou_threshold):
    """
    1枚の生画像に対して5つのステップを順番に実行し、各ステップの時間を測る。

    返り値は (計測結果の辞書, りんごの3次元座標のリスト) です。
    時間の単位はすべてミリ秒です。
    """
    # --- ステップ1：画像取得 ---
    # 計測を開始する
    timer_start = timing_utils.start_timer(device)
    # カラー画像と点群を読み込む
    color_image_bgr, point_cloud_array = step1_image_acquisition(
        color_image_path, point_cloud_npy_path
    )
    # 経過時間を記録する
    step1_elapsed_ms = timing_utils.stop_timer(timer_start, device)

    # --- ステップ2：深度アライメント ---
    timer_start = timing_utils.start_timer(device)
    # 点群をカラー画像の座標系に投影して深度画像を作る
    depth_image_meters = step2_depth_alignment(point_cloud_array, alignment_transform)
    step2_elapsed_ms = timing_utils.stop_timer(timer_start, device)

    # --- ステップ3：推論前処理 ---
    timer_start = timing_utils.start_timer(device)
    # タイルに分けてモデルの入力形式に整える
    input_tensor_list = step3_inference_preprocessing(
        color_image_bgr, tile_list, model_input_size, device, use_half_precision
    )
    step3_elapsed_ms = timing_utils.stop_timer(timer_start, device)

    # --- ステップ4a：画像推論（NMSを含まない） ---
    timer_start = timing_utils.start_timer(device)
    # ニューラルネットワークの順伝播を行う
    raw_prediction_list = step4a_image_inference(detection_model, input_tensor_list)
    step4a_elapsed_ms = timing_utils.stop_timer(timer_start, device)

    # --- ステップ4b：後処理(NMS) ---
    timer_start = timing_utils.start_timer(device)
    # 重なった枠をまとめ、生画像の座標に戻す
    detection_boxes, detection_scores = step4b_postprocess_nms(
        raw_prediction_list, tile_list, model_input_size,
        confidence_threshold, iou_threshold, merge_iou_threshold
    )
    step4b_elapsed_ms = timing_utils.stop_timer(timer_start, device)

    # --- ステップ5：3D空間位置推定 ---
    timer_start = timing_utils.start_timer(device)
    # 検出したりんごの3次元座標を求める
    apple_position_list = step5_spatial_localization(
        depth_image_meters, detection_boxes, localization_settings
    )
    step5_elapsed_ms = timing_utils.stop_timer(timer_start, device)

    # 深度が取れたりんごの数を数えるための変数を0で用意する
    detections_with_depth_count = 0
    # りんごを1個ずつ見て、深度が取れているものを数える
    for one_apple in apple_position_list:
        if one_apple["median_depth_m"] > 0.0:
            detections_with_depth_count = detections_with_depth_count + 1

    # 5つのステップの時間を全部足して、1枚あたりの合計時間を求める
    total_elapsed_ms = (
        step1_elapsed_ms + step2_elapsed_ms + step3_elapsed_ms
        + step4a_elapsed_ms + step4b_elapsed_ms + step5_elapsed_ms
    )

    # 計測結果を辞書にまとめる
    measurement_result = {
        "step1_acquisition_ms": step1_elapsed_ms,
        "step2_depth_alignment_ms": step2_elapsed_ms,
        "step3_preprocessing_ms": step3_elapsed_ms,
        "step4a_inference_ms": step4a_elapsed_ms,
        "step4b_nms_ms": step4b_elapsed_ms,
        "step5_localization_ms": step5_elapsed_ms,
        "total_ms": total_elapsed_ms,
        # 参考情報：タイルの枚数
        "num_tiles": len(tile_list),
        # 参考情報：点群の点の数
        "num_points": int(point_cloud_array.shape[0]),
        # 参考情報：検出したりんごの数
        "num_detections": int(len(detection_boxes)),
        # 参考情報：そのうち深度が取れたりんごの数
        "num_detections_with_depth": detections_with_depth_count,
    }
    # 計測結果とりんごの3次元座標を返す
    return measurement_result, apple_position_list


# --------------------------------------------------------------------------
# ここからメインの処理
# --------------------------------------------------------------------------

# 計測するステップの名前を並べたリスト（CSVの列の順番もこの順になる）
STEP_NAME_LIST = [
    "step1_acquisition_ms",
    "step2_depth_alignment_ms",
    "step3_preprocessing_ms",
    "step4a_inference_ms",
    "step4b_nms_ms",
    "step5_localization_ms",
    "total_ms",
]


def read_scene_name_list(scene_list_file_path):
    """テキストファイルからシーン名の一覧を読み込んでリストで返す。"""
    # 結果を入れるための空のリストを用意する
    scene_name_list = []
    # ファイルを開く
    with open(scene_list_file_path, "r", encoding="utf-8") as opened_file:
        # 1行ずつ順番に読む
        for one_line in opened_file:
            # 前後の空白や改行を取り除く
            stripped_line = one_line.strip()
            # 空の行でなければリストに追加する
            if stripped_line != "":
                scene_name_list.append(stripped_line)
    # 完成したリストを返す
    return scene_name_list


def main():
    """コマンドから実行されたときに動く、このスクリプトの本体。"""
    # コマンドライン引数の設定を作る
    argument_parser = argparse.ArgumentParser(
        description="集中型計算方式の各処理ステップの実行時間を計測します。"
    )
    # 学習済みの重みファイルを指定する引数を追加する
    argument_parser.add_argument(
        "--weights",
        required=True,
        help="ファインチューニング済みの重みファイル(.pt)のパス",
    )
    # マシン名を指定する引数を追加する
    argument_parser.add_argument(
        "--machine-name",
        default=None,
        help="結果を区別するためのマシン名（例 workstation / jetson_orin）。省略するとホスト名を使います",
    )
    # 計測の繰り返し回数を指定する引数を追加する
    argument_parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="1枚の画像あたり何回繰り返して計測するか（既定値 10）",
    )
    # ウォームアップ回数を指定する引数を追加する
    argument_parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="計測から除外する準備運転の回数（既定値 5）",
    )
    # 画像ごとのウォームアップ回数を指定する引数を追加する
    argument_parser.add_argument(
        "--warmup-per-image",
        type=int,
        default=1,
        help=(
            "画像を切り替えるたびに行う準備運転の回数（既定値 1）。"
            "GPUは検出数などテンソルの形が変わるたびに内部の準備をやり直すため、"
            "これを入れないと各画像の1回目だけ極端に遅い値が出ます"
        ),
    )
    # 計測する画像の枚数の上限を指定する引数を追加する
    argument_parser.add_argument(
        "--limit-images",
        type=int,
        default=0,
        help="計測に使う画像の枚数の上限。0 なら test セット全部を使います（既定値 0）",
    )
    # 使用する計算装置を指定する引数を追加する
    argument_parser.add_argument(
        "--device",
        default="auto",
        help="使用する装置。auto / cuda / mps / cpu から選びます（既定値 auto）",
    )
    # 半精度を使うかどうかを指定する引数を追加する
    argument_parser.add_argument(
        "--half",
        action="store_true",
        help="指定すると半精度(float16)で推論します。NVIDIAのGPUでは速くなることがあります",
    )
    # モデルの入力サイズを指定する引数を追加する
    argument_parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="モデルに入力する画像の一辺のサイズ。学習時と合わせます（既定値 640）",
    )
    # 検出の自信度のしきい値を指定する引数を追加する
    argument_parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="この値より自信度の低い検出は捨てます（既定値 0.25）",
    )
    # NMS の重なりのしきい値を指定する引数を追加する
    argument_parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="タイル内のNMSで、枠がこの割合以上重なっていたら同じ物体とみなします（既定値 0.45）",
    )
    # タイルをまたいだ統合のしきい値を指定する引数を追加する
    argument_parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.5,
        help="タイルをまたいだ枠の統合で使う重なりのしきい値（既定値 0.5）",
    )
    # カメラパラメータのファイルを指定する引数を追加する
    argument_parser.add_argument(
        "--camera-params",
        default=None,
        help="camera_params.yaml のパス。省略するとプロジェクト直下のものを使います",
    )
    # 計測対象のシーン一覧ファイルを指定する引数を追加する
    argument_parser.add_argument(
        "--scene-list",
        default=None,
        help="計測するシーン名の一覧ファイル。省略すると outputs/splits/scenes_test.txt を使います",
    )
    # データセットの場所を指定する引数を追加する
    argument_parser.add_argument(
        "--dataset-root",
        default=None,
        help="KFuji_RGB-DS_dataset フォルダのパス。省略すると既定の場所を使います",
    )
    # 実際に引数を読み取る
    parsed_arguments = argument_parser.parse_args()

    # データセットの場所を決める
    if parsed_arguments.dataset_root is None:
        dataset_root = common_paths.get_default_dataset_root()
    else:
        dataset_root = parsed_arguments.dataset_root

    # カメラパラメータのファイルの場所を決める
    if parsed_arguments.camera_params is None:
        camera_parameters_path = os.path.join(
            common_paths.get_project_root(), "camera_params.yaml"
        )
    else:
        camera_parameters_path = parsed_arguments.camera_params

    # 計測対象のシーン一覧ファイルの場所を決める
    if parsed_arguments.scene_list is None:
        scene_list_file_path = os.path.join(
            common_paths.get_outputs_directory(), "splits", "scenes_test.txt"
        )
    else:
        scene_list_file_path = parsed_arguments.scene_list

    # シーン一覧ファイルが無ければ、先にデータ準備が必要なのでエラーを出して止める
    if not os.path.exists(scene_list_file_path):
        raise FileNotFoundError(
            "計測対象のシーン一覧が見つかりません: " + scene_list_file_path + "\n"
            + "先に python src/prepare_dataset.py を実行してください。"
        )

    # 使用する計算装置を決める
    device = timing_utils.resolve_device(parsed_arguments.device)

    # 半精度を使うかどうかを決める（NVIDIAのGPU以外では使わない）
    use_half_precision = parsed_arguments.half
    # NVIDIAのGPU以外で --half が指定された場合は、警告を出して無効にする
    if use_half_precision and device.type != "cuda":
        print("[警告] --half は NVIDIA の GPU でのみ使えます。単精度で計測します。")
        use_half_precision = False

    # マシン情報を集める
    collected_machine_information = machine_info.collect_machine_information(
        parsed_arguments.machine_name, device
    )
    # 集めたマシン情報を画面に表示する
    machine_info.print_machine_information(collected_machine_information)
    print("")

    # カメラパラメータを読み込む
    camera_parameters = depth_alignment.load_camera_parameters(camera_parameters_path)
    # 深度アライメントに使う値をあらかじめ計算しておく
    alignment_transform = depth_alignment.build_alignment_transform(camera_parameters)
    # 3D位置推定に使う値をあらかじめ計算しておく
    localization_settings = localization_3d.build_localization_settings(camera_parameters)
    # タイルの位置を計算しておく
    tile_list = build_tile_list(
        camera_parameters["tiling"],
        camera_parameters["color_camera"]["image_width"],
        camera_parameters["color_camera"]["image_height"],
    )

    # 計測対象のシーン名を読み込む
    scene_name_list = read_scene_name_list(scene_list_file_path)
    # 上限が指定されていれば、その枚数までに絞る
    if parsed_arguments.limit_images > 0:
        scene_name_list = scene_name_list[: parsed_arguments.limit_images]

    # 点群の .npy ファイルが存在するシーンだけを残す
    available_scene_name_list = []
    # シーンを1つずつ確認する
    for one_scene_name in scene_name_list:
        # そのシーンの .npy ファイルのパスを求める
        npy_file_path = common_paths.get_point_cloud_npy_path(dataset_root, one_scene_name)
        # ファイルがあればリストに追加する
        if os.path.exists(npy_file_path):
            available_scene_name_list.append(one_scene_name)
        # 無ければ警告を表示する
        else:
            print("[警告] 点群の .npy が見つかりません（このシーンは飛ばします）: " + one_scene_name)
    # 使えるシーンが1つも無ければエラーを出して止める
    if len(available_scene_name_list) == 0:
        raise FileNotFoundError(
            "計測できるシーンがありません。\n"
            + "python src/prepare_dataset.py --convert-point-clouds test を実行してください。"
        )

    # --- 学習済みモデルを読み込む ---
    print("学習済みモデルを読み込みます: " + parsed_arguments.weights)
    # ultralytics のライブラリを読み込む
    from ultralytics import YOLO
    # 重みファイルを読み込む
    yolo_model = YOLO(parsed_arguments.weights)
    # NMS を別に計測したいので、中身のニューラルネットワークだけを取り出す
    detection_model = yolo_model.model
    # 推論モード（学習用の処理を止める）にする
    detection_model = detection_model.eval()
    # 畳み込み層とバッチ正規化層を1つにまとめて推論を速くする
    if hasattr(detection_model, "fuse"):
        detection_model = detection_model.fuse()
    # 指定した計算装置に転送する
    detection_model = detection_model.to(device)
    # 半精度を使う設定ならモデルも半精度にする
    if use_half_precision:
        detection_model = detection_model.half()
    else:
        detection_model = detection_model.float()
    print("")

    # 計測の設定を画面に表示する
    print("=" * 70)
    print("計測の設定")
    print("=" * 70)
    print("  計測する画像     : " + str(len(available_scene_name_list)) + " 枚（test セット）")
    print("  1枚あたりの回数  : " + str(parsed_arguments.repeats) + " 回")
    print("  ウォームアップ   : 最初に " + str(parsed_arguments.warmup)
          + " 回、画像を変えるたびに " + str(parsed_arguments.warmup_per_image) + " 回（計測から除外）")
    print("  タイル分割       : " + str(len(tile_list)) + " 枚 / 1画像")
    print("  モデル入力サイズ : " + str(parsed_arguments.imgsz))
    print("  バッチサイズ     : 1（固定）")
    print("  半精度(float16)  : " + str(use_half_precision))
    print("=" * 70)
    print("")

    # --- ウォームアップ ---
    # 1枚目の画像のパスを求める
    warmup_color_image_path = common_paths.get_raw_color_image_path(
        dataset_root, available_scene_name_list[0]
    )
    # 1枚目の点群のパスを求める
    warmup_point_cloud_path = common_paths.get_point_cloud_npy_path(
        dataset_root, available_scene_name_list[0]
    )
    # ウォームアップを行うことを表示する
    print("ウォームアップを実行します（" + str(parsed_arguments.warmup) + " 回）...")
    # 指定された回数だけ、計測せずに一通り処理を流す
    for warmup_index in range(parsed_arguments.warmup):
        measure_one_image(
            warmup_color_image_path, warmup_point_cloud_path, alignment_transform,
            localization_settings, tile_list, detection_model, parsed_arguments.imgsz,
            device, use_half_precision, parsed_arguments.conf, parsed_arguments.iou,
            parsed_arguments.merge_iou,
        )
    print("ウォームアップ完了")
    print("")

    # --- 本計測 ---
    print("計測を開始します")
    print("")
    # 1回ごとの計測結果を入れるためのリストを用意する
    all_measurement_row_list = []
    # ステップごとの計測値をまとめるための辞書を用意する
    measurements_by_step = {}
    # ステップ名を1つずつ見て、空のリストを用意する
    for one_step_name in STEP_NAME_LIST:
        measurements_by_step[one_step_name] = []
    # 最後に処理したりんごの3次元座標を保存しておくためのリストを用意する
    detection_row_list = []

    # 画像を1枚ずつ順番に計測する
    for scene_index in range(len(available_scene_name_list)):
        # このシーンの名前を取り出す
        one_scene_name = available_scene_name_list[scene_index]
        # カラー画像のパスを求める
        color_image_path = common_paths.get_raw_color_image_path(dataset_root, one_scene_name)
        # 点群のパスを求める
        point_cloud_npy_path = common_paths.get_point_cloud_npy_path(dataset_root, one_scene_name)

        # 画像を切り替えた直後は、指定回数だけ計測せずに処理を流しておく
        # （GPUがテンソルの形ごとに内部の準備を行うため、その分を計測から外す）
        for warmup_index in range(parsed_arguments.warmup_per_image):
            measure_one_image(
                color_image_path, point_cloud_npy_path, alignment_transform,
                localization_settings, tile_list, detection_model, parsed_arguments.imgsz,
                device, use_half_precision, parsed_arguments.conf, parsed_arguments.iou,
                parsed_arguments.merge_iou,
            )

        # 指定された回数だけ繰り返して計測する
        for repeat_index in range(parsed_arguments.repeats):
            # 5つのステップを実行して時間を測る
            measurement_result, apple_position_list = measure_one_image(
                color_image_path, point_cloud_npy_path, alignment_transform,
                localization_settings, tile_list, detection_model, parsed_arguments.imgsz,
                device, use_half_precision, parsed_arguments.conf, parsed_arguments.iou,
                parsed_arguments.merge_iou,
            )
            # CSVに書き出す1行分の辞書を作る
            one_row = {
                # どのマシンで測ったか
                "machine_name": collected_machine_information["machine_name"],
                # どの装置を使ったか
                "device_type": collected_machine_information["device_type"],
                # どの画像か
                "scene_name": one_scene_name,
                # 何回目の繰り返しか
                "repeat_index": repeat_index,
            }
            # 計測結果の内容を1行分の辞書に足す
            for one_key in measurement_result:
                one_row[one_key] = measurement_result[one_key]
            # 1行分をリストに追加する
            all_measurement_row_list.append(one_row)
            # ステップごとの集計用リストにも値を追加する
            for one_step_name in STEP_NAME_LIST:
                measurements_by_step[one_step_name].append(measurement_result[one_step_name])

            # 最後の繰り返しのときだけ、りんごの3次元座標を記録しておく
            if repeat_index == parsed_arguments.repeats - 1:
                # りんごを1個ずつ見ていく
                for one_apple in apple_position_list:
                    # 記録用の1行分の辞書を作る
                    one_detection_row = {
                        "machine_name": collected_machine_information["machine_name"],
                        "scene_name": one_scene_name,
                    }
                    # りんごの情報を1行分の辞書に足す
                    for one_key in one_apple:
                        one_detection_row[one_key] = one_apple[one_key]
                    # 1行分をリストに追加する
                    detection_row_list.append(one_detection_row)

        # 1枚分の計測が終わったので、その画像の平均時間を表示する
        # （この画像の繰り返し分だけを取り出して平均する）
        total_time_list_for_this_scene = []
        # 直近の繰り返し回数分の結果を取り出す
        for row_index in range(len(all_measurement_row_list) - parsed_arguments.repeats,
                               len(all_measurement_row_list)):
            total_time_list_for_this_scene.append(all_measurement_row_list[row_index]["total_ms"])
        # 平均を計算する
        average_total_ms = sum(total_time_list_for_this_scene) / len(total_time_list_for_this_scene)
        # 進捗を表示する
        print("  [%2d/%2d] %-24s 合計 %7.1f ms / 検出 %3d 個"
              % (scene_index + 1, len(available_scene_name_list), one_scene_name,
                 average_total_ms, all_measurement_row_list[-1]["num_detections"]))

    print("")

    # --- 結果をファイルに書き出す ---
    # 結果を保存するフォルダのパスを作る
    measurements_directory = os.path.join(
        common_paths.get_outputs_directory(), "measurements"
    )
    # そのフォルダを作る
    os.makedirs(measurements_directory, exist_ok=True)
    # ファイル名に使うマシン名を取り出す
    machine_name_for_file = collected_machine_information["machine_name"]

    # --- 1つ目：1回ごとの生の計測値のCSV ---
    # 書き出すファイルのパスを作る
    raw_csv_path = os.path.join(
        measurements_directory, "timings_raw_" + machine_name_for_file + ".csv"
    )
    # 列の名前を並べたリストを作る
    raw_csv_column_list = [
        "machine_name", "device_type", "scene_name", "repeat_index",
        "step1_acquisition_ms", "step2_depth_alignment_ms", "step3_preprocessing_ms",
        "step4a_inference_ms", "step4b_nms_ms", "step5_localization_ms", "total_ms",
        "num_tiles", "num_points", "num_detections", "num_detections_with_depth",
    ]
    # CSVファイルを書き込みモードで開く
    with open(raw_csv_path, "w", encoding="utf-8", newline="") as opened_file:
        # 辞書を1行として書き出すための道具を用意する
        csv_writer = csv.DictWriter(opened_file, fieldnames=raw_csv_column_list)
        # 1行目に列の名前を書き出す
        csv_writer.writeheader()
        # 計測結果を1行ずつ書き出す
        for one_row in all_measurement_row_list:
            csv_writer.writerow(one_row)

    # --- 2つ目：統計値のまとめのCSV ---
    # 書き出すファイルのパスを作る
    summary_csv_path = os.path.join(
        measurements_directory, "timings_summary_" + machine_name_for_file + ".csv"
    )
    # マシン情報の項目名を並べたリストを作る
    machine_information_key_list = list(collected_machine_information.keys())
    # 統計値の列の名前を並べたリストを作る
    statistics_column_list = [
        "step_name", "count", "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms", "p95_ms",
        "share_of_total_percent",
    ]
    # まとめのCSVの列は「マシン情報 + 統計値」の順に並べる
    summary_csv_column_list = machine_information_key_list + statistics_column_list

    # 合計時間の平均を求めておく（各ステップが全体の何%を占めるかの計算に使う）
    total_summary = timing_utils.summarize_measurements(measurements_by_step["total_ms"])
    # 平均合計時間を取り出す
    mean_total_ms = total_summary["mean"]

    # CSVファイルを書き込みモードで開く
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as opened_file:
        # 辞書を1行として書き出すための道具を用意する
        csv_writer = csv.DictWriter(opened_file, fieldnames=summary_csv_column_list)
        # 1行目に列の名前を書き出す
        csv_writer.writeheader()
        # ステップを1つずつ順番に書き出す
        for one_step_name in STEP_NAME_LIST:
            # そのステップの統計量を計算する
            step_summary = timing_utils.summarize_measurements(measurements_by_step[one_step_name])
            # 全体に占める割合を計算する（合計時間が0のときは0にする）
            if mean_total_ms > 0.0:
                share_percent = 100.0 * step_summary["mean"] / mean_total_ms
            else:
                share_percent = 0.0
            # 1行分の辞書を、まずマシン情報で作る
            one_summary_row = dict(collected_machine_information)
            # 統計値を足していく
            one_summary_row["step_name"] = one_step_name
            one_summary_row["count"] = step_summary["count"]
            one_summary_row["mean_ms"] = round(step_summary["mean"], 4)
            one_summary_row["median_ms"] = round(step_summary["median"], 4)
            one_summary_row["std_ms"] = round(step_summary["std"], 4)
            one_summary_row["min_ms"] = round(step_summary["min"], 4)
            one_summary_row["max_ms"] = round(step_summary["max"], 4)
            one_summary_row["p95_ms"] = round(step_summary["p95"], 4)
            one_summary_row["share_of_total_percent"] = round(share_percent, 2)
            # 1行分を書き出す
            csv_writer.writerow(one_summary_row)

    # --- 3つ目：マシン情報のJSON ---
    # 書き出すファイルのパスを作る
    machine_json_path = os.path.join(
        measurements_directory, "machine_info_" + machine_name_for_file + ".json"
    )
    # 計測の設定もマシン情報と一緒に記録する
    machine_information_with_settings = dict(collected_machine_information)
    # 使った重みファイル
    machine_information_with_settings["weights"] = os.path.abspath(parsed_arguments.weights)
    # 繰り返し回数
    machine_information_with_settings["repeats"] = parsed_arguments.repeats
    # ウォームアップ回数
    machine_information_with_settings["warmup"] = parsed_arguments.warmup
    # 画像ごとのウォームアップ回数
    machine_information_with_settings["warmup_per_image"] = parsed_arguments.warmup_per_image
    # 計測した画像の枚数
    machine_information_with_settings["number_of_images"] = len(available_scene_name_list)
    # タイルの枚数
    machine_information_with_settings["number_of_tiles"] = len(tile_list)
    # モデル入力サイズ
    machine_information_with_settings["model_input_size"] = parsed_arguments.imgsz
    # バッチサイズ（常に1）
    machine_information_with_settings["batch_size"] = 1
    # 半精度を使ったかどうか
    machine_information_with_settings["half_precision"] = use_half_precision
    # JSONファイルとして書き出す
    with open(machine_json_path, "w", encoding="utf-8") as opened_file:
        json.dump(machine_information_with_settings, opened_file, ensure_ascii=False, indent=2)

    # --- 4つ目：検出したりんごの3次元座標のCSV ---
    # 書き出すファイルのパスを作る
    detections_csv_path = os.path.join(
        measurements_directory, "detections_" + machine_name_for_file + ".csv"
    )
    # 検出が1つでもあれば書き出す
    if len(detection_row_list) > 0:
        # 列の名前は、最初の行の辞書のキーをそのまま使う
        detection_column_list = list(detection_row_list[0].keys())
        # CSVファイルを書き込みモードで開く
        with open(detections_csv_path, "w", encoding="utf-8", newline="") as opened_file:
            # 辞書を1行として書き出すための道具を用意する
            csv_writer = csv.DictWriter(opened_file, fieldnames=detection_column_list)
            # 1行目に列の名前を書き出す
            csv_writer.writeheader()
            # 検出結果を1行ずつ書き出す
            for one_row in detection_row_list:
                csv_writer.writerow(one_row)

    # --- 結果を画面に表示する ---
    print("=" * 78)
    print("計測結果のまとめ（" + machine_name_for_file + "）")
    print("=" * 78)
    # 表の見出しを表示する
    print("  %-26s %9s %9s %9s %8s" % ("ステップ", "平均(ms)", "中央値(ms)", "標準偏差", "割合(%)"))
    print("  " + "-" * 74)
    # ステップを1つずつ表示する
    for one_step_name in STEP_NAME_LIST:
        # そのステップの統計量を計算する
        step_summary = timing_utils.summarize_measurements(measurements_by_step[one_step_name])
        # 全体に占める割合を計算する
        if mean_total_ms > 0.0:
            share_percent = 100.0 * step_summary["mean"] / mean_total_ms
        else:
            share_percent = 0.0
        # 合計の行の前に区切り線を入れる
        if one_step_name == "total_ms":
            print("  " + "-" * 74)
        # 1行分を表示する
        print("  %-26s %9.2f %9.2f %9.2f %8.1f"
              % (one_step_name, step_summary["mean"], step_summary["median"],
                 step_summary["std"], share_percent))
    print("=" * 78)
    print("")
    # 1秒あたり何枚処理できるかを表示する
    if mean_total_ms > 0.0:
        print("  1画像あたりの平均処理時間 : %.1f ms" % mean_total_ms)
        print("  1秒あたりの処理枚数       : %.2f 枚/秒 (FPS)" % (1000.0 / mean_total_ms))
    print("")
    # 書き出したファイルの場所を表示する
    print("結果を保存しました:")
    print("  1回ごとの生データ : " + raw_csv_path)
    print("  統計値のまとめ     : " + summary_csv_path)
    print("  マシン情報         : " + machine_json_path)
    if len(detection_row_list) > 0:
        print("  検出した3D座標     : " + detections_csv_path)
    print("")
    print("2台のマシンで計測したら、次のコマンドで比較できます:")
    print("  python src/compare_machines.py "
          + "--summary-csv outputs/measurements/timings_summary_マシン1.csv "
          + "outputs/measurements/timings_summary_マシン2.csv")


# このファイルが直接実行されたときだけ main() を呼ぶ
if __name__ == "__main__":
    main()
