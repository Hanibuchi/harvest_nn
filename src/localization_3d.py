# ============================================================================
# このファイルは何をするものか
# ----------------------------------------------------------------------------
# 【ステップ5：3D空間位置推定】の本体です。
#
# ステップ4でりんごの検出枠（bounding box）が得られ、
# ステップ2でカラー画像と同じ座標系の深度画像が得られています。
# この2つを組み合わせて、りんご1個ずつの3次元座標(X, Y, Z)を求めます。
#
# 【計算の流れ】
#   1. 検出枠の中央部分だけを切り出す
#      （枠の端には背景の葉や空が写り込みやすく、深度がずれる原因になるため）
#   2. その範囲にある「深度が分かっているピクセル」を集めて中央値を取る
#      （平均ではなく中央値を使うのは、まれに混ざる大きく外れた値の影響を
#        受けにくくするためです）
#   3. 得られた深度と、枠の中心のピクセル位置から、逆投影の式で3次元座標を求める
#        X = (u - cx) * 深度 / fx
#        Y = (v - cy) * 深度 / fy
#        Z = 深度
#
# 【返す座標系について】
#   上の式で得られる (X, Y, Z) は「カメラ光学座標系」です。
#     X ... カメラから見て右がプラス（単位: メートル）
#     Y ... カメラから見て下がプラス（単位: メートル）
#     Z ... カメラから見て前（奥行き）がプラス（単位: メートル）
#   ロボットのアーム制御などで「上がプラス」の座標系が必要な場合のために、
#   センサ座標系（X=右, Y=前, Z=上）に戻した値も一緒に返します。
# ============================================================================

# 数値計算ライブラリ
import numpy as np


def estimate_apple_positions_3d(depth_image_meters, detection_boxes,
                                localization_settings):
    """
    【ステップ5の本体】検出枠と深度画像から、りんごの3次元座標を求める。

    引数:
        depth_image_meters  : ステップ2で作った深度画像。(高さ, 幅) でメートル単位
        detection_boxes     : 検出枠のリスト。1つの枠は [x1, y1, x2, y2] の形で、
                              左上と右下のピクセル座標を表す
        localization_settings : build_localization_settings() が作った設定の辞書

    返り値:
        りんご1個ごとの情報を入れた辞書のリスト。
    """
    # 深度画像の高さと幅を取り出す
    image_height = depth_image_meters.shape[0]
    image_width = depth_image_meters.shape[1]
    # 設定から、枠の中央何割を使うかの比率を取り出す
    box_shrink_ratio = localization_settings["box_shrink_ratio"]
    # 設定から、3D座標の計算に最低限必要な有効ピクセル数を取り出す
    minimum_valid_pixels = localization_settings["minimum_valid_pixels"]
    # 設定から内部パラメータを取り出す
    focal_length_x = localization_settings["focal_length_x"]
    focal_length_y = localization_settings["focal_length_y"]
    principal_point_x = localization_settings["principal_point_x"]
    principal_point_y = localization_settings["principal_point_y"]

    # 結果を入れるための空のリストを用意する
    apple_position_list = []

    # 検出枠を1つずつ順番に処理する
    for box_index in range(len(detection_boxes)):
        # いま処理する枠を取り出す
        one_box = detection_boxes[box_index]
        # 枠の左端・上端・右端・下端の座標を取り出す
        box_x1 = float(one_box[0])
        box_y1 = float(one_box[1])
        box_x2 = float(one_box[2])
        box_y2 = float(one_box[3])

        # 枠の中心の横位置を求める
        box_center_u = (box_x1 + box_x2) / 2.0
        # 枠の中心の縦位置を求める
        box_center_v = (box_y1 + box_y2) / 2.0
        # 枠の横幅を求める
        box_width = box_x2 - box_x1
        # 枠の縦幅を求める
        box_height = box_y2 - box_y1

        # 中央部分だけを使うため、縮めた後の横幅の半分を求める
        half_width_shrunk = (box_width * box_shrink_ratio) / 2.0
        # 中央部分だけを使うため、縮めた後の縦幅の半分を求める
        half_height_shrunk = (box_height * box_shrink_ratio) / 2.0

        # 切り出す範囲の左端を求める（整数にして、画像の外に出ないようにする）
        sample_x1 = int(max(0, np.floor(box_center_u - half_width_shrunk)))
        # 切り出す範囲の上端を求める
        sample_y1 = int(max(0, np.floor(box_center_v - half_height_shrunk)))
        # 切り出す範囲の右端を求める
        sample_x2 = int(min(image_width, np.ceil(box_center_u + half_width_shrunk)))
        # 切り出す範囲の下端を求める
        sample_y2 = int(min(image_height, np.ceil(box_center_v + half_height_shrunk)))

        # 範囲の幅か高さが0以下になってしまった場合は、深度なしとして扱う
        if sample_x2 <= sample_x1 or sample_y2 <= sample_y1:
            median_depth_meters = 0.0
            valid_pixel_count = 0
        else:
            # 深度画像から、その範囲の部分だけを切り出す
            depth_patch = depth_image_meters[sample_y1:sample_y2, sample_x1:sample_x2]
            # 切り出した中から、深度が分かっている（0より大きい）値だけを集める
            valid_depth_values = depth_patch[depth_patch > 0.0]
            # 集まった有効な値の個数を数える
            valid_pixel_count = int(valid_depth_values.size)
            # 十分な数の有効値があれば中央値を計算する
            if valid_pixel_count >= minimum_valid_pixels:
                median_depth_meters = float(np.median(valid_depth_values))
            # 足りなければ深度なしとして 0 を入れる
            else:
                median_depth_meters = 0.0

        # 深度が求まった場合だけ、3次元座標を計算する
        if median_depth_meters > 0.0:
            # 逆投影の式で、カメラ光学座標系での横方向の位置を求める
            position_x_optical = (
                (box_center_u - principal_point_x) * median_depth_meters / focal_length_x
            )
            # 逆投影の式で、カメラ光学座標系での縦方向の位置を求める
            position_y_optical = (
                (box_center_v - principal_point_y) * median_depth_meters / focal_length_y
            )
            # 奥行きはそのまま深度の値になる
            position_z_optical = median_depth_meters
        else:
            # 深度が分からない場合は、座標を「非数（NaN）」にしておく
            position_x_optical = float("nan")
            position_y_optical = float("nan")
            position_z_optical = float("nan")

        # カメラ光学座標系(x=右, y=下, z=前)から
        # センサ座標系(X=右, Y=前, Z=上)に戻した値も計算しておく
        position_x_sensor = position_x_optical
        position_y_sensor = position_z_optical
        position_z_sensor = -position_y_optical

        # このりんご1個分の結果を辞書にまとめる
        one_apple_result = {
            # 何番目の検出枠か
            "detection_index": box_index,
            # 検出枠の位置（元のカラー画像のピクセル座標）
            "box_x1": box_x1,
            "box_y1": box_y1,
            "box_x2": box_x2,
            "box_y2": box_y2,
            # 枠の中心のピクセル位置
            "center_u": box_center_u,
            "center_v": box_center_v,
            # 深度の計算に使えた有効ピクセルの数
            "valid_depth_pixels": valid_pixel_count,
            # 枠の中央部分の深度の中央値（メートル）
            "median_depth_m": median_depth_meters,
            # カメラ光学座標系での3次元座標（メートル）
            "position_x_optical_m": position_x_optical,
            "position_y_optical_m": position_y_optical,
            "position_z_optical_m": position_z_optical,
            # センサ座標系での3次元座標（メートル）
            "position_x_sensor_m": position_x_sensor,
            "position_y_sensor_m": position_y_sensor,
            "position_z_sensor_m": position_z_sensor,
        }
        # できた辞書を結果のリストに追加する
        apple_position_list.append(one_apple_result)

    # 全てのりんごの結果をまとめたリストを返す
    return apple_position_list


def build_localization_settings(camera_parameters):
    """
    camera_params.yaml の内容から、ステップ5で使う設定だけを取り出してまとめる。

    depth_alignment.py の build_alignment_transform と同じ考え方で、
    計測のたびに設定ファイルを読み直さなくて済むように、
    必要な値だけをあらかじめ辞書にまとめておきます。
    """
    # ステップ5の設定部分を取り出す
    localization_parameters = camera_parameters["spatial_localization"]
    # 必要な値だけを集めた辞書を作る
    localization_settings = {
        # 枠の中央何割を深度の取得に使うか
        "box_shrink_ratio": float(localization_parameters["box_shrink_ratio"]),
        # 3D座標の計算に最低限必要な有効ピクセル数
        "minimum_valid_pixels": int(localization_parameters["minimum_valid_pixels"]),
        # 横方向の焦点距離
        "focal_length_x": float(camera_parameters["color_camera"]["fx"]),
        # 縦方向の焦点距離
        "focal_length_y": float(camera_parameters["color_camera"]["fy"]),
        # 画像中心の横座標
        "principal_point_x": float(camera_parameters["color_camera"]["cx"]),
        # 画像中心の縦座標
        "principal_point_y": float(camera_parameters["color_camera"]["cy"]),
    }
    # 作った辞書を返す
    return localization_settings
