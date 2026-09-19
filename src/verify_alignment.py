# ============================================================================
# このファイルは何をするものか
# ----------------------------------------------------------------------------
# 【深度アライメントの検証スクリプト（任意実行）】です。
#
# ステップ2の深度アライメントは、カメラの内部パラメータ（fx, fy, cx, cy）が
# 正しくないと、深度がカラー画像とずれた位置に貼り付いてしまいます。
# そうなると、ステップ5で求めるりんごの3次元座標も間違ってしまいます。
#
# KFuji RGB-DS データセットにはキャリブレーション結果が同梱されていないため、
# camera_params.yaml には Kinect v2 の一般的な公称値を入れてあります。
# このスクリプトは、その値で本当に正しく重なるかを確認するためのものです。
#
# 【確認のしくみ】
#   点群の各点には「その点の色(R,G,B)」も一緒に記録されています。
#   そこで、点を1つずつカラー画像に投影し、
#
#       「その点が持っている色」 と 「投影先のピクセルの色」
#
#   を比べます。カメラパラメータが正しければ、この2つはよく一致するはずです。
#   ずれていれば、全く違う場所の色と比べることになるので一致しません。
#
#   一致の度合いは相関係数で表します。-1 〜 1 の値で、1 に近いほど良い一致です。
#   （木の葉のような細かい模様が相手なので、完璧に合っていても 1 にはなりません。
#     0.6 くらいあれば十分に合っていると考えてよいです）
#
#   数値に加えて、目で見て確認できる画像も保存します。
#
# 【このスクリプトで分かったこと（参考）】
#   Kinect v2 は深度センサとカラーカメラが約52mm離れた位置に付いているため、
#   点群をそのまま投影すると約30ピクセル横にずれます。
#   camera_params.yaml の extrinsics の平行移動でこれを補正しており、
#   補正前の一致度 0.31 が、補正後は 0.63 に改善しています。
#
# 【実行例】
#   # 1シーンについて確認する
#   python src/verify_alignment.py
#
#   # 平行移動量を自動で少しずつ変えて、一番よく一致する値を探す
#   python src/verify_alignment.py --search
#
#   # 複数のシーンをまとめて確認する
#   python src/verify_alignment.py --search --num-scenes 5
# ============================================================================

# コマンドライン引数を扱うための標準ライブラリ
import argparse
# ファイルやフォルダを操作するための標準ライブラリ
import os

# 数値計算ライブラリ
import numpy as np
# 画像処理ライブラリ
import cv2

# 自作のモジュールを読み込む
import common_paths
import kinect_io
import depth_alignment


def render_point_cloud_colors(point_cloud_array, alignment_transform):
    """
    点群をカラー画像の座標系に投影し、各点の色を塗った画像を作る。

    返り値は (作った画像, 色が塗られた場所を示す真偽値の画像) です。
    作った画像は OpenCV に合わせて B,G,R の順の並びになっています。
    """
    # 点群を投影して、投影先の位置・奥行き・色を受け取る
    pixel_u, pixel_v, point_depth, point_colors_rgb = depth_alignment.project_points_with_colors(
        point_cloud_array, alignment_transform
    )
    # 画像の幅と高さを取り出す
    image_width = alignment_transform["image_width"]
    image_height = alignment_transform["image_height"]

    # 手前の点で上書きするため、奥行きの大きい点から先に塗る。
    # そのために奥行きの大きい順に並べ替えたときの順番を求める
    drawing_order = np.argsort(-point_depth)

    # 色を塗るための、真っ黒な画像を用意する
    rendered_image_bgr = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    # 色が塗られた場所を記録するための、全て偽の画像を用意する
    painted_mask = np.zeros((image_height, image_width), dtype=bool)

    # 並べ替えた順番に従って、投影先の位置を取り出す
    ordered_pixel_v = pixel_v[drawing_order]
    ordered_pixel_u = pixel_u[drawing_order]
    # 並べ替えた順番に従って、点の色を取り出す
    ordered_colors_rgb = point_colors_rgb[drawing_order]

    # 色の値を 0〜255 の整数に直す
    clipped_colors = np.clip(ordered_colors_rgb, 0, 255).astype(np.uint8)

    # 画像の青(B)の面に、点の青の値を書き込む（RGBの3番目が青）
    rendered_image_bgr[ordered_pixel_v, ordered_pixel_u, 0] = clipped_colors[:, 2]
    # 画像の緑(G)の面に、点の緑の値を書き込む
    rendered_image_bgr[ordered_pixel_v, ordered_pixel_u, 1] = clipped_colors[:, 1]
    # 画像の赤(R)の面に、点の赤の値を書き込む
    rendered_image_bgr[ordered_pixel_v, ordered_pixel_u, 2] = clipped_colors[:, 0]
    # 色を塗った場所に印を付ける
    painted_mask[ordered_pixel_v, ordered_pixel_u] = True

    # 点はまばらなので、そのままでは見づらい。
    # 3x3 の膨張処理で少しだけ広げて、目で比べやすい画像にする
    # （この処理は見た目のためだけのもので、数値の評価には使わない）
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    rendered_image_bgr = cv2.dilate(rendered_image_bgr, dilation_kernel)

    # できた画像と印を返す
    return rendered_image_bgr, painted_mask


def compute_point_color_agreement(point_cloud_array, alignment_transform, color_image_bgr):
    """
    点群の色と、投影先のカラー画像の色がどれくらい一致するかを計算する。

    点を1つずつカラー画像に投影し、
        ・その点が持っている色
        ・投影先のピクセルの色
    の2つを比べます。カメラパラメータが正しければよく一致します。

    返り値は (相関係数, 色の平均誤差, 比べた点の数) です。
        相関係数     : -1〜1。1に近いほど良い一致（模様の形が合っているか）
        色の平均誤差 : 0〜255。小さいほど良い一致（色の値そのもののずれ）
    """
    # 点群を投影して、投影先の位置と点の色を受け取る
    pixel_u, pixel_v, point_depth, point_colors_rgb = depth_alignment.project_points_with_colors(
        point_cloud_array, alignment_transform
    )
    # 投影できた点が少なすぎる場合は、比べられないので0を返す
    if pixel_u.size < 100:
        return 0.0, 255.0, int(pixel_u.size)

    # 投影先のピクセルの色をカラー画像から取り出す（並びは B,G,R）
    image_colors_bgr = color_image_bgr[pixel_v, pixel_u].astype(np.float64)
    # 並びを B,G,R から R,G,B に入れ替えて、点の色と同じ並びにする
    image_colors_rgb = image_colors_bgr[:, ::-1]
    # 点の色を小数に変換する
    point_colors = point_colors_rgb.astype(np.float64)

    # 色の値そのもののずれ（平均絶対誤差）を計算する
    mean_absolute_error = float(np.mean(np.abs(image_colors_rgb - point_colors)))

    # 相関係数は明るさ（白黒の濃さ）で計算する。R,G,Bの平均を明るさとする
    image_brightness = image_colors_rgb.mean(axis=1)
    point_brightness = point_colors.mean(axis=1)
    # それぞれから平均を引いて、全体の明るさの違いをなくす
    image_centered = image_brightness - np.mean(image_brightness)
    point_centered = point_brightness - np.mean(point_brightness)
    # それぞれのばらつきの大きさを計算する
    image_norm = np.sqrt(np.sum(image_centered * image_centered))
    point_norm = np.sqrt(np.sum(point_centered * point_centered))
    # ばらつきが0だと割り算ができないので0を返す
    if image_norm == 0.0 or point_norm == 0.0:
        return 0.0, mean_absolute_error, int(pixel_u.size)
    # 相関係数を計算する
    correlation = float(np.sum(image_centered * point_centered) / (image_norm * point_norm))
    # 3つの値をまとめて返す
    return correlation, mean_absolute_error, int(pixel_u.size)


def make_depth_overlay_image(color_image_bgr, depth_image_meters):
    """
    深度画像に色を付けて、カラー画像に半透明で重ねた確認用の画像を作る。

    深度が近いところと遠いところが色で分かるので、
    りんごの位置と深度の位置が合っているかを目で確認できます。
    """
    # 深度が入っている場所を調べる
    has_depth = depth_image_meters > 0.0
    # 深度が1つも無ければ、元の画像をそのまま返す
    if not np.any(has_depth):
        return color_image_bgr.copy()
    # 深度の最小値を求める
    minimum_depth = float(np.min(depth_image_meters[has_depth]))
    # 深度の最大値を求める（遠すぎる外れ値の影響を避けるため95パーセンタイルを使う）
    maximum_depth = float(np.percentile(depth_image_meters[has_depth], 95))
    # 最大と最小が同じだと割り算ができないので、その場合は少しずらす
    if maximum_depth <= minimum_depth:
        maximum_depth = minimum_depth + 1.0
    # 深度を 0〜1 の範囲に変換する
    normalized_depth = (depth_image_meters - minimum_depth) / (maximum_depth - minimum_depth)
    # 0未満と1より大きい値を、0〜1の範囲に収める
    normalized_depth = np.clip(normalized_depth, 0.0, 1.0)
    # 0〜255 の整数に変換する
    depth_as_byte = (normalized_depth * 255.0).astype(np.uint8)
    # 深度に色を付ける（近いほど赤、遠いほど青になる配色）
    colored_depth = cv2.applyColorMap(depth_as_byte, cv2.COLORMAP_JET)
    # 深度が無い場所は黒くしておく
    colored_depth[np.logical_not(has_depth)] = 0
    # カラー画像と色付き深度を半分ずつ混ぜる
    overlay_image = cv2.addWeighted(color_image_bgr, 0.5, colored_depth, 0.5, 0.0)
    # できた画像を返す
    return overlay_image


def load_scene_data(dataset_root, scene_name):
    """あるシーンのカラー画像と点群を読み込んで返す。"""
    # カラー画像のパスを求める
    color_image_path = common_paths.get_raw_color_image_path(dataset_root, scene_name)
    # カラー画像を読み込む
    color_image_bgr = kinect_io.load_color_image(color_image_path)
    # 点群の .npy のパスを求める
    point_cloud_npy_path = common_paths.get_point_cloud_npy_path(dataset_root, scene_name)
    # .npy があればそちらから読み込む（速いため）
    if os.path.exists(point_cloud_npy_path):
        point_cloud_array = kinect_io.load_point_cloud_from_npy(point_cloud_npy_path)
    # 無ければ元の .mat から読み込む
    else:
        point_cloud_mat_path = common_paths.get_raw_point_cloud_mat_path(dataset_root, scene_name)
        point_cloud_array = kinect_io.load_point_cloud_from_mat(point_cloud_mat_path)
    # 2つをまとめて返す
    return color_image_bgr, point_cloud_array


def choose_scene_names(dataset_root, requested_scene_name, number_of_scenes):
    """確認に使うシーン名のリストを決めて返す。"""
    # シーンが指定されていれば、それだけを使う
    if requested_scene_name is not None:
        return [requested_scene_name]
    # test セットのシーン一覧ファイルのパスを作る
    scene_list_file_path = os.path.join(
        common_paths.get_outputs_directory(), "splits", "scenes_test.txt"
    )
    # 一覧ファイルがあれば、その中から使う
    if os.path.exists(scene_list_file_path):
        # 結果を入れるための空のリストを用意する
        candidate_scene_name_list = []
        # ファイルを開いて1行ずつ読む
        with open(scene_list_file_path, "r", encoding="utf-8") as opened_file:
            for one_line in opened_file:
                # 前後の空白を取り除く
                stripped_line = one_line.strip()
                # 空行でなければリストに追加する
                if stripped_line != "":
                    candidate_scene_name_list.append(stripped_line)
    # 無ければ、データセット全体のシーン名を使う
    else:
        candidate_scene_name_list = common_paths.list_all_scene_names(dataset_root)
    # 指定された枚数までに絞って返す
    return candidate_scene_name_list[:number_of_scenes]


def evaluate_scene_list(scene_data_list, alignment_transform,
                        translation_x, translation_y):
    """
    指定した平行移動量で、読み込み済みの全シーンの一致度の平均を計算する。

    探索のたびに画像を読み直すと遅いので、読み込み済みのデータを受け取ります。
    """
    # 設定を書き換えてしまわないように、控えを作る
    original_translation_vector = alignment_transform["translation_vector"]
    # 試したい平行移動量を設定に書き込む
    alignment_transform["translation_vector"] = np.array(
        [translation_x, translation_y, 0.0], dtype=np.float32
    )
    # 各シーンの相関係数を足していくための変数を0で用意する
    correlation_total = 0.0
    # シーンを1つずつ評価する
    for color_image_bgr, point_cloud_array in scene_data_list:
        # このシーンの一致度を計算する
        correlation, mean_absolute_error, point_count = compute_point_color_agreement(
            point_cloud_array, alignment_transform, color_image_bgr
        )
        # 相関係数を足す
        correlation_total = correlation_total + correlation
    # 設定を元に戻す
    alignment_transform["translation_vector"] = original_translation_vector
    # 平均を計算して返す
    return correlation_total / len(scene_data_list)


def main():
    """コマンドから実行されたときに動く、このスクリプトの本体。"""
    # コマンドライン引数の設定を作る
    argument_parser = argparse.ArgumentParser(
        description="深度アライメント（カメラパラメータ）が正しいかを確認します。"
    )
    # 確認に使うシーンを指定する引数を追加する
    argument_parser.add_argument(
        "--scene",
        default=None,
        help="確認に使うシーン名。省略すると test セットの先頭から使います",
    )
    # 確認に使うシーンの枚数を指定する引数を追加する
    argument_parser.add_argument(
        "--num-scenes",
        type=int,
        default=3,
        help="確認に使うシーンの枚数（既定値 3）。--scene を指定した場合は無視されます",
    )
    # カメラパラメータのファイルを指定する引数を追加する
    argument_parser.add_argument(
        "--camera-params",
        default=None,
        help="camera_params.yaml のパス。省略するとプロジェクト直下のものを使います",
    )
    # データセットの場所を指定する引数を追加する
    argument_parser.add_argument(
        "--dataset-root", default=None, help="KFuji_RGB-DS_dataset フォルダのパス"
    )
    # パラメータ探索を行うかどうかを指定する引数を追加する
    argument_parser.add_argument(
        "--search",
        action="store_true",
        help="指定すると、平行移動量を少しずつ変えて一番よく一致する値を探します",
    )
    # 探索する平行移動の範囲を指定する引数を追加する
    argument_parser.add_argument(
        "--search-range-mm",
        type=float,
        default=90.0,
        help="探索する平行移動の範囲（ミリメートル、既定値 90）",
    )
    # 探索する平行移動の刻みを指定する引数を追加する
    argument_parser.add_argument(
        "--search-step-mm",
        type=float,
        default=10.0,
        help="平行移動を変える刻み幅（ミリメートル、既定値 10）",
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

    # 確認に使うシーン名を決める
    scene_name_list = choose_scene_names(
        dataset_root, parsed_arguments.scene, parsed_arguments.num_scenes
    )

    # 処理の開始を表示する
    print("=" * 70)
    print("深度アライメントの確認")
    print("=" * 70)
    print("  確認するシーン : " + str(len(scene_name_list)) + " 枚")
    for one_scene_name in scene_name_list:
        print("    - " + one_scene_name)
    print("")

    # カメラパラメータを読み込む
    camera_parameters = depth_alignment.load_camera_parameters(camera_parameters_path)
    # 投影に使う値をあらかじめ計算しておく
    alignment_transform = depth_alignment.build_alignment_transform(camera_parameters)

    # 各シーンのデータを読み込んで、リストにまとめておく
    print("画像と点群を読み込んでいます...")
    scene_data_list = []
    # シーンを1つずつ読み込む
    for one_scene_name in scene_name_list:
        # そのシーンのカラー画像と点群を読み込む
        color_image_bgr, point_cloud_array = load_scene_data(dataset_root, one_scene_name)
        # リストに追加する
        scene_data_list.append((color_image_bgr, point_cloud_array))
    print("読み込み完了")
    print("")

    # --- いまの設定での一致度を計算する ---
    # 現在の平行移動量を取り出す
    current_translation_vector = alignment_transform["translation_vector"]
    # 現在の設定を表示する
    print("いまの camera_params.yaml の設定:")
    print("  fx = %.2f  fy = %.2f  cx = %.2f  cy = %.2f"
          % (alignment_transform["focal_length_x"], alignment_transform["focal_length_y"],
             alignment_transform["principal_point_x"], alignment_transform["principal_point_y"]))
    print("  平行移動 = [%.4f, %.4f, %.4f] m"
          % (current_translation_vector[0], current_translation_vector[1],
             current_translation_vector[2]))
    print("")
    # シーンごとの一致度を表示する
    print("シーンごとの一致度:")
    # 相関係数を足していくための変数を0で用意する
    correlation_total = 0.0
    # シーンを1つずつ評価する
    for scene_index in range(len(scene_name_list)):
        # このシーンのデータを取り出す
        color_image_bgr, point_cloud_array = scene_data_list[scene_index]
        # 一致度を計算する
        correlation, mean_absolute_error, point_count = compute_point_color_agreement(
            point_cloud_array, alignment_transform, color_image_bgr
        )
        # 相関係数を足す
        correlation_total = correlation_total + correlation
        # 結果を表示する
        print("  %-24s 相関係数 %.4f / 色の平均誤差 %5.1f / 投影できた点 %d"
              % (scene_name_list[scene_index], correlation, mean_absolute_error, point_count))
    # 平均の一致度を計算する
    current_average_correlation = correlation_total / len(scene_name_list)
    # 平均を表示する
    print("  " + "-" * 62)
    print("  平均の相関係数 = %.4f" % current_average_correlation)
    print("")

    # --- パラメータ探索（--search が指定されたときだけ） ---
    if parsed_arguments.search:
        print("平行移動量を少しずつ変えて、一番よく一致する値を探します...")
        print("")
        # 現在の平行移動量を基準にする
        base_translation_x = float(current_translation_vector[0])
        base_translation_y = float(current_translation_vector[1])
        # 一番良かった一致度を、いまの値で初期化する
        best_correlation = current_average_correlation
        # 一番良かった平行移動量を、いまの値で初期化する
        best_translation_x = base_translation_x
        best_translation_y = base_translation_y

        # 探索する範囲と刻みをメートル単位に直す
        search_range_meters = parsed_arguments.search_range_mm / 1000.0
        search_step_meters = parsed_arguments.search_step_mm / 1000.0

        # 試す横方向の平行移動量の一覧を作る
        translation_x_candidate_list = []
        # 基準値から範囲の分だけマイナス側にずらした値から始める
        one_value = base_translation_x - search_range_meters
        # 範囲のプラス側の端に達するまで、刻みずつ値を並べる
        while one_value <= base_translation_x + search_range_meters + 1e-9:
            translation_x_candidate_list.append(one_value)
            one_value = one_value + search_step_meters

        # 試す縦方向の平行移動量の一覧を作る（縦方向のずれは小さいので範囲は半分にする）
        translation_y_candidate_list = []
        one_value = base_translation_y - search_range_meters / 2.0
        while one_value <= base_translation_y + search_range_meters / 2.0 + 1e-9:
            translation_y_candidate_list.append(one_value)
            one_value = one_value + search_step_meters

        # 試す組み合わせの総数を数える
        total_combination_count = (
            len(translation_x_candidate_list) * len(translation_y_candidate_list)
        )
        # いま何番目を試しているかを数えるための変数を0で用意する
        tried_combination_count = 0

        # 横方向の候補を1つずつ試す
        for one_translation_x in translation_x_candidate_list:
            # 縦方向の候補を1つずつ試す
            for one_translation_y in translation_y_candidate_list:
                # その平行移動量での平均一致度を計算する
                trial_correlation = evaluate_scene_list(
                    scene_data_list, alignment_transform, one_translation_x, one_translation_y
                )
                # これまでで一番良ければ、その値を覚えておく
                if trial_correlation > best_correlation:
                    best_correlation = trial_correlation
                    best_translation_x = one_translation_x
                    best_translation_y = one_translation_y
                # 試した数を1増やす
                tried_combination_count = tried_combination_count + 1
                # 20回ごとに進捗を表示する
                if tried_combination_count % 20 == 0:
                    print("  %d / %d 通り試しました（現在の最良 = %.4f）"
                          % (tried_combination_count, total_combination_count, best_correlation))

        # 探索結果を表示する
        print("")
        print("探索が終わりました。")
        print("  一番よく一致した平行移動量:")
        print("    translation_meters: [%.4f, %.4f, 0.0]"
              % (best_translation_x, best_translation_y))
        print("  そのときの平均相関係数 = %.4f （いまの設定では %.4f）"
              % (best_correlation, current_average_correlation))
        print("")
        # 明らかに改善した場合だけ、設定ファイルの書き換えを勧める
        if best_correlation > current_average_correlation + 0.005:
            print("  いまの設定より良い値が見つかりました。")
            print("  camera_params.yaml の extrinsics の translation_meters を")
            print("  上の値に書き換えると、アライメントの精度が上がります。")
        else:
            print("  いまの設定がほぼ最良でした。書き換えの必要はありません。")
        print("")
        # 確認画像は一番良かった設定で作るため、その値を書き込んでおく
        alignment_transform["translation_vector"] = np.array(
            [best_translation_x, best_translation_y, 0.0], dtype=np.float32
        )

    # --- 確認用の画像を作って保存する ---
    # 保存先のフォルダのパスを作る
    output_directory = os.path.join(common_paths.get_outputs_directory(), "alignment_check")
    # そのフォルダを作る
    os.makedirs(output_directory, exist_ok=True)

    # 確認画像は1枚目のシーンについて作る
    color_image_bgr, point_cloud_array = scene_data_list[0]
    # 1枚目のシーン名を取り出す
    first_scene_name = scene_name_list[0]

    # 点群の色を投影した画像を作る
    rendered_image_bgr, painted_mask = render_point_cloud_colors(
        point_cloud_array, alignment_transform
    )
    # 深度画像を作る
    depth_image_meters = depth_alignment.align_depth_to_color(
        point_cloud_array, alignment_transform
    )
    # 深度をカラー画像に重ねた画像を作る
    overlay_image_bgr = make_depth_overlay_image(color_image_bgr, depth_image_meters)

    # 3枚の画像を横に並べた比較画像を作るため、見やすい大きさに縮める
    display_width = 640
    # 縮小後の高さを、縦横比を保って計算する
    display_height = int(color_image_bgr.shape[0] * display_width / color_image_bgr.shape[1])
    # 本物のカラー画像を縮小する
    small_color = cv2.resize(color_image_bgr, (display_width, display_height))
    # 点群を投影した画像を縮小する（INTER_AREA は縮小時にきれいに見える方法）
    small_rendered = cv2.resize(rendered_image_bgr, (display_width, display_height),
                                interpolation=cv2.INTER_AREA)
    # 深度を重ねた画像を縮小する
    small_overlay = cv2.resize(overlay_image_bgr, (display_width, display_height))
    # 3枚を横につなげる
    comparison_image = np.hstack([small_color, small_rendered, small_overlay])

    # それぞれの画像に説明の文字を書き込む
    cv2.putText(comparison_image, "1: real color image", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(comparison_image, "2: projected point cloud", (display_width + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(comparison_image, "3: depth overlay", (display_width * 2 + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 比較画像のパスを作る
    comparison_image_path = os.path.join(
        output_directory, "alignment_" + first_scene_name + ".jpg"
    )
    # 比較画像を保存する
    cv2.imwrite(comparison_image_path, comparison_image)

    # 最終的な一致度をもう一度計算する
    final_correlation, final_mean_error, final_point_count = compute_point_color_agreement(
        point_cloud_array, alignment_transform, color_image_bgr
    )

    # 結果を表示する
    print("確認画像を保存しました: " + comparison_image_path)
    print("")
    print("  画像の見方:")
    print("    左   … 本物のカラー画像")
    print("    中央 … 点群の色を投影して作った画像（左と同じ模様に見えれば正しい）")
    print("    右   … 深度をカラー画像に重ねたもの（赤いほど手前、青いほど奥）")
    print("")
    print("  1枚目のシーンの一致度（相関係数） = %.4f" % final_correlation)
    # 一致度に応じた目安を表示する
    if final_correlation > 0.55:
        print("  → よく一致しています。カメラパラメータは適切です。")
    elif final_correlation > 0.40:
        print("  → だいたい合っていますが、ずれがあるかもしれません。")
        print("     --search を付けて実行すると、より良い値を探せます。")
    else:
        print("  → 一致していません。camera_params.yaml の設定を見直してください。")
        print("     --search を付けて実行すると、より良い値を探せます。")


# このファイルが直接実行されたときだけ main() を呼ぶ
if __name__ == "__main__":
    main()
