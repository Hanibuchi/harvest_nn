# ============================================================================
# このファイルは何をするものか
# ----------------------------------------------------------------------------
# 【ステップ2：深度アライメント】の本体です。
#
# KFuji の生データの深度情報は「3次元の点の集まり（点群）」の形で保存されていて、
# カラー画像とは座標系が違います。そのままでは
# 「カラー画像のこのピクセルの奥行きは何メートルか」が分かりません。
#
# そこでこのファイルでは、点群の1点1点を
#
#     3次元の点 (X, Y, Z)  →  カラー画像上の位置 (u, v) と 奥行き z
#
# という計算でカラー画像の座標系に投影し、
# カラー画像と同じ大きさ(1920x1080)の「深度画像」を作ります。
# これが論文でいう Depth Alignment（深度の位置合わせ）にあたる処理です。
#
# 【計算の流れ】
#   1. 座標軸の向きをそろえる
#        KFuji の点群は「X=右, Y=前, Z=上」の向き。
#        カメラへの投影計算で使うのは「x=右, y=下, z=前」の向き。
#        これを axis_permutation_matrix（3x3の行列）を掛けて変換する。
#   2. 外部パラメータ（回転と平行移動）を適用して、カラーカメラの座標系に移す。
#   3. 内部パラメータ（fx, fy, cx, cy）を使って画像平面に投影する。
#        u = fx * (x / z) + cx
#        v = fy * (y / z) + cy
#   4. 同じピクセルに複数の点が飛んできたら、一番手前（zが小さい）の点を採用する。
#      （これを Zバッファ処理と呼びます。奥の点が手前の点を上書きしないようにする）
#   5. 点と点の間にできた隙間を必要に応じて埋める。
# ============================================================================

# 数値計算ライブラリ
import numpy as np
# 画像処理ライブラリ（隙間を埋める膨張処理に使う）
import cv2
# 設定ファイル(YAML)を読むためのライブラリ
import yaml


def load_camera_parameters(camera_parameters_path):
    """camera_params.yaml を読み込んで、辞書として返す。"""
    # 設定ファイルを文字コードUTF-8で開く
    with open(camera_parameters_path, "r", encoding="utf-8") as opened_file:
        # YAML形式の文章を Python の辞書に変換する
        camera_parameters = yaml.safe_load(opened_file)
    # 読み込んだ辞書を返す
    return camera_parameters


def build_intrinsic_matrix(color_camera_parameters):
    """
    内部パラメータ(fx, fy, cx, cy)から 3x3 のカメラ行列 K を作って返す。

    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    """
    # 横方向の焦点距離を取り出す
    focal_length_x = float(color_camera_parameters["fx"])
    # 縦方向の焦点距離を取り出す
    focal_length_y = float(color_camera_parameters["fy"])
    # 画像中心の横座標を取り出す
    principal_point_x = float(color_camera_parameters["cx"])
    # 画像中心の縦座標を取り出す
    principal_point_y = float(color_camera_parameters["cy"])
    # 3行3列の行列を作る
    intrinsic_matrix = np.array(
        [
            [focal_length_x, 0.0, principal_point_x],
            [0.0, focal_length_y, principal_point_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # 作った行列を返す
    return intrinsic_matrix


def build_rotation_matrix_from_degrees(rotation_degrees):
    """
    X軸・Y軸・Z軸まわりの回転角（度）から、3x3 の回転行列を作って返す。

    回転は X軸 → Y軸 → Z軸 の順で適用されます。
    ずれが無い（全部0度の）場合は、何もしない単位行列が返ります。
    """
    # 3つの角度をそれぞれ度からラジアンに変換する
    angle_x_radians = np.deg2rad(float(rotation_degrees[0]))
    angle_y_radians = np.deg2rad(float(rotation_degrees[1]))
    angle_z_radians = np.deg2rad(float(rotation_degrees[2]))
    # X軸まわりの回転行列を作る
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle_x_radians), -np.sin(angle_x_radians)],
            [0.0, np.sin(angle_x_radians), np.cos(angle_x_radians)],
        ],
        dtype=np.float64,
    )
    # Y軸まわりの回転行列を作る
    rotation_y = np.array(
        [
            [np.cos(angle_y_radians), 0.0, np.sin(angle_y_radians)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle_y_radians), 0.0, np.cos(angle_y_radians)],
        ],
        dtype=np.float64,
    )
    # Z軸まわりの回転行列を作る
    rotation_z = np.array(
        [
            [np.cos(angle_z_radians), -np.sin(angle_z_radians), 0.0],
            [np.sin(angle_z_radians), np.cos(angle_z_radians), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # 3つの回転をまとめて1つの行列にする（Z・Y・Xの順に掛ける）
    rotation_matrix = rotation_z @ rotation_y @ rotation_x
    # 作った行列を返す
    return rotation_matrix


def build_alignment_transform(camera_parameters):
    """
    camera_params.yaml の内容から、投影計算に必要な値をあらかじめ計算してまとめる。

    この関数は「毎回の計測のたびに同じ計算を繰り返さない」ために用意しています。
    ここで作った辞書を align_depth_to_color に渡して使います。
    （計測したいのは点群の投影処理そのものなので、
      設定ファイルの読み込みや行列の準備は計測の外で済ませておきます）
    """
    # 座標軸を並べ替える3x3行列を取り出して numpy 配列にする
    axis_permutation_matrix = np.array(
        camera_parameters["axis_permutation_matrix"], dtype=np.float64
    )
    # 外部パラメータの設定部分を取り出す
    extrinsics_parameters = camera_parameters["extrinsics"]
    # 回転角（度）から回転行列を作る
    rotation_matrix = build_rotation_matrix_from_degrees(
        extrinsics_parameters["rotation_degrees"]
    )
    # 平行移動（メートル）を numpy 配列にする
    translation_vector = np.array(
        extrinsics_parameters["translation_meters"], dtype=np.float64
    )
    # 「座標軸の並べ替え」と「外部パラメータの回転」を1つの行列にまとめておく
    # （こうすると、点群に掛ける行列が1回で済んで速くなる）
    combined_rotation_matrix = rotation_matrix @ axis_permutation_matrix
    # 計測のときに必要な値をひとまとめの辞書にする
    alignment_transform = {
        # 点群に掛ける3x3行列（転置した形で持っておくと後の計算が書きやすい）
        "rotation_matrix_transposed": combined_rotation_matrix.T.astype(np.float32),
        # 平行移動ベクトル
        "translation_vector": translation_vector.astype(np.float32),
        # 横方向の焦点距離
        "focal_length_x": float(camera_parameters["color_camera"]["fx"]),
        # 縦方向の焦点距離
        "focal_length_y": float(camera_parameters["color_camera"]["fy"]),
        # 画像中心の横座標
        "principal_point_x": float(camera_parameters["color_camera"]["cx"]),
        # 画像中心の縦座標
        "principal_point_y": float(camera_parameters["color_camera"]["cy"]),
        # 出力する深度画像の幅
        "image_width": int(camera_parameters["color_camera"]["image_width"]),
        # 出力する深度画像の高さ
        "image_height": int(camera_parameters["color_camera"]["image_height"]),
        # 点群のどの列がX,Y,Zかの対応表
        "column_indices": camera_parameters["point_cloud_columns"],
        # 隙間を埋めるかどうか
        "fill_holes": bool(camera_parameters["depth_alignment"]["fill_holes"]),
        # 隙間を埋めるときのフィルタの大きさ
        "fill_kernel_size": int(camera_parameters["depth_alignment"]["fill_kernel_size"]),
    }
    # 作った辞書を返す
    return alignment_transform


def align_depth_to_color(point_cloud_array, alignment_transform):
    """
    【ステップ2の本体】点群をカラー画像の座標系に投影して、深度画像を作る。

    引数:
        point_cloud_array   : (N行, 8列) の点群。kinect_io が読み込んだもの
        alignment_transform : build_alignment_transform() が作った設定の辞書

    返り値:
        深度画像。形は (高さ, 幅) で、値の単位はメートル。
        深度が分からなかったピクセルには 0 が入っています。
    """
    # --- 1. 点群から X, Y, Z の3列だけを取り出す ---
    # どの列がX,Y,Zかの対応表を取り出す
    column_indices = alignment_transform["column_indices"]
    # X列・Y列・Z列をこの順に並べた (N行,3列) の配列を作る
    points_xyz = point_cloud_array[
        :, [column_indices["x"], column_indices["y"], column_indices["z"]]
    ]
    # 計算を軽くするため、32ビット小数に変換しておく
    points_xyz = points_xyz.astype(np.float32, copy=False)

    # --- 2. 座標軸をそろえ、外部パラメータを適用してカメラ座標系に移す ---
    # 用意しておいた回転行列（転置済み）を取り出す
    rotation_matrix_transposed = alignment_transform["rotation_matrix_transposed"]
    # 全ての点に行列を掛ける（(N,3) x (3,3) の行列積で一気に計算する）
    points_in_camera_frame = points_xyz @ rotation_matrix_transposed
    # 平行移動の分を足す
    points_in_camera_frame = points_in_camera_frame + alignment_transform["translation_vector"]

    # --- 3. 内部パラメータを使って画像平面に投影する ---
    # カメラ座標系での横方向の座標を取り出す
    camera_x = points_in_camera_frame[:, 0]
    # カメラ座標系での縦方向の座標を取り出す
    camera_y = points_in_camera_frame[:, 1]
    # カメラ座標系での奥行き（カメラからの距離）を取り出す
    camera_z = points_in_camera_frame[:, 2]

    # カメラより後ろにある点（奥行きが0以下）は投影できないので、そこだけ印を付ける
    is_in_front_of_camera = camera_z > 0.0

    # 0で割るとエラーになるので、奥行きが0以下の点は仮に1.0に置き換えておく
    # （どうせ後で is_in_front_of_camera により除外されるので値は何でもよい）
    safe_camera_z = np.where(is_in_front_of_camera, camera_z, 1.0)

    # 透視投影の式で画像上の横位置(u)を計算する
    projected_u = (
        alignment_transform["focal_length_x"] * (camera_x / safe_camera_z)
        + alignment_transform["principal_point_x"]
    )
    # 透視投影の式で画像上の縦位置(v)を計算する
    projected_v = (
        alignment_transform["focal_length_y"] * (camera_y / safe_camera_z)
        + alignment_transform["principal_point_y"]
    )

    # 小数の位置を四捨五入して、整数のピクセル座標に直す
    pixel_u = np.rint(projected_u).astype(np.int32)
    pixel_v = np.rint(projected_v).astype(np.int32)

    # 出力する深度画像の幅と高さを取り出す
    image_width = alignment_transform["image_width"]
    image_height = alignment_transform["image_height"]

    # 画像の範囲内に収まっていて、かつカメラの前にある点だけを「有効」とする
    is_valid_point = (
        is_in_front_of_camera
        & (pixel_u >= 0)
        & (pixel_u < image_width)
        & (pixel_v >= 0)
        & (pixel_v < image_height)
    )

    # 有効な点の横位置だけを取り出す
    valid_pixel_u = pixel_u[is_valid_point]
    # 有効な点の縦位置だけを取り出す
    valid_pixel_v = pixel_v[is_valid_point]
    # 有効な点の奥行きだけを取り出す
    valid_depth = camera_z[is_valid_point]

    # --- 4. Zバッファ処理：同じピクセルに複数の点が来たら手前の点を残す ---
    # 「奥行きの逆数（視差）」を使うと、手前ほど値が大きくなる。
    # こうすると「最大値を採用する」だけで手前の点を選べるので、処理が簡単で速い。
    valid_inverse_depth = 1.0 / valid_depth
    # 深度画像と同じ大きさの、1次元に並べた入れ物を 0 で初期化して用意する
    inverse_depth_flat = np.zeros(image_height * image_width, dtype=np.float32)
    # 2次元の位置 (v, u) を 1次元の並びの位置に変換する
    flat_index = valid_pixel_v * image_width + valid_pixel_u
    # 同じ位置に複数の点が来た場合、大きい方（＝手前）を残しながら書き込む
    np.maximum.at(inverse_depth_flat, flat_index, valid_inverse_depth)
    # 1次元に並べた入れ物を、画像の形（高さ x 幅）に戻す
    inverse_depth_image = inverse_depth_flat.reshape(image_height, image_width)

    # --- 5. 点と点の隙間を埋める（設定で有効になっている場合だけ） ---
    if alignment_transform["fill_holes"]:
        # 膨張フィルタの大きさを取り出す
        kernel_size = alignment_transform["fill_kernel_size"]
        # 指定した大きさの四角いフィルタを作る
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        # 膨張処理を行い、周囲の最大値（＝一番手前の面）で隙間を埋める
        inverse_depth_image = cv2.dilate(inverse_depth_image, dilation_kernel)

    # --- 6. 逆数に戻して、メートル単位の深度画像にする ---
    # 値が入っている（0より大きい）ピクセルだけを「深度あり」とする
    has_depth = inverse_depth_image > 0.0
    # 0で割らないように、値が0のところは仮に1.0にしておく
    safe_inverse_depth = np.where(has_depth, inverse_depth_image, 1.0)
    # 逆数を取ってメートル単位の深度に戻す。深度なしのところは 0 を入れる
    depth_image_meters = np.where(has_depth, 1.0 / safe_inverse_depth, 0.0)
    # 32ビット小数に変換して返す
    return depth_image_meters.astype(np.float32)


def project_points_with_colors(point_cloud_array, alignment_transform):
    """
    点群を投影して「投影先のピクセル位置」と「その点の色」を返す。

    これは計測には使わず、アライメントが正しいかを目で確認するための
    verify_alignment.py 専用の関数です。
    """
    # どの列が何かの対応表を取り出す
    column_indices = alignment_transform["column_indices"]
    # X,Y,Z の3列を取り出して32ビット小数にする
    points_xyz = point_cloud_array[
        :, [column_indices["x"], column_indices["y"], column_indices["z"]]
    ].astype(np.float32, copy=False)
    # R,G,B の3列を取り出す
    colors_rgb = point_cloud_array[
        :, [column_indices["red"], column_indices["green"], column_indices["blue"]]
    ]
    # 座標軸をそろえ、外部パラメータを適用する
    points_in_camera_frame = (
        points_xyz @ alignment_transform["rotation_matrix_transposed"]
        + alignment_transform["translation_vector"]
    )
    # カメラ座標系での各成分を取り出す
    camera_x = points_in_camera_frame[:, 0]
    camera_y = points_in_camera_frame[:, 1]
    camera_z = points_in_camera_frame[:, 2]
    # カメラの前にある点かどうかを判定する
    is_in_front_of_camera = camera_z > 0.0
    # 0割りを避けるため、後ろの点の奥行きを仮に1.0にする
    safe_camera_z = np.where(is_in_front_of_camera, camera_z, 1.0)
    # 画像上の横位置を計算して整数にする
    pixel_u = np.rint(
        alignment_transform["focal_length_x"] * (camera_x / safe_camera_z)
        + alignment_transform["principal_point_x"]
    ).astype(np.int32)
    # 画像上の縦位置を計算して整数にする
    pixel_v = np.rint(
        alignment_transform["focal_length_y"] * (camera_y / safe_camera_z)
        + alignment_transform["principal_point_y"]
    ).astype(np.int32)
    # 画像の中に収まっている有効な点だけを選ぶ
    is_valid_point = (
        is_in_front_of_camera
        & (pixel_u >= 0)
        & (pixel_u < alignment_transform["image_width"])
        & (pixel_v >= 0)
        & (pixel_v < alignment_transform["image_height"])
    )
    # 有効な点の横位置・縦位置・奥行き・色をまとめて返す
    return (
        pixel_u[is_valid_point],
        pixel_v[is_valid_point],
        camera_z[is_valid_point],
        colors_rgb[is_valid_point],
    )
