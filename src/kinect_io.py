# ============================================================================
# このファイルは何をするものか
# ----------------------------------------------------------------------------
# KFuji RGB-DS データセットの「生データ」を読み込むための関数をまとめたものです。
#
#   ・カラー画像 (*_RGB.jpg, 1920x1080) の読み込み
#   ・点群データ (*_pc.mat) の読み込み
#   ・点群データを高速に読める .npy 形式へ変換する処理
#   ・点群の8つの列（X,Y,Z,R,G,B,IR,S）を取り出す処理
#
# 【点群データ(*_pc.mat)の中身について】
#   MATLAB 形式のファイルで、中には ptCloud という名前の変数が1つだけ入っています。
#   形は「N行 × 8列」で、N は有効な点の数（例: 172,648点）です。
#   1行が1つの3次元点に対応し、8つの列はそれぞれ次の意味を持ちます。
#       0列目: X座標（メートル、右方向がプラス）
#       1列目: Y座標（メートル、カメラから見て前方＝奥行き方向がプラス）
#       2列目: Z座標（メートル、上方向がプラス）
#       3列目: 赤(R) 0〜255
#       4列目: 緑(G) 0〜255
#       5列目: 青(B) 0〜255
#       6列目: 近赤外(IR)の強度
#       7列目: 距離で補正した近赤外強度(S)
#   点は「画像のように格子状に並んでいない」ことに注意してください。
#   （深度が取れなかった点は最初から入っていません）
#   そのため、深度画像を作るには自分でカラー画像の座標系へ投影する必要があります。
#   その処理が depth_alignment.py（ステップ2）です。
# ============================================================================

# ファイルパスを扱うための標準ライブラリ
import os

# 数値計算ライブラリ
import numpy as np
# 画像の読み込みに使うライブラリ
import cv2
# MATLAB形式(.mat)ファイルを読むためのライブラリ
import scipy.io

# 点群ファイルの中に入っている変数の名前（データセットで決まっている）
POINT_CLOUD_VARIABLE_NAME = "ptCloud"


def load_color_image(color_image_path):
    """
    カラー画像をファイルから読み込んで numpy 配列として返す。

    返り値の形は (高さ, 幅, 3) で、色の並びは OpenCV の標準である
    B(青), G(緑), R(赤) の順です。RGBの順ではないので注意してください。
    """
    # OpenCV で画像ファイルを読み込む
    color_image_bgr = cv2.imread(color_image_path, cv2.IMREAD_COLOR)
    # 読み込みに失敗すると None が返るので、その場合はエラーを出して止める
    if color_image_bgr is None:
        raise FileNotFoundError("カラー画像を読み込めませんでした: " + color_image_path)
    # 読み込んだ画像を返す
    return color_image_bgr


def load_point_cloud_from_mat(point_cloud_mat_path):
    """
    MATLAB形式の点群ファイル(*_pc.mat)を読み込み、(N行, 8列) の配列を返す。

    この読み込みは .mat の解凍と解析を伴うため比較的遅いです。
    計測では、あらかじめ .npy に変換したものを使います。
    """
    # scipy を使って .mat ファイルを読み込む（結果は辞書型で返ってくる）
    mat_contents = scipy.io.loadmat(point_cloud_mat_path)
    # 辞書の中に ptCloud という変数が無ければエラーを出して止める
    if POINT_CLOUD_VARIABLE_NAME not in mat_contents:
        raise KeyError(
            "点群ファイルに " + POINT_CLOUD_VARIABLE_NAME + " が見つかりません: "
            + point_cloud_mat_path
        )
    # ptCloud 変数を取り出す
    point_cloud_array = mat_contents[POINT_CLOUD_VARIABLE_NAME]
    # 列数が8でなければ、想定と違うデータなのでエラーを出して止める
    if point_cloud_array.ndim != 2 or point_cloud_array.shape[1] != 8:
        raise ValueError(
            "点群の形が想定(N行8列)と違います: " + str(point_cloud_array.shape)
        )
    # 取り出した配列を返す
    return point_cloud_array


def load_point_cloud_from_npy(point_cloud_npy_path):
    """
    .npy 形式に変換済みの点群ファイルを読み込み、(N行, 8列) の配列を返す。

    .npy は numpy がそのまま読める形式なので、.mat より大幅に速く読み込めます。
    ステップ1（画像取得）の計測ではこちらを使います。
    """
    # numpy の読み込み関数でファイルを読む
    point_cloud_array = np.load(point_cloud_npy_path)
    # 読み込んだ配列を返す
    return point_cloud_array


def convert_point_cloud_mat_to_npy(point_cloud_mat_path, point_cloud_npy_path,
                                   use_float32=True):
    """
    点群ファイルを .mat から .npy へ変換して保存する。

    use_float32 が True のとき、64ビット小数(float64)から
    32ビット小数(float32)に変換してからファイルに保存します。
    こうするとファイルサイズが半分になり、読み込みも速くなります。
    精度はミリメートル以下まで十分に保たれるので、実用上の問題はありません。
    """
    # .mat ファイルから点群を読み込む
    point_cloud_array = load_point_cloud_from_mat(point_cloud_mat_path)
    # float32 に変換する設定なら、型を変換する
    if use_float32:
        point_cloud_array = point_cloud_array.astype(np.float32)
    # 保存先のフォルダのパスを求める
    output_directory = os.path.dirname(point_cloud_npy_path)
    # 保存先のフォルダが無ければ作る
    os.makedirs(output_directory, exist_ok=True)
    # numpy の保存関数で .npy として書き出す
    np.save(point_cloud_npy_path, point_cloud_array)
    # 何点保存したかを呼び出し元に返す（進捗表示に使う）
    return point_cloud_array.shape[0]


def extract_xyz_coordinates(point_cloud_array, column_indices):
    """
    点群の配列から X, Y, Z の3列だけを取り出して (N行, 3列) の配列で返す。

    column_indices は camera_params.yaml の point_cloud_columns の内容
    （どの列が何を表すかの対応表）です。
    """
    # X座標が何列目かを取り出す
    x_column = column_indices["x"]
    # Y座標が何列目かを取り出す
    y_column = column_indices["y"]
    # Z座標が何列目かを取り出す
    z_column = column_indices["z"]
    # 3つの列をこの順番で並べた新しい配列を作る
    xyz_array = point_cloud_array[:, [x_column, y_column, z_column]]
    # 作った配列を返す
    return xyz_array


def extract_rgb_colors(point_cloud_array, column_indices):
    """
    点群の配列から R, G, B の3列だけを取り出して (N行, 3列) の配列で返す。

    この色情報は、深度アライメントが正しくできているかを
    検証する（verify_alignment.py）ためだけに使います。
    """
    # 赤が何列目かを取り出す
    red_column = column_indices["red"]
    # 緑が何列目かを取り出す
    green_column = column_indices["green"]
    # 青が何列目かを取り出す
    blue_column = column_indices["blue"]
    # 3つの列をこの順番で並べた新しい配列を作る
    rgb_array = point_cloud_array[:, [red_column, green_column, blue_column]]
    # 作った配列を返す
    return rgb_array
