# ============================================================================
# このファイルは何をするものか
# ----------------------------------------------------------------------------
# KFuji RGB-DS データセットの中の「どのファイルがどこにあるか」を
# 一か所にまとめたモジュールです。
# 他のスクリプトは、ファイルのパスを自分で組み立てずに、
# このファイルの関数を呼んでパスを受け取ります。
# こうしておくと、データセットを別の場所に置いた場合でも
# 直す箇所がこのファイルだけで済みます。
#
# 【データセットのファイル名の決まり】
#   生データ（位置合わせ前）: KFuji_RGB-DS_dataset/row data/
#       BD12_inf_201711_086_RGB.jpg   ... 1920x1080 のカラー画像
#       BD12_inf_201711_086_pc.mat    ... 点群（N行8列）
#   切り出し済みデータ: KFuji_RGB-DS_dataset/preprocessed data/
#       images/BD12_inf_201711_086_01_RGBhr.jpg  ... 548x373 の切り出し画像
#       annotations/BD12_inf_201711_086_01_RGB.csv ... その画像のアノテーション
#
#   "BD12_inf_201711_086" の部分を、このプロジェクトでは「シーン名」と呼びます。
#     BD12   ... 果樹園の区画の名前
#     inf    ... 木の下側(inferior)を撮ったか上側(superior=sup)かの区別
#     201711 ... 撮影グループの番号
#     086    ... そのグループの中での連番（フレーム番号）
#   "_01" の部分は、1枚の生画像を3x3に切り出したときの何枚目かを表します。
# ============================================================================

# ファイルパスを扱うための標準ライブラリ
import os
# 正規表現（文字列のパターン照合）を使うための標準ライブラリ
import re


def get_project_root():
    """このプロジェクトの一番上のフォルダ（harvest_nn/）の絶対パスを返す。"""
    # このファイル自身の絶対パスを求める（例: /Users/.../harvest_nn/src/common_paths.py）
    this_file_path = os.path.abspath(__file__)
    # そのファイルが入っているフォルダを求める（例: /Users/.../harvest_nn/src）
    src_directory = os.path.dirname(this_file_path)
    # さらに一つ上のフォルダを求める（例: /Users/.../harvest_nn）
    project_root = os.path.dirname(src_directory)
    # 求めたパスを呼び出し元に返す
    return project_root


def get_default_dataset_root():
    """
    データセットの置き場所（KFuji_RGB-DS_dataset フォルダ）の既定のパスを返す。

    環境変数 KFUJI_DATASET_ROOT が設定されていれば、その場所を使います。
    外付けSSDなど、プロジェクトの外にデータセットを置きたいときに便利です。
        例: export KFUJI_DATASET_ROOT=/mnt/ssd/KFuji_RGB-DS_dataset
    設定されていなければ、従来どおり harvest_nn/KFuji_RGB-DS_dataset を使います。
    （各スクリプトの --dataset-root を指定した場合は、そちらが優先されます）
    """
    # 環境変数が設定されているかを調べる（設定が無ければ None が返る）
    dataset_root_from_environment = os.environ.get("KFUJI_DATASET_ROOT")
    # 設定されていて、空文字でもなければ、その場所を使う
    if dataset_root_from_environment:
        # 相対パスで書かれていても困らないよう、絶対パスに直して返す
        return os.path.abspath(os.path.expanduser(dataset_root_from_environment))
    # プロジェクトの一番上のフォルダを取得する
    project_root = get_project_root()
    # その下にある KFuji_RGB-DS_dataset をつなげてパスを作る
    dataset_root = os.path.join(project_root, "KFuji_RGB-DS_dataset")
    # 作ったパスを返す
    return dataset_root


def get_outputs_directory():
    """計測結果などの出力先フォルダ（outputs/）のパスを返す。無ければ作る。"""
    # プロジェクトの一番上のフォルダを取得する
    project_root = get_project_root()
    # その下の outputs フォルダのパスを作る
    outputs_directory = os.path.join(project_root, "outputs")
    # そのフォルダがまだ無ければ作る（すでにあればそのまま）
    os.makedirs(outputs_directory, exist_ok=True)
    # 作ったパスを返す
    return outputs_directory


def get_raw_data_directory(dataset_root):
    """生データ（位置合わせ前）が入っているフォルダのパスを返す。"""
    # データセットフォルダの下の "row data" をつなげる
    # （"row" は本来 "raw" の綴り間違いだが、配布データがこの名前なのでそのまま使う）
    raw_data_directory = os.path.join(dataset_root, "row data")
    # 作ったパスを返す
    return raw_data_directory


def get_preprocessed_data_directory(dataset_root):
    """切り出し済みデータが入っているフォルダのパスを返す。"""
    # データセットフォルダの下の "preprocessed data" をつなげる
    preprocessed_directory = os.path.join(dataset_root, "preprocessed data")
    # 作ったパスを返す
    return preprocessed_directory


def get_raw_color_image_path(dataset_root, scene_name):
    """シーン名から、生のカラー画像(1920x1080)のパスを作って返す。"""
    # 生データフォルダのパスを取得する
    raw_data_directory = get_raw_data_directory(dataset_root)
    # "シーン名_RGB.jpg" というファイル名を作る
    file_name = scene_name + "_RGB.jpg"
    # フォルダとファイル名をつないだ完全なパスを返す
    return os.path.join(raw_data_directory, file_name)


def get_raw_point_cloud_mat_path(dataset_root, scene_name):
    """シーン名から、生の点群ファイル(*_pc.mat)のパスを作って返す。"""
    # 生データフォルダのパスを取得する
    raw_data_directory = get_raw_data_directory(dataset_root)
    # "シーン名_pc.mat" というファイル名を作る
    file_name = scene_name + "_pc.mat"
    # フォルダとファイル名をつないだ完全なパスを返す
    return os.path.join(raw_data_directory, file_name)


def get_point_cloud_npy_path(dataset_root, scene_name):
    """
    シーン名から、npy形式に変換済みの点群ファイルのパスを作って返す。

    .mat ファイルは MATLAB 形式で読み込みに時間がかかるため、
    prepare_dataset.py であらかじめ numpy の .npy 形式に変換しておきます。
    計測（ステップ1）ではこちらを読み込みます。
    """
    # 変換済みファイルは outputs/point_cloud_npy/ に置く
    npy_directory = os.path.join(get_outputs_directory(), "point_cloud_npy")
    # そのフォルダが無ければ作る
    os.makedirs(npy_directory, exist_ok=True)
    # "シーン名_pc.npy" というファイル名を作る
    file_name = scene_name + "_pc.npy"
    # フォルダとファイル名をつないだ完全なパスを返す
    return os.path.join(npy_directory, file_name)


def get_crop_image_path(dataset_root, crop_name, image_kind="RGBhr"):
    """
    切り出し画像の名前（例 BD12_inf_201711_086_01）から画像ファイルのパスを返す。

    image_kind には "RGBhr" か "RGBp" を指定します。
    どちらも548x373ですが、本プロジェクトでは高解像度側の "RGBhr" を使います。
    """
    # 切り出しデータのフォルダのパスを取得する
    preprocessed_directory = get_preprocessed_data_directory(dataset_root)
    # その下の images フォルダのパスを作る
    images_directory = os.path.join(preprocessed_directory, "images")
    # "切り出し名_RGBhr.jpg" のようなファイル名を作る
    file_name = crop_name + "_" + image_kind + ".jpg"
    # フォルダとファイル名をつないだ完全なパスを返す
    return os.path.join(images_directory, file_name)


def get_crop_annotation_csv_path(dataset_root, crop_name):
    """切り出し画像の名前から、アノテーションCSVファイルのパスを返す。"""
    # 切り出しデータのフォルダのパスを取得する
    preprocessed_directory = get_preprocessed_data_directory(dataset_root)
    # その下の annotations フォルダのパスを作る
    annotations_directory = os.path.join(preprocessed_directory, "annotations")
    # "切り出し名_RGB.csv" というファイル名を作る
    file_name = crop_name + "_RGB.csv"
    # フォルダとファイル名をつないだ完全なパスを返す
    return os.path.join(annotations_directory, file_name)


def get_crop_annotation_xml_path(dataset_root, crop_name):
    """切り出し画像の名前から、アノテーションXMLファイルのパスを返す。"""
    # 切り出しデータのフォルダのパスを取得する
    preprocessed_directory = get_preprocessed_data_directory(dataset_root)
    # その下の square_annotations1 フォルダのパスを作る
    annotations_directory = os.path.join(preprocessed_directory, "square_annotations1")
    # "切り出し名_RGB.xml" というファイル名を作る
    file_name = crop_name + "_RGB.xml"
    # フォルダとファイル名をつないだ完全なパスを返す
    return os.path.join(annotations_directory, file_name)


def list_all_crop_names(dataset_root):
    """
    アノテーションが付いている切り出し画像の名前を、すべて並べたリストを返す。

    返り値の例: ["BD04_inf_201724_004_01", "BD04_inf_201724_004_02", ...]
    """
    # 切り出しデータのフォルダのパスを取得する
    preprocessed_directory = get_preprocessed_data_directory(dataset_root)
    # その下の annotations フォルダのパスを作る
    annotations_directory = os.path.join(preprocessed_directory, "annotations")
    # フォルダが無ければ、データセットがまだ用意できていないので分かりやすく知らせる
    if not os.path.isdir(annotations_directory):
        raise FileNotFoundError(
            "データセットのアノテーションフォルダが見つかりません: " + annotations_directory + "\n"
            + "先に python src/fetch_dataset.py を実行してデータセットを取得してください。\n"
            + "（すでに手元にある場合は --dataset-root か環境変数 KFUJI_DATASET_ROOT で"
            + "置き場所を指定してください）"
        )
    # 結果を入れるための空のリストを用意する
    crop_name_list = []
    # annotations フォルダの中のファイル名を名前順に1つずつ見ていく
    for file_name in sorted(os.listdir(annotations_directory)):
        # ".csv" で終わらないファイル（隠しファイルなど）は飛ばす
        if not file_name.endswith("_RGB.csv"):
            continue
        # ファイル名の末尾の "_RGB.csv"（8文字）を取り除いて切り出し名にする
        crop_name = file_name[: -len("_RGB.csv")]
        # できた名前をリストに追加する
        crop_name_list.append(crop_name)
    # 完成したリストを返す
    return crop_name_list


def get_scene_name_from_crop_name(crop_name):
    """
    切り出し画像の名前から、元の生画像のシーン名を取り出す。

    例: "BD12_inf_201711_086_01" → "BD12_inf_201711_086"
    """
    # 名前をアンダースコアで区切ってリストにする
    name_parts = crop_name.split("_")
    # 最後の要素（切り出し番号 "01" など）を除いた部分をアンダースコアでつなぎ直す
    scene_name = "_".join(name_parts[:-1])
    # できたシーン名を返す
    return scene_name


def list_all_scene_names(dataset_root):
    """
    アノテーションが存在するシーン名（生画像の単位）をすべて並べたリストを返す。

    生データには110シーンありますが、そのうち1シーン
    (BD11_sup_201710_151) は切り出し画像もアノテーションも無いため、
    ここでは除かれて109シーンが返ります。
    """
    # 切り出し名の一覧を取得する
    crop_name_list = list_all_crop_names(dataset_root)
    # 重複を除くために集合（set）を用意する
    scene_name_set = set()
    # 切り出し名を1つずつ見ていく
    for crop_name in crop_name_list:
        # 切り出し名からシーン名を取り出す
        scene_name = get_scene_name_from_crop_name(crop_name)
        # 集合に追加する（同じ名前を何度追加しても1つにまとまる）
        scene_name_set.add(scene_name)
    # 集合をリストに直し、名前順に並べ替えて返す
    return sorted(scene_name_set)


def list_crop_names_for_scene(dataset_root, scene_name):
    """あるシーンから切り出された画像の名前を、すべて並べたリストを返す。"""
    # 切り出し名の一覧を取得する
    all_crop_name_list = list_all_crop_names(dataset_root)
    # 結果を入れるための空のリストを用意する
    matched_crop_name_list = []
    # 切り出し名を1つずつ見ていく
    for crop_name in all_crop_name_list:
        # その切り出しの元シーン名が、探しているシーン名と同じなら
        if get_scene_name_from_crop_name(crop_name) == scene_name:
            # 結果のリストに追加する
            matched_crop_name_list.append(crop_name)
    # 完成したリストを返す
    return matched_crop_name_list


def parse_scene_name(scene_name):
    """
    シーン名を「撮影グループ名」と「フレーム番号」に分解する。

    例: "BD12_inf_201711_086" → ("BD12_inf_201711", 86)

    フレーム番号は、隣り合うシーン同士が似た構図かどうかを判定するために使います
    （＝データ分割のときにリークを防ぐために使います）。
    """
    # 「英数字と_の並び」＋「_」＋「数字だけ」という形かどうかを調べる
    matched = re.match(r"^(.*)_(\d+)$", scene_name)
    # 想定した形でなければ、フレーム番号なしとして返す
    if matched is None:
        return scene_name, -1
    # 1番目のカッコの中身が撮影グループ名
    acquisition_group_name = matched.group(1)
    # 2番目のカッコの中身が連番。文字列なので整数に変換する
    frame_number = int(matched.group(2))
    # 2つの値をまとめて返す
    return acquisition_group_name, frame_number
