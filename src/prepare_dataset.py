# ============================================================================
# このファイルは何をするものか
# ----------------------------------------------------------------------------
# 【(A) データ準備スクリプト】です。次の3つの仕事をします。
#
#   1. アノテーション形式の変換
#        KFuji のアノテーション(csv)を、YOLOが読める形式(txt)に変換します。
#        csv:  個体ID, 左上x, 左上y, 幅, 高さ, クラス番号   （単位はピクセル）
#        YOLO: クラス番号 中心x 中心y 幅 高さ             （0〜1に正規化）
#
#   2. train / val / test への分割
#        果樹園を移動しながら連続撮影したデータなので、
#        何も考えずにランダム分割すると「同じりんごが学習用とテスト用の
#        両方に入る」データリークが起きます。
#        （実際、データセットに最初から付いている sets/ フォルダの分割は
#          テスト用93シーンすべてが学習用にも入っており、使えません）
#        そこで、このスクリプトでは次の3段階で分割単位を作ります。
#
#          段階1: 同じ生画像から切り出した9枚は必ず同じセットに入れる
#          段階2: 同じ撮影グループ（例 BD12_inf_201711）の中で、フレーム番号順に
#                 並べて連続する数シーンを1つのブロックにまとめる
#                 （果樹園を移動しながら3フレーム間隔で撮影されているため、
#                   隣り合うシーンは同じ木・同じりんごが写っている）
#
#        【注意：アノテーションの個体IDについて】
#          csvの1列目には個体IDが入っていますが、これは
#          「シーンごとに連番のブロックを割り当てたもの」であって、
#          シーンをまたいで同じりんごを追跡した番号ではありません。
#          （例: BD11_inf_201710_003 は ID 9244〜9361、
#                BD11_inf_201710_039 は ID 9237〜9430 と、
#                番号の範囲がたまたま重なっているだけで、
#                同じIDの枠は画像上の全く違う位置にあります）
#          そのため、個体IDを「同じりんごかどうか」の判定には使えません。
#          1枚の生画像から切り出した9枚の間では、同じIDは同じりんごを指します。
#
#        こうして作った「単位」ごとに、乱数シードを固定してシャッフルし、
#        train / val / test に振り分けます。
#        最後に、異なるセットの境界にあるシーンを指定枚数だけ捨てて
#        （ガードバンド）、境目のリークも防ぎます。
#
#   3. 点群ファイルの変換
#        計測スクリプトで使うために、点群を .mat から .npy に変換します。
#
# 【実行例】
#   python src/prepare_dataset.py
#   python src/prepare_dataset.py --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2
#   python src/prepare_dataset.py --seed 123 --block-size 5 --guard-scenes 0
# ============================================================================

# コマンドライン引数を扱うための標準ライブラリ
import argparse
# 乱数を扱うための標準ライブラリ
import random
# ファイルやフォルダを操作するための標準ライブラリ
import os
# 結果を JSON ファイルに保存するための標準ライブラリ
import json
# XMLファイル（画像サイズの読み取りに使う）を解析するための標準ライブラリ
import xml.etree.ElementTree as ElementTree

# 自作のモジュール（データセットのパスを扱う）を読み込む
import common_paths
# 自作のモジュール（点群の読み込みと変換）を読み込む
import kinect_io


# --------------------------------------------------------------------------
# ここからアノテーション変換に関する関数
# --------------------------------------------------------------------------

def read_annotation_csv(csv_file_path):
    """
    アノテーションCSVを読み込み、1行ずつの内容をリストにして返す。

    返り値は辞書のリストで、1つの辞書が1個のりんごを表します。
        apple_id : りんごの個体ID（同じ番号は同じりんごを指す）
        x_min    : 枠の左端のx座標（ピクセル）
        y_min    : 枠の上端のy座標（ピクセル）
        width    : 枠の横幅（ピクセル）
        height   : 枠の縦幅（ピクセル）
    """
    # 結果を入れるための空のリストを用意する
    annotation_list = []
    # CSVファイルを開く
    with open(csv_file_path, "r", encoding="utf-8") as opened_file:
        # ファイルの中身を1行ずつ順番に読む
        for one_line in opened_file:
            # 行の前後の改行や空白を取り除く
            stripped_line = one_line.strip()
            # 空の行は飛ばす
            if stripped_line == "":
                continue
            # カンマで区切って項目のリストにする
            field_list = stripped_line.split(",")
            # 項目が6個未満の行は形式が違うので飛ばす
            if len(field_list) < 6:
                continue
            # 1個のりんごの情報を辞書にまとめる
            one_annotation = {
                # 1列目は個体ID（整数に変換する）
                "apple_id": int(float(field_list[0])),
                # 2列目は枠の左端のx座標
                "x_min": float(field_list[1]),
                # 3列目は枠の上端のy座標
                "y_min": float(field_list[2]),
                # 4列目は枠の横幅
                "width": float(field_list[3]),
                # 5列目は枠の縦幅
                "height": float(field_list[4]),
            }
            # できた辞書をリストに追加する
            annotation_list.append(one_annotation)
    # 完成したリストを返す
    return annotation_list


def read_image_size_from_xml(xml_file_path):
    """
    アノテーションXMLファイルから、画像の幅と高さを読み取って返す。

    画像ファイル自体を開くより速いので、こちらを使います。
    """
    # XMLファイルを解析して木構造として読み込む
    xml_tree = ElementTree.parse(xml_file_path)
    # 木構造の一番上の要素を取り出す
    root_element = xml_tree.getroot()
    # <size> という要素を探す
    size_element = root_element.find("size")
    # <size> の中の <width> の中身を整数にする
    image_width = int(size_element.find("width").text)
    # <size> の中の <height> の中身を整数にする
    image_height = int(size_element.find("height").text)
    # 幅と高さをまとめて返す
    return image_width, image_height


def convert_one_annotation_file_to_yolo(annotation_list, image_width, image_height,
                                        class_id=0, minimum_box_size_pixels=2.0):
    """
    1枚分のアノテーションを、YOLO形式の行のリストに変換する。

    YOLO形式は1行に1個の物体を書き、次の5つの値を半角スペースで区切ります。
        クラス番号 中心x 中心y 幅 高さ
    座標と大きさは画像サイズで割って 0〜1 の範囲に正規化します。
    """
    # 結果を入れるための空のリストを用意する
    yolo_line_list = []
    # アノテーションを1個ずつ順番に処理する
    for one_annotation in annotation_list:
        # 枠の左端の座標を取り出す
        box_x1 = one_annotation["x_min"]
        # 枠の上端の座標を取り出す
        box_y1 = one_annotation["y_min"]
        # 枠の右端の座標を計算する（左端＋横幅）
        box_x2 = one_annotation["x_min"] + one_annotation["width"]
        # 枠の下端の座標を計算する（上端＋縦幅）
        box_y2 = one_annotation["y_min"] + one_annotation["height"]

        # 枠が画像の外にはみ出している場合は、画像の内側に収まるように切り詰める
        if box_x1 < 0.0:
            box_x1 = 0.0
        if box_y1 < 0.0:
            box_y1 = 0.0
        if box_x2 > image_width:
            box_x2 = float(image_width)
        if box_y2 > image_height:
            box_y2 = float(image_height)

        # 切り詰めた後の横幅を計算する
        clipped_width = box_x2 - box_x1
        # 切り詰めた後の縦幅を計算する
        clipped_height = box_y2 - box_y1

        # 幅か高さが小さすぎる枠（画像の端でほとんど切れている枠）は使わない
        if clipped_width < minimum_box_size_pixels or clipped_height < minimum_box_size_pixels:
            continue

        # 枠の中心のx座標を求めて、画像の幅で割って0〜1にする
        normalized_center_x = (box_x1 + clipped_width / 2.0) / image_width
        # 枠の中心のy座標を求めて、画像の高さで割って0〜1にする
        normalized_center_y = (box_y1 + clipped_height / 2.0) / image_height
        # 枠の横幅を画像の幅で割って0〜1にする
        normalized_width = clipped_width / image_width
        # 枠の縦幅を画像の高さで割って0〜1にする
        normalized_height = clipped_height / image_height

        # 5つの値を半角スペースでつないだ1行の文字列を作る（小数点以下6桁）
        one_yolo_line = "%d %.6f %.6f %.6f %.6f" % (
            class_id,
            normalized_center_x,
            normalized_center_y,
            normalized_width,
            normalized_height,
        )
        # できた行をリストに追加する
        yolo_line_list.append(one_yolo_line)
    # 完成したリストを返す
    return yolo_line_list


# --------------------------------------------------------------------------
# ここから分割（train/val/test）に関する関数
# --------------------------------------------------------------------------

class UnionFind:
    """
    「どれとどれが同じグループか」を管理するための、よく使われる仕組み。

    たとえば「シーンAとシーンBは同じりんごが写っている」という情報を
    次々に登録していくと、最終的に「まとめて扱うべきシーンの集まり」が
    自動的に求まります。
    """

    def __init__(self, item_list):
        """最初は、すべての要素がそれぞれ別々のグループに属している状態にする。"""
        # 各要素の「親」を記録する辞書を作る
        self.parent_of = {}
        # 要素を1つずつ順番に見ていく
        for one_item in item_list:
            # 最初は自分自身が親（＝自分だけのグループ）とする
            self.parent_of[one_item] = one_item

    def find_root(self, item):
        """ある要素が属するグループの代表（根）を返す。"""
        # 自分が親になるまで、親をたどり続ける
        while self.parent_of[item] != item:
            # 途中の要素の親を、一段上の親に付け替えて次回以降を速くする
            self.parent_of[item] = self.parent_of[self.parent_of[item]]
            # 親に進む
            item = self.parent_of[item]
        # 見つかった代表を返す
        return item

    def union(self, item_a, item_b):
        """2つの要素を同じグループにまとめる。"""
        # それぞれの代表を求める
        root_a = self.find_root(item_a)
        root_b = self.find_root(item_b)
        # 代表が違うなら（＝別々のグループなら）、片方をもう片方につなぐ
        if root_a != root_b:
            self.parent_of[root_a] = root_b

    def get_groups(self):
        """まとまったグループを、代表ごとのリストにして返す。"""
        # 代表 → その仲間のリスト、という辞書を作る
        groups_by_root = {}
        # 全ての要素を1つずつ見ていく
        for one_item in self.parent_of:
            # その要素の代表を求める
            root = self.find_root(one_item)
            # 代表がまだ辞書に無ければ、空のリストを用意する
            if root not in groups_by_root:
                groups_by_root[root] = []
            # 仲間のリストに自分を追加する
            groups_by_root[root].append(one_item)
        # 代表ごとのリストをまとめて返す
        return groups_by_root


def build_split_units(scene_name_list, block_size):
    """
    分割の「単位」を作る。同じ単位に入ったシーンは必ず同じセットに入ります。

    同じ撮影グループの中で、フレーム番号順に並べたとき連続する block_size 枚を
    1つの単位にまとめます。果樹園を移動しながら撮影しているため、
    隣り合うシーンには同じ木・同じりんごが写っている可能性が高いからです。
    """
    # シーンをまとめるための仕組みを用意する
    union_find = UnionFind(scene_name_list)

    # 撮影グループごとにシーンを分けるための辞書を用意する
    scenes_by_acquisition_group = {}
    # シーンを1つずつ見ていく
    for one_scene_name in scene_name_list:
        # シーン名を「撮影グループ名」と「フレーム番号」に分解する
        acquisition_group_name, frame_number = common_paths.parse_scene_name(one_scene_name)
        # その撮影グループがまだ辞書に無ければ、空のリストを用意する
        if acquisition_group_name not in scenes_by_acquisition_group:
            scenes_by_acquisition_group[acquisition_group_name] = []
        # 撮影グループのリストに、シーン名とフレーム番号の組を追加する
        scenes_by_acquisition_group[acquisition_group_name].append(
            (frame_number, one_scene_name)
        )

    # 撮影グループを1つずつ順番に処理する
    for acquisition_group_name in sorted(scenes_by_acquisition_group):
        # そのグループのシーンを、フレーム番号の小さい順に並べ替える
        scenes_in_group = sorted(scenes_by_acquisition_group[acquisition_group_name])

        # --- フレーム番号順に連続する block_size 枚を1つの単位にまとめる ---
        # 並べ替えたシーンを、先頭から順番に番号を付けて見ていく
        for position_index in range(len(scenes_in_group)):
            # 何番目のブロックに属するかを計算する（0,0,0,1,1,1,... のようになる）
            block_index = position_index // block_size
            # 同じブロックの先頭のシーンの位置を求める
            first_position_of_block = block_index * block_size
            # 自分がブロックの先頭でなければ、先頭のシーンと同じ単位にまとめる
            if position_index != first_position_of_block:
                union_find.union(
                    scenes_in_group[first_position_of_block][1],
                    scenes_in_group[position_index][1],
                )

    # まとまったグループを取り出す
    groups_by_root = union_find.get_groups()
    # 結果を入れるための空のリストを用意する
    split_unit_list = []
    # グループを代表の名前順に1つずつ見ていく（順番を固定して再現性を保つため）
    for root_name in sorted(groups_by_root):
        # そのグループに属するシーンを名前順に並べてリストに追加する
        split_unit_list.append(sorted(groups_by_root[root_name]))
    # 完成したリストを返す
    return split_unit_list


def assign_units_to_splits(split_unit_list, scene_to_crop_count,
                           train_ratio, val_ratio, test_ratio, random_seed):
    """
    分割の単位を train / val / test に振り分ける。

    振り分け方は次のとおりです。
        1. 乱数シードを固定して単位の順番をシャッフルする
        2. 大きい単位から順に、「目標枚数に対して一番足りていないセット」へ入れる
    こうすると、指定した比率にできるだけ近い枚数配分になります。
    """
    # 全部で何枚の切り出し画像があるかを数える
    total_crop_count = 0
    # シーンごとの枚数を1つずつ足していく
    for one_scene_name in scene_to_crop_count:
        total_crop_count = total_crop_count + scene_to_crop_count[one_scene_name]

    # それぞれのセットの目標枚数を計算する
    target_crop_count = {
        "train": total_crop_count * train_ratio,
        "val": total_crop_count * val_ratio,
        "test": total_crop_count * test_ratio,
    }

    # 単位ごとの画像枚数を数えて、(枚数, 単位) の組のリストを作る
    unit_with_size_list = []
    # 単位を1つずつ見ていく
    for one_unit in split_unit_list:
        # その単位に含まれる画像枚数を数えるための変数を0で用意する
        crop_count_in_unit = 0
        # 単位に含まれるシーンを1つずつ見ていく
        for one_scene_name in one_unit:
            # そのシーンの画像枚数を足す
            crop_count_in_unit = crop_count_in_unit + scene_to_crop_count[one_scene_name]
        # 枚数と単位の組をリストに追加する
        unit_with_size_list.append((crop_count_in_unit, one_unit))

    # 乱数の種を固定した乱数生成器を用意する（毎回同じ結果になるようにするため）
    random_generator = random.Random(random_seed)
    # 単位の順番をシャッフルする
    random_generator.shuffle(unit_with_size_list)
    # 大きい単位から先に配ると比率が合いやすいので、枚数の多い順に並べ替える
    unit_with_size_list.sort(key=get_first_element, reverse=True)

    # 現在の割り当て枚数を数えるための辞書を0で用意する
    current_crop_count = {"train": 0, "val": 0, "test": 0}
    # シーン名 → どのセットか、を記録する辞書を用意する
    scene_to_split_name = {}

    # 単位を1つずつ順番に配っていく
    for crop_count_in_unit, one_unit in unit_with_size_list:
        # 一番不足しているセットを探すための変数を用意する
        best_split_name = "train"
        # 不足量の最大値を、あり得ないほど小さい値で初期化する
        largest_shortage = -1.0e18
        # 3つのセットを順番に調べる
        for one_split_name in ["train", "val", "test"]:
            # そのセットの目標枚数に対する不足量を計算する
            shortage = target_crop_count[one_split_name] - current_crop_count[one_split_name]
            # これまでで一番不足しているセットなら、覚えておく
            if shortage > largest_shortage:
                largest_shortage = shortage
                best_split_name = one_split_name
        # 一番不足しているセットに、この単位のシーンを全部入れる
        for one_scene_name in one_unit:
            scene_to_split_name[one_scene_name] = best_split_name
        # そのセットの割り当て枚数を増やす
        current_crop_count[best_split_name] = current_crop_count[best_split_name] + crop_count_in_unit

    # 完成した対応表を返す
    return scene_to_split_name


def get_first_element(pair):
    """
    (枚数, 単位) という組から、枚数（1つ目の要素）を取り出す。

    並べ替えのときに「何を基準に並べるか」を指定するために使います。
    （ラムダ式を避けて、普通の関数として書いています）
    """
    # 組の1つ目の要素を返す
    return pair[0]


def apply_guard_band(scene_name_list, scene_to_split_name, guard_scenes):
    """
    異なるセットの境目にあるシーンを、指定した枚数だけ「除外」にする。

    フレーム番号順に並べたとき、たとえば
        ... 041(train) 047(train) | 050(test) 053(test) ...
    のようにセットが切り替わる場所があります。
    この境目の前後のシーンは撮影位置が近く、構図が似ている可能性が高いため、
    どちらのセットにも入れずに捨てます（これをガードバンドと呼びます）。

    guard_scenes に 0 を指定すると、何も捨てません。
    """
    # 捨てる枚数が0以下なら、何もせずそのまま返す
    if guard_scenes <= 0:
        return scene_to_split_name, []

    # 撮影グループごとにシーンを分けるための辞書を用意する
    scenes_by_acquisition_group = {}
    # シーンを1つずつ見ていく
    for one_scene_name in scene_name_list:
        # シーン名を撮影グループ名とフレーム番号に分解する
        acquisition_group_name, frame_number = common_paths.parse_scene_name(one_scene_name)
        # その撮影グループがまだ辞書に無ければ、空のリストを用意する
        if acquisition_group_name not in scenes_by_acquisition_group:
            scenes_by_acquisition_group[acquisition_group_name] = []
        # フレーム番号とシーン名の組を追加する
        scenes_by_acquisition_group[acquisition_group_name].append(
            (frame_number, one_scene_name)
        )

    # 除外することにしたシーンの名前を入れるための空のリストを用意する
    excluded_scene_name_list = []

    # 境目を判定するために、除外を始める前の割り当てを控えとして取っておく。
    # （除外した結果を判定に使ってしまうと、「除外」と「test」が違うセットだと
    #   みなされて、除外が次々に連鎖してしまうため）
    original_split_name_of = dict(scene_to_split_name)

    # 撮影グループを1つずつ順番に処理する
    for acquisition_group_name in sorted(scenes_by_acquisition_group):
        # そのグループのシーンをフレーム番号順に並べ替える
        scenes_in_group = sorted(scenes_by_acquisition_group[acquisition_group_name])
        # 隣り合うシーンの組を順番に調べる
        for position_index in range(len(scenes_in_group) - 1):
            # 手前のシーンの名前を取り出す
            earlier_scene_name = scenes_in_group[position_index][1]
            # 次のシーンの名前を取り出す
            later_scene_name = scenes_in_group[position_index + 1][1]
            # 手前のシーンがどのセットだったかを、控えの方から調べる
            earlier_split_name = original_split_name_of[earlier_scene_name]
            # 次のシーンがどのセットだったかを、控えの方から調べる
            later_split_name = original_split_name_of[later_scene_name]
            # 同じセットなら境目ではないので、何もしない
            if earlier_split_name == later_split_name:
                continue
            # 境目なので、境目より後ろ側のシーンを guard_scenes 枚だけ除外する
            for offset in range(guard_scenes):
                # 除外する候補の位置を計算する
                candidate_position = position_index + 1 + offset
                # 位置がリストの範囲を超えたら、そこで終わりにする
                if candidate_position >= len(scenes_in_group):
                    break
                # 除外する候補のシーン名を取り出す
                candidate_scene_name = scenes_in_group[candidate_position][1]
                # すでに除外済みなら、何もしない
                if scene_to_split_name[candidate_scene_name] == "excluded":
                    continue
                # そのシーンを「除外」に変更する
                scene_to_split_name[candidate_scene_name] = "excluded"
                # 除外したシーンの名前を記録する
                excluded_scene_name_list.append(candidate_scene_name)

    # 更新した対応表と、除外したシーンのリストを返す
    return scene_to_split_name, sorted(excluded_scene_name_list)


# --------------------------------------------------------------------------
# ここからファイル出力に関する関数
# --------------------------------------------------------------------------

def create_symbolic_link(link_path, target_path, use_copy):
    """
    画像ファイルへのショートカット（シンボリックリンク）を作る。

    データセットは合計2.9GBあるので、画像をコピーすると
    ディスクを無駄に使ってしまいます。そこでリンクだけを作ります。
    use_copy が True のときは、リンクではなく実際にコピーします。
    """
    # すでに同じ名前のファイルやリンクがあれば、先に消す
    if os.path.islink(link_path) or os.path.exists(link_path):
        os.remove(link_path)
    # コピーする設定なら、ファイルをコピーする
    if use_copy:
        # ファイルをコピーするための標準ライブラリを読み込む
        import shutil
        # 中身と更新日時をコピーする
        shutil.copy2(target_path, link_path)
    # コピーしない設定なら、リンクを作る
    else:
        os.symlink(target_path, link_path)


def write_text_lines(file_path, line_list):
    """文字列のリストを、1行ずつファイルに書き出す。"""
    # 保存先のフォルダのパスを求める
    output_directory = os.path.dirname(file_path)
    # 保存先のフォルダが無ければ作る
    if output_directory != "":
        os.makedirs(output_directory, exist_ok=True)
    # ファイルを書き込みモードで開く
    with open(file_path, "w", encoding="utf-8") as opened_file:
        # 行を1つずつ順番に書き出す
        for one_line in line_list:
            # 行の内容と改行を書き込む
            opened_file.write(one_line + "\n")


# --------------------------------------------------------------------------
# ここからメインの処理
# --------------------------------------------------------------------------

def main():
    """コマンドから実行されたときに動く、このスクリプトの本体。"""
    # コマンドライン引数の設定を作る
    argument_parser = argparse.ArgumentParser(
        description="KFuji RGB-DS データセットを YOLO 形式に変換し、train/val/test に分割します。"
    )
    # データセットの場所を指定する引数を追加する
    argument_parser.add_argument(
        "--dataset-root",
        default=None,
        help="KFuji_RGB-DS_dataset フォルダのパス。省略するとプロジェクト内の既定の場所を使います",
    )
    # 出力先を指定する引数を追加する
    argument_parser.add_argument(
        "--output-dir",
        default=None,
        help="YOLO形式のデータセットを作る場所。省略すると outputs/dataset_yolo になります",
    )
    # 学習用の比率を指定する引数を追加する
    argument_parser.add_argument(
        "--train-ratio", type=float, default=0.70, help="学習用に使う割合（既定値 0.70）"
    )
    # 検証用の比率を指定する引数を追加する
    argument_parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="検証用に使う割合（既定値 0.15）"
    )
    # テスト用の比率を指定する引数を追加する
    argument_parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="テスト用に使う割合（既定値 0.15）"
    )
    # 連続何シーンを1ブロックにまとめるかを指定する引数を追加する
    argument_parser.add_argument(
        "--block-size",
        type=int,
        default=3,
        help="フレーム番号順に連続する何シーンを1つの単位にまとめるか（既定値 3）",
    )
    # ガードバンドの枚数を指定する引数を追加する
    argument_parser.add_argument(
        "--guard-scenes",
        type=int,
        default=1,
        help="異なるセットの境目で捨てるシーン数（既定値 1。0にすると捨てません）",
    )
    # 乱数シードを指定する引数を追加する
    argument_parser.add_argument(
        "--seed", type=int, default=42, help="乱数の種。同じ値なら毎回同じ分割になります（既定値 42）"
    )
    # 点群の変換対象を指定する引数を追加する
    argument_parser.add_argument(
        "--convert-point-clouds",
        default="test",
        choices=["test", "all", "none"],
        help="点群を .mat から .npy に変換する対象（既定値 test。計測はtestセットだけ使います）",
    )
    # 画像をコピーするかリンクにするかを指定する引数を追加する
    argument_parser.add_argument(
        "--copy-images",
        action="store_true",
        help="指定するとリンクではなく実際に画像をコピーします（ディスクを多く使います）",
    )
    # 使う画像の種類を指定する引数を追加する
    argument_parser.add_argument(
        "--image-kind",
        default="RGBhr",
        choices=["RGBhr", "RGBp"],
        help="学習に使う切り出し画像の種類（既定値 RGBhr）",
    )
    # 実際に引数を読み取る
    parsed_arguments = argument_parser.parse_args()

    # 比率の合計が1になっているか確認する
    ratio_sum = parsed_arguments.train_ratio + parsed_arguments.val_ratio + parsed_arguments.test_ratio
    # 合計が1から大きくずれていたらエラーを出して止める
    if abs(ratio_sum - 1.0) > 0.001:
        raise ValueError("train/val/test の比率の合計が1になっていません: " + str(ratio_sum))

    # データセットの場所を決める（指定が無ければ既定の場所を使う）
    if parsed_arguments.dataset_root is None:
        dataset_root = common_paths.get_default_dataset_root()
    else:
        dataset_root = parsed_arguments.dataset_root

    # 出力先を決める（指定が無ければ outputs/dataset_yolo を使う）
    if parsed_arguments.output_dir is None:
        output_directory = os.path.join(common_paths.get_outputs_directory(), "dataset_yolo")
    else:
        output_directory = parsed_arguments.output_dir

    # 処理の開始を画面に表示する
    print("=" * 70)
    print("データ準備を開始します")
    print("=" * 70)
    print("  データセットの場所 : " + dataset_root)
    print("  出力先             : " + output_directory)
    print("  乱数シード         : " + str(parsed_arguments.seed))
    print("")

    # ---------------- 手順1: アノテーションを読み込んで YOLO 形式に変換する ----
    print("[手順1] アノテーション(csv)を YOLO 形式に変換します")

    # 切り出し画像の名前を全部取得する
    all_crop_name_list = common_paths.list_all_crop_names(dataset_root)
    # 画像名 → YOLO形式の行のリスト、という辞書を用意する
    crop_to_yolo_lines = {}
    # シーン名 → そのシーンの切り出し枚数、という辞書を用意する
    scene_to_crop_count = {}
    # 変換したりんごの総数を数えるための変数を0で用意する
    total_apple_count = 0

    # 切り出し画像を1枚ずつ順番に処理する
    for one_crop_name in all_crop_name_list:
        # その画像のアノテーションCSVのパスを求める
        csv_file_path = common_paths.get_crop_annotation_csv_path(dataset_root, one_crop_name)
        # CSVを読み込む
        annotation_list = read_annotation_csv(csv_file_path)
        # その画像のアノテーションXMLのパスを求める
        xml_file_path = common_paths.get_crop_annotation_xml_path(dataset_root, one_crop_name)
        # XMLから画像の幅と高さを読み取る
        image_width, image_height = read_image_size_from_xml(xml_file_path)
        # アノテーションを YOLO 形式に変換する
        yolo_line_list = convert_one_annotation_file_to_yolo(
            annotation_list, image_width, image_height
        )
        # 変換結果を辞書に記録する
        crop_to_yolo_lines[one_crop_name] = yolo_line_list
        # 変換したりんごの数を足す
        total_apple_count = total_apple_count + len(yolo_line_list)

        # この画像の元になったシーン名を求める
        scene_name = common_paths.get_scene_name_from_crop_name(one_crop_name)
        # そのシーンがまだ辞書に無ければ、0 を用意する
        if scene_name not in scene_to_crop_count:
            scene_to_crop_count[scene_name] = 0
        # このシーンの切り出し枚数を1枚ぶん増やす
        scene_to_crop_count[scene_name] = scene_to_crop_count[scene_name] + 1

    # 変換結果の要約を表示する
    print("  切り出し画像 : " + str(len(all_crop_name_list)) + " 枚")
    print("  元のシーン   : " + str(len(scene_to_crop_count)) + " 枚（生画像の数）")
    print("  りんごの枠   : " + str(total_apple_count) + " 個")
    print("")

    # ---------------- 手順2: 分割の単位を作る ------------------------------
    print("[手順2] データリークを防ぐため、まとめて扱うべきシーンの単位を作ります")

    # シーン名の一覧を名前順に作る
    scene_name_list = sorted(scene_to_crop_count)
    # 分割の単位を作る
    split_unit_list = build_split_units(scene_name_list, parsed_arguments.block_size)
    # 単位の数を表示する
    print("  分割の単位の数 : " + str(len(split_unit_list)))
    # 一番大きい単位に何シーン入っているかを調べるための変数を0で用意する
    largest_unit_size = 0
    # 単位を1つずつ見て、一番大きいものを探す
    for one_unit in split_unit_list:
        if len(one_unit) > largest_unit_size:
            largest_unit_size = len(one_unit)
    # 一番大きい単位の大きさを表示する
    print("  最大の単位     : " + str(largest_unit_size) + " シーン")
    print("")

    # ---------------- 手順3: train / val / test に振り分ける ----------------
    print("[手順3] 単位ごとに train / val / test へ振り分けます")

    # 単位をセットに振り分ける
    scene_to_split_name = assign_units_to_splits(
        split_unit_list,
        scene_to_crop_count,
        parsed_arguments.train_ratio,
        parsed_arguments.val_ratio,
        parsed_arguments.test_ratio,
        parsed_arguments.seed,
    )
    # 境目のシーンを捨てる（ガードバンド）
    scene_to_split_name, excluded_scene_name_list = apply_guard_band(
        scene_name_list, scene_to_split_name, parsed_arguments.guard_scenes
    )
    # 捨てたシーンの数を表示する
    print("  ガードバンドで除外したシーン : " + str(len(excluded_scene_name_list)) + " 枚")
    print("")

    # ---------------- 手順4: YOLO用のフォルダを作ってファイルを書き出す ------
    print("[手順4] YOLO が読み込める形にファイルを配置します")

    # セットごとの画像名リストを入れるための辞書を用意する
    crop_names_by_split = {"train": [], "val": [], "test": [], "excluded": []}
    # セットごとのシーン名リストを入れるための辞書を用意する
    scene_names_by_split = {"train": [], "val": [], "test": [], "excluded": []}
    # セットごとのりんごの数を数えるための辞書を用意する
    apple_count_by_split = {"train": 0, "val": 0, "test": 0, "excluded": 0}

    # 切り出し画像を1枚ずつ見て、どのセットに属するかを調べる
    for one_crop_name in all_crop_name_list:
        # 元のシーン名を求める
        scene_name = common_paths.get_scene_name_from_crop_name(one_crop_name)
        # そのシーンが属するセットの名前を調べる
        split_name = scene_to_split_name[scene_name]
        # そのセットの画像名リストに追加する
        crop_names_by_split[split_name].append(one_crop_name)
        # そのセットのりんごの数を足す
        apple_count_by_split[split_name] = apple_count_by_split[split_name] + len(
            crop_to_yolo_lines[one_crop_name]
        )
    # シーンを1つずつ見て、セットごとのシーン名リストを作る
    for one_scene_name in scene_name_list:
        scene_names_by_split[scene_to_split_name[one_scene_name]].append(one_scene_name)

    # train / val / test の3つのセットについて、フォルダを作りファイルを配置する
    for split_name in ["train", "val", "test"]:
        # 画像を置くフォルダのパスを作る
        images_directory = os.path.join(output_directory, "images", split_name)
        # ラベルを置くフォルダのパスを作る
        labels_directory = os.path.join(output_directory, "labels", split_name)
        # 画像用のフォルダを作る
        os.makedirs(images_directory, exist_ok=True)
        # ラベル用のフォルダを作る
        os.makedirs(labels_directory, exist_ok=True)
        # そのセットの画像を1枚ずつ順番に処理する
        for one_crop_name in crop_names_by_split[split_name]:
            # 元の画像ファイルのパスを求める
            source_image_path = common_paths.get_crop_image_path(
                dataset_root, one_crop_name, parsed_arguments.image_kind
            )
            # 配置先の画像ファイルのパスを作る
            link_image_path = os.path.join(images_directory, one_crop_name + ".jpg")
            # リンク（またはコピー）を作る
            create_symbolic_link(
                link_image_path, os.path.abspath(source_image_path), parsed_arguments.copy_images
            )
            # 配置先のラベルファイルのパスを作る
            label_file_path = os.path.join(labels_directory, one_crop_name + ".txt")
            # YOLO形式の行をラベルファイルとして書き出す
            write_text_lines(label_file_path, crop_to_yolo_lines[one_crop_name])
        # このセットの処理が終わったことを表示する
        print(
            "  "
            + split_name.ljust(5)
            + " : 画像 "
            + str(len(crop_names_by_split[split_name])).rjust(4)
            + " 枚 / シーン "
            + str(len(scene_names_by_split[split_name])).rjust(3)
            + " 枚 / りんご "
            + str(apple_count_by_split[split_name]).rjust(5)
            + " 個"
        )
    # 除外されたシーンの情報も表示する
    print(
        "  除外  : 画像 "
        + str(len(crop_names_by_split["excluded"])).rjust(4)
        + " 枚 / シーン "
        + str(len(scene_names_by_split["excluded"])).rjust(3)
        + " 枚 / りんご "
        + str(apple_count_by_split["excluded"]).rjust(5)
        + " 個"
    )
    print("")

    # ---------------- 手順5: YOLO用の設定ファイル(dataset.yaml)を書き出す ----
    # ultralytics に「画像とラベルがどこにあるか」を伝えるための設定ファイルを作る
    dataset_yaml_lines = [
        "# このファイルは prepare_dataset.py が自動生成しました",
        "# YOLOv8 の学習時に --data でこのファイルを指定します",
        "",
        "# データセットの一番上のフォルダ（絶対パス）",
        "path: " + os.path.abspath(output_directory),
        "# 学習用画像のフォルダ（path からの相対パス）",
        "train: images/train",
        "# 検証用画像のフォルダ",
        "val: images/val",
        "# テスト用画像のフォルダ",
        "test: images/test",
        "",
        "# 検出するクラスの一覧。今回は「りんご」1種類だけ",
        "names:",
        "  0: apple",
    ]
    # 設定ファイルのパスを作る
    dataset_yaml_path = os.path.join(output_directory, "dataset.yaml")
    # 設定ファイルを書き出す
    write_text_lines(dataset_yaml_path, dataset_yaml_lines)
    # 書き出したことを表示する
    print("[手順5] YOLO用の設定ファイルを作りました: " + dataset_yaml_path)
    print("")

    # ---------------- 手順6: 分割結果を記録して再現できるようにする ----------
    # 分割結果を保存するフォルダのパスを作る
    splits_directory = os.path.join(common_paths.get_outputs_directory(), "splits")
    # そのフォルダを作る
    os.makedirs(splits_directory, exist_ok=True)

    # 4つのセットについて、シーン名と画像名の一覧をテキストファイルに書き出す
    for split_name in ["train", "val", "test", "excluded"]:
        # シーン名の一覧を書き出す
        write_text_lines(
            os.path.join(splits_directory, "scenes_" + split_name + ".txt"),
            scene_names_by_split[split_name],
        )
        # 画像名の一覧を書き出す
        write_text_lines(
            os.path.join(splits_directory, "crops_" + split_name + ".txt"),
            crop_names_by_split[split_name],
        )

    # 分割の条件と結果をまとめた辞書を作る（後から再現するための記録）
    split_record = {
        # 使った乱数シード
        "random_seed": parsed_arguments.seed,
        # 指定した比率
        "train_ratio": parsed_arguments.train_ratio,
        "val_ratio": parsed_arguments.val_ratio,
        "test_ratio": parsed_arguments.test_ratio,
        # 連続何シーンを1ブロックにしたか
        "block_size": parsed_arguments.block_size,
        # 境目で何シーン捨てたか
        "guard_scenes": parsed_arguments.guard_scenes,
        # 使った画像の種類
        "image_kind": parsed_arguments.image_kind,
        # 分割の単位の数
        "number_of_split_units": len(split_unit_list),
        # セットごとの枚数
        "crop_counts": {
            "train": len(crop_names_by_split["train"]),
            "val": len(crop_names_by_split["val"]),
            "test": len(crop_names_by_split["test"]),
            "excluded": len(crop_names_by_split["excluded"]),
        },
        # セットごとのシーン数
        "scene_counts": {
            "train": len(scene_names_by_split["train"]),
            "val": len(scene_names_by_split["val"]),
            "test": len(scene_names_by_split["test"]),
            "excluded": len(scene_names_by_split["excluded"]),
        },
        # セットごとのりんごの数
        "apple_counts": apple_count_by_split,
        # シーンごとの割り当て（どの画像がどのセットに入ったかの完全な記録）
        "scene_to_split": scene_to_split_name,
    }
    # 記録をJSONファイルのパスとして作る
    split_json_path = os.path.join(splits_directory, "split_record.json")
    # JSONファイルとして書き出す（日本語をそのまま読めるように ensure_ascii=False にする）
    with open(split_json_path, "w", encoding="utf-8") as opened_file:
        json.dump(split_record, opened_file, ensure_ascii=False, indent=2)
    # 書き出したことを表示する
    print("[手順6] 分割の記録を保存しました: " + split_json_path)
    print("")

    # ---------------- 手順7: データリークが無いことを確認する ---------------
    print("[手順7] データリークが無いか確認します")
    # 確認で問題が見つかったかどうかを記録する変数を用意する
    leak_was_found = False

    # --- 確認1: 同じシーンが2つのセットに入っていないか ---
    # train / val / test の組み合わせをすべて調べる
    for first_split_name in ["train", "val", "test"]:
        for second_split_name in ["train", "val", "test"]:
            # 同じセット同士、および同じ組み合わせの重複は調べなくてよいので飛ばす
            if first_split_name >= second_split_name:
                continue
            # それぞれのセットのシーン名を集合にする
            first_scene_set = set(scene_names_by_split[first_split_name])
            second_scene_set = set(scene_names_by_split[second_split_name])
            # 両方に入っているシーンを求める
            shared_scene_set = first_scene_set & second_scene_set
            # 両方に入っているシーンがあれば問題なので表示する
            if len(shared_scene_set) > 0:
                print("  [NG] " + first_split_name + " と " + second_split_name
                      + " に同じシーンが " + str(len(shared_scene_set)) + " 枚あります")
                leak_was_found = True
    # 問題が無ければその旨を表示する
    if not leak_was_found:
        print("  [OK] 同じ生画像が2つのセットに入っていることはありません")

    # --- 確認2: 違うセットのシーンが、撮影順で隣り合っていないか ---
    # 撮影グループごとにシーンを分けるための辞書を用意する
    scenes_by_acquisition_group = {}
    # シーンを1つずつ見ていく
    for one_scene_name in scene_name_list:
        # シーン名を撮影グループ名とフレーム番号に分解する
        acquisition_group_name, frame_number = common_paths.parse_scene_name(one_scene_name)
        # その撮影グループがまだ辞書に無ければ、空のリストを用意する
        if acquisition_group_name not in scenes_by_acquisition_group:
            scenes_by_acquisition_group[acquisition_group_name] = []
        # フレーム番号とシーン名の組を追加する
        scenes_by_acquisition_group[acquisition_group_name].append(
            (frame_number, one_scene_name)
        )

    # ガードバンドで隔てられずに直接隣り合ってしまった箇所を数える変数
    directly_adjacent_count = 0
    # 違うセット同士のフレーム番号の差の最小値を記録する変数
    smallest_frame_gap_between_splits = None

    # 撮影グループを1つずつ順番に調べる
    for acquisition_group_name in sorted(scenes_by_acquisition_group):
        # そのグループのシーンをフレーム番号順に並べ替える
        scenes_in_group = sorted(scenes_by_acquisition_group[acquisition_group_name])
        # 「除外」になっていないシーンだけを、元の並び順の位置とともに集める
        remaining_scenes_in_group = []
        # 並べ替えたシーンを、先頭から位置番号を付けて1つずつ見ていく
        for position_index in range(len(scenes_in_group)):
            # フレーム番号とシーン名を取り出す
            frame_number = scenes_in_group[position_index][0]
            one_scene_name = scenes_in_group[position_index][1]
            # そのシーンが除外されていなければ、位置番号と一緒にリストに追加する
            if scene_to_split_name[one_scene_name] != "excluded":
                remaining_scenes_in_group.append((position_index, frame_number, one_scene_name))

        # 残ったシーンの隣り合う組を順番に調べる
        for list_index in range(len(remaining_scenes_in_group) - 1):
            # 手前のシーンの情報を取り出す
            earlier_position, earlier_frame_number, earlier_scene_name = remaining_scenes_in_group[list_index]
            # 次のシーンの情報を取り出す
            later_position, later_frame_number, later_scene_name = remaining_scenes_in_group[list_index + 1]
            # 2つのシーンが同じセットなら、調べる必要が無いので次に進む
            if scene_to_split_name[earlier_scene_name] == scene_to_split_name[later_scene_name]:
                continue
            # フレーム番号の差を計算する
            frame_gap = later_frame_number - earlier_frame_number
            # これまでで一番小さい差なら記録する
            if smallest_frame_gap_between_splits is None or frame_gap < smallest_frame_gap_between_splits:
                smallest_frame_gap_between_splits = frame_gap
            # 元の並び順で位置が1つしか離れていない場合は、
            # 間にガードバンドのシーンが入っていないということなので問題とみなす
            if later_position - earlier_position == 1:
                directly_adjacent_count = directly_adjacent_count + 1

    # 直接隣り合ってしまった箇所があれば問題として表示する
    if directly_adjacent_count > 0:
        print("  [NG] 違うセットのシーンが、間に何も挟まずに隣り合っている箇所: "
              + str(directly_adjacent_count) + " 箇所")
        leak_was_found = True
    else:
        print("  [OK] 違うセットのシーンの間には必ずガードバンドのシーンが入っています")
    # 違うセット同士の最小のフレーム番号差を情報として表示する
    if smallest_frame_gap_between_splits is not None:
        print("  [情報] 違うセットに属するシーン同士の、フレーム番号の差の最小値: "
              + str(smallest_frame_gap_between_splits)
              + "（--guard-scenes を増やすとこの差を広げられます）")

    # ガードバンドが0の場合は、注意を表示する
    if parsed_arguments.guard_scenes == 0:
        print("  [注意] --guard-scenes 0 が指定されています。")
        print("         セットの境目にあるシーンは構図が似ている可能性があります。")
    print("")

    # ---------------- 手順8: 点群を .npy 形式に変換する ---------------------
    # 変換する対象を決める
    if parsed_arguments.convert_point_clouds == "none":
        scenes_to_convert = []
    elif parsed_arguments.convert_point_clouds == "all":
        scenes_to_convert = scene_name_list
    else:
        scenes_to_convert = scene_names_by_split["test"]

    # 変換対象があるときだけ処理する
    if len(scenes_to_convert) > 0:
        print("[手順8] 点群を .mat から .npy に変換します（" + str(len(scenes_to_convert)) + " 件）")
        # 変換したファイル数を数えるための変数を0で用意する
        converted_file_count = 0
        # シーンを1つずつ順番に変換する
        for one_scene_name in scenes_to_convert:
            # 元の .mat ファイルのパスを求める
            mat_file_path = common_paths.get_raw_point_cloud_mat_path(dataset_root, one_scene_name)
            # 変換先の .npy ファイルのパスを求める
            npy_file_path = common_paths.get_point_cloud_npy_path(dataset_root, one_scene_name)
            # 元ファイルが無ければ、警告を出して次に進む
            if not os.path.exists(mat_file_path):
                print("  [警告] 点群ファイルが見つかりません: " + mat_file_path)
                continue
            # 変換を実行する
            kinect_io.convert_point_cloud_mat_to_npy(mat_file_path, npy_file_path)
            # 変換したファイル数を1増やす
            converted_file_count = converted_file_count + 1
            # 10件ごとに進捗を表示する
            if converted_file_count % 10 == 0:
                print("  " + str(converted_file_count) + " / " + str(len(scenes_to_convert)) + " 件 完了")
        # 変換が終わったことを表示する
        print("  変換完了: " + str(converted_file_count) + " 件")
        print("  保存先  : " + os.path.dirname(
            common_paths.get_point_cloud_npy_path(dataset_root, scene_name_list[0])))
    else:
        print("[手順8] 点群の変換は行いません（--convert-point-clouds none が指定されました）")
    print("")

    # 全ての処理が終わったことを表示する
    print("=" * 70)
    print("データ準備が完了しました")
    print("=" * 70)
    print("次は学習を実行してください:")
    print("  python src/train_yolo.py --model-size n --epochs 100")


# このファイルが直接実行されたときだけ main() を呼ぶ
# （他のファイルから読み込まれたときは実行しない、という決まり文句）
if __name__ == "__main__":
    main()
