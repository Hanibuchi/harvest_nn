# ============================================================================
# このファイルは何をするものか
# ----------------------------------------------------------------------------
# 【(0) データ取得スクリプト】です。
# KFuji RGB-DS データセットを Zenodo から自動でダウンロードして、
# このプロジェクトが期待する形（KFuji_RGB-DS_dataset/row data/ など）に
# 展開します。新しいマシンで作業を始めるとき、
# 手でデータセットを持ち運ぶ必要をなくすためのものです。
#
# 【必要な分だけ取ってくる仕組み】
#   配布されている zip は 2.9GB ありますが、そのうち 2.3GB は
#   preprocessed data/images/*_DS.mat（位置合わせ済みの深度データ）で、
#   このプロジェクトでは1バイトも使いません（README の 5-3 を参照）。
#   そこで、このスクリプトは zip 全体を落とさずに、
#   「HTTP レンジ要求」という仕組みで必要なファイルの部分だけを取り出します。
#
#     手順1  zip の末尾にある「目次（中央ディレクトリ）」だけを読む
#     手順2  目次から、欲しいファイルが zip の何バイト目にあるかを知る
#     手順3  その範囲だけをまとめて要求する（1回の要求に最大50範囲まで入れられる）
#     手順4  受け取ったデータを展開し、CRC32（誤り検出符号）で壊れていないか確かめる
#
#   既定では約585MB（zip全体の19%）の取得で済みます。
#
# 【実行例】
#   python src/fetch_dataset.py
#
#   # 何をどれだけ取得するのか、通信せずに確認する
#   python src/fetch_dataset.py --dry-run
#
#   # 未使用の _DS.mat も含めて、配布物と全く同じ内容をそろえる（2.9GB）
#   python src/fetch_dataset.py --parts all
#
#   # 計測だけを行うマシン（Jetson など）で、生データと注釈だけを取得する
#   python src/fetch_dataset.py --parts raw,annotations
#
#   # 動作確認として、少しだけ取得してみる
#   python src/fetch_dataset.py --parts images --limit-files 20
#
#   # 置き場所を変える（環境変数 KFUJI_DATASET_ROOT でも指定できます）
#   python src/fetch_dataset.py --dataset-root /mnt/ssd/KFuji_RGB-DS_dataset
#
# 【注意】
#   このデータセットは CC-BY-NC-SA 4.0（研究・教育目的のみ、商用利用不可）です。
#   利用する場合は README の「9. データセットの利用について」の論文を引用してください。
# ============================================================================

# コマンドライン引数を扱うための標準ライブラリ
import argparse
# md5（書庫全体の照合に使う）を計算するための標準ライブラリ
import hashlib
# 通信が途中で切れたときの例外（IncompleteRead など）を判別するための標準ライブラリ
import http.client
# Zenodo が返す JSON を読むための標準ライブラリ
import json
# ファイルパスを扱うための標準ライブラリ
import os
# zip の中の数値（何バイト目か、など）を読み取るための標準ライブラリ
import struct
# 待ち時間や速度の計算に使う標準ライブラリ
import time
# 通信エラーの種類を判別するための標準ライブラリ
import urllib.error
# インターネットからデータを取得するための標準ライブラリ
import urllib.request
# 書庫全体を落とす方式のときに使う、zip を読むための標準ライブラリ
import zipfile
# zip の中身（deflate 形式）を展開し、CRC32 を計算するための標準ライブラリ
import zlib

# このプロジェクトの、ファイルの置き場所をまとめたモジュール
import common_paths


# --------------------------------------------------------------------------
# 設定値（ふつうは変更しなくてよい）
# --------------------------------------------------------------------------

# Zenodo に登録されている、このデータセットの記録番号
# （https://zenodo.org/records/3715991 の末尾の数字）
DEFAULT_RECORD_ID = "3715991"

# Zenodo の記録情報を問い合わせるための入口（API）のアドレス
ZENODO_API_BASE_URL = "https://zenodo.org/api/records/"

# 記録の中にある、データセット本体の zip のファイル名
ARCHIVE_FILE_NAME = "KFuji_RGB-DS_dataset.zip"

# 複数の範囲をまとめて要求したときに、サーバが返事の区切りに使う印
# （ふつうは multipart/byteranges という形式だが、Zenodo は
#   この固定の文字列を区切りに使い、改行は \n だけである）
MULTIPART_BOUNDARY_MARKER = b"--EOSMULTIPARTBOUNDARY"

# 1回の要求にまとめる範囲の最大数（サーバへの負担を抑えるための上限）
MAX_RANGES_PER_REQUEST = 50

# 1回の要求で受け取る最大バイト数
# （メモリ使用量を抑えるためと、途中で止めたときに失う量を小さくするための上限。
#   1回ぶんを受け取り終えるたびにファイルが書き出され、進み具合も表示されます）
MAX_BYTES_PER_REQUEST = 8 * 1024 * 1024

# 隣り合うファイルの隙間がこのバイト数以下なら、1つの範囲にまとめる
# （要求の回数を減らすため。少しだけ余計に受け取るが、そのほうが速い）
RANGE_GAP_TOLERANCE_BYTES = 4096

# zip の目次の位置を探すために、末尾から読むバイト数
ARCHIVE_TAIL_READ_SIZE = 65536

# 通信に失敗したときに、何回までやり直すか
MAX_RETRY_COUNT = 5

# このプログラムが名乗る名前（サーバ側のログに残る）
USER_AGENT = "harvest_nn-fetch-dataset/1.0"

# --parts に指定できる名前の一覧
AVAILABLE_PART_NAMES = ["raw", "annotations", "images", "ds-mat", "all"]

# --parts を省略したときに取得する範囲
DEFAULT_PART_NAMES = ["raw", "annotations", "images"]

# zip の中のどのフォルダにも属さないが、必ず一緒に置きたい小さなファイル
ALWAYS_INCLUDED_FILE_NAMES = ["README.txt", "LICENSE.txt"]


# --------------------------------------------------------------------------
# 表示を整えるための小さな道具
# --------------------------------------------------------------------------

def format_size(byte_count):
    """バイト数を人間に読みやすい文字列にする（例: 583.0MB）。"""
    # 1024の3乗（ギガ）以上なら GB で表示する
    if byte_count >= 1024 * 1024 * 1024:
        return "%.2fGB" % (float(byte_count) / (1024.0 * 1024.0 * 1024.0))
    # 1024の2乗（メガ）以上なら MB で表示する
    if byte_count >= 1024 * 1024:
        return "%.1fMB" % (float(byte_count) / (1024.0 * 1024.0))
    # 1024以上なら KB で表示する
    if byte_count >= 1024:
        return "%.1fKB" % (float(byte_count) / 1024.0)
    # それより小さければ、そのままバイト数で表示する
    return str(byte_count) + "B"


def format_duration(second_count):
    """秒数を人間に読みやすい文字列にする（例: 1分20秒）。"""
    # マイナスや非常に小さい値は 0 秒として扱う
    if second_count < 1:
        return "1秒未満"
    # 分と秒に分ける
    minute_count = int(second_count) // 60
    remaining_second_count = int(second_count) % 60
    # 1分未満なら秒だけを表示する
    if minute_count == 0:
        return str(remaining_second_count) + "秒"
    # 分と秒をつないで返す
    return str(minute_count) + "分" + str(remaining_second_count) + "秒"


def print_license_notice():
    """データセットの利用条件を画面に表示する。"""
    print("  このデータセットは CC-BY-NC-SA 4.0（研究・教育目的のみ、商用利用不可）です。")
    print("  論文などで利用する場合は、次の2本を引用してください。")
    print("    - Gené-Mola et al. (2019) Computers and Electronics in Agriculture, 162, 689-698.")
    print("    - Gené-Mola et al. (2019) Data in Brief, 25, 104289.")


# --------------------------------------------------------------------------
# 通信の部品
# --------------------------------------------------------------------------

class RangeNotSupportedError(Exception):
    """サーバが「一部だけ取得」に対応していないときに投げる例外。"""


# やり直せば直るかもしれない通信の失敗の一覧
# （OSError は接続の切断など、http.client.HTTPException は
#   「本文が途中で終わっている(IncompleteRead)」などを含む）
RETRYABLE_NETWORK_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    TimeoutError,
    OSError,
)


def open_url_with_retry(url, extra_header_dict=None):
    """
    指定したアドレスに接続し、応答オブジェクトを返す。

    混雑エラー(429)や一時的な失敗のときは、少し待ってから自動でやり直します。
    返ってきた応答は、呼び出した側が必ず close() してください。
    """
    # ヘッダ（通信につける付加情報）を組み立てる
    header_dict = {"User-Agent": USER_AGENT}
    # 範囲指定などの追加ヘッダがあれば足す
    if extra_header_dict is not None:
        header_dict.update(extra_header_dict)

    # 最後に起きたエラーを覚えておくための変数
    last_error = None
    # 決めた回数だけ、やり直しながら接続を試す
    for attempt_index in range(MAX_RETRY_COUNT):
        try:
            # 要求を組み立てて送る
            request = urllib.request.Request(url, headers=header_dict)
            # 応答を受け取って、そのまま呼び出し元に返す
            return urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as error:
            # 見つからない・権限が無いなど、やり直しても直らないエラーはすぐ諦める
            if error.code not in (429, 500, 502, 503, 504):
                raise
            # 覚えておく
            last_error = error
            # サーバが「何秒待って」と言っていればその秒数、無ければ倍々で待つ
            retry_after_text = error.headers.get("Retry-After") if error.headers else None
            if retry_after_text is not None and retry_after_text.isdigit():
                wait_second_count = int(retry_after_text)
            else:
                wait_second_count = 2 ** attempt_index
            print("  [警告] サーバが混雑しています（" + str(error.code) + "）。"
                  + str(wait_second_count) + "秒待ってやり直します")
            time.sleep(wait_second_count)
        except RETRYABLE_NETWORK_ERRORS as error:
            # 回線が切れた場合なども、少し待ってからやり直す
            last_error = error
            wait_second_count = 2 ** attempt_index
            print("  [警告] 通信に失敗しました。" + str(wait_second_count) + "秒待ってやり直します")
            time.sleep(wait_second_count)

    # 決めた回数やっても駄目だったので、エラーにして止める
    raise RuntimeError(
        "インターネットからのデータ取得に繰り返し失敗しました: " + url + "\n"
        + "原因: " + str(last_error) + "\n"
        + "回線の状態を確かめて、もう一度実行してください（途中まで取得したファイルは残ります）。"
    )


def read_url_with_retry(url, extra_header_dict=None):
    """
    指定したアドレスに接続し、本文を最後まで受け取って返す。

    open_url_with_retry と違い、**本文の受信中に切れた場合もやり直します**。
    大きなデータを長時間かけて受け取るときは、こちらを使ってください。
    返り値は (応答コード, ヘッダ, 本文のバイト列) です。
    """
    # 最後に起きたエラーを覚えておくための変数
    last_error = None
    # 決めた回数だけ、やり直しながら試す
    for attempt_index in range(MAX_RETRY_COUNT):
        response = None
        try:
            # 接続する（接続そのものの失敗は open_url_with_retry の中でやり直される）
            response = open_url_with_retry(url, extra_header_dict)
            # 本文を最後まで受け取る
            status_code = response.status
            response_headers = response.headers
            body_bytes = response.read()
            # 受け取れたので、利用回数制限に近づいていたら少し待ってから返す
            respect_rate_limit(response_headers)
            return status_code, response_headers, body_bytes
        except RETRYABLE_NETWORK_ERRORS as error:
            # 受信の途中で切れた場合は、少し待ってから最初から受け取り直す
            last_error = error
            wait_second_count = 2 ** attempt_index
            print("  [警告] 受信が途中で切れました。"
                  + str(wait_second_count) + "秒待ってやり直します")
            time.sleep(wait_second_count)
        finally:
            # 接続を必ず閉じる
            if response is not None:
                response.close()

    # 決めた回数やっても駄目だったので、エラーにして止める
    raise RuntimeError(
        "インターネットからのデータ取得に繰り返し失敗しました: " + url + "\n"
        + "原因: " + str(last_error) + "\n"
        + "回線の状態を確かめて、もう一度実行してください（取得済みのファイルはそのまま残ります）。"
    )


def respect_rate_limit(response_headers):
    """
    Zenodo の利用回数制限に引っかからないよう、必要なら少し待つ。

    応答には「あと何回要求できるか」と「いつ制限が戻るか」が入っています。
    残りが少ないときだけ、戻る時刻まで待ちます。
    """
    # 残り回数が書かれていなければ、何もしない
    remaining_text = response_headers.get("x-ratelimit-remaining")
    reset_text = response_headers.get("x-ratelimit-reset")
    if remaining_text is None or reset_text is None:
        return
    # 数字として読めなければ、何もしない
    if not remaining_text.isdigit() or not reset_text.isdigit():
        return
    # 残りが5回より多ければ、まだ余裕があるので何もしない
    if int(remaining_text) > 5:
        return
    # 制限が戻る時刻まで、あと何秒あるかを計算する
    wait_second_count = int(reset_text) - int(time.time()) + 1
    # すでに過ぎていれば待たない
    if wait_second_count <= 0:
        return
    # 長すぎる待ちは異常値とみなして60秒で切り上げる
    if wait_second_count > 60:
        wait_second_count = 60
    print("  [情報] Zenodo の利用回数制限に近づいたので "
          + str(wait_second_count) + "秒待ちます")
    time.sleep(wait_second_count)


def fetch_bytes_range(archive_url, range_list):
    """
    zip の指定した範囲（複数可）だけを取得して、範囲ごとのデータを返す。

    range_list は [(開始バイト, 終了バイト), ...] の形で、終了バイトも含みます。
    返り値は {開始バイト: そのデータ} という辞書です。
    """
    # 範囲指定の文字列を組み立てる（例: "bytes=0-99,200-299"）
    range_text_list = []
    for start_offset, end_offset in range_list:
        range_text_list.append(str(start_offset) + "-" + str(end_offset))
    range_header_value = "bytes=" + ",".join(range_text_list)

    # 範囲を指定して取得する（受信の途中で切れた場合もやり直される）
    status_code, response_headers, body_bytes = read_url_with_retry(
        archive_url, {"Range": range_header_value}
    )
    # 206（一部だけ返した）でなければ、このサーバは範囲取得に対応していない
    if status_code != 206:
        raise RangeNotSupportedError(
            "サーバが範囲指定に対応していません（応答: " + str(status_code) + "）"
        )

    # 範囲が1つだけなら、本文がそのままその範囲のデータになる
    if len(range_list) == 1:
        return {range_list[0][0]: body_bytes}

    # 範囲が複数なら、区切り入りの本文を分解する
    return parse_multipart_byteranges(body_bytes)


def parse_multipart_byteranges(body_bytes):
    """
    複数範囲の応答（区切り文字入り）を分解して、{開始バイト: データ} を返す。

    応答は次のような形をしています。
        --EOSMULTIPARTBOUNDARY
        Content-Type: application/octet-stream
        Content-Range: bytes 100-199/3113452347
        （空行）
        ここに実際のデータ
        --EOSMULTIPARTBOUNDARY
        ...
    """
    # 結果を入れる辞書を用意する
    data_by_start_offset = {}
    # 区切り文字で本文を分ける（最初の一片は空になることが多い）
    for one_piece in body_bytes.split(MULTIPART_BOUNDARY_MARKER):
        # この一片の中から「Content-Range:」の行を探す
        header_position = one_piece.find(b"Content-Range: bytes ")
        # 見つからなければ、データの無い一片なので飛ばす
        if header_position < 0:
            continue
        # 「Content-Range: bytes 100-199/3113452347」の数字の部分を取り出す
        value_start = header_position + len(b"Content-Range: bytes ")
        value_end = one_piece.find(b"\n", value_start)
        if value_end < 0:
            continue
        range_text = one_piece[value_start:value_end].decode("ascii", "replace")
        # 「100-199/全体サイズ」を「100」だけにする
        start_offset_text = range_text.split("-")[0].strip()
        if not start_offset_text.isdigit():
            continue
        # ヘッダ部分の終わり（空行）を探す。そこから後ろが実際のデータ
        blank_line_position = one_piece.find(b"\n\n", value_end)
        if blank_line_position < 0:
            continue
        data_start = blank_line_position + 2
        one_data = one_piece[data_start:]
        # 次の区切りの直前に付いている改行を取り除く
        if one_data.endswith(b"\n"):
            one_data = one_data[:-1]
        # 辞書に記録する
        data_by_start_offset[int(start_offset_text)] = one_data

    # 1つも取り出せなかった場合は、形式が想定と違うということなのでエラーにする
    if len(data_by_start_offset) == 0:
        raise RangeNotSupportedError("複数範囲の応答を読み取れませんでした")

    # できあがった辞書を返す
    return data_by_start_offset


# --------------------------------------------------------------------------
# Zenodo の記録情報を調べる
# --------------------------------------------------------------------------

def fetch_record_metadata(record_id):
    """
    Zenodo から記録情報を取得して、zip のアドレス・サイズ・md5 を返す。

    返り値は {"url": ..., "size": ..., "md5": ...} という辞書です。
    """
    # 問い合わせ先のアドレスを組み立てる
    api_url = ZENODO_API_BASE_URL + record_id
    # 接続して本文（JSON）を受け取る
    unused_status, unused_headers, body_bytes = read_url_with_retry(api_url)
    record_json_text = body_bytes.decode("utf-8")
    # 文字列を辞書に変換する
    record_dictionary = json.loads(record_json_text)

    # 記録に含まれるファイルの一覧から、データセット本体の zip を探す
    for one_file in record_dictionary.get("files", []):
        if one_file.get("key") != ARCHIVE_FILE_NAME:
            continue
        # チェックサムは "md5:4e4a..." の形なので、"md5:" を取り除く
        checksum_text = one_file.get("checksum", "")
        if checksum_text.startswith("md5:"):
            checksum_text = checksum_text[len("md5:"):]
        # 必要な情報だけを取り出して返す
        return {
            "url": one_file["links"]["self"],
            "size": int(one_file["size"]),
            "md5": checksum_text,
            "title": record_dictionary.get("title", ""),
        }

    # 見つからなかった場合は、記録番号が違う可能性が高い
    raise RuntimeError(
        "Zenodo の記録 " + record_id + " の中に " + ARCHIVE_FILE_NAME + " が見つかりません。\n"
        + "--record-id の指定を確かめてください（既定値は " + DEFAULT_RECORD_ID + " です）。"
    )


# --------------------------------------------------------------------------
# zip の目次（中央ディレクトリ）を読む
# --------------------------------------------------------------------------

def read_central_directory(archive_url, archive_size):
    """
    zip の末尾にある目次だけを取得して、入っているファイルの一覧を返す。

    返り値は辞書のリストで、各辞書は次の項目を持ちます。
        name              ... zip の中でのファイル名
        method            ... 圧縮方式（8=deflate, 0=無圧縮）
        crc               ... 展開後のデータの CRC32（壊れていないかの確認用）
        compressed_size   ... 圧縮されたままの大きさ
        uncompressed_size ... 展開後の大きさ
        local_offset      ... zip の何バイト目にこのファイルが置かれているか
    """
    # 末尾から少しだけ読む（目次の位置を書いた印がこの中にある）
    tail_read_size = ARCHIVE_TAIL_READ_SIZE
    if tail_read_size > archive_size:
        tail_read_size = archive_size
    tail_start_offset = archive_size - tail_read_size
    tail_bytes = fetch_bytes_range(
        archive_url, [(tail_start_offset, archive_size - 1)]
    )[tail_start_offset]

    # 目次の位置を書いた印（EOCD）を、末尾側から探す
    eocd_position = tail_bytes.rfind(b"PK\x05\x06")
    if eocd_position < 0:
        raise RuntimeError(
            "zip の目次が見つかりませんでした。書庫の形式が想定と違う可能性があります。\n"
            + "--full-archive を付けて実行すると、書庫全体を取得する方式に切り替わります。"
        )

    # 印の中から、目次の大きさと位置を読み取る
    central_directory_size, = struct.unpack_from("<I", tail_bytes, eocd_position + 12)
    central_directory_offset, = struct.unpack_from("<I", tail_bytes, eocd_position + 16)

    # 0xFFFFFFFF は「4GBを超える zip（zip64）」の印。この方式では扱えない
    if central_directory_offset == 0xFFFFFFFF or central_directory_size == 0xFFFFFFFF:
        raise RuntimeError(
            "この書庫は zip64 形式のため、必要な部分だけの取得に対応できません。\n"
            + "--full-archive を付けて実行してください。"
        )

    # 目次の部分だけを取得する
    central_directory_bytes = fetch_bytes_range(
        archive_url,
        [(central_directory_offset, central_directory_offset + central_directory_size - 1)],
    )[central_directory_offset]

    # 目次を先頭から順に読み、1ファイルぶんずつ取り出す
    entry_list = []
    read_position = 0
    while read_position + 46 <= len(central_directory_bytes):
        # 1件ぶんの始まりの印を確かめる
        if central_directory_bytes[read_position:read_position + 4] != b"PK\x01\x02":
            break
        # 必要な数値を決まった位置から読み取る
        general_flag, compression_method = struct.unpack_from(
            "<HH", central_directory_bytes, read_position + 8
        )
        crc_value, compressed_size, uncompressed_size = struct.unpack_from(
            "<III", central_directory_bytes, read_position + 16
        )
        name_length, extra_length, comment_length = struct.unpack_from(
            "<HHH", central_directory_bytes, read_position + 28
        )
        local_offset, = struct.unpack_from("<I", central_directory_bytes, read_position + 42)
        # ファイル名を取り出す（文字コードは印(0x800)があれば UTF-8、無ければ CP437）
        name_bytes = central_directory_bytes[
            read_position + 46:read_position + 46 + name_length
        ]
        if general_flag & 0x800:
            entry_name = name_bytes.decode("utf-8", "replace")
        else:
            entry_name = name_bytes.decode("cp437", "replace")
        # 暗号化されている書庫は扱えない
        if general_flag & 0x1:
            raise RuntimeError("この書庫は暗号化されているため展開できません: " + entry_name)
        # 1件ぶんの情報をリストに加える
        entry_list.append({
            "name": entry_name,
            "method": compression_method,
            "crc": crc_value,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "local_offset": local_offset,
        })
        # 次の1件へ進む
        read_position = read_position + 46 + name_length + extra_length + comment_length

    # 1件も読めなければ、形式が想定と違う
    if len(entry_list) == 0:
        raise RuntimeError("zip の目次を読み取れませんでした。")

    # 各ファイルが「どこまで続くか」を、次のファイルの開始位置から求めておく
    # （最後のファイルの終わりは、目次の開始位置になる）
    sorted_entry_list = sorted(entry_list, key=lambda one_entry: one_entry["local_offset"])
    for index in range(len(sorted_entry_list)):
        if index + 1 < len(sorted_entry_list):
            sorted_entry_list[index]["end_offset"] = sorted_entry_list[index + 1]["local_offset"]
        else:
            sorted_entry_list[index]["end_offset"] = central_directory_offset

    # ファイル名順に並べ直して返す
    return sorted(entry_list, key=lambda one_entry: one_entry["name"])


# --------------------------------------------------------------------------
# 取得するファイルを選ぶ
# --------------------------------------------------------------------------

def entry_belongs_to_part(entry_name, part_name):
    """ファイル名が、指定された部分（--parts の値）に含まれるかどうかを返す。"""
    # "all" はすべてのファイルを含む
    if part_name == "all":
        return True
    # "raw" は生データ（位置合わせ前）のフォルダ
    if part_name == "raw":
        return entry_name.startswith("row data/")
    # "annotations" は注釈まわりの小さなフォルダ3つ
    if part_name == "annotations":
        return (
            entry_name.startswith("preprocessed data/annotations/")
            or entry_name.startswith("preprocessed data/square_annotations1/")
            or entry_name.startswith("preprocessed data/sets/")
        )
    # "images" は切り出し画像のうち jpg だけ（学習に使うのはこちら）
    if part_name == "images":
        return (
            entry_name.startswith("preprocessed data/images/")
            and entry_name.endswith(".jpg")
        )
    # "ds-mat" は本プロジェクトでは使わない位置合わせ済みの深度データ
    if part_name == "ds-mat":
        return (
            entry_name.startswith("preprocessed data/images/")
            and entry_name.endswith("_DS.mat")
        )
    # 知らない名前なら含めない
    return False


def describe_entry_part(entry_name):
    """ファイル名を、表示用の分類名（row data など）に対応づける。"""
    if entry_name.startswith("row data/"):
        return "row data"
    if entry_name.startswith("preprocessed data/images/"):
        if entry_name.endswith("_DS.mat"):
            return "images(_DS.mat)"
        return "images(jpg)"
    if entry_name.startswith("preprocessed data/"):
        return "annotations"
    return "その他"


def select_entries(entry_list, part_name_list):
    """--parts の指定に合うファイルだけを選んで返す。"""
    # 選んだものを入れるリストを用意する
    selected_entry_list = []
    # 1件ずつ見ていく
    for one_entry in entry_list:
        entry_name = one_entry["name"]
        # 名前が "/" で終わるものはフォルダなので、ファイルとしては扱わない
        if entry_name.endswith("/"):
            continue
        # README と LICENSE は小さいので、常に一緒に置く
        if entry_name in ALWAYS_INCLUDED_FILE_NAMES:
            selected_entry_list.append(one_entry)
            continue
        # 指定された部分のどれかに当てはまれば選ぶ
        for one_part_name in part_name_list:
            if entry_belongs_to_part(entry_name, one_part_name):
                selected_entry_list.append(one_entry)
                break
    # 選んだリストを返す
    return selected_entry_list


def remove_already_downloaded_entries(selected_entry_list, dataset_root):
    """
    すでに正しい大きさで置いてあるファイルを、取得対象から外す。

    返り値は (まだ取得が必要なリスト, すでにあった件数) です。
    """
    # まだ必要なものを入れるリストを用意する
    remaining_entry_list = []
    # すでにあった件数を数えるための変数を0で用意する
    already_present_count = 0
    # 1件ずつ確かめる
    for one_entry in selected_entry_list:
        # 置き場所のパスを作る
        destination_path = build_destination_path(dataset_root, one_entry["name"])
        # ファイルがあり、大きさも一致していれば取得しなくてよい
        if os.path.exists(destination_path):
            if os.path.getsize(destination_path) == one_entry["uncompressed_size"]:
                already_present_count = already_present_count + 1
                continue
        # そうでなければ取得が必要
        remaining_entry_list.append(one_entry)
    # 結果を返す
    return remaining_entry_list, already_present_count


def build_destination_path(dataset_root, entry_name):
    """zip の中のファイル名から、ディスク上の置き場所のパスを作る。"""
    # zip の中は必ず "/" 区切りなので、その OS の区切りに直してつなぐ
    return os.path.join(dataset_root, *entry_name.split("/"))


# --------------------------------------------------------------------------
# 取得する範囲を組み立てる
# --------------------------------------------------------------------------

def build_byte_runs(selected_entry_list):
    """
    取得したいファイルを、まとめて要求できる「連続した範囲」に整理する。

    返り値は辞書のリストで、各辞書は次の項目を持ちます。
        start    ... 範囲の開始バイト
        end      ... 範囲の終了バイト（この位置も含む）
        entries  ... この範囲に入っているファイルの一覧
    """
    # zip の中での位置の順に並べる
    sorted_entry_list = sorted(
        selected_entry_list, key=lambda one_entry: one_entry["local_offset"]
    )
    # できあがった範囲を入れるリストを用意する
    run_list = []
    # 1件ずつ、前の範囲につなげられるかを見ていく
    for one_entry in sorted_entry_list:
        entry_start = one_entry["local_offset"]
        entry_end = one_entry["end_offset"] - 1
        # 直前の範囲があり、隙間が小さく、まとめても大きくなりすぎないならつなげる
        if len(run_list) > 0:
            last_run = run_list[-1]
            gap_size = entry_start - (last_run["end"] + 1)
            merged_size = entry_end - last_run["start"] + 1
            if gap_size <= RANGE_GAP_TOLERANCE_BYTES and merged_size <= MAX_BYTES_PER_REQUEST:
                last_run["end"] = entry_end
                last_run["entries"].append(one_entry)
                continue
        # つなげられなければ、新しい範囲として始める
        run_list.append({"start": entry_start, "end": entry_end, "entries": [one_entry]})
    # できあがった範囲のリストを返す
    return run_list


def build_request_batches(run_list):
    """
    範囲のリストを、1回の要求にまとめられる単位に分ける。

    1回の要求には、範囲の数と合計バイト数の両方に上限があります。
    """
    # できあがったまとまりを入れるリストを用意する
    batch_list = []
    # 今まとめている途中のものを入れるリストと、その合計サイズ
    current_batch = []
    current_batch_size = 0
    # 1つずつ、今のまとまりに入れられるかを見ていく
    for one_run in run_list:
        one_run_size = one_run["end"] - one_run["start"] + 1
        # 数か大きさの上限を超えるなら、今のまとまりをここで区切る
        if len(current_batch) > 0:
            if (len(current_batch) >= MAX_RANGES_PER_REQUEST
                    or current_batch_size + one_run_size > MAX_BYTES_PER_REQUEST):
                batch_list.append(current_batch)
                current_batch = []
                current_batch_size = 0
        # 今のまとまりに加える
        current_batch.append(one_run)
        current_batch_size = current_batch_size + one_run_size
    # 最後のまとまりが残っていれば加える
    if len(current_batch) > 0:
        batch_list.append(current_batch)
    # できあがったリストを返す
    return batch_list


# --------------------------------------------------------------------------
# 受け取ったデータを展開して保存する
# --------------------------------------------------------------------------

def extract_one_entry(range_bytes, range_start_offset, one_entry, dataset_root):
    """
    受け取ったデータの中から1ファイル分を取り出し、展開して保存する。

    保存の前に CRC32 と大きさを照合し、壊れていないことを確かめます。
    """
    # このファイルが、受け取ったデータの何バイト目から始まるかを求める
    relative_position = one_entry["local_offset"] - range_start_offset
    # そこに zip のファイル開始の印があるかを確かめる
    if range_bytes[relative_position:relative_position + 4] != b"PK\x03\x04":
        raise RuntimeError("書庫の中のファイル位置がずれています: " + one_entry["name"])
    # 実際のデータが始まる位置は、名前と追加情報の長さのぶんだけ後ろにある
    name_length, extra_length = struct.unpack_from(
        "<HH", range_bytes, relative_position + 26
    )
    data_start = relative_position + 30 + name_length + extra_length
    data_end = data_start + one_entry["compressed_size"]
    # 必要な分が受け取れているかを確かめる
    if data_end > len(range_bytes):
        raise RuntimeError("受け取ったデータが足りません: " + one_entry["name"])
    # 圧縮されたままのデータを取り出す
    compressed_bytes = range_bytes[data_start:data_end]

    # 圧縮方式に応じて展開する（8=deflate、0=無圧縮）
    if one_entry["method"] == 8:
        file_bytes = zlib.decompress(compressed_bytes, -15)
    elif one_entry["method"] == 0:
        file_bytes = compressed_bytes
    else:
        raise RuntimeError(
            "対応していない圧縮方式です（" + str(one_entry["method"]) + "）: " + one_entry["name"]
        )

    # 展開後の大きさが目次の記録と一致するかを確かめる
    if len(file_bytes) != one_entry["uncompressed_size"]:
        raise RuntimeError("展開後の大きさが合いません: " + one_entry["name"])
    # CRC32（誤り検出符号）が一致するかを確かめる
    if (zlib.crc32(file_bytes) & 0xFFFFFFFF) != one_entry["crc"]:
        raise RuntimeError("データが壊れています（CRC不一致）: " + one_entry["name"])

    # 置き場所のパスを作り、必要ならフォルダを作る
    destination_path = build_destination_path(dataset_root, one_entry["name"])
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    # いきなり本番の名前で書かず、.part に書いてから名前を変える
    # （途中で止まったときに、中途半端なファイルが残らないようにするため）
    temporary_path = destination_path + ".part"
    with open(temporary_path, "wb") as output_file:
        output_file.write(file_bytes)
    os.replace(temporary_path, destination_path)


def fetch_ranges_one_by_one(archive_url, range_list):
    """範囲を1つずつ別々に要求して取得する（まとめて要求できない環境のための方式）。"""
    # 結果を入れる辞書を用意する
    data_by_start_offset = {}
    # 1範囲ずつ取得して辞書に入れる
    for one_range in range_list:
        one_result = fetch_bytes_range(archive_url, [one_range])
        data_by_start_offset.update(one_result)
    # できあがった辞書を返す
    return data_by_start_offset


def download_selected_entries(archive_url, selected_entry_list, dataset_root):
    """必要なファイルだけを取得して展開する。展開したファイル数を返す。"""
    # 取得する範囲を組み立てる
    run_list = build_byte_runs(selected_entry_list)
    # 1回の要求にまとめる単位に分ける
    batch_list = build_request_batches(run_list)
    # 取得する合計バイト数を数える
    total_byte_count = 0
    for one_run in run_list:
        total_byte_count = total_byte_count + (one_run["end"] - one_run["start"] + 1)

    # 進み具合を表示するための準備
    print("  範囲 " + str(len(run_list)) + " 個を " + str(len(batch_list))
          + " 回の要求に分けて取得します（合計 " + format_size(total_byte_count) + "）")
    start_time = time.perf_counter()
    downloaded_byte_count = 0
    extracted_file_count = 0

    # 複数範囲をまとめて要求できない環境だと分かったら、1つずつに切り替えるための印
    use_single_range_mode = False

    # まとまりごとに取得して展開する
    for batch_index, one_batch in enumerate(batch_list):
        # このまとまりで要求する範囲の一覧を作る
        range_list = []
        for one_run in one_batch:
            range_list.append((one_run["start"], one_run["end"]))
        # まとめて取得する
        # （まとめて要求できない環境なら、1範囲ずつに切り替えてやり直す。
        #   要求の回数は増えるが、取れる内容は同じ）
        if use_single_range_mode:
            data_by_start_offset = fetch_ranges_one_by_one(archive_url, range_list)
        else:
            try:
                data_by_start_offset = fetch_bytes_range(archive_url, range_list)
            except RangeNotSupportedError as error:
                # 範囲が1つだけで失敗したのなら、範囲取得そのものが使えない
                if len(range_list) == 1:
                    raise
                print("  [警告] " + str(error))
                print("  [情報] 範囲を1つずつ要求する方式に切り替えます")
                use_single_range_mode = True
                data_by_start_offset = fetch_ranges_one_by_one(archive_url, range_list)
        # 範囲ごとに、中に入っているファイルを取り出して保存する
        for one_run in one_batch:
            range_bytes = data_by_start_offset.get(one_run["start"])
            # まとめた応答に含まれていなかった範囲は、その範囲だけ単独で取り直す
            if range_bytes is None:
                single_result = fetch_bytes_range(
                    archive_url, [(one_run["start"], one_run["end"])]
                )
                range_bytes = single_result.get(one_run["start"])
            # それでも取れなければ、これ以上は続けられないので止める
            if range_bytes is None:
                raise RuntimeError(
                    "要求した範囲が返ってきませんでした（開始位置 "
                    + str(one_run["start"]) + "）"
                )
            for one_entry in one_run["entries"]:
                extract_one_entry(range_bytes, one_run["start"], one_entry, dataset_root)
                extracted_file_count = extracted_file_count + 1
            downloaded_byte_count = downloaded_byte_count + len(range_bytes)

        # 進み具合を1行で表示する
        elapsed_second_count = time.perf_counter() - start_time
        if elapsed_second_count > 0:
            speed_text = format_size(int(downloaded_byte_count / elapsed_second_count)) + "/s"
        else:
            speed_text = "計測中"
        percent_value = 100.0 * downloaded_byte_count / max(total_byte_count, 1)
        print("  [%3d%%] " % int(percent_value)
              + format_size(downloaded_byte_count) + " / " + format_size(total_byte_count)
              + "   " + speed_text
              + "   (" + str(batch_index + 1) + "/" + str(len(batch_list)) + " 回目)",
              flush=True)

    # 展開したファイル数を返す
    return extracted_file_count


# --------------------------------------------------------------------------
# 代替の方式: 書庫全体を落としてから展開する
# --------------------------------------------------------------------------

def download_whole_archive(archive_url, archive_path, expected_size, expected_md5):
    """
    zip 全体をダウンロードする（途中まで取得済みなら続きから）。

    範囲取得が使えない環境のための、単純で確実な方式です。
    通信が途中で切れた場合は、切れたところから自動でやり直します。
    """
    # 途中まで取得したものは .part という名前で置いておく
    temporary_path = archive_path + ".part"

    # 切れてもやり直せるよう、決めた回数まで繰り返す
    last_error = None
    for attempt_index in range(MAX_RETRY_COUNT):
        try:
            # 続きから受け取る（すでに全部あれば何もしない）
            download_archive_once(archive_url, temporary_path, expected_size)
        except RETRYABLE_NETWORK_ERRORS as error:
            # 通信が切れた場合は、少し待ってから続きのところからやり直す
            last_error = error
            wait_second_count = 2 ** attempt_index
            print("  [警告] ダウンロードが途中で切れました。"
                  + str(wait_second_count) + "秒待って続きから再開します")
            time.sleep(wait_second_count)
            continue
        # 例外が出なくても途中で終わっていることがあるので、大きさを確かめる
        if os.path.getsize(temporary_path) >= expected_size:
            last_error = None
            break
        last_error = RuntimeError("受信が途中で終わりました")
        wait_second_count = 2 ** attempt_index
        print("  [警告] 受信が途中で終わりました。"
              + str(wait_second_count) + "秒待って続きから再開します")
        time.sleep(wait_second_count)

    # 決めた回数やっても終わらなかった場合はエラーにする
    if last_error is not None:
        raise RuntimeError(
            "書庫のダウンロードに繰り返し失敗しました: " + str(last_error) + "\n"
            + "もう一度実行すると、続きから取得し直します。"
        )

    # 大きさが合っているかを確かめる
    if os.path.getsize(temporary_path) != expected_size:
        raise RuntimeError(
            "ダウンロードした書庫の大きさが合いません: "
            + str(os.path.getsize(temporary_path)) + " / " + str(expected_size) + "\n"
            + "次のファイルを削除してから、もう一度実行してください: " + temporary_path
        )

    # md5（内容の指紋）を照合する
    if expected_md5:
        print("  md5 を照合しています（数十秒かかります）")
        md5_calculator = hashlib.md5()
        with open(temporary_path, "rb") as input_file:
            while True:
                chunk_bytes = input_file.read(4 * 1024 * 1024)
                if not chunk_bytes:
                    break
                md5_calculator.update(chunk_bytes)
        if md5_calculator.hexdigest() != expected_md5:
            raise RuntimeError(
                "書庫の md5 が一致しません（壊れている可能性があります）。\n"
                + "次のファイルを削除してから、もう一度実行してください: " + temporary_path
            )
        print("  [OK] md5 が一致しました")

    # 正式な名前に変えて、そのパスを返す
    os.replace(temporary_path, archive_path)
    return archive_path


def download_archive_once(archive_url, temporary_path, expected_size):
    """
    zip 全体を、途中まで取得済みならその続きから受け取ってファイルに書き足す。

    通信が切れた場合は例外がそのまま外に出ます（呼び出し側でやり直します）。
    """
    # すでにどこまで取得できているかを調べる
    if os.path.exists(temporary_path):
        already_byte_count = os.path.getsize(temporary_path)
    else:
        already_byte_count = 0
    # すでに全部取得できていれば、何もしない
    if already_byte_count >= expected_size:
        return

    # 続きから取得するための範囲指定を作る（最初からなら指定しない）
    extra_header_dict = None
    if already_byte_count > 0:
        print("  途中まで取得済みです（" + format_size(already_byte_count)
              + "）。続きから取得します")
        extra_header_dict = {"Range": "bytes=" + str(already_byte_count) + "-"}

    # 接続する
    response = open_url_with_retry(archive_url, extra_header_dict)
    try:
        # 続きから取得を頼んだのに最初から返ってきた場合は、最初から書き直す
        if already_byte_count > 0 and response.status != 206:
            print("  [警告] 続きからの取得に対応していないため、最初から取得します")
            already_byte_count = 0
            file_mode = "wb"
        elif already_byte_count > 0:
            file_mode = "ab"
        else:
            file_mode = "wb"
        # 少しずつ読みながらファイルに書き足す
        start_time = time.perf_counter()
        last_report_byte_count = already_byte_count
        with open(temporary_path, file_mode) as output_file:
            while True:
                chunk_bytes = response.read(1024 * 1024)
                if not chunk_bytes:
                    break
                output_file.write(chunk_bytes)
                already_byte_count = already_byte_count + len(chunk_bytes)
                # 8MB ごとに進み具合を表示する
                if already_byte_count - last_report_byte_count >= 8 * 1024 * 1024:
                    last_report_byte_count = already_byte_count
                    elapsed_second_count = time.perf_counter() - start_time
                    percent_value = 100.0 * already_byte_count / max(expected_size, 1)
                    print("  [%3d%%] " % int(percent_value)
                          + format_size(already_byte_count) + " / "
                          + format_size(expected_size)
                          + "   経過 " + format_duration(elapsed_second_count),
                          flush=True)
    finally:
        response.close()


def extract_from_archive_file(archive_path, selected_entry_list, dataset_root):
    """ダウンロードした zip から、選んだファイルだけを取り出す。取り出した数を返す。"""
    # 取り出す名前の一覧を作る
    wanted_name_list = []
    for one_entry in selected_entry_list:
        wanted_name_list.append(one_entry["name"])
    # zip を開いて、1つずつ取り出す
    extracted_file_count = 0
    with zipfile.ZipFile(archive_path) as zip_file:
        for one_name in wanted_name_list:
            zip_file.extract(one_name, dataset_root)
            extracted_file_count = extracted_file_count + 1
    # 取り出した数を返す
    return extracted_file_count


# --------------------------------------------------------------------------
# ここからメインの処理
# --------------------------------------------------------------------------

def parse_part_names(part_text):
    """--parts に渡された文字列（例 "raw,annotations"）をリストにする。"""
    # カンマで区切り、前後の空白を取り除く
    part_name_list = []
    for one_text in part_text.split(","):
        one_name = one_text.strip()
        if one_name == "":
            continue
        # 知らない名前が来たら、その場で分かるようにエラーにする
        if one_name not in AVAILABLE_PART_NAMES:
            raise ValueError(
                "--parts に指定できない名前です: " + one_name + "\n"
                + "指定できるのは " + ", ".join(AVAILABLE_PART_NAMES) + " です。"
            )
        part_name_list.append(one_name)
    # 1つも指定が無ければエラーにする
    if len(part_name_list) == 0:
        raise ValueError("--parts が空です。指定できるのは " + ", ".join(AVAILABLE_PART_NAMES) + " です。")
    # できたリストを返す
    return part_name_list


def pad_label(label_text, width):
    """表示用に、文字列の右側を空白で埋めて幅をそろえる。

    日本語の文字は半角2文字ぶんの幅で表示されるため、その分を数えて調整します。
    """
    # 表示したときの幅を数える
    display_width = 0
    for one_character in label_text:
        # 全角（日本語など）なら2、それ以外は1として数える
        if ord(one_character) > 0x2E80:
            display_width = display_width + 2
        else:
            display_width = display_width + 1
    # 足りない分を空白で埋めて返す
    if display_width >= width:
        return label_text
    return label_text + " " * (width - display_width)


def print_selection_summary(selected_entry_list, archive_size):
    """取得するファイルの内訳を表示する。"""
    # 分類ごとの件数と合計サイズを数えるための辞書を用意する
    count_by_part = {}
    size_by_part = {}
    total_size = 0
    # 1件ずつ数える
    for one_entry in selected_entry_list:
        part_label = describe_entry_part(one_entry["name"])
        if part_label not in count_by_part:
            count_by_part[part_label] = 0
            size_by_part[part_label] = 0
        count_by_part[part_label] = count_by_part[part_label] + 1
        size_by_part[part_label] = size_by_part[part_label] + one_entry["uncompressed_size"]
        total_size = total_size + one_entry["uncompressed_size"]
    # 分類ごとに1行ずつ表示する
    for part_label in sorted(count_by_part):
        print("    " + pad_label(part_label, 16)
              + "%6d ファイル %10s" % (
                  count_by_part[part_label], format_size(size_by_part[part_label])
              ))
    # 合計を表示する（書庫全体の何%かも出す）
    percent_value = 100.0 * total_size / max(archive_size, 1)
    print("    " + pad_label("合計", 16)
          + "%6d ファイル %10s （書庫全体の %d%%）" % (
              len(selected_entry_list), format_size(total_size), int(percent_value)
          ))
    # 合計サイズを返す
    return total_size


def read_entries_from_archive_file(archive_path):
    """手元にある zip から、入っているファイルの一覧を読む。

    範囲取得で目次を読んだときと同じ形の辞書のリストを返します。
    （この経路では zip 全体が手元にあるので、位置の情報は使いません）
    """
    # 結果を入れるリストを用意する
    entry_list = []
    # zip を開いて、1件ずつ情報を写し取る
    with zipfile.ZipFile(archive_path) as zip_file:
        for one_info in zip_file.infolist():
            entry_list.append({
                "name": one_info.filename,
                "method": one_info.compress_type,
                "crc": one_info.CRC,
                "compressed_size": one_info.compress_size,
                "uncompressed_size": one_info.file_size,
                "local_offset": one_info.header_offset,
                "end_offset": one_info.header_offset,
            })
    # できあがったリストを返す
    return entry_list


def choose_entries_to_download(entry_list, part_name_list, dataset_root,
                               archive_size, force_flag, limit_file_count=0):
    """
    取得するファイルを選び、内訳を表示する。

    limit_file_count に1以上を指定すると、その数までしか取得しません（動作確認用）。
    返り値は (これから取得するファイルの一覧, すでに置いてあった件数) です。
    """
    # --parts の指定に合うファイルを選ぶ
    selected_entry_list = select_entries(entry_list, part_name_list)
    # 選んだ結果が空なら、指定が間違っている
    if len(selected_entry_list) == 0:
        raise RuntimeError("取得対象のファイルがありません。--parts の指定を確かめてください。")
    # 内訳を表示する
    print_selection_summary(selected_entry_list, archive_size)
    # すでに置いてあるものは飛ばす（--force が付いていれば飛ばさない）
    already_present_count = 0
    if not force_flag:
        selected_entry_list, already_present_count = remove_already_downloaded_entries(
            selected_entry_list, dataset_root
        )
        if already_present_count > 0:
            print("    すでに取得済み : " + str(already_present_count) + " ファイル（飛ばします）")
    # 動作確認用に枚数が絞られていれば、先頭からその数だけにする
    if limit_file_count > 0 and len(selected_entry_list) > limit_file_count:
        selected_entry_list = selected_entry_list[:limit_file_count]
        print("    [情報] --limit-files により " + str(limit_file_count)
              + " ファイルだけ取得します")
    # 結果を返す
    return selected_entry_list, already_present_count


def print_next_step_message():
    """次に実行すべきコマンドを表示する。"""
    print("次は下記を実行してください:")
    print("  python src/prepare_dataset.py")


def main():
    """コマンドから実行されたときに動く、このスクリプトの本体。"""
    # コマンドライン引数の設定を作る
    argument_parser = argparse.ArgumentParser(
        description="KFuji RGB-DS データセットを Zenodo から取得して配置します。"
    )
    # データセットの置き場所を指定する引数を追加する
    argument_parser.add_argument(
        "--dataset-root",
        default=None,
        help="KFuji_RGB-DS_dataset フォルダを作る場所。"
             "省略すると環境変数 KFUJI_DATASET_ROOT、それも無ければ既定の場所を使います",
    )
    # 取得する範囲を指定する引数を追加する
    argument_parser.add_argument(
        "--parts",
        default=",".join(DEFAULT_PART_NAMES),
        help="取得する範囲をカンマ区切りで指定します。"
             "raw=生データ / annotations=注釈 / images=切り出しjpg / "
             "ds-mat=未使用の深度データ / all=すべて"
             "（既定値 " + ",".join(DEFAULT_PART_NAMES) + "）",
    )
    # Zenodo の記録番号を指定する引数を追加する
    argument_parser.add_argument(
        "--record-id",
        default=DEFAULT_RECORD_ID,
        help="Zenodo の記録番号（既定値 " + DEFAULT_RECORD_ID + "）",
    )
    # 動作確認のため、取得する枚数を絞る引数を追加する
    argument_parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="動作確認用。取得するファイル数をこの数までにします（0 なら制限なし）",
    )
    # すでにあるファイルも取り直す引数を追加する
    argument_parser.add_argument(
        "--force",
        action="store_true",
        help="すでに置いてあるファイルも取得し直します",
    )
    # 取得せずに内訳だけ見る引数を追加する
    argument_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際には取得せず、何をどれだけ取得するかだけを表示します",
    )
    # 書庫全体を落とす方式に切り替える引数を追加する
    argument_parser.add_argument(
        "--full-archive",
        action="store_true",
        help="必要な部分だけではなく、zip 全体(2.9GB)を取得してから展開します",
    )
    # 書庫を消さずに残す引数を追加する
    argument_parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="--full-archive のとき、展開後も zip を削除しません",
    )
    # 指定された引数を読み取る
    parsed_arguments = argument_parser.parse_args()

    # 取得する範囲を読み取る
    part_name_list = parse_part_names(parsed_arguments.parts)

    # データセットの置き場所を決める
    if parsed_arguments.dataset_root is None:
        dataset_root = common_paths.get_default_dataset_root()
    else:
        dataset_root = os.path.abspath(parsed_arguments.dataset_root)

    # 処理の開始を画面に表示する
    print("=" * 70)
    print("データセットの取得を開始します")
    print("=" * 70)
    print("  置き場所     : " + dataset_root)
    print("  取得する範囲 : " + ",".join(part_name_list))
    print("")
    print_license_notice()
    print("")

    # ---------------- 手順1: Zenodo の記録情報を取得する --------------------
    print("[手順1] Zenodo の記録情報を取得します (record " + parsed_arguments.record_id + ")")
    record_information = fetch_record_metadata(parsed_arguments.record_id)
    archive_url = record_information["url"]
    archive_size = record_information["size"]
    print("  タイトル : " + record_information["title"])
    print("  書庫     : " + ARCHIVE_FILE_NAME + "  " + format_size(archive_size))
    print("")

    # ---------------- 手順2: 書庫の目次を読み込む ---------------------------
    print("[手順2] 書庫の目次を読み込みます（ここでは数百KBしか通信しません）")
    try:
        entry_list = read_central_directory(archive_url, archive_size)
        can_use_range_request = True
    except RangeNotSupportedError as error:
        # 範囲取得が使えない環境なので、書庫全体を落とす方式に切り替える
        print("  [警告] " + str(error))
        print("  [情報] 書庫全体を取得する方式に切り替えます")
        entry_list = None
        can_use_range_request = False
    if can_use_range_request:
        print("  書庫の中のファイル数 : " + str(len(entry_list)))
    print("")

    # 書庫全体を落とす方式を使うかどうかを決める
    # （--full-archive が指定されたときと、範囲取得が使えない環境のとき）
    use_full_archive = parsed_arguments.full_archive or (not can_use_range_request)
    archive_path = None
    selected_entry_list = None
    already_present_count = 0

    # ---------------- 手順3: 取得するファイルを選ぶ -------------------------
    # 目次が読めていれば、何かを落とす前にここで選んでおく。
    # こうしておけば、すでに全部そろっている場合に1バイトも落とさずに済む。
    if entry_list is not None:
        print("[手順3] 取得するファイルを選びます")
        selected_entry_list, already_present_count = choose_entries_to_download(
            entry_list, part_name_list, dataset_root, archive_size,
            parsed_arguments.force, parsed_arguments.limit_files
        )
        print("")

        # --dry-run なら、ここで終わる
        if parsed_arguments.dry_run:
            print("[情報] --dry-run が指定されているため、ここで終了します")
            print("  これから取得するファイル数 : " + str(len(selected_entry_list)))
            if len(selected_entry_list) > 0 and can_use_range_request:
                run_list = build_byte_runs(selected_entry_list)
                batch_list = build_request_batches(run_list)
                print("  通信の回数（見込み）       : " + str(len(batch_list)) + " 回")
            return

        # 取得するものが無ければ、何もせずに終わる
        if len(selected_entry_list) == 0:
            print("[完了] 必要なファイルはすべて揃っています: " + dataset_root)
            print("")
            print_next_step_message()
            return
    elif parsed_arguments.dry_run:
        # 目次が読めない環境では、--dry-run で内訳を出せない
        print("[情報] 範囲取得が使えないため、--dry-run では内訳を表示できません")
        return

    # ---------------- 手順4: ダウンロードして展開する -----------------------
    print("[手順4] ダウンロードして展開します")
    # 置き場所のフォルダを作る
    os.makedirs(dataset_root, exist_ok=True)

    if use_full_archive:
        # 書庫全体を落としてから、その中の必要なファイルだけを取り出す
        print("  書庫全体をダウンロードします（" + format_size(archive_size) + "）")
        archive_path = download_whole_archive(
            archive_url,
            os.path.join(dataset_root, ARCHIVE_FILE_NAME),
            archive_size,
            record_information["md5"],
        )
        # 目次がまだ読めていなければ、手元の zip から読んでここで選ぶ
        if selected_entry_list is None:
            entry_list = read_entries_from_archive_file(archive_path)
            print("")
            print("[手順3] 取得するファイルを選びます")
            selected_entry_list, already_present_count = choose_entries_to_download(
                entry_list, part_name_list, dataset_root, archive_size,
                parsed_arguments.force, parsed_arguments.limit_files
            )
            print("")
        # 落とし済みの zip から取り出す
        extracted_file_count = extract_from_archive_file(
            archive_path, selected_entry_list, dataset_root
        )
    else:
        try:
            # 必要な範囲だけを取得して展開する
            extracted_file_count = download_selected_entries(
                archive_url, selected_entry_list, dataset_root
            )
        except RangeNotSupportedError as error:
            # 途中で範囲取得が使えないと分かった場合は、書庫全体の方式に切り替える
            print("  [警告] " + str(error))
            print("  [情報] 書庫全体を取得する方式に切り替えます（"
                  + format_size(archive_size) + "）")
            archive_path = download_whole_archive(
                archive_url,
                os.path.join(dataset_root, ARCHIVE_FILE_NAME),
                archive_size,
                record_information["md5"],
            )
            extracted_file_count = extract_from_archive_file(
                archive_path, selected_entry_list, dataset_root
            )
            use_full_archive = True
    print("")

    # ---------------- 手順5: 後片付けと結果の表示 ---------------------------
    print("[手順5] 結果")
    if not use_full_archive:
        print("  [OK] " + str(extracted_file_count)
              + " ファイルを展開し、すべて CRC が一致しました")
    else:
        print("  [OK] " + str(extracted_file_count) + " ファイルを展開しました")
    if already_present_count > 0:
        print("  [情報] すでにあった " + str(already_present_count) + " ファイルはそのまま使います")
    # 書庫全体の方式を使った場合は、zip を消す（残す指定があれば消さない）
    if use_full_archive and archive_path is not None:
        if parsed_arguments.keep_archive:
            print("  [情報] 書庫を残しました: " + archive_path)
        else:
            os.remove(archive_path)
            print("  [情報] 展開に使った zip を削除しました（--keep-archive で残せます）")
    print("")

    # 全ての処理が終わったことを表示する
    print("=" * 70)
    print("データセットの取得が完了しました")
    print("=" * 70)
    print("  置き場所 : " + dataset_root)
    print("")
    print_next_step_message()


# このファイルが直接実行されたときだけ main() を呼ぶ
# （他のファイルから読み込まれたときは実行しない、という決まり文句）
if __name__ == "__main__":
    main()
