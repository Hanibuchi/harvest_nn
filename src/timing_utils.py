# ============================================================================
# このファイルは何をするものか
# ----------------------------------------------------------------------------
# 処理時間を正しく測るための道具をまとめたファイルです。
#
# 【GPUの処理時間を測るときの注意】
#   PyTorch がGPU上で計算を行うとき、Python のプログラムは
#   「GPUに計算を頼んだ時点」で次の行に進んでしまいます（非同期実行）。
#   そのため、何もしないで時間を測ると
#   「GPUの計算が終わる前に測り終える」ことになり、実際よりずっと短い
#   時間が出てしまいます。
#   これを防ぐために、時間を測る前と後に torch.cuda.synchronize() を呼んで
#   「GPUの計算が全部終わるまで待つ」必要があります。
#   このファイルの start_timer / stop_timer は、その待ち合わせを
#   自動で行います。
#
# 【時刻の取り方について】
#   time.time() ではなく time.perf_counter() を使います。
#   perf_counter は経過時間を測る専用の時計で、精度が高く、
#   途中でシステムの時刻設定が変わっても影響を受けません。
# ============================================================================

# 時間を測るための標準ライブラリ
import time

# 数値計算ライブラリ（統計量の計算に使う）
import numpy as np
# ディープラーニングのライブラリ（GPUの待ち合わせに使う）
import torch


def resolve_device(requested_device_name):
    """
    使用する計算装置（GPUかCPUか）を決めて、torch のデバイスとして返す。

    requested_device_name に "auto" を指定すると、
    使えるものを自動で選びます。優先順位は次のとおりです。
        1. CUDA（NVIDIAのGPU。ワークステーションやJetsonで使われる）
        2. MPS （Appleシリコン搭載MacのGPU）
        3. CPU （GPUが使えない場合）
    これにより、同じスクリプトをGPUのない環境でもそのまま動かせます。
    """
    # 明示的に装置名が指定されている場合は、その指定をそのまま使う
    if requested_device_name is not None and requested_device_name != "auto":
        return torch.device(requested_device_name)
    # NVIDIAのGPUが使えるなら、それを選ぶ
    if torch.cuda.is_available():
        return torch.device("cuda")
    # AppleシリコンのGPU(MPS)が使えるなら、それを選ぶ。
    # 古いバージョンの PyTorch には mps の項目自体が無いことがあるので、
    # エラーが起きても止まらないように try で囲んでおく
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    # どちらも使えないならCPUを選ぶ
    return torch.device("cpu")


def synchronize_device(device):
    """
    計算装置の処理が全部終わるまで待つ。

    GPUを使っているときだけ意味のある処理で、CPUのときは何もしません。
    """
    # 装置の種類が cuda（NVIDIAのGPU）の場合
    if device.type == "cuda":
        # GPUに頼んだ計算が全部終わるまで待つ
        torch.cuda.synchronize()
    # 装置の種類が mps（AppleのGPU）の場合
    elif device.type == "mps":
        # MPS にも同じ待ち合わせの命令があるので呼ぶ
        torch.mps.synchronize()


def start_timer(device):
    """
    時間の計測を開始し、開始時刻を返す。

    計測を始める前に、それまでの計算が全部終わっているか確認します。
    （前の処理の残りが今回の計測時間に混ざらないようにするため）
    """
    # それまでの計算が終わるまで待つ
    synchronize_device(device)
    # 現在時刻を記録して返す
    return time.perf_counter()


def stop_timer(start_time, device):
    """
    計測を終了し、start_timer を呼んでからの経過時間をミリ秒で返す。
    """
    # 測りたい処理が終わるまで待つ
    synchronize_device(device)
    # 現在時刻から開始時刻を引いて経過秒数を求める
    elapsed_seconds = time.perf_counter() - start_time
    # 秒をミリ秒に直して返す（1秒 = 1000ミリ秒）
    return elapsed_seconds * 1000.0


def summarize_measurements(measurement_list):
    """
    計測値（ミリ秒）のリストから、平均・中央値・標準偏差などの統計量を計算する。

    返り値は辞書で、次のキーを持ちます。
        count  : 計測した回数
        mean   : 平均値
        median : 中央値（小さい順に並べたときの真ん中の値）
        std    : 標準偏差（値のばらつきの大きさ）
        min    : 最小値
        max    : 最大値
        p95    : 95パーセンタイル（遅い方から5%の位置にある値）
    """
    # 計測値が1つも無い場合は、全て0にした辞書を返す
    if len(measurement_list) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p95": 0.0,
        }
    # 計算しやすいように numpy の配列に変換する
    measurement_array = np.array(measurement_list, dtype=np.float64)
    # 標準偏差を計算する。計測が1回だけのときは、ばらつきを0とする
    if measurement_array.size > 1:
        # ddof=1 は「標本標準偏差」を意味する（実験データではこちらを使うのが一般的）
        standard_deviation = float(np.std(measurement_array, ddof=1))
    else:
        standard_deviation = 0.0
    # 統計量をまとめた辞書を作って返す
    return {
        "count": int(measurement_array.size),
        "mean": float(np.mean(measurement_array)),
        "median": float(np.median(measurement_array)),
        "std": standard_deviation,
        "min": float(np.min(measurement_array)),
        "max": float(np.max(measurement_array)),
        "p95": float(np.percentile(measurement_array, 95)),
    }
