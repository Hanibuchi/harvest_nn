# ============================================================================
# このファイルは何をするものか
# ----------------------------------------------------------------------------
# 計測を実行したマシンの情報（CPU名・GPU名・メモリ量・ライブラリのバージョン）を
# 集めるためのファイルです。
#
# 2台のマシン（研究室のワークステーションとNVIDIA Jetson）で計測した結果を
# 後から比較するとき、「どのマシンで測った結果か」が結果ファイルに
# 記録されていないと比較になりません。
# そこで計測スクリプトは、実行のたびにこのファイルの関数を呼んで
# マシン情報を集め、結果のCSVに一緒に書き込みます。
# ============================================================================

# ファイルやフォルダの存在確認に使う標準ライブラリ
import os
# OS名やPythonのバージョンを調べるための標準ライブラリ
import platform
# 外部コマンド（sysctlなど）を実行するための標準ライブラリ
import subprocess
# ホスト名を調べるための標準ライブラリ
import socket
# 日時を扱うための標準ライブラリ
import datetime

# CPUのコア数やメモリ量を調べるためのライブラリ
import psutil
# ディープラーニングのライブラリ（GPU情報とバージョンの取得に使う）
import torch


def get_cpu_model_name():
    """CPUの製品名（例: "Apple M2 Pro"）を調べて文字列で返す。"""
    # 現在のOSの種類を調べる（"Darwin"=macOS, "Linux"=Linux, "Windows"=Windows）
    operating_system_name = platform.system()

    # macOS の場合は sysctl コマンドで製品名を取得する
    if operating_system_name == "Darwin":
        try:
            # sysctl コマンドを実行して結果を文字列で受け取る
            command_output = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            )
            # 前後の余分な改行や空白を取り除いて返す
            return command_output.strip()
        except Exception:
            # コマンドが失敗した場合は、分かる範囲の情報を返す
            return platform.processor()

    # Linux の場合は /proc/cpuinfo というファイルから製品名を探す
    if operating_system_name == "Linux":
        try:
            # CPU情報が書かれたファイルを開く
            with open("/proc/cpuinfo", "r", encoding="utf-8") as opened_file:
                # ファイルの中身を1行ずつ順番に見ていく
                for one_line in opened_file:
                    # "model name" で始まる行がCPUの製品名を表す（x86系のPCの場合）
                    if one_line.startswith("model name"):
                        # ":" の後ろの部分を取り出して返す
                        return one_line.split(":", 1)[1].strip()
                    # Jetson などのARM系では "Model" という行に書かれていることがある
                    if one_line.startswith("Model"):
                        return one_line.split(":", 1)[1].strip()
        except Exception:
            # ファイルが読めなかった場合は何もしない（次の処理に進む）
            pass
        # 上で見つからなかった場合は、platform モジュールの情報を返す
        return platform.processor()

    # macOS でも Linux でもない場合は、platform モジュールの情報を返す
    return platform.processor()


def get_jetson_model_name():
    """
    NVIDIA Jetson で動いている場合、そのモデル名を返す。
    Jetson でない場合は空文字を返す。
    """
    # Jetson ではこのファイルにボードのモデル名が書かれている
    device_tree_model_path = "/proc/device-tree/model"
    # そのファイルが存在するか確認する
    if os.path.exists(device_tree_model_path):
        try:
            # ファイルを開いて中身を読む
            with open(device_tree_model_path, "r", encoding="utf-8", errors="ignore") as opened_file:
                model_name = opened_file.read()
            # 文字列の終端に入ることがある特殊な文字と空白を取り除く
            model_name = model_name.replace("\x00", "").strip()
            # 読み取ったモデル名を返す
            return model_name
        except Exception:
            # 読めなかった場合は空文字を返す
            return ""
    # ファイルが無ければ Jetson ではないので空文字を返す
    return ""


def get_gpu_information(device):
    """
    使用している計算装置(GPU/CPU)の名前とメモリ量を調べて辞書で返す。
    """
    # 装置が cuda（NVIDIAのGPU）の場合
    if device.type == "cuda":
        # GPUの製品名を取得する
        gpu_name = torch.cuda.get_device_name(0)
        # GPUの搭載メモリ量（バイト単位）を取得する
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        # バイトをギガバイトに直す（小数第2位まで）
        gpu_memory_gigabytes = round(gpu_memory_bytes / (1024 ** 3), 2)
    # 装置が mps（AppleシリコンのGPU）の場合
    elif device.type == "mps":
        # MPS では製品名を取得する仕組みが無いので、固定の文字列にする
        gpu_name = "Apple Silicon GPU (MPS)"
        # MPS はCPUとメモリを共有するため、専用メモリ量は不明として0にする
        gpu_memory_gigabytes = 0.0
    # 装置がCPUの場合
    else:
        # GPUを使っていないことが分かる文字列を入れる
        gpu_name = "CPU only (no GPU)"
        # GPUメモリは無いので0にする
        gpu_memory_gigabytes = 0.0
    # 調べた内容を辞書にまとめて返す
    return {"gpu_name": gpu_name, "gpu_memory_gb": gpu_memory_gigabytes}


def collect_machine_information(machine_name, device):
    """
    計測結果に一緒に記録するための、マシン情報一式を集めて辞書で返す。

    引数:
        machine_name : 結果を区別するための名前（例: "workstation", "jetson_orin"）
                       None を渡した場合は、そのマシンのホスト名が使われます
        device       : timing_utils.resolve_device() が返した計算装置
    """
    # マシン名が指定されていなければ、ホスト名を使う
    if machine_name is None or machine_name == "":
        machine_name = socket.gethostname()

    # GPUの情報を取得する
    gpu_information = get_gpu_information(device)

    # NVIDIA Jetson のモデル名を取得する（Jetsonでなければ空文字になる）
    jetson_model_name = get_jetson_model_name()

    # PyTorch がどのCUDAバージョン向けにビルドされているかを取得する
    # （CPU版のPyTorchでは None になるので、その場合は "none" という文字列にする）
    if torch.version.cuda is None:
        cuda_version_text = "none"
    else:
        cuda_version_text = torch.version.cuda

    # cuDNN（GPU用の深層学習ライブラリ）のバージョンを取得する
    if torch.backends.cudnn.is_available():
        cudnn_version_text = str(torch.backends.cudnn.version())
    else:
        cudnn_version_text = "none"

    # ultralytics のバージョンを取得する（読み込みに失敗しても止まらないようにする）
    try:
        import ultralytics
        ultralytics_version_text = ultralytics.__version__
    except Exception:
        ultralytics_version_text = "unknown"

    # OpenCV のバージョンを取得する
    try:
        import cv2
        opencv_version_text = cv2.__version__
    except Exception:
        opencv_version_text = "unknown"

    # numpy のバージョンを取得する
    try:
        import numpy
        numpy_version_text = numpy.__version__
    except Exception:
        numpy_version_text = "unknown"

    # 集めた情報を1つの辞書にまとめる
    machine_information = {
        # 結果を区別するためのマシン名
        "machine_name": machine_name,
        # 計測を実行した日時（例: 2026-09-19T14:30:00）
        "measured_at": datetime.datetime.now().isoformat(timespec="seconds"),
        # OSの名前とバージョン（例: Darwin 25.6.0）
        "os": platform.system() + " " + platform.release(),
        # CPUの命令セット（例: arm64, x86_64）
        "architecture": platform.machine(),
        # CPUの製品名
        "cpu_name": get_cpu_model_name(),
        # 物理的なCPUコアの数
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        # 論理的なCPUコアの数（ハイパースレッディングを含む）
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        # 搭載メモリ量（ギガバイト、小数第2位まで）
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        # Jetson の場合のボード名（Jetsonでなければ空文字）
        "jetson_model": jetson_model_name,
        # 計測に使った装置の種類（cuda / mps / cpu）
        "device_type": device.type,
        # GPUの製品名
        "gpu_name": gpu_information["gpu_name"],
        # GPUの搭載メモリ量（ギガバイト）
        "gpu_memory_gb": gpu_information["gpu_memory_gb"],
        # Python のバージョン
        "python_version": platform.python_version(),
        # PyTorch のバージョン
        "torch_version": torch.__version__,
        # PyTorch が対応しているCUDAのバージョン
        "cuda_version": cuda_version_text,
        # cuDNN のバージョン
        "cudnn_version": cudnn_version_text,
        # ultralytics のバージョン
        "ultralytics_version": ultralytics_version_text,
        # OpenCV のバージョン
        "opencv_version": opencv_version_text,
        # numpy のバージョン
        "numpy_version": numpy_version_text,
    }
    # 作った辞書を返す
    return machine_information


def print_machine_information(machine_information):
    """集めたマシン情報を、画面に見やすく表示する。"""
    # 見出しの線を表示する
    print("=" * 70)
    # 見出しの文字を表示する
    print("計測マシンの情報")
    # 見出しの線を表示する
    print("=" * 70)
    # 辞書のキーと値を1組ずつ順番に表示する
    for information_key in machine_information:
        # キーの名前を左詰めで22文字分の幅に揃えて、値と一緒に表示する
        print("  " + information_key.ljust(22) + ": " + str(machine_information[information_key]))
    # 最後に線を表示する
    print("=" * 70)
