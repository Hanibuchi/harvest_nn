# ============================================================================
# このファイルは何をするものか
# ----------------------------------------------------------------------------
# 【(B) ファインチューニング用スクリプト】です。
#
# COCOデータセットで学習済みの YOLOv8 のモデルを読み込み、
# prepare_dataset.py で用意したりんごのデータで追加学習（ファインチューニング）します。
#
# 「ファインチューニング」とは、たくさんの一般的な物体で学習済みのモデルを
# 出発点にして、自分のデータで学習し直すことです。
# ゼロから学習するより、少ないデータと時間で高い精度が得られます。
#
# 【モデルの大きさについて】
#   --model-size で n / s / m を選べます。右にいくほど精度が上がりますが遅くなります。
#       n (nano)  : 一番小さくて速い。Jetson などの非力な機器向け
#       s (small) : 中間
#       m (medium): 大きくて精度が高いが、その分遅い
#   論文の比較実験のように「軽いモデルと重いモデルで処理時間がどう変わるか」を
#   調べたい場合は、3つとも学習しておくとよいです。
#
# 【実行例】
#   python src/train_yolo.py --model-size n --epochs 100
#   python src/train_yolo.py --model-size s --epochs 100 --batch 8
#
# 【注意】
#   初めて実行するとき、学習済みモデル(yolov8n.pt など)を
#   インターネットから自動でダウンロードします。ネットにつながる環境で
#   実行してください。（ダウンロードは1回だけです）
# ============================================================================

# コマンドライン引数を扱うための標準ライブラリ
import argparse
# ファイルやフォルダを操作するための標準ライブラリ
import os
# ファイルをコピーするための標準ライブラリ
import shutil

# 自作のモジュール（パスの管理）を読み込む
import common_paths
# 自作のモジュール（計算装置の判定）を読み込む
import timing_utils


def convert_device_for_ultralytics(device):
    """
    PyTorch のデバイス表現を、ultralytics が受け取れる形に変換する。

    ultralytics では NVIDIA のGPUを使うとき、"cuda" ではなく
    GPUの番号（0 など）を渡す決まりになっているため、ここで変換します。
    """
    # NVIDIAのGPUの場合は、GPU番号の 0 を返す
    if device.type == "cuda":
        return 0
    # AppleシリコンのGPUの場合は、"mps" という文字列を返す
    if device.type == "mps":
        return "mps"
    # それ以外（CPU）の場合は、"cpu" という文字列を返す
    return "cpu"


def main():
    """コマンドから実行されたときに動く、このスクリプトの本体。"""
    # コマンドライン引数の設定を作る
    argument_parser = argparse.ArgumentParser(
        description="YOLOv8 をりんごデータセットでファインチューニングします。"
    )
    # モデルの大きさを指定する引数を追加する
    argument_parser.add_argument(
        "--model-size",
        default="n",
        choices=["n", "s", "m"],
        help="モデルの大きさ。n(小)/ s(中) / m(大) から選びます（既定値 n）",
    )
    # 学習の繰り返し回数を指定する引数を追加する
    argument_parser.add_argument(
        "--epochs", type=int, default=100, help="学習を何周するか（既定値 100）"
    )
    # 学習時の画像サイズを指定する引数を追加する
    argument_parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="学習時にモデルへ入力する画像の一辺のサイズ（既定値 640）",
    )
    # 一度に処理する画像の枚数を指定する引数を追加する
    argument_parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="一度に処理する画像の枚数。GPUのメモリが足りない場合は小さくします（既定値 16）",
    )
    # 使用する計算装置を指定する引数を追加する
    argument_parser.add_argument(
        "--device",
        default="auto",
        help="使用する装置。auto / cuda / mps / cpu から選びます（既定値 auto）",
    )
    # 乱数シードを指定する引数を追加する
    argument_parser.add_argument(
        "--seed", type=int, default=42, help="乱数の種。同じ値なら学習結果を再現しやすくなります（既定値 42）"
    )
    # データセット設定ファイルの場所を指定する引数を追加する
    argument_parser.add_argument(
        "--data",
        default=None,
        help="dataset.yaml のパス。省略すると outputs/dataset_yolo/dataset.yaml を使います",
    )
    # 学習を打ち切るまでの待ち回数を指定する引数を追加する
    argument_parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="精度が何周続けて改善しなかったら学習を打ち切るか（既定値 30）",
    )
    # データ読み込みの並列数を指定する引数を追加する
    argument_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="データ読み込みを並列で行う数。Jetsonでは 2 程度が無難です（既定値 4）",
    )
    # 実行名を指定する引数を追加する
    argument_parser.add_argument(
        "--name",
        default=None,
        help="学習結果を保存するフォルダの名前。省略すると自動で付けます",
    )
    # 実際に引数を読み取る
    parsed_arguments = argument_parser.parse_args()

    # データセット設定ファイルの場所を決める
    if parsed_arguments.data is None:
        dataset_yaml_path = os.path.join(
            common_paths.get_outputs_directory(), "dataset_yolo", "dataset.yaml"
        )
    else:
        dataset_yaml_path = parsed_arguments.data

    # 設定ファイルが無ければ、先にデータ準備が必要なのでエラーを出して止める
    if not os.path.exists(dataset_yaml_path):
        raise FileNotFoundError(
            "データセット設定ファイルが見つかりません: " + dataset_yaml_path + "\n"
            + "先に python src/prepare_dataset.py を実行してください。"
        )

    # 使用する計算装置を決める
    device = timing_utils.resolve_device(parsed_arguments.device)
    # ultralytics が受け取れる形に変換する
    ultralytics_device = convert_device_for_ultralytics(device)

    # 学習結果を保存するフォルダの名前を決める
    if parsed_arguments.name is None:
        run_name = "yolov8" + parsed_arguments.model_size + "_apple"
    else:
        run_name = parsed_arguments.name

    # 学習結果を保存する親フォルダのパスを作る
    training_output_directory = os.path.join(common_paths.get_outputs_directory(), "training")

    # 出発点にする学習済みモデルのファイル名を作る（例: yolov8n.pt）
    pretrained_model_name = "yolov8" + parsed_arguments.model_size + ".pt"

    # 学習の設定を画面に表示する
    print("=" * 70)
    print("YOLOv8 のファインチューニングを開始します")
    print("=" * 70)
    print("  出発点のモデル   : " + pretrained_model_name)
    print("  データセット設定 : " + dataset_yaml_path)
    print("  学習の周回数     : " + str(parsed_arguments.epochs))
    print("  入力画像サイズ   : " + str(parsed_arguments.imgsz))
    print("  バッチサイズ     : " + str(parsed_arguments.batch))
    print("  計算装置         : " + str(ultralytics_device))
    print("  乱数シード       : " + str(parsed_arguments.seed))
    print("  保存先           : " + os.path.join(training_output_directory, run_name))
    print("=" * 70)
    print("")

    # ultralytics のライブラリを読み込む
    # （ここで読み込むのは、読み込みに数秒かかるため。引数のミスがあれば先に気付ける）
    from ultralytics import YOLO

    # 学習済みモデルを読み込む（手元に無ければ自動でダウンロードされる）
    yolo_model = YOLO(pretrained_model_name)

    # 学習を実行する
    yolo_model.train(
        # どのデータで学習するか
        data=dataset_yaml_path,
        # 何周学習するか
        epochs=parsed_arguments.epochs,
        # 入力画像の一辺のサイズ
        imgsz=parsed_arguments.imgsz,
        # 一度に処理する枚数
        batch=parsed_arguments.batch,
        # 使用する装置
        device=ultralytics_device,
        # 乱数の種
        seed=parsed_arguments.seed,
        # 何周改善しなければ打ち切るか
        patience=parsed_arguments.patience,
        # データ読み込みの並列数
        workers=parsed_arguments.workers,
        # 結果を保存する親フォルダ
        project=training_output_directory,
        # 結果を保存するフォルダ名
        name=run_name,
        # 同じ名前のフォルダがあれば上書きする
        exist_ok=True,
        # 学習の経過をグラフにして保存する
        plots=True,
    )

    # 学習で一番成績の良かった重みファイルのパスを組み立てる
    best_weights_path = os.path.join(
        training_output_directory, run_name, "weights", "best.pt"
    )

    # 学習が終わったことを表示する
    print("")
    print("=" * 70)
    print("学習が完了しました")
    print("=" * 70)
    print("  学習済みの重み: " + best_weights_path)
    print("")

    # --- テストセットでの精度を測る ---
    print("テストセットで精度を評価します（学習にもハイパーパラメータ調整にも")
    print("使っていない、完全に未知のデータでの成績です）")
    print("")

    # 学習済みの重みを読み込む
    trained_model = YOLO(best_weights_path)
    # テストセットで評価を実行する
    validation_result = trained_model.val(
        # どのデータで評価するか
        data=dataset_yaml_path,
        # "test" と指定すると、dataset.yaml の test: の画像が使われる
        split="test",
        # 入力画像の一辺のサイズ（学習時と合わせる）
        imgsz=parsed_arguments.imgsz,
        # 使用する装置
        device=ultralytics_device,
        # 結果を保存する親フォルダ
        project=training_output_directory,
        # 結果を保存するフォルダ名
        name=run_name + "_test_eval",
        # 同じ名前のフォルダがあれば上書きする
        exist_ok=True,
    )

    # 評価結果の主な数値を表示する
    print("")
    print("=" * 70)
    print("テストセットでの精度")
    print("=" * 70)
    # mAP50 は「重なり50%以上を正解とみなしたときの平均精度」。1に近いほど良い
    print("  mAP50     : %.4f" % validation_result.box.map50)
    # mAP50-95 はより厳しい基準で測った平均精度
    print("  mAP50-95  : %.4f" % validation_result.box.map)
    # 適合率は「検出したもののうち、本当にりんごだった割合」
    print("  適合率(P) : %.4f" % validation_result.box.mp)
    # 再現率は「実際にあるりんごのうち、検出できた割合」
    print("  再現率(R) : %.4f" % validation_result.box.mr)
    print("=" * 70)
    print("")

    # --- 学習済みの重みを、分かりやすい場所にコピーする ---
    # 重みを置くフォルダのパスを作る
    weights_directory = os.path.join(common_paths.get_outputs_directory(), "weights")
    # そのフォルダを作る
    os.makedirs(weights_directory, exist_ok=True)
    # コピー先のファイル名を作る
    copied_weights_path = os.path.join(weights_directory, run_name + "_best.pt")
    # 重みファイルをコピーする
    shutil.copy2(best_weights_path, copied_weights_path)
    # コピーしたことを表示する
    print("学習済みの重みをコピーしました: " + copied_weights_path)
    print("")
    print("次は計測を実行してください:")
    print("  python src/measure_centralized.py --weights " + copied_weights_path
          + " --machine-name あなたのマシン名")


# このファイルが直接実行されたときだけ main() を呼ぶ
if __name__ == "__main__":
    main()
