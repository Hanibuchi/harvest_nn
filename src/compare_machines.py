# ============================================================================
# このファイルは何をするものか
# ----------------------------------------------------------------------------
# 【2台以上のマシンの計測結果を比較するスクリプト】です。
#
# measure_centralized.py が出力した「統計値のまとめCSV」
# （timings_summary_<マシン名>.csv）を2つ以上読み込んで、
# 次の4つのファイルを作ります。
#
#   1. comparison_table.csv    … 比較表（表計算ソフトで開けます）
#   2. comparison_table.md     … 比較表（文章に貼り付けやすい形式）
#   3. comparison_steps.png    … ステップごとの処理時間を並べた棒グラフ
#   4. comparison_composition.png … 処理時間の内訳（何が全体の何%か）のグラフ
#
# 【実行例】
#   python src/compare_machines.py \
#       --summary-csv outputs/measurements/timings_summary_workstation.csv \
#                     outputs/measurements/timings_summary_jetson_orin.csv
#
# 【グラフの文字が英語になっている理由】
#   グラフの中の文字は英語にしてあります。
#   日本語をグラフに描くには日本語フォントの設定が必要で、
#   Jetson など環境によってはフォントが入っておらず、
#   文字が「□□□」と豆腐のように化けてしまうためです。
#   それぞれのラベルの意味は次のとおりです。
#       1. Image acquisition … ステップ1 画像取得
#       2. Depth alignment   … ステップ2 深度アライメント
#       3. Preprocessing     … ステップ3 推論前処理
#       4a. Inference        … ステップ4a 画像推論
#       4b. NMS              … ステップ4b 後処理(NMS)
#       5. 3D localization   … ステップ5 3D空間位置推定
#       Total                … 合計
# ============================================================================

# コマンドライン引数を扱うための標準ライブラリ
import argparse
# ファイルやフォルダを操作するための標準ライブラリ
import os
# CSVファイルを読み書きするための標準ライブラリ
import csv

# グラフを描くためのライブラリ
import matplotlib
# 画面を持たない環境（サーバーやJetsonのSSH接続など）でも動くように、
# 画面に表示せずファイルに保存する方式を指定する。
# この指定は pyplot を読み込む前に行う必要がある
matplotlib.use("Agg")
# グラフを描くための道具を読み込む
import matplotlib.pyplot as plt
# 数値計算ライブラリ
import numpy as np

# 自作のモジュール（パスの管理）を読み込む
import common_paths


# CSVに入っているステップ名と、グラフに表示する英語の名前の対応表
STEP_DISPLAY_NAME = {
    "step1_acquisition_ms": "1. Image acquisition",
    "step2_depth_alignment_ms": "2. Depth alignment",
    "step3_preprocessing_ms": "3. Preprocessing",
    "step4a_inference_ms": "4a. Inference",
    "step4b_nms_ms": "4b. NMS",
    "step5_localization_ms": "5. 3D localization",
    "total_ms": "Total",
}

# ステップを表示する順番（CSVの中の名前）
STEP_ORDER = [
    "step1_acquisition_ms",
    "step2_depth_alignment_ms",
    "step3_preprocessing_ms",
    "step4a_inference_ms",
    "step4b_nms_ms",
    "step5_localization_ms",
]

# グラフに使う色。data-viz の標準パレットの1番目から順に使う。
# 色覚特性のある人でも見分けられることを検証済みの並びなので、
# 順番を変えたり別の色を混ぜたりしないこと
MACHINE_COLOR_LIST = [
    "#2a78d6",  # 青
    "#eb6834",  # オレンジ
    "#1baf7a",  # 青緑
    "#eda100",  # 黄
    "#e87ba4",  # 赤紫
    "#008300",  # 緑
    "#4a3aa7",  # 紫
    "#e34948",  # 赤
]

def choose_readable_text_color(background_hex_color):
    """
    帯の色の上に文字を書くとき、読みやすい文字色（白か黒）を選んで返す。

    黄色のような明るい色の上に白い文字を書くと読めなくなるため、
    帯の明るさを計算して、明るい帯には黒い文字、暗い帯には白い文字を使います。
    """
    # "#eda100" のような文字列から、赤・緑・青の値（0〜255）を取り出す
    red_value = int(background_hex_color[1:3], 16)
    green_value = int(background_hex_color[3:5], 16)
    blue_value = int(background_hex_color[5:7], 16)
    # 人の目の感じ方に合わせた重みで、明るさを計算する（0が真っ黒、1が真っ白）
    # 緑が一番明るく見え、青が一番暗く見えるので、この重みになっている
    brightness = (0.299 * red_value + 0.587 * green_value + 0.114 * blue_value) / 255.0
    # 明るい帯なら黒い文字、暗い帯なら白い文字を返す
    if brightness > 0.6:
        return "#0b0b0b"
    return "#ffffff"


# グラフの背景の色
SURFACE_COLOR = "#fcfcfb"
# 濃い文字の色
TEXT_PRIMARY_COLOR = "#0b0b0b"
# 薄い文字の色
TEXT_SECONDARY_COLOR = "#52514e"
# 目盛り線の色
GRID_COLOR = "#e3e3e0"


def read_summary_csv(summary_csv_path):
    """
    統計値のまとめCSVを1つ読み込み、中身を辞書にして返す。

    返り値は次の形の辞書です。
        {
          "machine_name": マシン名,
          "machine_information": マシン情報の辞書,
          "statistics_by_step": { ステップ名: {mean, median, std, ...}, ... }
        }
    """
    # ステップごとの統計値を入れるための辞書を用意する
    statistics_by_step = {}
    # マシン情報を入れるための辞書を用意する
    machine_information = {}

    # CSVファイルを開く
    with open(summary_csv_path, "r", encoding="utf-8", newline="") as opened_file:
        # 1行を辞書として読み込むための道具を用意する
        csv_reader = csv.DictReader(opened_file)
        # 1行ずつ順番に読む
        for one_row in csv_reader:
            # その行が表しているステップの名前を取り出す
            step_name = one_row["step_name"]
            # そのステップの統計値を辞書にまとめる
            statistics_by_step[step_name] = {
                "mean": float(one_row["mean_ms"]),
                "median": float(one_row["median_ms"]),
                "std": float(one_row["std_ms"]),
                "min": float(one_row["min_ms"]),
                "max": float(one_row["max_ms"]),
                "p95": float(one_row["p95_ms"]),
                "share": float(one_row["share_of_total_percent"]),
                "count": int(one_row["count"]),
            }
            # マシン情報はどの行にも同じ値が入っているので、1回だけ記録する
            if len(machine_information) == 0:
                # 統計値の列以外を、マシン情報として取り出す
                statistics_column_name_list = [
                    "step_name", "count", "mean_ms", "median_ms", "std_ms",
                    "min_ms", "max_ms", "p95_ms", "share_of_total_percent",
                ]
                # 行の中の項目を1つずつ見ていく
                for one_column_name in one_row:
                    # 統計値の列でなければ、マシン情報として記録する
                    if one_column_name not in statistics_column_name_list:
                        machine_information[one_column_name] = one_row[one_column_name]

    # ステップが1つも読み込めなければ、ファイルの形式がおかしいのでエラーにする
    if len(statistics_by_step) == 0:
        raise ValueError("統計値を読み込めませんでした: " + summary_csv_path)

    # 読み込んだ内容をまとめて返す
    return {
        "machine_name": machine_information.get("machine_name", "unknown"),
        "machine_information": machine_information,
        "statistics_by_step": statistics_by_step,
    }


def write_comparison_csv(output_csv_path, machine_result_list):
    """比較表をCSVファイルとして書き出す。"""
    # 列の名前を並べたリストを作る。最初は「ステップ名」
    column_name_list = ["step"]
    # マシンごとに「平均・中央値・標準偏差」の3列を足していく
    for one_machine_result in machine_result_list:
        machine_name = one_machine_result["machine_name"]
        column_name_list.append(machine_name + "_mean_ms")
        column_name_list.append(machine_name + "_median_ms")
        column_name_list.append(machine_name + "_std_ms")
    # 最後に「1台目に対して何倍の時間がかかったか」の列を、2台目以降について足す
    for machine_index in range(1, len(machine_result_list)):
        column_name_list.append(
            machine_result_list[machine_index]["machine_name"] + "_ratio_vs_"
            + machine_result_list[0]["machine_name"]
        )

    # CSVファイルを書き込みモードで開く
    with open(output_csv_path, "w", encoding="utf-8", newline="") as opened_file:
        # 書き出すための道具を用意する
        csv_writer = csv.writer(opened_file)
        # 1行目に列の名前を書き出す
        csv_writer.writerow(column_name_list)
        # ステップを1つずつ、合計も含めて書き出す
        for one_step_name in STEP_ORDER + ["total_ms"]:
            # 1行分の内容を入れるリストを、ステップの表示名で始める
            one_output_row = [STEP_DISPLAY_NAME[one_step_name]]
            # マシンごとに平均・中央値・標準偏差を足していく
            for one_machine_result in machine_result_list:
                # そのマシンのそのステップの統計値を取り出す
                step_statistics = one_machine_result["statistics_by_step"][one_step_name]
                one_output_row.append(round(step_statistics["mean"], 3))
                one_output_row.append(round(step_statistics["median"], 3))
                one_output_row.append(round(step_statistics["std"], 3))
            # 1台目の平均時間を取り出す（比率の計算に使う）
            first_machine_mean = machine_result_list[0]["statistics_by_step"][one_step_name]["mean"]
            # 2台目以降について、1台目の何倍かを計算して足す
            for machine_index in range(1, len(machine_result_list)):
                # そのマシンの平均時間を取り出す
                other_machine_mean = (
                    machine_result_list[machine_index]["statistics_by_step"][one_step_name]["mean"]
                )
                # 1台目が0だと割り算できないので、その場合は0を入れる
                if first_machine_mean > 0.0:
                    one_output_row.append(round(other_machine_mean / first_machine_mean, 3))
                else:
                    one_output_row.append(0.0)
            # 1行分を書き出す
            csv_writer.writerow(one_output_row)


def write_comparison_markdown(output_markdown_path, machine_result_list):
    """比較表を、文章に貼り付けやすい形式（Markdown）で書き出す。"""
    # 書き出す行を入れるためのリストを用意する
    output_line_list = []
    # 見出しを追加する
    output_line_list.append("# 集中型計算方式のステップ別処理時間の比較")
    output_line_list.append("")

    # マシンの情報を表にする
    output_line_list.append("## 計測に使ったマシン")
    output_line_list.append("")
    output_line_list.append("| 項目 | " + " | ".join(
        one_machine_result["machine_name"] for one_machine_result in machine_result_list) + " |")
    output_line_list.append("|---|" + "---|" * len(machine_result_list))
    # 表に載せるマシン情報の項目を選ぶ
    information_key_list = [
        "cpu_name", "gpu_name", "ram_total_gb", "device_type",
        "torch_version", "cuda_version", "os", "measured_at",
    ]
    # 項目を1つずつ行にする
    for one_information_key in information_key_list:
        # 1行分の内容を項目名で始める
        one_line = "| " + one_information_key + " |"
        # マシンごとの値を足していく
        for one_machine_result in machine_result_list:
            one_line = one_line + " " + str(
                one_machine_result["machine_information"].get(one_information_key, "-")) + " |"
        # できた行を追加する
        output_line_list.append(one_line)
    output_line_list.append("")

    # ステップ別の時間の表を作る
    output_line_list.append("## ステップ別の処理時間（平均 ± 標準偏差、単位: ミリ秒）")
    output_line_list.append("")
    # 見出しの行を作る
    header_line = "| ステップ |"
    for one_machine_result in machine_result_list:
        header_line = header_line + " " + one_machine_result["machine_name"] + " |"
    # 2台以上あれば、比率の列も足す
    for machine_index in range(1, len(machine_result_list)):
        header_line = header_line + " " + machine_result_list[machine_index]["machine_name"] \
            + " / " + machine_result_list[0]["machine_name"] + " |"
    output_line_list.append(header_line)
    # 見出しの下の区切り線を作る
    separator_count = 1 + len(machine_result_list) + (len(machine_result_list) - 1)
    output_line_list.append("|" + "---|" * separator_count)

    # ステップを1つずつ行にする
    for one_step_name in STEP_ORDER + ["total_ms"]:
        # 1行分の内容をステップの表示名で始める
        one_line = "| " + STEP_DISPLAY_NAME[one_step_name] + " |"
        # マシンごとの平均と標準偏差を足していく
        for one_machine_result in machine_result_list:
            step_statistics = one_machine_result["statistics_by_step"][one_step_name]
            one_line = one_line + " %.2f ± %.2f |" % (step_statistics["mean"], step_statistics["std"])
        # 1台目の平均時間を取り出す
        first_machine_mean = machine_result_list[0]["statistics_by_step"][one_step_name]["mean"]
        # 2台目以降について比率を足していく
        for machine_index in range(1, len(machine_result_list)):
            other_machine_mean = (
                machine_result_list[machine_index]["statistics_by_step"][one_step_name]["mean"]
            )
            if first_machine_mean > 0.0:
                one_line = one_line + " %.2f 倍 |" % (other_machine_mean / first_machine_mean)
            else:
                one_line = one_line + " - |"
        # できた行を追加する
        output_line_list.append(one_line)
    output_line_list.append("")

    # 1秒あたりの処理枚数(FPS)も載せる
    output_line_list.append("## 1秒あたりの処理枚数（FPS）")
    output_line_list.append("")
    for one_machine_result in machine_result_list:
        # 合計時間の平均を取り出す
        total_mean_ms = one_machine_result["statistics_by_step"]["total_ms"]["mean"]
        # 0で割らないように確認してから計算する
        if total_mean_ms > 0.0:
            frames_per_second = 1000.0 / total_mean_ms
        else:
            frames_per_second = 0.0
        # 1行分を追加する
        output_line_list.append("- %s : %.2f 枚/秒（1枚あたり %.1f ミリ秒）"
                                % (one_machine_result["machine_name"], frames_per_second, total_mean_ms))
    output_line_list.append("")

    # ファイルに書き出す
    with open(output_markdown_path, "w", encoding="utf-8") as opened_file:
        for one_line in output_line_list:
            opened_file.write(one_line + "\n")


def apply_chart_style(axes):
    """グラフの見た目を整える（線を細く、目盛りを控えめにする）。"""
    # グラフの背景の色を設定する
    axes.set_facecolor(SURFACE_COLOR)
    # 上・右・左の枠線を消す（データを邪魔しないようにするため）
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_visible(False)
    # 下の枠線だけ、細く薄い色で残す
    axes.spines["bottom"].set_color(GRID_COLOR)
    axes.spines["bottom"].set_linewidth(0.8)
    # 横方向の目盛り線だけを、薄く細く表示する
    axes.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    # 目盛り線をデータの後ろに描く（データが線に隠れないようにするため）
    axes.set_axisbelow(True)
    # 目盛りの文字の色と大きさを設定する
    axes.tick_params(colors=TEXT_SECONDARY_COLOR, labelsize=9, length=0)


def draw_step_comparison_chart(output_image_path, machine_result_list):
    """ステップごとの処理時間を、マシン別に並べた横棒グラフを描く。"""
    # 表示するステップの数を数える
    number_of_steps = len(STEP_ORDER)
    # 比較するマシンの数を数える
    number_of_machines = len(machine_result_list)

    # 棒を描く縦方向の位置を、ステップごとに用意する
    step_positions = np.arange(number_of_steps)
    # 1本の棒の太さを決める（マシンが増えるほど細くする）
    bar_thickness = 0.8 / number_of_machines

    # グラフの土台を作る。マシンが多いほど縦を長くする
    figure, axes = plt.subplots(
        figsize=(9.0, 1.0 + 0.62 * number_of_steps * number_of_machines / 2.0)
    )
    # グラフ全体の背景の色を設定する
    figure.patch.set_facecolor(SURFACE_COLOR)
    # 見た目を整える
    apply_chart_style(axes)

    # 棒の右端に数値を書くとき、どれだけ右にずらすかを決めるために、
    # 先に「一番長い棒の長さ」を求めておく。
    # （描きながら求めると、最初のマシンのラベルだけ位置がずれてしまうため）
    largest_value = 0.0
    # マシンを1台ずつ見ていく
    for one_machine_result in machine_result_list:
        # ステップを1つずつ見ていく
        for one_step_name in STEP_ORDER:
            # そのステップの統計値を取り出す
            step_statistics = one_machine_result["statistics_by_step"][one_step_name]
            # 棒の長さと標準偏差の線の長さを足した値を求める
            bar_end_position = step_statistics["mean"] + step_statistics["std"]
            # これまでで一番長ければ記録する
            if bar_end_position > largest_value:
                largest_value = bar_end_position
    # すべての値が0だと後の計算で困るので、その場合は1にしておく
    if largest_value <= 0.0:
        largest_value = 1.0

    # マシンを1台ずつ順番に描く
    for machine_index in range(number_of_machines):
        # そのマシンの結果を取り出す
        one_machine_result = machine_result_list[machine_index]
        # そのマシンの各ステップの平均時間を集めるリストを用意する
        mean_value_list = []
        # そのマシンの各ステップの標準偏差を集めるリストを用意する
        standard_deviation_list = []
        # ステップを1つずつ見ていく
        for one_step_name in STEP_ORDER:
            # そのステップの統計値を取り出す
            step_statistics = one_machine_result["statistics_by_step"][one_step_name]
            # 平均をリストに追加する
            mean_value_list.append(step_statistics["mean"])
            # 標準偏差をリストに追加する
            standard_deviation_list.append(step_statistics["std"])

        # この棒を描く縦位置を計算する（マシンごとに少しずつずらす）
        offset_from_center = (machine_index - (number_of_machines - 1) / 2.0) * bar_thickness
        bar_positions = step_positions + offset_from_center

        # 合計時間の平均を取り出して、凡例に添える
        total_mean_ms = one_machine_result["statistics_by_step"]["total_ms"]["mean"]
        legend_label = "%s  (total %.0f ms)" % (one_machine_result["machine_name"], total_mean_ms)

        # 横棒を描く
        axes.barh(
            bar_positions,
            mean_value_list,
            height=bar_thickness * 0.86,
            color=MACHINE_COLOR_LIST[machine_index % len(MACHINE_COLOR_LIST)],
            label=legend_label,
            # 棒どうしの間に隙間を作るため、背景色の細い縁を付ける
            edgecolor=SURFACE_COLOR,
            linewidth=1.0,
        )
        # 標準偏差を細い横線で示す（ばらつきの大きさが分かるようにするため）
        axes.errorbar(
            mean_value_list,
            bar_positions,
            xerr=standard_deviation_list,
            fmt="none",
            ecolor=TEXT_SECONDARY_COLOR,
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
        )
        # 棒の右端に数値を書く（色だけに頼らず値が読めるようにするため）
        for step_index in range(number_of_steps):
            axes.text(
                mean_value_list[step_index] + standard_deviation_list[step_index] + largest_value * 0.015,
                bar_positions[step_index],
                "%.1f" % mean_value_list[step_index],
                va="center",
                ha="left",
                fontsize=8,
                color=TEXT_SECONDARY_COLOR,
            )

    # 縦軸にステップの名前を表示する
    axes.set_yticks(step_positions)
    axes.set_yticklabels([STEP_DISPLAY_NAME[one_step_name] for one_step_name in STEP_ORDER])
    # 上のステップが1番になるように、縦軸の向きを逆にする
    axes.invert_yaxis()
    # 横軸の説明を書く
    axes.set_xlabel("Mean processing time per image (ms)",
                    color=TEXT_SECONDARY_COLOR, fontsize=9)
    # 数値を書くぶんの余白を右に作る
    axes.set_xlim(0, largest_value * 1.18)
    # グラフの題名を書く
    axes.set_title("Centralized computing scheme: time per step",
                   color=TEXT_PRIMARY_COLOR, fontsize=12, pad=14, loc="left")
    # 凡例をグラフの下に表示する（棒や数値と重ならないようにするため）
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
                ncol=min(number_of_machines, 3), frameon=False, fontsize=9,
                labelcolor=TEXT_SECONDARY_COLOR)
    # 余白を自動で調整する
    figure.tight_layout()
    # 画像ファイルとして保存する
    figure.savefig(output_image_path, dpi=150, facecolor=SURFACE_COLOR,
                   bbox_inches="tight")
    # 描き終わったグラフを閉じて、メモリを解放する
    plt.close(figure)


def draw_composition_chart(output_image_path, machine_result_list):
    """処理時間の内訳（どのステップが全体の何%を占めるか）を積み上げ棒グラフで描く。"""
    # 比較するマシンの数を数える
    number_of_machines = len(machine_result_list)
    # 棒を描く縦方向の位置を用意する
    machine_positions = np.arange(number_of_machines)

    # グラフの土台を作る
    figure, axes = plt.subplots(figsize=(9.0, 1.6 + 0.8 * number_of_machines))
    # グラフ全体の背景の色を設定する
    figure.patch.set_facecolor(SURFACE_COLOR)
    # 見た目を整える
    apply_chart_style(axes)

    # ステップごとに色を割り当てる（マシンではなくステップで色分けする）
    step_color_list = MACHINE_COLOR_LIST[: len(STEP_ORDER)]

    # 積み上げの開始位置を、マシンごとに0で用意する
    stack_left_position = np.zeros(number_of_machines)

    # ステップを1つずつ、積み上げながら描く
    for step_index in range(len(STEP_ORDER)):
        # このステップの名前を取り出す
        one_step_name = STEP_ORDER[step_index]
        # マシンごとの割合を集めるリストを用意する
        share_value_list = []
        # マシンを1台ずつ見ていく
        for one_machine_result in machine_result_list:
            # このステップが全体に占める割合を取り出す
            share_value_list.append(
                one_machine_result["statistics_by_step"][one_step_name]["share"]
            )
        # numpy の配列に変換する
        share_value_array = np.array(share_value_list)

        # 積み上げ棒の1段分を描く
        axes.barh(
            machine_positions,
            share_value_array,
            left=stack_left_position,
            height=0.5,
            color=step_color_list[step_index],
            label=STEP_DISPLAY_NAME[one_step_name],
            # 段どうしの境目に隙間を作るため、背景色の細い縁を付ける
            edgecolor=SURFACE_COLOR,
            linewidth=1.5,
        )
        # 幅が十分にある段にだけ、割合の数値を書き込む
        for machine_index in range(number_of_machines):
            # 段が狭すぎると文字が重なるので、6%より広い段にだけ書く
            if share_value_array[machine_index] > 6.0:
                axes.text(
                    stack_left_position[machine_index] + share_value_array[machine_index] / 2.0,
                    machine_positions[machine_index],
                    "%.0f%%" % share_value_array[machine_index],
                    va="center",
                    ha="center",
                    fontsize=8,
                    # 帯の明るさに合わせて、読みやすい文字色を選ぶ
                    color=choose_readable_text_color(step_color_list[step_index]),
                )
        # 次の段の開始位置を、いま描いた分だけ右にずらす
        stack_left_position = stack_left_position + share_value_array

    # 縦軸にマシン名を表示する
    axes.set_yticks(machine_positions)
    axes.set_yticklabels(
        [one_machine_result["machine_name"] for one_machine_result in machine_result_list]
    )
    # 上のマシンが1台目になるように、縦軸の向きを逆にする
    axes.invert_yaxis()
    # 横軸は0%から100%までにする
    axes.set_xlim(0, 100)
    # 横軸の説明を書く
    axes.set_xlabel("Share of total processing time (%)",
                    color=TEXT_SECONDARY_COLOR, fontsize=9)
    # グラフの題名を書く
    axes.set_title("Where the time goes", color=TEXT_PRIMARY_COLOR,
                   fontsize=12, pad=14, loc="left")
    # 凡例をグラフの下に横並びで表示する
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3,
                frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY_COLOR)
    # 余白を自動で調整する
    figure.tight_layout()
    # 画像ファイルとして保存する
    figure.savefig(output_image_path, dpi=150, facecolor=SURFACE_COLOR,
                   bbox_inches="tight")
    # 描き終わったグラフを閉じて、メモリを解放する
    plt.close(figure)


def main():
    """コマンドから実行されたときに動く、このスクリプトの本体。"""
    # コマンドライン引数の設定を作る
    argument_parser = argparse.ArgumentParser(
        description="2台以上のマシンの計測結果を比較し、表とグラフを作ります。"
    )
    # 読み込むCSVファイルを指定する引数を追加する
    argument_parser.add_argument(
        "--summary-csv",
        nargs="+",
        required=True,
        help="比較する timings_summary_*.csv のパス。スペース区切りで2つ以上指定します",
    )
    # 出力先を指定する引数を追加する
    argument_parser.add_argument(
        "--output-dir",
        default=None,
        help="結果の保存先。省略すると outputs/comparison になります",
    )
    # 実際に引数を読み取る
    parsed_arguments = argument_parser.parse_args()

    # 出力先を決める
    if parsed_arguments.output_dir is None:
        output_directory = os.path.join(common_paths.get_outputs_directory(), "comparison")
    else:
        output_directory = parsed_arguments.output_dir
    # 出力先のフォルダを作る
    os.makedirs(output_directory, exist_ok=True)

    # 処理の開始を表示する
    print("=" * 70)
    print("計測結果の比較")
    print("=" * 70)

    # 指定された各CSVを読み込む
    machine_result_list = []
    # ファイルを1つずつ順番に読み込む
    for one_csv_path in parsed_arguments.summary_csv:
        # ファイルが無ければエラーを出して止める
        if not os.path.exists(one_csv_path):
            raise FileNotFoundError("CSVファイルが見つかりません: " + one_csv_path)
        # 読み込む
        one_machine_result = read_summary_csv(one_csv_path)
        # リストに追加する
        machine_result_list.append(one_machine_result)
        # 読み込んだことを表示する
        print("  読み込み: " + one_machine_result["machine_name"] + "  (" + one_csv_path + ")")
    print("")

    # 比較するマシンが1台だけの場合は、注意を表示する（グラフ自体は作れる）
    if len(machine_result_list) < 2:
        print("[注意] CSVが1つだけ指定されています。比較するには2つ以上指定してください。")
        print("")

    # --- 比較表を画面に表示する ---
    print("=" * 78)
    print("ステップ別の処理時間（平均、単位: ミリ秒）")
    print("=" * 78)
    # 見出しの行を作る
    header_line = "  %-22s" % "ステップ"
    for one_machine_result in machine_result_list:
        header_line = header_line + "%16s" % one_machine_result["machine_name"][:15]
    # 2台以上あれば比率の列も足す
    if len(machine_result_list) >= 2:
        header_line = header_line + "%10s" % "比率"
    print(header_line)
    print("  " + "-" * 74)
    # ステップを1つずつ表示する
    for one_step_name in STEP_ORDER + ["total_ms"]:
        # 合計の行の前に区切り線を入れる
        if one_step_name == "total_ms":
            print("  " + "-" * 74)
        # 1行分をステップの表示名で始める
        one_line = "  %-22s" % STEP_DISPLAY_NAME[one_step_name]
        # マシンごとの平均を足していく
        for one_machine_result in machine_result_list:
            one_line = one_line + "%16.2f" % (
                one_machine_result["statistics_by_step"][one_step_name]["mean"]
            )
        # 2台以上あれば、1台目に対する2台目の比率を足す
        if len(machine_result_list) >= 2:
            first_mean = machine_result_list[0]["statistics_by_step"][one_step_name]["mean"]
            second_mean = machine_result_list[1]["statistics_by_step"][one_step_name]["mean"]
            if first_mean > 0.0:
                one_line = one_line + "%9.2f倍" % (second_mean / first_mean)
            else:
                one_line = one_line + "%10s" % "-"
        # 1行分を表示する
        print(one_line)
    print("=" * 78)
    print("")

    # --- ファイルに書き出す ---
    # 比較表のCSVのパスを作る
    comparison_csv_path = os.path.join(output_directory, "comparison_table.csv")
    # 比較表のCSVを書き出す
    write_comparison_csv(comparison_csv_path, machine_result_list)

    # 比較表のMarkdownのパスを作る
    comparison_markdown_path = os.path.join(output_directory, "comparison_table.md")
    # 比較表のMarkdownを書き出す
    write_comparison_markdown(comparison_markdown_path, machine_result_list)

    # ステップ別グラフのパスを作る
    step_chart_path = os.path.join(output_directory, "comparison_steps.png")
    # ステップ別グラフを描く
    draw_step_comparison_chart(step_chart_path, machine_result_list)

    # 内訳グラフのパスを作る
    composition_chart_path = os.path.join(output_directory, "comparison_composition.png")
    # 内訳グラフを描く
    draw_composition_chart(composition_chart_path, machine_result_list)

    # 保存したファイルを表示する
    print("結果を保存しました:")
    print("  比較表(CSV)         : " + comparison_csv_path)
    print("  比較表(Markdown)    : " + comparison_markdown_path)
    print("  ステップ別グラフ     : " + step_chart_path)
    print("  処理時間の内訳グラフ : " + composition_chart_path)


# このファイルが直接実行されたときだけ main() を呼ぶ
if __name__ == "__main__":
    main()
