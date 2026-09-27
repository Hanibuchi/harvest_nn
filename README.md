# 農業用収穫ロボット 集中型計算方式の処理時間計測

Xie et al. (2024) *"Boosting Cost-Efficiency in Robotics: A Distributed Computing
Approach for Harvesting Robots"* (Journal of Field Robotics) の Figure 6 上段にある
**集中型計算方式（Centralized Computing Scheme）** を再現し、
各処理ステップの実行時間を実測するためのプログラム一式です。

データセットは [KFuji RGB-DS database](https://doi.org/10.1016/j.dib.2019.104289)
（Fuji りんごの RGB-D 画像 967枚）を使います。

## 計測する5つのステップ

| ステップ | 内容 | 実装ファイル |
|---|---|---|
| 1. 画像取得 | カラー画像(1920×1080)と深度データ(点群)の読み込み | `src/kinect_io.py` |
| 2. 深度アライメント | 点群をカラー画像の座標系に投影して深度画像を作る | `src/depth_alignment.py` |
| 3. 推論前処理 | タイル分割・リサイズ・色変換・正規化・テンソル化 | `src/measure_centralized.py` |
| 4a. 画像推論 | YOLOv8 の順伝播（NMS を含まない） | `src/measure_centralized.py` |
| 4b. 後処理(NMS) | 重なった検出枠をまとめる処理 | `src/measure_centralized.py` |
| 5. 3D空間位置推定 | 検出枠内の深度の中央値から3次元座標を計算 | `src/localization_3d.py` |

ステップ4は **推論そのもの (4a) と後処理 (4b) を分けて計測** できます。

---

## 1. 準備

### 1-1. データセットを用意する

**自動で取得できます。** 環境を増やすたびに手でコピーする必要はありません。

```bash
python src/fetch_dataset.py
```

[Zenodo](https://zenodo.org/records/3715991) から必要なファイルだけを取得して、
`KFuji_RGB-DS_dataset/` を作ります。追加のライブラリは要りません（標準ライブラリだけで動きます）。

- 取得量は **約585MB**（配布zip全体 2.9GB の19%）です。
  残りの大部分は `preprocessed data/images/*_DS.mat`（位置合わせ済みの深度データ）で、
  このプロジェクトでは使わないため取得しません（理由は後述の 5-3）。
- 取得したファイルは1つずつ CRC で照合するので、壊れたまま進むことはありません。
- 途中で止まっても、もう一度同じコマンドを実行すれば**残りだけ**を取得します。
- Zenodo 側が混んでいると 1MB/秒を下回ることがあり、その場合は20〜40分ほどかかります。
  待てないときは先に `--parts raw,annotations`（約500MB）だけ取得し、
  学習をするときに `--parts images` を追加で実行してください。

```bash
# 何をどれだけ取得するか、通信せずに確認する
python src/fetch_dataset.py --dry-run

# 新しいマシンで、まず少しだけ取得して動作を確かめる
python src/fetch_dataset.py --parts images --limit-files 20

# 計測だけを行うマシン（Jetson など）で、必要最小限（約500MB）にする
python src/fetch_dataset.py --parts raw,annotations

# 配布物と全く同じ内容をそろえる（未使用の _DS.mat も含む。2.9GB）
python src/fetch_dataset.py --parts all
```

できあがる構成:

```
harvest_nn/
└── KFuji_RGB-DS_dataset/
    ├── row data/            ← 生データ（位置合わせ前）。計測にはこちらを使う
    └── preprocessed data/   ← 切り出し済みデータ。学習にはこちらを使う
```

**すでにデータセットを持っている場合**は、上の構成になるように置くだけで構いません。

**別の場所に置きたい場合**（外付けSSDなど）は、次のどちらかで指定します。

```bash
# 方法1: 環境変数で指定する（一度書いておけば、すべてのスクリプトが従います）
export KFUJI_DATASET_ROOT=/mnt/ssd/KFuji_RGB-DS_dataset
python src/fetch_dataset.py

# 方法2: コマンドごとに指定する
python src/fetch_dataset.py --dataset-root /mnt/ssd/KFuji_RGB-DS_dataset
python src/prepare_dataset.py --dataset-root /mnt/ssd/KFuji_RGB-DS_dataset
```

### 1-2. ライブラリをインストールする

**Python 3.11 または 3.12 が必要です。**
（PyTorch と Ultralytics は、まだ Python 3.13 / 3.14 に対応していません）

#### Mac / Linux ワークステーションの場合

```bash
# 仮想環境を作る（プロジェクトごとにライブラリを分けるための仕組み）
python3.12 -m venv .venv

# 仮想環境を有効にする（これ以降 python と打つとこの環境のものが使われる）
source .venv/bin/activate

# ライブラリをまとめてインストールする
pip install --upgrade pip
pip install -r requirements.txt
```

一度ターミナルを閉じたあと、また作業するときは `source .venv/bin/activate` だけを実行します。

#### NVIDIA Jetson の場合

**Jetson では torch / torchvision を通常の pip で入れてはいけません。**
pip で配布されている標準版は x86 パソコン用のため、Jetson では動きません。
NVIDIA が配布している Jetson 専用の wheel を先に入れてください。

```bash
# 1. 仮想環境を作る（--system-site-packages を付けるのが重要。
#    Jetson にあらかじめ入っている OpenCV などを使えるようにするため）
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# 2. NVIDIA 提供の torch / torchvision を入れる
#    お使いの JetPack のバージョンに合う wheel を
#    https://developer.nvidia.com/embedded/downloads から入手してください
pip install torch-*.whl torchvision-*.whl

# 3. torch と torchvision の行を除いた残りを入れる
pip install numpy scipy ultralytics pandas matplotlib PyYAML psutil

# 4. 正しく入ったか確認する（True と表示されれば GPU が使えます）
python -c "import torch; print(torch.cuda.is_available())"
```

Jetson では `--workers 2` を付けて学習すると、メモリ不足になりにくいです。

---

## 2. 実行の順番

```
⓪ データ取得 → ① データ準備 → ②（任意）アライメント確認 → ③ 学習 → ④ 計測 → ⑤ 2台の比較
```

### ⓪ データ取得

```bash
python src/fetch_dataset.py
```

Zenodo からデータセットを取得します（詳しくは 1-1）。すでに手元にある場合は飛ばせます。

### ① データ準備

```bash
python src/prepare_dataset.py
```

やること:
- アノテーション(csv)を YOLO 形式(txt)に変換する
- train / val / test に分割する（リークが起きないように分割します。詳しくは後述）
- 計測に使う点群を `.mat` から読み込みの速い `.npy` に変換する

よく使う引数:

```bash
# 分割の比率を変える
python src/prepare_dataset.py --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2

# 乱数シードを変えて、別の分割を作る
python src/prepare_dataset.py --seed 123

# 境目のシーンを捨てずに、全シーンを使う（データは増えるがリークの危険も増える）
python src/prepare_dataset.py --guard-scenes 0

# 全109シーンの点群を .npy に変換する（テストセット以外でも計測したい場合）
python src/prepare_dataset.py --convert-point-clouds all
```

### ②（任意）深度アライメントの確認

```bash
python src/verify_alignment.py
```

点群がカラー画像の正しい位置に重なっているかを確認します。
`outputs/alignment_check/` に確認用の画像が保存されます。

```bash
# カメラの平行移動量を自動で調整したい場合
python src/verify_alignment.py --search --num-scenes 5
```

### ③ 学習（ファインチューニング）

```bash
python src/train_yolo.py --model-size n --epochs 100
```

- `--model-size` で `n`（小さい・速い）/ `s`（中）/ `m`（大きい・精度が高い）を切り替えます
- 初回実行時、学習済みモデル `yolov8n.pt` を自動でダウンロードします（インターネットが必要）
- 学習が終わると、テストセットでの精度（mAP など）が表示されます
- 学習済みの重みは `outputs/weights/yolov8n_apple_best.pt` にコピーされます

```bash
# GPUのメモリが足りないとき
python src/train_yolo.py --model-size s --batch 8 --workers 2
```

### ④ 計測（メイン）

```bash
python src/measure_centralized.py \
    --weights outputs/weights/yolov8n_apple_best.pt \
    --machine-name workstation
```

`--machine-name` には、そのマシンが分かる名前を付けてください（結果ファイルの名前と、
CSV の中の列に使われます）。2台目では `--machine-name jetson_orin` のようにします。

よく使う引数:

```bash
# 繰り返し回数を増やして、より安定した値を得る
python src/measure_centralized.py --weights ... --machine-name jetson_orin \
    --repeats 20 --warmup 10

# 半精度(float16)で推論する（NVIDIA GPU のみ。速くなることがあります）
python src/measure_centralized.py --weights ... --machine-name workstation --half

# 動作確認のため、画像3枚だけで試す
python src/measure_centralized.py --weights ... --machine-name test --limit-images 3
```

### ⑤ 2台の結果を比較する

2台それぞれで ④ を実行したあと、出力された CSV を1か所に集めて実行します。

```bash
python src/compare_machines.py \
    --summary-csv outputs/measurements/timings_summary_workstation.csv \
                  outputs/measurements/timings_summary_jetson_orin.csv
```

`outputs/comparison/` に比較表とグラフが保存されます。

---

## 3. 出力ファイルの見方

### `outputs/measurements/timings_raw_<マシン名>.csv`

**1回の計測につき1行**の、生のデータです。統計を自分で取り直したいときに使います。

| 列名 | 意味 |
|---|---|
| `machine_name` | マシン名（2台の結果を1つにまとめるときの目印） |
| `device_type` | 計算に使った装置（`cuda` / `mps` / `cpu`） |
| `scene_name` | どの画像を処理したか |
| `repeat_index` | その画像の何回目の繰り返しか（0から始まる） |
| `step1_acquisition_ms` | ステップ1の時間（ミリ秒） |
| `step2_depth_alignment_ms` | ステップ2の時間 |
| `step3_preprocessing_ms` | ステップ3の時間 |
| `step4a_inference_ms` | ステップ4a（推論のみ）の時間 |
| `step4b_nms_ms` | ステップ4b（NMS）の時間 |
| `step5_localization_ms` | ステップ5の時間 |
| `total_ms` | 上の6つの合計 |
| `num_tiles` | 1画像を何枚のタイルに分けたか |
| `num_points` | 点群の点の数 |
| `num_detections` | 検出したりんごの数 |
| `num_detections_with_depth` | そのうち深度が取れた数 |

### `outputs/measurements/timings_summary_<マシン名>.csv`

**ステップごとに1行**の、統計値のまとめです。**マシン情報も各行に入っています**ので、
このファイル1つだけで「どのマシンの結果か」が分かります。

| 列名 | 意味 |
|---|---|
| `machine_name` 〜 `numpy_version` | マシン情報（CPU名・GPU名・メモリ量・PyTorch/CUDAバージョンなど） |
| `step_name` | どのステップか |
| `count` | 計測回数 |
| `mean_ms` / `median_ms` | 平均 / 中央値（ミリ秒） |
| `std_ms` | 標準偏差（値のばらつき。小さいほど安定している） |
| `min_ms` / `max_ms` / `p95_ms` | 最小 / 最大 / 95パーセンタイル |
| `share_of_total_percent` | そのステップが全体の何%を占めるか |

> **中央値と平均が大きく違うときは**、どこかの回だけ極端に遅くなっています。
> `--warmup` や `--warmup-per-image` の回数を増やすと改善することがあります。

### `outputs/measurements/detections_<マシン名>.csv`

検出したりんご1個ごとの3次元座標です。パイプラインが妥当な結果を出しているかの確認に使います。

| 列名 | 意味 |
|---|---|
| `box_x1` 〜 `box_y2` | 検出枠の位置（カラー画像のピクセル座標） |
| `median_depth_m` | 枠の中央部分の深度の中央値（メートル）。0 なら深度が取れなかった |
| `position_x_optical_m` 〜 `position_z_optical_m` | カメラ光学座標系での3次元座標（x=右, y=下, z=前） |
| `position_x_sensor_m` 〜 `position_z_sensor_m` | センサ座標系での3次元座標（X=右, Y=前, Z=上） |

### `outputs/comparison/`

| ファイル | 内容 |
|---|---|
| `comparison_table.csv` | 比較表（表計算ソフトで開けます） |
| `comparison_table.md` | 比較表（論文やレポートに貼り付けやすい形式） |
| `comparison_steps.png` | ステップごとの処理時間の棒グラフ |
| `comparison_composition.png` | 処理時間の内訳（何が全体の何%か）のグラフ |

### `outputs/splits/`

| ファイル | 内容 |
|---|---|
| `split_record.json` | 分割の条件（乱数シード・比率など）と、どの画像がどのセットに入ったかの完全な記録 |
| `scenes_train.txt` など | セットごとの生画像（シーン）の一覧 |
| `crops_train.txt` など | セットごとの切り出し画像の一覧 |

**同じ乱数シードで実行すれば、必ず同じ分割になります。**

---

## 4. ファイル構成

```
harvest_nn/
├── README.md                     このファイル
├── requirements.txt              必要なライブラリの一覧
├── camera_params.yaml            カメラの内部/外部パラメータの設定（編集可）
├── KFuji_RGB-DS_dataset/         データセット（fetch_dataset.py が作る）
├── outputs/                      実行結果の出力先（自動で作られます）
├── docs/
│   └── measure_centralized_dependencies.md  計測スクリプトの処理の依存関係の図
└── src/
    ├── common_paths.py           データセットのファイルの場所を管理する
    ├── kinect_io.py              カラー画像と点群を読み込む
    ├── depth_alignment.py        【ステップ2】点群をカラー画像に投影する
    ├── localization_3d.py        【ステップ5】3次元座標を計算する
    ├── timing_utils.py           時間を正しく測るための道具
    ├── machine_info.py           CPU名・GPU名などマシン情報を集める
    ├── fetch_dataset.py          ⓪ データ取得
    ├── prepare_dataset.py        ① データ準備
    ├── verify_alignment.py       ② アライメントの確認（任意）
    ├── train_yolo.py             ③ 学習
    ├── measure_centralized.py    ④ 計測（メイン）
    └── compare_machines.py       ⑤ 2台の結果の比較
```

---

## 5. 設計上の判断とその理由

このデータセットを実際に調べた結果に基づく判断です。**結果を解釈するときに重要**なので、
一度目を通してください。

### 5-1. データセット付属の train/val/test 分割は使っていません

`preprocessed data/sets/` に train/val/test の分割ファイルが入っていますが、
**これを使うと精度が不当に高く出ます。**

調べたところ、テスト用93シーンの**すべて**が学習用にも入っていました。
1枚の生画像を3×3に切り出した9枚がバラバラに振り分けられており、
切り出し同士には重なりがあるため、**同じりんごが学習用とテスト用の両方に現れます。**

| 付属の分割 | 切り出し画像 | 元のシーン |
|---|---|---|
| train | 619枚 | 109シーン |
| test | 193枚 | 93シーン（**全部 train にも含まれる**） |

そこで `prepare_dataset.py` では、次の方針で分割し直しています。

1. **同じ生画像から切り出した9枚は、必ず同じセットに入れる**
2. **同じ撮影グループの中でフレーム番号順に並べ、連続する3シーンを1ブロックとして扱う**
   （果樹園を移動しながら3フレーム間隔で撮影されているため、隣のシーンは構図が似ています）
3. ブロック単位で、乱数シードを固定してシャッフルし、比率に合わせて振り分ける
4. **違うセットの境目にあるシーンを1枚捨てる**（ガードバンド）

実行すると、リークが無いことの検査結果が表示されます。

### 5-2. アノテーションの個体ID（csvの1列目）は、シーンをまたいでは使えません

csv の1列目は個体IDですが、これは**シーンごとに連番のブロックを割り当てたもの**で、
シーンをまたいで同じりんごを追跡した番号ではありません。

例: `BD11_inf_201710_003` は ID 9244〜9361、`BD11_inf_201710_039` は ID 9237〜9430。
番号の範囲がたまたま重なっているだけで、同じIDの枠は画像上の全く違う位置にあります。

**1枚の生画像から切り出した9枚の間では**、同じIDは同じりんごを指します（これは利用できます）。

### 5-3. 計測には「位置合わせ前」の生データを使っています

`preprocessed data/` の `_DS.mat` はすでにカラー画像に位置合わせ済みなので、
これを使うとステップ2の時間が測れません。

そこで `row data/` の生データを使います。

- カラー画像 `*_RGB.jpg` … 1920×1080
- 点群 `*_pc.mat` … N行8列（X, Y, Z, R, G, B, IR, 距離補正IR）、点は格子状に並んでいない

ステップ2では、この点群を1点ずつカラー画像に投影して深度画像を作っています。

### 5-4. 点群は `.npy` に変換してから計測します

`.mat` は MATLAB 形式で、読み込みに解凍と解析が必要です。実測で **約36ミリ秒**かかります。
一方、`.npy` に変換しておくと **約1.3ミリ秒**で読めます。

実際のロボットではカメラから直接データを受け取るので、`.mat` の解析時間は
本来の「画像取得」には含まれません。そこで `prepare_dataset.py` であらかじめ変換し、
ステップ1では `.npy` の読み込み時間を計測しています。

### 5-5. 推論は 548×373 のタイル9枚に分けて行います

学習は 548×373 の切り出し画像で行います。もし 1920×1080 の生画像をそのまま
640 に縮めて推論すると、りんごの見かけの大きさが学習時の約3分の1になり、
検出精度が大きく落ちます。

そこで生画像を**学習時と同じ大きさのタイル9枚**に分け、1枚ずつ推論します
（バッチサイズは1固定）。タイルは深度が取れる範囲（画像中央の約1576×1053）を覆います。
タイルの位置や枚数は `camera_params.yaml` の `tiling` で変更できます。

### 5-6. カメラの外部パラメータは自分で求めました

データセットにキャリブレーション結果が同梱されていないため、内部パラメータは
Kinect v2 の公称値を使っています。

ただし公称値のまま投影すると、**約30ピクセル横にずれます**。
Kinect v2 は深度センサとカラーカメラが物理的に離れた位置に付いているためです。

`verify_alignment.py` で、点群が持つ色と投影先のカラー画像の色を比べて
一番よく一致する平行移動量を探した結果、次の値になりました。

| 設定 | 一致度（相関係数） |
|---|---|
| 平行移動なし | 0.31 |
| `translation_meters: [-0.055, -0.009, 0.0]` | **0.63** |

求まった `-0.055 m` は、Kinect v2 の深度センサとカラーカメラの実際の間隔
（約52mm）とほぼ一致しており、物理的にも妥当な値です。この値を `camera_params.yaml`
に設定済みです。

---

## 6. 正しく時間を測るための工夫

このプログラムでは、計測が不正確にならないよう次のことを行っています。

- **`time.perf_counter()` を使う** … 経過時間の測定専用の、精度の高い時計です
- **GPU の処理を待ち合わせる** … GPU は非同期に動くため、`torch.cuda.synchronize()`
  （Apple の GPU では `torch.mps.synchronize()`）で計算の完了を待ってから時間を止めます。
  これをしないと、実際よりずっと短い時間が出てしまいます
- **ウォームアップを除外する** … 最初の数回は CUDA の初期化などで極端に遅くなるため、
  計測から外します（`--warmup`）
- **画像ごとにもウォームアップする** … GPU は検出数などテンソルの形が変わるたびに
  内部の準備をやり直します。これを入れないと各画像の1回目だけ極端に遅い値が出ます（`--warmup-per-image`）
- **同じ処理を複数回繰り返す** … 平均・中央値・標準偏差を出します（`--repeats`）
- **バッチサイズは1に固定** … ロボットは画像を1枚ずつ処理するためです

---

## 7. `measure_centralized.py` の処理の詳細

計測スクリプトの中の関数を、**入力（引数）と出力（返り値）** がわかるように1つずつ説明します。
数値の例は `camera_params.yaml` の既定値（生画像 1920×1080、タイル 548×373 を 3×3 枚、
`--imgsz 640`）の場合です。

各処理の依存関係の図（Mermaid 記法）は
[`docs/measure_centralized_dependencies.md`](docs/measure_centralized_dependencies.md) にあります。

### 7-1. 全体の流れ

```
main()
 ├─ 準備（計測の外）: 引数の読み取り → 装置の決定 → マシン情報 → camera_params.yaml
 │                    → alignment_transform / localization_settings / tile_list の事前計算
 │                    → シーン一覧の読み込み → モデルの読み込み
 ├─ ウォームアップ   : 1枚目の画像で measure_one_image() を --warmup 回（結果は捨てる）
 ├─ 本計測           : 画像ごとに
 │                      measure_one_image() を --warmup-per-image 回（結果は捨てる）
 │                      measure_one_image() を --repeats 回（ここだけ記録する）
 └─ 結果の書き出し   : CSV 3種類 + JSON 1つ、画面にまとめを表示
```

`measure_one_image()` の中でステップ1〜5を順番に実行し、ステップごとに
`timing_utils.start_timer()` / `stop_timer()` で時間を測ります。
**設定ファイルの読み込み・行列やタイル位置の計算・モデルの読み込みは計測の外で1回だけ**
行い、計測にはステップの処理そのものだけが入るようにしています。

### 7-2. 準備用の関数（計測の外で1回だけ呼ばれる）

#### `build_tile_list(tiling_settings, image_width, image_height)`

生画像を切り分けるタイルの位置を計算します。

| | 内容 |
|---|---|
| 入力 `tiling_settings` | `camera_params.yaml` の `tiling` の辞書（`tile_width`, `tile_height`, `columns`, `rows`, `stride_x`, `stride_y`, `origin_x`, `origin_y`） |
| 入力 `image_width`, `image_height` | 生画像の幅と高さ（1920, 1080） |
| 出力 | タイルの `(左端x, 上端y, 幅, 高さ)` のリスト。既定では 9 個（左上から右へ、行ごとの順） |

- `origin + 列(行)番号 × stride` で各タイルの左上を求めます。
- タイルが画像の外にはみ出す場合は、画像の内側に収まるように位置をずらします
  （タイルの大きさは変えません）。
- 既定値では x = 172, 686, 1200、y = 13, 353, 693 の組み合わせになります。

#### `compute_letterbox_parameters(source_width, source_height, model_input_size)`

横長のタイルを、縦横比を保ったまま正方形のモデル入力に収めるための値を計算します
（レターボックス）。ステップ3（縮小と余白の追加）とステップ4b（座標を元に戻す）の
**両方から同じ計算で呼ばれる**ので、2つのステップの座標の変換が必ず一致します。

| | 内容 |
|---|---|
| 入力 | タイルの幅・高さ（548, 373）、モデル入力の一辺（640） |
| 出力 | `(縮小率, 左の余白, 上の余白, 縮小後の幅, 縮小後の高さ)` |

- 縮小率は `min(640/548, 640/373)` ≈ 1.168（実際は拡大）です。
- 既定値では、縮小後 640×436、左の余白 0、上の余白 102 ピクセルになります。

#### `read_scene_name_list(scene_list_file_path)`

| | 内容 |
|---|---|
| 入力 | シーン名が1行に1つ書かれたテキストファイルのパス（既定は `outputs/splits/scenes_test.txt`） |
| 出力 | シーン名（文字列）のリスト。空の行は除きます |

#### 他のモジュールで用意するもの（`main()` から呼び出す）

| 関数 | 入力 | 出力 |
|---|---|---|
| `timing_utils.resolve_device()` | `--device` の値（`auto` / `cuda` / `mps` / `cpu`） | `torch.device`。`auto` なら cuda → mps → cpu の順に選ぶ |
| `machine_info.collect_machine_information()` | マシン名、装置 | マシン情報の辞書（CPU名・GPU名・メモリ量・各ライブラリのバージョンなど） |
| `depth_alignment.load_camera_parameters()` | `camera_params.yaml` のパス | 設定全体の辞書 |
| `depth_alignment.build_alignment_transform()` | 設定全体の辞書 | `alignment_transform`（ステップ2用。軸の並べ替えと回転を1つにまとめた行列、平行移動、fx/fy/cx/cy、画像サイズ、点群の列の対応、隙間を埋める設定） |
| `localization_3d.build_localization_settings()` | 設定全体の辞書 | `localization_settings`（ステップ5用。`box_shrink_ratio`, `minimum_valid_pixels`, fx/fy/cx/cy） |
| `ultralytics.YOLO(weights).model` | 重みファイル(.pt) | ニューラルネットワーク本体。`eval()` → `fuse()`（畳み込みとバッチ正規化の統合）→ 装置へ転送 → `--half` なら float16 にする |

モデルは `YOLO` オブジェクトの `predict()` を使わず、**中身のネットワーク (`.model`) だけを
取り出して**使います。`predict()` は前処理・推論・NMS を一度に行ってしまい、
ステップごとに時間を分けて測れないためです。

### 7-3. ステップ1〜5（計測の対象）

#### ステップ1 `step1_image_acquisition(color_image_path, point_cloud_npy_path)`

カラー画像と深度データ（点群）をファイルから読み込みます。実際のロボットでは、
カメラからデータを受け取る部分にあたります。

| | 内容 |
|---|---|
| 入力 `color_image_path` | 生のカラー画像 `<シーン名>_RGB.jpg` のパス |
| 入力 `point_cloud_npy_path` | `.npy` に変換済みの点群 `outputs/point_cloud_npy/<シーン名>_pc.npy` のパス |
| 出力 `color_image_bgr` | カラー画像。形 `(1080, 1920, 3)`、`uint8`、色は **B, G, R の順** |
| 出力 `point_cloud_array` | 点群。形 `(N, 8)`、`float32`。列は X, Y, Z（メートル。X=右, Y=前, Z=上）, R, G, B, IR, S。N はシーンによって変わる（例: 約17万点） |

- 中身は `kinect_io.load_color_image()`（`cv2.imread`）と
  `kinect_io.load_point_cloud_from_npy()`（`np.load`）です。
- 計測時間には **ファイルの読み込みと JPEG の展開** が含まれます。

#### ステップ2 `step2_depth_alignment(point_cloud_array, alignment_transform)`

点群をカラー画像の座標系に投影して、カラー画像と同じ大きさの深度画像を作ります。

| | 内容 |
|---|---|
| 入力 `point_cloud_array` | ステップ1の点群 `(N, 8)` |
| 入力 `alignment_transform` | `build_alignment_transform()` が作った辞書 |
| 出力 `depth_image_meters` | 深度画像。形 `(1080, 1920)`、`float32`、単位はメートル。**深度が分からないピクセルは 0** |

中身は `depth_alignment.align_depth_to_color()` で、次の順に計算します。

1. 点群から X, Y, Z の3列を取り出す
2. 座標軸の並べ替え（X=右,Y=前,Z=上 → x=右,y=下,z=前）と外部パラメータの回転を
   まとめた行列を掛け、平行移動を足してカラーカメラの座標系に移す
3. `u = fx·x/z + cx`、`v = fy·y/z + cy` で画像上の位置に投影し、
   カメラの後ろの点と画像の外に出た点を捨てる
4. 同じピクセルに複数の点が来たら一番手前の点を残す（Zバッファ。奥行きの逆数の最大値で判定）
5. `fill_holes: true` なら膨張処理で点と点の隙間を埋める
6. 逆数を元に戻してメートル単位の深度にする

- **CPU（numpy / OpenCV）だけで計算**します。GPU は使いません。
- カラー画像は使わないので、ステップ3・4とは互いに依存しません。

#### ステップ3 `step3_inference_preprocessing(color_image_bgr, tile_list, model_input_size, device, use_half_precision)`

カラー画像をタイルに分け、1枚ずつモデルに入力できる形に整えます。

| | 内容 |
|---|---|
| 入力 `color_image_bgr` | ステップ1のカラー画像 `(1080, 1920, 3)` BGR |
| 入力 `tile_list` | `build_tile_list()` が作ったタイル位置のリスト |
| 入力 `model_input_size` | モデル入力の一辺（`--imgsz`、既定 640） |
| 入力 `device` | 計算装置（`cuda` / `mps` / `cpu`） |
| 入力 `use_half_precision` | `True` なら float16、`False` なら float32 にする |
| 出力 `input_tensor_list` | テンソルのリスト（タイルの数 = 9 個）。1つの形は `(1, 3, 640, 640)`、値は 0.0〜1.0、色は **R, G, B の順**、`device` 上にある |

タイル1枚ごとの処理:

1. 生画像からタイルの範囲を切り出す（numpy のスライスなのでコピーは作らない）
2. `cv2.resize`（`INTER_LINEAR`）で 548×373 → 640×436 に変える
3. 640×640 の灰色（値 114）の画像の中に貼り付ける（上下に 102 ピクセルずつの余白）
4. BGR → RGB に並べ替える
5. `(高さ, 幅, 色)` → `(色, 高さ, 幅)` に並べ替え、メモリ上で連続にする
6. `torch` のテンソルにして **装置へ転送** する
7. float16 / float32 に変換し、255 で割って 0.0〜1.0 にする
8. 先頭にバッチの次元を足して `(1, 3, 640, 640)` にする

- 計測時間には **CPU から GPU へのデータ転送** も含まれます。
- 型の変換と正規化は転送後に行うので、GPU を使う場合は GPU 上で計算されます。

#### ステップ4a `step4a_image_inference(detection_model, input_tensor_list)`

ニューラルネットワークの順伝播だけを行います。NMS は行いません。

| | 内容 |
|---|---|
| 入力 `detection_model` | `main()` で準備したネットワーク本体 |
| 入力 `input_tensor_list` | ステップ3のテンソルのリスト |
| 出力 `raw_prediction_list` | タイルごとの生の予測のリスト（9 個）。YOLOv8 では1つの形が `(1, 4 + クラス数, 8400)`。このプロジェクトは「apple」1クラスなので `(1, 5, 8400)` |

- 8400 は 640×640 入力のときの候補の数（80×80 + 40×40 + 20×20）です。
  各候補は `[中心x, 中心y, 幅, 高さ, apple の自信度]`（640×640 の画像上の座標）を持ちます。
- `torch.no_grad()` の中で、**バッチサイズ 1 のまま 9 回** 順伝播します。
- モデルの出力が組（タプル）の場合は、先頭の要素を検出結果として使います。

#### ステップ4b `step4b_postprocess_nms(raw_prediction_list, tile_list, model_input_size, confidence_threshold, iou_threshold, merge_iou_threshold)`

重なった検出枠をまとめ、枠の座標を生画像の座標に戻します。

| | 内容 |
|---|---|
| 入力 `raw_prediction_list` | ステップ4aの生の予測のリスト |
| 入力 `tile_list` | タイル位置のリスト（座標を生画像に戻すのに使う） |
| 入力 `model_input_size` | モデル入力の一辺（レターボックスの計算をやり直すのに使う） |
| 入力 `confidence_threshold` | `--conf`（既定 0.25）。これより自信度の低い候補を捨てる |
| 入力 `iou_threshold` | `--iou`（既定 0.45）。タイル内の NMS のしきい値 |
| 入力 `merge_iou_threshold` | `--merge-iou`（既定 0.5）。タイルをまたいだ統合のしきい値 |
| 出力 `detection_boxes` | 検出枠。numpy 配列、形 `(M, 4)`、`[x1, y1, x2, y2]`（**生画像 1920×1080 のピクセル座標**）。M は検出数 |
| 出力 `detection_scores` | 各枠の自信度。numpy 配列、形 `(M,)` |

2段階で処理します。

1. **タイルごとの NMS** … ultralytics の `non_max_suppression()` で自信度の低い候補を捨て、
   重なった枠を1つにまとめます（結果は `[x1, y1, x2, y2, 自信度, クラス番号]`）。
   そのあと、余白を引く → 縮小率で割る → タイルの左上の位置を足す、の順で
   生画像の座標に戻します。
2. **タイルをまたいだ統合** … 全タイルの枠をつなげ、`torchvision.ops.nms()` で
   タイルの重なり部分で二重に検出された枠をまとめます。

- どのタイルでも何も検出されなかったときは、形 `(0, 4)` と `(0,)` の空の配列を返します。
- 計測時間には **GPU から CPU への結果の転送**（`.cpu().numpy()`）も含まれます。
- クラス番号は1クラスなので捨てています。

#### ステップ5 `step5_spatial_localization(depth_image_meters, detection_boxes, localization_settings)`

検出枠と深度画像から、りんご1個ずつの3次元座標を求めます。

| | 内容 |
|---|---|
| 入力 `depth_image_meters` | ステップ2の深度画像 `(1080, 1920)` |
| 入力 `detection_boxes` | ステップ4bの検出枠 `(M, 4)` |
| 入力 `localization_settings` | `build_localization_settings()` が作った辞書 |
| 出力 `apple_position_list` | りんご1個につき1つの辞書のリスト（長さ M）。キーは下の表のとおり |

中身は `localization_3d.estimate_apple_positions_3d()` で、枠ごとに次の計算をします。

1. 枠の中央 `box_shrink_ratio`（既定 0.5 = 縦横それぞれ中央 50%）の範囲を深度画像から切り出す
2. 深度が 0 より大きいピクセルだけを集め、`minimum_valid_pixels`（既定 10）個以上あれば
   その**中央値**を深度とする（足りなければ深度 0 = 不明）
3. 枠の中心 `(u, v)` と深度から、`X = (u−cx)·深度/fx`、`Y = (v−cy)·深度/fy`、`Z = 深度`
   で3次元座標を求める（深度が不明なら NaN）

出力の辞書のキー:

| キー | 内容 |
|---|---|
| `detection_index` | 何番目の検出枠か |
| `box_x1`, `box_y1`, `box_x2`, `box_y2` | 検出枠（生画像のピクセル座標） |
| `center_u`, `center_v` | 枠の中心のピクセル位置 |
| `valid_depth_pixels` | 深度の計算に使えたピクセルの数 |
| `median_depth_m` | 深度の中央値（メートル）。0 なら深度が取れなかった |
| `position_x/y/z_optical_m` | カメラ光学座標系の3次元座標（x=右, y=下, z=前） |
| `position_x/y/z_sensor_m` | センサ座標系の3次元座標（X=右, Y=前, Z=上） |

### 7-4. 1枚分の計測 `measure_one_image(...)`

ステップ1〜5を順番に呼び、それぞれの前後で時間を測ります。

| | 内容 |
|---|---|
| 入力 | カラー画像と点群のパス、`alignment_transform`、`localization_settings`、`tile_list`、モデル、`model_input_size`、`device`、`use_half_precision`、3つのしきい値 |
| 出力 `measurement_result` | 計測結果の辞書（下の表） |
| 出力 `apple_position_list` | ステップ5の出力そのもの |

| キー | 内容 |
|---|---|
| `step1_acquisition_ms` 〜 `step5_localization_ms` | 各ステップの時間（ミリ秒）。4a と 4b は別々 |
| `total_ms` | 上の6つの合計（ステップの間の、計測に含まれないわずかな時間は入らない） |
| `num_tiles` | タイルの数 |
| `num_points` | 点群の点の数 N |
| `num_detections` | 検出数 M |
| `num_detections_with_depth` | そのうち `median_depth_m > 0` の数 |

`start_timer()` / `stop_timer()` は時刻を取る前に `torch.cuda.synchronize()`
（MPS では `torch.mps.synchronize()`）を呼ぶので、GPU で非同期に動く計算も
そのステップの時間に正しく含まれます。

### 7-5. `main()` の入力と出力

**入力（コマンドライン引数）**

| 引数 | 既定値 | 内容 |
|---|---|---|
| `--weights` | （必須） | 学習済みの重みファイル(.pt) |
| `--machine-name` | ホスト名 | 結果ファイルの名前と CSV の列に使うマシン名 |
| `--repeats` | 10 | 1枚あたりの計測回数 |
| `--warmup` | 5 | 最初に行う準備運転の回数（計測から除外） |
| `--warmup-per-image` | 1 | 画像を切り替えるたびの準備運転の回数（計測から除外） |
| `--limit-images` | 0 | 計測する画像の枚数の上限（0 なら全部） |
| `--device` | `auto` | 計算装置 |
| `--half` | なし | 指定すると float16 で推論（CUDA 以外では警告を出して無効） |
| `--imgsz` | 640 | モデル入力の一辺 |
| `--conf` / `--iou` / `--merge-iou` | 0.25 / 0.45 / 0.5 | ステップ4bのしきい値 |
| `--camera-params` | `camera_params.yaml` | カメラパラメータのファイル |
| `--scene-list` | `outputs/splits/scenes_test.txt` | 計測するシーンの一覧 |
| `--dataset-root` | 既定の場所 | データセットの場所 |

**入力（ファイル）**

- シーン一覧（無ければ `prepare_dataset.py` を先に実行するよう促して止まる）
- シーンごとのカラー画像 `<シーン名>_RGB.jpg` と点群 `outputs/point_cloud_npy/<シーン名>_pc.npy`
  （`.npy` が無いシーンは警告を出して飛ばす。1つも無ければ止まる）
- `camera_params.yaml`、重みファイル

**出力（`outputs/measurements/` に保存。列の意味は [3. 出力ファイルの見方](#3-出力ファイルの見方)）**

| ファイル | 内容 |
|---|---|
| `timings_raw_<マシン名>.csv` | `measure_one_image()` の結果1回分が1行（画像の枚数 × `--repeats` 行）。ウォームアップの分は入らない |
| `timings_summary_<マシン名>.csv` | ステップごとに1行。`timing_utils.summarize_measurements()` で求めた平均・中央値・標準偏差・最小・最大・95パーセンタイルと、平均で見た全体に占める割合。マシン情報も各行に入る |
| `machine_info_<マシン名>.json` | マシン情報と計測の設定（重みファイル、回数、画像の枚数、タイルの数、入力サイズ、バッチサイズ 1、半精度の有無） |
| `detections_<マシン名>.csv` | 各画像の**最後の繰り返し**で得た、りんご1個ごとの3次元座標（ステップ5の出力の辞書 + マシン名 + シーン名）。検出が1つも無ければ作られない |

画面には、画像ごとの平均合計時間と検出数、最後にステップごとの平均・中央値・標準偏差・割合と
1秒あたりの処理枚数 (FPS) を表示します。

---

## 8. 設定ファイル `camera_params.yaml`

カメラのパラメータと、ステップ2・ステップ5・タイル分割の動作を変更できます。
すべての項目に日本語の説明を書いてあります。主な項目は次のとおりです。

| 項目 | 内容 |
|---|---|
| `color_camera` | カラーカメラの内部パラメータ（fx, fy, cx, cy）と画像サイズ |
| `axis_permutation_matrix` | 点群の座標軸をコンピュータビジョンの標準の向きに直す行列 |
| `extrinsics` | 深度センサとカラーカメラの位置のずれ（回転と平行移動） |
| `depth_alignment` | 投影後の隙間を埋めるかどうかと、その強さ |
| `spatial_localization` | 検出枠の中央何割を深度の取得に使うか |
| `tiling` | タイルの大きさ・枚数・位置 |

---

## 9. 困ったときは

**`ModuleNotFoundError: No module named 'torch'` と出る**
→ 仮想環境が有効になっていません。`source .venv/bin/activate` を実行してください。

**`データセット設定ファイルが見つかりません` と出る**
→ 先に `python src/prepare_dataset.py` を実行してください。

**`データセットのアノテーションフォルダが見つかりません` と出る**
→ データセットがまだありません。`python src/fetch_dataset.py` を実行してください。
すでに手元にある場合は `--dataset-root` か環境変数 `KFUJI_DATASET_ROOT` で置き場所を指定してください。

**データの取得が途中で止まった / 回線が切れた**
→ もう一度 `python src/fetch_dataset.py` を実行してください。
取得済みのファイルは飛ばして、残りだけを取得します。

**`サーバが範囲指定に対応していません` と出る**
→ 間にプロキシなどが入っている環境です。自動で「zip全体を取得する方式」に切り替わりますが、
明示的に `python src/fetch_dataset.py --full-archive` としても構いません（2.9GB 必要です）。

**`計測できるシーンがありません` と出る**
→ 点群の `.npy` がまだ作られていません。
`python src/prepare_dataset.py --convert-point-clouds test` を実行してください。

**GPU のメモリが足りないと出る**
→ 学習時は `--batch 8` や `--batch 4` に下げてください。

**学習が非常に遅い**
→ `torch.cuda.is_available()` が `True` になっているか確認してください。
`False` だと CPU で学習しており、GPU の数十倍の時間がかかります。

**計測結果のばらつき（標準偏差）が大きい**
→ `--warmup` と `--repeats` を増やしてください。
また、他の重いプログラムを同時に動かしていないか確認してください。

---

## 10. データセットの利用について

KFuji RGB-DS database は **CC-BY-NC-SA 4.0**（研究・教育目的のみ、商用利用不可）です。
配布元は Zenodo の <https://zenodo.org/records/3715991> です
（`fetch_dataset.py` はここから取得します）。
利用する場合は次の論文を引用してください。

- Gené-Mola J, Vilaplana V, Rosell-Polo JR, Morros JR, Ruiz-Hidalgo J, Gregorio E. (2019).
  *Multi-modal Deep Learning for Fruit Detection Using RGB-D Cameras and their Radiometric
  Capabilities.* Computers and Electronics in Agriculture, 162, 689-698.
- Gené-Mola J, Vilaplana V, Rosell-Polo JR, Morros JR, Ruiz-Hidalgo J, Gregorio E. (2019).
  *KFuji RGB-DS database: Fuji apple multi-modal images for fruit detection with color,
  depth and range-corrected IR data.* Data in Brief, 25, 104289.
