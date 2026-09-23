# `measure_centralized.py` の処理の依存関係

`src/measure_centralized.py` の各処理が、どのデータ・どの処理に依存しているかを
Mermaid 記法でまとめたものです。各処理の入力と出力の詳しい説明は
[README の「7. `measure_centralized.py` の処理の詳細」](../README.md#7-measure_centralizedpy-の処理の詳細)
を見てください。

## 1. データの流れ（ステップ間の依存関係）

1枚の画像を処理するとき（`measure_one_image()` の中）の、データの受け渡しです。
四角は処理、角の丸い四角はデータ、灰色は計測の外で1回だけ準備するものです。

```mermaid
flowchart TD
    %% ---- 計測の外で1回だけ準備するもの ----
    subgraph PREP["準備（計測の外・1回だけ）"]
        YAML[/"camera_params.yaml"/]
        WEIGHTS[/"重みファイル .pt"/]
        ARGS[/"コマンドライン引数<br/>--imgsz --conf --iou --merge-iou<br/>--device --half"/]
        AT(["alignment_transform"])
        LS(["localization_settings"])
        TL(["tile_list<br/>9個の (x, y, 幅, 高さ)"])
        MODEL(["detection_model<br/>eval + fuse + 装置へ転送"])
        YAML -->|build_alignment_transform| AT
        YAML -->|build_localization_settings| LS
        YAML -->|"build_tile_list（tiling）"| TL
        WEIGHTS -->|"YOLO(...).model"| MODEL
        ARGS -.-> MODEL
    end

    %% ---- 入力ファイル ----
    JPG[/"シーン名_RGB.jpg"/]
    NPY[/"シーン名_pc.npy"/]

    %% ---- ステップ1〜5 ----
    S1["ステップ1 画像取得<br/>step1_image_acquisition"]
    S2["ステップ2 深度アライメント<br/>step2_depth_alignment"]
    S3["ステップ3 推論前処理<br/>step3_inference_preprocessing"]
    S4A["ステップ4a 画像推論<br/>step4a_image_inference"]
    S4B["ステップ4b 後処理 NMS<br/>step4b_postprocess_nms"]
    S5["ステップ5 3D空間位置推定<br/>step5_spatial_localization"]

    %% ---- 中間データ ----
    COLOR(["color_image_bgr<br/>(1080, 1920, 3) uint8 BGR"])
    PC(["point_cloud_array<br/>(N, 8) float32"])
    DEPTH(["depth_image_meters<br/>(1080, 1920) float32"])
    TENSORS(["input_tensor_list<br/>9 × (1, 3, 640, 640)"])
    RAW(["raw_prediction_list<br/>9 × (1, 5, 8400)"])
    BOXES(["detection_boxes (M, 4)<br/>detection_scores (M,)"])
    APPLES(["apple_position_list<br/>M 個の辞書"])

    JPG --> S1
    NPY --> S1
    S1 --> COLOR
    S1 --> PC

    PC --> S2
    AT --> S2
    S2 --> DEPTH

    COLOR --> S3
    TL --> S3
    ARGS -.->|"imgsz, device, half"| S3
    S3 --> TENSORS

    TENSORS --> S4A
    MODEL --> S4A
    S4A --> RAW

    RAW --> S4B
    TL --> S4B
    ARGS -.->|"imgsz, conf, iou, merge-iou"| S4B
    S4B --> BOXES

    DEPTH --> S5
    BOXES --> S5
    LS --> S5
    S5 --> APPLES

    classDef prep fill:#eeeeee,stroke:#888888,color:#333333
    class YAML,WEIGHTS,ARGS,AT,LS,TL,MODEL prep
```

**読み取れること**

- ステップ2（深度アライメント）は点群だけを使い、ステップ3〜4b（カラー画像の処理）とは
  **互いに依存しません**。2つの流れが合流するのはステップ5だけです。
  集中型計算方式ではこれを1台で順番に実行していますが、分散型にする場合は
  この2つの流れを別々の装置で同時に動かせます。
- `tile_list` はステップ3（切り出し）とステップ4b（座標を生画像に戻す）の両方で使います。
  2つのステップは `compute_letterbox_parameters()` を同じ引数で呼ぶので、
  座標の変換が必ず一致します。

## 2. 実行順序と時間の計測範囲

`measure_one_image()` の中では、データの依存関係とは関係なく、
ステップ1 → 2 → 3 → 4a → 4b → 5 の順に1つずつ実行し、
それぞれを `start_timer()` / `stop_timer()` で囲んで測ります。

```mermaid
flowchart LR
    subgraph M["measure_one_image()"]
        direction LR
        T1["⏱ ステップ1<br/>ファイル読み込み<br/>（CPU）"]
        T2["⏱ ステップ2<br/>点群の投影<br/>（CPU）"]
        T3["⏱ ステップ3<br/>タイル化・レターボックス<br/>・GPUへ転送・正規化"]
        T4A["⏱ ステップ4a<br/>順伝播 × 9回<br/>（GPU）"]
        T4B["⏱ ステップ4b<br/>NMS × 9回 + 統合NMS<br/>・CPUへ転送"]
        T5["⏱ ステップ5<br/>中央値の深度から<br/>3D座標（CPU）"]
        T1 --> T2 --> T3 --> T4A --> T4B --> T5
    end
    T5 --> R(["measurement_result<br/>各ステップの ms + total_ms<br/>+ apple_position_list"])
```

⏱ の各処理の前後で `torch.cuda.synchronize()`（MPS では `torch.mps.synchronize()`）を
呼んでから時刻を取るので、GPU の非同期実行があってもステップの時間が正しく分かれます。
（「GPU」と書いたところは、GPU が使えない環境では CPU で動きます。）

## 3. 関数の呼び出し関係（モジュール間の依存）

```mermaid
flowchart LR
    subgraph MC["measure_centralized.py"]
        MAIN["main()"]
        RSL["read_scene_name_list()"]
        BTL["build_tile_list()"]
        MOI["measure_one_image()"]
        CLP["compute_letterbox_parameters()"]
        S1["step1_image_acquisition()"]
        S2["step2_depth_alignment()"]
        S3["step3_inference_preprocessing()"]
        S4A["step4a_image_inference()"]
        S4B["step4b_postprocess_nms()"]
        S5["step5_spatial_localization()"]
    end

    subgraph KIO["kinect_io.py"]
        LCI["load_color_image()"]
        LPN["load_point_cloud_from_npy()"]
    end

    subgraph DA["depth_alignment.py"]
        LCP["load_camera_parameters()"]
        BAT["build_alignment_transform()"]
        ADC["align_depth_to_color()"]
    end

    subgraph L3D["localization_3d.py"]
        BLS["build_localization_settings()"]
        EAP["estimate_apple_positions_3d()"]
    end

    subgraph TU["timing_utils.py"]
        RD["resolve_device()"]
        TIMER["start_timer() / stop_timer()"]
        SYNC["synchronize_device()"]
        SUM["summarize_measurements()"]
    end

    subgraph MI["machine_info.py"]
        CMI["collect_machine_information()"]
        PMI["print_machine_information()"]
    end

    subgraph CP["common_paths.py"]
        PATHS["get_default_dataset_root()<br/>get_project_root()<br/>get_outputs_directory()<br/>get_raw_color_image_path()<br/>get_point_cloud_npy_path()"]
    end

    subgraph EXT["外部ライブラリ"]
        YOLO["ultralytics.YOLO"]
        NMS["ultralytics non_max_suppression"]
        TVNMS["torchvision.ops.nms"]
    end

    MAIN --> PATHS
    MAIN --> RD
    MAIN --> CMI
    MAIN --> PMI
    MAIN --> LCP
    MAIN --> BAT
    MAIN --> BLS
    MAIN --> BTL
    MAIN --> RSL
    MAIN --> YOLO
    MAIN --> MOI
    MAIN --> SUM

    MOI --> TIMER
    TIMER --> SYNC
    MOI --> S1
    MOI --> S2
    MOI --> S3
    MOI --> S4A
    MOI --> S4B
    MOI --> S5

    S1 --> LCI
    S1 --> LPN
    S2 --> ADC
    S3 --> CLP
    S4B --> CLP
    S4B --> NMS
    S4B --> TVNMS
    S5 --> EAP
```

## 4. `main()` 全体の流れ

```mermaid
flowchart TD
    A["引数を読み取る"] --> B["データセット・camera_params.yaml・<br/>シーン一覧の場所を決める"]
    B --> C{"シーン一覧が<br/>ある？"}
    C -- "ない" --> X1["FileNotFoundError<br/>prepare_dataset.py を促す"]
    C -- "ある" --> D["resolve_device()<br/>--half は CUDA のときだけ有効"]
    D --> E["collect_machine_information()<br/>→ 画面に表示"]
    E --> F["load_camera_parameters()<br/>build_alignment_transform()<br/>build_localization_settings()<br/>build_tile_list()"]
    F --> G["read_scene_name_list()<br/>--limit-images で絞る"]
    G --> H{".npy がある<br/>シーンが1つ以上？"}
    H -- "ない" --> X2["FileNotFoundError<br/>--convert-point-clouds を促す"]
    H -- "ある" --> I["モデルを読み込む<br/>.model → eval → fuse → to(device) → half/float"]
    I --> J["ウォームアップ<br/>1枚目の画像で measure_one_image() × --warmup"]
    J --> K["次のシーンへ"]
    K --> L["measure_one_image() × --warmup-per-image<br/>（結果は捨てる）"]
    L --> M["measure_one_image() × --repeats<br/>時間を記録し、最後の回だけ3D座標も記録"]
    M --> N["この画像の平均合計時間を表示"]
    N --> O{"まだ<br/>シーンがある？"}
    O -- "ある" --> K
    O -- "ない" --> P["timings_raw_*.csv"]
    P --> Q["summarize_measurements()<br/>→ timings_summary_*.csv"]
    Q --> R["machine_info_*.json"]
    R --> S["detections_*.csv<br/>（検出があれば）"]
    S --> T["まとめの表と FPS を表示"]
```
