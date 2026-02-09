# Neural Network Weight Visualizer 仕様書

## Quick Review: ユーザに提供される機能

本ツールは Marimo notebook (`playground/weight_visualizer.py`) として動作し、以下の機能を提供する:

1. **モデル選択**: `models/` 内の PyTorch (.pt) ファイルをドロップダウンで選択
2. **レイヤー・テンソル選択**: モデル内のレイヤーを選び、weight または bias を切り替えて表示
3. **ヒストグラム**: Plotly によるインタラクティブな重み分布表示。ビン数はスライダーで調整可能
4. **量子化ビュー切替**: 量子化モデル選択時、int8整数値 / デ量子化FP32値 の2ビューを切り替え可能
5. **値範囲分析**: min/max を指定して、範囲内のパラメータ数と割合を表示。ヒストグラム上でハイライト
6. **統計情報**: Min, Max, Mean, Std, Shape, パラメータ数を常時表示

---

## 1. 概要

### 1.1 目的

ResNet8 モデルの各レイヤーの重み分布をインタラクティブに可視化・分析するツール。量子化前後の重み分布の変化を確認し、量子化パラメータの妥当性を検証するために使用する。

### 1.2 スコープ

- `models/` ディレクトリ内の PyTorch モデルファイルの重みを可視化
- FP32 モデルおよび量子化モデル (int8) に対応
- ONNX モデル (`.onnx`) にも対応可能な設計だが、現在のリポジトリには `.pt` ファイルのみ存在
- 単一モデル・単一レイヤーの重み分布を表示（複数モデル比較は対象外）

### 1.3 対象ユーザ

ローカル環境でモデルの量子化分析を行う開発者

---

## 2. 機能要件

### 2.1 モデル選択

- `models/` ディレクトリ内の `.onnx` および `.pt` ファイルを自動検出
- ドロップダウンで選択
- **実装**: `mo.ui.dropdown(options=model_files, allow_select_none=True, label="Model")`

### 2.2 レイヤー選択

- 選択されたモデルからレイヤー一覧を動的に取得
- ドロップダウンで選択
- **実装**: `mo.ui.dropdown(options=_layer_options, allow_select_none=True, label="Layer")`

### 2.3 テンソル選択

- レイヤー選択後、`weight` / `bias` をドロップダウンで切り替え可能
- デフォルトは `weight`
- **実装**: レイヤー・テンソルは `mo.hstack([layer_selector, tensor_selector])` で横並び表示

### 2.4 量子化モデル対応

- 量子化モデル (int8) の重みについて、2つのビューを切り替え可能:
  - **int8 raw values**: 量子化された整数値のヒストグラム
  - **dequantized (FP32)**: `scale × (value - zero_point)` でFP32相当に復元した値のヒストグラム
- FP32 モデルでは切り替えUIは非表示
- **実装**: `mo.ui.radio(options={"int8 raw values": "int", "dequantized (FP32)": "fp32"}, value="dequantized (FP32)")` を `mo.md().callout(kind="neutral")` で表示

### 2.5 ヒストグラム表示

- Plotly によるインタラクティブなヒストグラム
- ビン数はスライダーで調整可能 (10〜200, step=5, デフォルト=50)
- ホバーで各ビンの詳細値を表示
- パーセンテージラベル付き (`histnorm="percent"`)
- **実装**: `go.Histogram(x=data, nbinsx=bins, histnorm="percent")` + `mo.ui.plotly(fig)`

### 2.6 値範囲分析

- ユーザが min / max を入力して [Apply Range] で適用
- デフォルト値は `mean ± std` を自動計算
- 指定範囲をヒストグラム上でハイライト表示（オレンジ色、範囲外はグレー）
- 統計パネルに以下を表示:
  - 選択範囲
  - 範囲内のパラメータ数
  - 全体に占める割合 (%)
- **実装**: `mo.ui.text` × 2 + `mo.ui.run_button` → `barmode="overlay"` で2色ヒストグラム

### 2.7 統計情報表示

- Min, Max, Mean, Std
- Shape (テンソル形状)
- Total params (パラメータ数)
- 量子化テンソルの場合: Scale, Zero Point も表示
- **実装**: `mo.md().callout(kind="neutral")`

---

## 3. 非機能要件

### 3.1 パフォーマンス

- モデル読み込みは初回のみ（Marimo のリアクティブセルで必要時にのみ再実行）
- テンソルデータはモデル読み込み時に一括抽出し `tensor_data` dict に格納
- ResNet8 規模 (約7万パラメータ) で即時応答

### 3.2 セキュリティ

- ローカル実行のみ。外部通信なし

### 3.3 可用性

- `marimo edit` で起動する Marimo notebook として提供

---

## 4. 技術仕様

### 4.1 技術スタック

| 技術 | 用途 | バージョン |
| --- | --- | --- |
| Python | 言語 | 3.12+ |
| Marimo | UIフレームワーク / notebook | 0.19.9 |
| Plotly | ヒストグラム描画 | 5.0.0+ |
| PyTorch | PyTorchモデル読み込み | 2.0.0+ |
| NumPy | 数値計算 | 1.26.4+ |
| ONNX | ONNXモデル読み込み (optional) | 1.17.0+ |

### 4.2 アーキテクチャ

**ファイル**: `playground/weight_visualizer.py` (約560行)

セル構成 (15 cells):

| # | 責務 | 出力変数 |
|---|------|----------|
| 1 | インポート・定数定義 | `MODELS_DIR`, `go`, `mo`, `np` |
| 2 | タイトル表示 | - |
| 3 | モデルファイル検出 | `model_files` |
| 4 | モデル選択UI | `model_selector` |
| 5 | モデル読み込み・テンソル抽出 | `model_data`, `load_error` |
| 6 | エラー表示 | - |
| 7 | レイヤー選択UI | `layer_selector` |
| 8 | テンソル選択UI + レイヤー/テンソル横並び | `tensor_selector` |
| 9 | テンソルデータ参照 | `tensor_entry` |
| 10 | 量子化ビュー切替 | `quant_view` |
| 11 | ビンスライダー | `bins_slider` |
| 12 | ヒストグラム表示 | `histogram_fig` |
| 13 | 値範囲入力UI | `range_min_input`, `range_max_input`, `apply_button` |
| 14 | 値範囲ハイライト表示 | - |
| 15 | 統計情報パネル | - |

### 4.3 モデル読み込み方式

**PyTorch FP32** (`resnet8.pt`):
- `torch.load()` → `dict` の `"model"` キーから `GraphModule` を取得
- `model.state_dict()` から `weight`/`bias` テンソルを抽出

**PyTorch INT8** (`resnet8_int8.pt`):
- `torch.load()` → TorchScript archive を自動検出 → `torch.jit.load()` にディスパッチ
- `RecursiveScriptModule` の量子化レイヤーから packed params を抽出:
  1. `mod._c.hasattr("_packed_params")` で量子化モジュール検出
  2. `mod._c.__getattr__("_packed_params")` で packed params 取得
  3. `torch.ops.quantized.conv2d_unpack()` / `linear_unpack()` で `(weight, bias)` を展開
  4. `weight.int_repr()` で int8 値、`weight.dequantize()` で FP32 値を取得
  5. `weight.q_per_channel_scales()` / `q_per_channel_zero_points()` でスケール・ゼロ点取得

**ONNX** (対応済みだが現在モデルファイルなし):
- `onnx.load()` → `model.graph.initializer` からテンソル取得
- scale/zero_point initializer を参照して量子化パラメータ取得

### 4.4 Marimo 設計上の注意点

- ヘルパー関数 (`_load_onnx_model`, `_load_pytorch_model`) はモデル読み込みセル**内部**で定義
  - Marimo では `def _xxx()` (アンダースコアプレフィックス) はセルプライベート、他セルから呼び出し不可
- セルローカル変数は全て `_` プレフィックス (例: `_data`, `_is_q`, `_x_title`)
  - プレフィックスなしの変数はセル間エクスポートされ、重複定義がエラーになるため
- `mo.ui.radio` の `value` パラメータはオプションの**キー（ラベル）**を指定 (値ではない)
  - 正: `value="dequantized (FP32)"` / 誤: `value="fp32"`

---

## 5. 制約・前提条件

- モデルファイルは `models/` ディレクトリに配置されていること
- 対応フォーマット: `.onnx`, `.pt`
- ResNet8 CIFAR-10 のアーキテクチャを前提としたレイヤー名の表示
- ローカル環境 (Linux) での使用を前提

---

## 6. UIレイアウト

```
┌─────────────────────────────────────────────────────────────────┐
│              Neural Network Weight Visualizer                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model: [resnet8_int8.pt ▼]                                    │
│  Layer: [model_1/conv2d_1/BiasAdd ▼]   Tensor: [weight ▼]     │
│                                                                 │
│  ┌── Quantization View ──────────────────────────────┐         │
│  │ (FP32モデルでは非表示 / 量子化モデルのみ表示)       │         │
│  │  View: ○ int8 raw values  ● dequantized (FP32)    │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                 │
│  Bins: ──────●────── 50                                         │
│                                                                 │
│  ┌─ Histogram ──────────────────────────────────────────────┐   │
│  │                                                           │   │
│  │    ▐█▌                                                    │   │
│  │   ▐███▌         Plotly インタラクティブヒストグラム        │   │
│  │  ▐█████▌        - ホバーで値表示                          │   │
│  │ ▐███████▌       - パーセンテージラベル                    │   │
│  │▐█████████▌      - 範囲指定時: オレンジ/グレーで表示       │   │
│  │█████████████                                              │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─ Value Range Analysis ────────────────────────────────────┐  │
│  │  Min: [-0.270923]  Max: [0.266250]  [Apply Range]         │  │
│  │                                                           │  │
│  │  (Apply後) Selected Range: [-0.270923, 0.266250]          │  │
│  │  Count in range: 287 / 432                                │  │
│  │  Percentage: 66.44%                                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Statistics ──────────────────────────────────────────────┐  │
│  │  Min: -0.764241    Max: 0.904751                          │  │
│  │  Mean: -0.002337    Std: 0.268575                         │  │
│  │  Shape: (16, 3, 3, 3)    Total params: 432                │  │
│  │  Scale: 0.004477    Zero Point: 0.0                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 実際のモデルデータ (実行結果)

以下は `resnet8.pt` (FP32) および `resnet8_int8.pt` (INT8 TorchScript) から実際に抽出したデータである。

### 7.1 FP32 モデル (`resnet8.pt`) レイヤー一覧

| レイヤー | テンソル | Shape | Params | Min | Max | Mean | Std |
|---------|---------|-------|--------|-----|-----|------|-----|
| conv2d_1/BiasAdd | weight | (16, 3, 3, 3) | 432 | -0.7645 | 0.9083 | -0.0021 | 0.2687 |
| conv2d_1/BiasAdd | bias | (16,) | 16 | -0.0056 | 0.0009 | -0.0007 | 0.0016 |
| conv2d_1_2/BiasAdd | weight | (16, 16, 3, 3) | 2,304 | -0.6771 | 0.5802 | -0.0041 | 0.1295 |
| conv2d_1_2/BiasAdd | bias | (16,) | 16 | -0.0599 | 0.1370 | 0.0101 | 0.0469 |
| conv2d_2_1/BiasAdd | weight | (16, 16, 3, 3) | 2,304 | -0.5411 | 0.5084 | -0.0005 | 0.1172 |
| conv2d_3_1/BiasAdd.1 | weight | (32, 16, 3, 3) | 4,608 | -0.4830 | 0.3758 | -0.0060 | 0.0998 |
| conv2d_4_1/BiasAdd | weight | (32, 32, 3, 3) | 9,216 | -0.4152 | 0.3844 | -0.0041 | 0.0865 |
| conv2d_5_1/BiasAdd | weight | (32, 16, 1, 1) | 512 | -0.4724 | 0.5948 | 0.0139 | 0.1338 |
| conv2d_6_1/BiasAdd.1 | weight | (64, 32, 3, 3) | 18,432 | -0.2789 | 0.3226 | -0.0047 | 0.0693 |
| conv2d_7_1/BiasAdd | weight | (64, 64, 3, 3) | 36,864 | -0.2503 | 0.2733 | -0.0021 | 0.0581 |
| conv2d_8_1/BiasAdd | weight | (64, 32, 1, 1) | 2,048 | -0.6925 | 0.5840 | -0.0271 | 0.1356 |

**合計パラメータ数 (weight のみ)**: 76,732

### 7.2 INT8 量子化モデル (`resnet8_int8.pt`) レイヤー一覧

| レイヤー | Shape | Scale | ZP | INT8 Range | FP32 Range | FP32 Mean | FP32 Std |
|---------|-------|-------|-----|-----------|-----------|----------|---------|
| conv2d_1/BiasAdd | (16, 3, 3, 3) | 0.004477 | 0.0 | [-128, 127] | [-0.7642, 0.9048] | -0.0023 | 0.2686 |
| conv2d_1_2/BiasAdd | (16, 16, 3, 3) | 0.003202 | 0.0 | [-128, 127] | [-0.6797, 0.5788] | -0.0042 | 0.1295 |
| conv2d_2_1/BiasAdd | (16, 16, 3, 3) | 0.003032 | 0.0 | [-128, 127] | [-0.5432, 0.5065] | -0.0005 | 0.1173 |
| conv2d_3_1/BiasAdd.1 | (32, 16, 3, 3) | 0.002398 | 0.0 | [-128, 127] | [-0.4849, 0.3750] | -0.0060 | 0.0998 |
| conv2d_4_1/BiasAdd | (32, 32, 3, 3) | 0.002169 | 0.0 | [-128, 127] | [-0.4168, 0.3842] | -0.0041 | 0.0865 |
| conv2d_5_1/BiasAdd | (32, 16, 1, 1) | 0.002277 | 0.0 | [-128, 127] | [-0.4742, 0.5925] | 0.0138 | 0.1337 |
| conv2d_6_1/BiasAdd.1 | (64, 32, 3, 3) | 0.001749 | 0.0 | [-128, 127] | [-0.2800, 0.3214] | -0.0047 | 0.0693 |
| conv2d_7_1/BiasAdd | (64, 64, 3, 3) | 0.001554 | 0.0 | [-128, 127] | [-0.2513, 0.2722] | -0.0021 | 0.0581 |
| conv2d_8_1/BiasAdd | (64, 32, 1, 1) | 0.002705 | 0.0 | [-128, 127] | [-0.6952, 0.5866] | -0.0271 | 0.1357 |

### 7.3 量子化精度の観察

全レイヤーで per-channel symmetric quantization (zero_point = 0.0) が適用されている。

FP32 → INT8 の値変化:
- **conv2d_1**: FP32 range [-0.7645, 0.9083] → INT8 dequantized [-0.7642, 0.9048] (誤差 < 0.004)
- **conv2d_7_1**: FP32 range [-0.2503, 0.2733] → INT8 dequantized [-0.2513, 0.2722] (誤差 < 0.002)
- 全レイヤーで scale 値は 0.001〜0.005 の範囲に収まっており、量子化精度は良好

---

## 8. 実行方法

```bash
# 起動
marimo edit playground/weight_visualizer.py

# または読み取り専用モード
marimo run playground/weight_visualizer.py
```

## 9. 依存関係

`pyproject.toml` に以下が追加済み:
- `marimo>=0.15.5`
- `plotly>=5.0.0`

既存の依存関係:
- `torch`
- `numpy`
- `onnx` (ONNX モデル使用時)
