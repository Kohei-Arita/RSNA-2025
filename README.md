# RSNA-2025 — Intracranial Aneurysm Detection（Colab-First / Kaggle Notebooks Only 対応）

本プロジェクトは研究=Google Colab、提出=Kaggle Notebooks Only（インターネット遮断・最大実行時間≒12時間）の二層設計を前提とします。以降の手順・コマンドは Colab 上での利用（Drive 連携／Kaggle API／Weights & Biases）と、Kaggle 上でのオフライン推論を想定しています。Colab の Drive マウントや入出力の流れは Colab 公式ノートブック「Local Files, Drive, Sheets, and Cloud Storage」を基準にしています。

## 目次

- [プロジェクト概要](#プロジェクト概要)
- [リポジトリ構成](#リポジトリ構成)
- [前提条件](#前提条件)
- [Colab セットアップ手順（初回）](#colab-セットアップ手順初回)
- [データ取得（Kaggle API → DVC/Drive 連携）](#データ取得kaggle-api--dvc-drive-連携)
- [EDA（探索的データ分析）](#eda探索的データ分析)
- [学習（Hydra/CLI + W&B ロギング）](#学習hydra-cli--wb-ロギング)
- [推論・OOF・可視化](#推論oof可視化)
- [提出（Kaggle Notebooks Only / オフライン）](#提出kaggle-notebooks-only--オフライン)
- [提出（Submission 自動化）](#提出submission-自動化)
- [実験管理ポリシー](#実験管理ポリシー)
- [Make タスク](#make-タスク)
- [トラブルシュート](#トラブルシュート)
- [ライセンス](#ライセンス)

## プロジェクト概要

- **目的**：RSNA Intracranial Aneurysm Detection コンペにおける管理しやすさ最優先の実験基盤。
- **実行**：Google Colab（GPU 推奨）。
- **追跡**：Weights & Biases（実験ロギング、アーティファクト管理の補助）。
- **データ／成果物**：DVC + Google Drive remote でデータ版管理、submissions/ に提出物を一元管理。
- **コンフィグ**：Hydra によるグループ分割・defaults 合成・マルチラン対応。

## リポジトリ構成

このリポジトリは Cookiecutter-Data-Science の思想に沿った標準的な構成で、Hydra・DVC・W&B を組み合わせたワークフローを採用しています。

**outputs/ と reports/ の境界**：
- **outputs/** : 再生成可能・中間生成物（学習済みモデル、予測結果、キャッシュなど）
- **reports/** : 永続共有・論文/発表用図表（最終レポート、共有用可視化、プレゼン資料など）

### ディレクトリ構成

```
RSNA-2025/
├── .github/                    # GitHub設定・テンプレート
│   ├── CODEOWNERS             # コードオーナー設定
│   ├── CONTRIBUTING.md        # コントリビューションガイド
│   ├── ISSUE_TEMPLATE/        # Issue テンプレート
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── SECURITY.md            # セキュリティポリシー
│
├── configs/                   # Hydra設定ファイル群
│   ├── config.yaml           # メイン設定・デフォルト値
│   ├── aug/                  # データ拡張設定
│   │   ├── light.yaml        # 軽量拡張（高速実験用）
│   │   ├── medium.yaml       # 中程度拡張（バランス型）
│   │   └── heavy.yaml        # 重拡張（最終精度向上用）
│   ├── cv/                   # 交差検証設定
│   │   ├── patient_kfold.yaml # 患者単位のKFold
│   │   ├── groupkfold.yaml   # グループKFold
│   │   └── seeds.yaml        # シード・分割設定
│   ├── data/                 # データセット設定
│   │   ├── rsna.yaml         # RSNA公式データ設定
│   │   ├── cache.yaml        # キャッシュ戦略設定
│   │   └── external.yaml     # 外部データ設定
│   ├── model/                # モデルアーキテクチャ設定
│   │   ├── baseline_2d.yaml  # 2Dベースライン（CNN）
│   │   ├── efficientnet.yaml # EfficientNet系
│   │   ├── convnext.yaml     # ConvNeXt系
│   │   ├── vit.yaml          # Vision Transformer
│   │   └── three_d_cnn.yaml  # 3D CNN（ボリューム処理）
│   ├── train/                # 学習設定
│   │   ├── base.yaml         # 基本学習設定
│   │   ├── fp16.yaml         # 混合精度学習
│   │   ├── swa.yaml          # Stochastic Weight Averaging
│   │   ├── earlystop.yaml    # Early Stopping設定
│   │   └── tta.yaml          # Test Time Augmentation
│   ├── inference/            # 推論設定
│   │   ├── base.yaml         # 基本推論設定
│   │   └── export.yaml       # モデルエクスポート設定
│   └── paths/                # 環境別パス設定
│       ├── local.yaml        # ローカル環境
│       ├── colab.yaml        # Google Colab環境
│       └── kaggle.yaml       # Kaggle Notebook環境
│
├── data/                     # データディレクトリ（DVC管理）
│   ├── raw/                  # 生データ（Kaggle取得）
│   ├── interim/              # 中間処理データ
│   ├── processed/            # 前処理済みデータ
│   └── external/             # 外部データ（追加データセット等）
│
├── src/rsna_aneurysm/        # メインソースコード
│   ├── __init__.py
│   ├── cli.py                # CLIエントリーポイント（Hydra）
│   ├── datamodule.py         # Lightning DataModule
│   ├── dataset.py            # Dataset実装（DICOM処理）
│   ├── dicom_utils.py        # DICOM読み込み・前処理
│   ├── model.py              # Lightning Module（学習ループ）
│   ├── loss.py               # 損失関数（Focal Loss等）
│   ├── metrics.py            # 評価指標（AUC、Sensitivity等）
│   ├── transforms.py         # データ拡張・前処理
│   ├── optimizer.py          # オプティマイザー設定
│   ├── scheduler.py          # 学習率スケジューラー
│   ├── inference.py          # 推論・予測処理
│   ├── oof_utils.py          # Out-of-Fold予測・評価
│   ├── postprocess.py        # 後処理（アンサンブル等）
│   ├── visualization.py      # 可視化（Grad-CAM、学習曲線）
│   ├── registry.py           # モデル・コンポーネント登録
│   └── utils.py              # ユーティリティ関数
│
├── experiments/              # 実験管理（1ディレクトリ=1実験）
│   └── exp0001_baseline/     # 実験例：ベースライン
│       ├── config.yaml       # 実験時のHydra合成設定（再現性）
│       ├── training.ipynb    # 学習実行ノートブック
│       ├── evaluation.ipynb  # 評価・OOF生成ノートブック
│       ├── inference.ipynb   # 推論実行ノートブック
│       └── notes.md          # 実験メモ・W&Bリンク・振り返り
│
├── notebooks/                # 探索・分析用Jupyter Notebook
│   ├── 00_eda.ipynb         # 探索的データ分析（EDA）
│   ├── 01_error_analysis.ipynb # エラー分析・失敗例調査
│   ├── 99_playground.ipynb   # 自由実験・プロトタイプ
│   └── README.md             # Notebook使用ガイド
│
├── outputs/                  # 再生成可能な出力（.gitignore対象）
│   ├── oof/                  # Out-of-Fold予測結果
│   └── preds/                # テストセット予測結果
│
├── models/                   # 学習済みモデル（DVC/W&B管理）
│
├── reports/                  # 永続的な成果物・共有資料
│   └── figures/              # 論文・発表用図表
│
├── submissions/              # Kaggle提出ファイル
│
├── tests/                    # 単体テスト・統合テスト
│   ├── test_dataset.py       # データセットテスト
│   ├── test_transforms.py    # 前処理テスト
│   ├── test_metrics.py       # 評価指標テスト
│   └── test_inference.py     # 推論テスト
│
├── tools/                    # ユーティリティスクリプト
│   ├── submit.py             # Kaggle提出自動化
│   ├── export_onnx.py        # ONNXエクスポート
│   ├── kaggle_sync.py        # Kaggleデータ同期
│   ├── sweep_wandb.py        # W&Bスイープ実行
│   └── seed_everything.py    # 再現性確保
│
├── docs/                     # プロジェクト関連ドキュメント
│   ├── colab_setup.md        # Colab環境構築詳細手順
│   ├── experiment_workflow.md # 実験ワークフロー解説
│   ├── dvc_remote.md         # DVC remote設定ガイド
│   └── DATASET_CARD.md       # データセット詳細・制約
│
├── env/                      # 環境・依存関係管理
│   ├── requirements.txt      # Colab用依存関係（配布用）
│   ├── requirements.lock     # ロックファイル（再現性）
│   └── colab_setup.ipynb     # Colab初期設定ノートブック
│
├── .kaggle/                  # Kaggle API設定
│   └── kaggle.json.example   # API認証ファイル例
│
├── dvc.yaml                  # DVCパイプライン定義
├── dvc.lock                  # DVCロックファイル
├── dvc.config.example        # DVC設定例（Google Drive）
├── pyproject.toml            # Python依存関係・メタデータ（真実源）
├── Makefile                  # 開発タスク自動化
└── README.md                 # このファイル
```

### 主要ファイルの役割

**設定管理**：
- `configs/config.yaml` : Hydraのメイン設定、全体のデフォルト値
- `pyproject.toml` : Python依存関係の真実源、プロジェクトメタデータ
- `dvc.yaml` : データパイプライン定義、再現性確保

**実行エントリーポイント**：
- `src/rsna_aneurysm/cli.py` : メインCLI（学習・推論・評価の統一インターフェース）
- `tools/submit.py` : Kaggle提出自動化スクリプト

**実験管理**：
- `experiments/expXXXX/` : 各実験の設定・ノートブック・メモを一元管理
- `notebooks/` : EDA・分析・プロトタイプ用途

## 前提条件

- Google アカウント（Colab と Drive 利用）
- Kaggle アカウント & Kaggle API トークン（kaggle.json）
- Weights & Biases アカウント & API Key（wandb login）

## Colab セットアップ手順（初回）

この章は Colab 専用です。 Drive マウント、依存関係の導入、シークレット投入までを 1 セルずつ実施します。

### GPU 確認（任意）

```python
!nvidia-smi || true
```

### Google Drive をマウント
Colab 公式の方法に従います。マウント後は `/content/drive/MyDrive` が利用可能です。

```python
from google.colab import drive
drive.mount('/content/drive')
```

### リポジトリを取得（例：あなたの GitHub にミラー済みの想定）

```bash
%cd /content
!git clone https://github.com/<your-org-or-user>/RSNA-2025.git
%cd RSNA-2025
```

### Python 依存の導入
本プロジェクトでは pyproject.toml を真実源 とし、Colab 用には env/requirements.txt（自動生成物） を配布しています。Colab ではこれを直接インストールします。

```bash
!pip install -U pip
!pip install -r env/requirements.txt
```

### W&B ログイン（対話）
W&B のクイックスタートに従い、wandb login で API Key を貼り付けます。

```bash
!wandb login
# ブラウザからコピーした API Key を貼り付け
```

### Kaggle API のセットアップ
Kaggle ドキュメントに従い kaggle.json を配置（Colab では ~/.kaggle/ が手早い）。権限設定も必要です。

```bash
!mkdir -p ~/.kaggle
# kaggle.json をファイルアップロードして /content/kaggle.json に置く例
!cp /content/kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle --version
```

### DVC + Google Drive remote 準備（任意・推奨）
DVC 公式の手順に沿って gdrive 連携を使えます（dvc_gdrive が必要）。初回は OAuth 認可が入る場合があります。

```bash
!pip install "dvc[gdrive]"
# 例：サンプルの dvc.config をベースにユーザ環境用を作成
!cp dvc.config.example dvc.config
# 初回 pull（リモート URL は dvc.config で指す）
!dvc pull
```

💡 DVC の Google Drive 認可でブロック表示が出るケースがある旨は公式に注記があります（ワークアラウンドあり）。困った場合は該当ページを参照してください。

## フロー概観（Colab=研究 / Kaggle=提出）

- 研究（Colab）: データ取得→EDA→学習→評価（W&B可）
- 提出（Kaggle）: 前計算/重みの Dataset を追加→最小推論→CSV 提出（オフライン）
- 切替（例）: `paths=colab` / `paths=kaggle wandb=disabled inference=kaggle_fast`

## データ取得（Colab）

- Kaggle API で公式データ取得（既存手順）
- DVC remote（任意）でチーム共有

```bash
# 公式データ（概要、詳細は上のセクション参照）
kaggle competitions download -c rsna-intracranial-aneurysm-detection -p data/raw
unzip -q data/raw/rsna-intracranial-aneurysm-detection.zip -d data/raw
# DVC（任意）
dvc pull
```

## EDA（Colab）

- `notebooks/00_eda.ipynb` を使用
- 入出力は `configs/data/*.yaml` を尊重（コメントで方針のみ）

## 学習（Colab, Hydra/CLI）

- 基本: `python -m rsna_aneurysm.cli train ... paths=colab`
- W&B ログは有効（必要なら `wandb=disabled` で停止可）

```bash
python -m rsna_aneurysm.cli train \
  model=baseline_2d cv=patient_kfold train=base,fp16 data=rsna \
  paths=colab wandb.project=RSNA2025
```

## 推論・評価（Colab）

- OOF/可視化は既存ノート/スクリプトを利用
- 提出用推論は Kaggle 側に最小化して分離

```bash
python -m rsna_aneurysm.cli infer \
  inference=base model=baseline_2d paths=colab \
  checkpoint_path=models/exp0001_fold0_best.ckpt
```

## 提出準備（前計算と重みの固定, Colab もしくはローカル）

- 前計算を梱包（スケルトン）: `make kaggle-prep`
- 学習済み重みを `dist/rsna2025-weights/` に集約（手動/スクリプト）
- それぞれ Kaggle Dataset として登録（Private で可）

```bash
make kaggle-prep  # dist/rsna2025-precompute/ を生成（現状は雛形）
```

## 提出（Kaggle Notebooks Only / オフライン）

- Notebook: `kaggle/notebook_template.ipynb` をアップロード
- 「Add data」で `rsna2025-precompute` と `rsna2025-weights` を追加
- `kaggle/kaggle_infer.py` を実行 → `/kaggle/working/submission.csv`
- 軽量検証（スケルトン）: `tools/verify_submission.py`

```bash
# Kaggle 環境（概念図）
python kaggle/kaggle_infer.py  # 実装後、submission.csv を生成
```

### オフラインpip（wheel / Add data）

```bash
# 依存が標準コンテナに無い場合のみ。wheel を事前に収集して Dataset 化（例: rsna2025-wheels）
pip install --no-index --find-links /kaggle/input/rsna2025-wheels -r kaggle/offline_requirements.txt
python - <<'PY'
import torch, os
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
PY
```

### 時間ガード（推論用の自動ダウングレード指針）

- 概算 ETA が上限（≒12h）を超えそうな場合に優先順位で無効化/粗化
  - TTA 停止 → パッチストライドを粗く → 候補数上限を縮小 → 入力解像度を縮小
- “完走最優先” を原則とし、ダウングレードの切替は `kaggle_infer.py` 内で実装（コメント済、後続で実装）

### CSV 検証強化（軽量）

```bash
python tools/verify_submission.py /kaggle/working/submission.csv
```

- 列名・dtype・NaN/Inf を検査
- スコア/座標の値域・ID 形式・重複行を検出
- 仕様確定後に `tools/verify_submission.py` 内コメントへ反映（実装は後続）

## ローカル乾式リハーサル

```bash
make kaggle-dryrun  # .work/submission.csv を生成（現状は空の雛形）
```

## 変更点（この改訂で追加された骨組み）

- `configs/paths/kaggle.yaml`, `configs/wandb/disabled.yaml`, `configs/inference/kaggle_fast.yaml`（Kaggle 12h 完走前提の軽量設定）
- `kaggle/` ディレクトリ（README_KAGGLE, kaggle_infer.py, kaggle_utils.py, notebook_template.ipynb, offline_requirements.txt）
- `tools/pack_precompute.py`（再サンプル/脳マスク/候補点の前計算梱包スケルトン）, `tools/verify_submission.py`（提出検証スケルトン）
- `tests/test_dicom_geometry.py`（現状 skip）
- `Makefile` の kaggle タスク（prep/dryrun/wheels）

## 次ステップ（実装方針だけ明記）

- kaggle_infer: 候補点ロード→3Dパッチ切り出し→モデル推論→NMS→CSV
- pack_precompute: 再サンプル/脳マスク/候補点の形式確定→梱包
- verify_submission: 列名/dtype/NaN・Inf/重複/値域の軽量検証

## 実験管理ポリシー

### 1 ディレクトリ = 1 実験（experiments/expXXXX）：

- **config.yaml**：実験時点の Hydra 合成結果のスナップショット（再現性確保）
- **training.ipynb / evaluation.ipynb / inference.ipynb**：CLI 呼び出しのセル化
- **notes.md**：所感・W&B リンク・次アクション

### outputs/ と reports/ の境界：

- **再生成可能・中間生成物＝outputs/**
- **永続共有・論文/発表用図表＝reports/**

（※この方針を README 冒頭でも明記）

## Make タスク

Colab では make が無い場合があるため、使わなくても進められるよう全コマンドを README に記載しています。make を使える場合は以下の簡略化が可能。

- **make train**：代表的な学習ジョブを実行
- **make infer**：推論実行
- **make submit**：提出 CSV 作成 → Kaggle 提出
- **make deps-freeze**：ローカルで pyproject.toml から env/requirements.txt を再生成（配布は生成物）

## トラブルシュート

- **Drive マウントがうまくいかない**：Colab 公式の I/O ノートブックの手順を再確認。権限ダイアログの再実行で解消することが多いです。
- **Kaggle API で 401/403**：~/.kaggle/kaggle.json の配置と chmod 600（権限）を確認。Kaggle 設定ページから再発行も有効です。
- **DVC × Google Drive で認可エラー／ブロック表示**：DVC 公式の gdrive remote ページにワークアラウンド記載あり。dvc remote modify の各種フラグ（gdrive_acknowledge_abuse, サービスアカウント使用 など）も検討。
- **Hydra のマルチランで組合せを制御したい**：hydra-filter-sweeper や List Sweeper で探索空間を制限可能。

## ライセンス

LICENSE を参照してください。

## 付録：よく使うコマンド集（Colab）

```bash
# 0) 準備
pip install -r env/requirements.txt
wandb login
mkdir -p ~/.kaggle && cp /content/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 1) データ取得（公式データ）
kaggle competitions download -c rsna-intracranial-aneurysm-detection -p data/raw
unzip -q data/raw/rsna-intracranial-aneurysm-detection.zip -d data/raw

# 2) DVC 同期（必要に応じて）
dvc pull   # 既存データ/加工物の取得
# dvc add data/processed && dvc push   # 新規の追加例

# 3) 学習（例）
python -m rsna_aneurysm.cli train model=baseline_2d cv=patient_kfold train=base,fp16 data=rsna paths=colab wandb.project=RSNA2025

# 4) 推論（例）
python -m rsna_aneurysm.cli infer inference=base model=baseline_2d paths=colab checkpoint_path=models/exp0001_fold0_best.ckpt

# 5) 提出（例）
python tools/submit.py --input outputs/preds/exp0001_fold-avg.csv --output submissions/exp0001.csv
kaggle competitions submit -c rsna-intracranial-aneurysm-detection -f submissions/exp0001.csv -m "exp0001"
```

## 参考

- [Kaggle API（Public API / GitHub README）](https://github.com/Kaggle/kaggle-api)
- [W&B Quickstart / Colab 例](https://docs.wandb.ai/quickstart)
- [DVC × Google Drive（remote 設定）](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive)
- [Hydra 基本のオーバライド構文](https://hydra.cc/docs/advanced/override_grammar/basic/)

## Kaggle Notebooks Only 対応（研究=Colab / 提出=Kaggle の二層設計）

> 最低限の案内のみ。実装詳細は後続ステップで追加予定。

- 切替例（Hydra）: `paths=kaggle wandb=disabled inference=kaggle_fast`
- 追加ファイル（骨組み）:
  - `configs/paths/kaggle.yaml` / `configs/wandb/disabled.yaml`
  - `configs/inference/kaggle_fast.yaml`
  - `kaggle/` ディレクトリ（README_KAGGLE, kaggle_infer.py, kaggle_utils.py, notebook_template.ipynb, offline_requirements.txt）
  - `tools/pack_precompute.py`（前計算梱包・スケルトン）
  - `tools/verify_submission.py`（提出検証・スケルトン）
  - `tests/test_dicom_geometry.py`（幾何テスト・後続で実装）
- Make タスク:
  - `make kaggle-prep` / `make kaggle-dryrun` / `make wheels`

TODO:
- 推論本体（パッチ推論・NMS・CSV出力）を `kaggle/kaggle_infer.py` に実装
- 前計算仕様（フォーマット/項目）と `tools/pack_precompute.py` の実装
- 検証ツール・幾何テストの具体化