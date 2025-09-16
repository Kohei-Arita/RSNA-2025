# RSNA-2025 — Intracranial Aneurysm Detection（Colab-First / Kaggle Notebooks Only 対応）

本プロジェクトは研究=Google Colab、提出=Kaggle Notebooks Only（Internet: Off／実行時間上限の公式値: CPU/GPU ≤ 12時間, TPU ≤ 9時間。設計上は安全側で9時間以内完走を目標）の二層設計に、Google Cloud Storage（GCS）を真実源とする三層構成（Colab=研究/学習、GCS=データ＆成果物ストア、Kaggle=オフライン提出）を採用します。以降の手順・コマンドは Colab 上での利用（GCS 連携／Kaggle API／Weights & Biases）と、Kaggle 上でのオフライン推論を前提にしています。DVC のリモートは GCS（gs://...）を前提とし、Colab からは Google Cloud の認証（ADC またはサービスアカウント）でアクセスします。

## 目次

- [プロジェクト概要](#プロジェクト概要)
- [提出要件（重要）](#提出要件重要)
- [リポジトリ構成](#リポジトリ構成)
- [前提条件](#前提条件)
- [Colab セットアップ手順（初回）](#colab-セットアップ手順初回)
- [データ取得（Kaggle API → DVC/GCS 連携）](#データ取得kaggle-api--dvcgcs-連携)
- [データセット運用（配置・同期・Kaggle搬入）](#データセット運用配置同期kaggle搬入)
- [EDA（探索的データ分析）](#eda探索的データ分析)
- [学習（Hydra/CLI + W&B ロギング）](#学習hydra-cli--wb-ロギング)
- [推論・OOF・可視化](#推論oof可視化)
- [提出（サービングAPI / Notebook-Only）](#提出サービングapi--notebook-only)
- [実験管理ポリシー](#実験管理ポリシー)
- [Make タスク](#make-タスク)
- [トラブルシュート](#トラブルシュート)
- [ライセンス](#ライセンス)

> 運用メモ: 進捗・計画に更新があれば、必ずリポジトリ直下の `PROGRESS_CHECKLIST.md` を更新してください（最新エントリを先頭に追記）。

## プロジェクト概要

- **目的**：RSNA Intracranial Aneurysm Detection コンペにおける管理しやすさ最優先の実験基盤。
- **実行**：Google Colab（GPU 推奨）。
- **追跡**：Weights & Biases（実験ロギング、アーティファクト管理の補助）。
- **データ／成果物**：DVC + GCS（Google Cloud Storage）remote でデータ版管理、submissions/ に提出物を一元管理。
- **コンフィグ**：Hydra によるグループ分割・defaults 合成・マルチラン対応。

## 提出要件（重要）

- Kaggle Notebooks Only：提出 Notebook は「Internet: Off」で保存・実行する（オンライン取得・外部通信は禁止）。
- 実行時間上限の目安：CPU/GPU ≤ 12時間、TPU ≤ 9時間（Code要件）。推論は `configs/inference/kaggle_fast.yaml` の `time_budget_hours` を基準に自動ダウングレード（解像度→TTA→stride→候補数）。

本READMEは、当該コンペがサービングAPIを前提とする Notebooks Only 形式であることを前提に記述しています。時間上限は公式値（CPU/GPU ≤ 12時間、TPU ≤ 9時間）を主としつつ、設計目標は安全側で9時間以内完走とします。

- 本番提出は評価APIのみ（シリーズ単位に14確率を返答）。CSV 提出は不要で、CSV はローカルDry-run専用ツール（`tools/verify_submission.py`）のみで使用する。

（注）時間上限は運営の告知や時点の仕様で変動することがあるため目安として記載。最新情報は各コンペのルール/フォーラムを確認。

## 評価指標（公式）

- 指標：Mean Weighted Columnwise AUCROC（列ごとの AUC を重み付き平均）
- 重み：`aneurysm_present` に 13、各部位ラベルに 1（合計 26）。README 上では「重み付きAUC」と表現しつつ、実際の重み比 13:1 を明記して混乱を避ける。
- 実務上の示唆：presence の寄与が大きいため、presence ヘッドの損失・校正を重視（例：`configs/train/presence_calibration.yaml`、温度スケーリング等）。

## リポジトリ構成

このリポジトリは Cookiecutter-Data-Science の思想に沿った標準的な構成で、Hydra・DVC・W&B を組み合わせたワークフローを採用しています。

**outputs/ と reports/ の境界**：
- **outputs/** : 再生成可能・中間生成物（学習済みモデル、予測結果、キャッシュなど）
- **reports/** : 永続共有・論文/発表用図表（最終レポート、共有用可視化、プレゼン資料など）

### ディレクトリ構成

（注）以下は現時点の構成（抜粋）です。追加予定や派生物は後段の「変更点」を参照してください。

```
RSNA-2025/
├── .github/                    # GitHub設定・テンプレート
│   ├── CODEOWNERS             # コードオーナー設定
│   ├── CONTRIBUTING.md        # コントリビューションガイド
│   ├── ISSUE_TEMPLATE/        # Issue テンプレート
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── SECURITY.md            # セキュリティポリシー
│
├── .gcloud/                  # GCS 認証関連（例のみ、実鍵は追跡しない）
│   ├── service-account.example.json   # サービスアカウント鍵の例（コメントのみ）
│   └── README.md                     # 認証手順メモ（ADC/SA, コメントのみ）
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
│   │   ├── three_d_cnn.yaml  # 3D CNN（ボリューム処理）
│   │   └── two_point_five_d.yaml # 2.5D（スライス+時系列）
│   ├── train/                # 学習設定
│   │   ├── base.yaml         # 基本学習設定
│   │   ├── fp16.yaml         # 混合精度学習
│   │   ├── swa.yaml          # Stochastic Weight Averaging
│   │   ├── earlystop.yaml    # Early Stopping設定
│   │   ├── tta.yaml          # Test Time Augmentation
│   │   └── presence_calibration.yaml # presence 校正・しきい値関連
│   ├── inference/            # 推論設定
│   │   ├── base.yaml         # 基本推論設定
│   │   ├── export.yaml       # モデルエクスポート設定
│   │   ├── kaggle_fast.yaml  # Kaggle向け軽量・時間ガード設定
│   │   └── modality_thresholds.yaml # モダリティ別しきい値
│   ├── wandb/                # ロギング設定
│   │   └── disabled.yaml     # Kaggle向け無効化設定
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
├── kaggle/                   # Kaggle サービング/ノートブック資材
│   ├── kaggle_infer.py       # サービングAPIエントリポイント
│   ├── kaggle_utils.py       # 付随ユーティリティ
│   ├── notebook_template.ipynb
│   └── offline_requirements.txt
│
├── submissions/              # Kaggle提出ファイル（サービングAPI移行によりCSVはDry-run用途）
│
├── tests/                    # 単体テスト・統合テスト
│   ├── test_dataset.py       # データセットテスト
│   ├── test_transforms.py    # 前処理テスト
│   ├── test_metrics.py       # 評価指標テスト
│   └── test_inference.py     # 推論テスト
│
├── tools/                    # ユーティリティスクリプト
│   ├── export_onnx.py        # ONNXエクスポート
│   ├── kaggle_sync.py        # Kaggleデータ同期
│   ├── sweep_wandb.py        # W&Bスイープ実行
│   └── seed_everything.py    # 再現性確保
│
├── docs/                     # プロジェクト関連ドキュメント
│   ├── README_ARCH.md        # GCS三層設計の要旨（コメント）
│   ├── colab_setup.md        # Colab環境構築（GCS認証のコメント追記）
│   ├── experiment_workflow.md # 実験ワークフロー解説
│   ├── dvc_remote_gcs.md     # DVC×GCS 連携手順（コメント）
│   ├── kaggle_offline.md     # Kaggle Internet: Off 運用（コメント）
│   ├── DATASET_CARD.md       # データセット詳細・制約
│   └── SUBMISSION_CONTRACT.md # 提出仕様
│
├── env/                      # 環境・依存関係管理
│   ├── requirements.txt      # Colab用依存関係（配布用）
│   ├── requirements.lock     # ロックファイル（再現性）
│   └── colab_setup.ipynb     # Colab初期設定ノートブック
│
├── .kaggle/                  # Kaggle API設定
│   └── kaggle.json.example   # API認証ファイル例
│
├── gcs/                      # GCS固有の運用補助（コメントのみ）
│   ├── buckets.example.yaml      # バケット構成例（コメント）
│   ├── iam.bindings.example.yaml # ロール付与の雛形（コメント）
│   ├── acl_preset.json           # ACL プリセット例（コメント）
│   ├── sync_patterns.txt         # 軽量同期パターン（コメント）
│   └── README.md                 # 補足メモ（コメント）
│
├── infra/                    # 任意（IaC: Terraform 等, コメントのみ）
│   └── terraform/
│       ├── main.tf
│       ├── variables.tf
│       ├── outputs.tf
│       ├── providers.tf
│       └── terraform.tfvars.example
│
├── scripts/                  # 認証/初期化スクリプト（コメントのみ）
│   ├── colab_auth_gcs.sh
│   ├── dvc_init_gcs.sh
│   ├── gcsfuse_mount_colab.sh
│   └── verify_env.py
│
├── dvc.yaml                  # DVCパイプライン定義
├── dvc.lock                  # DVCロックファイル
├── dvc.config.example        # DVC設定例（GCS remote の例, コメント）
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
- `kaggle/kaggle_infer.py` : Kaggle サービングAPIのエントリポイント（起動→初期化→serve）

**実験管理**：
- `experiments/expXXXX/` : 各実験の設定・ノートブック・メモを一元管理
- `notebooks/` : EDA・分析・プロトタイプ用途

## 前提条件

- Google アカウント（Colab 利用）＋ GCP プロジェクト/権限（GCS アクセス）
- Kaggle アカウント & Kaggle API トークン（kaggle.json）
- Weights & Biases アカウント & API Key（wandb login）

※ 外部データ・事前学習重みの利用可否は各大会の Rules が最優先です。使用する場合は出所・ライセンス・再現手順を `docs/DATASET_CARD.md` に一度だけ明記してください（本 README からも参照）。

## Colab セットアップ手順（初回）

この章は Colab 専用です。 GCS 認証（ADC/サービスアカウント）、依存関係の導入、シークレット投入までを 1 セルずつ実施します。

### GPU 確認（任意）

```python
!nvidia-smi || true
```

### Google Cloud 認証（ADC 推奨 / SA でも可）
```python
from google.colab import auth
auth.authenticate_user()  # ADC を有効化（対話許可）
```
（コメント）サービスアカウント鍵を使う場合の例：
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/content/RSNA-2025/.gcloud/service-account.json"
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

### DVC + GCS remote 準備（推奨）
（コメント）DVC 公式手順に準拠。リモートは gs:// をデフォルトに設定します。
```bash
!pip install "dvc[gcs]"
!dvc init
!dvc remote add -d gcsremote gs://<your-gcs-bucket>/rsna2025
!dvc remote modify gcsremote credentialpath .gcloud/service-account.json  # SA 使用時の例
!dvc pull
```

## フロー概観（Colab=研究 / Kaggle=提出）

- 研究（Colab）: データ取得→EDA→学習→評価（W&B可）
- 提出（Kaggle）: 前計算/重みの Dataset を追加→サーバ起動（起動後15分以内に初期化完了）→評価APIの逐次配信に応答（シリーズごとに14確率）
- 切替（例）: `paths=colab` / `paths=kaggle wandb=disabled inference=kaggle_fast`

## データ取得（Kaggle API → DVC/GCS 連携）

- Kaggle API で公式データ取得（既存手順）
- DVC remote（任意）でチーム共有

```bash
# 公式データ（概要、詳細は上のセクション参照）
kaggle competitions download -c rsna-intracranial-aneurysm-detection -p data/raw
unzip -q data/raw/rsna-intracranial-aneurysm-detection.zip -d data/raw
# DVC（任意）
dvc pull
```

## データセット運用（配置・同期・Kaggle搬入）

### 方針（公式データは数百GB規模［参考；最新はKaggleのDataタブ参照］）
- 研究（Colab/ローカル）: リポジトリ直下 `data/` をルートに、実体は DVC + GCS remote で管理。必要分のみ `dvc pull` で取得。
  - `data/raw/` に公式データ、`data/interim/` に中間生成物、`data/processed/` に前処理済みを格納。
- 提出（Kaggle Notebooks Only）: 公式データは Kaggle 側 `/kaggle/input/rsna-intracranial-aneurysm-detection/` を参照。巨大データは持ち込まず、必要最小の前計算と重みのみを Add data（各Dataset上限200GB［公式Docs準拠］。ただし `/kaggle/working` は20GiB［永続］、一時領域にも制約があるため搬入は圧縮・分割を推奨）で追加。

#### 局所化データ（train_localizers.csv）の活用

- 公式配布の `train_localizers.csv` は動脈瘤の座標・位置情報を含む。3D 候補検出の教師や、候補パッチ分類（3D/2.5D）での正例サンプリングに活用可能。
- 運用案：Colab 側で localizers を取り込み、`tools/pack_precompute.py` で `<case_id>/candidates.csv`（z,y,x,score 等）として梱包→Kaggle に Add data。評価・NMS は mm スケールで行い、`docs/SUBMISSION_CONTRACT.md` と整合させる。

### ローカル/Colab: 初回取得と同期
- 公式データの取得（Kaggle API）
```bash
kaggle competitions download -c rsna-intracranial-aneurysm-detection -p data/raw
unzip -q data/raw/rsna-intracranial-aneurysm-detection.zip -d data/raw
```
- DVC remote（GCS）の利用（例）
```bash
cp dvc.config.example dvc.config
dvc pull  # 既存の共有データ/成果物を取得（gs:// を参照）
```
- 新規生成物を共有に載せる場合（例）
```bash
dvc add data/processed
git add data/processed.dvc data/.gitignore
git commit -m "Add processed artifacts to DVC"
dvc push
```
- 容量ガイド: 数百GB規模（参考；最新はKaggleのDataタブ参照）は remote を真実源にし、手元は必要分のみ取得。中間生成物は `npz/float16` や疎表現で圧縮。

### Kaggle Notebooks: 搬入物と参照先（Internet: Off）
- 参照先
  - 公式データ: `/kaggle/input/rsna-intracranial-aneurysm-detection/`
  - 追加データ（Add data）: 前計算 `rsna2025-precompute`、重み `rsna2025-weights`
    - `/kaggle/working` は≒20GiB 上限（永続保存領域）
    - 入力データセットは1件あたり上限200GB（公式Docs準拠）。ただし実運用では `/kaggle/working` の20GiBや一時ディスク容量がボトルネックになりやすいため、分割・圧縮を推奨
- パス（Hydra の Kaggle プロファイル）
```yaml
# Kaggle 環境専用パス設定（/kaggle/working, /kaggle/input を前提）
paths:
  work_dir: /kaggle/working
  input_dir: /kaggle/input
  output_dir: ${paths.work_dir}/outputs
  models_dir: ${paths.input_dir}/rsna2025-weights
  precompute_dir: ${paths.input_dir}/rsna2025-precompute
  preds_dir: ${paths.output_dir}/preds
  oof_dir: ${paths.output_dir}/oof

wandb:
  enabled: false
  mode: disabled
```
- 実行フロー（例）
```bash
# Add data で rsna2025-precompute / rsna2025-weights を追加
python kaggle/kaggle_infer.py --serve
```

### 前計算（持込フォーマットの指針）
- 例（スケルトン、後続で仕様確定）
  - `<case_id>/volume.npz`（等方再サンプル済, float16）
  - `<case_id>/brain_mask.npz`
  - `<case_id>/candidates.csv`（z,y,x,score 等の最小列）
- ひな形生成（ローカル/Colab）
```bash
python tools/pack_precompute.py
# dist/rsna2025-precompute/ と _meta.json を作成（スケルトン）
```

### 契約・検証
- 真実源: 本コンペの**公式評価API仕様**に完全追従（Overview/Rules/評価実装の期待形式）。
  - サーバ応答はシリーズIDごとの 14 ラベル確率 [0,1]（`aneurysm_present` + 13 部位ラベル）。
  - `docs/SUBMISSION_CONTRACT.md` は評価API仕様と常に同期（差異があれば必ず更新）。
  - ローカルDry-runでは CSV 雛形を `tools/verify_submission.py` で検証（本番は未使用）。
  - 列名とラベル順序は公式定義に一致させる。列順の真実源は `docs/SUBMISSION_CONTRACT.md`（`series_id` と 14 ラベルの順序を固定）。

### よくある質問
- データ総量は数百GB規模（参考）。どこに置く？ → 研究側の DVC remote（GCS）を真実源にし、`data/raw/` を DVC 管理。Kaggle では `/kaggle/input/...` を参照し、巨大データは持ち込まない。最新サイズは Kaggle のDataタブを確認。
- どのくらい持ち込める？ → `/kaggle/working` は≒20GiB 上限（永続）。入力データセットは1件あたり上限200GB（公式Docs準拠）。ただし一時領域や実行時間の制約が実運用のボトルネックになりやすいので、前計算+重みは圧縮・分割設計を推奨。

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
- 提出用推論は Kaggle 側に最小化して分離（シリーズ→14ラベル確率を第一経路に、ローカライゼーションは可視化/特徴補助）

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

## 提出（サービングAPI / Notebook-Only）

### Notebook設定
- 「Edit » Notebook settings」等から Internet: Off を選択し、保存してから実行する。
  - Notebook-only 競技ではインターネット無効や外部データ可否がルールで定義。詳細はコンペの Code 要件パネル/FAQ を参照。

- Notebook: `kaggle/notebook_template.ipynb` をアップロード
- 「Add data」で `rsna2025-precompute` と `rsna2025-weights` を追加（依存wheelも必要なら `rsna2025-wheels` を追加）
- `kaggle/kaggle_infer.py` をサーバ実装として起動し、起動後15分以内に初期化完了→`serve()` を呼び出して待受（シリーズごとに14確率を応答）

- **重要**: サーバ初期化は起動後15分以内に完了し、必ず `serve()` を呼び出すこと（評価API要件）

```bash
# Kaggle 環境（概念図）
python kaggle/kaggle_infer.py --serve   # 起動→初期化→serve() で待受開始（15分以内）
# 評価APIがテストシリーズを逐次配信→サーバが14確率を返す（CSV提出は不要）
```

### オフラインpip（wheel / Add data）

```bash
# 依存が標準コンテナに無い場合のみ。wheel を事前に収集して Dataset 化（例: rsna2025-wheels）
pip install --no-index --find-links /kaggle/input/rsna2025-wheels -r kaggle/offline_requirements.txt
python - <<'PY'
import torch, sys
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
PY
```

### 時間ガード（推論用の自動ダウングレード指針）

※ 実行中にシリーズ単位の ETA を見積もり、予算超過が見えたら 基準解像度の段階的低下 → TTA停止 → パッチストライド粗化 → 候補上限縮小 の順で自動ダウングレード（`kaggle_infer.py` 内）。`time_budget_hours` 内完走を最優先。

- シリーズ単位の概算 ETA が `time_budget_hours` を超えそうな場合の優先順位
  - 入力解像度（短辺基準）を段階的に下げる → TTA 停止 → パッチストライドを粗く → 候補数上限を縮小
- “完走最優先” を原則とし、ダウングレードの切替は `kaggle_infer.py` 内で実装（コメント済、後続で実装）
 - 推論高速化の実務指針（本追記）: AMP（半精度）/TorchScript/ONNX を優先適用し、適用可能な層に対して dynamic quantization（int8）を使用。自動ダウングレード（解像度→TTA→stride→候補数）と併用し `time_budget_hours` 制限での完走率とスループットを最大化する（詳細は `experiments/exp0007_2p5d_mainline/` を参照）。

### Dry-run 検証（CSV雛形・ローカル専用）

```bash
python tools/verify_submission.py .work/submission.csv \
  --id_regex '^[A-Za-z0-9_.-]+$' --deny_duplicates --check-range
```

- 本検証はローカル/Colab の乾式リハーサル専用。Kaggle 本番では使用しない（本番はサービングAPIのみ）。
- 列名（`series_id` と 14 ラベル）、dtype、NaN/Inf、値域 [0,1]、重複行を軽量検査
- 仕様確定後に `tools/verify_submission.py` 内コメントへ反映（実装は後続）

### サービングAPI契約（必ず固定）

- 真実源は本コンペの**公式評価API仕様**。Overview/Rules/評価実装で定義されたAPIに完全追従。
  - サーバはシリーズIDごとに 14 ラベル（`aneurysm_present` + 13 部位ラベル）の確率 [0,1] を応答する。
  - スキーマ（キー名・dtype・NaN/Inf 取り扱い・値域）は評価APIの期待に合わせる。
- 反映先：`docs/SUBMISSION_CONTRACT.md`（API契約の同期要約）と `configs/inference/kaggle_fast.yaml` に同一仕様を反映（`time_budget_hours` を含む）。
- `tools/verify_submission.py` はローカルDry-run（CSV雛形）のみで利用し、Kaggle 本番では使用しない。

## ローカル乾式リハーサル

```bash
make kaggle-dryrun  # .work/submission.csv を生成（現状は空の雛形）
```

- Kaggle 側 GPU（P100/T4）を想定した小サンプル実測で、`kaggle/kaggle_infer.py` の自動ダウングレード閾値（TTA 停止→stride 粗化→候補数上限→解像度）を校正します。
- 最悪条件でも `time_budget_hours` 内完走する設定を優先します。
- 本手順はローカルDry-run用（CSV雛形）であり、Kaggle 本番ではサービングAPIのみを使用します。

## 変更点（この改訂で追加された骨組み）

- `configs/paths/kaggle.yaml`, `configs/wandb/disabled.yaml`, `configs/inference/kaggle_fast.yaml`（time_budget_hours を基準に完走を優先する軽量設定）
- `kaggle/` ディレクトリ（README_KAGGLE, kaggle_infer.py, kaggle_utils.py, notebook_template.ipynb, offline_requirements.txt）
- `tools/pack_precompute.py`（再サンプル/脳マスク/候補点の前計算梱包スケルトン）, `tools/verify_submission.py`（提出検証スケルトン）
- `tests/test_dicom_geometry.py`（現状 skip）
- `Makefile` の kaggle タスク（prep/dryrun/wheels）
- tests/test_inference.py：候補→パッチ→NMS→CSV の最小経路を検証
- Kaggle Dataset 合計≦20GB を前提に、前計算は `npz/float16`、候補は疎表現で圧縮（重みも必要最小限に絞る）

## 次ステップ（実装方針だけ明記）

- kaggle_infer: 候補点ロード→3Dパッチ切り出し→モデル推論→NMS→CSV
- pack_precompute: 再サンプル/脳マスク/候補点の形式確定→梱包
- verify_submission: 列名/dtype/NaN・Inf/重複/値域の軽量検証（公式評価APIの期待形式に直接照合）
- dicom_geometry: `tests/test_dicom_geometry.py` の skip を最優先で解除し、CI/ローカル前提チェックとして実行（失敗時は学習/推論を停止する運用）。

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
- **make deps-freeze**：ローカルで pyproject.toml から env/requirements.txt を再生成（配布は生成物）

## トラブルシュート

- **GCS 認証がうまくいかない**：Colab で `from google.colab import auth; auth.authenticate_user()` を実行。サービスアカウント使用時は `GOOGLE_APPLICATION_CREDENTIALS` を設定し、鍵は追跡しない。
- **Kaggle API で 401/403**：~/.kaggle/kaggle.json の配置と chmod 600（権限）を確認。Kaggle 設定ページから再発行も有効です。
- **DVC × GCS での権限エラー**：`dvc remote modify gcsremote credentialpath .gcloud/service-account.json` を確認。バケット IAM/ロールを点検（Storage Object Admin など最小権限）。
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

# 5) （参考）Kaggle 本番はサービングAPI（起動後15分以内に初期化→serve）。
#     ローカルDry-runは上記 kaggle-dryrun を参照（CSV雛形の検証のみ）。
```

## 参考

- [Kaggle API（Public API / GitHub README）](https://github.com/Kaggle/kaggle-api)
- [Kaggle Notebooks（Internet: Off / Add data の考え方）](https://www.kaggle.com/docs/notebooks)
- [W&B Quickstart / wandb login（公式Docs）](https://docs.wandb.ai/quickstart)
- [DVC × GCS（remote 設定）](https://dvc.org/doc/user-guide/data-management/remote-storage/google-cloud-storage)
- [Google Cloud 認証（ADC / サービスアカウント）](https://cloud.google.com/docs/authentication)
- [Hydra 基本のオーバライド構文（Basic Override）](https://hydra.cc/docs/advanced/override_grammar/basic/)
- [Hydra マルチラン（Multi-run / Sweeper）](https://hydra.cc/docs/advanced/multi-run/)
- Kaggle Datasets（サイズ上限の目安と運用注意）

## Kaggle Notebooks Only 対応（研究=Colab / 提出=Kaggle の二層設計）

> 最低限の案内のみ。実装詳細は後続ステップで追加予定。

- 切替例（Hydra）: `paths=kaggle wandb=disabled inference=kaggle_fast`（時間ガードは `time_budget_hours` 基準、シリーズ単位 ETA）
- 追加ファイル（骨組み）:
  - `configs/paths/kaggle.yaml` / `configs/wandb/disabled.yaml`
  - `configs/inference/kaggle_fast.yaml`（time_budget_hours を持つ）
  - `kaggle/` ディレクトリ（README_KAGGLE, kaggle_infer.py, kaggle_utils.py, notebook_template.ipynb, offline_requirements.txt）
  - `tools/pack_precompute.py`（前計算梱包・スケルトン）
  - `tools/verify_submission.py`（Dry-run用CSV検証・スケルトン：`series_id` + 14 ラベル確率の検証を想定）
  - `tests/test_dicom_geometry.py`（幾何テスト・後続で実装）
  - 追加（本追記）: `configs/train/presence_calibration.yaml`, `configs/inference/modality_thresholds.yaml`, `configs/model/two_point_five_d.yaml`
- Make タスク:
  - `make kaggle-prep` / `make kaggle-dryrun` / `make wheels`

TODO:
- 推論本体の第一経路（シリーズ→14ラベル確率の直行分類）を `kaggle/kaggle_infer.py` のサーバ実装として完成（ETA に基づく自動ダウングレードは `time_budget_hours` 内完走を第1目標）
- 候補→パッチ分類の経路は補助/アンサンブル要員（必要に応じて併用）
- 前計算仕様（フォーマット/項目）と `tools/pack_precompute.py` の実装
- 検証ツール・幾何テストの具体化

## 設計レビュー（強み・改善アクション）

### よかった点（強み）
- Colab=ロギング＆可視化／Kaggle=最小限推論の二層設計が明快（`paths=kaggle wandb=disabled` で切替一貫）
- 再現性に強い構成（`experiments/expXXXX/config.yaml` をスナップショット、`pyproject.toml` を真実源）
- 提出のオフライン最適化（wheels Dataset、時間ガード、検証スクリプト）で“完走最優先”の思想

### リスクと埋めたい穴
- DICOM 幾何の不整合（方位・符号・間隔/厚み・欠損）
  - `tests/test_dicom_geometry.py` の skip を最優先で解除し、等方再サンプルと座標系の統一を先に固定（失敗時は学習/推論を停止する運用）。
  - 運用冗長化: 学習アーティファクトの真実源を Kaggle Datasets（Private）または別リモート（S3/GCS など）にも二重化して保持。
- リーク防止と CV の一意性
  - 患者単位 split を CSV 化し、常に同じ fold を再利用（DVC / W&B artifact）
- Kaggle 時間ガードの実装不足
  - ETA 推定＋段階的ダウングレード（TTA→stride→候補N→解像度→2.5D）を自動切替
- 提出 CSV 仕様の早期固定
  - 列名・dtype・座標系・単位・スコア範囲の契約を `tools/verify_submission.py` と `docs/SUBMISSION_CONTRACT.md` に集約
- オフライン依存の脆さ
  - `offline_requirements.txt` + wheels で `--no-index` が通ることをローカル乾式/CIで検証

- Kaggle Dataset サイズ上限の遵守（各Dataset上限200GB［公式Docs準拠］）
  - 前計算は `npz/float16`、メタは parquet、重みは必要最小構成に絞る（`/kaggle/working` の20GiB制約も考慮）
- 外部データ・事前学習の可否と記載の徹底
  - Rules 準拠を徹底し、使用時は出所とライセンスを `docs/DATASET_CARD.md` に明記

### すぐ効く改善（この改訂で追加/更新）
- 提出契約ドキュメント: `docs/SUBMISSION_CONTRACT.md` を追加（検証ルールの真実源）
- Precompute スキーマの明文化: `docs/DATASET_CARD.md` に追記
- CV 分割の固定化: サンプル `data/processed/cv_fold_assign.example.csv` を追加
- README に設計レビューと契約リンクを追記
 - 最小の高インパクト改善（本追記で方針のみ固定）:
   - presence 校正の徹底強化（重み13に直結）: presence ヘッドの損失を class-balanced BCE + focal（γ>0）で強化し、OOF で温度スケーリング/等分位校正を適用。方針スケルトンを `configs/train/presence_calibration.yaml` に追加し、検証実験を `experiments/exp0006_presence_calib/` に分離。モダリティ（CTA/MRA/MRI）差に対しては `configs/inference/modality_thresholds.yaml` で modality-wise threshold を CV で同定。
   - 2.5D 主力 + 軽量3D補助 + 推論最適化: スライス CNN + RNN（LSTM/GRU）による 2.5D を presence/部位の第一主力に据え、3D は候補パッチ分類の補助に限定。推論では AMP + TorchScript/ONNX + dynamic quantization を活用し、時間ガード（解像度→TTA→stride→候補数）と併用して `time_budget_hours` 制限下での“速くて強い”実行を担保。方針スケルトンを `configs/model/two_point_five_d.yaml` と `experiments/exp0007_2p5d_mainline/` に追加。
   - 公開外部の事前学習の活用（ルール順守・出所明記）: 頭部 CT/MR 近縁の公開データや自己教師あり（例: 3D MAE 系）で backbone を事前学習し、本データで微調整。学習は Colab、重み搬入は Kaggle Datasets（Private, Add data）。出所・ライセンス・再現手順は `docs/DATASET_CARD.md` に明記。検証実験は `experiments/exp0008_external_pretrain/` に分離。

参考: 現時点のデフォルトは voxel 座標（`z,y,x`）と `confidence∈[0,1]`。ただし差異があれば**公式評価API仕様を常に優先**し、`docs/SUBMISSION_CONTRACT.md` を同期更新。

## この改訂での“最小追加”（実装はせずコメント方針のみ固定）

- **幾何テストの強制化（コメント方針）**: `tests/test_dicom_geometry.py` の skip を解除する前提で、以下の一致をテストで担保する方針を明記。
  - **spacing/厚み/傾斜**: PixelSpacing, SliceThickness, gantry tilt
  - **方位・符号**: ImageOrientationPatient, 左右/前後/頭尾の符号一貫性
  - **強度**: RescaleSlope/Intercept の適用と窓関数の一貫性
  - **等方 resample**: 3D→等方 resample のパイプライン一貫性
  - 失敗時は学習/推論を停止する運用（コメントで明記、実装は後続）

- **二段構え（候補→パッチ分類）の確定（コメント方針）**:
  - Colab: 軽量 3D U-Net/CenterNet で候補ヒートマップ→Top-K 座標抽出→ `rsna2025-precompute` に梱包（`candidates.csv`, `volume.npz`）。
  - Kaggle: 候補パッチのみ高精度 3D/2.5D 分類→NMS→CSV。時間ガードで TTA→stride→候補N→解像度の順に縮退。

- **mm スケール準拠の NMS/評価（コメント方針）**:
  - NMS 半径および一致判定を mm 単位で固定（PixelSpacing/SliceThickness 由来）。
  - `tools/verify_submission.py` に voxel⇄mm の往復検証（コメント）を追加予定。
  - 真実源は `docs/SUBMISSION_CONTRACT.md` に記載し、`configs/inference/kaggle_fast.yaml` と整合。

- **モダリティ別前処理の明文化（コメントのみ）**:
  - `src/rsna_aneurysm/transforms.py` に CTA/MRA/MRI 等モダリティ別の窓設定・強度正規化・等方リサンプルの方針をコメントで1段落追記（実装は後続）。
  - 目的: 候補検出（3D U-Net/CenterNet）→パッチ分類（3D/2.5D）間での前処理揺れを抑制。
  - コメント例（コードは書かない）：
    - CTA: 軟部/骨抑制ウィンドウの選択基準、HU→標準化の範囲、1.0mm 等方 resample を基本。
    - MRA/MRI: シーケンス別（TOF/T1/T2 等）の強度正規化方法、bias field 補正の適用可否、等方 resample の補間種別。
    - 共通: voxel→mm の換算を transforms 内に統一実装（方針のみ記述）。

### 追加された実験ディレクトリ（コメントのみのスケルトン）

```
experiments/
├── exp0001_baseline/
│   ├── config.yaml
│   ├── training.ipynb
│   ├── evaluation.ipynb
│   ├── inference.ipynb
│   └── notes.md
├── exp0002_candidates/
│   ├── config.yaml    # コメントのみ（候補ヒートマップ→Top-K 抽出の方針）
│   └── notes.md       # コメントのみ（運用・搬入ポリシー）
├── exp0003_patchclf/
│   ├── config.yaml    # コメントのみ（3D/2.5D パッチ分類の方針）
│   └── notes.md       # コメントのみ（TTA/時間ガード運用）
├── exp0004_nms_contract/
│   ├── config.yaml    # コメントのみ（mm スケール NMS/判定の契約）
│   └── notes.md       # コメントのみ（検証方針）
└── exp0005_ensemble_light/
    ├── config.yaml    # コメントのみ（軽量アンサンブル + 時間ガード）
    └── notes.md       # コメントのみ（P100/T4 実測で閾値校正）
```

- 追加予定の実験ディレクトリ（本追記、コメントのみのスケルトン）

```
experiments/
├── exp0006_presence_calib/
│   ├── config.yaml    # コメントのみ（presence 損失重み・校正・しきい値検証）
│   └── notes.md       # コメントのみ（OOF 校正・評価指標の扱い）
├── exp0007_2p5d_mainline/
│   ├── config.yaml    # コメントのみ（2.5D 主力 + 軽量3D補助 + 推論最適化）
│   └── notes.md       # コメントのみ（AMP/TorchScript/ONNX/quantization の運用）
├── exp0008_external_pretrain/
│   ├── config.yaml    # コメントのみ（公開外部事前学習→微調整の手順）
│   └── notes.md       # コメントのみ（ルール順守・出所/ライセンス記載）
└── exp0009_anatomy_aware/
    ├── config.yaml    # コメントのみ（13部位セグ/マスク活用の補助Loss/NMS重み付け）
    └── notes.md       # コメントのみ（設計メモとCV/ルール遵守/時間ガード方針）
```

- **運用**: すべて Colab で学習・検証し、Kaggle では `kaggle/kaggle_infer.py` のサーバ起動に限定（Internet: Off, wheels データセット併用）。
- **契約**: サーバ応答はシリーズごとの 14 ラベル確率 [0,1]。真実源は公式評価API仕様で、`docs/SUBMISSION_CONTRACT.md` はその同期要約とする。