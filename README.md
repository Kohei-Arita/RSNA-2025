# RSNA-2025 — Intracranial Aneurysm Detection (Colab-First)

本プロジェクトの実行環境は Google Colab を前提に設計しています。以降の手順・コマンドは Colab 上での利用を想定しています（Drive 連携、Kaggle API、Weights & Biases を使用）。Colab での Drive マウントや入出力の流れは Colab 公式ノートブック「Local Files, Drive, Sheets, and Cloud Storage」を基準にしています。

## 目次

- [プロジェクト概要](#プロジェクト概要)
- [リポジトリ構成](#リポジトリ構成)
- [前提条件](#前提条件)
- [Colab セットアップ手順（初回）](#colab-セットアップ手順初回)
- [データ取得（Kaggle API → DVC/Drive 連携）](#データ取得kaggle-api--dvc-drive-連携)
- [EDA（探索的データ分析）](#eda探索的データ分析)
- [学習（Hydra/CLI + W&B ロギング）](#学習hydra-cli--wb-ロギング)
- [推論・OOF・可視化](#推論oof可視化)
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

## データ取得（Kaggle API → DVC/Drive 連携）

### A. Kaggle から公式データを取得（Colab）

Kaggle API の基本コマンドを用います（競技名は rsna-intracranial-aneurysm-detection）。

```bash
# 競技規約に同意済みであることが前提
!kaggle competitions download -c rsna-intracranial-aneurysm-detection -p data/raw
!unzip -q data/raw/rsna-intracranial-aneurysm-detection.zip -d data/raw
```

### B. DVC で同期（チーム共有・復元用）

チームで Drive を DVC remote にしている場合、dvc pull / dvc push で同期します。

既存の加工物（data/processed など）を復元する場合も dvc pull。新たに生成したら dvc add → dvc push。

## EDA（探索的データ分析）

- 共通 EDA は notebooks/00_eda.ipynb を Colab で開き、入出力は configs/data/*.yaml を尊重する方針。
- 大規模 EDA でキャッシュが必要な場合は configs/data/cache.yaml を調整してから実行。
- Colab での Drive I/O は公式手順（上記）に準拠。

## 学習（Hydra/CLI + W&B ロギング）

### 1) CLI の基本

Hydra のオーバライド構文を使って、1 コマンドで構成を合成・実行します。

```bash
# 例：2D ベースラインを患者KFoldで学習、FP16有効、W&B プロジェクト名指定
python -m rsna_aneurysm.cli train \
  model=baseline_2d \
  cv=patient_kfold \
  train=base,fp16 \
  data=rsna \
  paths=colab \
  wandb.project=RSNA2025 \
  train.batch_size=16 train.max_epochs=10
```

### 2) マルチラン（簡易スイープ）

Hydra の multirun（-m）で複数設定を一括実行可能です。必要に応じてスイーパープラグインも利用できます。

```bash
# 例：学習率と重み減衰のグリッド
python -m rsna_aneurysm.cli -m train=base,fp16 \
  train.optimizer.lr=1e-3,3e-4 \
  train.optimizer.weight_decay=1e-4,1e-5
```

### 3) Weights & Biases でのトラッキング

初回は wandb login（上記）→ 以降は自動でログされます。

スクリプト側では wandb.init(project=..., config=...)・メトリクス/画像/アーティファクトをロギングできます。

## 推論・OOF・可視化

**推論**：

```bash
python -m rsna_aneurysm.cli infer \
  inference=base \
  model=baseline_2d \
  paths=colab \
  checkpoint_path=models/exp0001_fold0_best.ckpt
```

**OOF 生成・評価**：experiments/expXXXX/evaluation.ipynb を実行、または src/rsna_aneurysm/oof_utils.py のラッパーを呼びます。

**可視化（学習曲線・Grad-CAM 等）**：src/rsna_aneurysm/visualization.py / reports/figures/ を参照。

## 提出（Submission 自動化）

提出は Kaggle API で行います。競技ページの規約に従い、submissions/に生成した CSV を送信します。

```bash
# 例：tools/submit.py で CSV を生成した後、Kaggle API による提出
python tools/submit.py --input outputs/preds/exp0001_fold-avg.csv \
                       --output submissions/exp0001.csv

# Kaggle に投稿（コメント付き）
kaggle competitions submit \
  -c rsna-intracranial-aneurysm-detection \
  -f submissions/exp0001.csv \
  -m "exp0001: baseline_2d + fp16 + patient_kfold"
```

Colab での API 利用は Kaggle 公式の Public API/GitHub の README に準拠しています。

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