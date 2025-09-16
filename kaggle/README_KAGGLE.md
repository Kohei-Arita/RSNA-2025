# Kaggle 提出ワークフロー（Notebooks Only / オフライン）

> 最低限の流れのみ記載。実装や詳細手順はステップ実装の後で追記。

## 流れ（概要）
- 事前計算・重みを Dataset 化して追加（"Add data"）
  - 重み（weights）は Public な Kaggle Dataset として公開し、提出Notebookのバージョンにリンク（必須）
- 必要最小限の依存解決（標準イメージ優先）
- `kaggle_infer.py` を実行して `submission.csv` を生成
- 軽量検証を行い、そのまま Submit

## 事前準備（ローカル/Colab）
- `tools/pack_precompute.py` で前計算成果物を梱包（TODO: 実装）
- 学習済み重みを収集して Dataset 化（Public 必須）
 - 追加依存が必要な場合は wheel を収集して Dataset 化（例: `rsna2025-wheels`）

## ノートブック
- `notebook_template.ipynb` をベースにセルを最小構成で実行
- 実装はコメント/TODO のみ。後続で差し替え予定

### 先頭セルの固定例（オフラインpip）

```bash
pip install --no-index --find-links /kaggle/input/rsna2025-wheels -r kaggle/offline_requirements.txt
python - <<'PY'
import torch, os
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
PY
```

### 時間ガード（完走最優先）

- 目安: 実行時間の上限は ≒12 時間（GPU/CPU）
- 進捗から ETA を推定し、超過リスク時に自動ダウングレード
  - TTA 停止 → ストライド粗化 → 候補数上限縮小 → 入力解像度縮小
- 実装は `kaggle/kaggle_infer.py` にコメントで指針を記載（後続で実装）

### CSV 検証（軽量）

```bash
python tools/verify_submission.py /kaggle/working/submission.csv
```

- 列名・dtype・NaN/Inf・重複・スコア/座標の値域をチェック（仕様確定後に更新）
