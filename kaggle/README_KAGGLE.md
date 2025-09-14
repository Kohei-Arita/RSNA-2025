# Kaggle 提出ワークフロー（Notebooks Only / オフライン）

> 最低限の流れのみ記載。実装や詳細手順はステップ実装の後で追記。

## 流れ（概要）
- 事前計算・重みを Dataset 化して追加（"Add data"）
- 必要最小限の依存解決（標準イメージ優先）
- `kaggle_infer.py` を実行して `submission.csv` を生成
- 軽量検証を行い、そのまま Submit

## 事前準備（ローカル/Colab）
- `tools/pack_precompute.py` で前計算成果物を梱包（TODO: 実装）
- 学習済み重みを収集して Dataset 化

## ノートブック
- `notebook_template.ipynb` をベースにセルを最小構成で実行
- 実装はコメント/TODO のみ。後続で差し替え予定
