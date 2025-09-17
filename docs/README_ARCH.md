# GCS三層設計の要旨（コメント）
このファイルは README の構成に合わせたダミーです。実装方針やアーキ概要をここに記載してください。
 
## 三層設計の真実源=GCS

- 役割分担：
  - 研究/学習（Colab）… ロギング・可視化・前処理の実行。生成物は DVC/W&B 併用で管理
  - 共有ストア（GCS）… データセット・中間生成物・学習済み重みの真実源（Source of Truth）
  - 提出（Kaggle Notebooks Only, Internet: Off）… GCS 由来の前計算・重みを Dataset として搬入し、最小実装でサービング

- 真実源としての要件：
  - バケットは原則プロダクション用 1 系統を固定し、命名・ロケーション・権限制御を明文化
  - DVC remote は `gs://` を既定とし、取得は `dvc pull`、登録は `dvc add && dvc push`
  - 認証は ADC（Colab 推奨）またはサービスアカウント鍵（鍵はリポジトリで追跡しない）

- 運用定義（本リポジトリでの固定事項）：
  - バケット定義は `gcs/buckets.yaml` に記載
  - 本番バケット… `rsna2025-prod`（ロケーション: `asia-northeast2` 大阪、UBLA: on）
  - 例外設定や IAM 付与の雛形は `gcs/*.example.yaml` を参照し、実鍵や機微情報は `.gitignore` で除外

- 参照: `README.md`（Kaggle Notebooks Only 方針、時間ガード）、`docs/dvc_remote_gcs.md`（DVC×GCS 連携）、`configs/paths/*.yaml`（環境別パス）

## GCS 運用の基本方針
- UBLA（Uniform bucket-level access）を有効化し、ACL を使わず **IAM 一元管理**とする。
- バケット/オブジェクトに **ラベル**（env/team/purpose 等）を付与し、コスト分析や棚卸しを容易にする。
- アクセス頻度に応じて **ライフサイクル**で Standard → Nearline → Coldline → Archive に自動移行。
- 実設定はコンソールで行い、設計は `gcs/buckets.example.yaml` に記録（変更点は PR レビュー対象）。
