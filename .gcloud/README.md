# このディレクトリは GCS 認証情報の配置ガイド用です（コメントのみ）。
#
# 注意:
# - 実運用のサービスアカウント鍵は絶対にコミットしないでください（.gitignore 対象）。
# - Colab では ADC を推奨: from google.colab import auth; auth.authenticate_user()
# - サービスアカウント鍵を使う場合の環境変数例:
#     export GOOGLE_APPLICATION_CREDENTIALS="/content/RSNA-2025/.gcloud/service-account.json"
# - DVC での credentialpath 指定例（コメント）:
#     dvc remote modify gcsremote credentialpath .gcloud/service-account.json
