# Colab + Drive + Kaggle API + W&B の初期設定

以下のセルは `env/colab_setup.ipynb` に追加して運用してください（ノートブックは Git にも保存しておくと、チーム全員が同じ手順で再現できます）。

## セルA: リポジトリ＆依存インストール（再掲・簡略）
Colab: ランタイム起動直後に一度だけ実行。

```bash
pip -q install -U pip
pip -q install -r env/requirements.txt "dvc[gcs]"
```

## セルB: ADC（ブラウザ連携）で GCP 認証

```python
from google.colab import auth
auth.authenticate_user()  # ブラウザで認可
print("✅ ADC 認証OK")
```

## セルC: GCS へ疎通テスト（google-cloud-storage）

```python
from google.cloud import storage
client = storage.Client()  # ADC を使う
bucket_name = "<YOUR_BUCKET_NAME>"  # 例: rsna2025-<org>-prod
bucket = client.bucket(bucket_name)
print("✅ バケット存在:", bucket.exists())
```

## セルD: DVC リモート設定の読み込み＆pull テスト

既に `dvc.config.example` がある前提。自分の環境ファイル `dvc.config` を作るならここで。

```bash
cp -n dvc.config.example dvc.config || true
dvc remote list
dvc pull -q || echo "（初回は DVC で管理されているファイルが無い/権限不足の場合があります）"
python - << 'PY'
print("✅ DVC pull 実行完了（メッセージを確認）")
PY
```

補足:
- **YOUR_BUCKET_NAME** は `gcs/buckets.yaml` またはあなたの GCS 設定に合わせて置き換えてください。
- 事前に `env/requirements.txt` に `google-cloud-storage` と `dvc[gcs]` が含まれていることを確認してください。
- Colab 環境で `dvc` コマンドを使うため、上記インストールセルを先に実行してください。
