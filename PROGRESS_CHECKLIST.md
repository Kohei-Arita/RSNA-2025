# PROGRESS_CHECKLIST — 進捗プラン・更新ルール

このファイルは、実験・実装の進捗と次アクションを簡潔に可視化するためのチェックリストです。

## 更新ルール
- 進捗や計画の変更があれば必ず本ファイルを更新する（README にも明記）。
- 最新のエントリを上（先頭）に追記する。
- 各エントリには少なくとも「日付（YYYY-MM-DD）」「担当」「要点」を記載する。
- 可能であれば関連する `experiments/expXXXX_*` ディレクトリ、Pull Request、W&B Run などへの参照を添える。



## RSNA2025 脳動脈瘤検出コンペ：作業チェックリスト

### 前提条件・環境設定

- [x] Kaggleアカウント作成＆APIトークン取得: Kaggleのマイページ「Account」タブからAPIトークン（`kaggle.json`）を生成し、`~/.kaggle/kaggle.json`に配置（権限は`600`）。Kaggle CLI（`kaggle`コマンド）が動作することを確認。
 - [x] Google Cloud/GCS準備: GCPプロジェクトを用意し、コンペ用のGCSバケット（例: `gs://<your-bucket>/rsna2025`）を作成。DVC用サービスアカウントにStorage Object Admin権限を付与。Colabでは`GOOGLE_APPLICATION_CREDENTIALS`またはADCで認証可能に。（ADCで確認済み、SA鍵は未使用）
- [ ] 依存環境構築: リポジトリをクローンし、`pip install -r env/requirements.txt`でPythonライブラリをインストール。PyTorch/TensorFlow等のバージョン確認、必要に応じてGPU(T4/P100)やTPUを有効化。
- [ ] コードレポ・実験管理: `src/rsna_aneurysm/cli.py`に倣い、`experiments/`以下に実験ディレクトリを作成・管理。各実験で`config.yaml`にHydra設定を固定し、Notebookで再現可能に。

- [ ] Kaggle Notebook のアクセラレータ確認：Code タブの Notebook settings で選択肢を確認。GPU/CPU を基本とし、TPU は“要確認”（提出環境に依存）。

### データ取得と管理

- [ ] 公式データの取得（Kaggle API）: `kaggle competitions download -c rsna-intracranial-aneurysm-detection -p data/raw`でダウンロード・解凍。総容量は数百GB規模（参考）。最新サイズはKaggleのDataタブを確認。Colab単独は厳しいため、ローカル/大容量ストレージで取得・分割し、必要分のみGCSへ格納を検討。
- [ ] RSNA公式情報: 提供データはCT/MR脳画像で、13解剖学的領域の動脈瘤有無・位置のアノテーションを含む。生データはDVC+GCSでリモート管理し、Colabでは必要部分のみ`dvc pull`で取得する運用とする。
- [ ] DVC remote設定: `dvc init`後、`dvc remote add -d gcsremote gs://<your-bucket>/rsna2025`でリモート追加。`dvc pull`で取得、`dvc push`でアップロード。`dvc add data/raw`で生データを管理下に置き、`.dvc`ファイルをコミット共有。
- [ ] 部分取得の活用: DVCは必要ファイルのみ取得可能。Colabでは学習に必要なサブセット（例: 一部症例のみ）だけ`dvc pull`し、メモリ節約。
- [ ] 前処理データの管理: 等方化npz、脳マスク、候補座標などを`data/processed/`に生成し、DVCで管理。`tools/pack_precompute.py`で生成フォルダを整える。完成した前処理データは`dvc push`でGCSへ。
- [ ] Kaggle Notebook用データセット: Notebook Only提出に備え、前処理データと学習済み重みをKaggle Dataset化。`dist/rsna2025-precompute/`と`dist/rsna2025-weights/`に集約。各Dataset上限は200GB（公式Docs準拠）だが、`/kaggle/working` は20GiB（永続）かつ一時領域にも制約があるため、`npz/float16`や疎フォーマットで圧縮・分割を推奨。
- [ ] Kaggleへのデータ追加: 公式データは`/kaggle/input/rsna-intracranial-aneurysm-detection/`で参照可能。自作データセット（`rsna2025-precompute`, `rsna2025-weights`）はNotebook右上「+ Add data」で追加。

- [ ] localizers 欠落の許容：一部シリーズで localizer が欠落する可能性に備え、missing 許容・presence/部位分類はフォールバックで動作することを確認。

### 探索的データ分析（EDA）

- [ ] 基本統計・分布確認: `notebooks/00_eda.ipynb`等で症例数、モダリティ別数（CTA/MRA/MRI）、陽性率、部位ラベル分布を可視化。クラス不均衡を把握し、損失重みへ反映。
- [ ] 画像サンプル確認: スライス例を表示し、ウィンドウ幅/レベル、ノイズ、アーチファクトを確認。モダリティ間のピクセル値分布差に留意し、前処理（正規化・ウィンドウ適用）の差異を検討。
- [ ] 形状・幾何情報の検証: DICOMヘッダーの`PixelSpacing`、`SliceThickness`、`ImageOrientationPatient`を確認し、一貫性をテスト。幾何的不整合があればリサンプリングを検討し、`tests/test_dicom_geometry.py`で検査。
- [ ] ローカルライズ情報: `train_localizers.csv`等の座標から陽性領域サイズ・位置を把握。候補検出・Patch生成に活用するため、座標系（voxel vs mm）の扱いを明確化。
- [ ] CV分割計画: 患者ID単位でCVを設定し、患者リークを防止。`configs/cv/patient_kfold.yaml`等を参考に、再現シードを固定。

### モデル学習

- [ ] モデル設計: まず2D CNN（例: EfficientNet, ConvNeXt）でベースライン。必要に応じて2.5D（連続スライス+RNN）や軽量3Dも検討。2ヘッド構成で存在有無（`aneurysm_present`）と13部位ラベルを多値分類で出力。
- [ ] 前処理・データ拡張: DataModuleで3Dを2D入力へリサンプリングし、ウィンドウ/正規化（モダリティ別補正）を適用。学習は回転・左右反転・ランダム歪みなど（`configs/aug/*`）を適用、検証は最小限。候補パッチ学習時は陽例中心Patchと陰例ランダム抽出でバランス。
- [ ] 損失関数・重み付け: presenceヘッドは稀少性を考え`BCE+Focal`や重み付きBCE（陽:陰=13:13）を検討。部位ヘッドはラベル別BCE（陽例のみカウント）。重み付きColumnAUC対策としてpresence損失を重視（例: `configs/train/presence_calibration.yaml`）。
- [ ] 交差検証学習: `python -m rsna_aneurysm.cli train`でHydra設定を組み合わせて実行（例: `model=baseline_2d cv=patient_kfold train=base,fp16 data=rsna paths=colab wandb.project=RSNA2025`）。fold毎の重みは`models/expXXXX_foldY.ckpt`に保存し、W&Bへログ。
- [ ] W&Bログ: `wandb.login()`し、Runにパラメータ・メトリクス・ROC等を記録。Sweepでハイパラ自動化。Kaggle提出では`wandb=disabled`。
- [ ] モデル選択: 各Foldの検証AUC・損失を比較。presenceヘッドAUCを重視して選定し、`models/expXXXX_foldY_best.ckpt`としてマーク。

### 推論・Kaggle提出準備

- [ ] OOF予測と評価: CVのOOF予測を作成し、各ラベルのROC/AUCを算出。スコアは score = (13·AUC_presence + Σ AUC_parts) / 26 で評価（parts は13ラベル）。
- [ ] 出力フォーマット確認: 提出はサービングAPI形式。シリーズIDごとに14要素（`aneurysm_present` + 13ラベル）の確率を返す。予測が[0,1]範囲で重複行がないことを`tools/verify_submission.py`で検証。
- [ ] 前処理データのパッケージ化: スケルトン（等方化ボリューム、脳マスク、候補座標）を`tools/pack_precompute.py`で生成し、`dist/rsna2025-precompute/`へ。
- [ ] モデル重みの準備: 学習済み重み（各Fold/アンサンブル）を`dist/rsna2025-weights/`にまとめ、サイズ制限内に整理。`kaggle/kaggle_utils.py`でロード可能に管理。
- [ ] Kaggle推論スクリプト: `kaggle/kaggle_infer.py`をサービングとして使用。Notebook内で`python kaggle/kaggle_infer.py --serve`を実行し、起動後15分以内に初期化完了→必ず`serve()`を呼び出して待機（評価API要件）。
- [ ] 時間管理と簡易化: 制限時間（既定9h）を超えないよう、ETAを監視し「解像度縮小→TTA停止→ストライド粗化→候補数削減」の順でダウングレード。AMP（半精度）・TorchScript/ONNX・動的量子化で高速化。
- [ ] 提出の最終確認: Notebookは「Internet: Off」。必須ライブラリは`kaggle/offline_requirements.txt`からwheelを使用。`make kaggle-dryrun`等でローカルDry-run（形式チェック）を実行し、問題なければ「Save Version」→提出（CSV提出は不要・例外なし、Serving API のみ）。

- [ ] 学習済み重みをPublic Dataset化: `rsna2025-weights` を Kaggle で Public とし、提出Notebookのバージョンとリンク（URL/IDを本チェックリストに記載）。

- [ ] SUBMISSION_CONTRACT に従い、14 ラベルの正式名と配列順を固定（コード・検証双方が同一の真実源を参照）。
- [ ] `serve()` は起動直後に先呼びし、重い初期化は Lazy Load/ウォームアップで吸収（評価APIの15分要件を厳守）。
- [ ] テストはランダム順で約 2,500 シリーズ（目安）を想定し、ETA と自動ダウングレードの閾値を設計。
- [ ] CSV は Dry-run 専用（本番は Serving API）。テンプレの「submission.csv」表記に惑わされないようチーム内で明文化。

### 実験管理・共有

- [ ] 実験ディレクトリ管理: `experiments/expXXXX_name/`に`config.yaml`・学習/評価/推論Notebook・`notes.md`を集約。重要ハイパラは`config.yaml`で固定。
- [ ] 成果物の整理: 再生成可能な成果物（モデル重み、予測、キャッシュ）は`outputs/`へ。共有資料・図表は`reports/figures/`へ。DVCで共有できるデータはリモートへPushし、必要に応じてW&B Artifactsでもバックアップ。
- [ ] コード品質・テスト: `tests/`のユニット/統合テストを整備。前処理、DICOM読み込み、推論パイプラインにテストを用意し、CIで自動検証。特に`tests/test_dicom_geometry.py`は最優先で動作確認し、失敗時は学習/推論を停止する運用。
- [ ] ルール遵守・ドキュメント: 外部データや事前学習モデルの利用は`docs/DATASET_CARD.md`に出所・ライセンス・再現手順を明記。提出契約仕様（シリーズID順、14列、0-1範囲）は`docs/SUBMISSION_CONTRACT.md`に従い、`tools/verify_submission.py`と整合。

- [ ] LB 運用の注意：公開LBは約32%、最終LBは残り約68%で再計算。過学習に注意し、CV/OOF を優先判断に。

## 現在の進捗（最新が上）

- まだ記入なし。初回更新時に上記テンプレートをコピーして記入。



