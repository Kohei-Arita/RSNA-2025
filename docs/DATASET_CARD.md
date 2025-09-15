# データカード（前処理・倫理配慮・分布等）

本カードは、本プロジェクトで扱う DICOM/体積データの前処理契約、座標系、precompute アーティファクトのスキーマ、ならびにデータ利用上の注意をまとめます。

## Precompute アーティファクト（rsna2025-precompute）スキーマ

Kaggle Notebooks Only を想定し、前計算済み成果物を Dataset として持ち込みます。最小構成（案）は以下です。

- `spacing.json`: 各 `study_id` の元 `spacing` と派生メタ（JSON）
- `brain_mask.npz`: 3D脳マスク（bool、`np.savez_compressed`）。将来的にRLEに置換可
- `candidates.csv`: 候補点リスト（`study_id,z,y,x,score0..n`）
- `meta.csv`: 各症例の `n_slices, shape, vendor` 等の統計

注意: ファイル名・列名は固定。Notebook 側は“読むだけ”で、スキーマ逸脱時は失敗させる方針。

## DICOM 幾何と座標系（契約）

- 等方再サンプル: 1.0–1.25 mm を推奨（最終値は `configs/inference/*.yaml` で固定）
- 強度正規化: HU クリップ（例: [-100, 300]）→ z-score（脳マスクで限定可能）
- 座標系: 体積のボクセル座標 `z, y, x`
- voxel→mm 変換: `mm = voxel * spacing + origin`（`origin` は DICOM Series origin を採用）
- Orientation: `ImageOrientationPatient`/`ImagePositionPatient` に基づき、LPS 系の一貫性を確保
- 欠損・不整合: 欠損時はフェイルファスト。`tests/test_dicom_geometry.py` で単体検証

詳細な提出フォーマットの契約は `docs/SUBMISSION_CONTRACT.md` を参照。

## CV 分割とリーク防止

- 患者単位 split を CSV 化してバージョン管理（DVC/W&B artifact）
- サンプル: `data/processed/cv_fold_assign.example.csv`
- 外部データ混在時は `GroupKFold`（施設/装置 ID を group 列で指定）

## 倫理配慮・バイアス

- 医用画像データの取り扱いは各データ提供元の規約に従う
- 施設差・装置差のばらつきを評価し、過適合や非公平性を抑制

