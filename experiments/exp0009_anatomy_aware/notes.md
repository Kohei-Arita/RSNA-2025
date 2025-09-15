# exp0009_anatomy_aware（コメントのみ）

- 目的: 13解剖学的ロケーション（血管領域マップ/セグ）を補助信号として活用し、
  - 部位別ヘッドの監督（ROIプーリング/マスク付きAttention）
  - NMS/一致判定の再重み付け（mm + 部位マスク内優先）
  を試す追加実験。コード本線は変更しない（実験側で有効化）。

- 背景: RSNA公式は13部位の検出・局在化を明記。一部MRIで3Dセグ提供。
  - 提供セグは弱教師/補助Lossとして扱う。
  - 最終提出の契約（series_id + 14列の確率）は不変。

- 設定メモ（実装は後続）:
  - data.use_anatomy_masks: true で補助を有効化
  - train.losses に aux_anatomy_roi_loss を追加（weight≈0.2-0.4 を探索）
  - inference.nms に anatomy_mask_priority を追加（距離は mm 単位固定）

- CV/評価:
  - 患者単位 + 施設/モダリティ GroupKFold の併用も検討
  - 部位別感度/適合率、mm 一致判定（半径固定）

- ルール遵守:
  - 外部データ/事前学習の可否は Rules 最優先。利用時は docs/DATASET_CARD.md に出所/ライセンス/再現手順を明記。

- 時間ガード:
  - Kaggle は GPU 9h 基準。追加計算はオプショナルで無効化可にし、時間予算に応じて縮退可能な設計。
