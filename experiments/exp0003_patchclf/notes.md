<!--
exp0003_patchclf（コメントのみ）
- 目標: 候補→パッチ分類の精度最適化（AMP/channels-last/勾配スケーリング）
- ダウングレード順: TTA停止 → stride粗化 → 候補N → 解像度
- 評価: OOF AUC/Sensitivity、ケース単位の検出感度を主指標に
- 実装は行わない（骨組みと契約のみ）
-->
