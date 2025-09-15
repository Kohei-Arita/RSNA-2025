<!--
exp0002_candidates（コメントのみ）
- 目標: 候補生成器の確定（軽量 3D U-Net or CenterNet-like）
- 成果物: candidates.csv（z,y,x,score）, volume.npz（float16, 等方）
- 搬入: Kaggle Dataset rsna2025-precompute として Add data（≦20GB 目安）
- 検討: スライス厚/PixelSpacing の揺れに対する正規化、施設ID層化CV
- 実装は行わない（骨組みと運用のみ固定）
-->
