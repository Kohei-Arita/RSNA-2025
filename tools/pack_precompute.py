"""前計算成果物を Kaggle Dataset 用に梱包するスクリプト（スケルトン）。
- 実装は書かない。コメントで構成と項目の最小指針のみ残す。

想定入力（後続で仕様確定）:
- data/processed/precompute/
  - <case_id>/
    - volume.npz  # 再サンプル済みボリューム（コメント: 仕様確定後に拡張子/形式を決定）
    - brain_mask.npz  # 脳マスク
    - candidates.csv  # 候補点（x,y,z,score 等）

想定出力:
- dist/rsna2025-precompute/
  - <case_id>/...
  - _meta.json  # バージョン/作成日時/座標系などのメタ情報

メモ:
- Kaggle ではインターネット OFF のため、この Dataset 単体で 12h 以内に完走できる粒度にする
"""
from __future__ import annotations
import os
import json

SRC = "data/processed/precompute"  # TODO: 後続で確定
DST = "dist/rsna2025-precompute"

os.makedirs(DST, exist_ok=True)

# 最小メタ情報（後続で更新）
meta = {
    "version": 0,
    "notes": "skeleton only; fill after implementation",
}
with open(os.path.join(DST, "_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"prepared precompute dataset skeleton at {DST}")
