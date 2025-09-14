"""前計算成果物を Kaggle Dataset 用に梱包するスクリプト（スケルトン）。
- 入力: data/processed などの前計算（後続で仕様確定）
- 出力: dist/rsna2025-precompute/ に症例単位のファイル群を配置
- ここではディレクトリ作成と最小メタのみ作成。
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
