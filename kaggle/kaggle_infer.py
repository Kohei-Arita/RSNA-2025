"""
Kaggle 提出用の最小推論スクリプト（スケルトン）。
- ここではフローと I/O の形だけ定義し、実装は後続ステップで追加。
- 出力: /kaggle/working/submission.csv（環境変数 WORK_DIR で上書き可）
"""
from __future__ import annotations
import os
from dataclasses import dataclass

WORK_DIR = os.getenv("WORK_DIR", "/kaggle/working")
INPUT_DIR = os.getenv("INPUT_DIR", "/kaggle/input")
PRECOMP_DIR = os.path.join(INPUT_DIR, "rsna2025-precompute")
WEIGHTS_DIR = os.path.join(INPUT_DIR, "rsna2025-weights")
SUBMIT_PATH = os.path.join(WORK_DIR, "submission.csv")

@dataclass
class InferenceCfg:
    # TODO: 必要な設定（閾値、バッチサイズ等）を後続で追加
    pass


def main() -> None:
    """推論のエントリポイント（ダミー）。
    - 実装は後続で: 前計算の候補点読込 → パッチ推論 → NMS → CSV出力
    - ここでは空の submission.csv を生成するのみ（列は後続で定義）
    """
    os.makedirs(os.path.dirname(SUBMIT_PATH), exist_ok=True)
    with open(SUBMIT_PATH, "w") as f:
        f.write("")


if __name__ == "__main__":
    main()
