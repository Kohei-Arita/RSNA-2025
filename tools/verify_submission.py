"""提出 CSV の軽量検証（スケルトン）。
- 列や値域の仕様は後続で確定
"""
from __future__ import annotations
import sys

REQ_COLS = ["id", "confidence", "x", "y", "z", "label"]  # TODO: 確定後更新


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python tools/verify_submission.py path/to/submission.csv")
        return 1
    # TODO: 実装（CSV 読み込み→列検証→値域検証）
    print("OK (skeleton)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
