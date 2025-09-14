"""提出 CSV の軽量検証（スケルトン）。
- 実装は書かない。コメントで検証観点のみを明記。

検証観点（仕様確定後に更新）:
- 列名/順序: REQ_COLS に一致（追加列は許容/非許容の方針も明記）
- dtype: id/label は文字列 or 整数、confidence は浮動小数、座標は数値
- NaN/Inf: いずれの列にも存在しないこと
- 値域: confidence ∈ [0,1] など、座標の下限/上限（データ座標系に依存）
- ID 形式: 正規表現でバリデーション（例: \d+_\d+ など、コンペ仕様に従う）
- 重複: id 単位や（id,x,y,z,label）タプル単位で重複が無いこと

使い方:
    python tools/verify_submission.py /path/to/submission.csv
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
