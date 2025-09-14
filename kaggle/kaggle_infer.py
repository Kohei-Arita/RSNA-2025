"""
Kaggle 提出用の最小推論スクリプト（スケルトン）。
- 実装は書かない。コメントで最小限の指針のみ残す。
- 出力: /kaggle/working/submission.csv（環境変数 WORK_DIR で上書き可）

実装方針（コメントのみ）:
- 前計算（/kaggle/input/rsna2025-precompute）から候補点とメタをロード
- 3D パッチ切り出し → モデル推論 → NMS → CSV 出力
- 時間ガード: ETA を推定し、上限（≒12h）超過が見えたら段階的にダウングレード
  1) TTA 停止 2) パッチストライドを粗く 3) 候補数上限を縮小 4) 入力解像度を縮小
  ※ “完走最優先” を原則とし、上記は自動で切替える
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
    - 実装は後続で: 前計算の候補点読込 → パッチ推論 → NMS → CSV 出力
    - ここでは空の submission.csv を生成するのみ（列は後続で定義）

    # 注意: Kaggle はインターネット OFF が前提。追加依存は wheel を Add data で供給
    """
    os.makedirs(os.path.dirname(SUBMIT_PATH), exist_ok=True)
    with open(SUBMIT_PATH, "w") as f:
        f.write("")


if __name__ == "__main__":
    main()
