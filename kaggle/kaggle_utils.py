"""Kaggle 提出ユーティリティ（最小スケルトン）
- 依存を最小化するため、実装は後続ステップで追加
"""
from __future__ import annotations
from typing import Any, Iterable
import os
import json
import numpy as np

KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORK = "/kaggle/working"

class CandidateStore:
    """前計算された候補点のローダ（スケルトン）。
    期待フォーマット: 1症例につき {"series_uid": str, "points": [[z,y,x], ...]}
    実装は後続で追加し、ここでは I/O 形だけ整備。
    """
    def __init__(self, precompute_root: str):
        self.root = precompute_root
        # TODO: 例外やログ方針は後続実装で定義

    def load_points(self, series_id: str) -> np.ndarray:
        """候補点配列 (K,3) を返す。未実装時は空配列。
        """
        # TODO: npz/json を探して返す実装を追加
        return np.zeros((0, 3), dtype=np.float32)

    def iter_series_ids(self) -> Iterable[str]:
        """前計算ディレクトリから series_id の列挙を返す（暫定）。"""
        # TODO: 前計算ディレクトリから列挙
        return []
