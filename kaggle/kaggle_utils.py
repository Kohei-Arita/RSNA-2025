"""Kaggle 提出ユーティリティ（最小実装）
- 依存は標準ライブラリ + numpy のみに限定
- ベースライン（メタ特徴→GBM）のためのヘルパー
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, List
import os
import json
import pickle
from pathlib import Path
import numpy as np

KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORK = "/kaggle/working"

# 14 ラベル（README/inference.ipynb と同期）
LABEL_COLS: List[str] = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]


def load_gbm_and_metadata(models_dir: str) -> tuple[Any, Dict[str, Any]]:
    """gbm_baseline.pkl と metadata.json を読み込む。
    - models_dir 例: "/kaggle/input/rsna2025-weights/exp0001_baseline"
    """
    models_path = Path(models_dir)
    with open(models_path / 'gbm_baseline.pkl', 'rb') as f:
        gbm = pickle.load(f)
    with open(models_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    return gbm, metadata


def build_feature_row_from_series(row: Dict[str, Any], mod_columns: List[str], age_median: float) -> np.ndarray:
    """メタデータ行からベースライン特徴量を numpy 配列で作成。
    順序は [age, sex, mod_* ...]（metadata['mod_columns'] の順）に一致させる。
    """
    age_str = str(row.get('PatientAge', ''))
    try:
        # "xx - yy" の前段から数値抽出を簡略化
        first = age_str.split(' - ')[0]
        age_val = float(''.join(ch for ch in first if (ch.isdigit() or ch == '.')))
    except Exception:
        age_val = float(age_median)
    if not np.isfinite(age_val):
        age_val = float(age_median)

    sex_val = 1.0 if row.get('PatientSex', '') == 'Male' else 0.0

    modality = row.get('Modality', '')
    mod_feats: List[float] = []
    for m in mod_columns:
        mod_feats.append(1.0 if m == f"mod_{modality}" else 0.0)

    feats = np.array([age_val, sex_val] + mod_feats, dtype=np.float32)
    return feats.reshape(1, -1)


class CandidateStore:
    """前計算された候補点のローダ（スケルトン）。
    期待フォーマット: 1症例につき {"series_uid": str, "points": [[z,y,x], ...]}
    実装は後続で追加し、ここでは I/O 形だけ整備。
    """
    def __init__(self, precompute_root: str):
        self.root = precompute_root

    def load_points(self, series_id: str) -> np.ndarray:
        """候補点配列 (K,3) を返す。未実装時は空配列。
        """
        return np.zeros((0, 3), dtype=np.float32)

    def iter_series_ids(self) -> Iterable[str]:
        """前計算ディレクトリから series_id の列挙を返す（暫定）。"""
        return []

