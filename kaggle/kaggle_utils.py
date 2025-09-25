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
    """学習済みGBMとメタデータを読み込む。
    優先順位:
      1) 環境変数 MODEL_PATH, METADATA_PATH があればそれを使用
      2) models_dir 配下で既知の候補名を探索（ファイル名バリアント対応）
    既知の候補:
      - モデル: ["gbm_baseline.pkl", "GBM Baseline.pkl", "GBM_Baseline.pkl"]
      - メタ:   ["metadata.json", "Metadata.json", "Complete Workflow Metadata.json", "Complete_Workflow_Metadata.json"]
    """
    env_model = os.getenv('MODEL_PATH')
    env_meta = os.getenv('METADATA_PATH')
    if env_model and env_meta:
        with open(env_model, 'rb') as f:
            gbm = pickle.load(f)
        with open(env_meta, 'r') as f:
            metadata = json.load(f)
        return gbm, metadata

    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"models_dir not found: {models_dir}")

    model_candidates = [
        'gbm_baseline.pkl',
        'GBM Baseline.pkl',
        'GBM_Baseline.pkl',
    ]
    meta_candidates = [
        'metadata.json',
        'Metadata.json',
        'Complete Workflow Metadata.json',
        'Complete_Workflow_Metadata.json',
    ]

    def _find_first(path: Path, names: list[str]) -> Path | None:
        for name in names:
            p = path / name
            if p.exists():
                return p
        return None

    model_path = _find_first(models_path, model_candidates)
    meta_path = _find_first(models_path, meta_candidates)
    if model_path is None:
        raise FileNotFoundError(f"Model file not found under {models_dir}. Tried: {model_candidates}")
    if meta_path is None:
        raise FileNotFoundError(f"Metadata file not found under {models_dir}. Tried: {meta_candidates}")

    with open(model_path, 'rb') as f:
        gbm = pickle.load(f)
    with open(meta_path, 'r') as f:
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

