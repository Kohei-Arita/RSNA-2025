"""
Kaggle 提出用の最小推論スクリプト（ベースライン実装）。
- 本番は Serving API でシリーズ単位の 14 ラベル確率 [0,1] を応答。
- ここでは評価APIの代替として、STDIN/STDOUT で JSON 行を受け取り・返す簡易 serve を提供。
- さらに --dry でローカル Dry-run（submission.csv 生成のみ）にも対応。

使い方（Kaggle Notebook 概念図）:
  python kaggle/kaggle_infer.py --serve   # 起動→15分以内に待受開始
  # STDIN で {"SeriesInstanceUID": ..., "PatientAge": ..., "PatientSex": ..., "Modality": ...} を1行/JSONで送ると
  # STDOUT で {"SeriesInstanceUID": ..., "probs": {label: prob, ...}} を返す

補足:
- 学習済み GBM（gbm_baseline.pkl）と metadata.json（mod_columns 等）を
  /kaggle/input/rsna2025-weights/exp0001_baseline からロードする想定。
  必要に応じて環境変数 MODELS_SUBDIR でサブディレクトリ名を上書き可能。

時間ガード（設計のみ）:
- 将来的に ETA に基づく自動ダウングレード（解像度→TTA→stride→候補数）を組み込む。
"""
from __future__ import annotations
import os
import sys
import json
from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path

import numpy as np

from kaggle.kaggle_utils import (
    LABEL_COLS,
    load_gbm_and_metadata,
    build_feature_row_from_series,
)

WORK_DIR = os.getenv("WORK_DIR", "/kaggle/working")
INPUT_DIR = os.getenv("INPUT_DIR", "/kaggle/input")
PRECOMP_DIR = os.path.join(INPUT_DIR, "rsna2025-precompute")
WEIGHTS_DIR = os.path.join(INPUT_DIR, "rsna2025-weights")
MODELS_SUBDIR = os.getenv("MODELS_SUBDIR", "exp0001_baseline")
SUBMIT_PATH = os.path.join(WORK_DIR, "submission.csv")

@dataclass
class InferenceCfg:
    # TODO: 必要な設定（閾値、バッチサイズ等）を後続で追加
    pass


def predict_labels_from_meta(gbm: Any, metadata: Dict[str, Any], series_meta: Dict[str, Any]) -> Dict[str, float]:
    """シリーズメタ情報から 14 ラベルの確率辞書を返す（ベースライン）。"""
    mod_columns = metadata.get('mod_columns') or metadata.get('mod_columns'.upper()) or metadata.get('MOD_COLUMNS')
    if mod_columns is None:
        raise RuntimeError("metadata['mod_columns'] が見つかりません")
    age_median = float(metadata.get('age_median', 60))
    feats = build_feature_row_from_series(series_meta, mod_columns, age_median)
    prob_presence = float(gbm.predict_proba(feats)[:, 1][0])
    # 簡略: 全ラベルに presence と同一確率を割当
    return {label: prob_presence for label in LABEL_COLS}


def _serve_loop(gbm: Any, metadata: Dict[str, Any]) -> None:
    """STDIN/STDOUT ベースの簡易サービングループ。
    入力: 1行に1 JSON: {"SeriesInstanceUID": str, "PatientAge": str, "PatientSex": str, "Modality": str}
    出力: 1行に1 JSON: {"SeriesInstanceUID": str, "probs": {label: float, ...}}
    """
    sys.stdout.reconfigure(line_buffering=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            series_id = payload.get("SeriesInstanceUID") or payload.get("series_id")
            probs = predict_labels_from_meta(gbm, metadata, payload)
            out = {"SeriesInstanceUID": series_id, "probs": probs}
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {"error": str(e)}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()


def _dry_run_csv(gbm: Any, metadata: Dict[str, Any]) -> None:
    """サンプル数件のダミー行で submission.csv を生成（Dry-run用）。"""
    os.makedirs(os.path.dirname(SUBMIT_PATH), exist_ok=True)
    sample_series = [
        {"SeriesInstanceUID": "ID_0001", "PatientAge": "60 - 65", "PatientSex": "Male", "Modality": "CT"},
        {"SeriesInstanceUID": "ID_0002", "PatientAge": "45 - 50", "PatientSex": "Female", "Modality": "MR"},
    ]
    with open(SUBMIT_PATH, "w") as f:
        # Kaggle の Serving 本番では CSV を使わないが、Dry-run 用に列を固定
        header = ["SeriesInstanceUID"] + LABEL_COLS
        f.write(",".join(header) + "\n")
        for row in sample_series:
            probs = predict_labels_from_meta(gbm, metadata, row)
            values = [row["SeriesInstanceUID"]] + [f"{probs[label]:.6f}" for label in LABEL_COLS]
            f.write(",".join(values) + "\n")


if __name__ == "__main__":
    # 引数解釈（最小）
    args = set(sys.argv[1:])
    models_dir = os.path.join(WEIGHTS_DIR, MODELS_SUBDIR)
    gbm, metadata = load_gbm_and_metadata(models_dir)
    if "--serve" in args:
        _serve_loop(gbm, metadata)
    else:
        # 既定は Dry-run CSV を生成（Notebook 手動実行やローカル確認用）
        _dry_run_csv(gbm, metadata)
