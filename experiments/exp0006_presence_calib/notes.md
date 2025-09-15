# exp0006_presence_calib（コメントのみ）

- 目的: presence ヘッドの損失重み（class-balanced BCE + focal γ）と OOF 校正（温度スケーリング/等分位）を比較。
- 方針: 患者単位CVを固定し、presence の再現率重視で閾値最適化。モダリティ別閾値は `configs/inference/modality_thresholds.yaml` に反映。
- 成果物: OOF曲線、閾値、校正パラメータ。

（実装は後続・このファイルはメモ用）
