## 提出CSV 仕様契約（Contract） v0

この文書は Kaggle 提出用 CSV の最小仕様（列名・dtype・座標系・単位・値域）を定義します。検証は `tools/verify_submission.py` で行い、本ドキュメントのルールに準拠します。

### 目的
- 提出フォーマットの早期固定により、推論処理と検証処理を安定化する
- チーム内・Notebook環境での解釈齟齬を防ぎ、オフライン提出を堅牢化する

### CSV スキーマ（v0 案）
- 必須列: `study_id`, `x`, `y`, `z`, `confidence`
- 任意列: なし（将来拡張時は後方互換の列追加のみを許可）

### データ型
- `study_id`: string（正規表現: `^[A-Za-z0-9_.-]+$` を推奨）
- `x`, `y`, `z`: number（float64 推奨）
- `confidence`: number（float32/float64）範囲 [0.0, 1.0]

### 座標系と単位
- 座標系: 体積ボリュームのボクセル座標系（`z, y, x` の順に軸が増加）
- 単位: voxel（整数ではなく小数も可。小数はサブボクセル中心を表す）
- 範囲: `0 <= x < width`, `0 <= y < height`, `0 <= z < depth`
- メモ: mm 単位が必要な場合は、`configs/inference/*.yaml` 側で統一変換を行う。提出は voxel に固定。

### 値域と妥当性
- `confidence` は [0,1] にクリップされていること
- NaN/Inf はいずれの列にも含まれないこと
- 行重複は不可（`study_id,x,y,z` が同一の行を許可しない）

### 行ソート（任意）
- 安全性と比較容易性のため、最終CSVは `study_id` 昇順、続いて `confidence` 降順での並びを推奨

### サンプル
```
study_id,x,y,z,confidence
ID_0001,128.5,96.0,42.0,0.87
ID_0001,220.0,85.5,30.5,0.41
ID_0002,64.0,100.0,55.0,0.76
```

### 検証
- `tools/verify_submission.py` で以下を検査
  - 列名・dtype
  - NaN/Inf の有無
  - `confidence` の値域 [0,1]
  - `study_id` のID形式（正規表現）
  - 重複行の禁止
  - 可能であれば座標レンジ（体積形状が分かる場合に限る）

### 変更管理
- 本ドキュメントの改訂は PR ベースで行い、`tools/verify_submission.py` にも同時反映する
- 大きな互換性変更（列名・単位）は原則禁止。必要時は `v1` として明示（Notebook側の移行を伴う）


