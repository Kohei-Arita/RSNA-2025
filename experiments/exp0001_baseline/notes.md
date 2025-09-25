# exp0001_baseline 実験メモ

## 概要
初期ベースライン実験 - GradientBoostingClassifierを使用した単純な分類

## 実験日時
2024年12月

## 使用した特徴量
- 年齢（数値、中央値で欠損値補完）
- 性別（バイナリ、Male=1）
- モダリティ（one-hotエンコード）

## モデル
- GradientBoostingClassifier
- パラメータ：n_estimators=100, learning_rate=0.1, max_depth=8

## 結果
- Validation AUC: 記録予定
- OOF AUC: 記録予定

## 所感/改善点
- 画像データを使わない最小限のベースライン
- 次のステップ：画像特徴量の追加、深層学習モデルの検討

## ファイル構成
- `training.ipynb`: モデル学習
- `evaluation.ipynb`: OOF評価と閾値最適化
- `inference.ipynb`: 推論と予測生成
- `config.yaml`: 実験設定のスナップショット

## W&Bリンク
（記録予定）
