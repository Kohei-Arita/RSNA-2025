# exp0007_2p5d_mainline（コメントのみ）

- 目的: 2.5D を第一経路に据え、3D は候補パッチ補助に限定。
- 推論: AMP + TorchScript/ONNX + dynamic quantization。時間ガード（解像度→TTA→stride→候補数）併用。
- 計測: presence 優先の指標、実行時間、ETA の挙動を記録。

（メモのみ）
