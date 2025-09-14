import pytest

# スケルトン（実装完了までスキップ）
pytestmark = pytest.mark.skip(reason="geometry tests to be implemented in later step")

# 幾何テストの方針（コメント、実装は後続）：
# - RescaleSlope/Intercept の適用確認（DICOM ピクセル値→HU 変換の往復性）
# - 非等方 voxel（異方性解像度）のリサンプル前後での座標整合
# - 方向行列（ImageOrientationPatient）を考慮した index↔patient座標の往復
# - 座標系（LPS/RAS）の取り扱いを明示し、ラベル/候補点の整合を検証

def test_placeholder_geometry_roundtrip():
    assert True
