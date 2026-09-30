"""球台透视变换测试。"""
import numpy as np
import pytest

from pingpong_analyst.core.table_geometry import (
    TABLE_LENGTH_M,
    TABLE_WIDTH_M,
    TableGeometry,
    landing_label,
    net_placement_label,
    table_displacement_m,
    table_speed_kmh,
)


def test_table_geometry_maps_corners_to_unit_square():
    geometry = TableGeometry([[100, 100], [500, 120], [540, 320], [80, 300]])

    mapped = np.array([
        geometry.transform_point(point)
        for point in geometry.corners
    ])
    expected = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert np.allclose(mapped, expected, atol=1e-6)


def test_unconfigured_geometry_keeps_image_coordinates():
    geometry = TableGeometry()
    assert geometry.transform_point((12, 34)) == (12.0, 34.0)


def test_table_displacement_uses_ittf_length_and_width():
    # 半台长 = 1.37 m，不是半台宽
    assert table_displacement_m((0.0, 0.5), (0.5, 0.5)) == pytest.approx(TABLE_LENGTH_M / 2)
    assert table_displacement_m((0.5, 0.0), (0.5, 1.0)) == pytest.approx(TABLE_WIDTH_M)


def test_table_speed_kmh_from_known_dt():
    # 沿台长飞过全台 0.1 秒 → 2.74 m / 0.1 s = 98.64 km/h
    assert table_speed_kmh((0.0, 0.5), (1.0, 0.5), 0.1) == pytest.approx(
        TABLE_LENGTH_M / 0.1 * 3.6
    )


def test_table_speed_rejects_invalid_inputs():
    assert table_speed_kmh((0.0, 0.5), (1.0, 0.5), 0.0) is None
    assert table_speed_kmh(None, (1.0, 0.5), 0.1) is None
    assert table_speed_kmh((0.0, 0.5), (float("nan"), 0.5), 0.1) is None
    # 0.01 秒飞过全台 ≈ 986 km/h，超出合理上限
    assert table_speed_kmh((0.0, 0.5), (1.0, 0.5), 0.01) is None


def test_net_placement_label_by_width_lane():
    assert net_placement_label((0.5, 0.1)) == "过网 · 左路"
    assert net_placement_label((0.52, 0.5)) == "过网 · 中路"
    assert net_placement_label((0.48, 0.9)) == "过网 · 右路"
    assert net_placement_label((0.2, 0.5)) is None


def test_landing_label_uses_half_depth_and_lane():
    assert landing_label((0.72, 0.2)) == "右半台 · 中台 · 左路"
    assert landing_label((0.08, 0.5)) == "左半台 · 底线 · 中路"
    assert landing_label((0.58, 0.9)) == "右半台 · 近网 · 右路"
