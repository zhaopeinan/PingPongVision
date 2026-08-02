"""球台透视变换测试。"""
import numpy as np

from pingpong_analyst.core.table_geometry import TableGeometry


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

