"""球台标定数据和值存储测试。"""

import pytest

from pingpong_analyst.core.table_calibration import TableCalibration, TableCalibrationStore
from pingpong_analyst.core.table_geometry import TableGeometry


def valid_calibration():
    return TableCalibration(
        frame_index=12,
        corners=((10, 20), (630, 25), (620, 340), (15, 335)),
        net_points=((320, 22), (320, 338)),
    )


def test_table_calibration_round_trips(tmp_path):
    store = TableCalibrationStore(tmp_path)
    calibration = valid_calibration()

    store.save("video_1", calibration)
    loaded = store.load("video_1")

    assert loaded == calibration
    geometry = TableGeometry.from_calibration(loaded)
    assert geometry.transform_point(calibration.corners[0]) == pytest.approx((0.0, 0.0))
    assert geometry.transform_point(calibration.corners[2]) == pytest.approx((1.0, 1.0))


@pytest.mark.parametrize(
    "corners",
    [
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (1, 0), (0, 1)),
        ((0, 0), (1, 0), (1, 1), (0, 1.0)),
    ],
)
def test_table_calibration_rejects_invalid_corners(corners):
    if len(corners) == 4 and corners[-1] == (0, 1.0):
        corners = ((0, 0), (1, 0), (0, 1), (1, 1))
    with pytest.raises(ValueError):
        TableCalibration(frame_index=0, corners=corners)


def test_table_calibration_rejects_invalid_net_point_count():
    with pytest.raises(ValueError):
        TableCalibration(
            frame_index=0,
            corners=((0, 0), (10, 0), (10, 10), (0, 10)),
            net_points=((5, 0),),
        )


def test_store_delete_returns_none_after_delete(tmp_path):
    store = TableCalibrationStore(tmp_path)
    store.save("video_1", valid_calibration())
    store.delete("video_1")
    assert store.load("video_1") is None
