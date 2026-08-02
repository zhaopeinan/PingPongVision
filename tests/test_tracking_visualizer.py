"""实时追踪可视化元数据测试。"""

import numpy as np

from pingpong_analyst.core.rally_detector import RallyState
from pingpong_analyst.tracking_visualizer import TrackingVisualizer


class FakeTrackNet:
    is_loaded = True
    is_cv_fallback = False

    def __init__(self):
        self.positions = iter([(200.0, 180.0), (220.0, 180.0)])

    def detect_ball_position(self, _frame):
        return next(self.positions)

    def notify_rally_end(self):
        pass


def test_live_rally_count_includes_active_rally():
    visualizer = TrackingVisualizer(mode="rally")
    visualizer._tracknet = FakeTrackNet()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    _annotated, first = visualizer.process_frame(frame, fps=30.0)
    _annotated, second = visualizer.process_frame(frame, fps=30.0)

    assert first["rally_count"] == 0
    assert second["rally_state"] == RallyState.RALLY_ACTIVE.value
    assert second["rally_count"] == 1
