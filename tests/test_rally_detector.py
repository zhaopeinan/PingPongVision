"""测试回合检测状态机"""
import numpy as np
import pytest

from pingpong_analyst.core.data_aligner import DataAligner, FrameData, HitEvent
from pingpong_analyst.core.rally_detector import RallyDetector, RallyState, RallySegment


@pytest.fixture
def rally_config():
    return {
        "min_boards": 2,
        "ball_speed_threshold": 5.0,
        "rally_timeout_frames": 10,
        "pickup_duration_threshold": 5,
        "no_crossing_timeout_seconds": 1.0,
    }


@pytest.fixture
def detector(rally_config):
    return RallyDetector(rally_config)


def make_frame(idx, timestamp, ball_pos=None, ball_speed=0.0, persons=None, ball_table_pos=None, calibrated=False):
    """构造测试用 FrameData"""
    return FrameData(
        frame_idx=idx,
        timestamp=timestamp,
        ball_pos=ball_pos,
        ball_speed=ball_speed,
        persons=persons or [],
        ball_table_pos=ball_table_pos,
        calibrated=calibrated,
    )


def test_initial_state_idle(detector):
    assert detector.state == RallyState.IDLE


def test_rally_start_on_ball_movement(detector):
    """球进入一侧 -> 开始得分段"""
    fd = make_frame(0, 0.0, ball_pos=(100, 200), ball_speed=10.0, ball_table_pos=(0.2, 0.5))
    result = detector.update(fd)
    assert detector.state == RallyState.RALLY_ACTIVE
    assert result is None  # 回合未结束


def test_rally_end_on_timeout(detector):
    """超过无跨区超时 -> 得分段结束且板数清零"""
    fd = make_frame(0, 0.0, ball_pos=(100, 200), ball_speed=10.0, ball_table_pos=(0.2, 0.5))
    detector.update(fd)

    detector.update(make_frame(1, 0.1, ball_pos=(200, 200), ball_table_pos=(0.8, 0.5)))
    detector.update(make_frame(2, 0.2, ball_pos=(100, 200), ball_table_pos=(0.2, 0.5)))
    result = detector.update(make_frame(3, 1.3))

    assert detector.state == RallyState.IDLE
    assert result is not None
    assert result.board_count == 2
    assert detector.current_board_count == 0
    assert [hit.hitter_side for hit in result.hit_events] == ["left", "right"]


def test_low_board_count_filtered(rally_config):
    """板数不足的回合被过滤"""
    rally_config["min_boards"] = 10  # 高阈值
    detector = RallyDetector(rally_config)

    fd = make_frame(0, 0.0, ball_pos=(100, 200), ball_speed=10.0, ball_table_pos=(0.2, 0.5))
    detector.update(fd)

    for i in range(1, 4):
        fd = make_frame(i, 1.0 + i * 0.1)
        result = detector.update(fd)

    assert len(detector.get_valid_segments()) == 0


def test_valid_rally_detected(rally_config):
    """有效回合被检测到"""
    rally_config["min_boards"] = 2
    detector = RallyDetector(rally_config)

    fd = make_frame(0, 0.0, ball_pos=(100, 200), ball_speed=10.0, ball_table_pos=(0.2, 0.5))
    detector.update(fd, [HitEvent(0, 0.0, (100, 200), "left", {})] * 10)
    detector.update(make_frame(1, 0.1, ball_pos=(200, 200), ball_table_pos=(0.8, 0.5)))
    detector.update(make_frame(2, 0.2, ball_pos=(100, 200), ball_table_pos=(0.2, 0.5)))
    detector.update(make_frame(3, 1.3))

    segments = detector.get_valid_segments()
    assert len(segments) == 1
    assert segments[0].board_count == 2
    assert segments[0].is_valid
    assert [hit.board_index for hit in segments[0].hit_events] == [1, 2]
    assert [hit.hitter_side for hit in segments[0].hit_events] == ["left", "right"]


def test_rally_segment_properties():
    """测试 RallySegment 属性"""
    seg = RallySegment(
        start_frame=10, end_frame=100,
        start_time=0.5, end_time=3.5,
        board_count=8, hit_events=[],
    )
    assert seg.duration == 3.0


def test_reset_clears_previous_rally():
    """新视频开始前不应继承上一视频的回合和击球事件。"""
    detector = RallyDetector({
        "min_boards": 1,
        "rally_timeout_frames": 3,
        "pickup_duration_threshold": 0,
        "no_crossing_timeout_seconds": 0.5,
    })
    detector.update(make_frame(0, 0.0, ball_pos=(100, 200), ball_speed=10.0, ball_table_pos=(0.2, 0.5)))
    detector.update(make_frame(1, 0.1, ball_pos=(200, 200), ball_table_pos=(0.8, 0.5)))
    detector.update(make_frame(2, 0.7))

    assert detector.get_valid_segments()
    assert detector.get_valid_segments()[0].hit_events
    detector.reset()

    assert detector.state == RallyState.IDLE
    assert detector.get_valid_segments() == []
    assert detector._current_hit_events == []


def test_crossing_hit_copies_calibrated_speed_and_placement():
    detector = RallyDetector({
        "min_boards": 1,
        "no_crossing_timeout_seconds": 1.0,
    })
    detector.update(make_frame(0, 0.0, ball_pos=(100, 200), ball_speed=10.0, ball_table_pos=(0.2, 0.5)))
    detector.update(FrameData(
        frame_idx=1,
        timestamp=0.1,
        ball_pos=(200, 200),
        ball_speed=40.0,
        ball_speed_kmh=72.5,
        ball_table_pos=(0.56, 0.2),
        calibrated=True,
    ))
    hit = detector._current_hit_events[-1]
    assert hit.ball_speed == pytest.approx(40.0)
    assert hit.ball_speed_kmh == pytest.approx(72.5)
    assert hit.placement == "过网 · 左路"
    assert hit.calibrated is True
    assert hit.table_pos == (0.56, 0.2)


def test_bounce_landing_updates_hit_after_crossing():
    detector = RallyDetector({
        "min_boards": 1,
        "no_crossing_timeout_seconds": 2.0,
    })
    detector.update(make_frame(
        0, 0.0, ball_pos=(100, 90), ball_speed=10.0, ball_table_pos=(0.2, 0.5), calibrated=True,
    ))
    detector.update(make_frame(
        1, 0.03, ball_pos=(180, 100), ball_speed=40.0, ball_table_pos=(0.56, 0.5), calibrated=True,
    ))
    detector.update(make_frame(
        2, 0.06, ball_pos=(200, 110), ball_table_pos=(0.64, 0.22), calibrated=True,
    ))
    detector.update(make_frame(
        3, 0.09, ball_pos=(220, 120), ball_table_pos=(0.72, 0.22), calibrated=True,
    ))
    detector.update(make_frame(
        4, 0.12, ball_pos=(230, 108), ball_table_pos=(0.78, 0.22), calibrated=True,
    ))
    hit = detector._current_hit_events[-1]
    assert hit.placement_kind == "bounce"
    assert hit.landing_pos == (0.72, 0.22)
    assert hit.placement == "右半台 · 中台 · 左路"
    assert hit.table_pos[0] == pytest.approx(0.56)
