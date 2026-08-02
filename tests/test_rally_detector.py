"""测试回合检测状态机"""
import numpy as np
import pytest

from pingpong_analyst.core.data_aligner import DataAligner, FrameData, HitEvent
from pingpong_analyst.core.rally_detector import RallyDetector, RallyState, RallySegment


@pytest.fixture
def rally_config():
    return {
        "min_boards": 3,
        "ball_speed_threshold": 5.0,
        "rally_timeout_frames": 10,
        "pickup_duration_threshold": 5,
    }


@pytest.fixture
def detector(rally_config):
    return RallyDetector(rally_config)


def make_frame(idx, timestamp, ball_pos=None, ball_speed=0.0, persons=None):
    """构造测试用 FrameData"""
    return FrameData(
        frame_idx=idx,
        timestamp=timestamp,
        ball_pos=ball_pos,
        ball_speed=ball_speed,
        persons=persons or [],
    )


def test_initial_state_idle(detector):
    assert detector.state == RallyState.IDLE


def test_rally_start_on_ball_movement(detector):
    """球开始运动 -> 进入回合"""
    fd = make_frame(0, 0.0, ball_pos=(100, 200), ball_speed=10.0)
    result = detector.update(fd)
    assert detector.state == RallyState.RALLY_ACTIVE
    assert result is None  # 回合未结束


def test_rally_end_on_timeout(detector):
    """球静止超时 -> 回合结束"""
    # 开始回合
    fd = make_frame(0, 0.0, ball_pos=(100, 200), ball_speed=10.0)
    detector.update(fd)

    # 模拟击球事件
    hits = [HitEvent(frame_idx=0, timestamp=0.0, ball_pos=(100, 200),
                     hitter_side="left", arm_angles={})]
    hits.append(HitEvent(frame_idx=5, timestamp=0.15, ball_pos=(200, 200),
                         hitter_side="right", arm_angles={}))
    hits.append(HitEvent(frame_idx=10, timestamp=0.3, ball_pos=(100, 200),
                         hitter_side="left", arm_angles={}))

    # 球静止直到超时
    for i in range(1, 25):
        fd = make_frame(i, i * 0.033, ball_pos=(100, 200), ball_speed=0.0)
        detector.update(fd, hits if i == 1 else None)

    assert detector.state == RallyState.IDLE


def test_low_board_count_filtered(rally_config):
    """板数不足的回合被过滤"""
    rally_config["min_boards"] = 10  # 高阈值
    detector = RallyDetector(rally_config)

    fd = make_frame(0, 0.0, ball_pos=(100, 200), ball_speed=10.0)
    detector.update(fd, [HitEvent(0, 0.0, (100, 200), "left", {})])

    for i in range(1, 25):
        fd = make_frame(i, i * 0.033, ball_speed=0.0)
        result = detector.update(fd)

    assert len(detector.get_valid_segments()) == 0


def test_valid_rally_detected(rally_config):
    """有效回合被检测到"""
    rally_config["min_boards"] = 2
    detector = RallyDetector(rally_config)

    hits = [
        HitEvent(0, 0.0, (100, 200), "left", {}),
        HitEvent(3, 0.1, (200, 200), "right", {}),
        HitEvent(6, 0.2, (100, 200), "left", {}),
    ]

    fd = make_frame(0, 0.0, ball_pos=(100, 200), ball_speed=10.0)
    detector.update(fd, hits)

    for i in range(1, 25):
        fd = make_frame(i, i * 0.033, ball_speed=0.0)
        detector.update(fd)

    segments = detector.get_valid_segments()
    assert len(segments) == 1
    assert segments[0].board_count == 3
    assert segments[0].is_valid


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
    })
    detector.update(make_frame(0, 0.0, ball_pos=(100, 200), ball_speed=10.0), [
        HitEvent(0, 0.0, (100, 200), "left", {})
    ])
    for i in range(1, 6):
        detector.update(make_frame(i, i * 0.033, ball_speed=0.0))

    assert detector.get_valid_segments()
    detector.reset()

    assert detector.state == RallyState.IDLE
    assert detector.get_valid_segments() == []
    assert detector._current_hit_events == []
