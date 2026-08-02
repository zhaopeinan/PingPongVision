"""测试数据对齐器 (核心: 人-球时空对齐)"""
import numpy as np
import pytest

from pingpong_analyst.core.data_aligner import DataAligner, FrameData, HitEvent


@pytest.fixture
def aligner():
    config = {
        "ball_speed_threshold": 5.0,
        "turning_point_window": 3,
        "hit_match_window": 5,
    }
    return DataAligner(config)


def test_add_frame_basic(aligner):
    """测试帧数据添加与同步"""
    fd = aligner.add_frame(
        frame_idx=0,
        timestamp=0.0,
        ball_pos=(100, 200),
        ball_confidence=0.9,
        persons=[{"bbox": [50, 100, 150, 300], "keypoints": None, "conf": 0.8}],
    )
    assert fd.frame_idx == 0
    assert fd.ball_pos == (100, 200)
    assert len(fd.persons) == 1


def test_ball_speed_calculation(aligner):
    """测试球速计算"""
    # 帧0: 球在 (100, 200)
    aligner.add_frame(frame_idx=0, timestamp=0.0, ball_pos=(100, 200))
    # 帧1: 球移动到 (110, 200) -> 速度=10
    fd = aligner.add_frame(frame_idx=1, timestamp=0.033, ball_pos=(110, 200))
    assert fd.ball_speed == pytest.approx(10.0, abs=0.1)
    assert fd.ball_direction is not None


def test_ball_speed_no_movement(aligner):
    """球静止时速度为0"""
    aligner.add_frame(frame_idx=0, timestamp=0.0, ball_pos=(100, 200))
    fd = aligner.add_frame(frame_idx=1, timestamp=0.033, ball_pos=(100, 200))
    assert fd.ball_speed == 0.0


def test_turning_point_detection(aligner):
    """测试折返点检测"""
    # 模拟球向右运动后折返向左 (需足够帧数满足 window*2+1 缓冲)
    positions = [
        (100, 200),  # 向右
        (120, 200),
        (140, 200),
        (160, 200),  # 折返点
        (140, 200),  # 向左
        (120, 200),
        (100, 200),
        (80, 200),
        (60, 200),
    ]
    for i, pos in enumerate(positions):
        aligner.add_frame(frame_idx=i, timestamp=i * 0.033, ball_pos=pos)

    tps = aligner.detect_turning_points()
    # 应该检测到折返点在帧3附近
    assert len(tps) > 0


def test_turning_point_no_turn(aligner):
    """直线运动无折返点"""
    for i in range(10):
        aligner.add_frame(
            frame_idx=i, timestamp=i * 0.033, ball_pos=(100 + i * 20, 200)
        )
    tps = aligner.detect_turning_points()
    assert len(tps) == 0


def test_hit_event_matching(aligner):
    """测试击球事件匹配"""
    # 球折返 + 附近有人 (需足够帧数满足 window*2+1 缓冲)
    positions = [
        (100, 200), (120, 200), (140, 200),
        (160, 200),  # 折返点
        (140, 200), (120, 200), (100, 200),
        (80, 200), (60, 200),
    ]
    for i, pos in enumerate(positions):
        person = {
            "bbox": [130, 150, 170, 250],
            "keypoints": np.zeros((17, 3)),
            "conf": 0.8,
        }
        aligner.add_frame(
            frame_idx=i,
            timestamp=i * 0.033,
            ball_pos=pos,
            persons=[person],
            arm_angles={"left_elbow_angle": 150, "right_elbow_angle": 100},
        )

    events = aligner.match_hit_events()
    assert len(events) > 0
    assert isinstance(events[0], HitEvent)
    assert events[0].hit_type in ("drive", "spin", "unknown")


def test_hit_event_uses_matched_person_angles(aligner):
    """击球分类应使用被选中的人的角度，而不是 FrameData 摘要。"""
    positions = [
        (100, 200), (120, 200), (140, 200), (160, 200),
        (140, 200), (120, 200), (100, 200), (80, 200), (60, 200),
    ]
    for i, pos in enumerate(positions):
        aligner.add_frame(
            frame_idx=i,
            timestamp=i * 0.033,
            ball_pos=pos,
            persons=[{
                "bbox": [130, 150, 170, 250],
                "keypoints": np.zeros((17, 3)),
                "conf": 0.8,
                "arm_angles": {"left_elbow_angle": 160, "right_elbow_angle": 155},
            }],
            arm_angles={"left_elbow_angle": 100, "right_elbow_angle": 105},
        )

    events = aligner.match_hit_events()
    assert events
    assert events[0].hit_type == "drive"


def test_ball_only_hit_events_do_not_require_persons(aligner):
    """回合模式只依赖球轨迹，不应触发人体角度或击球者分析。"""
    positions = [
        (100, 200), (120, 200), (140, 200), (160, 200),
        (140, 200), (120, 200), (100, 200), (80, 200), (60, 200),
    ]
    for i, pos in enumerate(positions):
        aligner.add_frame(
            frame_idx=i,
            timestamp=i * 0.033,
            ball_pos=pos,
            persons=[],
            arm_angles=None,
        )

    events = aligner.match_hit_events(require_persons=False)
    assert events
    assert events[0].hitter_side == "unknown"
    assert events[0].hit_type == "unknown"
    assert events[0].arm_angles == {}


def test_infer_hit_type_drive():
    """大肘角度 -> 撞击"""
    angles = {"left_elbow_angle": 160, "right_elbow_angle": 155}
    assert DataAligner._infer_hit_type(angles) == "drive"


def test_infer_hit_type_spin():
    """小肘角度 -> 摩擦"""
    angles = {"left_elbow_angle": 100, "right_elbow_angle": 110}
    assert DataAligner._infer_hit_type(angles) == "spin"


def test_infer_hit_type_unknown():
    """无角度数据 -> unknown"""
    assert DataAligner._infer_hit_type(None) == "unknown"
