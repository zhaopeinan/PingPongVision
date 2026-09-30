"""跨区击球事件与 YOLO 姿态补全。"""
import numpy as np

from pingpong_analyst.core.data_aligner import HitEvent
from pingpong_analyst.core.hit_enrichment import (
    apply_pose_to_hit,
    arm_angles_from_coco_keypoints,
    crossing_hitter_side,
    hit_frame_indices,
    person_side,
    select_hitter,
)
from pingpong_analyst.core.rally_detector import RallySegment
from pingpong_analyst.core.table_geometry import TableGeometry


def _coco_keypoints(left_elbow_deg=180.0):
    points = np.zeros((17, 3), dtype=np.float64)
    points[:, 2] = 1.0
    points[6] = [40, 10, 1]
    points[8] = [50, 10, 1]
    points[10] = [60, 10, 1]
    points[11] = [10, 40, 1]
    points[12] = [40, 40, 1]
    if left_elbow_deg == 90:
        points[5] = [10, 10, 1]
        points[7] = [10, 20, 1]
        points[9] = [20, 20, 1]
    else:
        points[5] = [10, 10, 1]
        points[7] = [20, 10, 1]
        points[9] = [30, 10, 1]
    return points


def test_crossing_hitter_is_the_side_the_ball_left():
    assert crossing_hitter_side("right") == "left"
    assert crossing_hitter_side("left") == "right"
    assert crossing_hitter_side(None) == "unknown"


def test_coco_straight_arm_is_180():
    angles = arm_angles_from_coco_keypoints(_coco_keypoints(180.0))
    assert abs(angles["left_elbow_angle"] - 180.0) < 0.5
    assert abs(angles["right_elbow_angle"] - 180.0) < 0.5


def test_coco_right_angle_elbow():
    angles = arm_angles_from_coco_keypoints(_coco_keypoints(90.0))
    assert abs(angles["left_elbow_angle"] - 90.0) < 0.5


def test_low_confidence_keypoints_are_ignored():
    points = _coco_keypoints()
    points[:, 2] = 0.1
    assert arm_angles_from_coco_keypoints(points) == {}


def test_person_side_uses_frame_midline_when_uncalibrated():
    left = {"bbox": [10, 10, 40, 80]}
    right = {"bbox": [400, 10, 500, 80]}
    assert person_side(left, frame_size=(640, 360)) == "left"
    assert person_side(right, frame_size=(640, 360)) == "right"


def test_person_side_uses_table_geometry_when_calibrated():
    geometry = TableGeometry(((0, 0), (100, 0), (100, 50), (0, 50)))
    left = {"bbox": [5, 5, 15, 40]}
    right = {"bbox": [80, 5, 95, 40]}
    assert person_side(left, table_geometry=geometry, frame_size=(100, 50)) == "left"
    assert person_side(right, table_geometry=geometry, frame_size=(100, 50)) == "right"


def test_select_hitter_prefers_matching_side():
    persons = [
        {"bbox": [10, 10, 40, 80], "conf": 0.99, "keypoints": _coco_keypoints()},
        {"bbox": [400, 10, 500, 80], "conf": 0.4, "keypoints": _coco_keypoints()},
    ]
    chosen = select_hitter(persons, "right", frame_size=(640, 360))
    assert chosen["bbox"] == [400, 10, 500, 80]


def test_apply_pose_to_hit_fills_angles_and_hit_type():
    hit = HitEvent(
        frame_idx=10,
        timestamp=0.3,
        ball_pos=(420.0, 40.0),
        hitter_side="right",
        arm_angles={},
        board_index=1,
        ball_speed=18.0,
    )
    persons = [{
        "bbox": [400, 10, 500, 80],
        "conf": 0.9,
        "keypoints": _coco_keypoints(180.0),
    }]
    apply_pose_to_hit(hit, persons, frame_size=(640, 360))
    assert hit.hit_type == "drive"
    assert hit.arm_angles["right_elbow_angle"] > 140
    assert hit.confidence >= 0.9


def test_apply_pose_keeps_crossing_event_when_no_person():
    hit = HitEvent(8, 0.2, (100, 40), "left", {}, board_index=2, ball_speed=12)
    apply_pose_to_hit(hit, [], frame_size=(640, 360))
    assert hit.hitter_side == "left"
    assert hit.board_index == 2
    assert hit.arm_angles == {}
    assert hit.hit_type == "unknown"


def test_hit_event_to_dict_exposes_board_speed_and_aliases():
    payload = HitEvent(
        12, 1.234, (10.12, 20.49), "left",
        {"right_elbow_angle": 142.36},
        hit_type="drive",
        confidence=0.81,
        board_index=3,
        ball_speed=78.21,
    ).to_dict()
    assert payload["board_index"] == 3
    assert payload["side"] == "left"
    assert payload["type"] == "drive"
    assert payload["ball_speed"] == 78.2
    assert payload["ball_speed_kmh"] is None
    assert payload["speed_unit"] == "px/f"
    assert payload["calibrated"] is False
    assert payload["arm_angles"]["right_elbow_angle"] == 142.4


def test_hit_event_to_dict_exposes_calibrated_kmh_and_placement():
    payload = HitEvent(
        12, 1.234, (10.12, 20.49), "left",
        {},
        board_index=3,
        ball_speed=78.21,
        ball_speed_kmh=64.44,
        table_pos=(0.501, 0.22),
        placement="过网 · 左路",
        calibrated=True,
    ).to_dict()
    assert payload["ball_speed_kmh"] == 64.4
    assert payload["speed_unit"] == "km/h"
    assert payload["table_pos"] == [0.501, 0.22]
    assert payload["placement"] == "过网 · 左路"
    assert payload["calibrated"] is True
    assert payload["landing_pos"] is None
    assert payload["placement_kind"] is None


def test_hit_event_to_dict_exposes_bounce_landing():
    payload = HitEvent(
        12, 1.234, (10.12, 20.49), "left",
        {},
        board_index=3,
        ball_speed=78.21,
        ball_speed_kmh=64.44,
        table_pos=(0.501, 0.22),
        landing_pos=(0.72, 0.22),
        placement="右半台 · 中台 · 左路",
        placement_kind="bounce",
        calibrated=True,
    ).to_dict()
    assert payload["landing_pos"] == [0.72, 0.22]
    assert payload["placement_kind"] == "bounce"
    assert payload["placement"] == "右半台 · 中台 · 左路"


def test_hit_frame_indices_include_lookback():
    segment = RallySegment(
        start_frame=0, end_frame=20, start_time=0, end_time=1,
        board_count=1,
        hit_events=[HitEvent(10, 0.3, (1, 1), "left", {}, board_index=1)],
    )
    assert hit_frame_indices([segment], lookback_frames=2) == {8, 9, 10}


def test_enrich_segments_from_video_uses_pose_detector(tmp_path):
    import cv2
    from pingpong_analyst.core.hit_enrichment import enrich_segments_from_video

    path = tmp_path / "hits.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(12):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    class FakePose:
        def extract_keypoints(self, _frame):
            return [{
                "bbox": [40, 4, 60, 40],
                "conf": 0.8,
                "keypoints": _coco_keypoints(),
            }]

    hit = HitEvent(5, 0.16, (50, 20), "right", {}, board_index=1, ball_speed=9)
    segment = RallySegment(0, 11, 0.0, 0.3, 1, [hit])
    enrich_segments_from_video(str(path), [segment], FakePose(), lookback_frames=0)
    assert hit.hit_type == "drive"
    assert hit.arm_angles["right_elbow_angle"] > 140
