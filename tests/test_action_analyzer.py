"""动作分析中的身份关联、压缩结果和击球对齐。"""
import cv2
import numpy as np

from pingpong_analyst.core.action_analyzer import (
    ActionAnalyzer,
    AppearanceIdentityTracker,
    bind_hits_to_pose,
    compact_person,
    finalize_hit_events,
)


def test_identity_tracker_confirms_consistent_appearance():
    tracker = AppearanceIdentityTracker(min_confirmations=3)
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (80, 110), (0, 0, 255), -1)
    cv2.rectangle(frame, (160, 10), (230, 110), (255, 0, 0), -1)
    persons = [
        {"bbox": [10, 10, 80, 110], "conf": 0.9},
        {"bbox": [160, 10, 230, 110], "conf": 0.9},
    ]

    first = tracker.update(frame, persons, frame_idx=0)
    second = tracker.update(frame, persons, frame_idx=1)
    third = tracker.update(frame, persons, frame_idx=2)

    assert [person["identity"] for person in first] == ["unknown", "unknown"]
    assert [person["track_id"] for person in second] == ["track_1", "track_2"]
    assert [person["identity"] for person in third] == ["player_1", "player_2"]


def test_identity_tracker_does_not_force_unknown_person_into_existing_track():
    tracker = AppearanceIdentityTracker(min_confirmations=2)
    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_a[:, :50] = (0, 0, 255)
    first = tracker.update(frame_a, [{"bbox": [0, 0, 50, 100]}], frame_idx=0)

    frame_b = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_b[:, :50] = (0, 255, 0)
    second = tracker.update(frame_b, [{"bbox": [0, 0, 50, 100]}], frame_idx=1)

    assert first[0]["identity"] == "unknown"
    assert second[0]["identity"] == "unknown"
    assert second[0]["track_id"] != "track_1"


def test_compact_person_drops_landmarks():
    payload = compact_person({
        "track_id": "track_1",
        "identity": "unknown",
        "keypoints": np.zeros((17, 3)),
        "pose_landmarks": np.zeros((33, 4)),
        "arm_angles": {"right_elbow_angle": 142.2},
        "conf": 0.8,
        "bbox": [1, 2, 3, 4],
    })
    assert "keypoints" not in payload
    assert "pose_landmarks" not in payload
    assert payload["arm_angles"]["right_elbow_angle"] == 142.2


def test_bind_hits_to_pose_prefers_matching_side():
    hits = [{"timestamp": 0.1, "hitter_side": "right", "board_index": 1, "ball_speed": 20}]
    persons = [
        {"bbox": [10, 10, 40, 80], "conf": 0.99, "arm_angles": {"right_elbow_angle": 90}, "track_id": "track_1"},
        {"bbox": [400, 10, 500, 80], "conf": 0.4, "arm_angles": {"right_elbow_angle": 150}, "track_id": "track_2"},
    ]
    bind_hits_to_pose(hits, 0.1, persons, frame_size=(640, 360))
    finalized = finalize_hit_events(hits)
    assert finalized[0]["pose"]["track_id"] == "track_2"
    assert finalized[0]["hit_type"] == "drive"
    assert "_best_dt" not in finalized[0]


def test_bind_hits_ignores_empty_frames():
    hits = [{"timestamp": 0.1, "hitter_side": "left", "arm_angles": {"left_elbow_angle": 160}}]
    bind_hits_to_pose(
        hits, 0.1,
        [{"bbox": [10, 10, 40, 80], "conf": 0.8, "arm_angles": {"left_elbow_angle": 160}}],
        (640, 360),
    )
    bind_hits_to_pose(hits, 0.101, [], (640, 360))
    assert hits[0]["pose"]["arm_angles"]["left_elbow_angle"] == 160


def test_action_analyze_omits_full_frames(tmp_path):
    path = tmp_path / "action.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(10):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    analyzer = ActionAnalyzer()
    analyzer.load_models = lambda: None

    class FakeYolo:
        is_loaded = True

        def extract_keypoints(self, image):
            width = image.shape[1]
            return [{
                "bbox": [width * 0.6, 2, width - 2, image.shape[0] - 2],
                "conf": 0.9,
                "arm_angles": {"right_elbow_angle": 150},
            }]

    analyzer._yolo = FakeYolo()
    analyzer._mediapipe = None
    result = analyzer.analyze(
        str(path),
        hit_events=[{"timestamp": 0.1, "hitter_side": "right", "board_index": 1, "ball_speed": 22}],
    )
    assert "frames" not in result
    assert result["frames_analyzed"] >= 1
    assert result["posed_frames"] >= 1
    assert result["hit_events"][0]["pose"]["arm_angles"]["right_elbow_angle"] == 150
    debug = analyzer.analyze(str(path), include_frames=True)
    assert debug["frames"]
    assert "keypoints" not in debug["frames"][0]["persons"][0]

