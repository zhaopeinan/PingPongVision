"""动作分析中的身份关联和结果基础行为测试。"""
import cv2
import numpy as np

from pingpong_analyst.core.action_analyzer import AppearanceIdentityTracker


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
