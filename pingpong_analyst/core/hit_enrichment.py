"""用 YOLO 关键点给跨区击球事件补上击球者角度。

回合管线第一遍只跑 TrackNet 和跨区计数，保证板数算法不变。
有效得分段确定后，只对过网时刻附近的少量帧跑人体检测，避免和 TrackNet
同时占显存。
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
from loguru import logger

from .data_aligner import DataAligner, HitEvent
from .table_geometry import TableGeometry

# COCO-17 与 YOLO-Pose 一致
_COCO_LEFT_SHOULDER = 5
_COCO_RIGHT_SHOULDER = 6
_COCO_LEFT_ELBOW = 7
_COCO_RIGHT_ELBOW = 8
_COCO_LEFT_WRIST = 9
_COCO_RIGHT_WRIST = 10
_COCO_LEFT_HIP = 11
_COCO_RIGHT_HIP = 12


def crossing_hitter_side(current_side: str | None) -> str:
    """球刚进入的一侧是落点，对侧才是击球者。"""
    if current_side == "right":
        return "left"
    if current_side == "left":
        return "right"
    return "unknown"


def arm_angles_from_coco_keypoints(
    keypoints, min_confidence: float = 0.3
) -> dict[str, float]:
    """从 COCO-17 关键点计算肘/肩角度。关键点不足时返回空字典。"""
    if keypoints is None:
        return {}
    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 13 or points.shape[1] < 2:
        return {}

    def _point(index: int) -> np.ndarray | None:
        if index >= len(points):
            return None
        item = points[index]
        if len(item) >= 3 and not np.isfinite(item[2]):
            return None
        if len(item) >= 3 and float(item[2]) < min_confidence:
            return None
        if not np.isfinite(item[0]) or not np.isfinite(item[1]):
            return None
        return item[:2]

    angles: dict[str, float] = {}
    left_elbow = _joint_angle(
        _point(_COCO_LEFT_SHOULDER), _point(_COCO_LEFT_ELBOW), _point(_COCO_LEFT_WRIST)
    )
    right_elbow = _joint_angle(
        _point(_COCO_RIGHT_SHOULDER), _point(_COCO_RIGHT_ELBOW), _point(_COCO_RIGHT_WRIST)
    )
    left_shoulder = _joint_angle(
        _point(_COCO_LEFT_HIP), _point(_COCO_LEFT_SHOULDER), _point(_COCO_LEFT_ELBOW)
    )
    right_shoulder = _joint_angle(
        _point(_COCO_RIGHT_HIP), _point(_COCO_RIGHT_SHOULDER), _point(_COCO_RIGHT_ELBOW)
    )
    if left_elbow is not None:
        angles["left_elbow_angle"] = left_elbow
    if right_elbow is not None:
        angles["right_elbow_angle"] = right_elbow
    if left_shoulder is not None:
        angles["left_shoulder_angle"] = left_shoulder
    if right_shoulder is not None:
        angles["right_shoulder_angle"] = right_shoulder
    return angles


def _joint_angle(
    start: np.ndarray | None, joint: np.ndarray | None, end: np.ndarray | None
) -> float | None:
    if start is None or joint is None or end is None:
        return None
    vec_a = start - joint
    vec_b = end - joint
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-8:
        return None
    cosine = float(np.clip(np.dot(vec_a, vec_b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def person_side(
    person: dict,
    table_geometry: TableGeometry | None = None,
    frame_size: tuple[int, int] | None = None,
) -> str | None:
    """按人体框中心判断左右半区；标定后走球台坐标。"""
    bbox = person.get("bbox")
    if bbox is None or len(bbox) != 4:
        return None
    center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
    center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
    if table_geometry is not None and table_geometry.calibrated:
        table_x, _table_y = table_geometry.transform_point((center_x, center_y))
        if not np.isfinite(table_x):
            return None
        return "left" if table_x < 0.5 else "right"
    if frame_size and frame_size[0] > 0:
        return "left" if center_x < frame_size[0] / 2.0 else "right"
    return None


def select_hitter(
    persons: list[dict],
    hitter_side: str,
    table_geometry: TableGeometry | None = None,
    frame_size: tuple[int, int] | None = None,
    ball_pos: tuple[float, float] | None = None,
) -> dict | None:
    """在目标半区里挑离球最近、否则置信度最高的人。"""
    if not persons or hitter_side not in {"left", "right"}:
        return None
    candidates = []
    for person in persons:
        side = person_side(person, table_geometry, frame_size)
        if side != hitter_side:
            continue
        distance = DataAligner._person_ball_distance(person, ball_pos or (0.0, 0.0))
        if ball_pos is None:
            distance = 0.0
        confidence = float(person.get("conf", 0.5) or 0.5)
        candidates.append((distance, -confidence, person))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def apply_pose_to_hit(
    hit: HitEvent,
    persons: list[dict],
    frame_size: tuple[int, int] | None = None,
    table_geometry: TableGeometry | None = None,
) -> HitEvent:
    """用当前帧人体结果填充击球角度；找不到人时保持原事件。"""
    for person in persons:
        if not person.get("arm_angles"):
            person["arm_angles"] = arm_angles_from_coco_keypoints(person.get("keypoints"))
    chosen = select_hitter(
        persons,
        hit.hitter_side,
        table_geometry=table_geometry,
        frame_size=frame_size,
        ball_pos=hit.ball_pos,
    )
    if chosen is None:
        return hit
    angles = chosen.get("arm_angles") or {}
    if not angles:
        return hit
    hit.arm_angles = dict(angles)
    hit.hit_type = DataAligner._infer_hit_type(hit.arm_angles)
    hit.confidence = max(float(hit.confidence or 0.0), float(chosen.get("conf", 0.5) or 0.5))
    return hit


def hit_frame_indices(segments, lookback_frames: int = 2) -> set[int]:
    """过网帧及其前若干帧，供第二遍人体检测使用。"""
    lookback = max(0, int(lookback_frames))
    frames: set[int] = set()
    for segment in segments:
        for hit in getattr(segment, "hit_events", []) or []:
            last = int(hit.frame_idx)
            first = max(0, last - lookback)
            frames.update(range(first, last + 1))
    return frames


def enrich_segments_from_video(
    video_path: str,
    segments,
    pose_detector,
    table_geometry: TableGeometry | None = None,
    lookback_frames: int = 2,
    cancel_event=None,
    progress_callback: Callable[[float], None] | None = None,
) -> None:
    """原地给有效得分段的击球事件补角度。检测器需提供 extract_keypoints。"""
    wanted = hit_frame_indices(segments, lookback_frames=lookback_frames)
    if not wanted:
        if progress_callback:
            progress_callback(1.0)
        return

    hits_by_frame: dict[int, list[HitEvent]] = {}
    for segment in segments:
        for hit in segment.hit_events:
            for frame_idx in range(max(0, hit.frame_idx - lookback_frames), hit.frame_idx + 1):
                hits_by_frame.setdefault(frame_idx, []).append(hit)

    remaining = set(wanted)
    processed = 0
    total = max(1, len(wanted))
    for frame_idx, image in _iter_selected_frames(video_path, remaining, cancel_event):
        persons = pose_detector.extract_keypoints(image)
        frame_size = (int(image.shape[1]), int(image.shape[0]))
        for hit in hits_by_frame.get(frame_idx, []):
            apply_pose_to_hit(
                hit,
                persons,
                frame_size=frame_size,
                table_geometry=table_geometry,
            )
        processed += 1
        if progress_callback:
            progress_callback(min(1.0, processed / total))
        remaining.discard(frame_idx)
        if not remaining:
            break

    if progress_callback:
        progress_callback(1.0)


def _iter_selected_frames(
    video_path: str,
    wanted: Iterable[int],
    cancel_event=None,
):
    import av

    remaining = set(int(index) for index in wanted)
    if not remaining:
        return
    last_needed = max(remaining)
    container = av.open(video_path)
    try:
        frame_idx = 0
        for frame in container.decode(video=0):
            if cancel_event is not None and cancel_event.is_set():
                logger.info("击球姿态补全被取消")
                break
            if frame_idx in remaining:
                yield frame_idx, frame.to_ndarray(format="bgr24")
            if frame_idx >= last_needed:
                break
            frame_idx += 1
    finally:
        container.close()
