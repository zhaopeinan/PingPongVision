"""独立的人体动作分析管线。

动作分析与回合剪辑解耦：这里只加载 YOLO-Pose 和 MediaPipe，不运行 TrackNet
或回合状态机。身份关联是保守的外观/轨迹启发式，无法确认时返回 ``unknown``。
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from loguru import logger

from ..models import MediaPipePoseAnalyzer, YOLOPoseDetector
from ..utils.config import get_config
from ..utils.device_manager import DeviceManager
from .data_aligner import DataAligner
from .hit_enrichment import select_hitter

HIT_MATCH_SECONDS = 0.25


def _json_value(value: Any):
    """将 NumPy 标量/数组转换为 API 可序列化的原生值。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def compact_person(person: dict) -> dict:
    """去掉关键点和 3D 坐标，只保留报告需要的身份和角度。"""
    return {
        "track_id": person.get("track_id"),
        "identity": person.get("identity", "unknown"),
        "identity_confidence": person.get("identity_confidence", 0.0),
        "conf": float(person.get("conf", 0.0) or 0.0),
        "bbox": _json_value(person.get("bbox")),
        "arm_angles": _json_value(person.get("arm_angles") or {}),
        "body_metrics": _json_value(person.get("body_metrics") or {}),
    }


def bind_hits_to_pose(
    hits: list[dict],
    timestamp: float,
    persons: list[dict],
    frame_size: tuple[int, int],
    window: float = HIT_MATCH_SECONDS,
) -> None:
    """把当前帧姿态挂到时间最接近的击球事件上（原地更新）。"""
    if not persons:
        return
    for hit in hits:
        target = float(hit.get("timestamp") or 0.0)
        delta = abs(timestamp - target)
        if delta > window or delta > float(hit.get("_best_dt", window)):
            continue
        side = hit.get("hitter_side") or hit.get("side") or "unknown"
        chosen = select_hitter(
            persons, side, frame_size=frame_size
        ) if side in {"left", "right"} else None
        if chosen is None and persons:
            chosen = max(persons, key=lambda item: float(item.get("conf", 0.0) or 0.0))
        hit["_best_dt"] = delta
        hit["pose"] = compact_person(chosen) if chosen else None
        angles = (chosen or {}).get("arm_angles") or {}
        if angles and not hit.get("arm_angles"):
            hit["arm_angles"] = _json_value(angles)
            hit_type = DataAligner._infer_hit_type(angles)
            hit["hit_type"] = hit_type
            hit["type"] = hit_type


def finalize_hit_events(hits: list[dict]) -> list[dict]:
    """去掉内部匹配字段，输出可序列化击球快照。"""
    payload = []
    for hit in hits:
        item = {
            key: value
            for key, value in hit.items()
            if not str(key).startswith("_")
        }
        payload.append(_json_value(item))
    return payload


class AppearanceIdentityTracker:
    """基于服装颜色和短期轨迹的保守身份关联器。"""

    def __init__(self, min_confirmations: int = 3):
        self.min_confirmations = min_confirmations
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    @staticmethod
    def _feature(frame: np.ndarray, bbox: list[float] | None) -> np.ndarray | None:
        if bbox is None or len(bbox) != 4:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        # 服装颜色比背景更稳定，取人体框中间 60% 区域降低球台背景影响。
        crop = crop[int(crop.shape[0] * 0.15):int(crop.shape[0] * 0.8)]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32)

    @staticmethod
    def _similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
        if left is None or right is None:
            return 0.0
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1e-8:
            return 0.0
        return float(np.dot(left, right) / denominator)

    @staticmethod
    def _spatial_similarity(previous: list[float] | None, current: list[float] | None) -> float:
        if previous is None or current is None:
            return 0.0
        px = (previous[0] + previous[2]) / 2
        py = (previous[1] + previous[3]) / 2
        cx = (current[0] + current[2]) / 2
        cy = (current[1] + current[3]) / 2
        diagonal = max(((previous[2] - previous[0]) ** 2 + (previous[3] - previous[1]) ** 2) ** 0.5, 1.0)
        distance = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        return float(np.exp(-distance / (diagonal * 2.5)))

    def update(self, frame: np.ndarray, persons: list[dict], frame_idx: int) -> list[dict]:
        if not persons:
            return []

        features = [self._feature(frame, person.get("bbox")) for person in persons]
        pairs = []
        for person_idx, feature in enumerate(features):
            for track_id, track in self._tracks.items():
                appearance = self._similarity(track.get("feature"), feature)
                spatial = self._spatial_similarity(
                    track.get("bbox"), persons[person_idx].get("bbox")
                )
                # 跨片段时没有连续空间轨迹，仍允许外观特征参与匹配。
                if track.get("last_frame", -1) > frame_idx:
                    score = appearance
                else:
                    score = appearance * 0.65 + spatial * 0.35
                pairs.append((score, person_idx, track_id))

        assigned_people: set[int] = set()
        assigned_tracks: set[int] = set()
        assignments: dict[int, tuple[int, float]] = {}
        for score, person_idx, track_id in sorted(pairs, reverse=True):
            if person_idx in assigned_people or track_id in assigned_tracks:
                continue
            if score < 0.62:
                continue
            assignments[person_idx] = (track_id, score)
            assigned_people.add(person_idx)
            assigned_tracks.add(track_id)

        for person_idx in range(len(persons)):
            if person_idx not in assignments:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = {
                    "bbox": persons[person_idx].get("bbox"),
                    "feature": features[person_idx],
                    "last_frame": frame_idx,
                    "confirmations": 1,
                    "confidence": 0.35,
                }
                assignments[person_idx] = (track_id, 0.35)

        enriched = []
        for person_idx, person in enumerate(persons):
            track_id, score = assignments[person_idx]
            track = self._tracks[track_id]
            track["bbox"] = person.get("bbox")
            if features[person_idx] is not None:
                track["feature"] = features[person_idx]
            track["last_frame"] = frame_idx
            if score >= 0.62:
                track["confirmations"] += 1
            track["confidence"] = max(float(track.get("confidence", 0.35)), float(score))
            confirmed = track["confirmations"] >= self.min_confirmations
            item = dict(person)
            item["track_id"] = f"track_{track_id}"
            item["identity"] = f"player_{track_id}" if confirmed else "unknown"
            item["identity_confidence"] = round(
                min(1.0, float(track["confidence"])), 3
            )
            enriched.append(item)
        return enriched


class ActionAnalyzer:
    """对一段视频输出逐帧姿态和动作指标。"""

    def __init__(self, config_path: str | None = None):
        self.config = get_config(config_path)
        self.device_info = DeviceManager.detect(self.config.device_mode)
        self._yolo: YOLOPoseDetector | None = None
        self._mediapipe: MediaPipePoseAnalyzer | None = None
        self.identity_tracker = AppearanceIdentityTracker()

    def load_models(self):
        if self._yolo is not None:
            return
        yolo_cfg = self.config.get("models", "yolo_pose", default={})
        mp_cfg = self.config.get("models", "mediapipe_pose", default={})
        self._yolo = YOLOPoseDetector(self.device_info, yolo_cfg)
        if not self._yolo.load():
            raise RuntimeError("YOLO-Pose 加载失败，无法进行动作分析")
        if mp_cfg.get("enabled", True):
            self._mediapipe = MediaPipePoseAnalyzer(self.device_info, mp_cfg)
            self._mediapipe.load()

    def unload_models(self):
        if self._yolo is not None:
            self._yolo.unload()
            self._yolo = None
        if self._mediapipe is not None:
            self._mediapipe.unload()
            self._mediapipe = None

    def analyze(
        self,
        video_path: str,
        max_frames: int = -1,
        start_frame: int = 0,
        end_frame: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_event=None,
        include_frames: bool = False,
        hit_events: list[dict] | None = None,
    ) -> dict:
        """分析视频，返回摘要、击球快照；默认不返回逐帧关键点。

        ``hit_events`` 的 timestamp 必须落在这段视频自己的时间轴上
        （分析得分段剪辑时应先扣除 clip 起始时间）。
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")
        self.load_models()

        import av

        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 30.0)
        estimated_frames = int(stream.frames or 0)
        if estimated_frames <= 0 and stream.duration and stream.time_base:
            estimated_frames = int(float(stream.duration * stream.time_base) * fps)
        estimated_frames = max(0, estimated_frames - start_frame)
        if end_frame is not None:
            estimated_frames = min(estimated_frames, max(0, end_frame - start_frame + 1))
        if max_frames > 0:
            estimated_frames = min(estimated_frames, max_frames) if estimated_frames else max_frames
        frames = []
        posed_frames = 0
        pending_hits = [dict(hit) for hit in (hit_events or [])]
        stats: dict[str, dict] = defaultdict(lambda: {
            "frame_count": 0,
            "identity": "unknown",
            "identity_confidence": 0.0,
            "angle_sums": defaultdict(float),
            "body_sums": defaultdict(float),
        })
        decoded_idx = 0
        analyzed = 0
        try:
            for frame in container.decode(video=0):
                if cancel_event is not None and cancel_event.is_set():
                    logger.info("动作分析任务被取消")
                    break
                if decoded_idx < start_frame:
                    decoded_idx += 1
                    continue
                if end_frame is not None and decoded_idx > end_frame:
                    break
                if max_frames > 0 and analyzed >= max_frames:
                    break

                image = frame.to_ndarray(format="bgr24")
                persons = self._yolo.extract_keypoints(image)
                if self._mediapipe is not None and self._mediapipe.is_loaded:
                    persons = self._mediapipe.enrich_persons(image, persons)
                persons = self.identity_tracker.update(image, persons, decoded_idx)
                timestamp = round(decoded_idx / fps, 4)
                frame_size = (int(image.shape[1]), int(image.shape[0]))
                if persons:
                    posed_frames += 1
                if pending_hits:
                    bind_hits_to_pose(pending_hits, timestamp, persons, frame_size)

                compact_people = [compact_person(person) for person in persons]
                for item in compact_people:
                    track_id = item["track_id"]
                    current = stats[track_id]
                    current["frame_count"] += 1
                    current["identity"] = item["identity"]
                    current["identity_confidence"] = max(
                        current["identity_confidence"], item["identity_confidence"]
                    )
                    for key, value in (item["arm_angles"] or {}).items():
                        current["angle_sums"][key] += float(value)
                    for key, value in (item["body_metrics"] or {}).items():
                        current["body_sums"][key] += float(value)

                if include_frames:
                    frames.append({
                        "frame": decoded_idx,
                        "timestamp": timestamp,
                        "persons": compact_people,
                    })
                decoded_idx += 1
                analyzed += 1
                if progress_callback and (analyzed == 1 or analyzed % 5 == 0):
                    progress_callback(analyzed, estimated_frames)
        finally:
            if progress_callback:
                progress_callback(analyzed, estimated_frames)
            container.close()

        summaries = []
        for track_id, current in stats.items():
            count = max(current["frame_count"], 1)
            summaries.append({
                "track_id": track_id,
                "identity": current["identity"],
                "identity_confidence": round(current["identity_confidence"], 3),
                "frame_count": current["frame_count"],
                "average_arm_angles": {
                    key: round(value / count, 3)
                    for key, value in current["angle_sums"].items()
                },
                "average_body_metrics": {
                    key: round(value / count, 3)
                    for key, value in current["body_sums"].items()
                },
            })

        result = {
            "source": Path(video_path).name,
            "frames_analyzed": analyzed,
            "posed_frames": posed_frames,
            "fps": round(fps, 3),
            "persons": summaries,
            "hit_events": finalize_hit_events(pending_hits),
        }
        if include_frames:
            result["frames"] = frames
        return result
