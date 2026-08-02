"""Persistence and dataset construction for manual TrackNet ball labels."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from torch.utils.data import Dataset

from ..models.tracknet_preprocess import TrackNetPreprocessor


AnnotationLabel = Literal["ball", "absent"]
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class BallAnnotation:
    frame_index: int
    label: AnnotationLabel
    x: float | None = None
    y: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or int(self.frame_index) != self.frame_index:
            raise ValueError("frame_index 必须是整数")
        if self.frame_index < 0:
            raise ValueError("frame_index 不能为负数")
        if self.label not in {"ball", "absent"}:
            raise ValueError("label 必须是 ball 或 absent")

        if self.label == "ball":
            if self.x is None or self.y is None:
                raise ValueError("ball 标注必须包含 x 和 y")
            if not all(math.isfinite(float(value)) for value in (self.x, self.y)):
                raise ValueError("球坐标必须是有限数字")
            if float(self.x) < 0 or float(self.y) < 0:
                raise ValueError("球坐标不能为负数")
        elif self.x is not None or self.y is not None:
            raise ValueError("absent 标注不能包含球坐标")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "BallAnnotation":
        if not isinstance(value, dict):
            raise ValueError("标注必须是对象")
        return cls(
            frame_index=value.get("frame_index"),
            label=value.get("label"),
            x=value.get("x"),
            y=value.get("y"),
        )


class TrackNetAnnotationStore:
    """Store one validated JSON document per video."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, video_id: str) -> Path:
        if not isinstance(video_id, str) or not _VIDEO_ID_RE.fullmatch(video_id):
            raise ValueError("video_id 包含非法字符")
        return self.root / f"{video_id}.json"

    def save(self, video_id: str, annotations: list[BallAnnotation]) -> None:
        path = self._path(video_id)
        normalized = [item if isinstance(item, BallAnnotation) else BallAnnotation.from_dict(item) for item in annotations]
        by_frame: dict[int, BallAnnotation] = {}
        for annotation in normalized:
            by_frame[annotation.frame_index] = annotation
        ordered = [by_frame[index] for index in sorted(by_frame)]
        payload = {
            "video_id": video_id,
            "annotations": [item.to_dict() for item in ordered],
        }
        fd, temp_name = tempfile.mkstemp(prefix=f".{video_id}.", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def load(self, video_id: str) -> list[BallAnnotation]:
        path = self._path(video_id)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload.get("annotations", payload) if isinstance(payload, dict) else payload
            if not isinstance(values, list):
                raise ValueError("annotations 必须是数组")
            return [BallAnnotation.from_dict(value) for value in values]
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"标注文件无法读取: {path}") from exc

    def delete(self, video_id: str) -> None:
        path = self._path(video_id)
        path.unlink(missing_ok=True)


class TrackNetWindowDataset(Dataset):
    """Sparse annotated windows using the official TrackNetV3 input layout."""

    def __init__(
        self,
        video_path: str | Path,
        annotations: list[BallAnnotation],
        background: np.ndarray | None,
        width: int = 512,
        height: int = 288,
        temporal_frames: int = 8,
        split: str = "all",
        validation_fraction: float = 0.2,
        heatmap_sigma: float = 2.0,
    ) -> None:
        if split not in {"all", "train", "validation"}:
            raise ValueError("split 必须是 all、train 或 validation")
        self.video_path = str(video_path)
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.temporal_frames = max(1, int(temporal_frames))
        self.heatmap_sigma = max(0.1, float(heatmap_sigma))
        self.preprocessor = TrackNetPreprocessor(
            self.width, self.height, self.temporal_frames, background_mode="concat"
        )
        self.background = background
        self.annotations = sorted(
            [item if isinstance(item, BallAnnotation) else BallAnnotation.from_dict(item) for item in annotations],
            key=lambda item: item.frame_index,
        )
        self.total_frames, self.source_width, self.source_height = self._probe_video()
        for annotation in self.annotations:
            if annotation.frame_index >= self.total_frames:
                raise ValueError(
                    f"frame_index 超出视频范围: {annotation.frame_index} >= {self.total_frames}"
                )
            if annotation.label == "ball" and (
                annotation.x >= self.source_width or annotation.y >= self.source_height
            ):
                raise ValueError("球坐标超出视频尺寸")

        self._split_indices = self._make_split_indices(validation_fraction)
        if split == "all":
            selected = list(range(len(self.annotations)))
        else:
            selected = self._split_indices[split]
        self.indices = selected
        self.split = split
        self.split_counts = {
            "total": len(self.annotations),
            "train": len(self._split_indices["train"]),
            "validation": len(self._split_indices["validation"]),
        }

    def _probe_video(self) -> tuple[int, int, int]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开训练视频: {self.video_path}")
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        if total <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"训练视频缺少有效帧信息: {self.video_path}")
        return total, width, height

    def _make_split_indices(self, validation_fraction: float) -> dict[str, list[int]]:
        if not self.annotations:
            return {"train": [], "validation": []}
        gap_limit = self.temporal_frames * 2
        groups: list[list[int]] = [[]]
        for index in range(1, len(self.annotations)):
            previous = self.annotations[index - 1].frame_index
            current = self.annotations[index].frame_index
            if current - previous > gap_limit:
                groups.append([])
            groups[-1].append(index)
        groups[0].insert(0, 0)
        if len(groups) == 1:
            return {"train": [item for item in groups[0]], "validation": []}

        target_validation = max(1, int(round(len(self.annotations) * min(max(validation_fraction, 0.0), 0.5))))
        validation_groups: list[list[int]] = []
        validation_count = 0
        for group in reversed(groups[1:]):
            validation_groups.insert(0, group)
            validation_count += len(group)
            if validation_count >= target_validation:
                break
        validation_set = {item for group in validation_groups for item in group}
        train = [index for index in range(len(self.annotations)) if index not in validation_set]
        validation = [index for index in range(len(self.annotations)) if index in validation_set]
        if not train:
            train = validation[:-1]
            validation = validation[-1:]
        return {"train": train, "validation": validation}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        annotation = self.annotations[self.indices[item]]
        frames = self._read_window(annotation.frame_index)
        inputs = self.preprocessor.prepare(frames, self.background)[0]
        targets = np.zeros(
            (self.temporal_frames, self.height, self.width), dtype=np.float32
        )
        if annotation.label == "ball":
            x = float(annotation.x) * self.width / self.source_width
            y = float(annotation.y) * self.height / self.source_height
            targets[-1] = self._gaussian_heatmap(x, y)
        target_mask = np.zeros((self.temporal_frames,), dtype=np.float32)
        target_mask[-1] = 1.0
        metadata = {
            "frame_index": annotation.frame_index,
            "label": annotation.label,
            "x": annotation.x,
            "y": annotation.y,
        }
        return inputs, targets, target_mask, metadata

    def _read_window(self, target_index: int) -> list[np.ndarray]:
        start = max(0, target_index - self.temporal_frames + 1)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开训练视频: {self.video_path}")
        frames: list[np.ndarray] = []
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(start, target_index + 1):
                success, frame = cap.read()
                if not success:
                    break
                frames.append(frame)
        finally:
            cap.release()
        if not frames:
            raise ValueError(f"无法读取训练帧: {target_index}")
        while len(frames) < self.temporal_frames:
            frames.insert(0, frames[0])
        return frames[-self.temporal_frames :]

    def _gaussian_heatmap(self, x: float, y: float) -> np.ndarray:
        grid_x, grid_y = np.meshgrid(
            np.arange(self.width, dtype=np.float32),
            np.arange(self.height, dtype=np.float32),
        )
        distance = ((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * self.heatmap_sigma**2)
        return np.exp(-distance).astype(np.float32)
