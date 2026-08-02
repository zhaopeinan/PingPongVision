"""Per-video table calibration value objects and JSON persistence."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _normalize_points(value, name: str, expected_count: int | None = None) -> tuple[tuple[float, float], ...]:
    if value is None:
        points: tuple[tuple[float, float], ...] = ()
    else:
        try:
            points = tuple((float(point[0]), float(point[1])) for point in value)
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError(f"{name} 必须是 [x, y] 坐标数组") from exc
    if expected_count is not None and len(points) != expected_count:
        raise ValueError(f"{name} 必须包含 {expected_count} 个点")
    if not all(math.isfinite(coord) for point in points for coord in point):
        raise ValueError(f"{name} 必须包含有限数字")
    return points


def _validate_corners(corners: tuple[tuple[float, float], ...]) -> None:
    signed_crosses = []
    for index in range(4):
        first = corners[index]
        second = corners[(index + 1) % 4]
        third = corners[(index + 2) % 4]
        vector_a = (second[0] - first[0], second[1] - first[1])
        vector_b = (third[0] - second[0], third[1] - second[1])
        signed_crosses.append(vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0])
    if not all(abs(value) > 1e-9 for value in signed_crosses):
        raise ValueError("table corners 不能共线或形成退化四边形")
    if not all(value > 0 for value in signed_crosses) and not all(value < 0 for value in signed_crosses):
        raise ValueError("table corners 必须按顺时针或逆时针顺序排列")


@dataclass(frozen=True)
class TableCalibration:
    """A single video's table geometry in source-frame pixel coordinates."""

    frame_index: int
    corners: tuple[tuple[float, float], ...]
    net_points: tuple[tuple[float, float], ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or int(self.frame_index) != self.frame_index:
            raise ValueError("frame_index 必须是整数")
        if self.frame_index < 0:
            raise ValueError("frame_index 不能为负数")
        normalized_corners = _normalize_points(self.corners, "corners", expected_count=4)
        normalized_net_points = _normalize_points(self.net_points, "net_points")
        if len(normalized_net_points) not in {0, 2}:
            raise ValueError("net_points 必须包含 0 或 2 个点")
        _validate_corners(normalized_corners)
        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(self, "corners", normalized_corners)
        object.__setattr__(self, "net_points", normalized_net_points)
        object.__setattr__(self, "version", int(self.version))

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "corners": [list(point) for point in self.corners],
            "net_points": [list(point) for point in self.net_points],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "TableCalibration":
        if not isinstance(value, dict):
            raise ValueError("table calibration 必须是对象")
        return cls(
            frame_index=value.get("frame_index"),
            corners=value.get("corners"),
            net_points=value.get("net_points") or (),
            version=value.get("version", 1),
        )


class TableCalibrationStore:
    """Store one validated calibration JSON document per video."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, video_id: str) -> Path:
        if not isinstance(video_id, str) or not _VIDEO_ID_RE.fullmatch(video_id):
            raise ValueError("video_id 包含非法字符")
        return self.root / f"{video_id}.json"

    def save(self, video_id: str, calibration: TableCalibration) -> None:
        path = self._path(video_id)
        calibration = calibration if isinstance(calibration, TableCalibration) else TableCalibration.from_dict(calibration)
        payload = {"video_id": video_id, "calibration": calibration.to_dict()}
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

    def load(self, video_id: str) -> TableCalibration | None:
        path = self._path(video_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = payload.get("calibration", payload) if isinstance(payload, dict) else payload
            return TableCalibration.from_dict(value)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"标定文件无法读取: {path}") from exc

    def delete(self, video_id: str) -> None:
        self._path(video_id).unlink(missing_ok=True)
