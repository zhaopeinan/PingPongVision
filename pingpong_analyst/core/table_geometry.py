"""球台四点标定与图像到球台归一化坐标的透视变换。"""
from dataclasses import dataclass
from math import hypot

import numpy as np

# ITTF 标准球台。标定后 x 沿两端底线（过网方向），y 沿两侧边线。
# 只用宽度当各向同性尺度，会把沿台长的球速大约缩小 2.74/1.525 ≈ 1.8 倍。
TABLE_LENGTH_M = 2.74
TABLE_WIDTH_M = 1.525
# 世界纪录扣杀约 119 km/h；投影噪声偶发会更大，超过此值视为无效。
MAX_PLAUSIBLE_SPEED_KMH = 200.0
_NET_HALF_WIDTH = 0.12


def _finite_pair(point) -> tuple[float, float] | None:
    if point is None or len(point) < 2:
        return None
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y


def table_displacement_m(
    p0: tuple[float, float] | list[float] | None,
    p1: tuple[float, float] | list[float] | None,
) -> float | None:
    """标定台面坐标的平面位移（米）。未标定或无效时返回 None。"""
    start = _finite_pair(p0)
    end = _finite_pair(p1)
    if start is None or end is None:
        return None
    dx = (end[0] - start[0]) * TABLE_LENGTH_M
    dy = (end[1] - start[1]) * TABLE_WIDTH_M
    return float(hypot(dx, dy))


def table_speed_kmh(
    p0: tuple[float, float] | list[float] | None,
    p1: tuple[float, float] | list[float] | None,
    dt: float,
) -> float | None:
    """台面平面投影速度（km/h）。不含球的高度，过网附近仍有参考价值。"""
    try:
        delta = float(dt)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(delta) or delta <= 1e-4:
        return None
    distance = table_displacement_m(p0, p1)
    if distance is None:
        return None
    kmh = distance / delta * 3.6
    if not np.isfinite(kmh) or kmh < 0 or kmh > MAX_PLAUSIBLE_SPEED_KMH:
        return None
    return float(kmh)


def _lane_label(y: float) -> str:
    if y < 1.0 / 3.0:
        return "左路"
    if y > 2.0 / 3.0:
        return "右路"
    return "中路"


def net_placement_label(
    table_pos: tuple[float, float] | list[float] | None,
    net_half_width: float = _NET_HALF_WIDTH,
) -> str | None:
    """过网瞬间沿台宽的线路，不是弹跳落点。远离中线时不标注。"""
    point = _finite_pair(table_pos)
    if point is None:
        return None
    x, y = point
    if abs(x - 0.5) > net_half_width:
        return None
    return f"过网 · {_lane_label(y)}"


def landing_label(table_pos: tuple[float, float] | list[float] | None) -> str | None:
    """弹跳落点：半区 · 近网/中台/底线 · 线路。"""
    point = _finite_pair(table_pos)
    if point is None:
        return None
    x, y = point
    half = "左半台" if x <= 0.5 else "右半台"
    depth = abs(x - 0.5)
    if depth < 0.18:
        zone = "近网"
    elif depth < 0.35:
        zone = "中台"
    else:
        zone = "底线"
    return f"{half} · {zone} · {_lane_label(y)}"


@dataclass
class TableGeometry:
    """将图像中的球台四角映射到 [0, 1] x [0, 1]。"""

    corners: np.ndarray | None = None

    def __post_init__(self):
        if self.corners is None:
            return
        corners = np.asarray(self.corners, dtype=np.float64)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError("table corners 必须是 4 个有限的 [x, y] 坐标")
        self.corners = corners
        self._homography = self._compute_homography(corners)

    @classmethod
    def from_config(cls, config: dict | None) -> "TableGeometry":
        config = config or {}
        table_config = config.get("table") or config
        corners = table_config.get("corners")
        if not corners:
            return cls()
        return cls(corners=np.asarray(corners, dtype=np.float64))

    @classmethod
    def from_calibration(cls, calibration) -> "TableGeometry":
        """Build geometry from a persisted per-video calibration object."""
        return cls(corners=np.asarray(calibration.corners, dtype=np.float64))

    @property
    def calibrated(self) -> bool:
        return self.corners is not None

    @staticmethod
    def _compute_homography(corners: np.ndarray) -> np.ndarray:
        # 顶点顺序: 左上、右上、右下、左下。
        target = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        rows = []
        for (x, y), (u, v) in zip(corners, target):
            rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        matrix = np.asarray(rows, dtype=np.float64)
        if np.linalg.matrix_rank(matrix) < 8:
            raise ValueError("table corners 不能共线或形成退化四边形")
        _, _, vh = np.linalg.svd(matrix)
        homography = vh[-1].reshape(3, 3)
        if abs(homography[2, 2]) < 1e-12:
            raise ValueError("无法计算球台透视变换")
        return homography / homography[2, 2]

    def transform_point(self, point: tuple[float, float] | list[float]) -> tuple[float, float]:
        """将一个图像坐标映射为归一化球台坐标。"""
        if not self.calibrated:
            return float(point[0]), float(point[1])
        vector = self._homography @ np.array([point[0], point[1], 1.0], dtype=np.float64)
        if abs(vector[2]) < 1e-12:
            return float("nan"), float("nan")
        return float(vector[0] / vector[2]), float(vector[1] / vector[2])
