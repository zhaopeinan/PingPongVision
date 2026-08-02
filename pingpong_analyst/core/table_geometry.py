"""球台四点标定与图像到球台归一化坐标的透视变换。"""
from dataclasses import dataclass

import numpy as np


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
