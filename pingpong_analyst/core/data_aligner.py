"""
多模态数据融合与时空对齐 (任务一 - 最核心)

解决: YOLO/MediaPipe输出"人"坐标系, TrackNet输出"球"坐标系
实现: 帧同步 + 球轨迹折返点与击球动作匹配 -> 精准判定板数
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

from .table_geometry import TableGeometry, net_placement_label, table_speed_kmh


def _finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


@dataclass
class FrameData:
    """单帧多模态数据 (帧同步后的聚合体)"""
    frame_idx: int
    timestamp: float  # 秒

    # 球数据 (TrackNet)
    ball_pos: Optional[tuple[float, float]] = None  # (x, y) 像素坐标
    ball_confidence: float = 0.0

    # 人体数据 (YOLO11-Pose)
    persons: list[dict] = field(default_factory=list)  # [{"bbox", "keypoints", "conf"}]

    # MediaPipe精细数据
    pose_landmarks: Optional[np.ndarray] = None  # [33, 4]
    arm_angles: Optional[dict] = None  # {"left_elbow_angle": ..., ...}

    # 球速度 (由DataAligner计算)
    ball_speed: float = 0.0  # 像素/帧，供跨区与折返阈值使用
    ball_speed_kmh: Optional[float] = None  # 仅标定后：台面平面投影速度
    ball_direction: Optional[tuple[float, float]] = None  # 单位向量
    ball_table_pos: Optional[tuple[float, float]] = None  # 透视校正后的球台坐标
    frame_size: Optional[tuple[int, int]] = None  # (width, height), 用于未标定坐标归一化
    calibrated: bool = False


@dataclass
class HitEvent:
    """一次球台半区跨越对应的击球记录。

    板数仍由跨区计数器决定；这里只描述「第 N 板是谁打的」。
    ``board_index`` 从 1 起，与 ``CrossingResult.board_count`` 对齐。
    """
    frame_idx: int
    timestamp: float
    ball_pos: tuple[float, float]
    hitter_side: str  # "left" / "right" / "unknown"
    arm_angles: dict
    hit_type: str = "unknown"  # "drive" (撞击) / "spin" (摩擦) / "unknown"
    confidence: float = 0.0
    board_index: int = 0
    ball_speed: float = 0.0
    ball_speed_kmh: Optional[float] = None
    table_pos: Optional[tuple[float, float]] = None
    landing_pos: Optional[tuple[float, float]] = None
    placement: Optional[str] = None
    placement_kind: Optional[str] = None  # "net" / "bounce"
    calibrated: bool = False

    def to_dict(self) -> dict:
        """API / 前端共用的可序列化结构，保留 side/type 别名。"""
        angles = {}
        for key, value in (self.arm_angles or {}).items():
            number = _finite_float(value)
            if number is not None:
                angles[str(key)] = round(number, 1)
        ball_pos = None
        if self.ball_pos is not None:
            x = _finite_float(self.ball_pos[0])
            y = _finite_float(self.ball_pos[1])
            if x is not None and y is not None:
                ball_pos = [round(x, 1), round(y, 1)]
        speed = _finite_float(self.ball_speed) or 0.0
        kmh = _finite_float(self.ball_speed_kmh)
        return {
            "board_index": int(self.board_index),
            "frame_idx": int(self.frame_idx),
            "timestamp": round(float(self.timestamp), 3),
            "hitter_side": self.hitter_side,
            "side": self.hitter_side,
            "hit_type": self.hit_type,
            "type": self.hit_type,
            "ball_speed": round(speed, 1),
            "ball_speed_kmh": round(kmh, 1) if kmh is not None else None,
            "speed_unit": "km/h" if self.calibrated else "px/f",
            "table_pos": _serialize_table_point(self.table_pos),
            "landing_pos": _serialize_table_point(self.landing_pos),
            "placement": self.placement,
            "placement_kind": self.placement_kind,
            "calibrated": bool(self.calibrated),
            "ball_pos": ball_pos,
            "arm_angles": angles,
            "confidence": round(float(self.confidence or 0.0), 3),
        }


def _serialize_table_point(point) -> list[float] | None:
    if point is None:
        return None
    x = _finite_float(point[0])
    y = _finite_float(point[1])
    if x is None or y is None:
        return None
    return [round(x, 3), round(y, 3)]


def apply_ball_kinematics(hit: HitEvent, frame: FrameData) -> HitEvent:
    """把当前帧的像素速度、标定后 km/h 和过网线路写进击球事件。"""
    hit.ball_speed = float(frame.ball_speed or 0.0)
    hit.calibrated = bool(frame.calibrated)
    if frame.calibrated:
        hit.ball_speed_kmh = frame.ball_speed_kmh
        hit.table_pos = frame.ball_table_pos
        hit.placement = net_placement_label(frame.ball_table_pos)
        hit.placement_kind = "net" if hit.placement else None
    else:
        hit.ball_speed_kmh = None
        hit.table_pos = None
        hit.landing_pos = None
        hit.placement = None
        hit.placement_kind = None
    return hit


class DataAligner:
    """
    人-球数据时空对齐器

    工作流程:
    1. 帧同步: 将球轨迹、人体姿态、关节角度按帧索引对齐
    2. 球轨迹分析: 计算球速、方向、检测折返点
    3. 事件关联: 将折返点与击球动作帧匹配 -> 生成HitEvent
    """

    def __init__(self, config: dict, table_geometry: TableGeometry | None = None):
        self.ball_speed_threshold = config.get("ball_speed_threshold", 5.0)
        self.turning_point_window = config.get("turning_point_window", 5)
        self.hit_match_window = config.get("hit_match_window", 10)
        table_config = config.get("table", {})
        self.turning_min_displacement = config.get("turning_min_displacement", 5.0)
        self.table_turning_min_displacement = table_config.get("turning_min_displacement", 0.02)
        self.player_side_margin = table_config.get(
            "player_side_margin", config.get("player_side_margin", 0.15)
        )
        self.table_geometry = table_geometry or TableGeometry.from_config(config)
        # 缓冲区需同时容纳: 折返点检测窗口 + 击球匹配窗口
        self._buffer_size = max(
            self.turning_point_window * 2 + 1,
            self.turning_point_window * 2 + self.hit_match_window * 2 + 1,
        )
        self._frame_buffer: list[FrameData] = []
        # (frame, timestamp, x, y, table_xy)；table_xy 仅在标定时有值
        self._ball_history: list[tuple[int, float, float, float, tuple[float, float] | None]] = []
        self._processed_turning_points: set[int] = set()  # 已处理的折返点 (防重复)
        logger.info("DataAligner 初始化完成")

    def add_frame(
        self,
        frame_idx: int,
        timestamp: float,
        ball_pos: tuple[float, float] | None = None,
        ball_confidence: float = 0.0,
        persons: list[dict] = None,
        pose_landmarks: np.ndarray = None,
        arm_angles: dict = None,
        frame_size: tuple[int, int] | None = None,
    ) -> FrameData:
        """
        添加一帧的多模态数据 (帧同步入口)

        各模型输出按同一 frame_idx 对齐
        """
        frame_data = FrameData(
            frame_idx=frame_idx,
            timestamp=timestamp,
            ball_pos=ball_pos,
            ball_confidence=ball_confidence,
            persons=persons or [],
            pose_landmarks=pose_landmarks,
            arm_angles=arm_angles,
            frame_size=frame_size,
            calibrated=self.table_geometry.calibrated,
        )
        if ball_pos is not None:
            frame_data.ball_table_pos = self.table_geometry.transform_point(ball_pos)
            if not self.table_geometry.calibrated and frame_size:
                width, height = frame_size
                if width > 0 and height > 0:
                    frame_data.ball_table_pos = (
                        frame_data.ball_table_pos[0] / width,
                        frame_data.ball_table_pos[1] / height,
                    )

        # 计算球速与方向
        self._update_ball_motion(frame_data)

        self._frame_buffer.append(frame_data)
        if len(self._frame_buffer) > self._buffer_size:
            self._frame_buffer.pop(0)

        return frame_data

    def _update_ball_motion(self, frame_data: FrameData):
        """计算球速与方向 (基于历史位置)"""
        if frame_data.ball_pos is None:
            return

        x, y = frame_data.ball_pos
        table_xy = None
        if self.table_geometry.calibrated and frame_data.ball_table_pos is not None:
            tx, ty = frame_data.ball_table_pos
            if np.isfinite(tx) and np.isfinite(ty):
                table_xy = (float(tx), float(ty))
        self._ball_history.append(
            (frame_data.frame_idx, frame_data.timestamp, x, y, table_xy)
        )

        # 保留最近5帧
        if len(self._ball_history) > 5:
            self._ball_history.pop(0)

        if len(self._ball_history) >= 2:
            prev_frame, prev_ts, prev_x, prev_y, prev_table = self._ball_history[-2]
            dx = x - prev_x
            dy = y - prev_y
            frame_diff = max(frame_data.frame_idx - prev_frame, 1)
            frame_data.ball_speed = np.sqrt(dx**2 + dy**2) / frame_diff

            if frame_data.ball_speed > 0:
                frame_data.ball_direction = (dx / np.sqrt(dx**2 + dy**2 + 1e-8),
                                             dy / np.sqrt(dx**2 + dy**2 + 1e-8))

            if table_xy is not None and prev_table is not None:
                frame_data.ball_speed_kmh = table_speed_kmh(
                    prev_table, table_xy, frame_data.timestamp - prev_ts
                )

    def detect_turning_points(self) -> list[int]:
        """
        检测球轨迹折返点 (方向反转帧)

        折返点 = 球的x方向速度反转 -> 击球发生的近似时刻

        Returns:
            折返点帧索引列表
        """
        if len(self._frame_buffer) < self.turning_point_window * 2 + 1:
            return []

        turning_points = []
        window = self.turning_point_window
        min_displacement = (
            self.table_turning_min_displacement
            if self.table_geometry.calibrated
            else self.turning_min_displacement
        )
        min_gap = 5  # 折返点间最小帧间隔

        for i in range(window, len(self._frame_buffer) - window):
            frame = self._frame_buffer[i]
            if frame.ball_pos is None or frame.ball_speed < self.ball_speed_threshold:
                continue

            center = self._trajectory_position(frame)
            before_positions = self._neighbor_positions(i, -1, window)
            after_positions = self._neighbor_positions(i, 1, window)
            if center is None or not before_positions or not after_positions:
                continue

            before = np.mean(before_positions[-3:], axis=0)
            after = np.mean(after_positions[:3], axis=0)
            dx_before = center[0] - before[0]
            dx_after = after[0] - center[0]

            # x方向反转 = 折返点，要求足够大的位移
            if dx_before * dx_after < 0 and abs(dx_before) > min_displacement and abs(dx_after) > min_displacement:
                if self.table_geometry.calibrated and not self._near_player_side(center[0]):
                    continue
                # 最小间隔检查
                if turning_points and frame.frame_idx - turning_points[-1] < min_gap:
                    continue
                turning_points.append(frame.frame_idx)

        return turning_points

    def _trajectory_position(self, frame: FrameData) -> tuple[float, float] | None:
        """优先使用球台坐标，未标定时使用原图像素坐标。"""
        position = frame.ball_table_pos if self.table_geometry.calibrated else frame.ball_pos
        if position is None or not np.isfinite(position).all():
            return None
        return position

    def _neighbor_positions(self, index: int, direction: int, window: int) -> list[np.ndarray]:
        """在有限窗口内取有效位置，允许少量连续丢球帧。"""
        positions = []
        for offset in range(1, window + 1):
            neighbor_index = index + direction * offset
            if neighbor_index < 0 or neighbor_index >= len(self._frame_buffer):
                break
            position = self._trajectory_position(self._frame_buffer[neighbor_index])
            if position is not None:
                positions.append(np.asarray(position, dtype=np.float64))
        return positions

    def _near_player_side(self, table_x: float) -> bool:
        margin = float(self.player_side_margin)
        return table_x <= margin or table_x >= 1.0 - margin

    def match_hit_events(self, require_persons: bool = True) -> list[HitEvent]:
        """
        将折返点与击球动作帧匹配 -> 生成击球事件

        核心算法:
        1. 检测球轨迹折返点
        2. 有人体数据时，在折返点附近匹配击球者和关节角度
        3. 纯回合模式下，只把球轨迹折返点记为未知击球事件

        ``require_persons=False`` 是回合剪辑专用路径。它不会触发或依赖
        YOLO/MediaPipe，板数只由球轨迹折返点估计，人体相关字段保持未知。
        """
        turning_points = self.detect_turning_points()
        events = []
        min_hit_gap = 5  # 击球事件间最小帧间隔

        for tp_frame_idx in turning_points:
            # 跳过已处理的折返点 (防重复检测)
            if tp_frame_idx in self._processed_turning_points:
                continue
            # 跳过与已处理折返点太近的 (防噪声误检)
            if any(abs(tp_frame_idx - p) < min_hit_gap for p in self._processed_turning_points):
                continue
            self._processed_turning_points.add(tp_frame_idx)

            if not require_persons:
                candidates = [
                    frame for frame in self._frame_buffer
                    if abs(frame.frame_idx - tp_frame_idx) <= self.hit_match_window
                    and frame.ball_pos is not None
                ]
                if candidates:
                    best_frame = min(
                        candidates,
                        key=lambda frame: abs(frame.frame_idx - tp_frame_idx),
                    )
                    events.append(apply_ball_kinematics(
                        HitEvent(
                            frame_idx=tp_frame_idx,
                            timestamp=best_frame.timestamp,
                            ball_pos=best_frame.ball_pos,
                            hitter_side="unknown",
                            arm_angles={},
                            hit_type="unknown",
                            confidence=0.5,
                        ),
                        best_frame,
                    ))
                continue

            # 在折返点附近找最近的帧数据
            best_frame = None
            best_score = 0.0

            for frame in self._frame_buffer:
                if abs(frame.frame_idx - tp_frame_idx) > self.hit_match_window:
                    continue
                if not frame.persons:
                    continue

                # 选择离球最近的人作为击球者
                if frame.ball_pos is None:
                    continue

                for person in frame.persons:
                    if person.get("bbox") is None:
                        continue
                    bbox = person["bbox"]
                    person_center_x = (bbox[0] + bbox[2]) / 2
                    person_center_y = (bbox[1] + bbox[3]) / 2

                    dist = self._person_ball_distance(person, frame.ball_pos)
                    time_score = 1.0 - min(
                        abs(frame.frame_idx - tp_frame_idx) / (self.hit_match_window + 1),
                        1.0,
                    )
                    distance_score = 1.0 / (1.0 + dist)
                    score = person.get("conf", 0.5) * time_score * distance_score
                    if score > best_score:
                        best_score = score
                        best_frame = frame
                        best_person = person
                        best_person_center_x = person_center_x
                        best_person_center_y = person_center_y

            if best_frame is None:
                continue

            # 判断击球者左右
            hitter_side = self._hitter_side(
                best_person_center_x,
                best_person_center_y,
                best_frame.ball_pos,
            )

            # 使用被匹配到的具体选手角度，避免整帧姿态误配到另一位选手。
            person_arm_angles = best_person.get("arm_angles") or best_frame.arm_angles
            hit_type = self._infer_hit_type(person_arm_angles)

            event = apply_ball_kinematics(
                HitEvent(
                    frame_idx=tp_frame_idx,
                    timestamp=best_frame.timestamp,
                    ball_pos=best_frame.ball_pos,
                    hitter_side=hitter_side,
                    arm_angles=person_arm_angles or {},
                    hit_type=hit_type,
                    confidence=float(min(best_score, 1.0)),
                ),
                best_frame,
            )
            events.append(event)
            logger.debug(
                f"击球事件 @帧{tp_frame_idx} ({best_frame.timestamp:.2f}s) "
                f"侧:{hitter_side} 类型:{hit_type}"
            )

        return events

    @staticmethod
    def _person_ball_distance(person: dict, ball_pos: tuple[float, float]) -> float:
        """优先使用腕部/肘部关键点，否则回退到人体框中心。"""
        kpts = person.get("keypoints")
        if kpts is not None:
            candidates = []
            for index in (9, 10, 7, 8):
                if index < len(kpts) and len(kpts[index]) >= 3 and kpts[index][2] > 0.3:
                    candidates.append(kpts[index][:2])
            if candidates:
                return float(min(np.linalg.norm(np.asarray(point) - np.asarray(ball_pos)) for point in candidates))

        bbox = person.get("bbox")
        if bbox is None:
            return float("inf")
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        return float(np.linalg.norm(np.asarray(center) - np.asarray(ball_pos)))

    def _hitter_side(
        self,
        person_center_x: float,
        person_center_y: float,
        ball_pos: tuple[float, float],
    ) -> str:
        if self.table_geometry.calibrated:
            table_pos = self.table_geometry.transform_point((person_center_x, person_center_y))
            return "left" if table_pos[0] < 0.5 else "right"
        return "left" if person_center_x < ball_pos[0] else "right"

    @staticmethod
    def _infer_hit_type(arm_angles: dict | None) -> str:
        """
        根据关节角度推断击球类型

        简化规则:
        - 肘关节角度大(>140°) + 肩关节角度大 -> 撞击(drive)
        - 肘关节角度小(<120°) -> 摩擦(spin)
        """
        if arm_angles is None:
            return "unknown"

        # 取左右肘角度的较大值
        left_elbow = arm_angles.get("left_elbow_angle", 130)
        right_elbow = arm_angles.get("right_elbow_angle", 130)
        max_elbow = max(left_elbow, right_elbow)

        if max_elbow > 140:
            return "drive"  # 撞击
        elif max_elbow < 120:
            return "spin"   # 摩擦
        else:
            return "unknown"

    def reset(self):
        """重置状态, 准备处理下一段"""
        self._frame_buffer.clear()
        self._ball_history.clear()
        self._processed_turning_points.clear()
