"""
回合判定状态机 (任务二)

状态流转: IDLE -> RALLY_ACTIVE -> RALLY_END -> IDLE
基于球的运动状态: 静止 -> 运动 -> 出界/落地

同时实现:
- 误检过滤: 捡球、擦汗等非比赛动作
- 板数统计: 基于 DataAligner 的击球事件
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger

from .data_aligner import DataAligner, HitEvent


class RallyState(Enum):
    """回合状态"""
    IDLE = "idle"              # 等待发球
    BALL_MOVING = "ball_moving"  # 球在运动中
    RALLY_ACTIVE = "rally_active"  # 回合进行中
    RALLY_END = "rally_end"    # 回合结束


@dataclass
class RallySegment:
    """一个回合片段"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    board_count: int           # 板数
    hit_events: list[HitEvent]
    # 误检标记
    is_valid: bool = True
    invalid_reason: str = ""

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class RallyDetector:
    """
    回合检测状态机

    输入: DataAligner 的 FrameData 流
    输出: 符合阈值的 RallySegment 列表
    """

    def __init__(self, config: dict):
        self.min_boards = config.get("min_boards", 6)
        self.ball_speed_threshold = config.get("ball_speed_threshold", 5.0)
        self.rally_timeout_frames = config.get("rally_timeout_frames", 60)
        self.pickup_duration_threshold = config.get("pickup_duration_threshold", 15)

        self.state: RallyState = RallyState.IDLE
        self.segments: list[RallySegment] = []

        # 当前回合跟踪
        self._current_start_frame: Optional[int] = None
        self._current_start_time: Optional[float] = None
        self._current_hit_events: list[HitEvent] = []
        self._last_ball_moving_frame: int = 0
        self._static_frame_count: int = 0  # 连续静止帧计数 (捡球检测)

        logger.info(
            f"RallyDetector 初始化: min_boards={self.min_boards}, "
            f"timeout={self.rally_timeout_frames}帧"
        )

    def update(self, frame_data, hit_events: list[HitEvent] = None) -> Optional[RallySegment]:
        """
        处理一帧, 返回已完成的RallySegment (如有)

        Args:
            frame_data: DataAligner.FrameData
            hit_events: 当前帧匹配到的击球事件
        Returns:
            如果回合结束, 返回 RallySegment; 否则返回 None
        """
        hit_events = hit_events or []
        self._current_hit_events.extend(hit_events)

        ball_moving = (
            frame_data.ball_pos is not None
            and frame_data.ball_speed >= self.ball_speed_threshold
        )

        completed_segment = None

        if self.state == RallyState.IDLE:
            if ball_moving:
                # 检测到球开始运动 -> 进入回合
                self._current_start_frame = frame_data.frame_idx
                self._current_start_time = frame_data.timestamp
                self._current_hit_events = list(hit_events)
                self._last_ball_moving_frame = frame_data.frame_idx
                self._static_frame_count = 0
                self.state = RallyState.RALLY_ACTIVE
                logger.debug(f"回合开始 @帧{frame_data.frame_idx}")

        elif self.state == RallyState.RALLY_ACTIVE:
            if ball_moving:
                self._last_ball_moving_frame = frame_data.frame_idx
                self._static_frame_count = 0
            else:
                # 球静止, 检查是否捡球
                self._static_frame_count += 1

                # 捡球检测: 长时间静止 + 人弯腰 -> 误检
                if self._static_frame_count > self.pickup_duration_threshold:
                    if self._is_pickup_action(frame_data):
                        logger.debug("检测到捡球动作, 终止当前回合")
                        completed_segment = self._end_rally(frame_data, is_valid=False, reason="捡球")
                    elif frame_data.frame_idx - self._last_ball_moving_frame > self.rally_timeout_frames:
                        # 超时 -> 回合结束
                        completed_segment = self._end_rally(frame_data)

        if completed_segment:
            self.state = RallyState.IDLE
            self._clear_current_rally()

        return completed_segment

    def _end_rally(
        self, frame_data, is_valid: bool = True, reason: str = ""
    ) -> RallySegment:
        """结束当前回合, 生成RallySegment"""
        segment = RallySegment(
            start_frame=self._current_start_frame,
            end_frame=frame_data.frame_idx,
            start_time=self._current_start_time,
            end_time=frame_data.timestamp,
            board_count=len(self._current_hit_events),
            hit_events=list(self._current_hit_events),
            is_valid=is_valid,
            invalid_reason=reason,
        )

        # 板数过滤
        if is_valid and segment.board_count < self.min_boards:
            segment.is_valid = False
            segment.invalid_reason = f"板数不足({segment.board_count}<{self.min_boards})"

        if segment.is_valid:
            self.segments.append(segment)
            logger.info(
                f"有效回合: 帧{segment.start_frame}-{segment.end_frame} "
                f"({segment.start_time:.1f}s-{segment.end_time:.1f}s) "
                f"板数:{segment.board_count}"
            )
        else:
            logger.debug(f"无效回合(已过滤): {segment.invalid_reason}")

        return segment

    def _is_pickup_action(self, frame_data) -> bool:
        """
        误检过滤: 检测捡球动作

        判断依据: 人体重心明显下降(弯腰) + 球静止
        简化: 检查YOLO关键点中髋部与头部y坐标差是否缩小
        """
        if not frame_data.persons:
            return False

        for person in frame_data.persons:
            kpts = person.get("keypoints")
            if kpts is None or len(kpts) < 16:
                continue
            # COCO关键点: 0=nose, 11=left_hip, 12=right_hip
            nose_y = kpts[0][1] if kpts[0][2] > 0.3 else None
            hip_y = (kpts[11][1] + kpts[12][1]) / 2 if kpts[11][2] > 0.3 and kpts[12][2] > 0.3 else None

            if nose_y is not None and hip_y is not None:
                # 弯腰: 头部接近髋部高度
                if abs(nose_y - hip_y) < 30:  # 像素阈值
                    return True

        return False

    def flush(self, last_frame_idx: int, last_timestamp: float) -> Optional[RallySegment]:
        """视频结束时, 强制结束进行中的回合"""
        if self.state == RallyState.RALLY_ACTIVE and self._current_start_frame is not None:
            from .data_aligner import FrameData
            dummy = FrameData(frame_idx=last_frame_idx, timestamp=last_timestamp)
            segment = self._end_rally(dummy)
            self.state = RallyState.IDLE
            self._clear_current_rally()
            return segment
        return None

    def _clear_current_rally(self):
        """清理当前回合状态, 保留已完成的有效片段。"""
        self._current_start_frame = None
        self._current_start_time = None
        self._current_hit_events = []
        self._last_ball_moving_frame = 0
        self._static_frame_count = 0

    def reset(self):
        """重置整个检测器, 准备分析一个新视频。"""
        self.state = RallyState.IDLE
        self.segments.clear()
        self._clear_current_rally()

    def get_valid_segments(self) -> list[RallySegment]:
        """获取所有有效回合片段"""
        return [s for s in self.segments if s.is_valid]
