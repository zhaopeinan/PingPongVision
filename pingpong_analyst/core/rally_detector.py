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

from .ball_crossing import BallCrossingCounter
from .ball_landing import TrajectorySample, apply_landing, find_bounce
from .data_aligner import HitEvent, apply_ball_kinematics
from .hit_enrichment import crossing_hitter_side


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
        self.min_boards = config.get("min_boards", 4)
        self.ball_speed_threshold = config.get("ball_speed_threshold", 5.0)
        self.rally_timeout_frames = config.get("rally_timeout_frames", 60)
        self.pickup_duration_threshold = config.get("pickup_duration_threshold", 15)
        self.no_crossing_timeout_seconds = float(
            config.get("no_crossing_timeout_seconds", 4.0)
        )
        self.crossing_counter = BallCrossingCounter(self.no_crossing_timeout_seconds)

        self.state: RallyState = RallyState.IDLE
        self.segments: list[RallySegment] = []

        # 当前回合跟踪
        self._current_start_frame: Optional[int] = None
        self._current_start_time: Optional[float] = None
        self._current_hit_events: list[HitEvent] = []
        self._last_ball_moving_frame: int = 0
        self._static_frame_count: int = 0  # 连续静止帧计数 (捡球检测)
        self._current_board_count: int = 0
        self._landing_hit: HitEvent | None = None
        self._landing_side: str | None = None
        self._landing_samples: list[TrajectorySample] = []

        logger.info(
            f"RallyDetector 初始化: min_boards={self.min_boards}, "
            f"timeout={self.rally_timeout_frames}帧"
        )

    def update(self, frame_data, hit_events: list[HitEvent] = None) -> Optional[RallySegment]:
        """
        处理一帧, 返回已完成的RallySegment (如有)

        Args:
            frame_data: DataAligner.FrameData
            hit_events: 当前帧匹配到的击球事件（板数不再使用该参数）
        Returns:
            如果回合结束, 返回 RallySegment; 否则返回 None
        """
        result = self.crossing_counter.update(frame_data)
        completed_segment = None

        if result.timed_out and result.ended_start_frame is not None:
            self._finalize_landing()
            completed_segment = self._end_rally(
                frame_data,
                board_count=result.ended_board_count or 0,
                start_frame=result.ended_start_frame,
                start_time=result.ended_start_time,
            )
            self.state = RallyState.IDLE
            self._clear_current_rally()

        if result.segment_started and self.crossing_counter.active:
            self._current_start_frame = self.crossing_counter.segment_start_frame
            self._current_start_time = self.crossing_counter.segment_start_time
            self._current_hit_events = []
            self._static_frame_count = 0
            self.state = RallyState.RALLY_ACTIVE
            logger.debug(f"得分段开始 @帧{self._current_start_frame}")

        if result.crossed:
            self._finalize_landing()
            hit = self._hit_from_crossing(frame_data, result.current_side, result.board_count)
            self._current_hit_events.append(hit)
            self._start_landing_search(hit, result.current_side)

        self._observe_landing(frame_data)
        self._current_board_count = result.board_count
        return completed_segment

    def _hit_from_crossing(self, frame_data, current_side: str | None, board_index: int) -> HitEvent:
        """跨区即一板；击球者是球离开的半区。角度可在第二遍人体检测中补全。"""
        ball_pos = frame_data.ball_pos or (0.0, 0.0)
        return apply_ball_kinematics(
            HitEvent(
                frame_idx=int(frame_data.frame_idx),
                timestamp=float(frame_data.timestamp),
                ball_pos=ball_pos,
                hitter_side=crossing_hitter_side(current_side),
                arm_angles={},
                hit_type="unknown",
                confidence=0.5 if frame_data.ball_pos is not None else 0.2,
                board_index=int(board_index),
            ),
            frame_data,
        )

    def _start_landing_search(self, hit: HitEvent, destination_side: str | None) -> None:
        self._landing_hit = hit
        self._landing_side = destination_side
        self._landing_samples = []

    def _observe_landing(self, frame_data) -> None:
        if self._landing_hit is None or not getattr(frame_data, "calibrated", False):
            return
        if frame_data.ball_pos is None or frame_data.ball_table_pos is None:
            return
        table_pos = (
            float(frame_data.ball_table_pos[0]),
            float(frame_data.ball_table_pos[1]),
        )
        self._landing_samples.append(
            TrajectorySample(
                frame_idx=int(frame_data.frame_idx),
                image_y=float(frame_data.ball_pos[1]),
                table_pos=table_pos,
            )
        )
        bounce = find_bounce(self._landing_samples, self._landing_side)
        if bounce is not None:
            apply_landing(self._landing_hit, bounce)
            self._landing_hit = None
            self._landing_samples = []

    def _finalize_landing(self) -> None:
        if self._landing_hit is None:
            self._landing_samples = []
            return
        bounce = find_bounce(self._landing_samples, self._landing_side)
        if bounce is not None:
            apply_landing(self._landing_hit, bounce)
        self._landing_hit = None
        self._landing_side = None
        self._landing_samples = []

    def _end_rally(
        self,
        frame_data,
        is_valid: bool = True,
        reason: str = "",
        board_count: int | None = None,
        start_frame: int | None = None,
        start_time: float | None = None,
    ) -> RallySegment:
        """结束当前回合, 生成RallySegment"""
        segment = RallySegment(
            start_frame=start_frame if start_frame is not None else self._current_start_frame,
            end_frame=frame_data.frame_idx,
            start_time=start_time if start_time is not None else self._current_start_time,
            end_time=frame_data.timestamp,
            board_count=board_count if board_count is not None else self._current_board_count,
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
                f"有效得分段: 帧{segment.start_frame}-{segment.end_frame} "
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
        if self.crossing_counter.active and self._current_start_frame is not None:
            from .data_aligner import FrameData
            dummy = FrameData(frame_idx=last_frame_idx, timestamp=last_timestamp)
            self._finalize_landing()
            result = self.crossing_counter.close(dummy)
            segment = self._end_rally(
                dummy,
                board_count=result.ended_board_count or 0,
                start_frame=result.ended_start_frame,
                start_time=result.ended_start_time,
            )
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
        self._current_board_count = 0
        self._landing_hit = None
        self._landing_side = None
        self._landing_samples = []

    def reset(self):
        """重置整个检测器, 准备分析一个新视频。"""
        self.state = RallyState.IDLE
        self.segments.clear()
        self.crossing_counter.reset()
        self._clear_current_rally()

    @property
    def current_board_count(self) -> int:
        return self._current_board_count

    def get_valid_segments(self) -> list[RallySegment]:
        """获取所有有效回合片段"""
        return [s for s in self.segments if s.is_valid]
