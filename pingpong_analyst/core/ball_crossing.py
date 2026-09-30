"""Count ball exchanges between the two normalized table halves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CrossingResult:
    board_count: int
    current_side: str | None
    crossed: bool = False
    timed_out: bool = False
    segment_started: bool = False
    ended_board_count: int | None = None
    ended_start_frame: int | None = None
    ended_start_time: float | None = None


class BallCrossingCounter:
    """Use a hysteresis band to count left/right table side changes."""

    def __init__(
        self,
        timeout_seconds: float = 2.0,
        left_enter: float = 0.45,
        right_enter: float = 0.55,
    ) -> None:
        timeout_seconds = float(timeout_seconds)
        if not 0.5 <= timeout_seconds <= 10.0:
            raise ValueError("timeout_seconds 必须在 0.5 到 10.0 秒之间")
        if not 0.0 < left_enter < right_enter < 1.0:
            raise ValueError("左右半区阈值必须满足 0 < left_enter < right_enter < 1")
        self.timeout_seconds = timeout_seconds
        self.left_enter = float(left_enter)
        self.right_enter = float(right_enter)
        self.reset()

    @property
    def active(self) -> bool:
        return self.segment_start_frame is not None

    def reset(self) -> None:
        self.current_side: Optional[str] = None
        self.board_count = 0
        self.segment_start_frame: int | None = None
        self.segment_start_time: float | None = None
        self.last_crossing_time: float | None = None

    def update(self, frame_data) -> CrossingResult:
        timestamp = float(frame_data.timestamp)
        timed_out = False
        ended_board_count = None
        ended_start_frame = None
        ended_start_time = None

        if (
            self.active
            and self.last_crossing_time is not None
            and timestamp - self.last_crossing_time > self.timeout_seconds
        ):
            timed_out = True
            ended_board_count = self.board_count
            ended_start_frame = self.segment_start_frame
            ended_start_time = self.segment_start_time
            self.reset()

        segment_started = False
        crossed = False
        table_x = self._table_x(frame_data)
        if table_x is not None:
            side = self._confirmed_side(table_x)
            if side is not None and not self.active:
                self.segment_start_frame = int(frame_data.frame_idx)
                self.segment_start_time = timestamp
                self.last_crossing_time = timestamp
                segment_started = True

            if side is not None:
                if self.current_side is None:
                    self.current_side = side
                elif side != self.current_side:
                    self.current_side = side
                    self.board_count += 1
                    self.last_crossing_time = timestamp
                    crossed = True

        return CrossingResult(
            board_count=self.board_count,
            current_side=self.current_side,
            crossed=crossed,
            timed_out=timed_out,
            segment_started=segment_started,
            ended_board_count=ended_board_count,
            ended_start_frame=ended_start_frame,
            ended_start_time=ended_start_time,
        )

    def close(self, frame_data) -> CrossingResult:
        """Close the current segment at video end without starting another one."""
        if not self.active:
            return CrossingResult(board_count=0, current_side=None)
        result = CrossingResult(
            board_count=0,
            current_side=None,
            timed_out=True,
            ended_board_count=self.board_count,
            ended_start_frame=self.segment_start_frame,
            ended_start_time=self.segment_start_time,
        )
        self.reset()
        return result

    def _table_x(self, frame_data) -> float | None:
        position = frame_data.ball_table_pos
        if position is None:
            return None
        value = float(position[0])
        return value if np.isfinite(value) and 0.0 <= value <= 1.0 else None

    def _confirmed_side(self, table_x: float) -> str | None:
        if table_x <= self.left_enter:
            return "left"
        if table_x >= self.right_enter:
            return "right"
        return None
