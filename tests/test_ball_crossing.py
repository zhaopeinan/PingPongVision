"""跨球台半区板数计数测试。"""

import pytest

from pingpong_analyst.core.ball_crossing import BallCrossingCounter
from pingpong_analyst.core.data_aligner import FrameData


def frame(index, timestamp, x=None):
    return FrameData(
        frame_idx=index,
        timestamp=timestamp,
        ball_table_pos=(x, 0.5) if x is not None else None,
    )


def test_left_right_left_counts_two_boards():
    counter = BallCrossingCounter(timeout_seconds=2.0)

    assert counter.update(frame(0, 0.0, 0.20)).board_count == 0
    assert counter.update(frame(1, 0.1, 0.80)).board_count == 1
    assert counter.update(frame(2, 0.2, 0.20)).board_count == 2


def test_hysteresis_does_not_count_center_jitter():
    counter = BallCrossingCounter(timeout_seconds=2.0)

    for index, x in enumerate((0.48, 0.52, 0.48, 0.52)):
        result = counter.update(frame(index, index * 0.1, x))

    assert result.board_count == 0


def test_timeout_ends_segment_and_starts_new_side_state():
    counter = BallCrossingCounter(timeout_seconds=2.0)
    counter.update(frame(0, 0.0, 0.20))
    crossing = counter.update(frame(1, 0.1, 0.80))
    timed_out = counter.update(frame(2, 2.11, 0.80))

    assert crossing.board_count == 1
    assert timed_out.timed_out is True
    assert timed_out.ended_board_count == 1
    assert timed_out.board_count == 0
    assert timed_out.segment_started is True


def test_missing_ball_before_timeout_can_resume_crossing():
    counter = BallCrossingCounter(timeout_seconds=2.0)
    counter.update(frame(0, 0.0, 0.20))
    counter.update(frame(1, 0.5, None))
    result = counter.update(frame(2, 1.0, 0.80))

    assert result.crossed is True
    assert result.board_count == 1


def test_timeout_range_is_validated():
    with pytest.raises(ValueError):
        BallCrossingCounter(timeout_seconds=0.1)
    with pytest.raises(ValueError):
        BallCrossingCounter(timeout_seconds=11.0)
