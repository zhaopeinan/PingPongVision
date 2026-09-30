"""过网后弹跳落点估计。"""
from pingpong_analyst.core.ball_landing import TrajectorySample, apply_landing, find_bounce
from pingpong_analyst.core.data_aligner import HitEvent


def _sample(frame, image_y, table_pos):
    return TrajectorySample(frame_idx=frame, image_y=image_y, table_pos=table_pos)


def test_find_bounce_at_image_y_peak_on_destination_half():
    samples = [
        _sample(1, 100, (0.56, 0.5)),
        _sample(2, 110, (0.64, 0.22)),
        _sample(3, 120, (0.72, 0.22)),
        _sample(4, 108, (0.78, 0.22)),
    ]
    bounce = find_bounce(samples, "right")
    assert bounce is not None
    assert bounce.frame_idx == 3
    assert bounce.table_pos == (0.72, 0.22)


def test_find_bounce_ignores_net_and_departure_half():
    samples = [
        _sample(1, 100, (0.5, 0.5)),
        _sample(2, 130, (0.5, 0.5)),
        _sample(3, 90, (0.5, 0.5)),
        _sample(4, 100, (0.2, 0.5)),
        _sample(5, 120, (0.3, 0.5)),
        _sample(6, 90, (0.25, 0.5)),
    ]
    assert find_bounce(samples, "right") is None


def test_apply_landing_sets_bounce_label():
    hit = HitEvent(1, 0.1, (10, 10), "left", {}, board_index=1, placement="过网 · 中路")
    apply_landing(hit, _sample(8, 120, (0.72, 0.22)))
    assert hit.placement_kind == "bounce"
    assert hit.landing_pos == (0.72, 0.22)
    assert hit.placement == "右半台 · 中台 · 左路"
