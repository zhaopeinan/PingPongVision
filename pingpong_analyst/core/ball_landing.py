"""过网后用画面高度折返估计台面弹跳落点。"""
from dataclasses import dataclass

from .table_geometry import landing_label


@dataclass(frozen=True)
class TrajectorySample:
    frame_idx: int
    image_y: float
    table_pos: tuple[float, float]


def _on_table(pos: tuple[float, float], margin: float = 0.08) -> bool:
    x, y = pos
    return -margin <= x <= 1.0 + margin and -margin <= y <= 1.0 + margin


def _on_destination(pos: tuple[float, float], side: str | None) -> bool:
    x = pos[0]
    if side == "right":
        return x >= 0.52
    if side == "left":
        return x <= 0.48
    return False


def find_bounce(
    samples: list[TrajectorySample],
    destination_side: str | None,
    min_drop_px: float = 3.0,
    max_frame_gap: int = 4,
) -> TrajectorySample | None:
    """画面 y 向下增大。下落后再上升的局部最高点视为弹跳。"""
    usable = [
        sample
        for sample in samples
        if _on_table(sample.table_pos) and _on_destination(sample.table_pos, destination_side)
    ]
    if len(usable) < 3:
        return None
    for index in range(1, len(usable) - 1):
        prev, current, nxt = usable[index - 1], usable[index], usable[index + 1]
        if (
            current.frame_idx - prev.frame_idx > max_frame_gap
            or nxt.frame_idx - current.frame_idx > max_frame_gap
        ):
            continue
        drop = current.image_y - prev.image_y
        rise = current.image_y - nxt.image_y
        if drop >= min_drop_px and rise >= min_drop_px:
            return current
    return None


def apply_landing(hit, sample: TrajectorySample) -> None:
    hit.landing_pos = sample.table_pos
    hit.placement = landing_label(sample.table_pos)
    hit.placement_kind = "bounce"
