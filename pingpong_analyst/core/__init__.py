"""核心业务逻辑模块"""
from .data_aligner import DataAligner, FrameData, HitEvent
from .rally_detector import RallyDetector, RallyState, RallySegment
from .ball_crossing import BallCrossingCounter, CrossingResult
from .clip_exporter import ClipExporter

__all__ = [
    "DataAligner",
    "FrameData",
    "HitEvent",
    "RallyDetector",
    "RallyState",
    "RallySegment",
    "BallCrossingCounter",
    "CrossingResult",
    "ClipExporter",
    "ActionAnalyzer",
    "VideoAnalyzer",
]


def __getattr__(name):
    """按需加载重模型依赖, 允许纯算法模块独立测试。"""
    if name == "VideoAnalyzer":
        from .video_analyzer import VideoAnalyzer
        return VideoAnalyzer
    if name == "ActionAnalyzer":
        from .action_analyzer import ActionAnalyzer
        return ActionAnalyzer
    raise AttributeError(name)
