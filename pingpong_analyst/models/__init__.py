"""模型封装模块"""
from .base import BaseModelWrapper

__all__ = [
    "BaseModelWrapper",
    "YOLOPoseDetector",
    "MediaPipePoseAnalyzer",
    "TrackNetTracker",
]


def __getattr__(name):
    """按需加载 OpenCV、MediaPipe 和 Ultralytics 模型封装。"""
    modules = {
        "YOLOPoseDetector": ".yolo_pose",
        "MediaPipePoseAnalyzer": ".mediapipe_pose",
        "TrackNetTracker": ".tracknet",
    }
    module_name = modules.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = __import__(f"{__name__}{module_name}", fromlist=[name])
    return getattr(module, name)
