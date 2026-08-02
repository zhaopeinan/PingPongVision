"""
视频分析器 - 主管线编排

    回合流水线: 解码 -> 追踪(TrackNet) -> 球轨迹对齐 -> 回合检测 -> 剪辑
    动作分析由独立的 ActionAnalyzer 负责，避免回合剪辑加载人体模型。
采用流水线并行: 分阶段释放显存
"""
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from ..utils.device_manager import DeviceManager, DeviceType
from ..utils.config import get_config
from ..models import YOLOPoseDetector, MediaPipePoseAnalyzer, TrackNetTracker
from .data_aligner import DataAligner, HitEvent
from .rally_detector import RallyDetector, RallySegment
from .clip_exporter import ClipExporter


class VideoAnalyzer:
    """
    视频分析主管线

    用法:
        analyzer = VideoAnalyzer(config_path="config.yaml")
        segments = analyzer.analyze("input.mp4")
        analyzer.export_clips("input.mp4", segments)
    """

    def __init__(self, config_path: str = None, tracknet_model_path: str | None = None):
        self.config = get_config(config_path)
        self.tracknet_model_path = tracknet_model_path

        # 设备检测
        self.device_info = DeviceManager.detect(self.config.device_mode)

        # 初始化组件 (延迟加载模型)
        self._yolo: Optional[YOLOPoseDetector] = None
        self._mediapipe: Optional[MediaPipePoseAnalyzer] = None
        self._tracknet: Optional[TrackNetTracker] = None

        self.aligner = DataAligner(self.config.get("analysis", "rally", default={}))
        self.rally_detector = RallyDetector(
            self.config.get("analysis", "rally", default={})
        )
        self.clip_exporter = ClipExporter(
            self.config.get("video", default={}),
            project_root=str(Path(__file__).parent.parent.parent),
        )

        self.frame_stride = self.config.get("analysis", "frame_stride", default=1)
        self.mp_enabled = self.config.get("models", "mediapipe_pose", "enabled", default=True)

        logger.info("VideoAnalyzer 初始化完成")

    def _load_models_stage1(self, include_pose: bool = False):
        """加载回合所需模型；默认只加载 TrackNet。"""
        mp_cfg = self.config.get("models", "yolo_pose", default={})
        tn_cfg = dict(self.config.get("models", "tracknet", default={}))
        if self.tracknet_model_path:
            tn_cfg["weights"] = self.tracknet_model_path

        self._tracknet = TrackNetTracker(self.device_info, tn_cfg)
        self._tracknet.load()

        if include_pose:
            self._yolo = YOLOPoseDetector(self.device_info, mp_cfg)
            self._yolo.load()

    def _load_mediapipe(self):
        """阶段2: 按需加载MediaPipe (流水线并行时省显存)"""
        if not self.mp_enabled:
            return
        mp_cfg = self.config.get("models", "mediapipe_pose", default={})
        self._mediapipe = MediaPipePoseAnalyzer(self.device_info, mp_cfg)
        self._mediapipe.load()

    def _unload_mediapipe(self):
        """卸载MediaPipe释放显存"""
        if self._mediapipe is not None:
            self._mediapipe.unload()
            self._mediapipe = None

    def analyze(self, video_path: str, max_frames: int = -1) -> list[RallySegment]:
        """
        分析视频, 返回有效回合片段

        Args:
            video_path: 视频文件路径
            max_frames: 最大处理帧数 (-1=全部, 测试用)
        """
        video_path = str(video_path)
        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")

        logger.info(f"开始分析视频: {video_path}")
        self._reset_run_state()

        # 回合分析只加载 TrackNet，不执行人体/躯干分析。
        self._load_models_stage1(include_pose=False)

        try:
            segments = self._run_pipeline(video_path, max_frames)
        finally:
            # 释放所有模型显存
            if self._yolo:
                self._yolo.unload()
            if self._tracknet:
                self._tracknet.unload()
            self._unload_mediapipe()

        logger.info(f"分析完成, 共 {len(segments)} 个有效回合")
        return segments

    def _reset_run_state(self):
        """清理上一段视频的时序数据, 保证一次 analyze 对应一个独立运行。"""
        self.aligner.reset()
        self.rally_detector.reset()

    def _run_pipeline(self, video_path: str, max_frames: int) -> list[RallySegment]:
        """执行处理管线"""
        import av
        from tqdm import tqdm

        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        total_frames = stream.frames or 0

        if self._tracknet and not self._tracknet.is_cv_fallback:
            self._tracknet.prepare_background(video_path)

        if max_frames > 0:
            total_frames = min(total_frames, max_frames)

        logger.info(f"视频信息: {video_path} | FPS={fps:.1f} | 总帧数={total_frames}")

        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="分析进度", unit="帧")

        for frame in container.decode(video=0):
            if max_frames > 0 and frame_idx >= max_frames:
                break

            if frame_idx % self.frame_stride != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            # 解码帧 -> numpy
            img = frame.to_ndarray(format="bgr24")
            timestamp = frame_idx / fps

            # 回合模式只运行球检测；人体模型由独立的动作分析入口负责。
            if self._tracknet.is_cv_fallback:
                ball_pos = None
            else:
                ball_pos = self._tracknet.detect_ball_position(img)
            ball_conf = 1.0 if ball_pos else 0.0
            persons = []
            arm_angles = None
            pose_landmarks = None

            # 时空对齐
            frame_data = self.aligner.add_frame(
                frame_idx=frame_idx,
                timestamp=timestamp,
                ball_pos=ball_pos,
                ball_confidence=ball_conf,
                persons=persons,
                pose_landmarks=pose_landmarks,
                arm_angles=arm_angles,
            )

            # 击球事件匹配
            if self._tracknet.is_cv_fallback:
                hit_events = []
            else:
                hit_events = self.aligner.match_hit_events(require_persons=False)

            # 回合检测状态机
            self.rally_detector.update(frame_data, hit_events)

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        container.close()

        # flush 最后一个回合
        self.rally_detector.flush(frame_idx, frame_idx / fps)

        return self.rally_detector.get_valid_segments()

    def export_clips(self, video_path: str, segments: list[RallySegment]) -> list[str]:
        """导出剪辑片段"""
        return self.clip_exporter.export_all(video_path, segments)

    def analyze_and_clip(self, video_path: str, max_frames: int = -1) -> list[str]:
        """一键分析+剪辑"""
        segments = self.analyze(video_path, max_frames)
        if not segments:
            logger.info("无有效回合, 跳过剪辑")
            return []
        return self.export_clips(video_path, segments)
