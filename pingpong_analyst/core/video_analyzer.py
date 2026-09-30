"""
视频分析器 - 主管线编排

    回合流水线: 解码 -> TrackNet 跨区板数 -> 剪辑 -> 过网帧 YOLO 补击球角度
    动作分析由独立的 ActionAnalyzer 负责，避免与 TrackNet 同时占显存。
采用流水线并行: 分阶段释放显存
帧预取线程 + 批量推理以最大化 GPU 利用率。
"""
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from loguru import logger

from ..utils.device_manager import DeviceManager, DeviceType
from ..utils.config import get_config
from ..models import YOLOPoseDetector, MediaPipePoseAnalyzer, TrackNetTracker
from .data_aligner import DataAligner
from .rally_detector import RallyDetector, RallySegment
from .clip_exporter import ClipExporter
from .hit_enrichment import enrich_segments_from_video
from .table_calibration import TableCalibration
from .table_geometry import TableGeometry


class PipelineStuckError(Exception):
    """管线卡死异常：长时间无帧处理进展。"""
    pass


class VideoAnalyzer:
    """
    视频分析主管线

    用法:
        analyzer = VideoAnalyzer(config_path="config.yaml")
        segments = analyzer.analyze("input.mp4")
        analyzer.export_clips("input.mp4", segments)
    """

    def __init__(
        self,
        config_path: str = None,
        tracknet_model_path: str | None = None,
        table_calibration: TableCalibration | None = None,
        no_crossing_timeout_seconds: float | None = None,
    ):
        self.config = get_config(config_path)
        self.tracknet_model_path = tracknet_model_path
        self.table_calibration = table_calibration

        # 设备检测
        self.device_info = DeviceManager.detect(self.config.device_mode)

        # 初始化组件 (延迟加载模型)
        self._yolo: Optional[YOLOPoseDetector] = None
        self._mediapipe: Optional[MediaPipePoseAnalyzer] = None
        self._tracknet: Optional[TrackNetTracker] = None

        rally_config = dict(self.config.get("analysis", "rally", default={}))
        if no_crossing_timeout_seconds is not None:
            rally_config["no_crossing_timeout_seconds"] = float(no_crossing_timeout_seconds)
        table_geometry = (
            TableGeometry.from_calibration(table_calibration)
            if table_calibration is not None
            else None
        )
        self.aligner = DataAligner(rally_config, table_geometry=table_geometry)
        self.rally_detector = RallyDetector(rally_config)
        self.clip_exporter = ClipExporter(
            self.config.get("video", default={}),
            project_root=str(Path(__file__).parent.parent.parent),
        )

        self.frame_stride = self.config.get("analysis", "frame_stride", default=1)
        self.batch_size = self.config.get("analysis", "batch_size", default=8)
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

    def analyze(
        self,
        video_path: str,
        max_frames: int = -1,
        progress_callback: Callable[[float], None] | None = None,
        cancel_event: threading.Event | None = None,
        start_frame: int = 0,
        resume: bool = False,
        enrich_hits: bool | None = None,
    ) -> list[RallySegment]:
        """
        分析视频, 返回有效回合片段

        Args:
            video_path: 视频文件路径
            max_frames: 最大处理帧数 (-1=全部, 测试用)
            cancel_event: 取消信号, set() 后尽快终止
            enrich_hits: 是否在 TrackNet 之后对过网帧跑 YOLO 补角度；
                默认读取 ``analysis.rally.enrich_hits``。
        """
        video_path = str(video_path)
        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")

        logger.info(f"开始分析视频: {video_path}")
        if not resume:
            self._reset_run_state()

        if enrich_hits is None:
            enrich_hits = bool(
                self.config.get("analysis", "rally", "enrich_hits", default=True)
            )

        def on_pipeline_progress(ratio: float) -> None:
            if progress_callback:
                progress_callback(min(0.9, max(0.0, float(ratio)) * 0.9))

        # 第一遍只加载 TrackNet 做跨区板数；人体检测放到板数完成之后。
        self._load_models_stage1(include_pose=False)
        segments: list[RallySegment] = []

        try:
            segments = self._run_pipeline(
                video_path, max_frames, on_pipeline_progress, cancel_event, start_frame=start_frame
            )
            if self._tracknet:
                self._tracknet.unload()
                self._tracknet = None
            if enrich_hits and segments and not (
                cancel_event is not None and cancel_event.is_set()
            ):
                self._enrich_hits_with_pose(
                    video_path, segments, progress_callback, cancel_event
                )
            elif progress_callback:
                progress_callback(1.0)
        finally:
            if self._yolo:
                self._yolo.unload()
                self._yolo = None
            if self._tracknet:
                self._tracknet.unload()
                self._tracknet = None
            self._unload_mediapipe()

        logger.info(f"分析完成, 共 {len(segments)} 个有效回合")
        return segments

    def _enrich_hits_with_pose(
        self,
        video_path: str,
        segments: list[RallySegment],
        progress_callback: Callable[[float], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """卸载 TrackNet 后，只对过网附近帧跑 YOLO 补击球角度。"""
        yolo_cfg = self.config.get("models", "yolo_pose", default={})
        detector = YOLOPoseDetector(self.device_info, yolo_cfg)
        if not detector.load():
            logger.warning("YOLO 不可用，击球事件仅保留半区与球速")
            if progress_callback:
                progress_callback(1.0)
            return

        lookback = int(
            self.config.get("analysis", "rally", "hit_pose_lookback_frames", default=2)
        )

        def on_enrich_progress(ratio: float) -> None:
            if progress_callback:
                progress_callback(0.9 + min(max(float(ratio), 0.0), 1.0) * 0.1)

        self._yolo = detector
        try:
            logger.info("开始为击球事件补充人体角度")
            enrich_segments_from_video(
                video_path,
                segments,
                detector,
                table_geometry=self.aligner.table_geometry,
                lookback_frames=lookback,
                cancel_event=cancel_event,
                progress_callback=on_enrich_progress,
            )
        except Exception as exc:
            logger.warning(f"击球姿态补全失败，保留半区与球速: {exc}")
            if progress_callback:
                progress_callback(1.0)
        finally:
            detector.unload()
            self._yolo = None

    def _reset_run_state(self):
        """清理上一段视频的时序数据, 保证一次 analyze 对应一个独立运行。"""
        self.aligner.reset()
        self.rally_detector.reset()
        self._checkpoint_frame_idx = 0

    def _run_pipeline(
        self,
        video_path: str,
        max_frames: int,
        progress_callback: Callable[[float], None] | None = None,
        cancel_event: threading.Event | None = None,
        start_frame: int = 0,
    ) -> list[RallySegment]:
        """执行处理管线（帧预取 + 批量推理）"""
        import av
        from tqdm import tqdm

        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        total_frames = stream.frames or 0

        if self._tracknet and not self._tracknet.is_cv_fallback:
            self._tracknet.prepare_background(video_path, cancel_event=cancel_event)

        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("任务已取消")

        if max_frames > 0:
            total_frames = min(total_frames, max_frames)

        logger.info(
            f"视频信息: {video_path} | FPS={fps:.1f} | 总帧数={total_frames} | batch_size={self.batch_size}"
        )

        # ---------- 帧预取线程 ----------
        frame_queue: queue.Queue = queue.Queue(maxsize=64)
        stop_event = threading.Event()

        _decoder_alive = threading.Event()
        _decoder_alive.set()

        def _decode_thread():
            """后台解码线程：将帧放入队列供主线程消费。"""
            try:
                local_idx = 0
                for frame in container.decode(video=0):
                    if stop_event.is_set():
                        break
                    # 断点续传：跳过已处理帧
                    if local_idx < start_frame:
                        local_idx += 1
                        continue
                    img = frame.to_ndarray(format="bgr24")
                    frame_queue.put(("frame", img, frame.pts))
                    local_idx += 1
            except Exception as e:
                frame_queue.put(("error", e))
            finally:
                _decoder_alive.clear()
                frame_queue.put(None)  # 结束信号

        t = threading.Thread(target=_decode_thread, daemon=True)
        t.start()

        # ---------- 主循环：批量消费 ----------
        frame_idx = start_frame
        last_progress = -1 if start_frame == 0 else int(start_frame / max(1, total_frames) * 100)
        pbar = tqdm(total=total_frames, desc="分析进度", unit="帧", initial=start_frame)
        use_batch = self._tracknet and not self._tracknet.is_cv_fallback
        import time
        watchdog_timeout = 120  # 秒：无进展超时
        last_activity_time = time.time()

        while True:
            # 看门狗：检查解码线程是否存活且队列无数据
            if not _decoder_alive.is_set() and frame_queue.empty():
                break  # 解码正常结束
            if time.time() - last_activity_time > watchdog_timeout:
                stop_event.set()
                t.join(timeout=2)
                container.close()
                pbar.close()
                raise PipelineStuckError(
                    f"管线卡死：{watchdog_timeout}秒无进展 (已处理 {frame_idx} 帧)"
                )

            # 检查取消信号
            if cancel_event is not None and cancel_event.is_set():
                stop_event.set()
                t.join(timeout=2)
                container.close()
                logger.info("分析任务被取消")
                return self.rally_detector.get_valid_segments()

            # 攒一批帧
            batch_frames = []
            batch_indices = []
            while len(batch_frames) < self.batch_size:
                try:
                    item = frame_queue.get(timeout=1)
                except queue.Empty:
                    # 超时后检查取消信号，然后继续等
                    if cancel_event is not None and cancel_event.is_set():
                        stop_event.set()
                        t.join(timeout=2)
                        container.close()
                        logger.info("分析任务被取消")
                        return self.rally_detector.get_valid_segments()
                    continue
                if item is None:
                    break  # 解码结束
                if item[0] == "error":
                    raise item[1]
                _, img, _ = item
                if max_frames > 0 and frame_idx >= max_frames:
                    break
                if frame_idx % self.frame_stride != 0:
                    frame_idx += 1
                    pbar.update(1)
                    last_activity_time = time.time()
                    continue
                batch_frames.append(img)
                batch_indices.append(frame_idx)
                frame_idx += 1
                pbar.update(1)
                last_activity_time = time.time()

            if not batch_frames:
                break

            # 批量推理
            if use_batch:
                ball_positions = self._tracknet.detect_ball_positions_batch(batch_frames)
            else:
                ball_positions = [None] * len(batch_frames)

            # 逐帧后处理
            for img, fidx, ball_pos in zip(batch_frames, batch_indices, ball_positions):
                timestamp = fidx / fps
                ball_conf = 1.0 if ball_pos else 0.0

                frame_data = self.aligner.add_frame(
                    frame_idx=fidx,
                    timestamp=timestamp,
                    ball_pos=ball_pos,
                    ball_confidence=ball_conf,
                    persons=[],
                    pose_landmarks=None,
                    arm_angles=None,
                    frame_size=(img.shape[1], img.shape[0]),
                )

                completed_segment = self.rally_detector.update(frame_data)
                if completed_segment is not None:
                    self._tracknet.notify_rally_end()

            # 保存 checkpoint（供断点续传）
            self._checkpoint_frame_idx = frame_idx

            if progress_callback and total_frames > 0:
                progress = min(1.0, frame_idx / total_frames)
                progress_percent = int(progress * 100)
                if progress_percent != last_progress:
                    progress_callback(progress)
                    last_progress = progress_percent

        # 清理
        stop_event.set()
        t.join(timeout=2)
        pbar.close()
        container.close()

        # flush 最后一个回合
        self.rally_detector.flush(frame_idx, frame_idx / fps)
        if progress_callback:
            progress_callback(1.0)

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
