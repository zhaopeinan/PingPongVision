"""
实时追踪可视化引擎

逐帧处理视频, 在画面上叠加:
- 球体轨迹 (拖尾效果)
- 人体骨骼关键点
- 击球事件高亮
- 回合状态指示

输出为 MJPEG 流, 供前端实时观看
"""
import io
import time
from collections import deque
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from loguru import logger

from .utils.device_manager import DeviceManager
from .utils.config import get_config
from .models import YOLOPoseDetector, MediaPipePoseAnalyzer, TrackNetTracker
from .core.data_aligner import DataAligner
from .core.rally_detector import RallyDetector, RallyState
from .core.table_calibration import TableCalibration
from .core.table_geometry import TableGeometry


# COCO 17关键点骨骼连接 (YOLO-Pose)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
    (5, 11), (6, 12), (11, 12),             # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),  # 腿
]

# 配色方案 (OKLCH-inspired, 暖橙主色)
COLOR_BALL = (0, 165, 255)          # 橙色 - 球
COLOR_BALL_TRAIL = (0, 100, 200)    # 深橙 - 拖尾
COLOR_SKELETON = (255, 200, 80)     # 青黄 - 骨骼
COLOR_KEYPOINT = (80, 255, 180)     # 青绿 - 关键点
COLOR_HIT_FLASH = (0, 0, 255)       # 红色 - 击球闪光
COLOR_RALLY_ACTIVE = (0, 255, 100)  # 绿色 - 回合进行
COLOR_RALLY_IDLE = (128, 128, 128)  # 灰色 - 等待
COLOR_HUD_BG = (20, 20, 30)         # HUD背景
COLOR_HUD_TEXT = (220, 220, 230)    # HUD文字
COLOR_HUD_ACCENT = (0, 165, 255)    # HUD强调


class TrackingVisualizer:
    """实时追踪可视化器"""

    def __init__(
        self,
        config_path: str = None,
        mode: str = "rally",
        tracknet_model_path: str | None = None,
        table_calibration: TableCalibration | None = None,
        no_crossing_timeout_seconds: float | None = None,
    ):
        if mode not in {"rally", "action"}:
            raise ValueError(f"未知追踪模式: {mode}")
        self.config = get_config(config_path)
        self.mode = mode
        self.tracknet_model_path = tracknet_model_path
        self.device_info = DeviceManager.detect(self.config.device_mode)

        self._yolo: YOLOPoseDetector | None = None
        self._tracknet: TrackNetTracker | None = None
        self._mediapipe: MediaPipePoseAnalyzer | None = None

        rally_config = dict(self.config.get("analysis", "rally", default={}))
        if no_crossing_timeout_seconds is not None:
            rally_config["no_crossing_timeout_seconds"] = float(no_crossing_timeout_seconds)
        table_geometry = (
            TableGeometry.from_calibration(table_calibration)
            if table_calibration is not None
            else None
        )
        self._rally_config = rally_config
        self._table_geometry = table_geometry
        self.aligner = DataAligner(rally_config, table_geometry=table_geometry)
        self.rally_detector = RallyDetector(rally_config)

        self._ball_trail: deque = deque(maxlen=30)  # 球轨迹拖尾
        self._hit_flash_frames: int = 0  # 击球闪光剩余帧数
        self._board_count: int = 0  # 当前得分段板数
        self._frame_idx: int = 0
        self._person_filter: list[list[float]] | None = None  # 指定追踪的人物bbox列表
        self._tracking_warning: str | None = None

        logger.info(f"TrackingVisualizer 初始化完成: mode={mode}")

    @property
    def tracking_warning(self) -> str | None:
        """当前追踪链路的可用性警告，供 API 状态消息和前端展示。"""
        return self._tracking_warning

    def set_person_filter(self, boxes: list[list[float]]):
        """设置人物过滤区域，只追踪bbox与这些区域有重叠的人"""
        self._person_filter = boxes
        logger.info(f"已设置人物过滤: {len(boxes)} 个目标区域")

    def prepare_background(self, video_path: str) -> bool:
        """为回合模式准备 TrackNetV3 的全局中值背景。"""
        if self.mode != "rally" or self._tracknet is None or self._tracknet.is_cv_fallback:
            return False
        return self._tracknet.prepare_background(video_path)

    def _filter_persons(self, persons: list[dict]) -> list[dict]:
        """按person_filter过滤人物"""
        if not self._person_filter or not persons:
            return persons

        def _iou(box1, box2) -> float:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            if x2 <= x1 or y2 <= y1:
                return 0.0
            inter = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            return inter / min(area1, area2)

        filtered = []
        for p in persons:
            bbox = p.get("bbox")
            if not bbox:
                continue
            for target in self._person_filter:
                if _iou(bbox, target) > 0.3:
                    filtered.append(p)
                    break
        return filtered

    def load_models(self, progress_callback=None):
        """加载所有模型

        Args:
            progress_callback: 可选回调 fn(percent: int, message: str)
        """
        def _report(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        tn_cfg = dict(self.config.get("models", "tracknet", default={}))
        if self.tracknet_model_path:
            tn_cfg["weights"] = self.tracknet_model_path
        mp_mp_cfg = self.config.get("models", "mediapipe_pose", default={})

        if self.mode == "rally":
            _report(10, "正在加载 TrackNet (球的追踪)...")
            self._tracknet = TrackNetTracker(self.device_info, tn_cfg)
            self._tracknet.load()
            if self._tracknet.is_cv_fallback:
                self._tracking_warning = "TrackNet 权重不可用，已禁用可信球与板数计算"
                _report(100, self._tracking_warning)
        else:
            mp_cfg = self.config.get("models", "yolo_pose", default={})
            _report(10, "正在加载 YOLO-Pose (人体检测)...")
            self._yolo = YOLOPoseDetector(self.device_info, mp_cfg)
            self._yolo.load()
            if mp_mp_cfg.get("enabled", True):
                _report(60, "正在加载 MediaPipe (关节角度)...")
                self._mediapipe = MediaPipePoseAnalyzer(self.device_info, mp_mp_cfg)
                self._mediapipe.load()

        _report(100, "模型加载完成")

    def unload_models(self):
        """卸载所有模型"""
        if self._yolo:
            self._yolo.unload()
        if self._tracknet:
            self._tracknet.unload()
        if self._mediapipe:
            self._mediapipe.unload()

    def reset(self):
        """重置状态, 准备新视频"""
        self.aligner.reset()
        self.rally_detector.reset()
        self._ball_trail.clear()
        self._hit_flash_frames = 0
        self._board_count = 0
        self._frame_idx = 0

    def process_frame(self, img: np.ndarray, fps: float) -> tuple[np.ndarray, dict]:
        """
        处理单帧, 返回标注画面 + 元数据

        Args:
            img: BGR 帧图像
            fps: 视频FPS
        Returns:
            (annotated_frame, metadata_dict)
        """
        timestamp = self._frame_idx / fps
        annotated = img.copy()

        # 1. 回合模式追踪球；动作模式只处理人体姿态。
        ball_pos = None
        if (
            self.mode == "rally"
            and self._tracknet
            and self._tracknet.is_loaded
            and not self._tracknet.is_cv_fallback
        ):
            ball_pos = self._tracknet.detect_ball_position(img)

        if ball_pos:
            self._ball_trail.append(ball_pos)
            self._draw_ball(annotated, ball_pos)
        else:
            self._ball_trail.append(None)

        # 2. 人体姿态
        persons = []
        if self.mode == "action" and self._yolo and self._yolo.is_loaded:
            persons = self._yolo.extract_keypoints(img)
            persons = self._filter_persons(persons)
            self._draw_skeleton(annotated, persons)

        # 3. MediaPipe 关节角度
        arm_angles = None
        if self.mode == "action" and self._mediapipe and self._mediapipe.is_loaded:
            persons = self._mediapipe.enrich_persons(img, persons)
            arm_angles = self._mediapipe.select_display_angles(persons)

        # 4. 时空对齐 + 事件检测
        frame_data = self.aligner.add_frame(
            frame_idx=self._frame_idx,
            timestamp=timestamp,
            ball_pos=ball_pos,
            ball_confidence=1.0 if ball_pos else 0.0,
            persons=persons,
            arm_angles=arm_angles,
            frame_size=(img.shape[1], img.shape[0]),
        )

        # 回合模式的板数由球跨区状态机计算，动作模式不运行板数逻辑。
        hit_events = []
        completed_segment = self.rally_detector.update(frame_data, hit_events)
        prev_boards = self._board_count
        self._board_count = self.rally_detector.current_board_count if self.mode == "rally" else 0
        if self.mode == "rally" and self._board_count > prev_boards:
            self._hit_flash_frames = 8
        if completed_segment is not None and self._tracknet is not None:
            self._tracknet.notify_rally_end()

        # 5. 绘制叠加层
        if self._hit_flash_frames > 0:
            self._draw_hit_flash(annotated)
            self._hit_flash_frames -= 1

        # 6. HUD
        kmh = frame_data.ball_speed_kmh
        metadata = {
            "frame": self._frame_idx,
            "timestamp": round(timestamp, 2),
            "ball_pos": [round(p, 1) for p in ball_pos] if ball_pos else None,
            "ball_speed": round(frame_data.ball_speed, 1),
            "ball_speed_kmh": round(kmh, 1) if kmh is not None else None,
            "calibrated": bool(frame_data.calibrated),
            "persons": len(persons),
            # 实时追踪要把当前进行中的回合也展示出来；视频结束后，
            # 未达到最小板数的回合仍会被离线结果过滤掉。
            "rally_count": len(self.rally_detector.get_valid_segments()) + (
                1 if self.rally_detector.state == RallyState.RALLY_ACTIVE else 0
            ),
            "board_count": self._board_count,
            "rally_state": self.rally_detector.state.value,
            "arm_angles": arm_angles,
            "mode": self.mode,
            "ball_tracking": "tracknet" if not self._tracking_warning else "unavailable",
            "warning": self._tracking_warning,
        }
        self._draw_hud(annotated, metadata)

        self._frame_idx += 1
        return annotated, metadata

    def _draw_ball(self, img: np.ndarray, pos: tuple[float, float]):
        """绘制球 + 拖尾"""
        x, y = int(pos[0]), int(pos[1])

        # 拖尾
        trail_points = [p for p in self._ball_trail if p is not None]
        if len(trail_points) > 1:
            for i in range(len(trail_points) - 1):
                alpha = (i + 1) / len(trail_points)
                thickness = max(1, int(alpha * 4))
                color = tuple(int(c * alpha) for c in COLOR_BALL_TRAIL)
                cv2.line(img,
                         (int(trail_points[i][0]), int(trail_points[i][1])),
                         (int(trail_points[i + 1][0]), int(trail_points[i + 1][1])),
                         color, thickness, cv2.LINE_AA)

        # 球本体 (带光晕)
        cv2.circle(img, (x, y), 14, COLOR_BALL, 1, cv2.LINE_AA)  # 外圈
        cv2.circle(img, (x, y), 10, COLOR_BALL, -1, cv2.LINE_AA)  # 实心
        cv2.circle(img, (x, y), 6, (255, 255, 255), -1, cv2.LINE_AA)  # 高光

    def _draw_skeleton(self, img: np.ndarray, persons: list[dict]):
        """绘制人体骨骼"""
        for person in persons:
            kpts = person.get("keypoints")
            if kpts is None or len(kpts) == 0:
                continue

            # 骨骼连线
            for (i, j) in SKELETON_CONNECTIONS:
                if i < len(kpts) and j < len(kpts):
                    p1 = kpts[i]
                    p2 = kpts[j]
                    if p1[2] > 0.3 and p2[2] > 0.3:
                        cv2.line(img,
                                 (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])),
                                 COLOR_SKELETON, 2, cv2.LINE_AA)

            # 关键点
            for kpt in kpts:
                if kpt[2] > 0.3:
                    cv2.circle(img, (int(kpt[0]), int(kpt[1])), 4,
                               COLOR_KEYPOINT, -1, cv2.LINE_AA)

            # 检测框
            bbox = person.get("bbox")
            if bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_SKELETON, 1, cv2.LINE_AA)

    def _draw_hit_flash(self, img: np.ndarray):
        """击球闪光效果"""
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), COLOR_HIT_FLASH, -1)
        alpha = 0.15 * (self._hit_flash_frames / 8.0)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def _draw_hud(self, img: np.ndarray, meta: dict):
        """绘制HUD信息面板"""
        h, w = img.shape[:2]

        # 顶部状态栏
        bar_h = 44
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # 回合状态指示灯
        state = meta["rally_state"]
        if state == "rally_active":
            indicator_color = COLOR_RALLY_ACTIVE
            state_text = "RALLY ACTIVE"
        else:
            indicator_color = COLOR_RALLY_IDLE
            state_text = "STANDBY"

        cv2.circle(img, (24, bar_h // 2), 6, indicator_color, -1, cv2.LINE_AA)
        if state == "rally_active":
            # 脉冲效果
            pulse_r = 6 + int(4 * (1 + np.sin(self._frame_idx * 0.3)) / 2)
            cv2.circle(img, (24, bar_h // 2), pulse_r, indicator_color, 1, cv2.LINE_AA)

        cv2.putText(img, state_text, (40, bar_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HUD_TEXT, 1, cv2.LINE_AA)

        # 板数 (大号)
        board_text = f"BOARDS: {meta['board_count']}"
        cv2.putText(img, board_text, (w // 2 - 60, bar_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HUD_ACCENT, 2, cv2.LINE_AA)

        # 时间戳
        ts_text = f"{meta['timestamp']:.1f}s | F:{meta['frame']}"
        cv2.putText(img, ts_text, (w - 200, bar_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD_TEXT, 1, cv2.LINE_AA)

        # 左下角: 球速
        if meta["ball_pos"]:
            kmh = meta.get("ball_speed_kmh")
            if meta.get("calibrated"):
                ball_text = f"BALL  {kmh:.0f} km/h" if kmh is not None else "BALL  -- km/h"
            else:
                ball_text = f"BALL  {meta['ball_speed']:.0f} px/f"
            cv2.putText(img, ball_text, (12, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BALL, 1, cv2.LINE_AA)

        # 击球闪光时显示 HIT
        if self._hit_flash_frames > 0:
            hit_text = "HIT!"
            text_size = cv2.getTextSize(hit_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            tx = w // 2 - text_size[0] // 2
            ty = h // 2 + text_size[1] // 2
            cv2.putText(img, hit_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_HIT_FLASH, 3, cv2.LINE_AA)


def generate_mjpeg_stream(
    video_path: str,
    config_path: str = None,
    max_frames: int = -1,
    target_width: int = 960,
    mode: str = "rally",
    tracknet_model_path: str | None = None,
    table_calibration: TableCalibration | None = None,
    no_crossing_timeout_seconds: float | None = None,
) -> Generator[bytes, None, None]:
    """
    生成 MJPEG 流

    Yields:
        JPEG 编码的帧字节 (multipart/x-mixed-replace 格式)
    """
    visualizer = TrackingVisualizer(
        config_path,
        mode=mode,
        tracknet_model_path=tracknet_model_path,
        table_calibration=table_calibration,
        no_crossing_timeout_seconds=no_crossing_timeout_seconds,
    )
    visualizer.load_models()
    visualizer.prepare_background(video_path)

    try:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)

        for frame in container.decode(video=0):
            if max_frames > 0 and visualizer._frame_idx >= max_frames:
                break

            img = frame.to_ndarray(format="bgr24")

            annotated, meta = visualizer.process_frame(img, fps)
            # Calibration and ball coordinates use source-frame pixels. Resize
            # only after inference and drawing for the outgoing stream.
            h, w = annotated.shape[:2]
            if w > target_width:
                scale = target_width / w
                annotated = cv2.resize(annotated, (target_width, int(h * scale)))

            # 编码为 JPEG
            _, jpeg_buf = cv2.imencode(".jpg", annotated,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield jpeg_buf.tobytes()

        container.close()
    finally:
        visualizer.unload_models()
        logger.info("MJPEG 流生成结束")
