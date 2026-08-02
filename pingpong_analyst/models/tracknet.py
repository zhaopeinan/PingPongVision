"""
TrackNet 小球追踪器封装
- 服务器: TensorRT engine (FP16)
- 本地: PyTorch 权重 (MPS/CPU)
- 权重不可用时: 纯CV运动检测回退
"""
from collections import deque
from typing import Any

import cv2
import numpy as np
from loguru import logger

from .base import BaseModelWrapper
from .tracknet_preprocess import TrackNetPreprocessor
from ..utils.device_manager import DeviceInfo, DeviceType
from ..utils.config import resolve_project_path


class TrackNetTracker(BaseModelWrapper):
    """TrackNetV2/V3 乒乓球追踪器"""

    def __init__(self, device_info: DeviceInfo, config: dict):
        super().__init__(device_info, config)
        self.temporal_frames = max(1, int(config.get("temporal_frames", 3)))
        self.input_channels = max(3, int(config.get("input_channels", self.temporal_frames * 3)))
        self.output_channels = max(1, int(config.get("output_channels", self.temporal_frames)))
        self.background_mode = str(
            config.get("background_mode", config.get("bg_mode", "")) or ""
        )
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=self.temporal_frames)
        self._background_frame: np.ndarray | None = None
        self._prev_ball_pos: tuple[float, float] | None = None
        # 比赛中球绝大多数时间位于两名选手之间。用归一化区域做候选
        # 过滤，避免把地面、灯光和场边白色物体当成球。
        center_region = config.get("center_region", {}) or {}
        self._center_region = {
            "x_min": float(center_region.get("x_min", 0.25)),
            "x_max": float(center_region.get("x_max", 0.75)),
            "y_min": float(center_region.get("y_min", 0.10)),
            "y_max": float(center_region.get("y_max", 0.90)),
            "enabled": bool(center_region.get("enabled", True)),
            "outside_after_rally_frames": max(
                0, int(center_region.get("outside_after_rally_frames", 30))
            ),
        }
        self._outside_recovery_frames = 0
        self._allow_outside_current = False

    def _get_preprocessor(self) -> TrackNetPreprocessor:
        return TrackNetPreprocessor(
            width=self.config.get("input_width", 512),
            height=self.config.get("input_height", 288),
            temporal_frames=self.temporal_frames,
            background_mode=(
                self.background_mode
                if self.background_mode == "concat"
                else ("concat" if self.input_channels == self.temporal_frames * 3 + 3 else "")
            ),
        )

    def load(self) -> bool:
        if self._loaded:
            return True

        engine_path = resolve_project_path(self.config.get("engine"))
        weights_path = resolve_project_path(self.config.get("weights"))
        version = self.config.get("version", "v2")

        # 优先 TensorRT (服务器)
        if self.device_info.device_type == DeviceType.TENSORRT and engine_path:
            loaded = self._load_tensorrt(str(engine_path), version)
            if loaded:
                self._loaded = True
                return True

        # PyTorch权重
        if weights_path:
            loaded = self._load_pytorch(str(weights_path), version)
            if loaded:
                self._loaded = True
                return True

        # 回退: 纯CV运动检测 (macOS测试用)
        logger.warning("TrackNet 权重不可用, 回退到纯CV运动检测 (仅用于测试)")
        self._backend = "cv_fallback"
        self._load_cv_fallback()
        self._loaded = True
        return True

    @property
    def backend(self) -> str | None:
        """当前实际后端，用于区分真实模型和开发回退模式。"""
        return getattr(self, "_backend", None)

    @property
    def is_cv_fallback(self) -> bool:
        """是否没有可用于可信球追踪的 TrackNet 权重/engine。"""
        return self.backend == "cv_fallback"

    def _load_tensorrt(self, engine_path: str, version: str) -> bool:
        """
        TensorRT 加载 TrackNet

        导出流程:
            # TrackNetV2 PyTorch模型 -> ONNX -> TensorRT
            # 使用 trtexec 优化:
            trtexec --onnx=tracknet.onnx --saveEngine=tracknet.engine \
                    --fp16 --maxBatchSize=1 \
                    --workspace=2048

        加载示例:
            import tensorrt as trt
            import pycuda.driver as cuda

            runtime = trt.Runtime(trt.Logger())
            with open(engine_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()
        """
        try:
            import os
            import tensorrt as trt

            if not os.path.exists(engine_path):
                logger.warning(f"TensorRT engine 不存在: {engine_path}")
                return False

            logger.info(f"加载 TrackNet TensorRT engine: {engine_path}")

            # 初始化 TRT 运行时
            trt_logger = trt.Logger(trt.Logger.INFO)
            runtime = trt.Runtime(trt_logger)

            with open(engine_path, "rb") as f:
                engine_data = f.read()

            self._trt_engine = runtime.deserialize_cuda_engine(engine_data)
            self._trt_context = self._trt_engine.create_execution_context()
            self._backend = "tensorrt"

            # 分配缓冲区
            self._setup_trt_buffers()

            logger.info(f"TrackNet TensorRT 加载成功 (版本: {version})")
            return True
        except Exception as e:
            logger.error(f"TrackNet TensorRT 加载异常: {e}")
            return False

    def _setup_trt_buffers(self):
        """分配TensorRT推理缓冲区"""
        import pycuda.driver as cuda
        import pycuda.autoinit
        import numpy as np

        input_w = self.config.get("input_width", 640)
        input_h = self.config.get("input_height", 360)

        # 优先使用 engine 的实际绑定形状；老 engine 可能仍是单帧 3 通道。
        input_shape = None
        output_shape = None
        if hasattr(self._trt_engine, "get_binding_shape"):
            input_shape = tuple(int(v) for v in self._trt_engine.get_binding_shape(0))
            output_shape = tuple(int(v) for v in self._trt_engine.get_binding_shape(1))
        if not input_shape or any(v <= 0 for v in input_shape):
            input_shape = (1, self.input_channels, input_h, input_w)
        if not output_shape or any(v <= 0 for v in output_shape):
            output_shape = (1, input_h, input_w)

        if len(input_shape) != 4:
            raise ValueError(f"TrackNet 输入必须是 NCHW, 实际为 {input_shape}")
        self._trt_input_shape = input_shape
        self._trt_output_shape = output_shape
        self.input_channels = input_shape[1]
        self.output_channels = output_shape[1] if len(output_shape) == 4 else 1
        channel_frames = max(1, self.input_channels // 3)
        self.temporal_frames = max(
            1,
            channel_frames - 1 if self.background_mode == "concat" else channel_frames,
        )
        self._frame_buffer = deque(self._frame_buffer, maxlen=self.temporal_frames)

        # 分配GPU内存
        self._trt_input_size = int(np.prod(self._trt_input_shape) * np.float32().itemsize)
        self._trt_output_size = int(np.prod(self._trt_output_shape) * np.float32().itemsize)

        self._trt_d_input = cuda.mem_alloc(self._trt_input_size)
        self._trt_d_output = cuda.mem_alloc(self._trt_output_size)

        self._trt_bindings = [int(self._trt_d_input), int(self._trt_d_output)]

        logger.info(
            f"TrackNet TRT 缓冲区: input={self._trt_input_shape}, output={self._trt_output_shape}"
        )

    def _load_pytorch(self, weights_path: str, version: str) -> bool:
        """PyTorch权重加载"""
        try:
            import os
            import torch
            if not os.path.exists(weights_path):
                logger.warning(f"TrackNet权重不存在: {weights_path}")
                return False

            logger.info(f"加载 TrackNet PyTorch 权重: {weights_path}")
            self._device = self.device_info.torch_device
            self._backend = "pytorch"
            self._half = bool(self.config.get("half", True) and self.device_info.supports_half)

            # TrackNetV2 使用自定义网络结构
            # 这里简化为通用加载接口, 实际需根据TrackNet仓库实现
            try:
                # 先落到 CPU，再统一迁移整个模型，兼容 CPU/CUDA/MPS checkpoint。
                checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(weights_path, map_location="cpu")

            checkpoint_params = checkpoint.get("param_dict", {}) if isinstance(checkpoint, dict) else {}

            if isinstance(checkpoint, torch.nn.Module):
                self._model = checkpoint
            else:
                state_dict = checkpoint
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get("state_dict")
                    if state_dict is None:
                        state_dict = checkpoint.get("model_state_dict")
                    if state_dict is None:
                        state_dict = checkpoint.get("model", checkpoint)
                if isinstance(state_dict, torch.nn.Module):
                    self._model = state_dict
                elif isinstance(state_dict, dict):
                    from .tracknet_v3 import TrackNetV3

                    normalized_state = {
                        key.removeprefix("module.").removeprefix("model."): value
                        for key, value in state_dict.items()
                    }
                    input_weight = normalized_state.get("down_block_1.conv_1.conv.weight")
                    predictor_weight = normalized_state.get("predictor.weight")
                    if input_weight is None or predictor_weight is None:
                        raise KeyError(
                            "TrackNet checkpoint 缺少 down_block_1 或 predictor 权重"
                        )

                    # TrackNetV3_TableTennis 的 bg_mode=concat checkpoint 是
                    # 3 个背景通道 + seq_len * 3 个连续帧通道。
                    self.input_channels = int(input_weight.shape[1])
                    self.output_channels = int(predictor_weight.shape[0])
                    checkpoint_seq_len = checkpoint_params.get("seq_len")
                    if checkpoint_seq_len is not None:
                        self.temporal_frames = max(1, int(checkpoint_seq_len))
                    if not self.background_mode:
                        self.background_mode = str(checkpoint_params.get("bg_mode", "") or "")
                    self._frame_buffer = deque(
                        self._frame_buffer, maxlen=self.temporal_frames
                    )
                    self._model = TrackNetV3(self.input_channels, self.output_channels)
                    self._model.load_state_dict(normalized_state, strict=True)
                else:
                    raise TypeError("无法识别 TrackNet checkpoint 格式")
            if hasattr(self._model, "to"):
                self._model = self._model.to(self._device)
            if hasattr(self._model, "eval"):
                self._model.eval()
            if self._half and hasattr(self._model, "half"):
                self._model = self._model.half()

            logger.info("TrackNet PyTorch 加载成功")
            return True
        except Exception as e:
            logger.warning(f"TrackNet PyTorch 加载失败: {e}")
            self._model = None
            return False

    def _load_cv_fallback(self):
        """纯CV球检测回退 (macOS测试用)

        策略: 帧差法 + 亮度过滤 + 轨迹跟踪
        不依赖单帧检测，用轨迹连续性从噪声中提取球的运动路径
        """
        self._prev_frame = None
        self._prev_ball_pos = None
        self._ball_velocity = (0, 0)  # 球的速度估计
        self._ball_history = []  # 球位置历史
        self._miss_count = 0  # 连续丢失次数
        self._frame_buffer.clear()
        logger.info("TrackNet CV回退模式: 帧差法+亮度+轨迹跟踪")

    def notify_rally_end(self) -> None:
        """在回合结束后短暂允许球出现在中间走廊之外。"""
        if not self._center_region["enabled"]:
            return
        self._outside_recovery_frames = max(
            0,
            self._center_region["outside_after_rally_frames"],
        )
        # 新一分开始时不沿用上一回合的速度和位置，避免把场边物体
        # 通过距离连续性重新接回球轨迹。
        self._prev_ball_pos = None
        self._ball_velocity = (0, 0)

    def _is_ball_position_allowed(self, x: float, y: float, width: int, height: int) -> bool:
        if not self._center_region["enabled"] or self._allow_outside_current:
            return True
        return (
            self._center_region["x_min"] * width <= x <= self._center_region["x_max"] * width
            and self._center_region["y_min"] * height <= y <= self._center_region["y_max"] * height
        )

    def prepare_background(self, video_path: str) -> bool:
        """从视频采样生成 TrackNetV3 concat 所需的全局中值背景。"""
        if self.background_mode != "concat" or self._backend not in {"pytorch", "tensorrt"}:
            return False
        try:
            self._background_frame = TrackNetPreprocessor.sample_background(
                video_path,
                width=self.config.get("input_width", 512),
                height=self.config.get("input_height", 288),
                max_samples=max(8, int(self.config.get("background_samples", 180))),
            )
        except ValueError as exc:
            logger.warning(str(exc))
            return False
        logger.info(
            f"shape={self._background_frame.shape}"
        )
        return True

    def infer(self, frame: np.ndarray) -> Any:
        """推理并返回热力图；真实模型使用连续帧，CV回退使用当前帧。"""
        if not self._loaded:
            raise RuntimeError("模型未加载")

        if self._backend == "cv_fallback":
            return self._infer_cv(frame)

        self._frame_buffer.append(frame.copy())
        if self._backend == "tensorrt":
            return self._infer_tensorrt(frame)
        if self._backend == "pytorch":
            return self._infer_pytorch(frame)
        raise RuntimeError(f"未知 TrackNet backend: {self._backend}")

    def _prepare_temporal_input(self, frame: np.ndarray) -> np.ndarray:
        """将最近帧整理成 NCHW 的连续帧输入。"""
        frames = list(self._frame_buffer)
        if not frames:
            frames = [frame]

        stacked = self._get_preprocessor().prepare(frames, self._background_frame)[0]
        if stacked.shape[0] < self.input_channels:
            needed = self.input_channels - stacked.shape[0]
            repeats = (needed + 2) // 3
            padding = np.tile(stacked[-3:], (repeats, 1, 1))[:needed]
            stacked = np.concatenate([stacked, padding], axis=0)
        elif stacked.shape[0] > self.input_channels:
            stacked = stacked[-self.input_channels:]
        return np.ascontiguousarray(stacked[np.newaxis, ...])

    def _infer_tensorrt(self, frame: np.ndarray) -> np.ndarray:
        """TensorRT推理, 返回热力图"""
        import pycuda.driver as cuda
        import pycuda.autoinit
        import cv2

        input_w = self.config.get("input_width", 640)
        input_h = self.config.get("input_height", 360)

        batched = self._prepare_temporal_input(frame)

        cuda.memcpy_htod(self._trt_d_input, batched)
        self._trt_context.execute_v2(self._trt_bindings)
        heatmap = np.empty(self._trt_output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(heatmap, self._trt_d_output)

        return np.squeeze(heatmap)

    def _infer_pytorch(self, frame: np.ndarray) -> np.ndarray:
        """PyTorch推理, 返回热力图"""
        import torch
        import cv2

        input_w = self.config.get("input_width", 640)
        input_h = self.config.get("input_height", 360)

        batched = self._prepare_temporal_input(frame)
        tensor = torch.from_numpy(batched).to(self._device)

        if self._half:
            tensor = tensor.half()

        with torch.no_grad():
            heatmap = self._model(tensor)

        return heatmap.cpu().float().numpy().squeeze()

    def _infer_cv(self, frame: np.ndarray) -> np.ndarray:
        """纯CV球检测: 帧差法+亮度+轨迹跟踪

        步骤:
        1. 与上一帧做差，找运动区域
        2. 在运动区域中找小且亮的blob作为候选
        3. 用轨迹跟踪从候选中选出真正的球
        """
        h, w = frame.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            return heatmap

        # 1. 帧差
        diff = cv2.absdiff(frame, self._prev_frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_bin = cv2.threshold(diff_gray, 15, 255, cv2.THRESH_BINARY)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, kernel)

        # 2. 找候选: 运动区域中的小blob
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff_bin)
        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 2 or area > 80:  # 球很小
                continue
            cx, cy = centroids[i]
            # 亮度过滤: 球应该比画面平均亮
            g = gray[int(cy), int(cx)]
            if g < 80:  # 太暗，不是球
                continue
            candidates.append((cx, cy, area, g))

        candidates = [
            candidate
            for candidate in candidates
            if self._is_ball_position_allowed(candidate[0], candidate[1], w, h)
        ]

        # 3. 轨迹跟踪选球
        best = None
        if self._prev_ball_pos is not None and self._miss_count < 5:
            # 有上一帧位置: 预测当前位置，找最近的候选
            px, py = self._prev_ball_pos
            vx, vy = self._ball_velocity
            pred_x = px + vx
            pred_y = py + vy

            # 搜索半径: 速度越大，搜索范围越大
            search_r = max(50, (vx*vx + vy*vy)**0.5 * 2 + 30)

            near = []
            for cx, cy, area, g in candidates:
                dist = ((cx - pred_x)**2 + (cy - pred_y)**2)**0.5
                if dist < search_r:
                    # 评分: 距离越近越好，亮度越高越好
                    score = (1 - dist/search_r) * 0.7 + (g/255) * 0.3
                    near.append((cx, cy, area, g, score))

            if near:
                near.sort(key=lambda x: x[4], reverse=True)
                best = (near[0][0], near[0][1])
        else:
            # 无轨迹: 选最亮的小blob（偏好在画面中部）
            if candidates:
                candidates.sort(key=lambda x: x[3], reverse=True)
                # 从最亮的5个中选最接近画面中部的
                top = candidates[:5]
                center_x, center_y = w / 2, h / 2
                best = min(top, key=lambda c: ((c[0]-center_x)**2 + (c[1]-center_y)**2))[:2]
                best = (best[0], best[1])

        # 4. 更新轨迹
        if best is not None:
            bx, by = best
            if self._prev_ball_pos is not None:
                vx = bx - self._prev_ball_pos[0]
                vy = by - self._prev_ball_pos[1]
                # 平滑速度
                self._ball_velocity = (
                    self._ball_velocity[0] * 0.5 + vx * 0.5,
                    self._ball_velocity[1] * 0.5 + vy * 0.5,
                )
            self._prev_ball_pos = (bx, by)
            self._ball_history.append((bx, by))
            self._miss_count = 0

            # 画热力图峰
            heatmap[int(by), int(bx)] = 1.0
            cv2.circle(heatmap, (int(bx), int(by)), 5, 0.8, -1)
        else:
            self._miss_count += 1
            if self._miss_count > 10:
                # 丢失太久，重置轨迹
                self._prev_ball_pos = None
                self._ball_velocity = (0, 0)

        self._prev_frame = frame.copy()
        return heatmap

    def detect_ball_position(self, frame: np.ndarray) -> tuple[float, float] | None:
        """
        检测球的位置坐标

        Returns:
            (x, y) 像素坐标, 或 None
        """
        self._allow_outside_current = self._outside_recovery_frames > 0
        if self._outside_recovery_frames > 0:
            self._outside_recovery_frames -= 1

        frame_height, frame_width = frame.shape[:2]
        if (
            self._prev_ball_pos is not None
            and not self._allow_outside_current
            and not self._is_ball_position_allowed(
                self._prev_ball_pos[0], self._prev_ball_pos[1], frame_width, frame_height
            )
        ):
            self._prev_ball_pos = None

        heatmap = self.infer(frame)
        heatmap = np.asarray(heatmap)

        # TrackNetV3 输出 [N, T, H, W] 或 [T, H, W]，在线模式取窗口最后一帧。
        if heatmap.ndim == 4:
            heatmap = heatmap[0, -1]
        elif heatmap.ndim == 3:
            heatmap = heatmap[-1]
        if heatmap.ndim != 2:
            raise ValueError(f"TrackNet 热力图必须是二维，实际 shape={heatmap.shape}")

        threshold = self.config.get("peak_threshold", 0.3)

        if heatmap.max() < threshold:
            return None

        # 找所有候选峰值
        binary = (heatmap > threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        if num_labels <= 1:
            return None

        heatmap_height, heatmap_width = heatmap.shape
        scale_x = frame_width / heatmap_width
        scale_y = frame_height / heatmap_height
        candidates = []
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            # 取该区域的峰值强度
            peak_val = heatmap[labels == i].max()
            # 模型输入通常是 512x288，输出要映射回原视频像素坐标。
            x = cx * scale_x
            y = cy * scale_y
            if self._is_ball_position_allowed(x, y, frame_width, frame_height):
                candidates.append((x, y, peak_val))

        if not candidates:
            return None

        # 轨迹连续性: 如果有上一帧位置，优先选最近的候选
        if self._prev_ball_pos is not None:
            px, py = self._prev_ball_pos
            # 按距离排序，取距离最近的（但距离不能太远）
            max_dist = 150  # 最大允许移动距离（像素）
            near = [(c, ((c[0]-px)**2 + (c[1]-py)**2)**0.5) for c in candidates]
            near = [n for n in near if n[1] < max_dist]
            if near:
                near.sort(key=lambda n: n[1])
                best = near[0][0]
                self._prev_ball_pos = (best[0], best[1])
                return (best[0], best[1])

        # 否则取峰值最强的
        candidates.sort(key=lambda c: c[2], reverse=True)
        best = candidates[0]
        self._prev_ball_pos = (best[0], best[1])
        self._allow_outside_current = False
        return (best[0], best[1])

    def unload(self):
        """释放模型并清理跨视频的 temporal buffer。"""
        super().unload()
        self._frame_buffer.clear()
        self._background_frame = None
        self._prev_ball_pos = None
        self._outside_recovery_frames = 0
        self._allow_outside_current = False
