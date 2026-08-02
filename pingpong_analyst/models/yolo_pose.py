"""
 YOLO 人体检测/姿态检测器封装
- 服务器: 优先加载 TensorRT .engine (FP16)
- macOS Apple Silicon: 使用 PyTorch MPS
- 普通 detect 权重也可作为人物框提供器，姿态由 MediaPipe 补全
"""
from typing import Any

import numpy as np
from loguru import logger

from .base import BaseModelWrapper
from ..utils.device_manager import DeviceInfo, DeviceType
from ..utils.config import resolve_project_path


class YOLOPoseDetector(BaseModelWrapper):
    """YOLO11-Pose 封装"""

    def load(self) -> bool:
        if self._loaded:
            return True

        engine_path = resolve_project_path(self.config.get("engine"))
        weights_path = resolve_project_path(self.config.get("weights", "yolo11s-pose.pt"))
        use_half = self.config.get("half", True) and self.device_info.supports_half

        # 优先 TensorRT engine (服务器)
        if self.device_info.device_type == DeviceType.TENSORRT and engine_path:
            loaded = self._load_tensorrt(str(engine_path), use_half)
            if loaded:
                self._loaded = True
                return True
            logger.warning("TensorRT engine加载失败, 回退到PyTorch权重")

        # 回退: PyTorch权重 (macOS CPU / 服务器CUDA)
        loaded = self._load_pytorch(str(weights_path), use_half)
        if loaded:
            self._loaded = True
            return True

        logger.error(f"YOLO11-Pose 加载失败: {weights_path}")
        return False

    def _load_tensorrt(self, engine_path: str, half: bool) -> bool:
        """
        TensorRT 加载示例

        YOLO11 支持直接导出并加载 TensorRT engine:
            from ultralytics import YOLO
            model = YOLO("yolo11s-pose.pt")
            model.export(format="engine", half=True)  # 生成 .engine

        加载时只需将路径指向 .engine 文件即可
        """
        try:
            from ultralytics import YOLO
            import os

            if not os.path.exists(engine_path):
                logger.warning(f"TensorRT engine 不存在: {engine_path}")
                return False

            logger.info(f"加载 TensorRT engine: {engine_path} (FP16={half})")
            self._model = YOLO(engine_path, task=self.config.get("task", "pose"))
            self._backend = "tensorrt"
            logger.info("YOLO11-Pose TensorRT 加载成功")
            return True
        except Exception as e:
            logger.error(f"TensorRT 加载异常: {e}")
            return False

    def _load_pytorch(self, weights_path: str, half: bool) -> bool:
        """PyTorch权重加载 (macOS CPU / 服务器CUDA)"""
        try:
            from ultralytics import YOLO
            import os

            if not os.path.exists(weights_path):
                logger.warning(f"权重文件不存在: {weights_path}")
                # 尝试从ultralytics自动下载
                logger.info(f"尝试自动下载: {weights_path}")
                model_name = os.path.basename(weights_path)
                self._model = YOLO(model_name, task=self.config.get("task"))
            else:
                self._model = YOLO(weights_path, task=self.config.get("task"))

            self._device = self.device_info.torch_device
            self._backend = "pytorch"
            self._half = half and self.device_info.device_type != DeviceType.CPU
            logger.info(
                f"YOLO 加载成功: task={self._model.task}, device={self._device}, half={self._half}"
            )
            return True
        except Exception as e:
            logger.error(f"PyTorch 加载异常: {e}")
            return False

    def infer(self, frame: np.ndarray) -> Any:
        """
        单帧推理

        Returns:
            ultralytics 结果对象, 包含:
            - result.keypoints: 关键点坐标 [N, K, 2] + conf [N, K]
            - result.boxes: 检测框
        """
        if not self._loaded:
            raise RuntimeError("模型未加载, 请先调用 load()")

        if self._backend == "tensorrt":
            # TensorRT engine 推理
            results = self._model(
                frame,
                conf=self.config.get("conf", 0.25),
                iou=self.config.get("iou", 0.45),
                imgsz=self.config.get("imgsz", 640),
                verbose=False,
            )
        else:
            # PyTorch 推理
            results = self._model(
                frame,
                device=self._device,
                conf=self.config.get("conf", 0.25),
                iou=self.config.get("iou", 0.45),
                imgsz=self.config.get("imgsz", 640),
                half=self._half,
                verbose=False,
            )

        return results[0]

    def extract_keypoints(self, frame: np.ndarray) -> list[dict]:
        """
        提取人体关键点

        Returns:
            [{"bbox": [x1,y1,x2,y2], "keypoints": np.ndarray[K,3], "conf": float}, ...]
            keypoints 格式: [x, y, confidence] per keypoint
        """
        result = self.infer(frame)
        persons = []

        boxes = result.boxes
        # Pose 权重带 keypoints；普通 detect 权重没有 keypoints，但仍可提供
        # person bbox，后续由 MediaPipe 对每个框单独计算姿态。
        kpts = result.keypoints.data if result.keypoints is not None else None
        if boxes is None:
            return persons

        for i in range(len(boxes)):
            class_id = int(boxes[i].cls[0].cpu()) if boxes[i].cls is not None else 0
            class_name = str(result.names.get(class_id, "person"))
            if class_name.lower() != "person":
                continue
            person = {
                "bbox": boxes[i].xyxy[0].cpu().numpy().tolist(),
                "keypoints": kpts[i].cpu().numpy() if kpts is not None else None,
                "conf": float(boxes[i].conf[0].cpu()),
            }
            persons.append(person)

        return persons
