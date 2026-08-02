"""模型封装基类"""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from loguru import logger

from ..utils.device_manager import DeviceInfo, DeviceType


class BaseModelWrapper(ABC):
    """所有模型封装的基类, 负责设备适配与显存管理"""

    def __init__(self, device_info: DeviceInfo, config: dict):
        self.device_info = device_info
        self.config = config
        self._loaded = False
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> bool:
        """加载模型权重, 返回是否成功"""
        pass

    @abstractmethod
    def infer(self, frame: np.ndarray) -> Any:
        """单帧推理"""
        pass

    def unload(self):
        """释放模型, 释放显存 (流水线并行时使用)"""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False

        # 主动清理GPU缓存
        if self.device_info.device_type in (DeviceType.CUDA, DeviceType.TENSORRT, DeviceType.MPS):
            try:
                import torch
                if self.device_info.device_type == DeviceType.MPS:
                    torch.mps.empty_cache()
                else:
                    torch.cuda.empty_cache()
                logger.info(f"{self.__class__.__name__} 模型已卸载, 加速器缓存已清理")
            except Exception:
                pass
        else:
            logger.info(f"{self.__class__.__name__} 模型已卸载")

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False
