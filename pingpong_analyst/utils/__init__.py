"""设备抽象层 - 自动检测并切换 CPU/GPU/TensorRT"""
from .device_manager import DeviceManager, DeviceType

__all__ = ["DeviceManager", "DeviceType"]
