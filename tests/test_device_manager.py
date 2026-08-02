"""测试设备管理器"""
import platform
from pingpong_analyst.utils.device_manager import DeviceManager, DeviceType


def test_auto_detect_macos():
    """macOS 优先使用 MPS，不可用时回退到 CPU。"""
    info = DeviceManager.detect("auto")
    if platform.system() == "Darwin":
        assert info.device_type in (DeviceType.MPS, DeviceType.CPU)
        if info.device_type == DeviceType.MPS:
            assert info.torch_device == "mps"
        else:
            assert info.torch_device == "cpu"
            assert info.supports_half is False
    # Linux/Windows 服务器测试时, 应该是 CUDA 或 TENSORRT


def test_force_cpu():
    """强制 CPU 模式"""
    info = DeviceManager.detect("cpu")
    assert info.device_type == DeviceType.CPU
    assert info.torch_device == "cpu"


def test_singleton():
    """单例模式测试"""
    m1 = DeviceManager()
    m2 = DeviceManager()
    assert m1 is m2


def test_get_info_returns_device_info():
    info = DeviceManager.get_info()
    assert info.device_type in DeviceType
    assert isinstance(info.device_name, str)
    assert isinstance(info.total_memory_mb, int)
    assert isinstance(info.supports_half, bool)
