"""测试设备管理器"""
import platform
from types import SimpleNamespace

from pingpong_analyst.utils.device_manager import DeviceInfo, DeviceManager, DeviceType


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


def test_nvidia_smi_metrics_are_parsed(monkeypatch):
    info = DeviceInfo(DeviceType.CUDA, "NVIDIA CUDA", "cuda:0", 16384, True)
    output = "0, Tesla T4, 42, 3276, 16384, 54, 35.4, 70.0\n"
    monkeypatch.setattr(
        "pingpong_analyst.utils.device_manager.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output),
    )

    metrics = DeviceManager._query_nvidia_smi(info, {
        "available": False,
        "source": "unavailable",
        "gpu_utilization_percent": None,
        "memory_used_mb": None,
        "memory_total_mb": info.total_memory_mb,
        "temperature_c": None,
        "power_draw_w": None,
        "power_limit_w": None,
        "message": "实时指标不可用",
    })

    assert metrics["available"] is True
    assert metrics["gpu_name"] == "Tesla T4"
    assert metrics["gpu_utilization_percent"] == 42
    assert metrics["memory_used_mb"] == 3276
    assert metrics["memory_total_mb"] == 16384
    assert metrics["temperature_c"] == 54
    assert metrics["power_draw_w"] == 35.4
    assert metrics["power_limit_w"] == 70.0


def test_nvidia_smi_missing_is_reported(monkeypatch):
    info = DeviceInfo(DeviceType.CUDA, "NVIDIA CUDA", "cuda:0", 16384, True)

    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("pingpong_analyst.utils.device_manager.subprocess.run", missing)
    metrics = DeviceManager._query_nvidia_smi(info, {
        "available": False,
        "source": "unavailable",
        "gpu_utilization_percent": None,
        "memory_used_mb": None,
        "memory_total_mb": 16384,
        "temperature_c": None,
        "power_draw_w": None,
        "power_limit_w": None,
        "message": "实时指标不可用",
    })

    assert metrics["available"] is False
    assert metrics["message"] == "未找到 nvidia-smi"
