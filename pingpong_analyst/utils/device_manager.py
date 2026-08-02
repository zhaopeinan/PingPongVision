"""
设备抽象层
- Apple Silicon macOS: 优先使用 PyTorch MPS, 不可用时回退 CPU
- 服务器(T4): 优先使用 TensorRT, 其次 CUDA
"""
import os
import platform
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class DeviceType(Enum):
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"
    TENSORRT = "tensorrt"


@dataclass
class DeviceInfo:
    """设备信息"""
    device_type: DeviceType
    device_name: str
    # PyTorch设备字符串: "cpu" / "mps" / "cuda:0"
    torch_device: str
    # 显存总量(MB), CPU为0
    total_memory_mb: int
    # 是否支持半精度
    supports_half: bool


class DeviceManager:
    """统一设备管理器"""

    _instance = None
    _info: DeviceInfo = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def detect(cls, mode: str = "auto") -> DeviceInfo:
        """
        检测可用设备

        Args:
            mode: "auto" 自动检测 / "cpu" / "mps" / "cuda" / "tensorrt"
        """
        if mode == "auto":
            info = cls._auto_detect()
        elif mode == "cpu":
            info = cls._force_cpu()
        elif mode == "mps":
            info = cls._try_mps() or cls._force_cpu()
        elif mode == "cuda":
            info = cls._try_cuda()
        elif mode == "tensorrt":
            info = cls._try_tensorrt()
        else:
            logger.warning(f"未知设备模式: {mode}, 回退到auto")
            info = cls._auto_detect()

        cls._info = info
        cls._apply_env_settings(info)
        logger.info(
            f"设备检测完成: {info.device_type.value} | "
            f"{info.device_name} | 显存: {info.total_memory_mb}MB | "
            f"半精度: {info.supports_half}"
        )
        return info

    @classmethod
    def get_info(cls) -> DeviceInfo:
        """获取当前设备信息"""
        if cls._info is None:
            return cls.detect("auto")
        return cls._info

    # ---------- 检测逻辑 ----------

    @staticmethod
    def _auto_detect() -> DeviceInfo:
        system = platform.system()

        # Apple Silicon macOS: 使用 Metal 加速 YOLO；不可用时回退 CPU。
        if system == "Darwin":
            info = DeviceManager._try_mps()
            if info is not None:
                return info
            logger.info("检测到 macOS, MPS 不可用, 使用 CPU 模式")
            return DeviceManager._force_cpu()

        # Linux/Windows: 尝试 TensorRT -> CUDA -> CPU
        info = DeviceManager._try_tensorrt()
        if info is None:
            info = DeviceManager._try_cuda()
        if info is None:
            info = DeviceManager._force_cpu()
        return info

    @staticmethod
    def _force_cpu() -> DeviceInfo:
        return DeviceInfo(
            device_type=DeviceType.CPU,
            device_name=platform.processor() or "CPU",
            torch_device="cpu",
            total_memory_mb=0,
            supports_half=False,
        )

    @staticmethod
    def _try_mps() -> DeviceInfo | None:
        """检测 Apple Silicon 的 PyTorch Metal 后端。"""
        if platform.system() != "Darwin":
            return None
        try:
            import torch

            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                return None
            return DeviceInfo(
                device_type=DeviceType.MPS,
                device_name="Apple Silicon GPU (Metal)",
                torch_device="mps",
                total_memory_mb=0,
                supports_half=True,
            )
        except Exception as e:
            logger.debug(f"MPS 不可用: {e}")
            return None

    @staticmethod
    def _try_cuda() -> DeviceInfo | None:
        try:
            import torch
            if not torch.cuda.is_available():
                return None

            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_mb = props.total_memory // (1024 * 1024)

            logger.info(f"检测到 CUDA GPU: {gpu_name} ({total_mb}MB)")
            return DeviceInfo(
                device_type=DeviceType.CUDA,
                device_name=gpu_name,
                torch_device="cuda:0",
                total_memory_mb=total_mb,
                supports_half=True,
            )
        except Exception as e:
            logger.debug(f"CUDA 不可用: {e}")
            return None

    @staticmethod
    def _try_tensorrt() -> DeviceInfo | None:
        # TensorRT 依赖 CUDA, 先检测CUDA
        cuda_info = DeviceManager._try_cuda()
        if cuda_info is None:
            return None

        try:
            import tensorrt as trt
            logger.info(f"检测到 TensorRT: {trt.__version__}")
            return DeviceInfo(
                device_type=DeviceType.TENSORRT,
                device_name=f"{cuda_info.device_name} + TensorRT",
                torch_device=cuda_info.torch_device,
                total_memory_mb=cuda_info.total_memory_mb,
                supports_half=True,
            )
        except ImportError:
            logger.debug("TensorRT 未安装, 使用 CUDA")
            return cuda_info
        except Exception as e:
            logger.warning(f"TensorRT 初始化失败: {e}, 使用 CUDA")
            return cuda_info

    @staticmethod
    def _apply_env_settings(info: DeviceInfo):
        """应用环境变量设置"""
        if info.device_type in (DeviceType.CUDA, DeviceType.TENSORRT):
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

            # 限制PyTorch显存预分配
            try:
                import torch
                if hasattr(torch.cuda, "set_per_process_memory_fraction"):
                    fraction = 0.8  # 默认80%
                    torch.cuda.set_per_process_memory_fraction(fraction)
                    logger.info(f"PyTorch 显存限制: {fraction*100:.0f}%")
            except Exception as e:
                logger.debug(f"设置显存限制失败: {e}")
