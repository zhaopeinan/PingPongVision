"""测试配置加载"""
from pathlib import Path
from pingpong_analyst.utils.config import Config, get_config, resolve_project_path


def test_default_config():
    config = Config(raw={})
    assert config.device_mode == "auto"
    assert config.gpu_memory_fraction == 0.8


def test_load_yaml(tmp_path):
    yaml_content = """
device:
  mode: cpu
  gpu_memory_fraction: 0.5
models:
  yolo_pose:
    weights: test.pt
    conf: 0.3
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = Config.load(config_path)
    assert config.device_mode == "cpu"
    assert config.gpu_memory_fraction == 0.5
    assert config.get("models", "yolo_pose", "conf") == 0.3


def test_get_nested_default():
    config = Config(raw={})
    assert config.get("nonexistent", "key", default="fallback") == "fallback"


def test_load_project_config():
    """加载项目默认 config.yaml"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.yaml"
    if config_path.exists():
        config = Config.load(config_path)
        assert config.device_mode in ("auto", "cpu", "cuda", "tensorrt")


def test_resolve_project_path_supports_root_fallback(tmp_path):
    root_weight = Path(__file__).parent.parent / "yolo11s-pose.pt"
    assert root_weight.exists()

    resolved = resolve_project_path("models/yolo11s-pose.pt")
    assert resolved == root_weight


def test_resolve_project_path_supports_models_fallback(tmp_path, monkeypatch):
    """本地根目录配置在服务器 models/ 布局下也能解析。"""
    import pingpong_analyst.utils.config as config_module

    models_weight = tmp_path / "models" / "yolo11s-pose.pt"
    models_weight.parent.mkdir()
    models_weight.write_bytes(b"weights")
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)

    assert config_module.resolve_project_path("yolo11s-pose.pt") == models_weight


def test_resolve_project_path_keeps_absolute_path(tmp_path):
    asset = tmp_path / "asset.engine"
    assert resolve_project_path(asset) == asset
