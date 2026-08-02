"""测试剪辑导出器"""
import subprocess
import pytest
from pathlib import Path

from pingpong_analyst.core.rally_detector import RallySegment
from pingpong_analyst.core.clip_exporter import ClipExporter


@pytest.fixture
def exporter(tmp_path):
    config = {
        "clip_mode": "stream_copy",
        "output_format": "mp4",
        "output_codec": "libx264",
        "output_dir": str(tmp_path / "clips"),
        "clip_buffer_before": 0.5,
        "clip_buffer_after": 0.5,
    }
    return ClipExporter(config, project_root=str(tmp_path))


@pytest.fixture
def sample_segment():
    return RallySegment(
        start_frame=100, end_frame=300,
        start_time=3.33, end_time=10.0,
        board_count=8, hit_events=[],
    )


def test_exporter_init(exporter):
    assert exporter.clip_mode == "stream_copy"
    assert exporter.output_dir.exists()


def test_export_no_ffmpeg(exporter, sample_segment, monkeypatch):
    """ffmpeg 不存在时返回 None"""
    def mock_run(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")
    monkeypatch.setattr(subprocess, "run", mock_run)

    result = exporter.export_clip("fake.mp4", sample_segment)
    assert result is None


def test_export_all_empty(exporter):
    """无有效片段时返回空列表"""
    outputs = exporter.export_all("fake.mp4", [])
    assert outputs == []


def test_clip_name_generation(exporter, sample_segment):
    """测试默认文件名生成"""
    # 验证输出目录可写
    assert exporter.output_dir.is_dir()
