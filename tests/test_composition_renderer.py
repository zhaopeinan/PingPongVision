"""Focused tests for precise highlight composition."""

from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pingpong_analyst.core.composition_renderer import (
    CompositionError,
    CompositionRenderer,
)
from pingpong_analyst.utils.device_manager import DeviceInfo, DeviceManager, DeviceType


def segments(*ranges):
    return [
        {"edit_start_time": start, "edit_end_time": end}
        for start, end in ranges
    ]


def test_build_filter_graph_with_audio_concats_trimmed_segments(tmp_path):
    renderer = CompositionRenderer("ffmpeg", output_dir=tmp_path)
    graph = renderer.build_filter_graph(
        segments((1.25, 3.5), (8.0, 10.0)),
        has_audio=True,
    )

    assert "trim=start=1.250:end=3.500" in graph.filter_complex
    assert "atrim=start=8.000:end=10.000" in graph.filter_complex
    assert "concat=n=2:v=1:a=1" in graph.filter_complex
    assert graph.maps == ["[vout]", "[aout]"]


def test_build_filter_graph_without_audio_maps_only_video(tmp_path):
    renderer = CompositionRenderer("ffmpeg", output_dir=tmp_path)
    graph = renderer.build_filter_graph(segments((0.0, 2.0)), has_audio=False)

    assert "concat=n=1:v=1:a=0" in graph.filter_complex
    assert graph.maps == ["[vout]"]
    assert "atrim" not in graph.filter_complex


@pytest.mark.parametrize(
    "value",
    [[], [{"edit_start_time": 2.0, "edit_end_time": 2.0}],
     [{"edit_start_time": -1.0, "edit_end_time": 2.0}],
     [{"edit_start_time": float("nan"), "edit_end_time": 2.0}]],
)
def test_invalid_segments_are_rejected(tmp_path, value):
    renderer = CompositionRenderer(output_dir=tmp_path)
    with pytest.raises(CompositionError):
        renderer.build_filter_graph(value, has_audio=False)


def test_build_command_uses_safe_argv_and_audio_mapping(tmp_path):
    renderer = CompositionRenderer(
        ffmpeg_binary="ffmpeg",
        output_dir=tmp_path,
        preset="slow",
        crf=19,
    )
    source = tmp_path / "source file.mp4"
    output = tmp_path / "final cut.mp4"
    command = renderer.build_command(
        source,
        segments((1.0, 2.0)),
        output,
        has_audio=True,
        encoder="libx264",
    )

    assert isinstance(command, list)
    assert str(source) in command
    assert str(output) in command
    assert "-filter_complex" in command
    assert ["-map", "[vout]"] == command[command.index("-map") : command.index("-map") + 2]
    assert "[aout]" in command
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-preset") + 1] == "slow"
    assert command[command.index("-crf") + 1] == "19"
    assert "-c:a" in command and command[command.index("-c:a") + 1] == "aac"
    assert "-movflags" in command and "+faststart" in command


def test_detect_audio_uses_pyav_and_closes_container(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    class Container:
        streams = [SimpleNamespace(type="video"), SimpleNamespace(type="audio")]

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    container = Container()
    fake_av = SimpleNamespace(open=lambda path: container)
    monkeypatch.setitem(sys.modules, "av", fake_av)

    assert CompositionRenderer().detect_audio(source) is True
    assert container.closed is True


def test_auto_encoder_uses_nvenc_only_for_cuda_with_ffmpeg_support(monkeypatch):
    info = DeviceInfo(DeviceType.TENSORRT, "Tesla T4", "cuda:0", 16384, True)
    monkeypatch.setattr(DeviceManager, "get_info", classmethod(lambda cls: info))
    monkeypatch.setattr(
        "pingpong_analyst.core.composition_renderer.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=" V.... h264_nvenc           NVIDIA NVENC H.264 encoder\n",
        ),
    )

    assert CompositionRenderer().select_encoder() == "h264_nvenc"


def test_auto_encoder_falls_back_without_cuda_or_ffmpeg_support(monkeypatch):
    info = DeviceInfo(DeviceType.MPS, "Apple GPU", "mps", 0, True)
    monkeypatch.setattr(DeviceManager, "get_info", classmethod(lambda cls: info))
    renderer = CompositionRenderer()
    assert renderer.select_encoder() == "libx264"

    info = DeviceInfo(DeviceType.CUDA, "Tesla T4", "cuda:0", 16384, True)
    monkeypatch.setattr(DeviceManager, "get_info", classmethod(lambda cls: info))
    monkeypatch.setattr(
        "pingpong_analyst.core.composition_renderer.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=" V.... libx264\n"),
    )
    assert renderer.select_encoder() == "libx264"


def test_parse_progress_is_monotonic_and_capped():
    values = []
    progress = CompositionRenderer.parse_progress(
        [
            "out_time_ms=500000\n",
            "progress=continue\n",
            "out_time_ms=1500000\n",
            "out_time_ms=900000\n",
            "progress=end\n",
        ],
        total_duration=2.0,
        progress_callback=lambda value, message: values.append(value),
    )

    assert progress == 1.0
    assert values == sorted(values)
    assert all(0.0 <= value <= 1.0 for value in values)
    assert values[-1] == 1.0


def test_render_progress_does_not_regress(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(CompositionRenderer, "detect_audio", lambda self, path: False)
    monkeypatch.setattr(CompositionRenderer, "select_encoder", lambda self, requested=None: "libx264")
    monkeypatch.setattr(
        "pingpong_analyst.core.composition_renderer.subprocess.Popen",
        lambda command, **kwargs: FakeProcess(
            command,
            progress="out_time_ms=800000\nout_time_ms=200000\nprogress=end\n",
            output=True,
        ),
    )
    updates = []
    CompositionRenderer().render(
        source,
        segments((0.0, 1.0)),
        output,
        progress_callback=lambda value, message: updates.append(value),
    )

    assert updates == sorted(updates)
    assert updates[-1] == 1.0


class FakeProcess:
    def __init__(self, command, *, returncode=0, progress="", stderr="", output=False):
        self.command = command
        self.returncode = returncode
        self.stdout = io.StringIO(progress)
        self.stderr = io.StringIO(stderr)
        self.terminated = False
        if output:
            Path(command[-1]).write_bytes(b"mp4")

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


def test_render_parses_progress_and_returns_metadata(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "exports" / "highlight.mp4"
    source.write_bytes(b"source")
    created = {}

    monkeypatch.setattr(CompositionRenderer, "detect_audio", lambda self, path: False)
    monkeypatch.setattr(CompositionRenderer, "select_encoder", lambda self, requested=None: "libx264")

    def popen(command, **kwargs):
        created["kwargs"] = kwargs
        process = FakeProcess(
            command,
            progress="out_time_ms=500000\nprogress=continue\nout_time_ms=1000000\nprogress=end\n",
            output=True,
        )
        created["process"] = process
        return process

    monkeypatch.setattr("pingpong_analyst.core.composition_renderer.subprocess.Popen", popen)
    updates = []
    result = CompositionRenderer(timeout_seconds=2).render(
        source,
        segments((0.0, 1.0), (2.0, 3.0)),
        output,
        progress_callback=lambda value, message: updates.append(value),
    )

    assert result["output_path"] == str(output)
    assert result["has_audio"] is False
    assert updates == sorted(updates)
    assert updates[-1] == 1.0
    assert created["kwargs"]["shell"] is False
    assert created["kwargs"]["stdin"] is subprocess.DEVNULL
    assert str(source) in created["process"].command
    assert str(output) in created["process"].command


def test_render_nonzero_exit_includes_stderr_tail(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(CompositionRenderer, "detect_audio", lambda self, path: False)
    monkeypatch.setattr(CompositionRenderer, "select_encoder", lambda self, requested=None: "libx264")
    monkeypatch.setattr(
        "pingpong_analyst.core.composition_renderer.subprocess.Popen",
        lambda command, **kwargs: FakeProcess(
            command, returncode=1, stderr="invalid input video\n"
        ),
    )

    with pytest.raises(CompositionError, match="invalid input video"):
        CompositionRenderer().render(source, segments((0.0, 1.0)), output)


def test_render_timeout_is_converted_to_composition_error(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(CompositionRenderer, "detect_audio", lambda self, path: False)
    monkeypatch.setattr(CompositionRenderer, "select_encoder", lambda self, requested=None: "libx264")

    class TimeoutProcess(FakeProcess):
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(self.command, timeout)

    monkeypatch.setattr(
        "pingpong_analyst.core.composition_renderer.subprocess.Popen",
        lambda command, **kwargs: TimeoutProcess(command),
    )

    with pytest.raises(CompositionError, match="超时"):
        CompositionRenderer(timeout_seconds=0.1).render(
            source, segments((0.0, 1.0)), output
        )


def test_render_missing_output_is_an_error(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(CompositionRenderer, "detect_audio", lambda self, path: False)
    monkeypatch.setattr(CompositionRenderer, "select_encoder", lambda self, requested=None: "libx264")
    monkeypatch.setattr(
        "pingpong_analyst.core.composition_renderer.subprocess.Popen",
        lambda command, **kwargs: FakeProcess(command),
    )

    with pytest.raises(CompositionError, match="未生成输出文件"):
        CompositionRenderer().render(source, segments((0.0, 1.0)), output)


def test_render_reports_missing_ffmpeg(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(CompositionRenderer, "detect_audio", lambda self, path: False)
    monkeypatch.setattr(CompositionRenderer, "select_encoder", lambda self, requested=None: "libx264")

    def missing(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr("pingpong_analyst.core.composition_renderer.subprocess.Popen", missing)

    with pytest.raises(CompositionError, match="FFmpeg 不可用"):
        CompositionRenderer().render(source, segments((0.0, 1.0)), output)
