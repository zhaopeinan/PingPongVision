"""Render selected highlight segments into one precisely cut MP4.

The renderer deliberately has no API or project-store dependencies.  It accepts
plain segment dictionaries and returns plain metadata so it can be used by a
background task, a CLI, or a future service layer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import multiprocessing
from numbers import Real
from pathlib import Path
import os
import queue
import subprocess
import threading
import time
from typing import Callable, Iterable

from loguru import logger

from pingpong_analyst.utils.device_manager import DeviceManager


class CompositionError(RuntimeError):
    """Raised when a composition cannot be prepared or rendered."""


@dataclass(frozen=True)
class FilterGraph:
    """An FFmpeg filter graph and the output labels it exposes."""

    filter_complex: str
    maps: list[str]


_STREAM_END = object()


class CompositionRenderer:
    """Build and execute an FFmpeg composition for one source video."""

    def __init__(
        self,
        ffmpeg_binary: str = "ffmpeg",
        output_dir: Path = Path("output/exports"),
        encoder: str = "auto",
        preset: str = "fast",
        crf: int = 21,
        timeout_seconds: float = 3600.0,
    ):
        self.ffmpeg_binary = str(ffmpeg_binary)
        self.output_dir = Path(output_dir)
        self.encoder = encoder
        self.preset = str(preset)
        self.crf = int(crf)
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0 or not math.isfinite(self.timeout_seconds):
            raise ValueError("timeout_seconds must be a positive finite number")

    def build_filter_graph(
        self, segments: list[dict], has_audio: bool
    ) -> FilterGraph:
        """Create trim/normalize/concat filters in the supplied segment order."""
        normalized = self._validate_segments(segments)
        video_filters: list[str] = []
        audio_filters: list[str] = []

        for index, (start, end) in enumerate(normalized):
            video_filters.append(
                f"[0:v]trim=start={start:.3f}:end={end:.3f},"
                f"setpts=PTS-STARTPTS[v{index}]"
            )
            if has_audio:
                audio_filters.append(
                    f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
                    f"asetpts=PTS-STARTPTS[a{index}]"
                )

        if has_audio:
            concat_inputs = "".join(
                f"[v{index}][a{index}]" for index in range(len(normalized))
            )
            concat = (
                f"{concat_inputs}concat=n={len(normalized)}:v=1:a=1"
                "[vout][aout]"
            )
            filters = video_filters + audio_filters + [concat]
            return FilterGraph(";".join(filters), ["[vout]", "[aout]"])

        concat_inputs = "".join(f"[v{index}]" for index in range(len(normalized)))
        concat = f"{concat_inputs}concat=n={len(normalized)}:v=1:a=0[vout]"
        return FilterGraph(";".join(video_filters + [concat]), ["[vout]"])

    def build_command(
        self,
        source_path: Path,
        segments: list[dict],
        output_path: Path,
        has_audio: bool,
        encoder: str,
    ) -> list[str]:
        """Build an argv list; no shell interpolation is used anywhere."""
        graph = self.build_filter_graph(segments, has_audio)
        selected_encoder = self.select_encoder(encoder)
        command = [
            self.ffmpeg_binary,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(Path(source_path)),
            "-filter_complex",
            graph.filter_complex,
        ]
        for output_map in graph.maps:
            command.extend(["-map", output_map])

        command.extend(["-c:v", selected_encoder])
        if selected_encoder == "libx264":
            command.extend(["-preset", self.preset, "-crf", str(self.crf)])
            # 限制 CPU 线程数，避免占满所有核心导致 API 服务无响应
            max_threads = min(8, max(1, multiprocessing.cpu_count() // 4))
            command.extend(["-threads", str(max_threads)])
        elif selected_encoder == "h264_nvenc":
            command.extend(["-cq", "23"])

        if has_audio:
            command.extend(["-c:a", "aac"])
        command.extend(
            [
                "-movflags",
                "+faststart",
                "-progress",
                "pipe:1",
                "-nostats",
                str(Path(output_path)),
            ]
        )
        return command

    def render(
        self,
        source_path: Path,
        segments: list[dict],
        output_path: Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict:
        """Render selected segments and return output metadata.

        FFmpeg is invoked with one source input and a filter graph.  The source
        is inspected with PyAV first so files without audio do not receive a
        synthetic track.
        """
        source = Path(source_path)
        output = Path(output_path)
        normalized = self._validate_segments(segments)
        if not source.is_file():
            raise CompositionError(f"源视频不存在: {source}")
        if source.resolve() == output.resolve():
            raise CompositionError("输出文件不能覆盖源视频")

        output.parent.mkdir(parents=True, exist_ok=True)
        has_audio = self.detect_audio(source)
        selected_encoder = self.select_encoder(self.encoder)
        command = self.build_command(
            source,
            segments,
            output,
            has_audio,
            selected_encoder,
        )
        total_duration = sum(end - start for start, end in normalized)

        # A stale output must not make a failed FFmpeg invocation look successful.
        if output.exists():
            try:
                output.unlink()
            except OSError as exc:
                raise CompositionError(f"无法覆盖旧输出文件: {output}: {exc}") from exc

        if progress_callback:
            progress_callback(0.0, "准备渲染")

        progress, stderr_tail = self._execute_ffmpeg(
            command,
            total_duration,
            progress_callback,
        )
        if not output.is_file():
            detail = self._stderr_detail(stderr_tail)
            suffix = f": {detail}" if detail else ""
            raise CompositionError(f"FFmpeg 未生成输出文件: {output}{suffix}")

        if progress_callback and progress < 1.0:
            progress_callback(1.0, "渲染完成")

        return {
            "output_path": str(output),
            "output_filename": output.name,
            "duration": total_duration,
            "progress": 1.0,
            "has_audio": has_audio,
            "encoder": selected_encoder,
        }

    def detect_audio(self, source_path: Path) -> bool:
        """Return whether PyAV sees an audio stream in the source container."""
        try:
            import av
        except Exception as exc:
            raise CompositionError(f"PyAV 不可用，无法检测音频流: {exc}") from exc

        container = None
        try:
            container = av.open(str(source_path))
            streams = getattr(container, "streams", ())
            return any(getattr(stream, "type", None) == "audio" for stream in streams)
        except Exception as exc:
            raise CompositionError(f"无法读取源视频音频流: {exc}") from exc
        finally:
            if container is not None:
                close = getattr(container, "close", None)
                if callable(close):
                    close()

    def select_encoder(self, requested: str | None = None) -> str:
        """Select NVENC only for CUDA/TensorRT with confirmed FFmpeg support."""
        value = str(self.encoder if requested is None else requested).strip().lower()
        if value == "libx264":
            return "libx264"
        if value not in {"auto", "h264_nvenc"}:
            raise CompositionError(f"不支持的视频编码器: {value}")

        # Auto selection intentionally keeps macOS/MPS and CPU on libx264.
        if value == "auto" and self._cuda_or_tensorrt_available():
            if self._ffmpeg_supports_encoder("h264_nvenc"):
                return "h264_nvenc"
        return "libx264"

    @staticmethod
    def parse_progress(
        lines: Iterable[str],
        total_duration: float,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> float:
        """Parse FFmpeg ``-progress`` lines and return the last ratio."""
        if total_duration <= 0 or not math.isfinite(total_duration):
            raise CompositionError("渲染总时长必须为正数")

        last_progress = 0.0
        for raw_line in lines:
            line = raw_line.decode(errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
            key, separator, value = line.strip().partition("=")
            if not separator:
                continue

            seconds = None
            if key in {"out_time_ms", "out_time_us"}:
                try:
                    raw_value = float(value)
                    if raw_value >= 0:
                        seconds = raw_value / 1_000_000.0
                except ValueError:
                    continue
            elif key == "out_time":
                seconds = CompositionRenderer._parse_timestamp(value)

            if seconds is not None:
                ratio = min(1.0, max(0.0, seconds / total_duration))
                if ratio > last_progress:
                    last_progress = ratio
                    if progress_callback:
                        progress_callback(last_progress, "渲染中")
            elif key == "progress" and value.strip() == "end":
                last_progress = 1.0
                if progress_callback:
                    progress_callback(last_progress, "渲染完成")

        return last_progress

    @staticmethod
    def _parse_timestamp(value: str) -> float | None:
        try:
            hours, minutes, seconds = value.strip().split(":")
            parsed = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            return parsed if parsed >= 0 and math.isfinite(parsed) else None
        except (TypeError, ValueError):
            return None

    def _ffmpeg_supports_encoder(self, encoder: str) -> bool:
        try:
            result = subprocess.run(
                [self.ffmpeg_binary, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                shell=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False
        if result.returncode != 0:
            return False
        output = getattr(result, "stdout", "") or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return encoder in output.split()

    @staticmethod
    def _cuda_or_tensorrt_available() -> bool:
        try:
            info = DeviceManager.get_info()
            device_type = getattr(info.device_type, "value", info.device_type)
            return str(device_type).lower() in {"cuda", "tensorrt"}
        except Exception as exc:
            logger.debug(f"无法读取编码设备状态: {exc}")
            return False

    def _execute_ffmpeg(
        self,
        command: list[str],
        total_duration: float,
        progress_callback: Callable[[float, str], None] | None,
    ) -> tuple[float, list[str]]:
        stderr_tail: deque[str] = deque(maxlen=40)
        progress_queue: queue.Queue[object] = queue.Queue()

        try:
            # 降低 FFmpeg 进程优先级，避免 CPU 密集型编码抢占 API 服务
            if os.name == "posix":
                import signal
                preexec_fn = lambda: os.nice(10)
            else:
                preexec_fn = None
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                shell=False,
                preexec_fn=preexec_fn,
            )
        except FileNotFoundError as exc:
            raise CompositionError(
                f"FFmpeg 不可用，未找到可执行文件: {self.ffmpeg_binary}"
            ) from exc
        except OSError as exc:
            raise CompositionError(f"无法启动 FFmpeg: {exc}") from exc

        def pump(stream, destination, keep_tail: bool = False):
            try:
                if stream is not None:
                    readline = getattr(stream, "readline", None)
                    if callable(readline):
                        while True:
                            line = readline()
                            if line in ("", b""):
                                break
                            if keep_tail:
                                stderr_tail.append(self._line_text(line).rstrip())
                            else:
                                destination.put(line)
                    else:
                        for line in stream:
                            if keep_tail:
                                stderr_tail.append(self._line_text(line).rstrip())
                            else:
                                destination.put(line)
            finally:
                if not keep_tail:
                    destination.put(_STREAM_END)

        stdout_thread = threading.Thread(
            target=pump,
            args=(process.stdout, progress_queue),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=pump,
            args=(process.stderr, None, True),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        deadline = time.monotonic() + self.timeout_seconds
        progress_state = [0.0]

        def report_progress(value: float, message: str) -> None:
            if value <= progress_state[0]:
                return
            progress_state[0] = value
            if progress_callback:
                progress_callback(value, message)

        stdout_finished = False
        progress_lines: list[str] = []
        try:
            while not stdout_finished:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    item = progress_queue.get(timeout=min(remaining, 0.25))
                except queue.Empty:
                    continue
                if item is _STREAM_END:
                    stdout_finished = True
                    continue
                progress_lines.append(self._line_text(item))
                self.parse_progress(
                    [progress_lines[-1]],
                    total_duration,
                    report_progress,
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            try:
                return_code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError from exc
            stdout_thread.join(timeout=max(0.0, deadline - time.monotonic()))
            stderr_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        except TimeoutError as exc:
            self._stop_process(process)
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            detail = self._stderr_detail(stderr_tail)
            suffix = f": {detail}" if detail else ""
            raise CompositionError(
                f"FFmpeg 渲染超时（{self.timeout_seconds:g}秒）{suffix}"
            ) from exc

        if return_code != 0:
            detail = self._stderr_detail(stderr_tail)
            suffix = f": {detail}" if detail else ""
            raise CompositionError(
                f"FFmpeg 渲染失败（退出码 {return_code}）{suffix}"
            )

        progress = progress_state[0]
        if progress < 1.0:
            progress = max(
                progress,
                self.parse_progress(
                    progress_lines,
                    total_duration,
                    None,
                ),
            )
        return progress, list(stderr_tail)

    @staticmethod
    def _validate_segments(segments: list[dict]) -> list[tuple[float, float]]:
        if not isinstance(segments, list) or not segments:
            raise CompositionError("至少需要一个有效分段")

        normalized: list[tuple[float, float]] = []
        for index, segment in enumerate(segments, start=1):
            if not isinstance(segment, dict):
                raise CompositionError(f"第 {index} 个分段格式无效")
            try:
                start = CompositionRenderer._valid_time(
                    segment["edit_start_time"], "开始时间", index
                )
                end = CompositionRenderer._valid_time(
                    segment["edit_end_time"], "结束时间", index
                )
            except KeyError as exc:
                raise CompositionError(f"第 {index} 个分段缺少时间字段") from exc
            if end <= start:
                raise CompositionError(f"第 {index} 个分段结束时间必须大于开始时间")
            normalized.append((start, end))

        # 安全检查：消除相邻片段在源视频时间轴上的重叠
        # 按开始时间排序后，如果后一个片段的开始时间 < 前一个的结束时间，
        # 取中点作为分界线
        if len(normalized) > 1:
            # 排序后操作，确保级联修复使用的是已修正的值
            order = sorted(range(len(normalized)), key=lambda i: normalized[i][0])
            for pos in range(1, len(order)):
                prev_i = order[pos - 1]
                curr_i = order[pos]
                prev_start, prev_end = normalized[prev_i]
                curr_start, curr_end = normalized[curr_i]
                if curr_start < prev_end:
                    boundary = (prev_end + curr_start) / 2.0
                    normalized[prev_i] = (prev_start, boundary)
                    normalized[curr_i] = (boundary, curr_end)

        return normalized

    @staticmethod
    def _valid_time(value, label: str, index: int) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise CompositionError(f"第 {index} 个分段的{label}无效")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise CompositionError(f"第 {index} 个分段的{label}无效")
        return value

    @staticmethod
    def _line_text(value) -> str:
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return str(value)

    @staticmethod
    def _stderr_detail(lines: Iterable[str]) -> str:
        detail = "\n".join(line for line in lines if line).strip()
        return detail[-800:]

    @staticmethod
    def _stop_process(process) -> None:
        try:
            process.terminate()
        except (AttributeError, OSError):
            pass
        try:
            process.wait(timeout=1.0)
        except (AttributeError, OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except (AttributeError, OSError):
                pass
