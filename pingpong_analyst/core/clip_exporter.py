"""
自动剪辑导出器 (任务二)

使用 FFmpeg 流拷贝模式 (Stream Copy) 进行无损快速剪辑
- stream_copy: 不重编码, 速度极快, 按关键帧对齐
- reencode: 精确切割, 需重编码
"""
import subprocess
import os
from pathlib import Path
from typing import Optional

from loguru import logger

from .rally_detector import RallySegment


class ClipExporter:
    """视频片段导出器"""

    def __init__(self, config: dict, project_root: str = None):
        self.clip_mode = config.get("clip_mode", "stream_copy")
        self.output_format = config.get("output_format", "mp4")
        self.output_codec = config.get("output_codec", "libx264")
        output_dir = config.get("output_dir", "output/clips")
        self.buffer_before = config.get("clip_buffer_before", 0.5)
        self.buffer_after = config.get("clip_buffer_after", 0.5)

        if project_root:
            self.output_dir = Path(project_root) / output_dir
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ClipExporter 初始化: mode={self.clip_mode}, output={self.output_dir}")

    def export_clip(
        self,
        video_path: str,
        segment: RallySegment,
        clip_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        导出单个片段

        Args:
            video_path: 源视频路径
            segment: 回合片段
            clip_name: 输出文件名 (不含扩展名)
        Returns:
            输出文件路径, 失败返回 None
        """
        if clip_name is None:
            clip_name = f"clip_{segment.start_frame}_{segment.end_frame}_b{segment.board_count}"

        output_path = self.output_dir / f"{clip_name}.{self.output_format}"

        # 加缓冲时间
        start_time = max(0, segment.start_time - self.buffer_before)
        duration = (segment.end_time - segment.start_time) + self.buffer_before + self.buffer_after

        if self.clip_mode == "stream_copy":
            return self._export_stream_copy(video_path, str(output_path), start_time, duration)
        else:
            return self._export_reencode(video_path, str(output_path), start_time, duration)

    def _export_stream_copy(
        self, video_path: str, output_path: str, start_time: float, duration: float
    ) -> Optional[str]:
        """FFmpeg 流拷贝 (无损快速)"""
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_time:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c", "copy",        # 流拷贝, 不重编码
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            output_path,
        ]
        return self._run_ffmpeg(cmd, output_path, "stream_copy")

    def _export_reencode(
        self, video_path: str, output_path: str, start_time: float, duration: float
    ) -> Optional[str]:
        """FFmpeg 重编码 (精确切割)"""
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_time:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c:v", self.output_codec,
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_path,
        ]
        return self._run_ffmpeg(cmd, output_path, "reencode")

    @staticmethod
    def _run_ffmpeg(cmd: list[str], output_path: str, mode: str) -> Optional[str]:
        """执行FFmpeg命令"""
        logger.info(f"FFmpeg [{mode}]: {' '.join(cmd[:6])}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                logger.info(f"片段导出成功: {output_path}")
                return output_path
            else:
                logger.error(f"FFmpeg 失败 (code={result.returncode}): {result.stderr[-500:]}")
                return None
        except FileNotFoundError:
            logger.error("ffmpeg 未安装或不在PATH中")
            return None
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg 执行超时")
            return None

    def export_all(
        self, video_path: str, segments: list[RallySegment]
    ) -> list[str]:
        """批量导出所有有效片段"""
        outputs = []
        for i, segment in enumerate(segments):
            if not segment.is_valid:
                continue
            clip_name = f"rally_{i+1:03d}_b{segment.board_count}_{segment.start_time:.1f}s"
            path = self.export_clip(video_path, segment, clip_name)
            if path:
                outputs.append(path)

        logger.info(f"批量导出完成: {len(outputs)}/{len(segments)} 个片段")
        return outputs
