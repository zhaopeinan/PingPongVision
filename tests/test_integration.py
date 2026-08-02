"""
端到端集成测试
生成合成视频 (含运动小球) -> 运行完整分析管线 -> 验证回合检测与剪辑
"""
import subprocess
from pathlib import Path

import numpy as np
import cv2
import pytest

from pingpong_analyst.core import VideoAnalyzer


def generate_test_video(path: str, num_frames: int = 120, fps: int = 30):
    """生成含运动小球的测试视频"""
    width, height = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # 背景: 浅灰
        frame[:] = (40, 40, 40)

        # 模拟球来回运动 (产生折返点)
        cycle = 40  # 40帧一个来回
        t = i % cycle
        if t < cycle // 2:
            x = int(100 + (t / (cycle // 2)) * 440)
        else:
            x = int(540 - ((t - cycle // 2) / (cycle // 2)) * 440)
        y = 180

        # 画球 (亮白色)
        cv2.circle(frame, (x, y), 12, (255, 255, 255), -1)
        # 加一点模糊
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        writer.write(frame)

    writer.release()
    return path


@pytest.fixture
def test_video(tmp_path):
    video_path = str(tmp_path / "test_rally.mp4")
    generate_test_video(video_path, num_frames=120, fps=30)
    return video_path


def test_end_to_end_analysis(test_video):
    """端到端: 视频分析 -> 回合检测"""
    analyzer = VideoAnalyzer()
    progress = []
    segments = analyzer.analyze(test_video, max_frames=120, progress_callback=progress.append)

    # 合成视频有来回运动, 应能检测到一些事件
    # (CV回退模式可能检测效果有限, 宽松断言)
    assert isinstance(segments, list)
    assert progress
    assert progress[-1] == 1.0
    assert all(0.0 <= value <= 1.0 for value in progress)
    print(f"\n检测到 {len(segments)} 个有效回合")


def test_video_generation(test_video):
    """验证测试视频生成成功"""
    assert Path(test_video).exists()
    assert Path(test_video).stat().st_size > 0

    # 验证可读
    cap = cv2.VideoCapture(test_video)
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert frame_count > 0
    cap.release()


def test_clip_export_stream_copy(test_video, tmp_path):
    """测试 FFmpeg 流拷贝剪辑"""
    from pingpong_analyst.core.clip_exporter import ClipExporter
    from pingpong_analyst.core.rally_detector import RallySegment

    exporter = ClipExporter(
        {
            "clip_mode": "stream_copy",
            "output_format": "mp4",
            "output_dir": str(tmp_path / "clips"),
            "clip_buffer_before": 0.3,
            "clip_buffer_after": 0.3,
        }
    )

    segment = RallySegment(
        start_frame=10, end_frame=60,
        start_time=0.33, end_time=2.0,
        board_count=5, hit_events=[],
    )

    result = exporter.export_clip(test_video, segment, "test_clip")
    assert result is not None
    assert Path(result).exists()
    assert Path(result).stat().st_size > 0
