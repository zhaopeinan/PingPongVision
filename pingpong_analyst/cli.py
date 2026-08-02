#!/usr/bin/env python3
"""
PingPong AI Analyst CLI

用法:
    python -m pingpong_analyst.cli analyze input.mp4
    python -m pingpong_analyst.cli analyze input.mp4 --min-boards 6 --max-frames 1000
    python -m pingpong_analyst.cli clip input.mp4 --output-dir ./clips
    python -m pingpong_analyst.cli run input.mp4  # 分析+剪辑一键执行
"""
import argparse
import sys
from pathlib import Path

from loguru import logger


def cmd_analyze(args):
    """仅分析, 不剪辑"""
    from .core import VideoAnalyzer

    analyzer = VideoAnalyzer(config_path=args.config)
    segments = analyzer.analyze(args.video, max_frames=args.max_frames)

    print(f"\n{'='*60}")
    print(f"分析结果: 共 {len(segments)} 个有效回合")
    print(f"{'='*60}")
    for i, seg in enumerate(segments):
        print(
            f"  回合{i+1}: {seg.start_time:.1f}s - {seg.end_time:.1f}s "
            f"| 板数: {seg.board_count} | 时长: {seg.duration:.1f}s"
        )
        for hit in seg.hit_events:
            print(f"    - @{hit.timestamp:.1f}s 侧:{hit.hitter_side} 类型:{hit.hit_type}")


def cmd_clip(args):
    """仅剪辑 (需要先有分析结果, 此处简化为重新分析)"""
    from .core import VideoAnalyzer

    analyzer = VideoAnalyzer(config_path=args.config)
    segments = analyzer.analyze(args.video, max_frames=args.max_frames)
    outputs = analyzer.export_clips(args.video, segments)

    print(f"\n{'='*60}")
    print(f"剪辑完成: {len(outputs)} 个片段")
    print(f"{'='*60}")
    for p in outputs:
        print(f"  {p}")


def cmd_run(args):
    """分析+剪辑一键执行"""
    from .core import VideoAnalyzer

    analyzer = VideoAnalyzer(config_path=args.config)
    outputs = analyzer.analyze_and_clip(args.video, max_frames=args.max_frames)

    print(f"\n{'='*60}")
    print(f"完成: 生成 {len(outputs)} 个剪辑片段")
    print(f"{'='*60}")
    for p in outputs:
        print(f"  {p}")


def cmd_device(args):
    """检测设备信息"""
    from .utils.device_manager import DeviceManager
    info = DeviceManager.detect(args.mode)
    print(f"\n设备信息:")
    print(f"  类型: {info.device_type.value}")
    print(f"  名称: {info.device_name}")
    print(f"  PyTorch设备: {info.torch_device}")
    print(f"  显存: {info.total_memory_mb} MB")
    print(f"  支持半精度: {info.supports_half}")


def main():
    parser = argparse.ArgumentParser(
        description="PingPong AI Analyst - 乒乓球智能分析与自动化剪辑系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="分析视频 (不剪辑)")
    p_analyze.add_argument("video", help="视频文件路径")
    p_analyze.add_argument("--config", default=None, help="配置文件路径")
    p_analyze.add_argument("--min-boards", type=int, default=None, help="最小板数阈值")
    p_analyze.add_argument("--max-frames", type=int, default=-1, help="最大处理帧数")
    p_analyze.set_defaults(func=cmd_analyze)

    # clip
    p_clip = subparsers.add_parser("clip", help="分析并剪辑视频")
    p_clip.add_argument("video", help="视频文件路径")
    p_clip.add_argument("--config", default=None, help="配置文件路径")
    p_clip.add_argument("--output-dir", default=None, help="输出目录")
    p_clip.add_argument("--max-frames", type=int, default=-1, help="最大处理帧数")
    p_clip.set_defaults(func=cmd_clip)

    # run
    p_run = subparsers.add_parser("run", help="分析+剪辑一键执行")
    p_run.add_argument("video", help="视频文件路径")
    p_run.add_argument("--config", default=None, help="配置文件路径")
    p_run.add_argument("--max-frames", type=int, default=-1, help="最大处理帧数")
    p_run.set_defaults(func=cmd_run)

    # device
    p_dev = subparsers.add_parser("device", help="检测设备信息")
    p_dev.add_argument("--mode", default="auto", help="设备模式: auto/cpu/mps/cuda/tensorrt")
    p_dev.set_defaults(func=cmd_device)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
