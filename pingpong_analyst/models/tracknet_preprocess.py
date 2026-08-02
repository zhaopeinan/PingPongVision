"""Shared preprocessing for the official TrackNetV3 table-tennis format."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class TrackNetPreprocessor:
    """Build the background-plus-temporal-frame input expected by TrackNetV3."""

    def __init__(
        self,
        width: int,
        height: int,
        temporal_frames: int = 8,
        background_mode: str = "concat",
    ) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.temporal_frames = max(1, int(temporal_frames))
        self.background_mode = str(background_mode or "").lower()

    def prepare(
        self,
        frames: list[np.ndarray] | tuple[np.ndarray, ...],
        background: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a contiguous float32 NCHW batch in official TrackNet order.

        Frames and ``background`` are BGR uint8 images in the same convention as
        OpenCV. The returned tensor is RGB, normalized to ``[0, 1]``. For the
        public TrackNetV3 checkpoint the channel order is 3 background channels
        followed by eight RGB frame channels.
        """
        if not frames:
            raise ValueError("TrackNet preprocessing requires at least one frame")

        temporal = list(frames)[-self.temporal_frames :]
        while len(temporal) < self.temporal_frames:
            temporal.insert(0, temporal[0])

        resized_frames = [self._resize_rgb(frame) for frame in temporal]
        channels: list[np.ndarray] = []

        if self.background_mode == "concat":
            if background is None:
                background_rgb = np.median(
                    np.stack(resized_frames, axis=0), axis=0
                ).astype(np.uint8)
            else:
                background_rgb = self._resize_rgb(background)
            channels.append(np.transpose(background_rgb.astype(np.float32) / 255.0, (2, 0, 1)))

        channels.extend(
            np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1))
            for frame in resized_frames
        )
        return np.ascontiguousarray(np.concatenate(channels, axis=0)[np.newaxis, ...])

    def _resize_rgb(self, frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("TrackNet frames must be HxWx3 BGR arrays")
        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    @staticmethod
    def sample_background(
        video_path: str | Path,
        width: int,
        height: int,
        max_samples: int = 180,
    ) -> np.ndarray:
        """Return a resized BGR uint8 median background sampled from a video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频生成 TrackNet 背景: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_count = max(1, int(max_samples))
        sample_step = max(1, total_frames // sample_count) if total_frames else 1
        samples: list[np.ndarray] = []
        frame_index = 0
        try:
            while len(samples) < sample_count:
                success, image = cap.read()
                if not success:
                    break
                if frame_index % sample_step == 0:
                    samples.append(
                        cv2.resize(
                            image,
                            (max(1, int(width)), max(1, int(height))),
                            interpolation=cv2.INTER_AREA,
                        )
                    )
                frame_index += 1
        finally:
            cap.release()

        if not samples:
            raise ValueError(f"视频没有可用帧，无法生成 TrackNet 背景: {video_path}")
        return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)
