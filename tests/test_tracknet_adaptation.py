import cv2
import numpy as np
import pytest

from pingpong_analyst.core.tracknet_annotations import (
    BallAnnotation,
    TrackNetAnnotationStore,
    TrackNetWindowDataset,
)
from pingpong_analyst.models.tracknet_preprocess import TrackNetPreprocessor


def test_tracknet_preprocessor_builds_concat_batch():
    frames = [np.full((6, 8, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
    processor = TrackNetPreprocessor(
        width=8, height=6, temporal_frames=3, background_mode="concat"
    )

    batch = processor.prepare(frames)

    assert batch.shape == (1, 12, 6, 8)
    assert batch.dtype == np.float32
    assert np.allclose(batch[0, 0], 20 / 255.0)
    assert np.allclose(batch[0, 3], 10 / 255.0)
    assert np.allclose(batch[0, 9], 30 / 255.0)


def test_tracknet_preprocessor_pads_early_window():
    frame = np.full((4, 5, 3), 42, dtype=np.uint8)
    processor = TrackNetPreprocessor(width=5, height=4, temporal_frames=3, background_mode="")

    batch = processor.prepare([frame])

    assert batch.shape == (1, 9, 4, 5)
    assert np.allclose(batch[0, 0], 42 / 255.0)
    assert np.allclose(batch[0, 3], 42 / 255.0)
    assert np.allclose(batch[0, 6], 42 / 255.0)


def test_tracknet_preprocessor_samples_short_video(tmp_path):
    video_path = str(tmp_path / "short.mp4")
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (8, 6),
    )
    for value in (10, 20, 30):
        writer.write(np.full((6, 8, 3), value, dtype=np.uint8))
    writer.release()

    background = TrackNetPreprocessor.sample_background(video_path, 4, 3, max_samples=180)

    assert background.shape == (3, 4, 3)
    assert background.dtype == np.uint8
    assert np.allclose(background, 20, atol=4)


def _make_annotation_video(path, frame_count=16):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 36)
    )
    for index in range(frame_count):
        frame = np.full((36, 64, 3), 30, dtype=np.uint8)
        cv2.circle(frame, (10 + index, 18), 2, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_tracknet_annotation_store_round_trips_and_deduplicates(tmp_path):
    store = TrackNetAnnotationStore(tmp_path / "annotations")
    store.save(
        "video-a",
        [
            BallAnnotation(frame_index=8, label="ball", x=40, y=20),
            BallAnnotation(frame_index=8, label="absent"),
        ],
    )

    loaded = store.load("video-a")

    assert loaded == [BallAnnotation(frame_index=8, label="absent")]
    store.delete("video-a")
    assert store.load("video-a") == []


def test_tracknet_annotation_validation():
    with pytest.raises(ValueError):
        BallAnnotation(frame_index=1, label="unknown")
    with pytest.raises(ValueError):
        BallAnnotation(frame_index=1, label="ball", x=-1, y=3)
    with pytest.raises(ValueError):
        BallAnnotation(frame_index=-1, label="absent")
    with pytest.raises(ValueError):
        BallAnnotation(frame_index=1, label="absent", x=2, y=3)


def test_tracknet_window_dataset_masks_unlabeled_outputs(tmp_path):
    video_path = tmp_path / "annotated.mp4"
    _make_annotation_video(video_path)
    annotations = [BallAnnotation(frame_index=8, label="ball", x=40, y=20)]
    background = np.full((288, 512, 3), 30, dtype=np.uint8)
    dataset = TrackNetWindowDataset(
        video_path,
        annotations,
        background,
        width=512,
        height=288,
        temporal_frames=8,
    )

    inputs, targets, target_mask, metadata = dataset[0]

    assert inputs.shape == (27, 288, 512)
    assert targets.shape == (8, 288, 512)
    assert target_mask.shape == (8,)
    assert target_mask.sum() == 1
    assert targets[-1].max() == 1
    assert targets[:-1].max() == 0
    assert metadata["frame_index"] == 8


def test_tracknet_window_dataset_absent_has_zero_target(tmp_path):
    video_path = tmp_path / "absent.mp4"
    _make_annotation_video(video_path)
    dataset = TrackNetWindowDataset(
        video_path,
        [BallAnnotation(frame_index=2, label="absent")],
        np.zeros((288, 512, 3), dtype=np.uint8),
    )

    _, targets, target_mask, _ = dataset[0]

    assert targets.max() == 0
    assert target_mask[-1] == 1
