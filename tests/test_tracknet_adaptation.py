import cv2
import numpy as np

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
