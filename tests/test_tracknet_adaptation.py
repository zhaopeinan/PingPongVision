import cv2
import numpy as np
import pytest
from pathlib import Path

from pingpong_analyst.core.tracknet_annotations import (
    BallAnnotation,
    TrackNetAnnotationStore,
    TrackNetWindowDataset,
)
from pingpong_analyst.models.tracknet_preprocess import TrackNetPreprocessor
from pingpong_analyst.models.tracknet_finetune import masked_heatmap_loss
from pingpong_analyst.models.tracknet_registry import TrackNetModelRegistry
from pingpong_analyst.models.tracknet_finetune import TrackNetFineTuner
from pingpong_analyst.models.tracknet_v3 import TrackNetV3
from pingpong_analyst.utils.device_manager import DeviceInfo, DeviceType


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


def test_masked_heatmap_loss_ignores_unlabeled_outputs():
    predictions = np.zeros((1, 8, 2, 2), dtype=np.float32)
    targets = np.zeros_like(predictions)
    masks = np.zeros((1, 8), dtype=np.float32)
    masks[0, 7] = 1
    predictions[0, 0] = 100
    predictions[0, 7, 0, 0] = 1

    import torch

    loss = masked_heatmap_loss(torch.tensor(predictions), torch.tensor(targets), torch.tensor(masks))

    assert abs(float(loss) - 0.25) < 1e-6


def test_masked_heatmap_loss_weights_ball_peak():
    import torch

    predictions = torch.zeros((1, 8, 4, 4), dtype=torch.float32)
    targets = torch.zeros_like(predictions)
    masks = torch.zeros((1, 8), dtype=torch.float32)
    masks[0, 7] = 1
    targets[0, 7, 2, 2] = 1.0

    loss = masked_heatmap_loss(predictions, targets, masks, positive_weight=20.0)

    assert float(loss) > 0.0
    assert abs(float(loss) - 20.0 / 16.0) < 1e-6


def test_tracknet_registry_keeps_base_and_saves_run(tmp_path):
    base_path = tmp_path / "TrackNet_best.pt"
    base_path.write_bytes(b"base")
    registry = TrackNetModelRegistry(tmp_path / "models" / "runs", base_path)
    run = registry.create_temp_run({"annotation_count": 4})
    checkpoint = run.output_dir / "TrackNet_best.pt"
    checkpoint.write_bytes(b"fine-tuned")

    assert registry.list_models()[0].model_id == "base"
    saved = registry.save_run(run.run_id, checkpoint)

    assert saved.model_id == run.run_id
    assert registry.resolve(run.run_id).read_bytes() == b"fine-tuned"
    assert any(item.model_id == run.run_id for item in registry.list_models())
    registry.discard_run("other-temp-run")


def test_tracknet_finetuner_requires_minimum_positive_labels():
    from pingpong_analyst.models.tracknet_finetune import TrackNetFineTuner

    tuner = TrackNetFineTuner({"min_positive_annotations": 12})
    with pytest.raises(ValueError, match="至少需要"):
        tuner._validate_annotations(
            [BallAnnotation(frame_index=index, label="ball", x=2, y=2) for index in range(11)]
        )


def test_tracknet_finetuner_runs_one_small_cpu_epoch(tmp_path):
    video_path = tmp_path / "fine_tune.mp4"
    _make_annotation_video(video_path, frame_count=16)
    base_path = tmp_path / "base.pt"
    import torch

    torch.save(
        {
            "model": TrackNetV3(27, 8).state_dict(),
            "param_dict": {"seq_len": 8, "bg_mode": "concat"},
        },
        base_path,
    )
    annotations = [
        BallAnnotation(frame_index=index, label="ball", x=10 + index, y=18)
        for index in range(12)
    ]
    result = TrackNetFineTuner(
        {
            "input_width": 64,
            "input_height": 36,
            "temporal_frames": 8,
            "background_samples": 8,
            "epochs": 1,
            "batch_size": 2,
            "min_positive_annotations": 12,
        }
    ).train(
        base_model_path=base_path,
        video_path=video_path,
        annotations=annotations,
        output_dir=tmp_path / "run",
        device_info=DeviceInfo(DeviceType.CPU, "TestCPU", "cpu", 0, False),
    )

    assert result["status"] == "completed"
    assert Path(result["checkpoint_path"]).is_file()
    assert result["metrics"]["validation_loss"] >= 0
