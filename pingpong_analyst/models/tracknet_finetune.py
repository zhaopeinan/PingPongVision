"""Conservative TrackNetV3 fine-tuning for manually labelled videos."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..core.tracknet_annotations import BallAnnotation, TrackNetWindowDataset
from ..utils.device_manager import DeviceInfo, DeviceType
from .tracknet_v3 import TrackNetV3
from .tracknet_preprocess import TrackNetPreprocessor


def masked_heatmap_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    positive_weight: float = 20.0,
) -> torch.Tensor:
    """Compute foreground-weighted MSE for explicitly labelled output frames.

    TrackNet heatmaps are mostly zero pixels. Weighting the Gaussian ball peak
    keeps an all-background prediction from looking artificially excellent.
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"预测和目标形状不一致: {predictions.shape} != {targets.shape}")
    mask = target_mask.to(device=predictions.device, dtype=predictions.dtype)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape != predictions.shape[:2]:
        raise ValueError("target_mask 必须是 [batch, temporal_frames]")
    mask = mask[:, :, None, None]
    pixel_weight = 1.0 + max(0.0, float(positive_weight) - 1.0) * targets.clamp(0.0, 1.0)
    squared_error = (predictions - targets) ** 2 * pixel_weight * mask
    denominator = mask.sum() * predictions.shape[-1] * predictions.shape[-2]
    if denominator.item() == 0:
        return predictions.sum() * 0.0
    return squared_error.sum() / denominator


class TrackNetFineTuner:
    """Fine-tune late TrackNet blocks on a sparse annotation set."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})

    def _device(self, device_info: DeviceInfo) -> torch.device:
        requested = str(device_info.torch_device)
        if device_info.device_type in {DeviceType.CUDA, DeviceType.TENSORRT} and torch.cuda.is_available():
            return torch.device(requested)
        if device_info.device_type == DeviceType.MPS and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _load_model(checkpoint_path: str | Path) -> tuple[nn.Module, dict]:
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(checkpoint, nn.Module):
            return checkpoint, {}
        params = checkpoint.get("param_dict", {}) if isinstance(checkpoint, dict) else {}
        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
        if state_dict is None and isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        if isinstance(state_dict, nn.Module):
            return state_dict, params
        normalized = {
            key.removeprefix("module.").removeprefix("model."): value
            for key, value in state_dict.items()
        }
        input_weight = normalized.get("down_block_1.conv_1.conv.weight")
        predictor_weight = normalized.get("predictor.weight")
        if input_weight is None or predictor_weight is None:
            raise ValueError("TrackNet checkpoint 缺少网络结构权重")
        model = TrackNetV3(int(input_weight.shape[1]), int(predictor_weight.shape[0]))
        model.load_state_dict(normalized, strict=True)
        return model, params

    def _validate_annotations(self, annotations: list[BallAnnotation]) -> None:
        positive_count = sum(item.label == "ball" for item in annotations)
        minimum = int(self.config.get("min_positive_annotations", 12))
        if positive_count < minimum:
            raise ValueError(f"至少需要 {minimum} 个 ball 标注，当前只有 {positive_count} 个")

    def _notify(self, callback: Callable[[dict], None] | None, payload: dict) -> None:
        if callback:
            callback(payload)

    def train(
        self,
        base_model_path: str | Path,
        video_path: str | Path,
        annotations: list[BallAnnotation],
        output_dir: str | Path,
        device_info: DeviceInfo,
        progress_callback: Callable[[dict], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict:
        self._validate_annotations(annotations)
        cancel_event = cancel_event or threading.Event()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        device = self._device(device_info)
        width = int(self.config.get("input_width", 512))
        height = int(self.config.get("input_height", 288))
        temporal_frames = int(self.config.get("temporal_frames", 8))
        background = TrackNetPreprocessor.sample_background(
            video_path,
            width=width,
            height=height,
            max_samples=int(self.config.get("background_samples", 180)),
        )
        train_dataset = TrackNetWindowDataset(
            video_path,
            annotations,
            background,
            width=width,
            height=height,
            temporal_frames=temporal_frames,
            split="train",
            validation_fraction=float(self.config.get("validation_fraction", 0.2)),
            heatmap_sigma=float(self.config.get("heatmap_sigma", 2.0)),
        )
        validation_dataset = TrackNetWindowDataset(
            video_path,
            annotations,
            background,
            width=width,
            height=height,
            temporal_frames=temporal_frames,
            split="validation",
            validation_fraction=float(self.config.get("validation_fraction", 0.2)),
            heatmap_sigma=float(self.config.get("heatmap_sigma", 2.0)),
        )
        if len(validation_dataset) == 0:
            validation_dataset = train_dataset
        batch_size = max(1, int(self.config.get("batch_size", 1)))
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        model, checkpoint_params = self._load_model(base_model_path)
        model = model.to(device)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(("down_block_3", "bottleneck", "up_block", "predictor"))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(
            trainable,
            lr=float(self.config.get("learning_rate", 1e-4)),
        )
        epochs = max(1, int(self.config.get("epochs", 5)))
        positive_weight = float(self.config.get("heatmap_positive_weight", 20.0))
        best_loss = float("inf")
        best_metrics: dict = {}
        self._notify(progress_callback, {"status": "running", "epoch": 0, "epochs": epochs, "device": str(device)})

        for epoch in range(1, epochs + 1):
            if cancel_event.is_set():
                return {"status": "cancelled", "device": str(device)}
            model.train()
            train_losses: list[float] = []
            for inputs, targets, masks, _metadata in loader:
                if cancel_event.is_set():
                    return {"status": "cancelled", "device": str(device)}
                optimizer.zero_grad(set_to_none=True)
                predictions = model(inputs.float().to(device))
                loss = masked_heatmap_loss(
                    predictions,
                    targets.float().to(device),
                    masks.float().to(device),
                    positive_weight=positive_weight,
                )
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            metrics = self._evaluate(
                model,
                validation_loader,
                device,
                validation_dataset,
                positive_weight=positive_weight,
            )
            metrics.update({"train_loss": float(np.mean(train_losses)) if train_losses else 0.0, "epoch": epoch})
            self._notify(progress_callback, {"status": "running", "epoch": epoch, "epochs": epochs, **metrics})
            if metrics["validation_loss"] < best_loss:
                best_loss = metrics["validation_loss"]
                best_metrics = metrics
                torch.save(
                    {
                        "model": model.state_dict(),
                        "param_dict": {
                            "seq_len": temporal_frames,
                            "bg_mode": "concat",
                            "input_channels": int(next(model.parameters()).shape[1]) if next(model.parameters()).ndim == 4 else 27,
                            "output_channels": temporal_frames,
                        },
                    },
                    output_path / "TrackNet_best.pt",
                )

        result = {
            "status": "completed",
            "checkpoint_path": str((output_path / "TrackNet_best.pt").resolve()),
            "device": str(device),
            "device_type": device_info.device_type.value,
            "base_model_path": str(Path(base_model_path).resolve()),
            "annotation_count": len(annotations),
            "positive_count": sum(item.label == "ball" for item in annotations),
            "absent_count": sum(item.label == "absent" for item in annotations),
            "split_counts": train_dataset.split_counts,
            "epochs": epochs,
            "metrics": best_metrics,
            "checkpoint_params": checkpoint_params,
        }
        self._notify(progress_callback, result)
        return result

    @staticmethod
    def _evaluate(model, loader, device, dataset, positive_weight: float = 20.0) -> dict:
        model.eval()
        losses: list[float] = []
        positive_hits = 0
        positive_total = 0
        pixel_errors: list[float] = []
        with torch.no_grad():
            for inputs, targets, masks, metadata in loader:
                predictions = model(inputs.float().to(device))
                loss = masked_heatmap_loss(
                    predictions,
                    targets.float().to(device),
                    masks.float().to(device),
                    positive_weight=positive_weight,
                )
                losses.append(float(loss.cpu()))
                labels = metadata["label"]
                if isinstance(labels, str):
                    labels = [labels]
                for batch_index, label in enumerate(labels):
                    label_value = str(label)
                    if label_value != "ball":
                        continue
                    positive_total += 1
                    predicted = predictions[batch_index, -1].detach().cpu().numpy()
                    target = targets[batch_index, -1].numpy()
                    if float(predicted.max()) >= 0.5:
                        positive_hits += 1
                    pred_y, pred_x = np.unravel_index(int(np.argmax(predicted)), predicted.shape)
                    target_y, target_x = np.unravel_index(int(np.argmax(target)), target.shape)
                    pixel_errors.append(float(np.hypot(pred_x - target_x, pred_y - target_y)))
        return {
            "validation_loss": float(np.mean(losses)) if losses else 0.0,
            "positive_detection_rate": positive_hits / positive_total if positive_total else 0.0,
            "mean_pixel_error": float(np.mean(pixel_errors)) if pixel_errors else None,
        }
