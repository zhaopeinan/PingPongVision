# TrackNet Video Adaptation Design

## Goal

Allow a user to select a video, mark table-tennis ball locations on selected frames,
fine-tune the existing TrackNetV3 model for the video scene, and optionally keep the
result as a reusable model version. Saved versions must be selectable manually for
future analyses and must never replace the base model automatically.

This feature is an adaptation workflow, not training TrackNet from scratch.

## Decisions

- Fine-tune the existing TrackNetV3 checkpoint.
- Keep the base `TrackNet_best.pt` immutable.
- Store every accepted training result as an independent model version.
- Do not automatically make a new version the default model.
- Let the user select a model version for each analysis or tracking session.
- Support local MPS/CPU training and server-side CUDA training on the NVIDIA T4.
- Use point annotations rather than bounding boxes.
- Support three annotation states: ball present, ball absent/not visible, and skipped.

## User Workflow

1. Select a video from the library.
2. Open the ball annotation workspace.
3. Move through the video frame by frame or drag the timeline.
4. Click the ball center to create a positive annotation.
5. Mark a frame as absent/not visible when no reliable ball location exists.
6. Edit, delete, or undo annotations before training.
7. Start a training job and watch its progress.
8. Review before/after validation metrics and sample predictions.
9. Save the result as a model version or discard the temporary result.
10. Select a model version manually before future rally analysis or live tracking.

The recommended dataset size is at least 30 positive and 10 negative annotations.
Training is blocked below 12 positive annotations and a warning is shown when the
recommended size is not met.

## Annotation Data

Annotations are stored separately from the source video. Each annotation contains:

```json
{
  "video_id": "0045c4ef",
  "frame_index": 1234,
  "label": "ball",
  "x": 842,
  "y": 391,
  "created_at": "..."
}
```

For a negative frame, `label` is `absent` and `x`/`y` are omitted. A skipped frame
is not written as a training label. The original video and base weights are never
modified by annotation.

## Training Dataset Construction

TrackNetV3 receives an eight-frame temporal window. For an annotation at frame `t`,
the training sample uses frames `t-7` through `t`, padding the beginning with the
first available frame when necessary.

The input uses the checkpoint's official `bg_mode=concat` format:

```text
3-channel median background + 8 RGB frames = 27 input channels
8 output heatmaps, one per temporal frame
```

The annotated frame is converted into a small Gaussian ball heatmap at the model
resolution. Only the annotated output frame contributes to the loss. The other
outputs are masked so that sparse annotation does not imply an incorrect negative
label for unlabeled frames. An `absent` annotation contributes an all-zero target
for its target frame.

Samples are split by temporal groups rather than randomly splitting adjacent frames.
This prevents nearly identical neighboring frames from appearing in both training
and validation sets. The same background construction and normalization used during
inference must be used during fine-tuning and validation.

## Fine-Tuning

The first implementation uses conservative transfer learning:

- Load the selected base model or saved model version.
- Freeze early TrackNet feature blocks.
- Fine-tune the later feature block and heatmap predictor.
- Use a low learning rate and a small default epoch count, initially 5-10 epochs.
- Select the best checkpoint by validation loss and ball-location metrics.
- Save only the best model weights and metadata by default; optimizer state is temporary.

Device selection follows the existing device manager:

```text
Apple Silicon macOS -> MPS
Other local machines -> CPU
T4 server -> CUDA
```

TensorRT is not used for training. It remains an inference backend for server
deployment after a model version has been validated and exported if needed.

## Model Registry

Each saved version has its own directory, for example:

```text
models/runs/run_20260802_001/
  TrackNet_best.pt
  metadata.json
```

Metadata includes:

- stable `model_id` and display name;
- base model ID and checkpoint hash;
- source video IDs;
- annotation counts by label;
- training device and configuration;
- validation loss and location metrics;
- creation time and status;
- whether the version is available for manual selection.

The registry exposes the immutable base model and saved versions together. A saved
version can be selected, disabled, or deleted without changing another version.
There is no automatic default replacement.

## API and Component Boundaries

Training is separated from frame inference:

- `TrackNetFineTuner`: dataset creation, training, validation, and checkpoint output.
- `TrackNetModelRegistry`: model metadata, paths, status, and selection validation.
- `TrackNetTracker`: inference only; accepts a selected model path.
- Annotation storage: persists video/frame/label records independently of model files.

The API will provide equivalent operations for:

- create/read/update/delete annotations;
- create and cancel a training job;
- query training progress and validation results;
- list, save, disable, and delete model versions;
- select a `model_id` for rally analysis or live tracking.

Long-running training must run outside the request handler. The API returns a job ID,
and the frontend polls job status. A failed or cancelled job must not alter the base
model or an existing saved version.

## Frontend States

The annotation workspace must expose explicit states:

- loading video metadata;
- annotating;
- unsaved annotations;
- training queued/running;
- validation available;
- save confirmation;
- discarded/failed/cancelled;
- model selected for analysis.

The analysis controls must show which model version is selected before starting a
job or tracking stream. If a selected version cannot be loaded, the UI reports the
failure and offers the base model rather than silently using an unknown checkpoint.

## Acceptance Criteria

- A user can annotate positive, absent, and skipped frames.
- An annotation set can be edited and persisted without modifying the source video.
- A valid annotation set starts a background fine-tuning job on local MPS/CPU.
- The same job code can run on server CUDA.
- Training progress and validation results are visible to the user.
- A user can save or discard the result.
- Saved versions appear in a model list and are not automatically selected.
- A manually selected version is used by both rally analysis and live tracking.
- Base and existing saved versions remain usable after a failed training job.
- A corrupt or unavailable selected version produces a clear error and safe base-model fallback.
- Tests cover annotation persistence, temporal sample construction, masked loss,
  device selection, checkpoint save/discard, registry selection, and inference model switching.

## Non-Goals for the First Version

- Training a new TrackNet architecture from scratch.
- Automatically labeling all frames without user review.
- Integrating `InpaintNet_best.pt` into rally output.
- Automatically replacing the global default model.
- Multi-user permissions or remote model sharing.
