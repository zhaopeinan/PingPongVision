# PingPong Full Functionality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ping-pong video pipeline reliable on macOS CPU and NVIDIA T4, with correct model asset resolution, isolated analysis runs, temporal ball tracking, per-player pose analysis, and geometry-aware hit detection.

**Architecture:** Keep the existing FastAPI and model-wrapper structure. Add project-root path resolution at the configuration boundary, make each analysis run reset/isolate its state, feed temporal frame windows to real TrackNet backends while retaining the CV fallback, and attach MediaPipe-derived angles to the YOLO player record that produced them. Add optional four-corner table calibration so trajectory decisions can use rectified table coordinates; preserve an automatic image-coordinate fallback when calibration is not configured.

**Tech Stack:** Python 3.12, NumPy, OpenCV, PyAV, Ultralytics YOLO, MediaPipe, PyTorch/TensorRT, FastAPI, pytest.

---

### Task 1: Resolve model assets consistently

**Files:**
- Modify: `pingpong_analyst/utils/config.py`
- Modify: `pingpong_analyst/models/yolo_pose.py`
- Modify: `pingpong_analyst/models/tracknet.py`
- Modify: `config.yaml`
- Test: `tests/test_config.py`

- [ ] Add a project-root-aware `resolve_path()` helper that accepts an absolute path, a path relative to the project root, and the legacy `models/<basename>` path. The helper must return an existing root-level fallback such as `yolo11s-pose.pt` when `models/yolo11s-pose.pt` is absent.
- [ ] Resolve YOLO, TrackNet, and TensorRT paths before loading and log the selected path.
- [ ] Keep `models/...` as the server convention while making the checked-in root-level YOLO file usable locally.
- [ ] Add tests for absolute paths, normal project-relative paths, and the root-level fallback.
- [ ] Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_config.py`.

### Task 2: Isolate analysis runs and per-request thresholds

**Files:**
- Modify: `pingpong_analyst/core/rally_detector.py`
- Modify: `pingpong_analyst/core/video_analyzer.py`
- Modify: `pingpong_analyst/api.py`
- Test: `tests/test_rally_detector.py`

- [ ] Add `RallyDetector.reset()` that clears state, current events, and accumulated segments.
- [ ] Reset the aligner and rally detector at the start of every `VideoAnalyzer.analyze()` call and make `flush()` leave the detector idle.
- [ ] Stop sharing one mutable `VideoAnalyzer` between background tasks. Construct one analyzer per task and serialize GPU analysis with a process-local lock so concurrent requests cannot mix state or exhaust the T4.
- [ ] Always apply the request's `min_boards`, including the default value, without leaking a previous request's threshold.
- [ ] Add a regression test that runs the detector twice and confirms the second run has no segments or hit events from the first.
- [ ] Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_rally_detector.py`.

### Task 3: Make TrackNet temporal instead of single-frame

**Files:**
- Modify: `pingpong_analyst/models/tracknet.py`
- Modify: `config.yaml`
- Test: `tests/test_models.py`

- [ ] Add `temporal_frames` configuration with default `3`, maintain a bounded frame deque, and pad the first few frames by repeating the earliest frame.
- [ ] Build model inputs as `[1, temporal_frames * 3, H, W]` for PyTorch and TensorRT backends, while keeping CV fallback single-frame.
- [ ] Derive TensorRT input/output shapes from the engine when available instead of assuming three input channels.
- [ ] Normalize model output shapes consistently before peak extraction.
- [ ] Preserve the fallback detector and reset its temporal state on load/unload.
- [ ] Add a test using a fake temporal model that asserts three RGB frames become nine input channels.
- [ ] Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_models.py` once OpenCV dependencies are installed; run static checks before that.

### Task 4: Associate pose angles with both selected players

**Files:**
- Modify: `pingpong_analyst/models/mediapipe_pose.py`
- Modify: `pingpong_analyst/core/video_analyzer.py`
- Modify: `pingpong_analyst/tracking_visualizer.py`
- Modify: `pingpong_analyst/core/data_aligner.py`
- Test: `tests/test_models.py`
- Test: `tests/test_data_aligner.py`

- [ ] Add crop-based per-person MediaPipe inference using YOLO bounding boxes. Each player receives `pose_landmarks` and `arm_angles` on their own `persons` record.
- [ ] Keep a display-level angle summary for the current UI while using the matched player's angles for hit classification.
- [ ] Ensure person filtering is applied before pose inference, so selected-player mode does not mix angles from an unselected player.
- [ ] Add tests for two mocked player crops and for DataAligner using the matched player's angles instead of a frame-global angle dictionary.
- [ ] Preserve a graceful no-MediaPipe path that leaves angle fields absent.

### Task 5: Add optional table calibration and geometry-aware events

**Files:**
- Create: `pingpong_analyst/core/table_geometry.py`
- Modify: `pingpong_analyst/core/data_aligner.py`
- Modify: `pingpong_analyst/core/video_analyzer.py`
- Modify: `pingpong_analyst/tracking_visualizer.py`
- Modify: `config.yaml`
- Test: `tests/test_data_aligner.py`
- Test: `tests/test_table_geometry.py`

- [ ] Implement a four-corner projective transform from image coordinates to normalized table coordinates, with validation for malformed corners and a deterministic image-coordinate fallback when calibration is absent.
- [ ] Add optional table corners and event margins to configuration; do not require calibration for existing videos.
- [ ] Detect trajectory reversals from smoothed, gap-tolerant rectified x coordinates and require a player-side margin, ball visibility, and a minimum temporal gap.
- [ ] Keep the event output compatible with the existing API while adding confidence and normalized position internally.
- [ ] Add synthetic geometry and trajectory tests covering perspective correction, missing frames, and false reversals near the table center.
- [ ] Run the focused data-alignment tests.

### Task 6: Verify both deployment paths

**Files:**
- Modify: `requirements-mac.txt` if dependency pins need correction
- Modify: `requirements-server.txt` if dependency pins need correction
- Modify: `start.sh` only if the server path/config contract changes
- Create or modify: `docs/local-and-server.md`

- [ ] Verify Python compilation and JavaScript syntax.
- [ ] Run the full test suite in an environment containing OpenCV, PyAV, FastAPI, MediaPipe, Ultralytics, and PyTorch.
- [ ] Verify macOS selects CPU and uses the local YOLO weight plus temporal fallback when TrackNet weights are absent.
- [ ] Verify T4 selects TensorRT when engines exist, otherwise CUDA/PyTorch, and uses the same normalized interfaces.
- [ ] Document required model assets, config paths, startup commands, and the optional table calibration format.

