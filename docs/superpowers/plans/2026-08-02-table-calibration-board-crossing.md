# Table Calibration And Board Crossing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution with task-by-task checkpoints. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-video table calibration and replace turning-point board counts with hysteresis-based left/right table crossings, ending a scoring segment after a configurable period without a crossing.

**Architecture:** Store four table corners and optional net points in a per-video JSON calibration store. Inject the resulting `TableGeometry` into both `TrackingVisualizer` and `VideoAnalyzer`. Add a shared `BallCrossingCounter` that consumes normalized table x-coordinates, increments boards only on confirmed side changes, and resets after the configured timeout; keep internal segment/clip objects while removing the visible rally metric.

**Tech Stack:** Python 3.12, FastAPI, OpenCV, NumPy, PyTorch/TrackNet, vanilla JavaScript, HTML/CSS, pytest.

---

### Task 1: Update the timeout contract

**Files:**
- Modify: `docs/superpowers/specs/2026-08-02-table-calibration-board-crossing-design.md`
- Modify: `config.yaml:93-112`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add `analysis.rally.no_crossing_timeout_seconds: 2.0`.** Document the default and the request override range `0.5` through `10.0` seconds in the spec and config comments.
- [ ] **Step 2: Add a config assertion.** Extend the existing config test with:

```python
assert config.get("analysis", "rally", "no_crossing_timeout_seconds") == 2.0
```

- [ ] **Step 3: Run:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_config.py`. Expected: all config tests pass.
- [ ] **Step 4: Commit:** `git add config.yaml docs/superpowers/specs/2026-08-02-table-calibration-board-crossing-design.md tests/test_config.py && git commit -m "feat: configure board crossing timeout"`.

### Task 2: Add per-video table calibration persistence and API

**Files:**
- Create: `pingpong_analyst/core/table_calibration.py`
- Modify: `pingpong_analyst/core/table_geometry.py:7-61`
- Modify: `pingpong_analyst/api.py:40-115,350-395`
- Create: `tests/test_table_calibration.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing tests.** Test `TableCalibration(frame_index, corners, net_points)` round trips through JSON; reject non-four corners, non-finite/degenerate corners, and net point counts other than zero or two.
- [ ] **Step 2: Implement `TableCalibration` and `TableCalibrationStore`.** Use the same safe video-id validation and atomic temporary-file replacement as `TrackNetAnnotationStore`, rooted at `data/table_calibrations`.
- [ ] **Step 3: Add `TableGeometry.from_calibration(calibration)`.** Preserve the existing normalized target corners and derive the net line from `x=0.5`; retain optional net points as metadata.
- [ ] **Step 4: Add API routes:**

```text
GET    /api/videos/{video_id}/table-calibration
POST   /api/videos/{video_id}/table-calibration
DELETE /api/videos/{video_id}/table-calibration
```

POST must validate video id, frame index, image bounds, corner order/geometry, and optional net points. GET returns `calibration: null` when absent; DELETE returns `deleted: true`.
- [ ] **Step 5: Add API tests.** Upload the existing synthetic video, save four corners plus two net points, assert GET, reject an out-of-bounds point with `400`, then assert DELETE.
- [ ] **Step 6: Run:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_table_calibration.py tests/test_table_geometry.py tests/test_api.py`. Expected: all selected tests pass.
- [ ] **Step 7: Commit:** `git add pingpong_analyst/core/table_calibration.py pingpong_analyst/core/table_geometry.py pingpong_analyst/api.py tests/test_table_calibration.py tests/test_api.py && git commit -m "feat: persist per-video table calibration"`.

### Task 3: Implement shared side-crossing board counting

**Files:**
- Create: `pingpong_analyst/core/ball_crossing.py`
- Modify: `pingpong_analyst/core/data_aligner.py:15-119`
- Modify: `pingpong_analyst/core/rally_detector.py:28-213`
- Modify: `config.yaml:93-112`
- Create: `tests/test_ball_crossing.py`
- Modify: `tests/test_rally_detector.py`

- [ ] **Step 1: Write failing tests for these sequences:**

```python
0.20 -> 0.80 -> 0.20  # boards 2
0.48 -> 0.52 -> 0.48  # boards 0 with hysteresis
0.20 -> 0.80, then timestamp + 2.01  # segment ends and resets
0.20 -> missing frames -> 0.80 before 2 seconds  # crossing counts
```

Assert the first observed side initializes state without incrementing boards.
- [ ] **Step 2: Implement `BallCrossingCounter`.** Use `left_enter=0.45`, `right_enter=0.55`, `current_side`, `board_count`, segment start timestamps, and `last_crossing_time`. Ignore missing/non-finite points, emit one event only after a confirmed opposite side, emit a timeout result after `timestamp - last_crossing_time > timeout_seconds`, and validate timeout in `[0.5, 10.0]`.
- [ ] **Step 3: Normalize uncalibrated x.** Add optional `frame_size` to `FrameData`/`DataAligner.add_frame`; calibrated frames use homography output, uncalibrated frames normalize raw pixel x by frame width. Update video and visualizer callers to pass `(width, height)`.
- [ ] **Step 4: Integrate with `RallyDetector`.** Use the counter’s board count instead of `len(_current_hit_events)`, end segments on the counter timeout, reset the counter after ending, and keep `hit_events` accepted but ignored for rally board counts.
- [ ] **Step 5: Update detector tests.** A left/right/left sequence must produce `board_count == 2`; a custom `1.0` second timeout must end at `1.01`; old hit events must not change the count.
- [ ] **Step 6: Run:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_ball_crossing.py tests/test_rally_detector.py tests/test_data_aligner.py`. Expected: all selected tests pass.
- [ ] **Step 7: Commit:** `git add pingpong_analyst/core/ball_crossing.py pingpong_analyst/core/data_aligner.py pingpong_analyst/core/rally_detector.py config.yaml tests/test_ball_crossing.py tests/test_rally_detector.py && git commit -m "feat: count boards by table side crossings"`.

### Task 4: Wire calibration, timeout, and crossings into analysis

**Files:**
- Modify: `pingpong_analyst/core/video_analyzer.py:30-205`
- Modify: `pingpong_analyst/tracking_visualizer.py:50-264`
- Modify: `pingpong_analyst/api.py:653-823,871-923`
- Modify: `pingpong_analyst/static/app.js:760-910`
- Modify: `tests/test_tracking_visualizer.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Inject geometry and timeout.** Let `VideoAnalyzer` and `TrackingVisualizer` receive optional `TableGeometry` and `no_crossing_timeout_seconds`; construct `DataAligner` and `RallyDetector` with them. Add an API helper to load a video calibration.
- [ ] **Step 2: Pass request values.** Add `no_crossing_timeout_seconds` query parameters to `/api/analyze/{video_id}` and `/api/ws/{video_id}`, clamp to `0.5–10.0`, and pass loaded geometry/timeout into analyzers. Extend `generate_mjpeg_stream` with the same optional values.
- [ ] **Step 3: Remove turning-point board updates.** Stop calling `match_hit_events` for rally board increments. Pass frame dimensions, call `RallyDetector.update`, and expose the counter’s current board count as `board_count`. Keep rally mode free of YOLO/MediaPipe.
- [ ] **Step 4: Add the timeout input.** Add a numeric control beside the minimum-board threshold with default `2.0`, min `0.5`, max `10`, step `0.1`; include it in offline requests and WebSocket URLs. Remove visible rally-count updates while retaining board count and ball speed.
- [ ] **Step 5: Add real-time assertions.** Use a fake TrackNet returning `[0.2, 0.8, 0.2]` and assert metadata board counts `0, 1, 2`; assert the configured timeout ends the internal segment.
- [ ] **Step 6: Run:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_tracking_visualizer.py tests/test_integration.py tests/test_api.py`. Expected: all selected tests pass.
- [ ] **Step 7: Commit:** `git add pingpong_analyst/core/video_analyzer.py pingpong_analyst/tracking_visualizer.py pingpong_analyst/api.py pingpong_analyst/static/app.js tests/test_tracking_visualizer.py tests/test_integration.py tests/test_api.py && git commit -m "feat: use calibrated crossings in rally analysis"`.

### Task 5: Add the interactive calibration workflow

**Files:**
- Modify: `pingpong_analyst/static/index.html:288-345`
- Modify: `pingpong_analyst/static/app.js:1-535`
- Modify: `pingpong_analyst/static/style.css:1400-1565`

- [ ] **Step 1: Add a `标定球台` entry and reuse the current canvas.** Add table/net modes, point-order instructions, clear points, and save calibration while preserving ball annotation/training mode.
- [ ] **Step 2: Implement point interaction.** Reuse zoom/pan source-coordinate conversion; table mode accepts four ordered points and draws numbered markers; net mode accepts two points and draws a line. Store source-video pixels, never CSS pixels.
- [ ] **Step 3: Load/save state.** On open, GET `/table-calibration`, select the saved frame, load it, and draw saved points. Save `frame_index`, `corners`, and optional `net_points`; display API validation errors in the modal.
- [ ] **Step 4: Run:** `node --check pingpong_analyst/static/app.js`. In the running app, calibrate, save, close, reopen, zoom, and pan; confirm all points remain aligned.
- [ ] **Step 5: Commit:** `git add pingpong_analyst/static/index.html pingpong_analyst/static/app.js pingpong_analyst/static/style.css && git commit -m "feat: add interactive table calibration"`.

### Task 6: Full verification and handoff

**Files:** all modified implementation and test files.

- [ ] **Step 1: Run:**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python -m compileall -q pingpong_analyst
node --check pingpong_analyst/static/app.js
git diff --check
```

Expected: all tests pass and all static checks exit 0.
- [ ] **Step 2: Verify service:** `curl -sS http://127.0.0.1:8077/api/health`; expected `{"status":"ok", ...}`.
- [ ] **Step 3: Verify the user workflow.** With the 1GB video, calibrate one frame, save, start tracking, confirm only board count and ball speed are visible, then run offline analysis with default `2.0s` and a custom timeout.
- [ ] **Step 4: Inspect final state:** `git status --short` and `git log -1 --oneline`.

