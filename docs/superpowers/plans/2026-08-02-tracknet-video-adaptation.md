# TrackNet Video Adaptation Implementation Plan

Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a point-annotation workflow that fine-tunes TrackNetV3 on a selected video, stores accepted checkpoints as manually selectable model versions, and supports the workflow on local MPS/CPU and server CUDA.

**Architecture:** Keep `TrackNetTracker` inference-only and introduce shared temporal preprocessing, annotation storage, a fine-tuner, and a JSON-backed model registry. The API will pass an optional `model_id` into rally analysis and live tracking, while the frontend adds an annotation workspace and model selector without coupling training state to playback state.

**Tech Stack:** Existing FastAPI, vanilla HTML/CSS/JavaScript frontend, PyTorch, OpenCV/PyAV, JSON metadata, pytest.

---

## File Map

- Create `pingpong_analyst/models/tracknet_preprocess.py`: shared 27-channel background-plus-eight-frame preprocessing and sampled median background generation.
- Create `pingpong_analyst/core/tracknet_annotations.py`: annotation dataclass, validation, JSON persistence, and frame-label grouping.
- Create `pingpong_analyst/models/tracknet_finetune.py`: sparse-label temporal dataset, masked heatmap loss, fine-tuning loop, validation, and temporary checkpoints.
- Create `pingpong_analyst/models/tracknet_registry.py`: model version metadata, save/discard/list/resolve operations.
- Modify `pingpong_analyst/models/tracknet.py`: delegate preprocessing to the shared helper and accept a resolved selected checkpoint path.
- Modify `pingpong_analyst/core/video_analyzer.py`: accept an optional TrackNet model path and use it for rally analysis.
- Modify `pingpong_analyst/tracking_visualizer.py`: accept an optional TrackNet model path for live rally tracking.
- Modify `pingpong_analyst/api.py`: add annotation, training, registry, frame, and model-selection endpoints; pass `model_id` through rally endpoints.
- Modify `pingpong_analyst/static/index.html`: add annotation workspace, training controls, and manual TrackNet model selector.
- Modify `pingpong_analyst/static/app.js`: implement annotation interactions, training polling, model loading, and selected-model query parameters.
- Modify `pingpong_analyst/static/style.css`: style the annotation workspace, frame controls, labels, training state, and model selector.
- Create `tests/test_tracknet_adaptation.py`: unit tests for annotations, preprocessing, masked loss, training setup, and registry behavior.
- Modify `tests/test_api.py`: endpoint tests for annotations, frame retrieval, model list, training lifecycle, and selected model propagation.
- Modify `config.yaml`: add annotation storage, model registry, training defaults, and sample-count settings.
- Modify `docs/local-and-server.md`: document annotation training, model selection, and T4 training behavior.

### Task 1: Extract Shared TrackNet Preprocessing

**Files:**
- Create: `pingpong_analyst/models/tracknet_preprocess.py`
- Modify: `pingpong_analyst/models/tracknet.py:1-30,260-370`
- Modify: `config.yaml:45-67`
- Test: `tests/test_tracknet_adaptation.py`

- [ ] **Step 1: Write failing preprocessing tests**

Add tests that build three small BGR frames and assert:

```python
processor = TrackNetPreprocessor(width=8, height=6, temporal_frames=3, background_mode="concat")
batch = processor.prepare([frame10, frame20, frame30], background=None)
assert batch.shape == (1, 12, 6, 8)
assert np.allclose(batch[0, 0], 20 / 255.0)
assert np.allclose(batch[0, 3], 10 / 255.0)
assert np.allclose(batch[0, 9], 30 / 255.0)
```

Also assert `sample_background()` returns a resized BGR `uint8` median and handles a short video with fewer than the requested samples.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_tracknet_adaptation.py -k preprocessing
```

Expected: collection or import failure because `TrackNetPreprocessor` does not exist.

- [ ] **Step 3: Implement the shared helper**

Implement:

```python
class TrackNetPreprocessor:
    def __init__(self, width, height, temporal_frames=8, background_mode="concat"):
        ...

    def prepare(self, frames, background=None) -> np.ndarray:
        """Return contiguous float32 NCHW in official TrackNet order."""

    @staticmethod
    def sample_background(video_path, width, height, max_samples=180) -> np.ndarray:
        """Return an RGB-normalized source as resized BGR uint8 for later conversion."""
```

The helper must convert BGR to RGB, resize to `512x288` by configuration, prepend the sampled median background for `concat`, and pad the first temporal window by repeating its earliest frame.

Refactor `TrackNetTracker._prepare_temporal_input()` and `prepare_background()` to call this helper. Preserve CV fallback behavior and the existing public tracker methods.

- [ ] **Step 4: Run preprocessing and existing model tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_tracknet_adaptation.py tests/test_models.py
```

Expected: all focused tests pass and existing temporal-input tests remain green.

- [ ] **Step 5: Commit the preprocessing slice**

```bash
git add pingpong_analyst/models/tracknet_preprocess.py pingpong_analyst/models/tracknet.py config.yaml tests/test_tracknet_adaptation.py
git commit -m "refactor: share TrackNet temporal preprocessing"
```

### Task 2: Add Annotation Storage and Sparse Temporal Dataset

**Files:**
- Create: `pingpong_analyst/core/tracknet_annotations.py`
- Modify: `config.yaml:80-110`
- Test: `tests/test_tracknet_adaptation.py`

- [ ] **Step 1: Write failing annotation and dataset tests**

Cover these cases:

```python
store.save("video-a", [BallAnnotation(frame_index=8, label="ball", x=40, y=20)])
loaded = store.load("video-a")
assert loaded[0].frame_index == 8

sample = dataset[0]
inputs, targets, target_mask, metadata = sample
assert inputs.shape == (27, 288, 512)
assert targets.shape == (8, 288, 512)
assert target_mask.shape == (8,)
assert target_mask.sum() == 1
assert targets[-1].max() == 1
```

Test that `absent` produces an all-zero target and that an invalid label, negative coordinate, or out-of-range frame raises `ValueError`.

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_tracknet_adaptation.py -k "annotation or dataset"
```

Expected: failure because the annotation types and dataset are not present.

- [ ] **Step 3: Implement annotation persistence**

Create:

```python
@dataclass
class BallAnnotation:
    frame_index: int
    label: Literal["ball", "absent"]
    x: float | None = None
    y: float | None = None

class TrackNetAnnotationStore:
    def save(self, video_id: str, annotations: list[BallAnnotation]) -> None:
        """Validate and atomically persist annotations for one video."""

    def load(self, video_id: str) -> list[BallAnnotation]:
        """Load annotations for one video, returning an empty list when absent."""

    def delete(self, video_id: str) -> None:
        """Delete the annotation file for one video when it exists."""
```

Use one JSON file per video under the configured annotation directory. Validate frame indices against the probed video metadata at API boundaries, and use atomic replacement through a temporary file when saving.

- [ ] **Step 4: Implement the sparse temporal dataset**

Create `TrackNetWindowDataset(video_path, annotations, background, width=512, height=288, temporal_frames=8)`.

For each annotation at frame `t`, decode the eight-frame range `[t-7, t]`, pad the beginning, build the shared 27-channel input, create an eight-frame heatmap target, and set only index `7` in `target_mask`. Use a Gaussian disk centered at the annotation's scaled point. Use an all-zero heatmap for `absent`.

Group annotations by time gaps before splitting 80/20 into train and validation so adjacent frames remain in the same split. Expose `split_counts` for metadata and validation tests.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_tracknet_adaptation.py -k "annotation or dataset"
```

Then commit:

```bash
git add pingpong_analyst/core/tracknet_annotations.py tests/test_tracknet_adaptation.py config.yaml
git commit -m "feat: add TrackNet ball annotations and temporal dataset"
```

### Task 3: Implement Fine-Tuning and Model Registry

**Files:**
- Create: `pingpong_analyst/models/tracknet_finetune.py`
- Create: `pingpong_analyst/models/tracknet_registry.py`
- Modify: `pingpong_analyst/models/tracknet.py:160-245`
- Modify: `pingpong_analyst/core/video_analyzer.py:25-75`
- Modify: `pingpong_analyst/tracking_visualizer.py:45-145`
- Test: `tests/test_tracknet_adaptation.py`

- [ ] **Step 1: Write failing registry and masked-loss tests**

Test that registry operations are isolated:

```python
registry = TrackNetModelRegistry(tmp_path, base_path)
run = registry.create_temp_run({"annotation_count": 4})
assert registry.list_models()[0].model_id == "base"
registry.save_run(run.run_id, checkpoint_path)
assert any(item.model_id == run.run_id for item in registry.list_models())
registry.discard_run("other-temp-run")
```

Test masked loss with a target containing only index 7 and assert indices 0-6 do not affect the loss. Test that CUDA/MPS selection follows `DeviceManager` and that the fine-tuner refuses fewer than 12 positive labels.

- [ ] **Step 2: Run focused tests and verify failure**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_tracknet_adaptation.py -k "registry or loss or finetune"
```

Expected: failure because the registry and fine-tuner do not exist.

- [ ] **Step 3: Implement the model registry**

Implement `TrackNetModelRegistry` with:

```python
list_models() -> list[TrackNetModelRecord]
resolve(model_id: str) -> Path
create_temp_run(metadata: dict) -> TrackNetRun
save_run(run_id: str, checkpoint_path: Path, metadata: dict) -> TrackNetModelRecord
discard_run(run_id: str) -> None
disable(model_id: str) -> None
```

Use `models/runs/index.json` plus one directory per run. The `base` record points to the configured immutable `models/TrackNet_best.pt`. Resolve only paths under the project model root, and reject unknown or disabled IDs. Save only the best `.pt` and `metadata.json`.

- [ ] **Step 4: Implement fine-tuning**

Implement `TrackNetFineTuner.train()` with this flow:

```python
result = tuner.train(
    base_model_path=base_path,
    video_path=video_path,
    annotations=annotations,
    output_dir=temp_run_dir,
    device_info=device_info,
    progress_callback=callback,
    cancel_event=cancel_event,
)
```

Load the checkpoint using the existing TrackNet state-dict compatibility path, freeze early blocks, train the later feature block and predictor with `Adam(lr=1e-4)`, run 5-10 configured epochs, compute masked heatmap loss, and record validation loss, positive detection rate, mean pixel error, device, and annotation counts. Stop promptly when `cancel_event.is_set()` and never promote a temporary checkpoint automatically.

- [ ] **Step 5: Inject selected checkpoints into analysis**

Add optional `tracknet_model_path: str | None = None` to `VideoAnalyzer` and `TrackingVisualizer`. In `_load_models_stage1()` and `load_models()`, copy the TrackNet config before overriding `weights` with the resolved path. Keep the existing default config behavior when no path is supplied.

- [ ] **Step 6: Run focused tests and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_tracknet_adaptation.py tests/test_models.py
```

```bash
git add pingpong_analyst/models/tracknet_finetune.py pingpong_analyst/models/tracknet_registry.py pingpong_analyst/models/tracknet.py pingpong_analyst/core/video_analyzer.py pingpong_analyst/tracking_visualizer.py tests/test_tracknet_adaptation.py
git commit -m "feat: add TrackNet fine-tuning and model registry"
```

### Task 4: Add API Lifecycle and Manual Model Selection

**Files:**
- Modify: `pingpong_analyst/api.py:30-45,252-440,444-622`
- Modify: `tests/test_api.py`
- Modify: `config.yaml`

- [ ] **Step 1: Write failing API tests**

Add tests for:

```python
POST /api/videos/{video_id}/tracknet/annotations
GET  /api/videos/{video_id}/tracknet/annotations
GET  /api/videos/{video_id}/frame?frame_index=10
GET  /api/tracknet/models
POST /api/videos/{video_id}/tracknet/train
GET  /api/tracknet/train/{job_id}
```

Assert that invalid coordinates return `400`, frame retrieval returns `image/jpeg`, a training request returns a job ID, and `model_id=base` is passed to both `/api/analyze/{video_id}` and `/api/ws/{video_id}` without changing the global config.

- [ ] **Step 2: Run API tests and verify failure**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_api.py -k "tracknet or frame or model"
```

Expected: 404 responses because the new routes are not defined.

- [ ] **Step 3: Implement annotation and frame endpoints**

Add JSON endpoints backed by `TrackNetAnnotationStore` and a frame endpoint using PyAV. The frame endpoint returns the requested frame as JPEG plus frame metadata in response headers or a JSON envelope agreed with the frontend. Reject negative and out-of-range indices.

- [ ] **Step 4: Implement training job state**

Add `_training_jobs`, `_training_lock`, and `_run_tracknet_training_task()`. Store `queued`, `running`, `completed`, `failed`, `cancelled`, and `discarded` states. Use the existing process-local GPU serialization lock and keep the job ID response immediate. A save endpoint promotes the temporary run through the registry; discard removes only that run.

- [ ] **Step 5: Add model listing and selection propagation**

Add `GET /api/tracknet/models`. Resolve and validate `model_id` before starting an analysis. Add `model_id` query parameters to `/api/analyze/{video_id}`, `/api/stream/{video_id}`, and `/api/ws/{video_id}`. Pass the resolved path into `VideoAnalyzer` or `TrackingVisualizer`. Unknown IDs return `404`; disabled IDs return `409`.

- [ ] **Step 6: Run API tests and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_api.py
```

```bash
git add pingpong_analyst/api.py tests/test_api.py config.yaml
git commit -m "feat: expose TrackNet annotation and model APIs"
```

### Task 5: Build Annotation Workspace and Model Selector

**Files:**
- Modify: `pingpong_analyst/static/index.html`
- Modify: `pingpong_analyst/static/app.js`
- Modify: `pingpong_analyst/static/style.css`

- [ ] **Step 1: Add the workspace markup**

Add a `tracknetAnnotationModal` containing an image/canvas overlay, frame range input, frame step controls, `ball` and `absent` label buttons, annotation count, save button, train button, progress area, validation summary, save/discard controls, and a model selector. Use stable canvas dimensions and keep the existing playback modal behavior unchanged.

- [ ] **Step 2: Add annotation interaction state**

Extend `state` with `selectedModelId`, `annotationFrame`, `tracknetAnnotations`, and `trainingJobId`. Add functions:

```javascript
openTracknetAnnotationModal()
loadAnnotationFrame(frameIndex)
handleBallCanvasClick(event)
setAbsentAnnotation()
saveTracknetAnnotations()
startTracknetTraining()
pollTracknetTraining(jobId)
loadTracknetModels()
```

Convert canvas coordinates back to original video coordinates before sending them to the API. Prevent duplicate labels for the same frame by replacing the existing frame annotation.

- [ ] **Step 3: Add model selection to analysis requests**

Append `model_id=${encodeURIComponent(state.selectedModelId)}` to rally analysis, MJPEG/WS tracking, and any future rally request. Do not append a model ID to action analysis. Refresh the model list after a saved training run and leave the selected model unchanged until the user changes it.

- [ ] **Step 4: Add frontend state styling and error messages**

Style positive and absent markers differently, show unsaved state, disable training until the minimum label count is met, show train/save/discard states, and report unknown-model or failed-training errors without hiding the current video selection.

- [ ] **Step 5: Run browser smoke checks and commit**

Start the server and verify:

```text
Upload/select video -> open ball annotation -> load a frame -> click ball -> mark absent frame -> save labels -> list models -> select a model -> start rally tracking
```

Then run the existing test suite and commit:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
git add pingpong_analyst/static/index.html pingpong_analyst/static/app.js pingpong_analyst/static/style.css
git commit -m "feat: add TrackNet annotation workspace"
```

### Task 6: End-to-End Verification and Documentation

**Files:**
- Modify: `docs/local-and-server.md`
- Modify: `README.md` if created during repository setup
- Test: `tests/test_tracknet_adaptation.py`, `tests/test_api.py`

- [ ] **Step 1: Add end-to-end test coverage**

Use a short generated video and a synthetic point annotation to exercise annotation persistence, dataset construction, temporary fine-tuning, registry save, API model listing, and inference with the selected checkpoint. Keep the real 1 GB video out of automated tests.

- [ ] **Step 2: Run local verification**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python -m compileall -q pingpong_analyst
python -m pingpong_analyst.api --host 127.0.0.1 --port 8077
```

Verify `/api/health`, `/api/device`, model listing, and one short TrackNet inference on `data/uploads/0c42afad_test.mp4` using MPS.

- [ ] **Step 3: Document local and T4 operations**

Document annotation storage, minimum labels, training commands, model directories, manual model selection, MPS/CPU behavior, T4 CUDA training, and the fact that TensorRT export is an optional post-training inference step.

- [ ] **Step 4: Run final review and commit**

```bash
git diff --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
git status --short
git commit -m "docs: document TrackNet adaptation workflow"
```

The final report must state the exact test result, selected device, model backend, and any remaining accuracy limitations.
