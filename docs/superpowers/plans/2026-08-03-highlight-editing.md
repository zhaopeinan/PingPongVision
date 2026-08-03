# 精彩分段成片编辑器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为同一个原始乒乓球视频提供可恢复的精彩分段编辑器，支持逐段预览、精确微调、选择排序、异步硬切渲染和 MP4 导出。

**Architecture:** 分析完成后由 `EditProjectStore` 将分段元数据持久化为 JSON 项目。独立编辑器页面通过项目 API 保存编排状态；`CompositionRenderer` 使用原视频和精确时间范围调用 FFmpeg 合并视频/音频，渲染状态写回项目文件，前端轮询并提供预览下载。

**Tech Stack:** Python 3.12、FastAPI、PyAV、FFmpeg、原生 HTML/CSS/JavaScript、pytest、Playwright CLI。

---

## 文件结构

### 新增文件

- `pingpong_analyst/core/edit_project_store.py`
  - 定义项目 JSON 的创建、读取、列表、分段更新、渲染状态更新和原子写入。
- `pingpong_analyst/core/composition_renderer.py`
  - 定义 FFmpeg 输入探测、过滤链构造、编码器选择、渲染进度解析和错误转换。
- `pingpong_analyst/static/editor.html`
  - 独立成片编辑器页面骨架。
- `pingpong_analyst/static/editor.js`
  - 编辑器加载、逐段浏览、起止时间、选中状态、排序、自动保存、渲染轮询和导出交互。
- `pingpong_analyst/static/editor.css`
  - 编辑器三步工作区、分段卡片、播放区、编排列表和窄屏布局。
- `tests/test_edit_project_store.py`
  - 项目持久化和校验测试。
- `tests/test_composition_renderer.py`
  - FFmpeg 命令构造、进度解析和失败场景测试。

### 修改文件

- `pingpong_analyst/api.py`
  - 注册项目目录和导出目录；分析完成时创建项目；新增项目、渲染、导出 API；提供编辑器页面路由。
- `pingpong_analyst/static/index.html`
  - 分析完成后的结果区增加“进入成片编辑”入口。
- `pingpong_analyst/static/app.js`
  - 保存 `project_id`，渲染按钮跳转 `/editor/<project_id>`，处理旧结果没有项目 ID 的兼容创建。
- `pingpong_analyst/static/style.css`
  - 仅补充分析结果入口的样式；编辑器专属样式放在 `editor.css`。
- `config.yaml`
  - 增加成片输出目录、编码器、preset 和 CRF 配置。
- `tests/test_api.py`
  - 补充分析结果项目 ID 和项目 API 的端到端接口测试。

## Task 1: 实现可恢复的项目存储

**Files:**
- Create: `pingpong_analyst/core/edit_project_store.py`
- Test: `tests/test_edit_project_store.py`

- [ ] **Step 1: Write failing store tests for creation, defaults, and reload**

```python
def test_create_project_derives_buffered_edit_ranges(tmp_path):
    store = EditProjectStore(tmp_path / "projects")
    project = store.create_from_analysis(
        project_id="project-001",
        video_id="video-001",
        source_filename="match.mp4",
        source_storage_name="match.mp4",
        source_duration=20.0,
        rallies=[{"index": 1, "start_time": 3.0, "end_time": 8.0, "board_count": 12}],
        clips=["rally_001.mp4"],
        buffer_before=0.5,
        buffer_after=0.5,
    )

    assert project["status"] == "draft"
    assert project["segments"][0]["edit_start_time"] == 2.5
    assert project["segments"][0]["edit_end_time"] == 8.5
    assert store.get("project-001")["source_storage_name"] == "match.mp4"


def test_project_update_is_reloaded_from_disk(tmp_path):
    store = EditProjectStore(tmp_path / "projects")
    store.create_from_analysis(**sample_analysis_args())
    updated = store.update_segments("project-001", [
        {"segment_id": "segment-001", "selected": False, "order": 1,
         "edit_start_time": 2.8, "edit_end_time": 7.9}
    ])

    assert updated["segments"][0]["selected"] is False
    assert EditProjectStore(tmp_path / "projects").get("project-001")["segments"][0]["edit_start_time"] == 2.8
```

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_edit_project_store.py`

Expected: FAIL because `EditProjectStore` and the project schema do not exist.

- [ ] **Step 2: Add the minimal project schema and public store methods**

Implement these methods with plain JSON-compatible dictionaries so the API can serialize them directly:

```python
class EditProjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
    def create_from_analysis(self, *, project_id, video_id, source_filename,
                             source_storage_name, source_duration, rallies,
                             clips, buffer_before, buffer_after) -> dict: pass
    def get(self, project_id: str) -> dict: pass
    def list(self, video_id: str | None = None,
             source_storage_name: str | None = None) -> list[dict]: pass
    def update_segments(self, project_id: str, segments: list[dict]) -> dict: pass
    def update_render(self, project_id: str, **fields) -> dict: pass
```

For each rally/clip pair, create `segment-001` style IDs, copy `analysis_index`, `board_count`, detected times, and set `selected=true`, `order=index`. Clamp `clip_start_time` and `clip_end_time` to `[0, source_duration]`, then initialize the edit times to those values. Raise `ProjectNotFoundError` for missing IDs and `ProjectValidationError` for invalid JSON or missing required fields.

Use an 8-character lowercase hex project ID supplied by the API. Write with `tempfile.NamedTemporaryFile(dir=root, delete=False)` followed by `os.replace`, and never accept `/`, `\\`, `..`, or an absolute project ID as a filename.

- [ ] **Step 3: Add validation and list tests**

Add tests for an empty selection, `edit_start_time >= edit_end_time`, values outside the clip range, duplicate/negative order, missing project, path traversal, and `list(video_id=...)`. Assert that invalid updates leave the original JSON unchanged.

- [ ] **Step 4: Run store tests and commit**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_edit_project_store.py`

Expected: all store tests pass.

```bash
git add pingpong_analyst/core/edit_project_store.py tests/test_edit_project_store.py
git commit -m "feat: add persistent edit project store"
```

## Task 2: Implement precise FFmpeg composition

**Files:**
- Create: `pingpong_analyst/core/composition_renderer.py`
- Create: `tests/test_composition_renderer.py`
- Modify: `config.yaml`

- [ ] **Step 1: Write failing tests for filter graphs and encoder configuration**

```python
def test_build_filter_graph_with_audio_concats_trimmed_segments():
    renderer = CompositionRenderer("ffmpeg", output_dir=Path("output/exports"))
    graph = renderer.build_filter_graph([
        {"edit_start_time": 1.25, "edit_end_time": 3.5},
        {"edit_start_time": 8.0, "edit_end_time": 10.0},
    ], has_audio=True)

    assert "trim=start=1.250:end=3.500" in graph.filter_complex
    assert "atrim=start=8.000:end=10.000" in graph.filter_complex
    assert "concat=n=2:v=1:a=1" in graph.filter_complex
    assert graph.maps == ["[vout]", "[aout]"]


def test_build_filter_graph_without_audio_maps_only_video():
    renderer = CompositionRenderer("ffmpeg", output_dir=Path("output/exports"))
    graph = renderer.build_filter_graph([
        {"edit_start_time": 0.0, "edit_end_time": 2.0},
    ], has_audio=False)

    assert "concat=n=1:v=1:a=0" in graph.filter_complex
    assert graph.maps == ["[vout]"]
```

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_composition_renderer.py`

Expected: FAIL because the renderer and filter graph types do not exist.

- [ ] **Step 2: Add explicit renderer interfaces and command construction**

Implement:

```python
class CompositionError(RuntimeError):
    pass

@dataclass(frozen=True)
class FilterGraph:
    filter_complex: str
    maps: list[str]

class CompositionRenderer:
    def __init__(self, ffmpeg_binary: str = "ffmpeg", output_dir: Path = Path("output/exports"),
                 encoder: str = "auto", preset: str = "fast", crf: int = 21): pass
    def build_filter_graph(self, segments: list[dict], has_audio: bool) -> FilterGraph: pass
    def build_command(self, source_path: Path, segments: list[dict], output_path: Path,
                      has_audio: bool, encoder: str) -> list[str]: pass
    def render(self, source_path: Path, segments: list[dict], output_path: Path,
               progress_callback: Callable[[float, str], None] | None = None) -> dict: pass
```

The filter graph must emit one normalized video chain and, when audio exists, one normalized audio chain per selected segment, then concatenate in the supplied order. `build_command` must contain `-filter_complex`, `-map [vout]`, `-movflags +faststart`, and either `-map [aout] -c:a aac` or no audio map. Choose `h264_nvenc` only when `encoder=auto`, the device is CUDA/TensorRT, and `ffmpeg -encoders` confirms it; otherwise use `libx264`. Put `-preset` and `-crf` only on `libx264`; use `-cq 23` for NVENC.

Detect audio with PyAV before building the command. Do not create a silent synthetic audio stream. Normalize all numeric times to three decimal places and reject invalid segments before launching FFmpeg.

- [ ] **Step 3: Add progress parsing and failure tests**

Test a fake `-progress pipe:1` stream containing `out_time_ms`, `progress=continue`, and `progress=end`; verify callbacks are monotonic and capped at 1.0. Test `FileNotFoundError`, timeout, non-zero return code, and missing output file produce `CompositionError` with a short stderr tail. Test the command includes source and output paths as separate argv entries, never a shell string.

- [ ] **Step 4: Add config defaults, run renderer tests, and commit**

Add this block to `config.yaml`:

```yaml
composition:
  output_dir: output/exports
  encoder: auto
  preset: fast
  crf: 21
```

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_composition_renderer.py`

Expected: all renderer tests pass.

```bash
git add config.yaml pingpong_analyst/core/composition_renderer.py tests/test_composition_renderer.py
git commit -m "feat: add precise highlight composition renderer"
```

## Task 3: Connect analysis results and project APIs

**Files:**
- Modify: `pingpong_analyst/api.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing API tests for project creation and update**

Add a fixture that patches `EDIT_PROJECT_DIR` and `COMPOSITION_OUTPUT_DIR` to `tmp_path`, registers a test video, and seeds `_task_results[video_id]` with two rallies and two clip names. Define these test helpers in `tests/test_api.py` before the lifecycle test:

```python
def upload_test_video(client, test_video):
    with open(test_video, "rb") as source:
        response = client.post(
            "/api/upload", files={"file": ("edit-source.mp4", source, "video/mp4")}
        )
    return response.json()["video_id"]


def seed_rally_result(task_id):
    api_module._task_results[task_id] = {
        "status": "completed",
        "mode": "rally",
        "total_rallies": 2,
        "rallies": [
            {"index": 1, "start_time": 1.0, "end_time": 3.0, "board_count": 8},
            {"index": 2, "start_time": 5.0, "end_time": 7.0, "board_count": 10},
        ],
        "clips": ["rally_001.mp4", "rally_002.mp4"],
    }


def test_edit_project_lifecycle(client, test_video):
    video_id = upload_test_video(client, test_video)
    seed_rally_result(video_id)
    response = client.post(f"/api/edit-projects/from-analysis/{video_id}")
    assert response.status_code == 200
    project_id = response.json()["project_id"]

    loaded = client.get(f"/api/edit-projects/{project_id}")
    assert loaded.status_code == 200
    assert len(loaded.json()["segments"]) == 2

    saved = client.patch(f"/api/edit-projects/{project_id}", json={
        "segments": [{"segment_id": "segment-001", "selected": False,
                       "order": 1, "edit_start_time": 1.0, "edit_end_time": 2.5},
                      {"segment_id": "segment-002", "selected": True,
                       "order": 2, "edit_start_time": 4.0, "edit_end_time": 6.0}]
    })
    assert saved.status_code == 200
    assert saved.json()["segments"][0]["selected"] is False
```

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_api.py -k edit_project`

Expected: FAIL because the routes and project constants do not exist.

- [ ] **Step 2: Add API constants, source resolution, and project creation helper**

Define `EDIT_PROJECT_DIR = Path("data/edit_projects")`, `COMPOSITION_OUTPUT_DIR = Path("output/exports")`, `_render_lock = threading.Lock()`, and instantiate `EditProjectStore` through a helper so tests can patch the directory. Add `_source_storage_name(video_id)` using `Path(_video_registry[video_id]["path"]).name` and `_resolve_project_source(project)` that matches the stored storage name against registered paths after `_scan_library()`.

Add `create_edit_project_from_analysis(task_id)` that reads the completed rally task, calls `EditProjectStore.create_from_analysis`, uses `get_config().get("video", "clip_buffer_before", default=0.5)` and the corresponding after value, and returns the project. If a project ID is already present in the task result, return it without creating another JSON file.

Also add `create_edit_project_from_segments(task_id, video_path, segments, outputs)` for the live analysis path. It resolves the registry entry whose key is `task_id`, converts the `RallySegment` objects into the same rally dictionaries returned by the API, and delegates to `EditProjectStore.create_from_analysis`.

- [ ] **Step 3: Add project, render-status, export, and editor-page routes**

Implement the endpoints from the approved spec:

```python
@app.get("/api/edit-projects")
async def list_edit_projects(video_id: str | None = None):
    pass

@app.get("/api/edit-projects/{project_id}")
async def get_edit_project(project_id: str):
    pass

@app.patch("/api/edit-projects/{project_id}")
async def patch_edit_project(project_id: str, payload: dict = Body()):
    pass

@app.post("/api/edit-projects/{project_id}/render")
async def render_edit_project(project_id: str, background_tasks: BackgroundTasks):
    pass

@app.get("/api/edit-projects/{project_id}/render")
async def get_edit_render(project_id: str):
    pass

@app.get("/api/edit-projects/{project_id}/export")
async def export_edit_project(project_id: str):
    pass

@app.get("/editor/{project_id}")
async def editor_page(project_id: str):
    pass
```

Use 404 for unknown projects/files, 409 for missing source, invalid render state, or render prerequisites, and 400 for malformed patch payloads. Export must use a resolved path under `COMPOSITION_OUTPUT_DIR` and return `FileResponse(str(output_path), media_type="video/mp4", filename=output_filename)` only for a completed project.

- [ ] **Step 4: Integrate project creation into `_run_analysis_task`**

Immediately after `outputs = analyzer.export_clips(video_path, segments)`, call `create_edit_project_from_segments(task_id, video_path, segments, outputs)`, update the completed `_task_results` payload with `project_id`, and keep the existing `rallies` and `clips` fields unchanged for old clients. Persist `source_filename` from the registry’s display name and `source_storage_name` from the physical path name.

- [ ] **Step 5: Add the background render task**

Add `_run_edit_render_task(project_id)` that acquires `_render_lock`, marks the project `rendering`, resolves the source, selects `selected=true` segments sorted by `order`, invokes `CompositionRenderer.render`, updates progress through `EditProjectStore.update_render`, and marks the project `completed` with `output_filename`/duration or `failed` with the error summary. A second render request while the lock or project state is active returns the current task without starting another process.

- [ ] **Step 6: Run API tests and commit**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_api.py -k "edit_project or analyze_existing"`

Expected: project lifecycle, project ID propagation, invalid requests, render status, and export tests pass.

```bash
git add pingpong_analyst/api.py tests/test_api.py
git commit -m "feat: expose persistent edit project APIs"
```

## Task 4: Build the independent editor page

**Files:**
- Create: `pingpong_analyst/static/editor.html`
- Create: `pingpong_analyst/static/editor.js`
- Create: `pingpong_analyst/static/editor.css`

- [ ] **Step 1: Create the editor HTML structure**

Build semantic sections with stable IDs:

```html
<main id="editorApp">
  <header class="editor-header">
    <a href="/" class="editor-back">返回分析</a>
    <div><h1>精彩分段成片</h1><span id="projectSource"></span></div>
    <div id="editorSaveStatus">读取项目中</div>
  </header>
  <nav class="editor-steps" aria-label="成片流程">
    <button data-stage="browse">浏览分段</button>
    <button data-stage="arrange">编排成片</button>
    <button data-stage="render">渲染导出</button>
  </nav>
  <section class="editor-workspace">
    <div class="editor-preview">
      <video id="segmentPlayer" controls preload="metadata"></video>
      <div id="segmentMeta"></div>
      <div class="trim-controls" id="trimControls"></div>
      <div class="editor-actions" id="previewActions"></div>
    </div>
    <aside class="segment-panel">
      <div id="segmentSummary"></div>
      <div id="segmentList"></div>
    </aside>
  </section>
  <section id="renderPanel" hidden></section>
</main>
```

Include `/static/style.css` for shared tokens and `/static/editor.css` for page-specific layout. Use native controls and button labels; use drag handles plus explicit up/down buttons so sorting is keyboard accessible.

- [ ] **Step 2: Implement project loading and rendering state**

Define `const projectId = decodeURIComponent(location.pathname.split("/").filter(Boolean).pop());`, then use `GET /api/edit-projects/<id>`. Keep this state shape:

```javascript
const state = {
  project: null,
  stage: "browse",
  currentSegmentId: null,
  saveTimer: null,
  renderTimer: null,
  saveInFlight: false,
};
```

Implement `loadProject`, `renderSegmentList`, `renderCurrentSegment`, `renderSummary`, and `setStage`. Default `currentSegmentId` to the first segment and set the player source to `/api/clips/<clip_filename>`. Show the current segment’s board count, detected range, edited range, and duration.

- [ ] **Step 3: Implement segment selection, trimming, and navigation**

Render two synchronized range inputs and two number inputs for `edit_start_time` and `edit_end_time`. Clamp values to `clip_start_time`/`clip_end_time`, reject end values at or before start, update the in-memory project, refresh summary/list, and call `scheduleSave()` after a valid change. Implement previous/next navigation over all segments and clickable cards; selected state must not affect browsing.

- [ ] **Step 4: Implement ordering and auto-save**

Use HTML drag-and-drop for desktop and explicit up/down buttons for all devices. Recompute `order` after every move, update the selected summary, and call:

```javascript
function scheduleSave() {
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(saveProject, 350);
}

async function saveProject() {
  const response = await fetch(`/api/edit-projects/${projectId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ segments: state.project.segments }),
  });
  if (!response.ok) throw new Error((await response.json()).error || "保存失败");
  state.project = await response.json();
  document.querySelector("#editorSaveStatus").textContent = "已自动保存";
}
```

Show “保存失败，点击重试” without discarding local edits. Do not save while a render is active.

- [ ] **Step 5: Implement render polling, preview, and export**

On render confirmation, `POST /api/edit-projects/<id>/render`, set stage to `render`, disable trim/order/selection controls, and poll `GET /api/edit-projects/<id>/render` every 1500 ms. Render progress must update a width percentage and text such as `正在合并第 3/8 段 · 42%`. On `completed`, set the final player source to `/api/edit-projects/<id>/export`, show file metadata and a download link to the same endpoint. On `failed`, re-enable controls and show retry.

- [ ] **Step 6: Add editor CSS and run syntax checks**

Use a two-column desktop grid, a stacked narrow layout, `min-width: 0` on flex/grid children, and text ellipsis for long filenames. Keep cards at the existing small radius and use the shared dark dashboard tokens. Ensure the player has stable `aspect-ratio`, and controls cannot resize the layout.

Run: `node --check pingpong_analyst/static/editor.js`

Expected: exit code 0.

## Task 5: Add the analysis-page entry and editor browser flow

**Files:**
- Modify: `pingpong_analyst/static/index.html`
- Modify: `pingpong_analyst/static/app.js`
- Modify: `pingpong_analyst/static/style.css`

- [ ] **Step 1: Add the disabled entry button to the analysis result area**

Add a button with ID `editProjectBtn`, icon, and label `进入成片编辑` near the completed analysis status. Keep it disabled until a valid `project_id` is available, so older results do not produce a broken link.

- [ ] **Step 2: Wire project ID propagation and compatibility creation**

In `pollResult`, on a completed rally result set `state.editProjectId = data.project_id || null`. If no ID exists, call `POST /api/edit-projects/from-analysis/<task_id>` once, store the returned ID, and enable the button. The click handler must navigate with `location.href = `/editor/${encodeURIComponent(state.editProjectId)}``.

- [ ] **Step 3: Add entry-state styles and reset behavior**

Reset `state.editProjectId` and disable the button when a new video is uploaded or selected. If the project endpoint returns 409/404, show the error in the analysis status and keep the button disabled. Add only the entry styling to `style.css`; keep editor layout in `editor.css`.

- [ ] **Step 4: Verify the browser flow manually**

Start the service with `python -m pingpong_analyst.api --host 127.0.0.1 --port 8077`, open the existing app with Playwright, and verify the button is disabled before analysis and enabled with a real project ID after a seeded completed result. Open `/editor/<project_id>` and verify the page loads the persisted source and segment list.

## Task 6: Complete regression coverage and integration verification

**Files:**
- Modify: `tests/test_api.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Add API render tests with a fake renderer**

Monkeypatch `CompositionRenderer.render` to call the progress callback with `(0.4, "正在合并")`, create the expected output file, then return `{"duration": 3.0}`. Assert render returns immediately with `rendering`, the status endpoint reaches `completed`, and the export endpoint returns `video/mp4`. Add a second test where the fake renderer raises `CompositionError` and assert project status becomes `failed` while segment edits remain intact.

- [ ] **Step 2: Add a real small-video composition smoke test when FFmpeg is available**

Add a short audio/video fixture created with `ffmpeg` using a color video source and a sine audio source. Create two non-overlapping project segments, render to a temporary MP4, and assert the output exists, has video and audio streams, and has duration within 0.2 seconds of the sum of segment durations. Skip only when the `ffmpeg` executable is absent.

- [ ] **Step 3: Run the complete verification suite**

Run:

```bash
node --check pingpong_analyst/static/app.js
node --check pingpong_analyst/static/editor.js
python -m py_compile pingpong_analyst/core/edit_project_store.py pingpong_analyst/core/composition_renderer.py pingpong_analyst/api.py
git diff --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Expected: JavaScript and Python syntax checks exit 0, `git diff --check` is empty, and all tests pass with only documented environment skips.

- [ ] **Step 4: Verify desktop and narrow browser layouts**

Use Playwright CLI with a seeded project page at 1200x840 and 800x700. Capture screenshots under `output/playwright/`, verify no horizontal overflow, and exercise: select/unselect, time input, previous/next, order buttons, reload recovery, render progress, final player, and export link. Confirm the network log contains the project PATCH and render status polling requests.

- [ ] **Step 5: Commit the integrated feature**

```bash
git add pingpong_analyst/core/edit_project_store.py \
  pingpong_analyst/core/composition_renderer.py \
  pingpong_analyst/api.py \
  pingpong_analyst/static/editor.html \
  pingpong_analyst/static/editor.js \
  pingpong_analyst/static/editor.css \
  pingpong_analyst/static/index.html \
  pingpong_analyst/static/app.js \
  pingpong_analyst/static/style.css \
  config.yaml tests/test_edit_project_store.py \
  tests/test_composition_renderer.py tests/test_api.py tests/test_integration.py
git commit -m "feat: add highlight composition editor"
```
