/* PingPong AI Analyst — 前端交互逻辑 v2
   WebSocket 实时追踪 + 剪辑回放 + 分析报告
*/

const $ = (id) => document.getElementById(id);
const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
const wsBase = `${wsProtocol}//${location.host}`;

const state = {
  videoId: null,
  videoName: null,
  videoInfo: null,
  taskId: null,
  actionTaskId: null,
  ws: null,
  streaming: false,
  pollTimer: null,
  analysisData: null,
  playbackSpeed: 1,
  selectedModelId: "base",
  annotationFrame: 0,
  annotationMode: "ball",
  tracknetAnnotations: [],
  trainingJobId: null,
  trainingPollTimer: null,
  annotationLoadToken: 0,
  annotationImage: null,
  annotationZoom: 1,
  annotationPanX: 0,
  annotationPanY: 0,
  annotationPointer: null,
  annotationDragged: false,
  annotationSuppressClick: false,
};

// ========== 初始化 ==========

async function init() {
  await loadDeviceInfo();
  bindEvents();
  bindModals();
  await loadTracknetModels();
}

async function loadDeviceInfo() {
  try {
    const res = await fetch("/api/device");
    const data = await res.json();
    $("deviceText").textContent = `${data.device_name} · ${data.device_type.toUpperCase()}`;
    $("devicePill").querySelector(".status-dot").classList.remove("status-dot--idle");
  } catch {
    $("deviceText").textContent = "设备不可用";
  }
}

function bindEvents() {
  $("uploadBtn").addEventListener("click", () => $("fileInput").click());
  $("fileInput").addEventListener("change", (e) => {
    if (e.target.files[0]) handleUpload(e.target.files[0]);
  });
  $("trackBtn").addEventListener("click", toggleTracking);
  $("annotateBtn").addEventListener("click", openTracknetAnnotationModal);
  $("analyzeBtn").addEventListener("click", startAnalysis);
  $("actionBtn").addEventListener("click", openActionModal);
  $("libraryBtn").addEventListener("click", openLibrary);
  $("tracknetFrameRange").addEventListener("input", (event) => loadAnnotationFrame(Number(event.target.value)));
  $("tracknetFrameInput").addEventListener("change", (event) => loadAnnotationFrame(Number(event.target.value)));
  $("tracknetPrevFrame").addEventListener("click", () => loadAnnotationFrame(state.annotationFrame - 1));
  $("tracknetNextFrame").addEventListener("click", () => loadAnnotationFrame(state.annotationFrame + 1));
  $("tracknetBallMode").addEventListener("click", () => setTracknetAnnotationMode("ball"));
  $("tracknetAbsentBtn").addEventListener("click", setAbsentAnnotation);
  $("tracknetSkipBtn").addEventListener("click", skipAnnotationFrame);
  $("tracknetSaveAnnotationsBtn").addEventListener("click", saveTracknetAnnotations);
  $("tracknetTrainBtn").addEventListener("click", startTracknetTraining);
  $("tracknetCancelTrainingBtn").addEventListener("click", cancelTracknetTraining);
  $("tracknetSaveModelBtn").addEventListener("click", saveTracknetTrainingResult);
  $("tracknetDiscardModelBtn").addEventListener("click", discardTracknetTrainingResult);
  $("tracknetAnnotationCanvas").addEventListener("click", handleBallCanvasClick);
  $("tracknetAnnotationCanvas").addEventListener("pointerdown", beginAnnotationPan);
  $("tracknetAnnotationCanvas").addEventListener("pointermove", moveAnnotationPan);
  $("tracknetAnnotationCanvas").addEventListener("pointerup", endAnnotationPan);
  $("tracknetAnnotationCanvas").addEventListener("pointercancel", endAnnotationPan);
  $("tracknetAnnotationCanvas").addEventListener("wheel", handleAnnotationWheel, { passive: false });
  $("tracknetZoomOut").addEventListener("click", () => setAnnotationZoom(state.annotationZoom - 0.25));
  $("tracknetZoomReset").addEventListener("click", resetAnnotationView);
  $("tracknetZoomIn").addEventListener("click", () => setAnnotationZoom(state.annotationZoom + 0.25));
  $("tracknetModelSelect").addEventListener("change", (event) => {
    state.selectedModelId = event.target.value || "base";
    const selected = event.target.selectedOptions[0];
    $("tracknetModelStatus").textContent = selected?.textContent || "基础模型";
  });
  $("playbackSpeed").addEventListener("change", (event) => {
    state.playbackSpeed = Number(event.target.value) || 1;
    if (state.ws && state.streaming && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(`speed:${state.playbackSpeed}`);
    }
  });

  const zone = $("uploadZone");
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("upload-zone--dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("upload-zone--dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("upload-zone--dragover");
    if (e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0]);
  });
}

function bindModals() {
  $("modalClose").addEventListener("click", () => closeModals());
  $("rallyModalClose").addEventListener("click", () => closeModals());
  $("libraryModalClose").addEventListener("click", () => closeModals());
  $("personSelectModalClose").addEventListener("click", () => closeModals());
  $("actionModalClose").addEventListener("click", () => closeModals());
  $("tracknetAnnotationModalClose").addEventListener("click", () => closeModals());
  document.querySelectorAll(".modal__backdrop").forEach(el => {
    el.addEventListener("click", () => closeModals());
  });
  $("personSelectConfirm").addEventListener("click", () => confirmPersonSelect());
  $("personSelectSkip").addEventListener("click", () => {
    state.personFilter = null;
    closeModals();
    startTracking();
  });
  $("actionOriginalBtn").addEventListener("click", () => startActionAnalysis());
  $("actionAllBtn").addEventListener("click", () => startActionAnalysis({ allClips: true }));
}

function closeModals() {
  $("clipModal").classList.remove("modal--open");
  $("rallyModal").classList.remove("modal--open");
  $("actionModal").classList.remove("modal--open");
  $("libraryModal").classList.remove("modal--open");
  $("personSelectModal").classList.remove("modal--open");
  $("tracknetAnnotationModal").classList.remove("modal--open");
  const player = $("clipPlayer");
  player.pause();
  player.src = "";
}

// ========== 上传 ==========

async function handleUpload(file) {
  $("footerFile").textContent = file.name;
  $("uploadBtn").disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();

    state.videoId = data.video_id;
    state.videoName = data.filename;
    state.videoInfo = data.info;

    // 显示视频信息
    if (data.info) {
      $("footerRes").textContent = `${data.info.width}×${data.info.height}`;
      $("footerFps").textContent = data.info.fps;
      $("footerTotalFrame").textContent = data.info.total_frames;
    }

    $("uploadZone").classList.add("upload-zone--hidden");
    $("viewer").classList.remove("viewer--hidden");
    $("trackBtn").disabled = false;
    $("annotateBtn").disabled = false;
    $("analyzeBtn").disabled = false;
    $("actionBtn").disabled = false;
    $("uploadBtn").disabled = false;
    $("uploadBtn").innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>更换视频`;
  } catch (err) {
    alert("上传失败: " + err.message);
    $("uploadBtn").disabled = false;
  }
}

// ========== 视频库 ==========

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
  if (bytes < 1024 * 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + " MB";
  return (bytes / 1024 / 1024 / 1024).toFixed(2) + " GB";
}

function formatDuration(sec) {
  if (!sec) return "—";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

async function openLibrary() {
  $("libraryModal").classList.add("modal--open");
  $("libraryContent").innerHTML = '<div class="library__loading">加载中...</div>';

  try {
    const res = await fetch("/api/library");
    const data = await res.json();
    renderLibrary(data.videos || []);
  } catch (err) {
    $("libraryContent").innerHTML = `<div class="library__loading">加载失败: ${err.message}</div>`;
  }
}

function renderLibrary(videos) {
  const container = $("libraryContent");

  if (videos.length === 0) {
    container.innerHTML = '<div class="library__loading">视频库为空，请上传视频</div>';
    return;
  }

  let html = '<div class="library__grid">';
  videos.forEach(v => {
    const info = v.info || {};
    const res = info.width ? `${info.width}×${info.height}` : "—";
    const dur = formatDuration(info.duration);
    const size = formatSize(v.size);
    const fps = info.fps ? info.fps + "fps" : "";

    html += `
      <div class="library__item" data-vid="${v.video_id}" data-name="${v.filename}">
        <div class="library__item-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
        </div>
        <div class="library__item-info">
          <div class="library__item-name">${v.filename}</div>
          <div class="library__item-meta">
            <span>${res}</span>
            <span>${dur}</span>
            <span>${fps}</span>
            <span>${size}</span>
          </div>
        </div>
        <button class="library__item-select btn btn--primary btn--sm">选择</button>
      </div>
    `;
  });
  html += '</div>';

  container.innerHTML = html;

  // 绑定选择按钮
  container.querySelectorAll(".library__item").forEach(item => {
    const selectBtn = item.querySelector(".library__item-select");
    selectBtn.addEventListener("click", () => {
      const vid = item.dataset.vid;
      const name = item.dataset.name;
      selectVideo(vid, name);
    });
  });
}

function selectVideo(videoId, filename) {
  state.videoId = videoId;
  state.videoName = filename;

  // 获取视频信息
  fetch(`/api/videos/${videoId}/info`)
    .then(res => res.json())
    .then(info => {
      state.videoInfo = info;
      if (info.width) {
        $("footerRes").textContent = `${info.width}×${info.height}`;
        $("footerFps").textContent = info.fps;
        $("footerTotalFrame").textContent = info.total_frames;
      }
    });

  $("footerFile").textContent = filename;
  $("uploadZone").classList.add("upload-zone--hidden");
  $("viewer").classList.remove("viewer--hidden");
  $("trackBtn").disabled = false;
  $("annotateBtn").disabled = false;
  $("analyzeBtn").disabled = false;
  $("actionBtn").disabled = false;
  $("uploadBtn").innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>更换视频`;

  closeModals();
}

// ========== TrackNet 标注与模型选择 ==========

async function loadTracknetModels() {
  try {
    const res = await fetch("/api/tracknet/models");
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "模型列表加载失败");
    const select = $("tracknetModelSelect");
    const previous = state.selectedModelId;
    select.innerHTML = "";
    for (const model of data.models || []) {
      if (!model.available) continue;
      const option = document.createElement("option");
      option.value = model.model_id;
      option.textContent = model.display_name || model.model_id;
      select.appendChild(option);
    }
    if ([...select.options].some(option => option.value === previous)) {
      select.value = previous;
    } else if (select.options.length) {
      state.selectedModelId = select.options[0].value;
      select.value = state.selectedModelId;
    }
    const selected = select.selectedOptions[0];
    $("tracknetModelStatus").textContent = selected?.textContent || "无可用模型";
  } catch {
    $("tracknetModelStatus").textContent = "模型列表加载失败";
  }
}

async function openTracknetAnnotationModal() {
  if (!state.videoId) return;
  $("tracknetAnnotationModal").classList.add("modal--open");
  resetAnnotationView();
  $("tracknetAnnotationLoading").style.display = "flex";
  $("tracknetAnnotationStatus").textContent = "加载已有标注...";
  try {
    const res = await fetch(`/api/videos/${state.videoId}/tracknet/annotations`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "标注加载失败");
    state.tracknetAnnotations = data.annotations || [];
    const total = Math.max(1, Number(state.videoInfo?.total_frames || 1));
    const range = $("tracknetFrameRange");
    range.max = String(total - 1);
    state.annotationFrame = Math.min(state.annotationFrame, total - 1);
    await loadAnnotationFrame(state.annotationFrame);
    updateTracknetAnnotationSummary();
  } catch (err) {
    $("tracknetAnnotationStatus").textContent = "标注加载失败: " + err.message;
  }
}

async function loadAnnotationFrame(frameIndex) {
  if (!state.videoId) return;
  const total = Math.max(1, Number(state.videoInfo?.total_frames || 1));
  const nextFrame = Math.min(Math.max(0, Math.round(frameIndex)), total - 1);
  state.annotationFrame = nextFrame;
  $("tracknetFrameRange").value = String(nextFrame);
  $("tracknetFrameInput").value = String(nextFrame);
  $("tracknetFrameLabel").textContent = `帧 ${nextFrame} / ${total - 1}`;
  $("tracknetAnnotationLoading").style.display = "flex";
  const token = ++state.annotationLoadToken;
  try {
    const res = await fetch(`/api/videos/${state.videoId}/frame?frame_index=${nextFrame}`);
    if (!res.ok) throw new Error("帧加载失败");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => {
      if (token !== state.annotationLoadToken) return;
      const canvas = $("tracknetAnnotationCanvas");
      canvas.width = Number(state.videoInfo?.width || image.naturalWidth);
      canvas.height = Number(state.videoInfo?.height || image.naturalHeight);
      const canvasWrap = $("tracknetAnnotationCanvas").parentElement;
      canvasWrap.style.setProperty("--tracknet-aspect", String(canvas.width / canvas.height));
      canvasWrap.style.aspectRatio = `${canvas.width} / ${canvas.height}`;
      state.annotationImage = image;
      drawTracknetAnnotationFrame();
      URL.revokeObjectURL(url);
      $("tracknetAnnotationLoading").style.display = "none";
      $("tracknetAnnotationStatus").textContent = "点击球心添加标注";
    };
    image.src = url;
  } catch (err) {
    $("tracknetAnnotationLoading").style.display = "none";
    $("tracknetAnnotationStatus").textContent = err.message;
  }
}

function currentTracknetAnnotation() {
  return state.tracknetAnnotations.find(item => Number(item.frame_index) === state.annotationFrame);
}

function drawTracknetAnnotationFrame() {
  const canvas = $("tracknetAnnotationCanvas");
  const ctx = canvas.getContext("2d");
  if (state.annotationImage) ctx.drawImage(state.annotationImage, 0, 0, canvas.width, canvas.height);
  drawTracknetAnnotationMarker();
}

function clampAnnotationPan() {
  const canvas = $("tracknetAnnotationCanvas");
  const viewport = canvas.parentElement.getBoundingClientRect();
  const scaledWidth = viewport.width * state.annotationZoom;
  const scaledHeight = viewport.height * state.annotationZoom;
  const minX = Math.min(0, viewport.width - scaledWidth);
  const minY = Math.min(0, viewport.height - scaledHeight);
  state.annotationPanX = Math.min(0, Math.max(minX, state.annotationPanX));
  state.annotationPanY = Math.min(0, Math.max(minY, state.annotationPanY));
}

function applyAnnotationTransform() {
  const canvas = $("tracknetAnnotationCanvas");
  clampAnnotationPan();
  canvas.style.transform = `translate(${state.annotationPanX}px, ${state.annotationPanY}px) scale(${state.annotationZoom})`;
  $("tracknetZoomReset").textContent = `${Math.round(state.annotationZoom * 100)}%`;
}

function resetAnnotationView() {
  state.annotationZoom = 1;
  state.annotationPanX = 0;
  state.annotationPanY = 0;
  applyAnnotationTransform();
}

function setAnnotationZoom(nextZoom, clientX = null, clientY = null) {
  const canvas = $("tracknetAnnotationCanvas");
  const viewport = canvas.parentElement.getBoundingClientRect();
  const oldRect = canvas.getBoundingClientRect();
  const anchorX = clientX ?? (viewport.left + viewport.width / 2);
  const anchorY = clientY ?? (viewport.top + viewport.height / 2);
  const sourceX = ((anchorX - oldRect.left) / oldRect.width) * canvas.width;
  const sourceY = ((anchorY - oldRect.top) / oldRect.height) * canvas.height;
  const next = Math.min(4, Math.max(1, Math.round(nextZoom * 4) / 4));
  state.annotationZoom = next;
  state.annotationPanX = anchorX - viewport.left - (sourceX / canvas.width) * viewport.width * next;
  state.annotationPanY = anchorY - viewport.top - (sourceY / canvas.height) * viewport.height * next;
  applyAnnotationTransform();
}

function handleAnnotationWheel(event) {
  event.preventDefault();
  setAnnotationZoom(state.annotationZoom + (event.deltaY < 0 ? 0.25 : -0.25), event.clientX, event.clientY);
}

function beginAnnotationPan(event) {
  if (event.button !== 0) return;
  state.annotationPointer = {
    id: event.pointerId,
    x: event.clientX,
    y: event.clientY,
    panX: state.annotationPanX,
    panY: state.annotationPanY,
  };
  state.annotationDragged = false;
  event.currentTarget.setPointerCapture(event.pointerId);
}

function moveAnnotationPan(event) {
  const pointer = state.annotationPointer;
  if (!pointer || pointer.id !== event.pointerId) return;
  const dx = event.clientX - pointer.x;
  const dy = event.clientY - pointer.y;
  if (Math.abs(dx) > 3 || Math.abs(dy) > 3) state.annotationDragged = true;
  if (!state.annotationDragged || state.annotationZoom <= 1) return;
  state.annotationPanX = pointer.panX + dx;
  state.annotationPanY = pointer.panY + dy;
  applyAnnotationTransform();
}

function endAnnotationPan(event) {
  const pointer = state.annotationPointer;
  if (!pointer || pointer.id !== event.pointerId) return;
  state.annotationSuppressClick = state.annotationDragged;
  state.annotationPointer = null;
  try { event.currentTarget.releasePointerCapture(event.pointerId); } catch {}
}

function drawTracknetAnnotationMarker() {
  const annotation = currentTracknetAnnotation();
  const canvas = $("tracknetAnnotationCanvas");
  if (!annotation) return;
  const ctx = canvas.getContext("2d");
  if (annotation.label === "ball") {
    ctx.beginPath();
    ctx.arc(Number(annotation.x), Number(annotation.y), Math.max(6, canvas.width / 160), 0, Math.PI * 2);
    ctx.strokeStyle = "#64e6a5";
    ctx.lineWidth = Math.max(2, canvas.width / 480);
    ctx.stroke();
    ctx.fillStyle = "#64e6a5";
    ctx.fillRect(Number(annotation.x) - 2, Number(annotation.y) - 2, 4, 4);
  } else {
    ctx.fillStyle = "#ff8274";
    ctx.font = `${Math.max(14, canvas.width / 45)}px sans-serif`;
    ctx.fillText("无球", 18, 32);
  }
}

function upsertTracknetAnnotation(annotation) {
  state.tracknetAnnotations = state.tracknetAnnotations.filter(
    item => Number(item.frame_index) !== state.annotationFrame
  );
  state.tracknetAnnotations.push(annotation);
  state.tracknetAnnotations.sort((a, b) => Number(a.frame_index) - Number(b.frame_index));
}

function setTracknetAnnotationMode(mode) {
  state.annotationMode = mode;
  $("tracknetBallMode").classList.toggle("btn--primary", mode === "ball");
  $("tracknetAbsentBtn").classList.toggle("btn--primary", mode === "absent");
  $("tracknetAnnotationStatus").textContent = mode === "ball" ? "点击球心添加标注" : "当前帧将标记为无球";
}

function handleBallCanvasClick(event) {
  if (state.annotationMode !== "ball") return;
  if (state.annotationSuppressClick) {
    state.annotationSuppressClick = false;
    return;
  }
  const canvas = $("tracknetAnnotationCanvas");
  const rect = canvas.getBoundingClientRect();
  const x = Math.min(canvas.width, Math.max(0, (event.clientX - rect.left) * canvas.width / rect.width));
  const y = Math.min(canvas.height, Math.max(0, (event.clientY - rect.top) * canvas.height / rect.height));
  upsertTracknetAnnotation({ frame_index: state.annotationFrame, label: "ball", x, y });
  drawTracknetAnnotationFrame();
  updateTracknetAnnotationSummary();
  $("tracknetAnnotationStatus").textContent = `已标注帧 ${state.annotationFrame}，可继续拖动时间轴`;
}

function setAbsentAnnotation() {
  upsertTracknetAnnotation({ frame_index: state.annotationFrame, label: "absent" });
  loadAnnotationFrame(state.annotationFrame);
  updateTracknetAnnotationSummary();
  setTracknetAnnotationMode("absent");
}

function skipAnnotationFrame() {
  state.tracknetAnnotations = state.tracknetAnnotations.filter(
    item => Number(item.frame_index) !== state.annotationFrame
  );
  loadAnnotationFrame(state.annotationFrame);
  updateTracknetAnnotationSummary();
  $("tracknetAnnotationStatus").textContent = `已跳过帧 ${state.annotationFrame}`;
}

function updateTracknetAnnotationSummary() {
  const positive = state.tracknetAnnotations.filter(item => item.label === "ball").length;
  const absent = state.tracknetAnnotations.filter(item => item.label === "absent").length;
  $("tracknetAnnotationCount").textContent = `${state.tracknetAnnotations.length} 个标注 · 球 ${positive} · 无球 ${absent}`;
  $("tracknetTrainBtn").disabled = positive < 12 || Boolean(state.trainingJobId);
}

async function saveTracknetAnnotations() {
  if (!state.videoId) return false;
  try {
    const res = await fetch(`/api/videos/${state.videoId}/tracknet/annotations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ annotations: state.tracknetAnnotations }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "保存标注失败");
    state.tracknetAnnotations = data.annotations || state.tracknetAnnotations;
    $("tracknetAnnotationStatus").textContent = `已保存 ${data.count} 个标注`;
    updateTracknetAnnotationSummary();
    return true;
  } catch (err) {
    $("tracknetAnnotationStatus").textContent = "保存失败: " + err.message;
    return false;
  }
}

async function startTracknetTraining() {
  if (!state.videoId || state.tracknetAnnotations.filter(item => item.label === "ball").length < 12) return;
  if (!(await saveTracknetAnnotations())) return;
  $("tracknetTrainingPanel").hidden = false;
  $("tracknetTrainingStatus").textContent = "训练排队中";
  $("tracknetTrainingProgressBar").style.width = "0%";
  $("tracknetTrainingProgressText").textContent = "0%";
  $("tracknetSaveModelBtn").disabled = true;
  $("tracknetDiscardModelBtn").disabled = true;
  try {
    const res = await fetch(`/api/videos/${state.videoId}/tracknet/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: state.selectedModelId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "训练启动失败");
    state.trainingJobId = data.job_id;
    updateTracknetAnnotationSummary();
    pollTracknetTraining(data.job_id);
  } catch (err) {
    $("tracknetTrainingStatus").textContent = "启动失败: " + err.message;
    $("tracknetTrainBtn").disabled = false;
  }
}

function pollTracknetTraining(jobId) {
  if (state.trainingPollTimer) clearInterval(state.trainingPollTimer);
  state.trainingPollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/tracknet/train/${jobId}`);
      const data = await res.json();
      const progress = Number(data.progress || 0);
      $("tracknetTrainingProgressBar").style.width = progress + "%";
      $("tracknetTrainingProgressText").textContent = progress + "%";
      $("tracknetTrainingStatus").textContent = data.message || data.status;
      if (data.metrics?.validation_loss !== undefined) {
        const validationLoss = Number(data.metrics.validation_loss);
        const lossText = Math.abs(validationLoss) < 0.001
          ? validationLoss.toExponential(3)
          : validationLoss.toFixed(4);
        $("tracknetTrainingMetrics").textContent = `验证损失 ${lossText} · 检测率 ${(Number(data.metrics.positive_detection_rate || 0) * 100).toFixed(0)}%`;
      }
      if (["completed", "failed", "cancelled", "discarded", "saved"].includes(data.status)) {
        clearInterval(state.trainingPollTimer);
        state.trainingPollTimer = null;
        if (data.status === "completed") {
          $("tracknetSaveModelBtn").disabled = false;
          $("tracknetDiscardModelBtn").disabled = false;
        }
        if (["failed", "cancelled", "discarded"].includes(data.status)) {
          state.trainingJobId = null;
          updateTracknetAnnotationSummary();
        }
      }
    } catch {}
  }, 1200);
}

async function cancelTracknetTraining() {
  if (!state.trainingJobId) return;
  await fetch(`/api/tracknet/train/${state.trainingJobId}/cancel`, { method: "POST" });
}

async function saveTracknetTrainingResult() {
  if (!state.trainingJobId) return;
  const res = await fetch(`/api/tracknet/train/${state.trainingJobId}/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  const data = await res.json();
  if (!res.ok) {
    $("tracknetTrainingStatus").textContent = data.error || "模型保存失败";
    return;
  }
  $("tracknetTrainingStatus").textContent = "模型已保存，可在下拉框中手动选择";
  $("tracknetSaveModelBtn").disabled = true;
  $("tracknetDiscardModelBtn").disabled = true;
  state.trainingJobId = null;
  updateTracknetAnnotationSummary();
  await loadTracknetModels();
}

async function discardTracknetTrainingResult() {
  if (!state.trainingJobId) return;
  await fetch(`/api/tracknet/train/${state.trainingJobId}`, { method: "DELETE" });
  $("tracknetTrainingStatus").textContent = "训练结果已放弃";
  state.trainingJobId = null;
  $("tracknetSaveModelBtn").disabled = true;
  $("tracknetDiscardModelBtn").disabled = true;
  updateTracknetAnnotationSummary();
}

// ========== WebSocket 实时追踪 ==========

function toggleTracking() {
  if (!state.videoId) return;
  if (state.streaming) {
    stopTracking();
  } else {
    // 回合追踪不需要先做人框预览，避免误触发 YOLO/躯干分析。
    state.personFilter = null;
    startTracking();
  }
}

// ========== 人物选择 ==========

state.personFilter = null;
state.personPreviewData = null;
state.selectedPersons = [];

async function openPersonSelect() {
  state.selectedPersons = [];
  state.personFilter = null;
  $("personSelectModal").classList.add("modal--open");
  $("personSelectOverlay").innerHTML = "";
  $("personSelectLoading").style.display = "flex";
  $("personSelectConfirm").disabled = true;
  $("personSelectCount").innerHTML = "已选: 0/2";
  $("personSelectImg").src = "";

  try {
    const res = await fetch(`/api/videos/${state.videoId}/preview`);
    const data = await res.json();
    state.personPreviewData = data;

    $("personSelectImg").src = data.image;
    $("personSelectImg").onload = () => {
      renderPersonBoxes(data);
      $("personSelectLoading").style.display = "none";
    };
  } catch (err) {
    $("personSelectLoading").textContent = "检测失败，请跳过";
    $("personSelectLoading").style.display = "flex";
  }
}

function renderPersonBoxes(data) {
  const overlay = $("personSelectOverlay");
  const img = $("personSelectImg");
  const imgW = img.clientWidth;
  const imgH = img.clientHeight;
  overlay.innerHTML = "";

  if (!data.persons || data.persons.length === 0) {
    $("personSelectLoading").textContent = "未检测到人物，请跳过";
    $("personSelectLoading").style.display = "flex";
    return;
  }

  data.persons.forEach((p, idx) => {
    const [x1, y1, x2, y2] = p.bbox;
    const box = document.createElement("div");
    box.className = "person-select__box";
    box.style.left = (x1 / data.width * 100) + "%";
    box.style.top = (y1 / data.height * 100) + "%";
    box.style.width = ((x2 - x1) / data.width * 100) + "%";
    box.style.height = ((y2 - y1) / data.height * 100) + "%";

    const label = document.createElement("div");
    label.className = "person-select__box-label";
    label.textContent = "P" + (idx + 1);
    box.appendChild(label);

    box.addEventListener("click", (e) => {
      e.stopPropagation();
      togglePersonSelect(idx, p.bbox, box);
    });

    overlay.appendChild(box);
  });
}

function togglePersonSelect(idx, bbox, boxEl) {
  const existIdx = state.selectedPersons.findIndex(s => s.idx === idx);
  if (existIdx >= 0) {
    state.selectedPersons.splice(existIdx, 1);
    boxEl.classList.remove("person-select__box--selected");
  } else {
    if (state.selectedPersons.length >= 2) {
      return;
    }
    state.selectedPersons.push({ idx, bbox });
    boxEl.classList.add("person-select__box--selected");
  }

  const count = state.selectedPersons.length;
  $("personSelectCount").innerHTML = `已选: <strong>${count}</strong>/2`;
  $("personSelectConfirm").disabled = count === 0;
}

function confirmPersonSelect() {
  if (state.selectedPersons.length === 0) {
    state.personFilter = null;
  } else {
    state.personFilter = state.selectedPersons.map(s => s.bbox);
  }
  closeModals();
  startTracking();
}

function startTracking() {
  state.streaming = true;
  $("trackBtn").innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>停止追踪`;
  $("streamPlaceholder").style.display = "none";
  $("streamLoading").style.display = "flex";
  $("loadingText").textContent = "加载模型中...";
  $("modelProgress").style.display = "flex";
  $("modelProgressFill").style.width = "0%";
  $("modelProgressPct").textContent = "0%";
  $("streamCanvas").style.display = "none";
  $("trackingProgress").style.display = "none";
  $("liveBadge").textContent = "LOADING";
  $("streamDot").classList.remove("status-dot--idle");
  $("streamText").textContent = "加载中...";

  // 连接 WebSocket
  const wsUrl = `${wsBase}/api/ws/${state.videoId}?mode=rally&model_id=${encodeURIComponent(state.selectedModelId)}&playback_speed=${state.playbackSpeed}`;
  const ws = new WebSocket(wsUrl);
  state.ws = ws;

  const canvas = $("streamCanvas");
  const ctx = canvas.getContext("2d");

  ws.onmessage = async (event) => {
    if (typeof event.data === "string") {
      // JSON 消息
      const msg = JSON.parse(event.data);

      if (msg.type === "status") {
        if (msg.status === "loading") {
          $("loadingText").textContent = msg.message || "加载模型中...";
          $("modelProgress").style.display = "flex";
          $("modelProgressFill").style.width = (msg.percent || 0) + "%";
          $("modelProgressPct").textContent = (msg.percent || 0) + "%";
        } else if (msg.status === "tracking") {
          $("streamLoading").style.display = "none";
          $("modelProgress").style.display = "none";
          $("streamCanvas").style.display = "block";
          $("trackingProgress").style.display = "block";
          $("liveBadge").textContent = msg.warning ? "PREVIEW" : "LIVE";
          $("streamText").textContent = msg.warning ? "仅预览" : "追踪中";
          setTrackingNotice(msg.warning || "");
          if (msg.playback_speed) setPlaybackSpeedControl(msg.playback_speed);
        } else if (msg.status === "speed") {
          setPlaybackSpeedControl(msg.playback_speed);
        } else if (msg.status === "completed") {
          finishTracking();
        } else if (msg.status === "error") {
          $("streamLoading").innerHTML = `<div style="color: var(--data-red)">错误: ${msg.error}</div>`;
          stopTracking();
        }
      } else if (msg.type === "meta") {
        updateMetrics(msg);
      }
    } else {
      // Binary: JPEG 帧
      const blob = new Blob([event.data], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
      };
      img.src = url;
    }
  };

  ws.onerror = () => {
    $("streamLoading").innerHTML = `<div style="color: var(--data-red)">连接失败</div>`;
    stopTracking();
  };

  ws.onclose = () => {
    if (state.streaming) {
      finishTracking();
    }
  };
}

function stopTracking() {
  state.streaming = false;
  if (state.ws) {
    try { state.ws.send("stop"); } catch {}
    try { state.ws.close(); } catch {}
    state.ws = null;
  }
  resetTrackingUI();
}

function finishTracking() {
  state.streaming = false;
  state.ws = null;
  resetTrackingUI();
  $("liveBadge").textContent = "DONE";
  $("streamText").textContent = "已完成";
  $("footerState").textContent = "完成";
}

function resetTrackingUI() {
  $("trackBtn").innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>开始追踪`;
  $("streamDot").classList.add("status-dot--idle");
  $("streamLoading").style.display = "none";
  $("streamCanvas").style.display = "none";
  $("trackingProgress").style.display = "none";
  $("streamPlaceholder").style.display = "flex";
  setTrackingNotice("");
}

function setPlaybackSpeedControl(speed) {
  const control = $("playbackSpeed");
  if (!control || !speed) return;
  state.playbackSpeed = Number(speed);
  control.value = String(speed);
}

function setTrackingNotice(message) {
  const notice = $("trackingNotice");
  if (!notice) return;
  notice.textContent = message;
  notice.hidden = !message;
}

function updateMetrics(meta) {
  // 实时指标
  $("metricBoards").textContent = meta.board_count || 0;
  $("metricBallSpeed").innerHTML = `${meta.ball_speed || 0}<span class="metric__unit">px/f</span>`;
  if (meta.rally_count !== undefined) {
    $("metricRallies").textContent = meta.rally_count;
  }

  // 底部状态栏
  $("footerFrame").textContent = meta.frame;
  $("footerTime").textContent = meta.timestamp?.toFixed(1);
  $("footerState").textContent = meta.rally_state || "—";
  if (meta.warning) setTrackingNotice(meta.warning);
  if (meta.playback_speed) setPlaybackSpeedControl(meta.playback_speed);

  // 追踪进度条
  if (meta.total_frames > 0) {
    const pct = (meta.frame / meta.total_frames) * 100;
    $("trackingProgressBar").style.width = pct + "%";
  }

  // 关节角度
  if (meta.mode === "action" && meta.arm_angles) {
    $("armAnglesPanel").style.display = "block";
    $("angleLeftElbow").textContent = meta.arm_angles.left_elbow_angle?.toFixed(0) + "°" || "—";
    $("angleRightElbow").textContent = meta.arm_angles.right_elbow_angle?.toFixed(0) + "°" || "—";
    $("angleLeftShoulder").textContent = meta.arm_angles.left_shoulder_angle?.toFixed(0) + "°" || "—";
    $("angleRightShoulder").textContent = meta.arm_angles.right_shoulder_angle?.toFixed(0) + "°" || "—";
  } else {
    $("armAnglesPanel").style.display = "none";
  }
}

// ========== 深度分析 ==========

async function startAnalysis() {
  if (!state.videoId) return;

  const minBoards = parseInt($("minBoardsInput").value) || 6;

  $("analyzeBtn").disabled = true;
  $("analysisBadge").textContent = "分析中";
  $("analysisText").textContent = "正在进行纯球回合分析并生成剪辑...";
  updateAnalysisProgress(0);

  try {
    const res = await fetch(`/api/analyze/${state.videoId}?min_boards=${minBoards}&model_id=${encodeURIComponent(state.selectedModelId)}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "回合分析启动失败");
    state.taskId = data.task_id;
    pollResult(state.taskId);
  } catch (err) {
    $("analysisText").textContent = "分析失败: " + err.message;
    $("analysisBadge").textContent = "失败";
    $("analyzeBtn").disabled = false;
  }
}

function updateAnalysisProgress(progress) {
  const value = Math.min(100, Math.max(0, Math.round(Number(progress) || 0)));
  $("progressBar").style.width = value + "%";
  $("analysisProgressText").textContent = value + "%";
}

function pollResult(taskId) {
  if (state.pollTimer) clearInterval(state.pollTimer);

  state.pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/result/${taskId}`);
      const data = await res.json();

      if (data.progress !== undefined) updateAnalysisProgress(data.progress);
      if (data.message) $("analysisText").textContent = data.message;

      if (data.status === "completed") {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        state.analysisData = data;

        updateAnalysisProgress(100);
        $("analysisBadge").textContent = "完成";
        $("analysisText").textContent = `检测到 ${data.total_rallies} 个回合, ${data.clips.length} 个剪辑片段`;
        $("analyzeBtn").disabled = false;
        $("actionAllBtn").disabled = !data.clips || data.clips.length === 0;

        renderRallies(data.rallies, data.clips);
        $("metricRallies").textContent = data.total_rallies;
        $("rallyCountBadge").textContent = data.total_rallies;
      } else if (data.status === "failed") {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        $("analysisText").textContent = "分析失败: " + data.error;
        $("analysisBadge").textContent = "失败";
        $("analyzeBtn").disabled = false;
      }
    } catch {}
  }, 2000);
}

// ========== 回合列表 ==========

function renderRallies(rallies, clips) {
  const list = $("rallyList");
  list.innerHTML = "";

  if (!rallies || rallies.length === 0) {
    list.innerHTML = '<div class="rally-list--empty">未检测到符合阈值的回合</div>';
    return;
  }

  rallies.forEach((rally, i) => {
    const item = document.createElement("div");
    item.className = "rally-item";
    item.innerHTML = `
      <div class="rally-item__index">${String(i + 1).padStart(2, "0")}</div>
      <div class="rally-item__info">
        <div class="rally-item__time">${rally.start_time.toFixed(1)}s — ${rally.end_time.toFixed(1)}s</div>
        <div class="rally-item__duration">时长 ${rally.duration.toFixed(1)}s · 纯球轨迹分析</div>
      </div>
      <div class="rally-item__boards">${rally.board_count}</div>
    `;

    // 点击打开详情弹窗
    item.addEventListener("click", () => openRallyDetail(rally, clips?.[i]));

    list.appendChild(item);
  });
}

// ========== 弹窗 ==========

function openRallyDetail(rally, clipFilename) {
  const modal = $("rallyModal");
  $("rallyModalTitle").textContent = `回合 ${rally.index} · ${rally.board_count}板 · ${rally.duration.toFixed(1)}s`;

  let html = `
    <div class="rally-detail__summary">
      <div class="detail-stat"><div class="detail-stat__label">开始</div><div class="detail-stat__value">${rally.start_time}s</div></div>
      <div class="detail-stat"><div class="detail-stat__label">结束</div><div class="detail-stat__value">${rally.end_time}s</div></div>
      <div class="detail-stat"><div class="detail-stat__label">时长</div><div class="detail-stat__value">${rally.duration}s</div></div>
      <div class="detail-stat"><div class="detail-stat__label">板数</div><div class="detail-stat__value">${rally.board_count}</div></div>
    </div>
  `;

  if (rally.hit_events && rally.hit_events.length > 0) {
    html += `<div class="rally-detail__events"><div class="events__title">击球事件序列</div><div class="events__timeline">`;
    rally.hit_events.forEach((hit, j) => {
      const sideLabel = hit.side === "left" ? "左侧" : "右侧";
      const typeLabel = hit.type === "drive" ? "撞击" : hit.type === "spin" ? "摩擦" : "未知";
      const typeClass = hit.type === "drive" ? "event--drive" : hit.type === "spin" ? "event--spin" : "event--unknown";
      html += `
        <div class="event-item ${typeClass}">
          <div class="event-item__num">${j + 1}</div>
          <div class="event-item__info">
            <span class="event-item__time">${hit.timestamp}s</span>
            <span class="event-item__side">${sideLabel}</span>
          </div>
          <div class="event-item__type">${typeLabel}</div>
        </div>
      `;
    });
    html += `</div></div>`;
  }

  if (clipFilename) {
    html += `
      <div class="rally-detail__actions">
        <button class="btn btn--primary" id="playClipBtn">播放剪辑</button>
        <a class="btn" href="/api/clips/${clipFilename}" download>下载片段</a>
        <button class="btn" id="actionClipBtn">动作分析此回合</button>
      </div>
    `;
  }

  $("rallyDetailContent").innerHTML = html;
  modal.classList.add("modal--open");

  // 绑定播放按钮
  if (clipFilename) {
    $("playClipBtn").addEventListener("click", () => {
      closeModals();
      setTimeout(() => openClipPlayer(clipFilename, rally), 200);
    });
    $("actionClipBtn").addEventListener("click", () => {
      closeModals();
      startActionAnalysis({ clipFilename });
    });
  }
}

function openClipPlayer(clipFilename, rally) {
  const modal = $("clipModal");
  const player = $("clipPlayer");

  $("modalTitle").textContent = `剪辑 ${rally.index} · ${rally.board_count}板`;
  player.src = `/api/clips/${clipFilename}`;

  let detailsHtml = `
    <div class="modal__detail-row"><span>时间范围</span><strong>${rally.start_time}s — ${rally.end_time}s</strong></div>
    <div class="modal__detail-row"><span>时长</span><strong>${rally.duration}s</strong></div>
    <div class="modal__detail-row"><span>板数</span><strong>${rally.board_count}</strong></div>
    <div class="modal__detail-row"><span>击球次数</span><strong>${rally.hit_events?.length || 0}</strong></div>
    <div class="modal__detail-actions">
      <a class="btn" href="/api/clips/${clipFilename}" download>下载片段</a>
    </div>
  `;
  $("modalDetails").innerHTML = detailsHtml;

  modal.classList.add("modal--open");
}

// ========== 独立动作分析 ==========

function openActionModal() {
  if (!state.videoId) return;
  $("actionModalTitle").textContent = "动作分析";
  $("actionSourcePanel").style.display = "block";
  $("actionProgressPanel").style.display = "none";
  $("actionReportPanel").style.display = "none";
  $("actionAllBtn").disabled = !state.analysisData?.clips?.length;
  $("actionModal").classList.add("modal--open");
}

async function startActionAnalysis({ clipFilename = null, allClips = false } = {}) {
  if (!state.videoId) return;
  $("actionModal").classList.add("modal--open");
  $("actionSourcePanel").style.display = "none";
  $("actionProgressPanel").style.display = "flex";
  $("actionReportPanel").style.display = "none";
  $("actionProgressText").textContent = allClips ? "正在分析全部回合动作..." : "正在分析动作...";

  const params = new URLSearchParams();
  if (clipFilename) params.set("clip_filename", clipFilename);
  if (allClips) params.set("all_clips", "true");

  try {
    const query = params.toString() ? `?${params.toString()}` : "";
    const res = await fetch(`/api/action/analyze/${state.videoId}${query}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "动作分析启动失败");
    state.actionTaskId = data.task_id;
    pollActionResult(state.actionTaskId);
  } catch (err) {
    $("actionProgressPanel").style.display = "none";
    $("actionSourcePanel").style.display = "block";
    $("actionProgressText").textContent = "动作分析失败: " + err.message;
  }
}

function pollActionResult(taskId) {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/result/${taskId}`);
      const data = await res.json();
      if (data.status === "completed") {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        renderActionReport(data);
      } else if (data.status === "failed") {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        $("actionProgressPanel").style.display = "none";
        $("actionSourcePanel").style.display = "block";
        $("actionProgressText").textContent = "动作分析失败: " + data.error;
      }
    } catch {}
  }, 1500);
}

function renderActionReport(data) {
  $("actionModalTitle").textContent = "动作分析结果";
  $("actionProgressPanel").style.display = "none";
  $("actionSourcePanel").style.display = "none";
  $("actionReportPanel").style.display = "block";

  let html = `<div class="action-report__meta">已分析 ${data.analyses?.length || 0} 个视频片段</div>`;
  for (const analysis of data.analyses || []) {
    html += `<div class="action-report__source"><div class="action-report__source-title">${analysis.source} · ${analysis.frames_analyzed} 帧</div>`;
    if (!analysis.persons || analysis.persons.length === 0) {
      html += `<div class="action-report__empty">未检测到稳定的人体姿态</div></div>`;
      continue;
    }
    html += `<div class="action-report__people">`;
    for (const person of analysis.persons) {
      const arm = Object.entries(person.average_arm_angles || {})
        .map(([key, value]) => `${key}: ${Number(value).toFixed(1)}°`).join(" · ");
      const body = Object.entries(person.average_body_metrics || {})
        .map(([key, value]) => `${key}: ${Number(value).toFixed(1)}`).join(" · ");
      html += `<div class="action-report__person">
        <div><strong>${person.identity || "unknown"}</strong><span>${person.track_id} · 置信度 ${(Number(person.identity_confidence || 0) * 100).toFixed(0)}%</span></div>
        <div class="action-report__metrics">${arm || "暂无手臂角度"}</div>
        <div class="action-report__metrics">${body || "暂无躯干指标"}</div>
      </div>`;
    }
    html += `</div></div>`;
  }
  html += `<div class="action-report__hint">逐帧姿态、角度和身份字段已保存在本次任务结果中；无法确认身份的帧标记为 unknown。</div>`;
  $("actionReport").innerHTML = html;
}

// ========== 启动 ==========

init();
