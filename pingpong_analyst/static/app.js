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
  devicePollTimer: null,
  deviceMetricsInFlight: false,
  analysisData: null,
  editProjectId: null,
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
  tableCalibration: null,
  calibrationFrame: 0,
  calibrationMode: "corners",
  calibrationCorners: [],
  calibrationNetPoints: [],
  calibrationImage: null,
  calibrationLoadToken: 0,
};

// ========== 初始化 ==========

async function init() {
  await loadDeviceInfo();
  bindEvents();
  bindModals();
  startDeviceMetricsPolling();
  await loadTracknetModels();
  handleUrlParams();
}

function handleUrlParams() {
  const params = new URLSearchParams(window.location.search);
  const videoId = params.get("video");
  const taskId = params.get("task");
  if (videoId) {
    fetch(`/api/videos/${videoId}/info`)
      .then(res => res.ok ? res.json() : null)
      .then(info => {
        if (info) {
          selectVideo(videoId, info.filename || videoId);
        }
      })
      .catch(() => {});
  } else if (taskId) {
    openTaskFromHistory(taskId);
  }
}

async function openTaskFromHistory(taskId) {
  state.taskId = taskId;
  try {
    const res = await fetch(`/api/result/${taskId}`);
    const data = await res.json();
    if (data.mode === "action" || String(taskId).startsWith("action_")) {
      state.actionTaskId = taskId;
      if (data.video_id) {
        try {
          const infoRes = await fetch(`/api/videos/${data.video_id}/info`);
          if (infoRes.ok) {
            const info = await infoRes.json();
            selectVideo(data.video_id, info.filename || data.video_id);
          }
        } catch {}
      }
      if (data.status === "completed") {
        renderActionReport(data);
        return;
      }
      if (data.status === "failed") {
        openActionModal();
        $("actionSourcePanel").style.display = "block";
        $("actionProgressPanel").style.display = "none";
        $("actionProgressText").textContent = "动作分析失败: " + (data.error || data.message || "");
        return;
      }
      openActionModal();
      $("actionSourcePanel").style.display = "none";
      $("actionProgressPanel").style.display = "flex";
      $("actionReportPanel").style.display = "none";
      updateActionProgress(data.progress, data.message);
      pollActionResult(taskId);
      return;
    }
  } catch {}
  pollResult(taskId);
}

function deviceTypeLabel(deviceType) {
  if (deviceType === "tensorrt") return "CUDA";
  return String(deviceType || "cpu").toUpperCase();
}

function cleanDeviceName(name) {
  return String(name || "未知设备").replace(/\s*\+\s*TensorRT\s*$/i, "");
}

function formatMetricNumber(value, digits = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "--";
}

function formatDeviceMetrics(runtime, deviceType) {
  if (!runtime) return "实时指标不可用";

  if (deviceType === "cpu") {
    return runtime.message || "CPU 模式无 GPU 实时指标";
  }

  if (deviceType === "mps") {
    const memory = runtime.memory_used_mb === null || runtime.memory_used_mb === undefined
      ? "--"
      : `${formatMetricNumber(runtime.memory_used_mb, 0)}MB`;
    return `显存 ${memory} · ${runtime.message || "MPS 不提供利用率和温度"}`;
  }

  if (!runtime.available) {
    return runtime.message || "nvidia-smi 实时指标不可用";
  }

  const memoryUsed = runtime.memory_used_mb;
  const memoryTotal = runtime.memory_total_mb;
  const memoryText = memoryUsed === null || memoryUsed === undefined || memoryTotal === null || memoryTotal === undefined
    ? "显存 --"
    : `显存 ${formatMetricNumber(memoryUsed / 1024, 1)}/${formatMetricNumber(memoryTotal / 1024, 1)}GB`;
  const temperature = runtime.temperature_c === null || runtime.temperature_c === undefined
    ? "温度 --"
    : `${formatMetricNumber(runtime.temperature_c)}°C`;
  const power = runtime.power_draw_w === null || runtime.power_draw_w === undefined
    ? "功耗 --"
    : runtime.power_limit_w === null || runtime.power_limit_w === undefined
      ? `${formatMetricNumber(runtime.power_draw_w, 1)}W`
      : `${formatMetricNumber(runtime.power_draw_w, 1)}/${formatMetricNumber(runtime.power_limit_w, 1)}W`;
  return `GPU ${formatMetricNumber(runtime.gpu_utilization_percent)}% · ${memoryText} · ${temperature} · ${power}`;
}

function renderDeviceStatus(data) {
  const deviceType = data?.device_type || "cpu";
  const runtime = data?.runtime || null;
  const deviceName = cleanDeviceName(runtime?.gpu_name || data?.device_name);
  $("deviceText").textContent = `${deviceName} · ${deviceTypeLabel(deviceType)}`;
  $("deviceMetrics").textContent = formatDeviceMetrics(runtime, deviceType);
  const dot = $("devicePill").querySelector(".status-dot");
  dot.classList.toggle("status-dot--idle", !(runtime?.available || deviceType === "mps"));
}

async function loadDeviceInfo() {
  try {
    const res = await fetch("/api/device", { cache: "no-store" });
    if (!res.ok) throw new Error("device request failed");
    const data = await res.json();
    renderDeviceStatus(data);
  } catch {
    $("deviceText").textContent = "设备不可用";
    $("deviceMetrics").textContent = "无法读取设备状态";
  }
}

async function refreshDeviceMetrics() {
  if (state.deviceMetricsInFlight) return;
  state.deviceMetricsInFlight = true;
  try {
    const res = await fetch("/api/device/metrics", { cache: "no-store" });
    if (!res.ok) throw new Error("metrics request failed");
    renderDeviceStatus(await res.json());
  } catch {
    // Keep the last known device name and expose only the failed live sample.
    $("deviceMetrics").textContent = "实时指标读取失败";
  } finally {
    state.deviceMetricsInFlight = false;
  }
}

function startDeviceMetricsPolling() {
  if (state.devicePollTimer) clearInterval(state.devicePollTimer);
  state.devicePollTimer = setInterval(refreshDeviceMetrics, 1000);
}

function bindEvents() {
  $("uploadBtn").addEventListener("click", () => $("fileInput").click());
  $("fileInput").addEventListener("change", (e) => {
    if (e.target.files[0]) handleUpload(e.target.files[0]);
  });
  $("trackBtn").addEventListener("click", toggleTracking);
  $("annotateBtn").addEventListener("click", openTracknetAnnotationModal);
  $("calibrateBtn").addEventListener("click", openTableCalibrationModal);
  $("analyzeBtn").addEventListener("click", startAnalysis);
  $("actionBtn").addEventListener("click", openActionModal);
  $("editProjectBtn").addEventListener("click", openEditProject);
  $("libraryBtn").addEventListener("click", openLibrary);
  $("tracknetFrameRange").addEventListener("input", (event) => loadAnnotationFrame(Number(event.target.value)));
  $("tracknetFrameInput").addEventListener("change", (event) => loadAnnotationFrame(Number(event.target.value)));
  $("tracknetPrevFrame").addEventListener("click", () => loadAnnotationFrame(state.annotationFrame - 1));
  $("tracknetNextFrame").addEventListener("click", () => loadAnnotationFrame(state.annotationFrame + 1));
  document.querySelectorAll(".tracknet-annotation__jump-row [data-jump]").forEach(btn => {
    btn.addEventListener("click", () => {
      const step = Number(btn.dataset.jump);
      loadAnnotationFrame(state.annotationFrame + step);
    });
  });
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
  $("tableCalibrationFrameRange").addEventListener("input", (event) => loadTableCalibrationFrame(Number(event.target.value)));
  $("tableCalibrationFrameInput").addEventListener("change", (event) => loadTableCalibrationFrame(Number(event.target.value)));
  $("tableCalibrationPrevFrame").addEventListener("click", () => loadTableCalibrationFrame(state.calibrationFrame - 1));
  $("tableCalibrationNextFrame").addEventListener("click", () => loadTableCalibrationFrame(state.calibrationFrame + 1));
  $("tableCalibrationCornersMode").addEventListener("click", () => setCalibrationMode("corners"));
  $("tableCalibrationNetMode").addEventListener("click", () => setCalibrationMode("net"));
  $("tableCalibrationResetBtn").addEventListener("click", resetTableCalibrationPoints);
  $("tableCalibrationSaveBtn").addEventListener("click", saveTableCalibration);
  $("tableCalibrationCanvas").addEventListener("click", handleTableCalibrationClick);
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
  $("tableCalibrationModalClose").addEventListener("click", () => closeModals());
  document.querySelectorAll(".modal__backdrop").forEach(el => {
    el.addEventListener("click", () => closeModals());
  });
  $("personSelectConfirm").addEventListener("click", () => confirmPersonSelect());
  $("personSelectSkip").addEventListener("click", () => {
    state.personFilter = null;
    closeModals();
    startTracking();
  });
  $("actionOriginalBtn").addEventListener("click", () => startActionAnalysis({ source: "original" }));
  $("actionAllBtn").addEventListener("click", () => startActionAnalysis({ allClips: true }));
}

function closeModals() {
  $("clipModal").classList.remove("modal--open");
  $("rallyModal").classList.remove("modal--open");
  $("actionModal").classList.remove("modal--open");
  $("libraryModal").classList.remove("modal--open");
  $("personSelectModal").classList.remove("modal--open");
  $("tracknetAnnotationModal").classList.remove("modal--open");
  $("tableCalibrationModal").classList.remove("modal--open");
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
    state.tableCalibration = null;

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
    $("calibrateBtn").disabled = false;
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
  state.tableCalibration = null;

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
  $("calibrateBtn").disabled = false;
  $("analyzeBtn").disabled = false;
  $("actionBtn").disabled = false;
  $("uploadBtn").innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>更换视频`;

  closeModals();
}

// ========== 球台标定 ==========

async function fetchTableCalibration() {
  if (!state.videoId) return null;
  const res = await fetch(`/api/videos/${state.videoId}/table-calibration`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "标定读取失败");
  state.tableCalibration = data.calibration || null;
  return state.tableCalibration;
}

async function ensureTableCalibration() {
  try {
    const calibration = await fetchTableCalibration();
    if (calibration) return true;
    openTableCalibrationModal();
    $("tableCalibrationStatus").textContent = "请先完成四角标定，再开始板数分析";
    return false;
  } catch (err) {
    setTrackingNotice("标定读取失败: " + err.message);
    return false;
  }
}

function calibrationCanvasPoint(event) {
  const canvas = $("tableCalibrationCanvas");
  const rect = canvas.getBoundingClientRect();
  return [
    Math.min(canvas.width, Math.max(0, (event.clientX - rect.left) * canvas.width / rect.width)),
    Math.min(canvas.height, Math.max(0, (event.clientY - rect.top) * canvas.height / rect.height)),
  ];
}

function openTableCalibrationModal() {
  if (!state.videoId) return;
  $("tableCalibrationModal").classList.add("modal--open");
  $("tableCalibrationLoading").style.display = "flex";
  $("tableCalibrationStatus").textContent = "读取此视频的标定...";
  fetchTableCalibration()
    .then((calibration) => {
      const total = Math.max(1, Number(state.videoInfo?.total_frames || 1));
      const range = $("tableCalibrationFrameRange");
      range.max = String(total - 1);
      state.calibrationFrame = Math.min(Number(calibration?.frame_index || 0), total - 1);
      state.calibrationCorners = calibration?.corners?.map(point => [Number(point[0]), Number(point[1])]) || [];
      state.calibrationNetPoints = calibration?.net_points?.map(point => [Number(point[0]), Number(point[1])]) || [];
      $("tableCalibrationFrameRange").value = String(state.calibrationFrame);
      $("tableCalibrationFrameInput").value = String(state.calibrationFrame);
      loadTableCalibrationFrame(state.calibrationFrame);
    })
    .catch((err) => {
      $("tableCalibrationLoading").style.display = "none";
      $("tableCalibrationStatus").textContent = "标定读取失败: " + err.message;
    });
}

async function loadTableCalibrationFrame(frameIndex) {
  if (!state.videoId) return;
  const total = Math.max(1, Number(state.videoInfo?.total_frames || 1));
  const nextFrame = Math.min(Math.max(0, Math.round(frameIndex)), total - 1);
  if (nextFrame !== state.calibrationFrame && (state.calibrationCorners.length || state.calibrationNetPoints.length)) {
    state.calibrationCorners = [];
    state.calibrationNetPoints = [];
    state.tableCalibration = null;
    $("tableCalibrationStatus").textContent = "已切换帧，请重新标定点位";
  }
  state.calibrationFrame = nextFrame;
  $("tableCalibrationFrameRange").value = String(nextFrame);
  $("tableCalibrationFrameInput").value = String(nextFrame);
  $("tableCalibrationFrameLabel").textContent = `帧 ${nextFrame} / ${total - 1}`;
  $("tableCalibrationLoading").style.display = "flex";
  const token = ++state.calibrationLoadToken;
  try {
    const res = await fetch(`/api/videos/${state.videoId}/frame?frame_index=${nextFrame}`);
    if (!res.ok) throw new Error("帧加载失败");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => {
      if (token !== state.calibrationLoadToken) return;
      const canvas = $("tableCalibrationCanvas");
      canvas.width = Number(state.videoInfo?.width || image.naturalWidth);
      canvas.height = Number(state.videoInfo?.height || image.naturalHeight);
      const wrap = canvas.parentElement;
      wrap.style.aspectRatio = `${canvas.width} / ${canvas.height}`;
      state.calibrationImage = image;
      drawTableCalibrationFrame();
      URL.revokeObjectURL(url);
      $("tableCalibrationLoading").style.display = "none";
      updateTableCalibrationHint();
      if (state.tableCalibration) {
        $("tableCalibrationStatus").textContent = "已读取此视频标定，可检查或重新保存";
      } else if (!$("tableCalibrationStatus").textContent.includes("已切换帧")) {
        $("tableCalibrationStatus").textContent = "请按顺序点击四角，完成后保存标定";
      }
    };
    image.src = url;
  } catch (err) {
    $("tableCalibrationLoading").style.display = "none";
    $("tableCalibrationStatus").textContent = err.message;
  }
}

function drawTableCalibrationFrame() {
  const canvas = $("tableCalibrationCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.calibrationImage) ctx.drawImage(state.calibrationImage, 0, 0, canvas.width, canvas.height);

  const points = state.calibrationMode === "corners" ? state.calibrationCorners : state.calibrationNetPoints;
  if (state.calibrationCorners.length === 4) {
    ctx.beginPath();
    state.calibrationCorners.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.closePath();
    ctx.strokeStyle = "#64e6a5";
    ctx.lineWidth = Math.max(3, canvas.width / 360);
    ctx.stroke();
  }
  if (state.calibrationNetPoints.length === 2) {
    ctx.beginPath();
    ctx.moveTo(...state.calibrationNetPoints[0]);
    ctx.lineTo(...state.calibrationNetPoints[1]);
    ctx.strokeStyle = "#62c7f5";
    ctx.lineWidth = Math.max(3, canvas.width / 360);
    ctx.stroke();
  }
  points.forEach(([x, y], index) => {
    ctx.beginPath();
    ctx.arc(x, y, Math.max(9, canvas.width / 100), 0, Math.PI * 2);
    ctx.fillStyle = state.calibrationMode === "corners" ? "#64e6a5" : "#62c7f5";
    ctx.fill();
    ctx.fillStyle = "#171719";
    ctx.font = `600 ${Math.max(14, canvas.width / 55)}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(index + 1), x, y);
  });
}

function setCalibrationMode(mode) {
  state.calibrationMode = mode;
  $("tableCalibrationCornersMode").classList.toggle("btn--primary", mode === "corners");
  $("tableCalibrationNetMode").classList.toggle("btn--primary", mode === "net");
  updateTableCalibrationHint();
  drawTableCalibrationFrame();
}

function updateTableCalibrationHint() {
  const count = state.calibrationMode === "corners"
    ? state.calibrationCorners.length
    : state.calibrationNetPoints.length;
  $("tableCalibrationHint").textContent = state.calibrationMode === "corners"
    ? `按左上、右上、右下、左下顺序点击球台四角（${count}/4）`
    : `依次点击中网两端（${count}/2），中网标定可选`;
}

function handleTableCalibrationClick(event) {
  const point = calibrationCanvasPoint(event);
  if (state.calibrationMode === "corners") {
    if (state.calibrationCorners.length >= 4) {
      $("tableCalibrationStatus").textContent = "四角已完成，可切换到中网或保存";
      return;
    }
    state.calibrationCorners.push(point);
  } else {
    if (state.calibrationNetPoints.length >= 2) {
      $("tableCalibrationStatus").textContent = "中网两点已完成，可保存";
      return;
    }
    state.calibrationNetPoints.push(point);
  }
  drawTableCalibrationFrame();
  updateTableCalibrationHint();
  $("tableCalibrationStatus").textContent = "点位已记录，可继续点击或保存";
}

function resetTableCalibrationPoints() {
  state.calibrationCorners = [];
  state.calibrationNetPoints = [];
  state.tableCalibration = null;
  drawTableCalibrationFrame();
  updateTableCalibrationHint();
  $("tableCalibrationStatus").textContent = "点位已清空，请重新标定";
}

async function saveTableCalibration() {
  if (!state.videoId) return;
  if (state.calibrationCorners.length !== 4) {
    $("tableCalibrationStatus").textContent = "必须先按顺序标定四个球台角点";
    return;
  }
  try {
    const res = await fetch(`/api/videos/${state.videoId}/table-calibration`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frame_index: state.calibrationFrame,
        corners: state.calibrationCorners,
        net_points: state.calibrationNetPoints,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "标定保存失败");
    state.tableCalibration = data.calibration;
    $("tableCalibrationStatus").textContent = "标定已保存，可开始板数分析";
  } catch (err) {
    $("tableCalibrationStatus").textContent = "保存失败: " + err.message;
  }
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
    const loaded = state.tracknetAnnotations.length;
    $("tracknetAnnotationStatus").textContent = loaded > 0
      ? `已加载 ${loaded} 个已有标注，可继续标注`
      : "点击球心添加标注";
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

function canvasPointFromClient(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  return [
    Math.min(canvas.width, Math.max(0, (event.clientX - rect.left) * canvas.width / rect.width)),
    Math.min(canvas.height, Math.max(0, (event.clientY - rect.top) * canvas.height / rect.height)),
  ];
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
  const [x, y] = canvasPointFromClient(event, canvas);
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

async function toggleTracking() {
  if (!state.videoId) return;
  if (state.streaming) {
    stopTracking();
  } else {
    if (!(await ensureTableCalibration())) return;
    // 板数追踪不需要先做人框预览，避免误触发 YOLO/躯干分析。
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
  const crossingTimeout = normalizeCrossingTimeoutInput();
  const wsUrl = `${wsBase}/api/ws/${state.videoId}?mode=rally&model_id=${encodeURIComponent(state.selectedModelId)}&playback_speed=${state.playbackSpeed}&no_crossing_timeout_seconds=${crossingTimeout}&token=${encodeURIComponent(localStorage.getItem("pp_token") || "")}`;
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
          $("streamText").textContent = "模型加载中...";
          $("modelProgress").style.display = "flex";
          $("modelProgressFill").style.width = (msg.percent || 0) + "%";
          $("modelProgressPct").textContent = (msg.percent || 0) + "%";
        } else if (msg.status === "preparing") {
          $("loadingText").textContent = msg.message || "准备视频背景...";
          $("streamText").textContent = "准备中...";
          $("streamLoading").style.display = "flex";
          $("modelProgress").style.display = "none";
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
          stopTracking("连接失败");
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
    stopTracking("连接失败");
  };

  ws.onclose = () => {
    if (state.streaming) {
      finishTracking();
    }
  };
}

function stopTracking(statusText = "未连接") {
  state.streaming = false;
  if (state.ws) {
    try { state.ws.send("stop"); } catch {}
    try { state.ws.close(); } catch {}
    state.ws = null;
  }
  resetTrackingUI(statusText);
}

function finishTracking() {
  state.streaming = false;
  state.ws = null;
  resetTrackingUI();
  $("liveBadge").textContent = "DONE";
  $("streamText").textContent = "已完成";
  $("footerState").textContent = "完成";
}

function resetTrackingUI(statusText = "未连接") {
  $("trackBtn").innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>开始追踪`;
  $("streamDot").classList.add("status-dot--idle");
  $("streamLoading").style.display = "none";
  $("streamCanvas").style.display = "none";
  $("trackingProgress").style.display = "none";
  $("streamPlaceholder").style.display = "flex";
  $("streamText").textContent = statusText;
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

function normalizeCrossingTimeoutInput() {
  const input = $("crossingTimeoutInput");
  const value = Math.min(10, Math.max(0.5, Number(input?.value) || 4));
  if (input) input.value = value.toFixed(1);
  return value.toFixed(1);
}

function formatBallSpeedLabel(source) {
  const kmh = Number(source?.ball_speed_kmh);
  if (Number.isFinite(kmh) && kmh > 0) {
    return `${kmh.toFixed(0)} km/h`;
  }
  const px = Number(source?.ball_speed);
  if (Number.isFinite(px) && px > 0) {
    return `${px.toFixed(0)} px/f`;
  }
  return "";
}

function hitMarkPos(hit) {
  const landing = hit?.landing_pos;
  if (Array.isArray(landing) && landing.length >= 2) {
    const x = Number(landing[0]);
    const y = Number(landing[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      return { x, y, kind: "bounce" };
    }
  }
  const table = hit?.table_pos;
  if (hit?.calibrated && Array.isArray(table) && table.length >= 2) {
    const x = Number(table[0]);
    const y = Number(table[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      return { x, y, kind: hit.placement_kind || "net" };
    }
  }
  return null;
}

function renderTableMap(hits) {
  const events = Array.isArray(hits) ? hits : [];
  const marks = events.map((hit, index) => {
    const pos = hitMarkPos(hit);
    if (!pos) return "";
    const x = Math.max(0, Math.min(1, pos.x)) * 100;
    const y = Math.max(0, Math.min(1, pos.y)) * 100;
    const n = Number.isFinite(Number(hit.board_index)) && hit.board_index > 0
      ? hit.board_index
      : index + 1;
    return `<span class="table-map__dot table-map__dot--${pos.kind}" style="left:${x}%;top:${y}%">${n}</span>`;
  }).filter(Boolean);
  if (!marks.length) return "";
  return `<div class="table-map">
    <div class="table-map__surface">
      <div class="table-map__net"></div>
      ${marks.join("")}
    </div>
    <div class="table-map__legend">实心为弹跳落点，空心为过网位置</div>
  </div>`;
}

function updateMetrics(meta) {
  // 实时指标
  $("metricBoards").textContent = meta.board_count || 0;
  const kmh = Number(meta.ball_speed_kmh);
  if (meta.calibrated) {
    const value = Number.isFinite(kmh) ? kmh.toFixed(0) : "—";
    $("metricBallSpeed").innerHTML = `${value}<span class="metric__unit">km/h</span>`;
  } else {
    $("metricBallSpeed").innerHTML = `${Math.round(Number(meta.ball_speed) || 0)}<span class="metric__unit">px/f</span>`;
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

  if (!(await ensureTableCalibration())) return;

  const minBoards = parseInt($("minBoardsInput").value) || 4;
  const crossingTimeout = normalizeCrossingTimeoutInput();

  $("analyzeBtn").disabled = true;
  $("analysisBadge").textContent = "分析中";
  $("analysisText").textContent = "正在进行纯球板数分析并生成剪辑...";
  state.editProjectId = null;
  $("editProjectBtn").disabled = true;
  updateAnalysisProgress(0);
  showAnalysisView();

  try {
    const maxRetries = $("maxRetriesInput")?.value || 5;
    const res = await fetch(`/api/analyze/${state.videoId}?min_boards=${minBoards}&model_id=${encodeURIComponent(state.selectedModelId)}&no_crossing_timeout_seconds=${crossingTimeout}&max_retries=${maxRetries}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "板数分析启动失败");
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

// ---------- 分析进度视图 ----------

const RALLY_STEPS = [
  { key: "load_model", label: "加载 TrackNet 模型" },
  { key: "sample_bg", label: "采样背景帧" },
  { key: "analyze_frames", label: "分析视频帧" },
  { key: "detect_rallies", label: "检测得分回合" },
  { key: "generate_clips", label: "生成剪辑片段" },
  { key: "save_results", label: "保存结果" },
];

function showAnalysisView() {
  // 隐藏其他 viewer 元素
  $("streamPlaceholder").style.display = "none";
  $("streamLoading").style.display = "none";
  $("streamCanvas").style.display = "none";
  $("analysisView").style.display = "flex";
  $("analysisViewFill").style.width = "0%";
  $("analysisViewPct").textContent = "0%";
  $("analysisViewTitle").textContent = "正在分析视频...";
  $("analysisViewSteps").innerHTML = "";
  $("analysisViewDetail").textContent = "";
}

function hideAnalysisView() {
  $("analysisView").style.display = "none";
}

function updateAnalysisView(data) {
  const pct = Math.min(100, Math.max(0, Math.round(data.progress || 0)));
  $("analysisViewFill").style.width = pct + "%";
  $("analysisViewPct").textContent = pct + "%";

  if (data.message) {
    $("analysisViewTitle").textContent = data.message;
  }

  // 渲染步骤
  const steps = data.steps || [];
  if (steps.length > 0) {
    const stepsEl = $("analysisViewSteps");
    stepsEl.innerHTML = "";
    for (const step of steps) {
      const el = document.createElement("div");
      el.className = "analysis-view__step";
      if (step.status === "processing") el.classList.add("analysis-view__step--active");
      if (step.status === "done") el.classList.add("analysis-view__step--done");

      const icon = document.createElement("span");
      icon.className = "analysis-view__step-icon";
      if (step.status === "done") icon.textContent = "✓";
      else if (step.status === "processing") icon.textContent = "◐";
      else icon.textContent = "○";

      const label = document.createElement("span");
      label.className = "analysis-view__step-label";
      label.textContent = step.label || step.name || "";

      el.append(icon, label);
      stepsEl.appendChild(el);
    }
  }

  // 重试信息
  if (data.retry_info) {
    $("analysisViewDetail").textContent = data.retry_info;
  } else if (data.error) {
    $("analysisViewDetail").textContent = data.error;
  }
}

function pollResult(taskId) {
  if (state.pollTimer) clearInterval(state.pollTimer);

  state.pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/result/${taskId}`);
      const data = await res.json();

      if (data.progress !== undefined) updateAnalysisProgress(data.progress);
      if (data.message) $("analysisText").textContent = data.message;
      updateAnalysisView(data);

      if (data.status === "completed") {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        state.analysisData = data;
        state.editProjectId = data.project_id || null;
        const hasClips = data.clips && data.clips.length > 0;
        $("editProjectBtn").disabled = !state.editProjectId || !hasClips;

        // Results created by older server tasks may not have a project ID yet.
        if (!state.editProjectId && hasClips) {
          try {
            const projectResponse = await fetch(`/api/edit-projects/from-analysis/${encodeURIComponent(taskId)}`, { method: "POST" });
            const projectData = await projectResponse.json();
            if (projectResponse.ok) {
              state.editProjectId = projectData.project_id || null;
              $("editProjectBtn").disabled = !state.editProjectId;
            }
          } catch {}
        }

        updateAnalysisProgress(100);
        $("analysisBadge").textContent = "完成";
        const clipCount = (data.clips || []).length;
        $("analysisText").textContent = `检测到 ${data.total_rallies || 0} 个得分段, ${clipCount} 个剪辑片段`;
        $("analyzeBtn").disabled = false;
        $("actionAllBtn").disabled = !hasClips;
        hideAnalysisView();

        renderRallies(data.rallies, data.clips);
        $("segmentCountBadge").textContent = data.total_rallies;
      } else if (data.status === "failed") {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        $("analysisText").textContent = "分析失败: " + data.error;
        $("analysisBadge").textContent = "失败";
        state.editProjectId = null;
        $("editProjectBtn").disabled = true;
        $("analyzeBtn").disabled = false;
        $("analysisViewTitle").textContent = "分析失败";
        if (data.error) $("analysisViewDetail").textContent = data.error;
      }
    } catch {}
  }, 2000);
}

function openEditProject() {
  if (!state.editProjectId) return;
  window.location.href = `/editor/${encodeURIComponent(state.editProjectId)}`;
}

// ========== 得分段列表 ==========

function renderRallies(rallies, clips) {
  const list = $("rallyList");
  list.innerHTML = "";

  if (!rallies || rallies.length === 0) {
    list.innerHTML = '<div class="rally-list--empty">未检测到符合阈值的得分段</div>';
    return;
  }

  rallies.forEach((rally, i) => {
    const item = document.createElement("div");
    item.className = "rally-item";
    item.innerHTML = `
      <div class="rally-item__index">${String(i + 1).padStart(2, "0")}</div>
      <div class="rally-item__info">
        <div class="rally-item__time">${rally.start_time.toFixed(1)}s — ${rally.end_time.toFixed(1)}s</div>
        <div class="rally-item__duration">时长 ${rally.duration.toFixed(1)}s · ${rally.hit_events?.length || 0} 次击球</div>
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
  $("rallyModalTitle").textContent = `得分段 ${rally.index} · ${rally.board_count}板 · ${rally.duration.toFixed(1)}s`;

  let html = `
    <div class="rally-detail__summary">
      <div class="detail-stat"><div class="detail-stat__label">开始</div><div class="detail-stat__value">${rally.start_time}s</div></div>
      <div class="detail-stat"><div class="detail-stat__label">结束</div><div class="detail-stat__value">${rally.end_time}s</div></div>
      <div class="detail-stat"><div class="detail-stat__label">时长</div><div class="detail-stat__value">${rally.duration}s</div></div>
      <div class="detail-stat"><div class="detail-stat__label">板数</div><div class="detail-stat__value">${rally.board_count}</div></div>
    </div>
    ${renderTableMap(rally.hit_events)}
  `;

  if (rally.hit_events && rally.hit_events.length > 0) {
    html += `<div class="rally-detail__events"><div class="events__title">击球事件</div><div class="events__timeline">`;
    rally.hit_events.forEach((hit) => {
      const side = hit.hitter_side || hit.side;
      const hitType = hit.hit_type || hit.type;
      const sideLabel = side === "left" ? "左侧" : side === "right" ? "右侧" : "半区未知";
      const typeLabel = hitType === "drive" ? "撞击" : hitType === "spin" ? "摩擦" : "未知";
      const typeClass = hitType === "drive" ? "event--drive" : hitType === "spin" ? "event--spin" : "event--unknown";
      const boardIndex = Number.isFinite(Number(hit.board_index)) && Number(hit.board_index) > 0
        ? Number(hit.board_index)
        : null;
      const angles = hit.arm_angles || {};
      const rightElbow = Number(angles.right_elbow_angle);
      const leftElbow = Number(angles.left_elbow_angle);
      const elbowLabel = Number.isFinite(rightElbow)
        ? `右肘 ${rightElbow.toFixed(0)}°`
        : Number.isFinite(leftElbow)
          ? `左肘 ${leftElbow.toFixed(0)}°`
          : "";
      const speedLabel = formatBallSpeedLabel(hit);
      const placement = hit.placement || "";
      const parts = [
        boardIndex ? `第 ${boardIndex} 板` : "",
        sideLabel,
        speedLabel,
        placement,
        elbowLabel,
      ].filter(Boolean);
      html += `
        <div class="event-item ${typeClass}">
          <div class="event-item__num">${boardIndex ? String(boardIndex).padStart(2, "0") : "–"}</div>
          <div class="event-item__info">
            <span class="event-item__time">${Number(hit.timestamp).toFixed(1)}s</span>
            <span class="event-item__side">${parts.join(" · ")}</span>
          </div>
          <div class="event-item__type">${typeLabel}</div>
        </div>
      `;
    });
    html += `</div></div>`;
  } else {
    html += `<div class="rally-detail__events"><div class="events__empty">这一段还没有击球事件</div></div>`;
  }

  if (clipFilename) {
    html += `
      <div class="rally-detail__actions">
        <button class="btn btn--primary" id="playClipBtn">播放剪辑</button>
        <a class="btn" href="/api/clips/${clipFilename}?download=true" download>下载片段</a>
        <button class="btn" id="actionClipBtn">动作分析此得分段</button>
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

  $("modalTitle").textContent = `得分段 ${rally.index} · ${rally.board_count}板`;
  player.src = `/api/clips/${clipFilename}`;
  player.load();

  // 显式调用 play()（浏览器可能阻止 autoplay，需 catch）
  player.play().catch(() => {
    // 自动播放被阻止，用户需手动点击播放按钮
  });

  let detailsHtml = `
    <div class="modal__detail-row"><span>时间范围</span><strong>${rally.start_time}s — ${rally.end_time}s</strong></div>
    <div class="modal__detail-row"><span>时长</span><strong>${rally.duration}s</strong></div>
    <div class="modal__detail-row"><span>板数</span><strong>${rally.board_count}</strong></div>
    <div class="modal__detail-row"><span>击球次数</span><strong>${rally.hit_events?.length || 0}</strong></div>
    <div class="modal__detail-actions">
      <a class="btn" href="/api/clips/${clipFilename}?download=true" download>下载片段</a>
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

function updateActionProgress(progress, message) {
  const value = Math.min(100, Math.max(0, Math.round(Number(progress) || 0)));
  $("actionProgressBar").style.width = `${value}%`;
  $("actionProgressPercent").textContent = `${value}%`;
  if (message) $("actionProgressText").textContent = message;
}

async function startActionAnalysis({ clipFilename = null, allClips = false, source = null } = {}) {
  if (!state.videoId) return;
  $("actionModal").classList.add("modal--open");
  $("actionSourcePanel").style.display = "none";
  $("actionProgressPanel").style.display = "flex";
  $("actionReportPanel").style.display = "none";
  updateActionProgress(0, allClips || clipFilename ? "正在准备得分段动作..." : "正在准备动作分析...");

  const params = new URLSearchParams();
  if (clipFilename) params.set("clip_filename", clipFilename);
  if (allClips) params.set("all_clips", "true");
  if (source) params.set("source", source);

  try {
    const query = params.toString() ? `?${params.toString()}` : "";
    const res = await fetch(`/api/action/analyze/${state.videoId}${query}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "动作分析启动失败");
    state.actionTaskId = data.task_id;
    updateActionProgress(data.progress, data.message);
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
      if (data.progress !== undefined) updateActionProgress(data.progress, data.message);
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

const ACTION_ARM_LABELS = {
  left_elbow_angle: "左肘角度",
  right_elbow_angle: "右肘角度",
  left_shoulder_angle: "左肩角度",
  right_shoulder_angle: "右肩角度",
};

const ACTION_BODY_LABELS = {
  torso_lean_angle: "躯干倾斜角",
  torso_length: "躯干长度",
  shoulder_width: "肩宽",
  hip_width: "髋宽",
};

const ACTION_BODY_UNITS = {
  torso_lean_angle: "°",
};

const ACTION_ARM_UNITS = {
  left_elbow_angle: "°",
  right_elbow_angle: "°",
  left_shoulder_angle: "°",
  right_shoulder_angle: "°",
};

function escapeActionHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function getFiniteActionNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatActionNumber(value, digits = 1) {
  const number = getFiniteActionNumber(value);
  return number === null ? "暂无" : number.toFixed(digits);
}

function formatActionMetricLabel(key, labels) {
  if (labels[key]) return labels[key];
  return String(key)
    .replace(/_angle$/, "角度")
    .replace(/_length$/, "长度")
    .replace(/_width$/, "宽度")
    .replace(/_/g, " ");
}

function getActionMetricEntries(metrics, labels) {
  return Object.entries(metrics || {})
    .map(([key, value]) => ({
      key,
      label: formatActionMetricLabel(key, labels),
      value: getFiniteActionNumber(value),
    }))
    .filter((metric) => metric.value !== null);
}

function getActionCoverage(analysis) {
  const total = getFiniteActionNumber(analysis?.frames_analyzed);
  const posed = getFiniteActionNumber(analysis?.posed_frames);
  if (total && posed !== null) return Math.min(1, Math.max(0, posed / total));
  const frames = Array.isArray(analysis?.frames) ? analysis.frames : null;
  if (!total || !frames) return null;
  const visibleFrames = frames.filter((frame) => (
    Array.isArray(frame?.persons) && frame.persons.length > 0
  )).length;
  return Math.min(1, visibleFrames / total);
}

function getActionPersonCoverage(person, analysis) {
  const total = getFiniteActionNumber(analysis?.frames_analyzed);
  const tracked = getFiniteActionNumber(person?.frame_count);
  if (!total || tracked === null) return null;
  return Math.min(1, Math.max(0, tracked / total));
}

function getActionIdentity(person, index) {
  const confidence = Math.min(
    1,
    Math.max(0, getFiniteActionNumber(person?.identity_confidence) ?? 0),
  );
  const rawIdentity = String(person?.identity || "").trim();
  const confirmed = rawIdentity && rawIdentity !== "unknown" && confidence >= 0.6;
  const rawTrackId = String(person?.track_id || "").replace(/^track_/, "");
  return {
    label: confirmed ? rawIdentity : "身份未确认",
    trackLabel: `跟踪 ${rawTrackId || index + 1}`,
    confidence,
    status: confirmed ? "已确认" : "待确认",
  };
}

function getActionSummary(analyses) {
  const items = Array.isArray(analyses) ? analyses : [];
  const totalFrames = items.reduce(
    (sum, item) => sum + (getFiniteActionNumber(item?.frames_analyzed) || 0),
    0,
  );
  const trackedIds = new Set(items.flatMap((item) => (
    (item?.persons || []).map((person) => person?.track_id).filter(Boolean)
  )));
  const coverageValues = items.map(getActionCoverage).filter((value) => value !== null);
  const hitCount = items.reduce(
    (sum, item) => sum + (Array.isArray(item?.hit_events) ? item.hit_events.length : 0),
    0,
  );
  return {
    clipCount: items.length,
    totalFrames,
    trackedCount: trackedIds.size,
    hitCount,
    coverage: coverageValues.length
      ? coverageValues.reduce((sum, value) => sum + value, 0) / coverageValues.length
      : null,
  };
}

function renderActionMetricList(metrics, labels, units = {}) {
  const entries = getActionMetricEntries(metrics, labels);
  if (!entries.length) return `<div class="action-report__no-data">暂无有效数据</div>`;
  return entries.map((metric) => `
    <div class="action-metric">
      <span class="action-metric__label">${escapeActionHtml(metric.label)}</span>
      <strong class="action-metric__value">${formatActionNumber(metric.value)}${units[metric.key] || ""}</strong>
    </div>
  `).join("");
}

function renderActionPersonCard(person, index, analysis) {
  const identity = getActionIdentity(person, index);
  const coverage = getActionPersonCoverage(person, analysis);
  const confidencePercent = Math.round(identity.confidence * 100);
  const confidenceClass = identity.confidence >= 0.8
    ? "high"
    : identity.confidence >= 0.6 ? "medium" : "low";
  return `
    <article class="action-person-card">
      <div class="action-person-card__head">
        <div>
          <div class="action-person-card__identity">${escapeActionHtml(identity.label)}</div>
          <div class="action-person-card__track">${escapeActionHtml(identity.trackLabel)} · ${escapeActionHtml(identity.status)}</div>
        </div>
        <span class="action-confidence action-confidence--${confidenceClass}">${confidencePercent}%</span>
      </div>
      <div class="action-person-card__meta">
        <span>跟踪帧数 <strong>${formatActionNumber(person?.frame_count, 0)}</strong></span>
        <span>片段覆盖 <strong>${coverage === null ? "暂无" : `${Math.round(coverage * 100)}%`}</strong></span>
      </div>
      <div class="action-person-card__section-title">手臂角度</div>
      <div class="action-metrics">${renderActionMetricList(person?.average_arm_angles, ACTION_ARM_LABELS, ACTION_ARM_UNITS)}</div>
      <details class="action-person-card__details">
        <summary>查看身体指标</summary>
        <div class="action-metrics">${renderActionMetricList(person?.average_body_metrics, ACTION_BODY_LABELS, ACTION_BODY_UNITS)}</div>
      </details>
    </article>
  `;
}

function renderActionComparison(persons) {
  const candidates = (persons || []).slice(0, 2).map((person, index) => ({
    label: getActionIdentity(person, index).trackLabel,
    metrics: person?.average_arm_angles || {},
  }));
  if (candidates.length < 2) return "";
  const keys = Object.keys(ACTION_ARM_LABELS).filter((key) => (
    candidates.some((item) => getFiniteActionNumber(item.metrics[key]) !== null)
  ));
  if (!keys.length) return "";

  return `
    <section class="action-comparison">
      <div class="action-report__section-heading">手臂角度对比 <span>仅用于同片段相对比较</span></div>
      <div class="action-comparison__rows">
        ${keys.map((key) => {
          const values = candidates.map((item) => getFiniteActionNumber(item.metrics[key]) || 0);
          const scale = Math.max(...values, 1);
          return `<div class="action-comparison__row">
            <span class="action-comparison__label">${escapeActionHtml(ACTION_ARM_LABELS[key])}</span>
            ${candidates.map((item, index) => {
              const value = getFiniteActionNumber(item.metrics[key]);
              const width = value === null ? 0 : Math.max(4, Math.min(100, (Math.abs(value) / scale) * 100));
              return `<div class="action-comparison__value action-comparison__value--${index}">
                <span>${escapeActionHtml(item.label)} ${value === null ? "暂无" : `${formatActionNumber(value)}°`}</span>
                <span class="action-comparison__bar"><i style="width:${width}%"></i></span>
              </div>`;
            }).join("")}
          </div>`;
        }).join("")}
      </div>
    </section>
  `;
}

function renderActionHitEvents(hits) {
  const events = Array.isArray(hits) ? hits : [];
  if (!events.length) {
    return `<div class="action-report__empty-state action-report__empty-state--compact"><strong>没有对齐的击球事件</strong><span>下面是这段的平均姿态，供参考。</span></div>`;
  }
    return `<div class="rally-detail__events"><div class="events__title">击球事件</div>${renderTableMap(events)}<div class="events__timeline">${
    events.map((hit) => {
      const side = hit.hitter_side || hit.side;
      const hitType = hit.hit_type || hit.type;
      const sideLabel = side === "left" ? "左侧" : side === "right" ? "右侧" : "半区未知";
      const typeLabel = hitType === "drive" ? "撞击" : hitType === "spin" ? "摩擦" : "未知";
      const typeClass = hitType === "drive" ? "event--drive" : hitType === "spin" ? "event--spin" : "event--unknown";
      const boardIndex = Number(hit.board_index);
      const poseAngles = hit.pose?.arm_angles || hit.arm_angles || {};
      const rightElbow = Number(poseAngles.right_elbow_angle);
      const leftElbow = Number(poseAngles.left_elbow_angle);
      const elbowLabel = Number.isFinite(rightElbow)
        ? `右肘 ${rightElbow.toFixed(0)}°`
        : Number.isFinite(leftElbow)
          ? `左肘 ${leftElbow.toFixed(0)}°`
          : "";
      const poseTrack = hit.pose?.track_id ? String(hit.pose.track_id).replace(/^track_/, "") : "";
      const parts = [
        Number.isFinite(boardIndex) && boardIndex > 0 ? `第 ${boardIndex} 板` : "",
        sideLabel,
        formatBallSpeedLabel(hit),
        hit.placement || "",
        elbowLabel,
        poseTrack ? `跟踪 ${poseTrack}` : "",
      ].filter(Boolean);
      return `<div class="event-item ${typeClass}">
        <div class="event-item__num">${Number.isFinite(boardIndex) && boardIndex > 0 ? String(boardIndex).padStart(2, "0") : "–"}</div>
        <div class="event-item__info">
          <span class="event-item__time">${Number(hit.timestamp).toFixed(1)}s</span>
          <span class="event-item__side">${parts.map(escapeActionHtml).join(" · ")}</span>
        </div>
        <div class="event-item__type">${escapeActionHtml(typeLabel)}</div>
      </div>`;
    }).join("")
  }</div></div>`;
}

function renderActionReport(data) {
  const modal = $("actionModal");
  if (modal) modal.classList.add("modal--open");
  $("actionModalTitle").textContent = "动作分析报告";
  $("actionProgressPanel").style.display = "none";
  $("actionSourcePanel").style.display = "none";
  $("actionReportPanel").style.display = "block";

  const analyses = Array.isArray(data?.analyses) ? data.analyses : [];
  if (!analyses.length) {
    $("actionReport").innerHTML = `<div class="action-report__empty-state"><strong>没有可展示的动作分析结果</strong><span>本次任务没有返回可用片段。</span></div>`;
    return;
  }

  const summary = getActionSummary(analyses);
  let html = `
    <div class="action-report__overview">
      <div class="action-report__overview-title"><span>训练动作概览</span><small>按得分段击球事件对齐</small></div>
      <div class="action-report__overview-grid">
        <div><span>分析片段</span><strong>${summary.clipCount}</strong></div>
        <div><span>击球事件</span><strong>${summary.hitCount}</strong></div>
        <div><span>跟踪对象</span><strong>${summary.trackedCount}</strong></div>
        <div><span>平均姿态覆盖</span><strong>${summary.coverage === null ? "暂无" : `${Math.round(summary.coverage * 100)}%`}</strong></div>
      </div>
    </div>
  `;

  analyses.forEach((analysis, analysisIndex) => {
    const coverage = getActionCoverage(analysis);
    const persons = Array.isArray(analysis?.persons) ? analysis.persons : [];
    const rawFrames = Array.isArray(analysis?.frames) ? analysis.frames : [];
    html += `<section class="action-report__source">
      <header class="action-report__source-head">
        <div><span class="action-report__source-index">片段 ${analysisIndex + 1}</span><h3>${escapeActionHtml(analysis?.source || "未命名视频")}</h3></div>
        <div class="action-report__source-meta"><span>${formatActionNumber(analysis?.frames_analyzed, 0)} 帧</span><span>${formatActionNumber(analysis?.fps)} fps</span><span>姿态覆盖 ${coverage === null ? "暂无" : `${Math.round(coverage * 100)}%`}</span></div>
      </header>`;
    html += renderActionHitEvents(analysis?.hit_events);
    if (persons.length) {
      html += `<div class="action-report__people">${persons.map((person, index) => renderActionPersonCard(person, index, analysis)).join("")}</div>`;
      html += renderActionComparison(persons);
    } else if (!(analysis?.hit_events || []).length) {
      html += `<div class="action-report__empty-state action-report__empty-state--compact"><strong>未检测到稳定的人体姿态</strong><span>请检查视频画面是否清晰，或尝试分析更短的片段。</span></div>`;
    }
    if (rawFrames.length && rawFrames.length <= 30) {
      html += `<details class="action-report__raw"><summary>查看逐帧原始数据 <span>${rawFrames.length} 帧</span></summary><pre>${escapeActionHtml(JSON.stringify(rawFrames, null, 2))}</pre></details>`;
    }
    html += `</section>`;
  });
  html += `<p class="action-report__hint">击球侧来自过网方向；肘角来自过网时刻附近的姿态。身份未确认时只用跟踪编号区分。</p>`;
  $("actionReport").innerHTML = html;
}

// ========== 启动 ==========

init();
