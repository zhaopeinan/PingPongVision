/**
 * Workbench — 工作台前端逻辑
 */

const $ = (id) => document.getElementById(id);

// ---------- 状态 ----------
let currentFilter = "";

// ---------- 初始化 ----------
async function init() {
  loadUserInfo();
  loadDevice();
  pollDeviceMetrics();  // 页面加载时立即拉一次 GPU 指标
  loadStats();
  loadHistory();
  loadVideos();
  loadProjects();
  bindEvents();
  checkAdmin();
  startPolling();
}

let _pollTimer = null;

function startPolling() {
  if (_pollTimer) clearInterval(_pollTimer);
  _pollTimer = setInterval(async () => {
    // 检查是否有进行中的任务
    try {
      const res = await fetch("/api/history?status=processing&limit=1");
      const data = await res.json();
      const hasRunning = data.total > 0;

      if (hasRunning) {
        loadHistory();
        loadStats();
        if (!_hasRunningTask) {
          _hasRunningTask = true;
          startDevicePolling();
        }
      } else {
        if (_hasRunningTask) {
          _hasRunningTask = false;
          stopDevicePolling();
          loadHistory();
          loadStats();
        }
      }
    } catch {}
  }, 5000);
}

function loadUserInfo() {
  const user = window.PPAuth?.getUser();
  if (user) {
    $("userName").textContent = user.username;
    $("userPill").hidden = false;
  }
}

function checkAdmin() {
  const user = window.PPAuth?.getUser();
  if (user && user.role === "admin") {
    $("adminPanel").hidden = false;
    loadUsers();
  }
}

// ---------- 用户管理 ----------
async function loadUsers() {
  const container = $("userList");
  try {
    const res = await fetch("/api/admin/users");
    const data = await res.json();
    if (!data.users || data.users.length === 0) {
      container.innerHTML = '<div class="wb__empty">暂无用户</div>';
      return;
    }
    const currentUser = window.PPAuth?.getUser();
    container.innerHTML = data.users.map(u => {
      const roleLabel = u.role === "admin" ? "管理员" : "分析师";
      const statusLabel = u.active ? "启用" : "禁用";
      const statusClass = u.active ? "wb__badge--completed" : "wb__badge--failed";
      const time = formatTime(u.created_at);
      const isSelf = currentUser && currentUser.id === u.id;
      return `
        <div class="wb__user-row" data-user-id="${u.id}">
          <div class="wb__user-info">
            <div class="wb__user-name">${u.username}${isSelf ? ' (你)' : ''}</div>
            <div class="wb__user-meta">
              <span class="wb__badge ${statusClass}">${statusLabel}</span>
              ${roleLabel} · ${time}
            </div>
          </div>
          <div class="wb__user-actions">
            <button class="wb__user-btn" onclick="editUser(${u.id}, '${u.username}', '${u.role}')">编辑</button>
            ${u.active
              ? `<button class="wb__user-btn" onclick="toggleUser(${u.id}, false)">禁用</button>`
              : `<button class="wb__user-btn" onclick="toggleUser(${u.id}, true)">启用</button>`
            }
            ${!isSelf ? `<button class="wb__user-btn wb__user-btn--danger" onclick="deleteUser(${u.id}, '${u.username}')">删除</button>` : ''}
          </div>
        </div>
      `;
    }).join("");
  } catch (e) {
    container.innerHTML = '<div class="wb__empty">加载失败</div>';
  }
}

// ---------- 任务详情弹窗 ----------

let _detailPollTimer = null;
let _detailTaskId = null;

window.showTaskDetail = async function(taskId, event) {
  if (event) event.stopPropagation();
  _detailTaskId = taskId;
  const modal = $("taskDetailModal");
  modal.hidden = false;
  await refreshTaskDetail();
  // 开始 1 秒轮询
  if (_detailPollTimer) clearInterval(_detailPollTimer);
  _detailPollTimer = setInterval(refreshTaskDetail, 1000);
};

window.closeTaskDetail = function() {
  const modal = $("taskDetailModal");
  modal.hidden = true;
  if (_detailPollTimer) {
    clearInterval(_detailPollTimer);
    _detailPollTimer = null;
  }
  _detailTaskId = null;
};

async function refreshTaskDetail() {
  if (!_detailTaskId) return;
  try {
    const res = await fetch(`/api/result/${_detailTaskId}`);
    const data = await res.json();
    // 训练任务额外获取内存中的训练指标
    if (_detailTaskId.startsWith("train_")) {
      try {
        const trainRes = await fetch(`/api/tracknet/train/${_detailTaskId}`);
        if (trainRes.ok) {
          const trainData = await trainRes.json();
          data.epoch_history = trainData.epoch_history || [];
          data.metrics = trainData.metrics || {};
          data.checkpoint_path = trainData.checkpoint_path;
        }
      } catch {}
    }
    renderTaskDetail(data);
    // 如果任务已完成/失败/取消，停止轮询
    if (data.status !== "processing") {
      if (_detailPollTimer) {
        clearInterval(_detailPollTimer);
        _detailPollTimer = null;
      }
    }
  } catch {}
}

function renderTaskDetail(data) {
  const el = $("taskDetailBody");
  const titleEl = $("taskDetailTitle");
  const modeLabels = { rally: "得分段", action: "动作", tracknet_train: "模型微调" };
  const modeLabel = modeLabels[data.mode] || data.mode;
  titleEl.textContent = `${modeLabel} · ${data.video_filename || data.video_id || data.task_id}`;

  const statusLabels = { completed: "完成", processing: "进行中", failed: "失败", cancelled: "已取消" };
  const statusColors = { completed: "var(--data-green)", processing: "var(--accent)", failed: "var(--data-red)", cancelled: "var(--text-muted)" };
  const statusLabel = statusLabels[data.status] || data.status;
  const statusColor = statusColors[data.status] || "var(--text-secondary)";

  const steps = data.steps || [];
  const stepsHtml = steps.map(s => {
    const icons = { completed: "✓", processing: "◐", pending: "○", failed: "✕" };
    const colors = { completed: "var(--data-green)", processing: "var(--accent)", pending: "var(--text-muted)", failed: "var(--data-red)" };
    const icon = icons[s.status] || "○";
    const color = colors[s.status] || "var(--text-muted)";
    const pctText = s.status === "processing" && s.progress > 0 ? ` ${s.progress}%` : "";
    return `
      <div class="wb__step-item wb__step--${s.status}">
        <span class="wb__step-icon" style="color:${color}">${icon}</span>
        <span class="wb__step-name">${s.name}${pctText}</span>
      </div>
    `;
  }).join("");

  const pct = data.progress || 0;

  // 训练指标
  let metricsHtml = "";
  if (data.mode === "tracknet_train") {
    const hist = data.epoch_history || [];
    const metrics = data.metrics || {};
    const lr = hist.length > 0 ? hist[hist.length - 1]?.learning_rate : null;
    const device = metrics.device || "";

    if (hist.length > 0) {
      // 有 epoch 数据，显示完整指标
      const allLosses = [...hist.map(e => e.train_loss).filter(v => v != null), ...hist.map(e => e.validation_loss).filter(v => v != null)];
      const maxLoss = Math.max(...allLosses, 0.001);
      const minLoss = Math.min(...allLosses, 0);
      const range = maxLoss - minLoss || 1;

      const epochRows = hist.map(e => {
        const tl = e.train_loss != null ? e.train_loss.toFixed(4) : "—";
        const vl = e.validation_loss != null ? e.validation_loss.toFixed(4) : "—";
        const dr = e.positive_detection_rate != null ? `${(e.positive_detection_rate * 100).toFixed(0)}%` : "—";
        const tlBar = e.train_loss != null ? Math.max(2, ((e.train_loss - minLoss) / range) * 100) : 0;
        const vlBar = e.validation_loss != null ? Math.max(2, ((e.validation_loss - minLoss) / range) * 100) : 0;
        return `
          <div class="wb__epoch-row">
            <span class="wb__epoch-num">E${e.epoch}</span>
            <div class="wb__epoch-bars">
              <div class="wb__epoch-bar"><div class="wb__epoch-bar-fill wb__epoch-bar-fill--train" style="width:${tlBar}%"></div></div>
              <div class="wb__epoch-bar"><div class="wb__epoch-bar-fill wb__epoch-bar-fill--val" style="width:${vlBar}%"></div></div>
            </div>
            <span class="wb__epoch-val">train ${tl}</span>
            <span class="wb__epoch-val">val ${vl}</span>
            <span class="wb__epoch-val">检测率 ${dr}</span>
          </div>
        `;
      }).join("");

      metricsHtml = `
        <div class="wb__detail-metrics">
          <div class="wb__detail-metrics-header">
            <span>训练指标</span>
            ${lr ? `<span class="wb__detail-lr">学习率 ${lr.toExponential(2)}</span>` : ""}
          </div>
          <div class="wb__epoch-list">${epochRows}</div>
          <div class="wb__epoch-legend">
            <span class="wb__epoch-legend-item"><span class="wb__legend-dot wb__legend-dot--train"></span>训练损失</span>
            <span class="wb__epoch-legend-item"><span class="wb__legend-dot wb__legend-dot--val"></span>验证损失</span>
          </div>
        </div>
      `;
    } else {
      // 还没有 epoch 数据，显示等待信息
      metricsHtml = `
        <div class="wb__detail-metrics">
          <div class="wb__detail-metrics-header">
            <span>训练指标</span>
            <span class="wb__detail-lr">${device ? `设备 ${device}` : ""}</span>
          </div>
          <div class="wb__detail-metrics-waiting">等待第一轮训练完成...</div>
        </div>
      `;
    }
  }

  // 训练完成后显示保存/放弃按钮
  let trainActionsHtml = "";
  if (data.mode === "tracknet_train" && data.status === "completed") {
    trainActionsHtml = `
      <div class="wb__detail-train-actions">
        <button class="btn btn--primary btn--sm" onclick="saveTrainModel('${data.task_id}', event)">保存权重</button>
        <button class="btn btn--sm" onclick="discardTrainModel('${data.task_id}', event)">放弃</button>
      </div>
    `;
  }

  el.innerHTML = `
    <div class="wb__detail-status">
      <span class="wb__detail-status-dot" style="background:${statusColor}"></span>
      <span style="color:${statusColor};font-weight:600">${statusLabel}</span>
      <span class="wb__detail-progress">${pct}%</span>
    </div>
    <div class="wb__detail-progress-bar">
      <div class="wb__detail-progress-fill" style="width:${pct}%;background:${statusColor}"></div>
    </div>
    <div class="wb__detail-message">${data.message || ""}</div>
    ${metricsHtml}
    ${trainActionsHtml}
    <div class="wb__detail-steps">${stepsHtml}</div>
    ${data.error ? `<div class="wb__detail-error">${data.error}</div>` : ""}
  `;
}

window.cancelTask = async function(taskId, event) {
  if (event) event.stopPropagation();
  if (!confirm("确定停止该任务吗？")) return;
  try {
    const res = await fetch(`/api/tasks/${taskId}/cancel`, { method: "POST" });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.detail || "停止失败");
    }
    loadHistory();
    loadStats();
  } catch (e) {
    alert(e.message);
  }
};

window.deleteTask = async function(taskId, event) {
  if (event) event.stopPropagation();
  if (!confirm("确定删除该记录吗？")) return;
  try {
    const res = await fetch(`/api/tasks/${taskId}`, { method: "DELETE" });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.detail || "删除失败");
    }
    loadHistory();
    loadStats();
  } catch (e) {
    alert(e.message);
  }
};

window.saveTrainModel = async function(jobId, event) {
  if (event) event.stopPropagation();
  const displayName = prompt("请输入模型名称（可留空使用默认名）：", "");
  if (displayName === null) return; // 用户取消
  try {
    const res = await fetch(`/api/tracknet/train/${jobId}/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(displayName ? { display_name: displayName } : {}),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "保存失败");
    refreshTaskDetail();
    loadHistory();
  } catch (e) {
    alert(e.message);
  }
};

window.discardTrainModel = async function(jobId, event) {
  if (event) event.stopPropagation();
  if (!confirm("确定放弃该训练结果吗？")) return;
  try {
    const res = await fetch(`/api/tracknet/train/${jobId}`, { method: "DELETE" });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.error || "放弃失败");
    }
    refreshTaskDetail();
    loadHistory();
  } catch (e) {
    alert(e.message);
  }
};

window.rerenderProject = async function(projectId, event) {
  if (event) event.stopPropagation();
  if (!confirm("确定重新渲染该项目吗？")) return;
  try {
    const res = await fetch(`/api/edit-projects/${projectId}/render`, { method: "POST" });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.error || data.detail || "渲染启动失败");
    }
    loadHistory();
  } catch (e) {
    alert(e.message);
  }
};

window.deleteVideo = async function(videoId, event) {
  if (event) event.stopPropagation();
  if (!confirm("删除视频将同时删除其所有分析任务和剪辑文件，确定删除吗？")) return;
  try {
    const res = await fetch(`/api/videos/${videoId}`, { method: "DELETE" });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.detail || "删除失败");
    }
    loadVideos();
    loadHistory();
    loadStats();
  } catch (e) {
    alert(e.message);
  }
};

window.editUser = function(userId, username, role) {
  $("editUserId").value = userId;
  $("formUsername").value = username;
  $("formUsername").disabled = true;
  $("formPassword").value = "";
  $("formPassword").required = false;
  $("passwordHint").textContent = "(留空则不修改)";
  $("formRole").value = role;
  $("userModalTitle").textContent = "编辑用户";
  $("userModalSubmit").textContent = "保存";
  $("userModal").hidden = false;
};

window.toggleUser = async function(userId, active) {
  try {
    const res = await fetch(`/api/admin/users/${userId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ active }),
    });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.detail || "操作失败");
    }
    loadUsers();
  } catch (e) {
    alert(e.message);
  }
};

window.deleteUser = async function(userId, username) {
  if (!confirm(`确定删除用户 "${username}" 吗？`)) return;
  try {
    const res = await fetch(`/api/admin/users/${userId}`, { method: "DELETE" });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.detail || "删除失败");
    }
    loadUsers();
  } catch (e) {
    alert(e.message);
  }
};

// ---------- 设备信息 ----------
let _devicePollTimer = null;
let _hasRunningTask = false;

async function loadDevice() {
  try {
    const res = await fetch("/api/device");
    const data = await res.json();
    renderDevice(data);
    // 有 runtime 指标时立即渲染
    if (data.runtime) renderDeviceMetrics(data.runtime, data.device_type);
  } catch {
    $("deviceText").textContent = "设备不可用";
  }
}

function renderDevice(data) {
  const pill = $("devicePill");
  const text = $("deviceText");
  const dot = pill.querySelector(".wb__dot");
  if (data.device_type === "cuda" || data.device_type === "tensorrt") {
    dot.className = "wb__dot wb__dot--ok";
    text.textContent = `GPU · ${data.device_name || "CUDA"}`;
  } else if (data.device_type === "mps") {
    dot.className = "wb__dot wb__dot--ok";
    text.textContent = `MPS · ${data.device_name || "Apple Silicon"}`;
  } else {
    dot.className = "wb__dot wb__dot--err";
    text.textContent = "CPU 模式";
  }
}

function renderDeviceMetrics(rt, deviceType) {
  const el = $("deviceMetrics");
  if (!el) return;
  if (!rt || !rt.available) {
    el.textContent = rt && rt.message ? rt.message : "";
    el.hidden = !el.textContent;
    return;
  }
  const parts = [];
  if (rt.gpu_utilization_percent != null)
    parts.push(`GPU ${rt.gpu_utilization_percent}%`);
  if (rt.memory_used_mb != null && rt.memory_total_mb != null)
    parts.push(`显存 ${(rt.memory_used_mb / 1024).toFixed(1)}/${(rt.memory_total_mb / 1024).toFixed(1)}GB`);
  if (rt.temperature_c != null)
    parts.push(`${rt.temperature_c}°C`);
  if (rt.power_draw_w != null && rt.power_limit_w != null)
    parts.push(`${rt.power_draw_w.toFixed(1)}/${rt.power_limit_w.toFixed(0)}W`);
  el.textContent = parts.join(" · ");
  el.hidden = false;
}

async function pollDeviceMetrics() {
  try {
    const res = await fetch("/api/device/metrics");
    const data = await res.json();
    renderDeviceMetrics(data.runtime, data.device_type);
  } catch {}
}

function startDevicePolling() {
  if (_devicePollTimer) clearInterval(_devicePollTimer);
  _devicePollTimer = setInterval(pollDeviceMetrics, 1000);
}

function stopDevicePolling() {
  if (_devicePollTimer) {
    clearInterval(_devicePollTimer);
    _devicePollTimer = null;
  }
  // 停止后再拉一次最终状态
  pollDeviceMetrics();
}

// ---------- 统计 ----------
async function loadStats() {
  try {
    const [videos, history, projects] = await Promise.all([
      fetch("/api/library").then(r => r.json()),
      fetch("/api/history?limit=200").then(r => r.json()),
      fetch("/api/edit-projects").then(r => r.json()),
    ]);
    $("statVideos").textContent = videos.total || 0;
    $("statAnalyses").textContent = history.total || 0;
    $("statProjects").textContent = projects.total || 0;
  } catch (e) {
    console.error("加载统计失败:", e);
  }
}

// ---------- 历史记录 ----------
async function loadHistory() {
  const container = $("historyList");
  try {
    const params = new URLSearchParams({ limit: "50" });
    if (currentFilter) params.set("mode", currentFilter);
    const res = await fetch(`/api/history?${params}`);
    const data = await res.json();

    if (!data.tasks || data.tasks.length === 0) {
      container.innerHTML = '<div class="wb__empty">暂无分析记录</div>';
      return;
    }

    container.innerHTML = data.tasks.map(task => {
      const modeMap = { rally: { label: "得分段", icon: "🏓" }, action: { label: "动作", icon: "🏃" }, tracknet_train: { label: "模型微调", icon: "🧠" }, render: { label: "成片渲染", icon: "🎬" } };
      const modeInfo = modeMap[task.mode] || { label: task.mode, icon: "📋" };
      const modeLabel = modeInfo.label;
      const modeIcon = modeInfo.icon;
      const statusClass = task.status || "processing";
      const statusLabel = { completed: "完成", processing: "进行中", failed: "失败", cancelled: "已取消" }[task.status] || task.status;
      const time = formatTime(task.created_at);

      // 进度条和当前步骤
      let progressHtml = "";
      let detailBtn = "";
      if (task.status === "processing") {
        const pct = task.progress || 0;
        const stepText = task.message || "处理中";
        progressHtml = `
          <div class="wb__task-progress">
            <div class="wb__task-progress-bar">
              <div class="wb__task-progress-fill" style="width:${pct}%"></div>
            </div>
            <span class="wb__task-progress-text">${pct}% · ${stepText}</span>
          </div>
        `;
        detailBtn = `<button class="wb__history-link wb__detail-btn" onclick="showTaskDetail('${task.task_id}', event)">详情</button>`;
      }

      let link = "";
      if (task.status === "processing") {
        link = `<button class="wb__history-link wb__stop-btn" onclick="cancelTask('${task.task_id}', event)">停止</button>`;
      } else if (task.status === "completed" && task.mode === "render" && task.project_id) {
        link = `<a class="wb__history-link" href="/api/edit-projects/${task.project_id}/export" onclick="event.stopPropagation()">下载</a><a class="wb__history-link" href="/editor/${task.project_id}">编辑</a><button class="wb__history-link wb__rerender-btn" onclick="rerenderProject('${task.project_id}', event)">重渲染</button>`;
      } else if (task.status === "completed" && task.project_id) {
        link = `<a class="wb__history-link" href="/editor/${task.project_id}">编辑</a>`;
      } else if (task.status === "completed" && task.mode === "tracknet_train") {
        link = `<button class="wb__history-link wb__save-btn" onclick="saveTrainModel('${task.task_id}', event)">保存权重</button>`;
      } else if (task.status === "completed") {
        link = `<a class="wb__history-link" href="/analyze?task=${task.task_id}">查看</a>`;
      }
      const deleteBtn = task.status !== "processing"
        ? `<button class="wb__history-link wb__delete-btn" onclick="deleteTask('${task.task_id}', event)" title="删除">✕</button>`
        : "";

      // 结果信息
      let resultInfo = "";
      if (task.mode === "tracknet_train") {
        const eh = task.result?.epoch_history || [];
        if (eh.length > 0) {
          const last = eh[eh.length - 1];
          const detRate = last.positive_detection_rate != null ? `${(last.positive_detection_rate * 100).toFixed(0)}% 检测率` : "";
          const valLoss = last.validation_loss != null ? `val ${last.validation_loss.toFixed(4)}` : "";
          resultInfo = `${eh.length} 轮训练${detRate ? " · " + detRate : ""}${valLoss ? " · " + valLoss : ""}`;
        }
      } else if (task.result?.total_rallies != null) {
        resultInfo = `${task.result.total_rallies} 个回合`;
      } else {
        resultInfo = task.source_names || "";
      }

      // 视频时长 & 处理时长
      const videoDur = task.video_duration ? `视频 ${formatDuration(task.video_duration)}` : "";
      let procDur = "";
      if (task.created_at && task.updated_at && task.status !== "processing") {
        const procSec = (new Date(task.updated_at) - new Date(task.created_at)) / 1000;
        if (procSec > 0) procDur = `处理 ${formatDuration(procSec)}`;
      }

      return `
        <div class="wb__history-item" data-task-id="${task.task_id}" data-project-id="${task.project_id || ''}">
          <div class="wb__history-icon wb__history-icon--${task.mode}">${modeIcon}</div>
          <div class="wb__history-body">
            <div class="wb__history-title">${modeLabel} · ${task.video_filename || task.video_id || task.task_id}</div>
            <div class="wb__history-meta">
              <span class="wb__badge wb__badge--${statusClass}">${statusLabel}</span>
              <span>${resultInfo}</span>
              ${videoDur ? `<span>${videoDur}</span>` : ""}
              ${procDur ? `<span>${procDur}</span>` : ""}
              <span>${time}</span>
            </div>
            ${progressHtml}
          </div>
          ${detailBtn}
          ${link}
          ${deleteBtn}
        </div>
      `;
    }).join("");
  } catch (e) {
    container.innerHTML = '<div class="wb__empty">加载失败</div>';
    console.error("加载历史失败:", e);
  }
}

// ---------- 视频库 ----------
async function loadVideos() {
  const container = $("videoGrid");
  try {
    const res = await fetch("/api/library");
    const data = await res.json();

    $("videoCount").textContent = data.total || 0;

    if (!data.videos || data.videos.length === 0) {
      container.innerHTML = '<div class="wb__empty">暂无视频</div>';
      return;
    }

    container.innerHTML = data.videos.map(v => {
      const info = v.info || {};
      const duration = info.duration ? formatDuration(info.duration) : "";
      const resolution = info.width ? `${info.width}×${info.height}` : "";
      const size = formatSize(v.size);
      return `
        <div class="wb__video-card" data-video-id="${v.video_id}">
          <div class="wb__video-info">
            <div class="wb__video-name" title="${v.filename}">${v.filename}</div>
            <div class="wb__video-meta">${duration} · ${resolution} · ${size}</div>
          </div>
          <div class="wb__video-actions">
            <a class="wb__video-action" href="/api/videos/${v.video_id}/download" title="下载" onclick="event.stopPropagation()">⬇</a>
            <button class="wb__video-action" onclick="deleteVideo('${v.video_id}', event)" title="删除">✕</button>
          </div>
        </div>
      `;
    }).join("");
  } catch (e) {
    container.innerHTML = '<div class="wb__empty">加载失败</div>';
  }
}

// ---------- 成片项目 ----------
async function loadProjects() {
  const container = $("projectList");
  try {
    const res = await fetch("/api/edit-projects");
    const data = await res.json();

    $("projectCount").textContent = data.total || 0;

    if (!data.projects || data.projects.length === 0) {
      container.innerHTML = '<div class="wb__empty">暂无成片项目</div>';
      return;
    }

    container.innerHTML = data.projects.map(p => {
      const statusLabel = { draft: "草稿", rendering: "渲染中", completed: "已完成" }[p.status] || p.status;
      const segCount = (p.segments || []).filter(s => s.selected).length;
      const time = formatTime(p.created_at);
      return `
        <div class="wb__project-item" data-project-id="${p.project_id}">
          <div class="wb__project-body">
            <div class="wb__project-name">${p.source_filename || p.project_id}</div>
            <div class="wb__project-meta">${statusLabel} · ${segCount} 段 · ${time}</div>
          </div>
          <a class="wb__history-link" href="/editor/${p.project_id}">打开</a>
        </div>
      `;
    }).join("");
  } catch (e) {
    container.innerHTML = '<div class="wb__empty">加载失败</div>';
  }
}

// ---------- 上传 ----------
function bindEvents() {
  // 过滤按钮
  document.querySelectorAll(".wb__filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".wb__filter-btn").forEach(b => b.classList.remove("wb__filter-btn--active"));
      btn.classList.add("wb__filter-btn--active");
      currentFilter = btn.dataset.mode;
      loadHistory();
    });
  });

  // 上传按钮
  $("uploadBtn").addEventListener("click", () => $("fileInput").click());
  $("fileInput").addEventListener("change", handleUpload);

  // 登出
  $("logoutBtn").addEventListener("click", () => window.PPAuth?.logout());

  // 用户管理
  $("addUserBtn").addEventListener("click", openCreateUserModal);
  $("userModalCancel").addEventListener("click", closeUserModal);
  $("userForm").addEventListener("submit", handleUserFormSubmit);

  // 视频卡片点击 → 跳转分析
  $("videoGrid").addEventListener("click", (e) => {
    const card = e.target.closest(".wb__video-card");
    if (card) {
      window.location.href = `/analyze?video=${card.dataset.videoId}`;
    }
  });

  // 历史记录点击
  $("historyList").addEventListener("click", (e) => {
    const item = e.target.closest(".wb__history-item");
    if (item && !e.target.closest(".wb__history-link")) {
      const taskId = item.dataset.taskId;
      // 模型微调任务打开详情弹窗
      if (taskId.startsWith("train_")) {
        showTaskDetail(taskId, e);
      } else if (taskId.startsWith("render_")) {
        // 渲染任务跳转到编辑器页面
        const projectId = item.dataset.projectId;
        if (projectId) {
          window.location.href = `/editor/${projectId}`;
        }
      } else {
        window.location.href = `/analyze?task=${taskId}`;
      }
    }
  });
}

async function handleUpload(e) {
  const files = Array.from(e.target.files);
  if (!files.length) return;
  e.target.value = ""; // 允许再次选择相同文件

  const modal = $("uploadModal");
  const queueEl = $("uploadQueue");
  const titleEl = $("uploadModalTitle");
  const cancelBtn = $("uploadCancelBtn");
  modal.hidden = false;
  queueEl.innerHTML = "";

  // 取消控制
  let uploadCancelled = false;
  let currentUploadId = null;
  const onCancel = async () => {
    uploadCancelled = true;
    cancelBtn.disabled = true;
    cancelBtn.textContent = "取消中...";
    // 通知服务端清理当前分片
    if (currentUploadId) {
      try { await fetch(`/api/upload/abort/${currentUploadId}`, { method: "DELETE" }); } catch {}
    }
  };
  cancelBtn.disabled = false;
  cancelBtn.textContent = "取消";
  cancelBtn.onclick = onCancel;

  // 为每个文件创建队列项
  const items = files.map((file, idx) => {
    const item = document.createElement("div");
    item.className = "wb__upload-item";
    item.innerHTML = `
      <div class="wb__upload-item-header">
        <span class="wb__upload-item-name">${file.name.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</span>
        <span class="wb__upload-item-status" id="uploadStatus_${idx}">等待中</span>
      </div>
      <div class="wb__upload-item-bar">
        <div class="wb__upload-item-fill" id="uploadFill_${idx}"></div>
      </div>
    `;
    queueEl.appendChild(item);
    return { file, idx };
  });

  let successCount = 0;
  let lastVideoId = null;

  // 逐个上传
  for (const { file, idx } of items) {
    if (uploadCancelled) {
      const statusEl = $(`uploadStatus_${idx}`);
      statusEl.textContent = "已取消";
      statusEl.className = "wb__upload-item-status wb__upload-item-status--error";
      continue;
    }

    const statusEl = $(`uploadStatus_${idx}`);
    const fillEl = $(`uploadFill_${idx}`);
    titleEl.textContent = `上传中 (${successCount + 1}/${files.length})`;
    statusEl.textContent = "上传中...";
    statusEl.className = "wb__upload-item-status wb__upload-item-status--active";

    const result = await uploadSingleFile(file, (pct, text) => {
      fillEl.style.width = pct + "%";
      statusEl.textContent = text || `${pct}%`;
    }, (uid) => { currentUploadId = uid; }, () => uploadCancelled);

    currentUploadId = null;

    if (result.ok) {
      successCount++;
      lastVideoId = result.video_id;
      statusEl.textContent = "完成";
      statusEl.className = "wb__upload-item-status wb__upload-item-status--done";
      fillEl.style.width = "100%";
      fillEl.classList.add("wb__upload-item-fill--done");
    } else if (result.cancelled) {
      statusEl.textContent = "已取消";
      statusEl.className = "wb__upload-item-status wb__upload-item-status--error";
    } else {
      statusEl.textContent = result.error || "失败";
      statusEl.className = "wb__upload-item-status wb__upload-item-status--error";
      fillEl.classList.add("wb__upload-item-fill--error");
    }
  }

  cancelBtn.onclick = null;

  if (uploadCancelled) {
    titleEl.textContent = `已取消 (${successCount}/${files.length} 已完成)`;
    // 3 秒后关闭弹窗并刷新
    setTimeout(() => { modal.hidden = true; window.location.reload(); }, 3000);
    return;
  }

  // 全部完成
  titleEl.textContent = `上传完成 (${successCount}/${files.length})`;
  if (successCount > 0) {
    setTimeout(() => {
      modal.hidden = true;
      // 单文件直接跳转，多文件刷新工作台
      if (successCount === 1 && files.length === 1) {
        window.location.href = `/analyze?video=${lastVideoId}`;
      } else {
        window.location.reload();
      }
    }, 1500);
  }
}

async function uploadSingleFile(file, onProgress, onUploadId, isCancelled) {
  const CHUNK_SIZE = 5 * 1024 * 1024;
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  const MAX_RETRIES = 3;

  // 1. 初始化
  let uploadId, uploadedChunks;
  try {
    const initForm = new FormData();
    initForm.append("filename", file.name);
    initForm.append("total_size", file.size);
    const initResp = await fetch("/api/upload/init", { method: "POST", body: initForm });
    if (!initResp.ok) throw new Error("init failed");
    const initData = await initResp.json();
    uploadId = initData.upload_id;
    uploadedChunks = new Set(initData.uploaded_chunks || []);
    if (onUploadId) onUploadId(uploadId);
  } catch {
    return { ok: false, error: "初始化失败" };
  }

  // 2. 逐片上传
  for (let i = 0; i < totalChunks; i++) {
    if (isCancelled && isCancelled()) {
      try { await fetch(`/api/upload/abort/${uploadId}`, { method: "DELETE" }); } catch {}
      return { ok: false, cancelled: true };
    }

    if (uploadedChunks.has(i)) {
      const pct = Math.round(((i + 1) / totalChunks) * 100);
      onProgress(pct, `${pct}%`);
      continue;
    }

    let success = false;
    for (let retry = 0; retry < MAX_RETRIES; retry++) {
      if (isCancelled && isCancelled()) {
        try { await fetch(`/api/upload/abort/${uploadId}`, { method: "DELETE" }); } catch {}
        return { ok: false, cancelled: true };
      }
      try {
        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, file.size);
        const chunkBlob = file.slice(start, end);

        const chunkForm = new FormData();
        chunkForm.append("upload_id", uploadId);
        chunkForm.append("chunk_index", i);
        chunkForm.append("chunk", chunkBlob, `chunk_${i}`);

        const resp = await fetch("/api/upload/chunk", { method: "POST", body: chunkForm });
        if (!resp.ok) throw new Error(`chunk ${i} failed`);

        success = true;
        uploadedChunks.add(i);
        break;
      } catch {
        if (retry < MAX_RETRIES - 1) {
          onProgress(Math.round((i / totalChunks) * 100), `重试中 (${retry + 2}/${MAX_RETRIES})`);
          await new Promise(r => setTimeout(r, 1000 * (retry + 1)));
        }
      }
    }

    if (!success) {
      return { ok: false, error: `分片 ${i + 1} 失败` };
    }

    const pct = Math.round(((i + 1) / totalChunks) * 100);
    onProgress(pct, `${pct}%`);
  }

  // 3. 合并
  try {
    onProgress(100, "合并中...");
    const completeForm = new FormData();
    completeForm.append("upload_id", uploadId);
    const resp = await fetch("/api/upload/complete", { method: "POST", body: completeForm });
    if (!resp.ok) throw new Error("complete failed");
    const data = await resp.json();
    return { ok: true, video_id: data.video_id };
  } catch {
    return { ok: false, error: "合并失败" };
  }
}

// ---------- 用户管理模态框 ----------
function openCreateUserModal() {
  $("editUserId").value = "";
  $("formUsername").value = "";
  $("formUsername").disabled = false;
  $("formPassword").value = "";
  $("formPassword").required = true;
  $("passwordHint").textContent = "(至少 8 位)";
  $("formRole").value = "analyst";
  $("userModalTitle").textContent = "创建用户";
  $("userModalSubmit").textContent = "创建";
  $("userModal").hidden = false;
}

function closeUserModal() {
  $("userModal").hidden = true;
  $("userFormError").hidden = true;
}

async function handleUserFormSubmit(e) {
  e.preventDefault();
  const userId = $("editUserId").value;
  const username = $("formUsername").value.trim();
  const password = $("formPassword").value;
  const role = $("formRole").value;
  const errorEl = $("userFormError");

  const body = { role };
  if (password) body.password = password;
  if (!userId) body.username = username;

  const url = userId ? `/api/admin/users/${userId}` : "/api/admin/users";
  const method = userId ? "PATCH" : "POST";

  try {
    const res = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.detail || "操作失败");
    }
    closeUserModal();
    loadUsers();
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  }
}

// ---------- 工具函数 ----------
function formatTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  const now = new Date();
  const diff = (now - d) / 1000;
  if (diff < 60) return "刚刚";
  if (diff < 3600) return Math.floor(diff / 60) + " 分钟前";
  if (diff < 86400) return Math.floor(diff / 3600) + " 小时前";
  if (diff < 604800) return Math.floor(diff / 86400) + " 天前";
  return d.toLocaleDateString("zh-CN");
}

function formatDuration(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
  if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  return (bytes / (1024 * 1024 * 1024)).toFixed(1) + " GB";
}

// 启动
init();
