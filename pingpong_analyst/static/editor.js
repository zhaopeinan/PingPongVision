(function () {
  "use strict";

  const pathParts = window.location.pathname.split("/").filter(Boolean);
  const projectId = decodeURIComponent(pathParts[pathParts.length - 1] || "");
  const state = {
    project: null,
    stage: "browse",
    currentSegmentId: null,
    saveTimer: null,
    renderTimer: null,
    saveInFlight: false,
    savePromise: null,
    saveQueued: false,
    saveVersion: 0,
    draggedSegmentId: null,
    renderPollInFlight: false,
    renderStale: false,
  };

  const $ = (selector) => document.querySelector(selector);
  const editorApp = $("#editorApp");

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function finiteNumber(value, fallback = 0) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function formatSeconds(value) {
    const seconds = Math.max(0, finiteNumber(value));
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
  }

  function formatRange(start, end) {
    return `${finiteNumber(start).toFixed(1)}s - ${finiteNumber(end).toFixed(1)}s`;
  }

  function formatFileSize(bytes) {
    const number = finiteNumber(bytes, NaN);
    if (!Number.isFinite(number) || number < 0) return "文件大小读取中";
    if (number < 1024 * 1024) return `${(number / 1024).toFixed(0)} KB`;
    return `${(number / (1024 * 1024)).toFixed(1)} MB`;
  }

  function escapeText(value) {
    return String(value ?? "");
  }

  function getSegments() {
    return state.project?.segments || [];
  }

  function orderedSegments(segments = getSegments()) {
    return [...segments].sort((left, right) => {
      const orderDelta = finiteNumber(left.order, 0) - finiteNumber(right.order, 0);
      if (orderDelta !== 0) return orderDelta;
      return finiteNumber(left.analysis_index, 0) - finiteNumber(right.analysis_index, 0);
    });
  }

  function selectedSegments() {
    return orderedSegments().filter((segment) => segment.selected !== false);
  }

  function currentSegment() {
    return getSegments().find((segment) => segment.segment_id === state.currentSegmentId) || null;
  }

  function normalizeSegment(segment, index) {
    const clipStart = finiteNumber(segment.clip_start_time, finiteNumber(segment.detected_start_time));
    const clipEnd = Math.max(clipStart, finiteNumber(segment.clip_end_time, finiteNumber(segment.detected_end_time)));
    const editStart = clamp(finiteNumber(segment.edit_start_time, clipStart), clipStart, clipEnd);
    const editEnd = clamp(finiteNumber(segment.edit_end_time, clipEnd), clipStart, clipEnd);
    return {
      ...segment,
      segment_id: String(segment.segment_id || `segment-${String(index + 1).padStart(3, "0")}`),
      analysis_index: finiteNumber(segment.analysis_index, index + 1),
      clip_start_time: clipStart,
      clip_end_time: clipEnd,
      edit_start_time: Math.min(editStart, Math.max(clipStart, editEnd - 0.01)),
      edit_end_time: Math.max(editEnd, Math.min(clipEnd, editStart + 0.01)),
      board_count: finiteNumber(segment.board_count, 0),
      selected: segment.selected !== false,
      order: finiteNumber(segment.order, index + 1),
    };
  }

  function normalizeProject(project) {
    const segments = Array.isArray(project?.segments)
      ? project.segments.map(normalizeSegment)
      : [];
    return {
      ...project,
      segments,
      render: {
        status: "idle",
        progress: 0,
        message: "",
        task_id: null,
        output_filename: null,
        duration: null,
        error: null,
        ...(project?.render || {}),
      },
    };
  }

  async function requestJson(url, options = {}) {
    const response = await fetch(url, { cache: "no-store", ...options });
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) {
      const message = payload?.error || payload?.detail || `请求失败 (${response.status})`;
      const error = new Error(message);
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  function setError(message, canRetry = false) {
    const alert = $("#editorError");
    $("#editorErrorText").textContent = message;
    $("#editorErrorRetry").hidden = !canRetry;
    alert.hidden = false;
  }

  function clearError() {
    $("#editorError").hidden = true;
    $("#editorErrorText").textContent = "";
  }

  function setSaveStatus(text, stateName = "idle") {
    const status = $("#editorSaveStatus");
    status.textContent = text;
    status.dataset.state = stateName;
    status.disabled = stateName !== "error";
  }

  async function loadProject() {
    if (!projectId) {
      setError("项目编号无效。", false);
      setSaveStatus("项目读取失败", "error");
      return;
    }
    try {
      const project = await requestJson(`/api/edit-projects/${encodeURIComponent(projectId)}`);
      state.project = normalizeProject(project);
      const first = orderedSegments()[0];
      state.currentSegmentId = first?.segment_id || null;
      $("#projectSource").textContent = escapeText(state.project.source_filename || state.project.source_storage_name || "未命名视频");
      clearError();
      setSaveStatus("已加载项目", "saved");
      renderAll();

      const renderStatus = state.project.render.status;
      if (["rendering", "completed", "failed"].includes(renderStatus)) {
        state.stage = "render";
        renderAll();
        if (renderStatus === "rendering") startRenderPolling();
      }
    } catch (error) {
      setError(error.message || "项目读取失败。", true);
      setSaveStatus("项目读取失败", "error");
      $("#editorErrorRetry").onclick = loadProject;
    }
  }

  function setStage(stage) {
    if (!["browse", "arrange", "render"].includes(stage)) return;
    if (isRendering() && stage !== "render") return;
    state.stage = stage;
    editorApp.dataset.stage = stage;
    document.querySelectorAll(".editor-step").forEach((button) => {
      const active = button.dataset.stage === stage;
      button.classList.toggle("editor-step--active", active);
      button.setAttribute("aria-current", active ? "step" : "false");
    });
    $("#renderPanel").hidden = stage !== "render";
    $("#stageNote").textContent = stage === "browse"
      ? "逐个查看识别到的精彩分段"
      : stage === "arrange"
        ? "拖动或使用上移、下移调整顺序"
        : "确认顺序后渲染并导出 MP4";
    renderSegmentList();
    renderPreviewActions();
    renderCompositionPanel();
  }

  function renderAll() {
    setStage(state.stage);
    renderCurrentSegment();
    renderSegmentList();
    renderSummary();
    renderCompositionPanel();
    updateEditingLock();
  }

  function renderCurrentSegment() {
    const segment = currentSegment();
    const player = $("#segmentPlayer");
    const empty = $("#playerEmpty");
    if (!segment) {
      player.removeAttribute("src");
      player.load();
      player.hidden = true;
      empty.hidden = false;
      $("#previewHeading").textContent = "分段预览";
      $("#segmentMeta").replaceChildren();
      $("#trimControls").replaceChildren();
      renderPreviewActions();
      return;
    }

    player.hidden = false;
    empty.hidden = true;
    $("#previewHeading").textContent = `分段 ${String(segment.analysis_index).padStart(2, "0")}`;
    const clipUrl = `/api/clips/${encodeURIComponent(segment.clip_filename || "")}`;
    if (player.dataset.clipUrl !== clipUrl) {
      player.dataset.clipUrl = clipUrl;
      player.src = clipUrl;
      player.load();
    }
    renderSegmentMeta(segment);
    renderTrimControls(segment);
    renderPreviewActions();
  }

  function renderSegmentMeta(segment) {
    const meta = $("#segmentMeta");
    meta.replaceChildren();
    const values = [
      ["板数", `${segment.board_count} 板`],
      ["原始范围", formatRange(segment.detected_start_time, segment.detected_end_time)],
      ["当前范围", formatRange(segment.edit_start_time, segment.edit_end_time)],
      ["时长", formatSeconds(segment.edit_end_time - segment.edit_start_time)],
    ];
    values.forEach(([label, value]) => {
      const item = document.createElement("div");
      item.className = "segment-meta__item";
      const labelElement = document.createElement("span");
      labelElement.className = "segment-meta__label";
      labelElement.textContent = label;
      const valueElement = document.createElement("strong");
      valueElement.className = "segment-meta__value";
      valueElement.textContent = value;
      item.append(labelElement, valueElement);
      meta.appendChild(item);
    });
  }

  function createInput(type, id, value, min, max, label) {
    const input = document.createElement("input");
    input.type = type;
    input.id = id;
    input.name = id;
    input.value = Number(value).toFixed(2);
    input.min = Number(min).toFixed(2);
    input.max = Number(max).toFixed(2);
    input.step = "0.01";
    input.setAttribute("aria-label", label);
    input.disabled = isRendering();
    return input;
  }

  function renderTrimControls(segment) {
    const container = $("#trimControls");
    container.replaceChildren();
    const title = document.createElement("div");
    title.className = "trim-controls__heading";
    title.innerHTML = "<strong>微调分段范围</strong><span>可调整范围受原始剪辑窗口限制</span>";
    container.appendChild(title);

    const lower = segment.clip_start_time;
    const upper = segment.clip_end_time;
    const range = document.createElement("div");
    range.className = "trim-range-stack";
    const track = document.createElement("div");
    track.className = "trim-range-track";
    const startRange = createInput("range", "editStartRange", segment.edit_start_time, lower, upper, "开始时间");
    const endRange = createInput("range", "editEndRange", segment.edit_end_time, lower, upper, "结束时间");
    startRange.classList.add("trim-range", "trim-range--start");
    endRange.classList.add("trim-range", "trim-range--end");
    range.append(track, startRange, endRange);
    container.appendChild(range);

    const numbers = document.createElement("div");
    numbers.className = "trim-number-grid";
    const startField = document.createElement("label");
    startField.className = "trim-number-field";
    startField.innerHTML = "<span>开始</span>";
    startField.appendChild(createInput("number", "editStartNumber", segment.edit_start_time, lower, upper, "开始时间（秒）"));
    const endField = document.createElement("label");
    endField.className = "trim-number-field";
    endField.innerHTML = "<span>结束</span>";
    endField.appendChild(createInput("number", "editEndNumber", segment.edit_end_time, lower, upper, "结束时间（秒）"));
    numbers.append(startField, endField);
    container.appendChild(numbers);

    const bounds = document.createElement("div");
    bounds.className = "trim-bounds";
    bounds.textContent = `允许范围 ${formatRange(lower, upper)}`;
    container.appendChild(bounds);

    [$("#editStartRange"), $("#editStartNumber")].forEach((input) => {
      input.addEventListener("input", (event) => updateTrim("start", event.target.value));
      input.addEventListener("change", (event) => updateTrim("start", event.target.value));
    });
    [$("#editEndRange"), $("#editEndNumber")].forEach((input) => {
      input.addEventListener("input", (event) => updateTrim("end", event.target.value));
      input.addEventListener("change", (event) => updateTrim("end", event.target.value));
    });
  }

  function syncTrimControls(segment) {
    const controls = [
      ["#editStartRange", segment.edit_start_time],
      ["#editStartNumber", segment.edit_start_time],
      ["#editEndRange", segment.edit_end_time],
      ["#editEndNumber", segment.edit_end_time],
    ];
    controls.forEach(([selector, value]) => {
      const input = $(selector);
      if (input) input.value = Number(value).toFixed(2);
    });
  }

  function updateTrim(kind, rawValue) {
    const segment = currentSegment();
    if (!segment || isRendering()) return;
    const lower = segment.clip_start_time;
    const upper = segment.clip_end_time;
    const minimumGap = Math.min(0.01, Math.max(0, upper - lower));
    let value = clamp(finiteNumber(rawValue, kind === "start" ? segment.edit_start_time : segment.edit_end_time), lower, upper);
    if (kind === "start") {
      value = Math.min(value, Math.max(lower, segment.edit_end_time - minimumGap));
      segment.edit_start_time = Number(value.toFixed(2));
    } else {
      value = Math.max(value, Math.min(upper, segment.edit_start_time + minimumGap));
      segment.edit_end_time = Number(value.toFixed(2));
    }
    syncTrimControls(segment);
    renderSegmentMeta(segment);
    renderSummary();
    renderSegmentList();
    markEdited();
  }

  function renderPreviewActions() {
    const container = $("#previewActions");
    container.replaceChildren();
    const all = orderedSegments();
    const index = all.findIndex((segment) => segment.segment_id === state.currentSegmentId);
    const previous = document.createElement("button");
    previous.className = "btn";
    previous.type = "button";
    previous.dataset.action = "previous";
    previous.textContent = "上一段";
    previous.disabled = index <= 0;
    const next = document.createElement("button");
    next.className = "btn";
    next.type = "button";
    next.dataset.action = "next";
    next.textContent = "下一段";
    next.disabled = index < 0 || index >= all.length - 1;
    const arrange = document.createElement("button");
    arrange.className = "btn btn--primary";
    arrange.type = "button";
    arrange.dataset.action = "arrange";
    arrange.textContent = "进入编排";
    container.append(previous, next, arrange);
  }

  function renderSegmentList() {
    const list = $("#segmentList");
    list.replaceChildren();
    // 始终渲染所有分段，未选中的降低透明度但仍可勾选回来
    const segments = orderedSegments();
    $("#segmentCount").textContent = `${getSegments().length} 段`;
    if (!segments.length) {
      const empty = document.createElement("div");
      empty.className = "segment-list__empty";
      empty.textContent = "未找到精彩分段";
      list.appendChild(empty);
      return;
    }

    segments.forEach((segment, index) => {
      const card = document.createElement("article");
      card.className = "segment-card";
      card.dataset.segmentId = segment.segment_id;
      card.draggable = state.stage === "arrange" && !isRendering();
      card.classList.toggle("segment-card--current", segment.segment_id === state.currentSegmentId);
      card.classList.toggle("segment-card--unselected", segment.selected === false);
      card.addEventListener("dragstart", () => {
        state.draggedSegmentId = segment.segment_id;
        card.classList.add("segment-card--dragging");
      });
      card.addEventListener("dragend", () => {
        state.draggedSegmentId = null;
        card.classList.remove("segment-card--dragging");
      });
      card.addEventListener("dragover", (event) => {
        if (state.stage === "arrange" && !isRendering()) event.preventDefault();
      });
      card.addEventListener("drop", (event) => {
        event.preventDefault();
        moveSegment(state.draggedSegmentId, segment.segment_id);
      });

      const main = document.createElement("button");
      main.className = "segment-card__main";
      main.type = "button";
      main.setAttribute("aria-label", `查看分段 ${segment.analysis_index}`);
      main.addEventListener("click", () => selectCurrentSegment(segment.segment_id));
      const indexElement = document.createElement("span");
      indexElement.className = "segment-card__index";
      indexElement.textContent = String(index + 1).padStart(2, "0");
      const content = document.createElement("span");
      content.className = "segment-card__content";
      const title = document.createElement("strong");
      title.className = "segment-card__title";
      title.textContent = `分析分段 ${segment.analysis_index}`;
      const details = document.createElement("span");
      details.className = "segment-card__details";
      details.textContent = `${segment.board_count} 板 · ${formatSeconds(segment.edit_end_time - segment.edit_start_time)}`;
      const file = document.createElement("span");
      file.className = "segment-card__file";
      file.textContent = escapeText(segment.clip_filename || "无剪辑文件");
      content.append(title, details, file);
      main.append(indexElement, content);

      const tools = document.createElement("div");
      tools.className = "segment-card__tools";
      const selectionLabel = document.createElement("label");
      selectionLabel.className = "segment-select";
      selectionLabel.title = segment.selected === false ? "加入成片" : "从成片移除";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = segment.selected !== false;
      checkbox.disabled = isRendering();
      checkbox.setAttribute("aria-label", `${segment.analysis_index}号分段加入成片`);
      checkbox.addEventListener("click", (event) => event.stopPropagation());
      checkbox.addEventListener("change", () => toggleSelection(segment.segment_id, checkbox.checked));
      selectionLabel.appendChild(checkbox);
      tools.appendChild(selectionLabel);

      if (state.stage === "arrange") {
        const up = createOrderButton("上移", "up", index === 0, segment.segment_id);
        const down = createOrderButton("下移", "down", index === segments.length - 1, segment.segment_id);
        const remove = createOrderButton("移除", "remove", false, segment.segment_id);
        tools.append(up, down, remove);
      }
      card.append(main, tools);
      list.appendChild(card);
    });
  }

  function createOrderButton(label, action, disabled, segmentId) {
    const button = document.createElement("button");
    button.className = "segment-tool";
    button.type = "button";
    button.textContent = label;
    button.disabled = disabled || isRendering();
    button.setAttribute("aria-label", `${label}分段`);
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      if (action === "remove") toggleSelection(segmentId, false);
      else moveSegmentBy(segmentId, action === "up" ? -1 : 1);
    });
    return button;
  }

  function selectCurrentSegment(segmentId) {
    if (!getSegments().some((segment) => segment.segment_id === segmentId)) return;
    state.currentSegmentId = segmentId;
    $("#segmentPlayer").pause();
    renderCurrentSegment();
    renderSegmentList();
  }

  function navigateSegment(delta) {
    const segments = orderedSegments();
    const index = segments.findIndex((segment) => segment.segment_id === state.currentSegmentId);
    const nextIndex = index + delta;
    if (nextIndex < 0 || nextIndex >= segments.length) return;
    selectCurrentSegment(segments[nextIndex].segment_id);
  }

  function toggleSelection(segmentId, selected) {
    if (isRendering()) return;
    const segment = getSegments().find((item) => item.segment_id === segmentId);
    if (!segment) return;
    segment.selected = Boolean(selected);
    renderSegmentList();
    renderSummary();
    renderCompositionPanel();
    markEdited();
  }

  function recomputeOrder(segments) {
    segments.forEach((segment, index) => { segment.order = index + 1; });
  }

  function moveSegmentBy(segmentId, delta) {
    if (isRendering()) return;
    const segments = selectedSegments();
    const index = segments.findIndex((segment) => segment.segment_id === segmentId);
    const target = index + delta;
    if (index < 0 || target < 0 || target >= segments.length) return;
    moveSegment(segmentId, segments[target].segment_id);
  }

  function moveSegment(fromId, toId) {
    if (isRendering() || !fromId || !toId || fromId === toId) return;
    const selected = selectedSegments();
    const fromIndex = selected.findIndex((segment) => segment.segment_id === fromId);
    const toIndex = selected.findIndex((segment) => segment.segment_id === toId);
    if (fromIndex < 0 || toIndex < 0) return;
    const [moved] = selected.splice(fromIndex, 1);
    selected.splice(toIndex, 0, moved);
    const unselected = getSegments().filter((segment) => segment.selected === false);
    recomputeOrder([...selected, ...unselected]);
    selected.forEach((segment, index) => { segment.order = index + 1; });
    unselected.forEach((segment, index) => { segment.order = selected.length + index + 1; });
    renderSegmentList();
    renderSummary();
    renderCompositionPanel();
    markEdited();
  }

  function renderSummary() {
    const selected = selectedSegments();
    const totalDuration = selected.reduce((sum, segment) => sum + Math.max(0, segment.edit_end_time - segment.edit_start_time), 0);
    const totalBoards = selected.reduce((sum, segment) => sum + finiteNumber(segment.board_count), 0);
    const summary = $("#segmentSummary");
    summary.replaceChildren();
    [["已选择", `${selected.length}/${getSegments().length} 段`], ["预计时长", formatSeconds(totalDuration)], ["总板数", `${totalBoards} 板`]].forEach(([label, value]) => {
      const item = document.createElement("div");
      item.className = "summary-stat";
      const labelElement = document.createElement("span");
      labelElement.textContent = label;
      const valueElement = document.createElement("strong");
      valueElement.textContent = value;
      item.append(labelElement, valueElement);
      summary.appendChild(item);
    });
  }

  function renderCompositionPanel() {
    if (!state.project) return;
    const selected = selectedSegments();
    const totalDuration = selected.reduce((sum, segment) => sum + Math.max(0, segment.edit_end_time - segment.edit_start_time), 0);
    const totalBoards = selected.reduce((sum, segment) => sum + finiteNumber(segment.board_count), 0);
    const summary = $("#compositionSummary");
    summary.replaceChildren();
    [["成片分段", `${selected.length} 段`], ["预计时长", formatSeconds(totalDuration)], ["总板数", `${totalBoards} 板`]].forEach(([label, value]) => {
      const item = document.createElement("div");
      item.className = "composition-stat";
      const labelElement = document.createElement("span");
      labelElement.textContent = label;
      const valueElement = document.createElement("strong");
      valueElement.textContent = value;
      item.append(labelElement, valueElement);
      summary.appendChild(item);
    });

    const list = $("#compositionList");
    list.replaceChildren();
    if (!selected.length) {
      const empty = document.createElement("div");
      empty.className = "composition-list__empty";
      empty.textContent = "请至少选择一个分段后再渲染。";
      list.appendChild(empty);
    } else {
      selected.forEach((segment, index) => {
        const item = document.createElement("div");
        item.className = "composition-item";
        const number = document.createElement("span");
        number.className = "composition-item__number";
        number.textContent = String(index + 1).padStart(2, "0");
        const name = document.createElement("strong");
        name.className = "composition-item__name";
        name.textContent = `分析分段 ${segment.analysis_index}`;
        const details = document.createElement("span");
        details.className = "composition-item__details";
        details.textContent = `${segment.board_count} 板 · ${formatRange(segment.edit_start_time, segment.edit_end_time)} · ${formatSeconds(segment.edit_end_time - segment.edit_start_time)}`;
        item.append(number, name, details);
        list.appendChild(item);
      });
    }
    updateRenderUI();
  }

  function markEdited() {
    state.saveVersion += 1;
    state.saveQueued = true;
    state.renderStale = state.project?.render?.status === "completed" || state.renderStale;
    setSaveStatus("等待自动保存", "pending");
    scheduleSave();
  }

  function scheduleSave() {
    clearTimeout(state.saveTimer);
    state.saveTimer = setTimeout(() => saveProject(), 350);
  }

  function saveProject({ force = false } = {}) {
    if (!state.project || isRendering()) return Promise.resolve(false);
    if (!force && !state.saveQueued) return Promise.resolve(true);
    if (state.saveInFlight) return state.savePromise || Promise.resolve(false);
    const version = state.saveVersion;
    const payloadSegments = state.project.segments.map((segment) => ({
      segment_id: segment.segment_id,
      selected: segment.selected !== false,
      order: segment.order,
      edit_start_time: segment.edit_start_time,
      edit_end_time: segment.edit_end_time,
    }));
    state.saveQueued = false;
    state.saveInFlight = true;
    setSaveStatus("保存中...", "saving");
    const promise = (async () => {
      try {
        const response = await requestJson(`/api/edit-projects/${encodeURIComponent(projectId)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ segments: payloadSegments }),
        });
        if (version === state.saveVersion) {
          state.project = normalizeProject(response);
          state.currentSegmentId = state.project.segments.some((segment) => segment.segment_id === state.currentSegmentId)
            ? state.currentSegmentId
            : orderedSegments()[0]?.segment_id || null;
          renderAll();
          setSaveStatus(`已自动保存 · ${new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`, "saved");
        } else {
          state.saveQueued = true;
          scheduleSave();
        }
        return true;
      } catch (error) {
        state.saveQueued = true;
        setSaveStatus("保存失败，点击重试", "error");
        setError(error.message || "项目保存失败。", true);
        $("#editorErrorRetry").onclick = () => saveProject({ force: true });
        return false;
      } finally {
        state.saveInFlight = false;
        state.savePromise = null;
      }
    })();
    state.savePromise = promise;
    return promise;
  }

  function isRendering() {
    return state.project?.render?.status === "rendering";
  }

  function updateEditingLock() {
    editorApp.dataset.locked = isRendering() ? "true" : "false";
  }

  function normalizedProgress(value) {
    const number = finiteNumber(value, 0);
    const percent = number >= 0 && number <= 1 ? number * 100 : number;
    return clamp(Math.round(percent), 0, 100);
  }

  function applyRenderStatus(data) {
    if (!state.project) return;
    state.project.render = {
      ...state.project.render,
      ...(data || {}),
      progress: data?.progress ?? state.project.render.progress,
    };
    if (data?.status) state.project.status = data.status === "completed" ? "completed" : state.project.status;
    if (data?.status === "completed") state.renderStale = false;
    renderCompositionPanel();
    updateEditingLock();
  }

  function updateRenderUI() {
    if (!state.project) return;
    const render = state.project.render || {};
    const status = state.renderStale ? "idle" : render.status;
    const active = status === "rendering";
    const completed = status === "completed";
    const failed = status === "failed";
    $("#renderProgressPanel").hidden = !active;
    $("#renderFailurePanel").hidden = !failed;
    $("#finalResultPanel").hidden = !completed;
    $("#renderControls").hidden = active || completed;
    $("#renderPanelTitle").textContent = completed ? "成片已完成" : failed ? "渲染失败" : "确认成片顺序";

    const selected = selectedSegments();
    const confirmButton = $("#confirmRenderBtn");
    confirmButton.disabled = active || selected.length === 0;
    confirmButton.textContent = failed ? "重新渲染" : "确认并渲染";
    if (active) {
      const percent = normalizedProgress(render.progress);
      $("#renderProgressBar").style.width = `${percent}%`;
      $("#renderProgressPercent").textContent = `${percent}%`;
      $("#renderProgressMessage").textContent = render.message || "正在合并精彩分段";
      $("#renderProgressDetail").textContent = `正在生成成片 · ${percent}%`;
    }
    if (failed) {
      $("#renderFailureText").textContent = render.error || "渲染失败，请检查服务端日志后重试。";
    }
    if (completed) {
      const exportUrl = `/api/edit-projects/${encodeURIComponent(projectId)}/export`;
      const finalPlayer = $("#finalPlayer");
      if (finalPlayer.dataset.exportUrl !== exportUrl) {
        finalPlayer.dataset.exportUrl = exportUrl;
        finalPlayer.src = exportUrl;
      }
      $("#downloadExportBtn").href = exportUrl;
      const meta = $("#finalResultMeta");
      meta.replaceChildren();
      [["文件", render.output_filename || `${projectId}.mp4`], ["时长", formatSeconds(render.duration || selected.reduce((sum, segment) => sum + segment.edit_end_time - segment.edit_start_time, 0))], ["分段", `${selected.length} 段`], ["大小", formatFileSize(render.output_size_bytes || render.file_size)]].forEach(([label, value]) => {
        const item = document.createElement("div");
        item.className = "final-result__meta-item";
        const labelElement = document.createElement("span");
        labelElement.textContent = label;
        const valueElement = document.createElement("strong");
        valueElement.textContent = value;
        item.append(labelElement, valueElement);
        meta.appendChild(item);
      });
    }
  }

  async function startRender() {
    if (!state.project || isRendering()) return;
    if (!selectedSegments().length) {
      setError("请至少选择一个分段后再渲染。", false);
      return;
    }
    clearError();
    const saved = await saveProject({ force: true });
    if (!saved) return;
    try {
      const response = await requestJson(`/api/edit-projects/${encodeURIComponent(projectId)}/render`, { method: "POST" });
      state.renderStale = false;
      applyRenderStatus(response);
      setStage("render");
      startRenderPolling();
    } catch (error) {
      if (error.status === 409 && error.payload?.status === "rendering") {
        applyRenderStatus(error.payload);
        setStage("render");
        startRenderPolling();
        return;
      }
      setError(error.message || "渲染请求失败。", true);
      $("#editorErrorRetry").onclick = startRender;
    }
  }

  function startRenderPolling() {
    clearInterval(state.renderTimer);
    state.renderTimer = setInterval(pollRenderStatus, 1500);
    pollRenderStatus();
  }

  async function pollRenderStatus() {
    if (state.renderPollInFlight || !state.project) return;
    state.renderPollInFlight = true;
    try {
      const data = await requestJson(`/api/edit-projects/${encodeURIComponent(projectId)}/render`);
      applyRenderStatus(data);
      if (["completed", "failed"].includes(data?.status)) {
        clearInterval(state.renderTimer);
        state.renderTimer = null;
        if (data.status === "failed") {
          setError(data.error || "渲染失败。", true);
          $("#editorErrorRetry").onclick = startRender;
        } else {
          clearError();
        }
      }
    } catch (error) {
      setError(`渲染状态读取失败：${error.message || "网络错误"}`, true);
      $("#editorErrorRetry").onclick = pollRenderStatus;
    } finally {
      state.renderPollInFlight = false;
    }
  }

  function handleSegmentPlayerTime() {
    const segment = currentSegment();
    const player = $("#segmentPlayer");
    if (!segment || !Number.isFinite(player.currentTime)) return;
    const absoluteTime = segment.clip_start_time + player.currentTime;
    if (absoluteTime >= segment.edit_end_time && !player.paused) {
      player.pause();
      player.currentTime = Math.max(0, segment.edit_end_time - segment.clip_start_time);
    }
  }

  function syncPlayerToStart() {
    const segment = currentSegment();
    const player = $("#segmentPlayer");
    if (!segment || !Number.isFinite(player.duration)) return;
    player.currentTime = clamp(segment.edit_start_time - segment.clip_start_time, 0, player.duration);
  }

  function bindEvents() {
    document.querySelectorAll(".editor-step").forEach((button) => {
      button.addEventListener("click", () => setStage(button.dataset.stage));
    });
    $("#renderBackBtn").addEventListener("click", () => setStage("arrange"));
    $("#confirmRenderBtn").addEventListener("click", startRender);
    $("#retryRenderBtn").addEventListener("click", startRender);
    $("#returnEditBtn").addEventListener("click", () => {
      state.renderStale = false;
      setStage("arrange");
    });
    $("#editorSaveStatus").addEventListener("click", () => {
      if ($("#editorSaveStatus").dataset.state === "error") {
        if (state.project) saveProject({ force: true });
        else loadProject();
      }
    });
    $("#previewActions").addEventListener("click", (event) => {
      const action = event.target.closest("[data-action]")?.dataset.action;
      if (action === "previous") navigateSegment(-1);
      if (action === "next") navigateSegment(1);
      if (action === "arrange") setStage("arrange");
    });
    $("#segmentPlayer").addEventListener("loadedmetadata", syncPlayerToStart);
    $("#segmentPlayer").addEventListener("timeupdate", handleSegmentPlayerTime);
  }

  bindEvents();
  loadProject();
})();
