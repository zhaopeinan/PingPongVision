# 动作分析报告 UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将动作分析完成后的 JSON 风格文本改造成片段摘要、选手指标卡、手臂角度对比和可折叠技术详情组成的训练报告。

**Architecture:** 保持 `/api/action/analyze` 和任务结果协议不变，只在 `app.js` 中把现有响应转换为安全的展示 HTML，并在 `style.css` 中为报告定义层级、指标条和响应式布局。所有统计从现有 `analyses[].frames`、`persons` 和汇总字段计算，不新增模型或前端依赖。

**Tech Stack:** 原生 JavaScript、HTML 模板字符串、CSS Grid/Flex、原生 `details/summary`、FastAPI 现有动作分析结果。

---

### Task 1: 建立动作报告展示辅助函数

**Files:**
- Modify: `pingpong_analyst/static/app.js:1407`，在 `renderActionReport` 前加入展示辅助函数

- [ ] **Step 1: 添加中文指标映射和安全格式化函数**

在动作分析区域中加入以下函数。`escapeActionHtml` 用于文件名、身份和原始 JSON；`getFiniteActionNumber` 丢弃 `null`、非数字和非有限值；未知字段通过中文可读化名称展示，不把原始下划线键名直接暴露给用户。

```javascript
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

function escapeActionHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function getFiniteActionNumber(value) {
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
```

- [ ] **Step 2: 添加覆盖率、身份和数据摘要函数**

覆盖率只在分析帧数量为正且存在逐帧数组时计算；选手身份只有在后端给出非 `unknown` 身份且置信度达到 60% 时显示为已确认。跟踪标签始终使用列表序号，避免用画面左右推断身份。

```javascript
function getActionCoverage(analysis) {
  const total = getFiniteActionNumber(analysis?.frames_analyzed);
  const frames = Array.isArray(analysis?.frames) ? analysis.frames : null;
  if (!total || !frames) return null;
  const visibleFrames = frames.filter((frame) => Array.isArray(frame?.persons) && frame.persons.length > 0).length;
  return Math.min(1, visibleFrames / total);
}

function getActionPersonCoverage(person, analysis) {
  const total = getFiniteActionNumber(analysis?.frames_analyzed);
  const tracked = getFiniteActionNumber(person?.frame_count);
  if (!total || tracked === null) return null;
  return Math.min(1, Math.max(0, tracked / total));
}

function getActionIdentity(person, index) {
  const confidence = getFiniteActionNumber(person?.identity_confidence) ?? 0;
  const rawIdentity = String(person?.identity || "").trim();
  const confirmed = rawIdentity && rawIdentity !== "unknown" && confidence >= 0.6;
  return {
    label: confirmed ? rawIdentity : "身份未确认",
    trackLabel: `跟踪 ${index + 1}`,
    confidence,
    status: confirmed ? "已确认" : "待确认",
  };
}

function getActionSummary(analyses) {
  const items = Array.isArray(analyses) ? analyses : [];
  const totalFrames = items.reduce((sum, item) => sum + (getFiniteActionNumber(item?.frames_analyzed) || 0), 0);
  const trackedIds = new Set(items.flatMap((item) => (item?.persons || []).map((person) => person?.track_id).filter(Boolean)));
  const coverageValues = items.map(getActionCoverage).filter((value) => value !== null);
  return {
    clipCount: items.length,
    totalFrames,
    trackedCount: trackedIds.size,
    coverage: coverageValues.length ? coverageValues.reduce((sum, value) => sum + value, 0) / coverageValues.length : null,
  };
}
```

- [ ] **Step 3: 静态检查辅助函数语法**

Run: `node --check pingpong_analyst/static/app.js`

Expected: exit code 0 and no syntax error output.

### Task 2: 重写动作报告 DOM 结构

**Files:**
- Modify: `pingpong_analyst/static/app.js:1407-1435`，替换 `renderActionReport` 及其原始字段拼接逻辑

- [ ] **Step 1: 添加单项指标、选手卡片和对比区渲染函数**

新增渲染函数只接收已校验的数据，并为数值加单位。对比条的宽度按照同一指标中有效值的最大绝对值计算；它是相对比较，不标注技术好坏。身体指标使用 `details`，不放入首屏主卡片。

```javascript
function renderActionMetricList(metrics, labels, unit = "") {
  const entries = getActionMetricEntries(metrics, labels);
  if (!entries.length) return `<div class="action-report__no-data">暂无有效数据</div>`;
  return entries.map((metric) => `
    <div class="action-metric">
      <span class="action-metric__label">${escapeActionHtml(metric.label)}</span>
      <strong class="action-metric__value">${formatActionNumber(metric.value)}${unit}</strong>
    </div>
  `).join("");
}

function renderActionPersonCard(person, index, analysis) {
  const identity = getActionIdentity(person, index);
  const coverage = getActionPersonCoverage(person, analysis);
  const confidencePercent = Math.round(identity.confidence * 100);
  const confidenceClass = identity.confidence >= 0.8 ? "high" : identity.confidence >= 0.6 ? "medium" : "low";
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
      <div class="action-metrics">${renderActionMetricList(person?.average_arm_angles, ACTION_ARM_LABELS, "°")}</div>
      <details class="action-person-card__details">
        <summary>查看身体指标</summary>
        <div class="action-metrics">${renderActionMetricList(person?.average_body_metrics, ACTION_BODY_LABELS)}</div>
      </details>
    </article>
  `;
}

function renderActionComparison(persons) {
  const candidates = (persons || []).slice(0, 2).map((person, index) => ({
    person,
    label: getActionIdentity(person, index).trackLabel,
    metrics: person?.average_arm_angles || {},
  }));
  if (candidates.length < 2) return "";
  const keys = Object.keys(ACTION_ARM_LABELS).filter((key) => candidates.some((item) => getFiniteActionNumber(item.metrics[key]) !== null));
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
```

- [ ] **Step 2: 用总览、片段区和原始详情替换 `renderActionReport`**

`renderActionReport(data)` 保持现有面板切换逻辑，并生成以下结构：摘要 `.action-report__overview`、片段 `.action-report__source`、选手卡片 `.action-person-card`、对比 `.action-comparison`，以及在 `analysis.frames` 非空时生成的 `<details class="action-report__raw">`。原始数据必须经过 `escapeActionHtml(JSON.stringify(..., null, 2))`。

```javascript
function renderActionReport(data) {
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
      <div class="action-report__overview-title"><span>训练动作概览</span><small>基于本次姿态分析结果</small></div>
      <div class="action-report__overview-grid">
        <div><span>分析片段</span><strong>${summary.clipCount}</strong></div>
        <div><span>分析帧数</span><strong>${summary.totalFrames.toLocaleString("zh-CN")}</strong></div>
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
    if (!persons.length) {
      html += `<div class="action-report__empty-state action-report__empty-state--compact"><strong>未检测到稳定的人体姿态</strong><span>请检查视频画面是否清晰，或尝试分析更短的片段。</span></div>`;
    } else {
      html += `<div class="action-report__people">${persons.map((person, index) => renderActionPersonCard(person, index, analysis)).join("")}</div>`;
      html += renderActionComparison(persons);
    }
    if (rawFrames.length) {
      html += `<details class="action-report__raw"><summary>查看逐帧原始数据 <span>${rawFrames.length} 帧</span></summary><pre>${escapeActionHtml(JSON.stringify(rawFrames, null, 2))}</pre></details>`;
    }
    html += `</section>`;
  });
  html += `<p class="action-report__hint">身份只有在外观跟踪达到置信度阈值时才会确认；未确认时使用跟踪编号区分对象。</p>`;
  $("actionReport").innerHTML = html;
}
```

- [ ] **Step 3: 运行语法和差异检查**

Run: `node --check pingpong_analyst/static/app.js`

Expected: exit code 0.

Run: `git diff --check`

Expected: no output and exit code 0.

### Task 3: 实现报告样式和窄屏布局

**Files:**
- Modify: `pingpong_analyst/static/style.css:1348-1448`，替换旧的 `.action-report__person` 和文本指标规则
- Modify: `pingpong_analyst/static/style.css:1646`，在现有媒体查询中增加动作报告窄屏规则

- [ ] **Step 1: 添加总览、片段头部和选手卡片样式**

使用现有变量和完整边框，保持动作报告在模态框内的深色仪表盘层级。关键结构使用 `grid-template-columns: repeat(auto-fit, minmax(240px, 1fr))`，避免固定两列导致窄屏溢出。

```css
.action-report { display: flex; flex-direction: column; gap: var(--space-lg); }
.action-report__overview,
.action-report__source { padding: var(--space-lg); background: var(--bg-inset); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); }
.action-report__overview-title { display: flex; justify-content: space-between; gap: var(--space-md); align-items: baseline; }
.action-report__overview-title span,
.action-report__source-head h3 { color: var(--text-primary); font-family: var(--font-display); font-size: 15px; font-weight: 600; }
.action-report__overview-title small,
.action-report__source-index,
.action-report__source-meta,
.action-report__hint { color: var(--text-muted); font-family: var(--font-mono); font-size: 10px; }
.action-report__overview-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--space-sm); margin-top: var(--space-lg); }
.action-report__overview-grid div { padding: var(--space-md); background: var(--bg-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); }
.action-report__overview-grid span { display: block; color: var(--text-muted); font-family: var(--font-mono); font-size: 10px; }
.action-report__overview-grid strong { display: block; margin-top: var(--space-xs); color: var(--accent); font-family: var(--font-display); font-size: 22px; }
.action-report__source { display: flex; flex-direction: column; gap: var(--space-lg); }
.action-report__source-head { display: flex; align-items: flex-end; justify-content: space-between; gap: var(--space-md); padding-bottom: var(--space-md); border-bottom: 1px solid var(--border-subtle); }
.action-report__source-head h3 { margin: var(--space-xs) 0 0; overflow-wrap: anywhere; }
.action-report__source-meta { display: flex; flex-wrap: wrap; gap: var(--space-md); justify-content: flex-end; }
.action-report__people { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: var(--space-md); }
.action-person-card { display: flex; flex-direction: column; gap: var(--space-md); padding: var(--space-lg); background: var(--bg-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); }
.action-person-card__head { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--space-md); }
.action-person-card__identity { color: var(--text-primary); font-family: var(--font-display); font-size: 16px; font-weight: 600; }
.action-person-card__track { margin-top: var(--space-xs); color: var(--text-muted); font-family: var(--font-mono); font-size: 10px; }
.action-confidence { padding: 3px 7px; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 10px; }
.action-confidence--high { color: var(--data-green); background: color-mix(in oklch, var(--data-green) 14%, transparent); }
.action-confidence--medium { color: var(--accent); background: color-mix(in oklch, var(--accent) 14%, transparent); }
.action-confidence--low { color: var(--text-muted); background: var(--bg-inset); }
.action-person-card__meta { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-sm); color: var(--text-muted); font-family: var(--font-mono); font-size: 10px; }
.action-person-card__meta span { padding: var(--space-sm); background: var(--bg-inset); border-radius: var(--radius-sm); }
.action-person-card__meta strong { display: block; margin-top: 3px; color: var(--text-secondary); font-size: 12px; }
.action-person-card__section-title,
.action-report__section-heading { color: var(--text-secondary); font-family: var(--font-display); font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }
```

- [ ] **Step 2: 添加指标、对比、空状态和原始数据样式**

指标行使用稳定的两列布局；对比条使用不同的数据色但同时显示文本标签；原始数据设置最大高度并允许滚动，避免长视频撑开模态框。

```css
.action-metrics { display: grid; gap: var(--space-xs); }
.action-metric { display: flex; align-items: center; justify-content: space-between; gap: var(--space-md); min-height: 28px; padding: 5px 0; border-bottom: 1px solid color-mix(in oklch, var(--border-subtle) 60%, transparent); }
.action-metric:last-child { border-bottom: 0; }
.action-metric__label { color: var(--text-secondary); font-size: 11px; }
.action-metric__value { color: var(--data-cyan); font-family: var(--font-mono); font-size: 12px; }
.action-report__no-data { color: var(--text-muted); font-family: var(--font-mono); font-size: 10px; }
.action-person-card__details,
.action-report__raw { border-top: 1px solid var(--border-subtle); padding-top: var(--space-md); }
.action-person-card__details summary,
.action-report__raw summary { cursor: pointer; color: var(--text-muted); font-family: var(--font-mono); font-size: 10px; }
.action-person-card__details summary:hover,
.action-report__raw summary:hover { color: var(--text-primary); }
.action-comparison { display: flex; flex-direction: column; gap: var(--space-md); padding-top: var(--space-sm); }
.action-report__section-heading { display: flex; justify-content: space-between; gap: var(--space-md); color: var(--text-primary); }
.action-report__section-heading span { color: var(--text-muted); font-family: var(--font-mono); font-size: 9px; font-weight: 400; letter-spacing: 0; text-transform: none; }
.action-comparison__rows { display: grid; gap: var(--space-sm); }
.action-comparison__row { display: grid; grid-template-columns: 90px repeat(2, minmax(0, 1fr)); gap: var(--space-md); align-items: center; }
.action-comparison__label { color: var(--text-secondary); font-size: 11px; }
.action-comparison__value { display: grid; gap: 4px; min-width: 0; color: var(--text-muted); font-family: var(--font-mono); font-size: 9px; }
.action-comparison__bar { display: block; height: 5px; overflow: hidden; background: var(--bg-elevated); border-radius: 2px; }
.action-comparison__bar i { display: block; height: 100%; min-width: 4px; background: var(--data-cyan); border-radius: inherit; }
.action-comparison__value--1 .action-comparison__bar i { background: var(--accent); }
.action-report__raw pre { max-height: 240px; margin: var(--space-md) 0 0; padding: var(--space-md); overflow: auto; background: var(--bg-inset); color: var(--text-muted); font-family: var(--font-mono); font-size: 10px; line-height: 1.5; white-space: pre-wrap; overflow-wrap: anywhere; }
.action-report__empty-state { display: flex; flex-direction: column; gap: var(--space-xs); padding: var(--space-xl); color: var(--text-muted); background: var(--bg-inset); border: 1px dashed var(--border-subtle); border-radius: var(--radius-sm); }
.action-report__empty-state strong { color: var(--text-secondary); font-family: var(--font-display); font-size: 13px; }
.action-report__empty-state span { font-family: var(--font-mono); font-size: 10px; line-height: 1.5; }
.action-report__empty-state--compact { padding: var(--space-lg); }
.action-report__hint { margin: 0; line-height: 1.6; }
```

- [ ] **Step 3: 添加窄屏覆盖规则**

在现有 `@media (max-width: 900px)` 中加入以下规则，确保摘要四列和对比三列不会在小屏互相挤压。

```css
.action-report__overview-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.action-report__source-head { align-items: flex-start; flex-direction: column; }
.action-report__source-meta { justify-content: flex-start; }
.action-comparison__row { grid-template-columns: 1fr; gap: var(--space-xs); padding: var(--space-sm) 0; border-bottom: 1px solid var(--border-subtle); }
.action-comparison__value { grid-template-columns: 72px minmax(0, 1fr); align-items: center; }
```

### Task 4: 验证动作报告在真实页面中的行为

**Files:**
- Test: `pingpong_analyst/static/app.js`
- Test: `pingpong_analyst/static/style.css`
- Test: `docs/superpowers/specs/2026-08-03-action-report-design.md`

- [ ] **Step 1: 运行静态检查**

Run: `node --check pingpong_analyst/static/app.js`

Expected: exit code 0.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 2: 运行后端回归测试**

Run: `pytest -q`

Expected: all existing tests pass; this change must not alter the action result API or model behavior.

- [ ] **Step 3: 使用浏览器检查完成态报告**

启动当前服务后打开动作分析入口，使用已有真实视频或一个返回结果的动作任务检查：

```text
1. 完成动作分析并打开“动作分析报告”。
2. 确认顶部显示片段数、分析帧数、跟踪对象数和覆盖率。
3. 确认默认视图只显示中文指标卡，不显示原始 JSON 字段串。
4. 确认两名选手时出现四项手臂角度对比；身份不确定时显示“身份未确认”和跟踪编号。
5. 展开“查看身体指标”和“查看逐帧原始数据”，确认内容可读且不破坏布局。
6. 在桌面宽度和窄屏宽度检查无文字重叠、无横向溢出。
```

- [ ] **Step 4: 检查工作区变更范围**

Run: `git status --short`

Expected: only the intended action-report files and the implementation plan are changed; unrelated user changes remain untouched.

