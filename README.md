<p align="center">
  <img src="docs/assets/banner.png" alt="pingpong-analyst — Smart Cut · Deep Rally Analysis" width="100%" />
</p>

<h1 align="center">pingpong-analyst</h1>

<p align="center">
  <strong>Table tennis video intelligence</strong> — ball tracking, rally segmentation,<br/>
  pose-driven action reports, and highlight composition in one local-first stack.
</p>

<p align="center">
  <a href="#english">English</a> ·
  <a href="#中文">中文</a> ·
  <a href="docs/local-and-server.md">Deployment notes</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/api-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img alt="License" src="https://img.shields.io/badge/license-source%20available-lightgrey?style=flat-square" />
  <img alt="Platform" src="https://img.shields.io/badge/macOS%20MPS%20%7C%20CUDA%20%2F%20TensorRT-ready-F06A1A?style=flat-square" />
</p>

---

<a id="english"></a>

## Overview

**pingpong-analyst** turns raw table tennis footage into reviewable structure:

| Capability | What you get |
| --- | --- |
| **Rally analysis** | TrackNet / CV ball candidates → board crossings → rally clips |
| **Action analysis** | YOLO-Pose + MediaPipe angles, identity association, per-hit metrics |
| **Geometry** | Four-corner table calibration for rectified trajectory decisions |
| **Adaptation** | Manual ball labels → TrackNet fine-tune → versioned model registry |
| **Editing** | Highlight project store + composition export for shareable cuts |
| **Ops** | Auth, workbench, device auto-select (`auto` / MPS / CUDA / TensorRT) |

The UI follows a **tech-dashboard** direction: video is the hero surface; metrics stay secondary. Accent warm orange (ball), telemetry cyan, dark clinical chrome — built for coaches and athletes who need precision, not spectacle.

Rally and action pipelines are **intentionally isolated**: a rally job does not load pose models; an action job does not emit rally clips. That keeps memory predictable on Apple Silicon and on NVIDIA T4.

> **Privacy:** this repository ships **code and docs only**. Videos, SQLite state, token secrets, weights, and exports stay on your machine (see `.gitignore`).

---

## Architecture (at a glance)

```text
Video library (data/)
        │
        ▼
   FastAPI + Static UI  ── WebSocket live overlay
        │
   ┌────┴─────────────────────────────┐
   │                                  │
 Rally pipeline                  Action pipeline
 TrackNet / CV                   YOLO-Pose + MediaPipe
 board crossing / landing        angles & identity
 clip export                     action report
   │                                  │
   └──────────┬───────────────────────┘
              ▼
     Highlight editor → composition render
```

Configuration lives in [`config.yaml`](config.yaml). Runtime artifacts land under `data/`, `output/`, `logs/`, and `models/runs/` — none of which are required to clone the repo.

---

## Requirements

| Environment | Notes |
| --- | --- |
| **Python** | 3.11+ recommended (3.12 used in development) |
| **macOS** | Apple Silicon → PyTorch **MPS**; Intel / no MPS → CPU |
| **Linux + NVIDIA** | CUDA 12.x + optional **TensorRT 8.6+** for engines |
| **FFmpeg** | Available on `PATH` for stream-copy clips / composition |

Model weights are **not** committed. Place them yourself (paths are relative to the project root):

```text
models/yolo11s-pose.pt      # or root-level yolo11s-pose.pt (loader fallback)
models/TrackNet_best.pt     # TrackNetV3 TableTennis checkpoint (preferred)
```

Without TrackNet weights the rally path still runs with a **CV fallback** suitable for development — not production accuracy. See [docs/local-and-server.md](docs/local-and-server.md) for download pointers and TensorRT export shapes.

---

## Quick start (macOS / local)

```bash
git clone https://github.com/zhaopeinan/pingpong-analyst.git
cd pingpong-analyst

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

python -m pip install -U pip
python -m pip install -r requirements-mac.txt

# Optional but recommended for real rallies:
# place TrackNet + YOLO weights under models/ (or project root for YOLO)

python -m pingpong_analyst.api --host 127.0.0.1 --port 8077
```

Open **http://127.0.0.1:8077**.

On first boot an admin account is created:

| Field | Default |
| --- | --- |
| Username | `admin` |
| Password | `admin123` |

**Change this password immediately** in the workbench before exposing the service beyond localhost.

### Smoke checks

```bash
# Health
curl -s http://127.0.0.1:8077/api/health

# Unit / API tests (no large private videos required)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

---

## Server (CUDA / TensorRT)

```bash
python -m pip install -r requirements-server.txt
# Ensure CUDA + TensorRT match your host; install tensorrt wheels accordingly.

# Recommended engines (shapes must match config):
#   YOLO:    Ultralytics export → models/yolo11s-pose.engine
#   TrackNet:[1, 27, 288, 512] → [1, 8, 288, 512]

# Adjust APP_DIR / CONDA_ENV in start.sh, then:
bash start.sh start
bash start.sh status
```

Default bind in `start.sh` is `0.0.0.0:8077`. Treat that as an authenticated internal service — not a public anonymous upload endpoint.

---

## Typical workflow

1. **Sign in** → open the video library → upload or select footage.
2. **Calibrate the table** (fixed camera): four corners in image order TL → TR → BR → BL.
3. **Rally track** to detect board crossings and export rally clips.
4. **Action analyze** on the full video or a selected clip for pose / angle reports.
5. **Annotate & adapt** TrackNet when your venue lighting breaks the base model; save a versioned run and select it explicitly.
6. **Highlight editor** to assemble segments and render a composition.

CLI entry points remain available via `python -m pingpong_analyst` / package modules for scripting; the browser UI is the primary operator surface.

---

## Configuration

Key knobs in `config.yaml`:

- `device.mode`: `auto` | `cpu` | `mps` | `cuda` | `tensorrt`
- `models.tracknet.*`: weights, temporal window, center-region filter
- `analysis.rally.*`: board thresholds, timeouts, table corners
- `video.*` / `composition.*`: clip buffer, encoder, export paths

Per-video table calibration and TrackNet annotations persist under `data/` so re-analysis does not require re-clicking geometry.

---

## Repository layout

```text
pingpong_analyst/          Application package (API, core, models, static UI)
config.yaml                Runtime configuration
requirements-mac.txt       Local / Apple Silicon deps
requirements-server.txt    CUDA + TensorRT-oriented deps
start.sh                   Simple process supervisor for GPU hosts
docs/                      Design specs, plans, deployment notes
tests/                     Pytest suite
```

---

## Security notes

- Default admin credentials exist only to bootstrap an empty database — rotate on first login.
- JWT signing material is generated under `data/.token_secret` and must **never** be committed.
- Do not publish `data/`, `output/`, weight archives, or production databases.

---

## License & attribution

Source is published for inspection and local deployment. Upstream model weights (YOLO, TrackNetV3 TableTennis, MediaPipe) remain under their respective licenses — obtain and comply separately.

---

<a id="中文"></a>

## 概述

**pingpong-analyst** 把原始乒乓球训练 / 比赛视频变成可复盘的结构化结果：

| 能力 | 产出 |
| --- | --- |
| **回合分析** | TrackNet / 纯 CV 球候选 → 过板检测 → 回合剪辑 |
| **动作分析** | YOLO-Pose + MediaPipe 关节角、身份关联、击球指标 |
| **几何标定** | 球台四角标定，轨迹决策可走台面坐标系 |
| **模型适配** | 人工标球 → TrackNet 微调 → 可选择的版本化模型 |
| **成片编辑** | 精彩分段工程 + 合成导出 |
| **运维** | 账号体系、工作台、设备自动选择（`auto` / MPS / CUDA / TensorRT） |

产品调性是**科技仪表盘**：画面是主角，指标是配角；暖橙强调球迹，冷青服务遥测，深色临床质感——面向教练与运动员的精度需求，而不是演示特效。

回合管线与动作管线**刻意隔离**：回合任务不加载姿态模型，动作任务不产出回合剪辑，便于在 Apple Silicon 与 NVIDIA T4 上控制显存。

> **隐私：** 本仓库只发布**代码与文档**。视频、数据库、签名密钥、权重与导出文件均保留在本地（见 `.gitignore`）。

---

## 架构速览

```text
视频库 (data/)
      │
      ▼
 FastAPI + 静态前端 ── WebSocket 实时叠层
      │
 ┌────┴──────────────────────────┐
 │                               │
 回合管线                         动作管线
 TrackNet / CV                    YOLO-Pose + MediaPipe
 过板 / 落点                       角度与身份
 剪辑导出                         动作报告
 │                               │
 └──────────┬────────────────────┘
            ▼
     精彩编辑器 → 成片渲染
```

配置见 [`config.yaml`](config.yaml)。运行产物位于 `data/`、`output/`、`logs/`、`models/runs/`，克隆仓库时不需要这些目录。

---

## 环境要求

| 环境 | 说明 |
| --- | --- |
| **Python** | 建议 3.11+（开发环境为 3.12） |
| **macOS** | Apple Silicon → PyTorch **MPS**；无 MPS 则回退 CPU |
| **Linux + NVIDIA** | CUDA 12.x，可选 **TensorRT 8.6+** |
| **FFmpeg** | 需在 `PATH` 中，用于无损剪辑与成片 |

**权重不入库**，请自行放置（路径相对项目根目录）：

```text
models/yolo11s-pose.pt
models/TrackNet_best.pt
```

缺少 TrackNet 权重时，回合路径会走**纯 CV 回退**（适合开发联调，不等于正式精度）。下载与 TensorRT 导出形状见 [docs/local-and-server.md](docs/local-and-server.md)。

---

## 快速开始（macOS / 本地）

```bash
git clone https://github.com/zhaopeinan/pingpong-analyst.git
cd pingpong-analyst

python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements-mac.txt

# 建议放入 TrackNet / YOLO 权重后再做真实视频分析

python -m pingpong_analyst.api --host 127.0.0.1 --port 8077
```

浏览器打开 **http://127.0.0.1:8077**。

首次启动会创建默认管理员：

| 字段 | 默认值 |
| --- | --- |
| 用户名 | `admin` |
| 密码 | `admin123` |

**请立刻在工作台修改密码**；不要把默认口令暴露到公网。

### 冒烟验证

```bash
curl -s http://127.0.0.1:8077/api/health

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

---

## 服务器（CUDA / TensorRT）

```bash
python -m pip install -r requirements-server.txt

# 按需导出 engine，形状需与 config 一致
# TrackNet: [1, 27, 288, 512] → [1, 8, 288, 512]

# 修改 start.sh 中的 APP_DIR / CONDA_ENV 后：
bash start.sh start
bash start.sh status
```

`start.sh` 默认监听 `0.0.0.0:8077`。请按**内网已认证服务**对待，而不是匿名公网上传入口。

---

## 推荐操作流

1. **登录** → 视频库上传 / 选择素材  
2. **球台标定**（固定机位）：图像坐标顺序 左上 → 右上 → 右下 → 左下  
3. **回合追踪**，生成过板事件与回合剪辑  
4. **动作分析**（整段或某一剪辑），查看姿态与角度报告  
5. **标球微调** TrackNet，保存版本后在下拉框中**显式选择**  
6. **精彩编辑器** 编排分段并导出成片  

---

## 配置与安全

- 设备与模型阈值：`config.yaml`  
- JWT 密钥：`data/.token_secret`（自动生成，禁止提交）  
- 勿公开 `data/`、`output/`、权重压缩包或生产库  

更细的本地 / T4 说明、标定示例与适配流程见 **[docs/local-and-server.md](docs/local-and-server.md)**。

---

## 目录结构

```text
pingpong_analyst/     应用包（API、核心算法、模型封装、前端）
config.yaml           运行配置
requirements-*.txt    依赖清单
start.sh              GPU 主机进程脚本
docs/                 设计文档与部署说明
tests/                测试
```

---

<p align="center">
  <sub>Built for clinical review — not for dashboard theatre.</sub>
</p>
