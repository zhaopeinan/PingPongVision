"""
轻量 API 服务
提供视频分析、实时追踪可视化、自动剪辑的 HTTP 接口

启动: python -m pingpong_analyst.api --host 0.0.0.0 --port 8077
"""
import argparse
import asyncio
import json
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request, UploadFile, File, BackgroundTasks, Query, WebSocket, WebSocketDisconnect, Body, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .core import ActionAnalyzer, VideoAnalyzer
from .core.video_analyzer import PipelineStuckError
from .core import database as db
from .core.auth import (
    create_token,
    ensure_initial_admin,
    get_current_user,
    hash_password,
    require_admin,
    update_last_login,
    verify_password,
    verify_token,
)
from .core.captcha import (
    check_rate_limit,
    clear_login_failures,
    generate_captcha,
    record_login_failure,
    verify_captcha,
)
from .core.composition_renderer import CompositionError, CompositionRenderer
from .core.edit_project_store import (
    EditProjectStore,
    ProjectNotFoundError,
    ProjectValidationError,
)
from .core.tracknet_annotations import BallAnnotation, TrackNetAnnotationStore
from .core.table_calibration import TableCalibration, TableCalibrationStore
from .models.tracknet_finetune import TrackNetFineTuner
from .models.tracknet_registry import TrackNetModelRegistry
from .utils.config import get_config, resolve_project_path
from .utils.device_manager import DeviceManager
from .tracking_visualizer import TrackingVisualizer, generate_mjpeg_stream


app = FastAPI(title="PingPong AI Analyst", version="0.2.0")

VIDEO_LIBRARY_DIR = Path("data")
OUTPUT_DIR = Path("output/clips")
EDIT_PROJECT_DIR = Path("data/edit_projects")
COMPOSITION_OUTPUT_DIR = Path("output/exports")
STATIC_DIR = Path(__file__).parent / "static"
VIDEO_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EDIT_PROJECT_DIR.mkdir(parents=True, exist_ok=True)
COMPOSITION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

_video_registry: dict[str, dict] = {}
_task_results: dict[str, dict] = {}
_clip_registry: dict[str, list[str]] = {}
_cancel_events: dict[str, threading.Event] = {}
_analysis_lock = threading.Lock()
_task_results_lock = threading.Lock()
_edit_project_creation_lock = threading.Lock()
_render_lock = threading.Lock()
_training_jobs: dict[str, dict] = {}
_training_lock = threading.Lock()

# These defaults are replaceable in tests and keep all adaptation data outside
# source videos and the immutable base checkpoint.
ANNOTATION_DIR = Path("data/tracknet_annotations")
TABLE_CALIBRATION_DIR = Path("data/table_calibrations")
MODEL_REGISTRY_DIR = Path("models/runs")
TRACKNET_BASE_PATH: Path | None = None


def _normalize_playback_speed(value: str | float | None) -> float:
    """将播放速度限制在可控范围，避免异常客户端拖垮发送循环。"""
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(speed):
        return 1.0
    return round(min(max(speed, 0.1), 4.0), 2)


def _normalize_crossing_timeout(value: str | float | None) -> float:
    """Normalize the user-controlled score-segment timeout."""
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return 4.0
    if not math.isfinite(timeout):
        return 4.0
    return round(min(max(timeout, 0.5), 10.0), 1)

# 初始化数据库
db.init_db()

# 首次启动创建默认管理员
ensure_initial_admin()

# 启动时从数据库恢复视频注册表
for v in db.list_videos():
    if Path(v["path"]).exists():
        _video_registry[v["video_id"]] = v


def _scan_library():
    """扫描视频库目录，注册所有视频文件（增量扫描，已注册的跳过）。"""
    for f in VIDEO_LIBRARY_DIR.rglob("*"):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            # 先查数据库是否已有此路径
            existing = db.get_video_by_path(str(f))
            if existing:
                # 同步到内存（重启后首次扫描需要）
                _video_registry[existing["video_id"]] = existing
                continue
            # 新文件：用稳定 ID 注册
            size = f.stat().st_size
            vid = db.stable_video_id(f.name, size)
            info = _probe_video(str(f))
            record = db.upsert_video(vid, f.name, str(f), size, info)
            _video_registry[vid] = record


def get_analyzer(
    tracknet_model_path: str | None = None,
    table_calibration: TableCalibration | None = None,
    no_crossing_timeout_seconds: float | None = None,
) -> VideoAnalyzer:
    """创建一次性分析器, 避免跨任务共享时序状态和模型对象。"""
    return VideoAnalyzer(
        tracknet_model_path=tracknet_model_path,
        table_calibration=table_calibration,
        no_crossing_timeout_seconds=no_crossing_timeout_seconds,
    )


def _update_task_result(task_id: str, **updates) -> None:
    """原子更新后台任务状态，避免轮询读到半截结果。"""
    with _task_results_lock:
        result = dict(_task_results.get(task_id, {}))
        result.update(updates)
        _task_results[task_id] = result
    # 同步到数据库（忽略 clips/result 等大字段，只持久化状态信息）
    db_fields: dict[str, Any] = {}
    for key in ("status", "progress", "message", "project_id"):
        if key in updates:
            db_fields[key] = updates[key]
    if "clips" in updates:
        db_fields["clips"] = updates["clips"]
    if db_fields:
        db.update_task(task_id, **db_fields)


# ---------- 步骤追踪 ----------

_RALLY_STEPS = [
    ("load_model", "加载 TrackNet 模型"),
    ("sample_bg", "采样视频背景"),
    ("analyze_frames", "分析视频帧"),
    ("detect_rallies", "检测回合分段"),
    ("export_clips", "生成得分段剪辑"),
    ("save_results", "保存分析结果"),
]

_ACTION_STEPS = [
    ("load_model", "加载姿态模型"),
    ("analyze_clips", "分析动作片段"),
    ("save_results", "保存分析结果"),
]


def _init_steps(task_id: str, steps: list[tuple[str, str]]) -> None:
    """初始化任务步骤列表。"""
    _update_task_result(
        task_id,
        steps=[
            {"key": k, "name": n, "status": "pending", "progress": 0}
            for k, n in steps
        ],
    )


def _update_step(task_id: str, step_key: str, status: str, progress: int = 0) -> None:
    """更新某个步骤的状态。status: pending/processing/completed/failed"""
    with _task_results_lock:
        result = _task_results.get(task_id, {})
        steps = list(result.get("steps", []))
        for s in steps:
            if s["key"] == step_key:
                s["status"] = status
                s["progress"] = progress
                break
        _task_results[task_id] = {**result, "steps": steps}


def _edit_project_store() -> EditProjectStore:
    """Create a store instance from the patchable API data directory."""
    return EditProjectStore(EDIT_PROJECT_DIR)


def _source_storage_name(video_id: str) -> str:
    """Return the actual file name used on disk for a registered video."""
    entry = _video_registry.get(video_id)
    if not entry:
        raise LookupError("video not found")
    return Path(entry["path"]).name


def _resolve_project_source(project: dict) -> Path:
    """Resolve a persisted project to a currently available library file.

    ``video_id`` is an in-memory identifier and changes after a service
    restart.  The physical storage name is therefore the durable lookup key;
    registry entries are still preferred so the API never accepts a client
    supplied path.
    """
    storage_name = project["source_storage_name"]
    current = _video_registry.get(project.get("video_id"))
    candidates = []
    if current:
        candidates.append(Path(current["path"]))

    _scan_library()
    candidates.extend(
        Path(entry["path"])
        for entry in _video_registry.values()
        if Path(entry["path"]).name == storage_name
    )

    # This fallback supports a project after a restart before the library has
    # been requested, while still keeping the path below VIDEO_LIBRARY_DIR.
    library_root = VIDEO_LIBRARY_DIR.resolve()
    direct_path = (VIDEO_LIBRARY_DIR / storage_name).resolve()
    if library_root in direct_path.parents:
        candidates.append(direct_path)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or resolved.name != storage_name:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"源视频不存在: {storage_name}")


def _project_id() -> str:
    return f"edit_{uuid.uuid4().hex[:12]}"


def _clip_buffer_values() -> tuple[float, float]:
    config = get_config()
    before = config.get("video", "clip_buffer_before", default=0.5)
    after = config.get("video", "clip_buffer_after", default=0.5)
    return float(before), float(after)


def _serialize_hit_events(hits) -> list[dict]:
    payload = []
    for hit in hits or []:
        if hasattr(hit, "to_dict"):
            payload.append(hit.to_dict())
        elif isinstance(hit, dict):
            payload.append(hit)
    return payload


def _serialize_rallies(segments) -> list[dict]:
    rallies = []
    for index, segment in enumerate(segments, start=1):
        rallies.append({
            "index": index,
            "start_time": round(float(segment.start_time), 2),
            "end_time": round(float(segment.end_time), 2),
            "board_count": int(segment.board_count),
            "duration": round(float(segment.duration), 2),
            "hit_events": _serialize_hit_events(getattr(segment, "hit_events", [])),
        })
    return rallies


def _rally_payload_for_video(video_id: str) -> dict:
    """读取该视频最近一次板数分析的得分段和剪辑。"""
    with _task_results_lock:
        memory = dict(_task_results.get(video_id, {}))
    rallies = memory.get("rallies") or []
    clips = _clip_registry.get(video_id) or memory.get("clips") or []
    if not rallies or not clips:
        task = db.get_task(video_id)
        if task:
            task_result = task.get("result") or {}
            if not rallies:
                rallies = task_result.get("rallies") or []
            if not clips:
                clips = task.get("clips") or []
    return {"rallies": rallies, "clips": [Path(name).name for name in clips]}


def _clip_relative_hits(rally: dict | None, buffer_before: float) -> list[dict]:
    if not rally:
        return []
    start = float(rally.get("start_time") or 0.0) - float(buffer_before)
    hits = []
    for hit in rally.get("hit_events") or []:
        item = hit.to_dict() if hasattr(hit, "to_dict") else dict(hit)
        source_ts = float(item.get("timestamp") or 0.0)
        item["source_timestamp"] = source_ts
        item["timestamp"] = round(source_ts - start, 3)
        hits.append(item)
    return hits


def create_edit_project_from_analysis(task_id: str) -> dict:
    """Create or recover an editor project for a completed rally task."""
    with _edit_project_creation_lock:
        with _task_results_lock:
            result = dict(_task_results.get(task_id, {}))
        if result.get("project_id"):
            return _edit_project_store().get(result["project_id"])
        if result.get("status") != "completed" or result.get("mode") != "rally":
            raise LookupError("没有可用的已完成板数分析结果")
        entry = _video_registry.get(task_id)
        if not entry:
            raise LookupError("分析源视频不存在")

        before, after = _clip_buffer_values()
        project = _edit_project_store().create_from_analysis(
            project_id=_project_id(),
            video_id=task_id,
            source_filename=Path(entry.get("filename", entry["path"])).name,
            source_storage_name=Path(entry["path"]).name,
            source_duration=float(entry.get("info", {}).get("duration", 0.0)),
            rallies=result.get("rallies", []),
            clips=result.get("clips", []),
            buffer_before=before,
            buffer_after=after,
        )
        _update_task_result(task_id, project_id=project["project_id"])
        return project


def create_edit_project_from_segments(
    task_id: str,
    video_path: str,
    segments: list,
    outputs: list[str],
) -> dict:
    """Persist the live analysis output using the same schema as old tasks."""
    with _edit_project_creation_lock:
        with _task_results_lock:
            result = dict(_task_results.get(task_id, {}))
        if result.get("project_id"):
            return _edit_project_store().get(result["project_id"])
        entry = _video_registry.get(task_id, {})
        info = entry.get("info", {})
        rallies = [
            {
                "index": index + 1,
                "start_time": float(segment.start_time),
                "end_time": float(segment.end_time),
                "board_count": int(segment.board_count),
            }
            for index, segment in enumerate(segments)
        ]
        before, after = _clip_buffer_values()
        project = _edit_project_store().create_from_analysis(
            project_id=_project_id(),
            video_id=task_id,
            source_filename=Path(entry.get("filename", video_path)).name,
            source_storage_name=Path(video_path).name,
            source_duration=float(info.get("duration", 0.0)),
            rallies=rallies,
            clips=[Path(path).name for path in outputs],
            buffer_before=before,
            buffer_after=after,
        )
        _update_task_result(task_id, project_id=project["project_id"])
        return project


def _render_status_payload(project: dict) -> dict:
    return {"project_id": project["project_id"], **dict(project["render"])}


def _annotation_store() -> TrackNetAnnotationStore:
    return TrackNetAnnotationStore(ANNOTATION_DIR)


def _table_calibration_store() -> TableCalibrationStore:
    return TableCalibrationStore(TABLE_CALIBRATION_DIR)


def _required_table_calibration(video_id: str) -> TableCalibration:
    """Return a video's table calibration or raise a user-facing error."""
    try:
        calibration = _table_calibration_store().load(video_id)
    except ValueError as exc:
        raise LookupError(str(exc)) from exc
    if calibration is None:
        raise LookupError("请先标定球台四个角点，再开始板数分析")
    return calibration


def _parse_table_calibration(video_id: str, payload) -> TableCalibration:
    if not isinstance(payload, dict):
        raise ValueError("table calibration 必须是对象")
    calibration = TableCalibration.from_dict(payload)
    info = _video_registry[video_id].get("info", {})
    total_frames = int(info.get("total_frames", 0))
    width = int(info.get("width", 0))
    height = int(info.get("height", 0))
    if total_frames and calibration.frame_index >= total_frames:
        raise ValueError("frame_index 超出视频范围")
    for point in (*calibration.corners, *calibration.net_points):
        if width and not 0 <= point[0] < width:
            raise ValueError("标定点 x 超出视频尺寸")
        if height and not 0 <= point[1] < height:
            raise ValueError("标定点 y 超出视频尺寸")
    return calibration


def _tracknet_registry() -> TrackNetModelRegistry:
    config = get_config()
    base_value = TRACKNET_BASE_PATH or config.get("models", "tracknet", "weights", default="models/TrackNet_best.pt")
    base_path = resolve_project_path(base_value) or Path("models/TrackNet_best.pt").resolve()
    registry_dir = MODEL_REGISTRY_DIR
    if registry_dir == Path("models/runs"):
        configured = config.get("models", "tracknet_adaptation", "registry_dir", default=None)
        if configured:
            registry_dir = Path(configured)
    registry_dir = resolve_project_path(registry_dir) or registry_dir
    return TrackNetModelRegistry(registry_dir, base_path)


def _tracknet_training_config() -> dict:
    config = get_config()
    values = dict(config.get("models", "tracknet", default={}))
    values.update(config.get("models", "tracknet_adaptation", default={}) or {})
    return values


def _parse_tracknet_annotations(video_id: str, payload) -> list[BallAnnotation]:
    if isinstance(payload, dict):
        raw_annotations = payload.get("annotations")
    else:
        raw_annotations = payload
    if not isinstance(raw_annotations, list):
        raise ValueError("annotations 必须是数组")
    info = _video_registry[video_id].get("info", {})
    total_frames = int(info.get("total_frames", 0))
    width = int(info.get("width", 0))
    height = int(info.get("height", 0))
    annotations: list[BallAnnotation] = []
    for raw in raw_annotations:
        annotation = raw if isinstance(raw, BallAnnotation) else BallAnnotation.from_dict(raw)
        if total_frames and annotation.frame_index >= total_frames:
            raise ValueError("frame_index 超出视频范围")
        if annotation.label == "ball" and width and height and (
            annotation.x >= width or annotation.y >= height
        ):
            raise ValueError("球坐标超出视频尺寸")
        annotations.append(annotation)
    return annotations


def _resolve_tracknet_model(model_id: str):
    registry = _tracknet_registry()
    try:
        return registry.resolve(model_id)
    except PermissionError as exc:
        raise RuntimeError(str(exc)) from exc
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise LookupError(str(exc)) from exc


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "PingPong AI Analyst", "version": "0.2.0"}


# ---------- 认证 ----------

_PUBLIC_API_PATHS = {"/api/health", "/api/auth/login", "/api/auth/captcha", "/api/auth/status"}

# 媒体文件服务路径：浏览器 <video>/<img> 标签无法携带 Authorization header，
# 这些端点提供的是视频/图片文件而非敏感数据，跳过认证。
_MEDIA_PATH_PREFIXES = (
    "/api/clips/",       # 剪辑片段播放/下载
    "/api/stream/",      # MJPEG 实时追踪流
    "/api/edit-projects/",  # 包含 export 下载（按 project_id 限制，非敏感）
)
_MEDIA_PATH_SUFFIXES = (
    "/preview",          # 视频预览帧
    "/frame",            # 视频单帧截图
    "/export",           # 成片导出下载
    "/download",         # 原始视频下载
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """拦截 /api/ 请求，验证 JWT 令牌（公开路径和媒体文件除外）。"""
    path = request.url.path
    # WebSocket 路径跳过 HTTP 中间件（在端点内单独验证 token）
    if path.startswith("/api/ws/"):
        return await call_next(request)
    # 媒体文件服务路径跳过认证
    if any(path.startswith(p) for p in _MEDIA_PATH_PREFIXES):
        return await call_next(request)
    if any(path.endswith(s) for s in _MEDIA_PATH_SUFFIXES):
        return await call_next(request)
    if path.startswith("/api/") and path not in _PUBLIC_API_PATHS:
        # 优先从 Authorization header 读取，回退到 query param（用于 <video>/<img> 标签）
        token = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        else:
            token = request.query_params.get("token")
        if token:
            payload = verify_token(token)
            if payload:
                user = db.get_user_by_id(payload["user_id"])
                if user and user["active"]:
                    return await call_next(request)
        return JSONResponse(
            status_code=401,
            content={"detail": "未认证或认证已过期"},
        )
    return await call_next(request)


@app.get("/api/auth/captcha")
async def get_captcha():
    """获取登录验证码图片。"""
    return generate_captcha()


@app.get("/api/auth/status")
async def auth_status():
    """返回认证状态（是否需要初始化）。"""
    return {"needs_init": db.count_users() == 0}


@app.post("/api/auth/login")
async def login(request: Request, body: dict = Body(...)):
    """用户登录，需验证码，返回 JWT 令牌。"""
    username = body.get("username", "").strip()
    password = body.get("password", "")
    captcha_id = body.get("captcha_id", "")
    captcha_text = body.get("captcha_text", "")
    client_ip = request.client.host if request.client else "unknown"

    # 验证码校验
    if not captcha_id or not verify_captcha(captcha_id, captcha_text):
        return JSONResponse(status_code=400, content={"detail": "验证码错误或已过期"})

    if not username or not password:
        return JSONResponse(status_code=400, content={"detail": "用户名和密码不能为空"})

    # 限流检查
    allowed, retry = check_rate_limit(client_ip, username)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": f"登录失败次数过多，请 {retry} 秒后重试"},
        )

    user = db.get_user_by_username(username)
    if user is None or not user["active"]:
        remaining = record_login_failure(client_ip, username)
        return JSONResponse(
            status_code=401,
            content={"detail": f"用户名或密码错误（剩余 {remaining} 次尝试）"},
        )
    if not verify_password(password, user["password_hash"], user["password_salt"]):
        remaining = record_login_failure(client_ip, username)
        return JSONResponse(
            status_code=401,
            content={"detail": f"用户名或密码错误（剩余 {remaining} 次尝试）"},
        )

    clear_login_failures(client_ip, username)
    update_last_login(user["id"])
    token = create_token(user)
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
        },
    }


@app.get("/api/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    """获取当前登录用户信息。"""
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "created_at": user["created_at"],
        "last_login": user["last_login"],
    }


# ---------- 用户管理 (admin) ----------

@app.get("/api/admin/users")
async def admin_list_users(_: dict = Depends(require_admin)):
    """列出所有用户（仅管理员）。"""
    users = db.list_users()
    return {
        "total": len(users),
        "users": [
            {
                "id": u["id"],
                "username": u["username"],
                "role": u["role"],
                "active": u["active"],
                "created_at": u["created_at"],
                "last_login": u["last_login"],
            }
            for u in users
        ],
    }


@app.post("/api/admin/users")
async def admin_create_user(
    body: dict = Body(...),
    _: dict = Depends(require_admin),
):
    """创建新用户（仅管理员）。"""
    username = body.get("username", "").strip()
    password = body.get("password", "")
    role = body.get("role", "analyst")
    if not username or not password:
        return JSONResponse(status_code=400, content={"detail": "用户名和密码不能为空"})
    if role not in ("admin", "analyst"):
        return JSONResponse(status_code=400, content={"detail": "角色必须是 admin 或 analyst"})
    if len(password) < 8:
        return JSONResponse(status_code=400, content={"detail": "密码至少 8 位"})
    if db.get_user_by_username(username):
        return JSONResponse(status_code=409, content={"detail": "用户名已存在"})
    pw_hash, pw_salt = hash_password(password)
    user = db.create_user(username, pw_hash, pw_salt, role=role)
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "active": user["active"],
    }


@app.patch("/api/admin/users/{user_id}")
async def admin_update_user(
    user_id: int,
    body: dict = Body(...),
    current: dict = Depends(require_admin),
):
    """更新用户信息（仅管理员）。"""
    target = db.get_user_by_id(user_id)
    if target is None:
        return JSONResponse(status_code=404, content={"detail": "用户不存在"})
    fields: dict[str, Any] = {}
    if "role" in body:
        if body["role"] not in ("admin", "analyst"):
            return JSONResponse(status_code=400, content={"detail": "角色必须是 admin 或 analyst"})
        fields["role"] = body["role"]
    if "active" in body:
        fields["active"] = 1 if body["active"] else 0
    if "password" in body:
        if len(body["password"]) < 8:
            return JSONResponse(status_code=400, content={"detail": "密码至少 8 位"})
        pw_hash, pw_salt = hash_password(body["password"])
        fields["password_hash"] = pw_hash
        fields["password_salt"] = pw_salt
    if not fields:
        return JSONResponse(status_code=400, content={"detail": "没有需要更新的字段"})
    # 防止管理员把自己禁用或降级
    if current["id"] == user_id:
        if "active" in fields and fields["active"] == 0:
            return JSONResponse(status_code=400, content={"detail": "不能禁用自己的账号"})
        if "role" in fields and fields["role"] != "admin":
            return JSONResponse(status_code=400, content={"detail": "不能降低自己的角色"})
    db.update_user(user_id, **fields)
    updated = db.get_user_by_id(user_id)
    return {
        "id": updated["id"],
        "username": updated["username"],
        "role": updated["role"],
        "active": updated["active"],
    }


@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(
    user_id: int,
    current: dict = Depends(require_admin),
):
    """删除用户（仅管理员）。"""
    if current["id"] == user_id:
        return JSONResponse(status_code=400, content={"detail": "不能删除自己的账号"})
    target = db.get_user_by_id(user_id)
    if target is None:
        return JSONResponse(status_code=404, content={"detail": "用户不存在"})
    db.delete_user(user_id)
    return {"detail": "用户已删除"}


@app.get("/api/device")
async def device_info():
    from .utils.device_manager import DeviceManager
    info = DeviceManager.get_info()
    return {
        "device_type": info.device_type.value,
        "device_name": info.device_name,
        "torch_device": info.torch_device,
        "total_memory_mb": info.total_memory_mb,
        "supports_half": info.supports_half,
        "runtime": DeviceManager.get_runtime_metrics(),
    }


@app.get("/api/device/metrics")
async def device_metrics():
    """Return live accelerator metrics for the one-second header poll."""
    from .utils.device_manager import DeviceManager

    info = DeviceManager.get_info()
    return {
        "device_type": info.device_type.value,
        "device_name": info.device_name,
        "runtime": DeviceManager.get_runtime_metrics(),
    }


def _probe_video(video_path: str) -> dict:
    """探测视频元数据"""
    try:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        total_frames = stream.frames or 0
        width = stream.width
        height = stream.height
        duration = total_frames / fps if fps > 0 else 0
        container.close()
        return {
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration": round(duration, 2),
        }
    except Exception as e:
        logger.warning(f"视频探测失败: {e}")
        return {"width": 0, "height": 0, "fps": 0, "total_frames": 0, "duration": 0}


# 启动时扫描视频库
_scan_library()


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """上传视频文件（永久保存到视频库）"""
    safe_name = file.filename.replace(" ", "_")
    content = await file.read()
    video_path = VIDEO_LIBRARY_DIR / safe_name

    # 重名时加随机前缀
    if video_path.exists():
        prefix = str(uuid.uuid4())[:8]
        video_path = VIDEO_LIBRARY_DIR / f"{prefix}_{safe_name}"

    with open(video_path, "wb") as f:
        f.write(content)

    video_info = _probe_video(str(video_path))
    video_id = db.stable_video_id(video_path.name, len(content))
    record = db.upsert_video(video_id, safe_name, str(video_path), len(content), video_info)
    _video_registry[video_id] = record
    logger.info(f"视频上传: {video_id} -> {video_path} ({len(content)} bytes)")

    return {"video_id": video_id, "filename": safe_name, "size": len(content), "info": video_info}


# ---------- 分片上传（断点续传） ----------

CHUNK_UPLOAD_DIR = Path("data") / ".chunks"
CHUNK_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_CHUNK_SIZE = 5 * 1024 * 1024  # 5MB per chunk


@app.post("/api/upload/init")
async def upload_init(filename: str = Form(...), total_size: int = Form(...)):
    """初始化分片上传会话，返回 upload_id 和已上传的分片列表。"""
    safe_name = filename.replace(" ", "_")
    # 生成 upload_id
    upload_id = f"{uuid.uuid4().hex[:16]}_{safe_name}"
    session_dir = CHUNK_UPLOAD_DIR / upload_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # 写入元数据
    meta = {"filename": safe_name, "total_size": total_size, "chunk_size": _CHUNK_SIZE}
    (session_dir / "meta.json").write_text(json.dumps(meta))

    # 检查已上传的分片（断点续传）
    uploaded = set()
    for f in session_dir.iterdir():
        if f.name.startswith("chunk_"):
            try:
                uploaded.add(int(f.name.split("_")[1]))
            except (ValueError, IndexError):
                pass

    return {"upload_id": upload_id, "chunk_size": _CHUNK_SIZE, "uploaded_chunks": sorted(uploaded)}


@app.post("/api/upload/chunk")
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...),
):
    """上传单个分片。"""
    session_dir = CHUNK_UPLOAD_DIR / upload_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"error": "upload session not found"})

    chunk_path = session_dir / f"chunk_{chunk_index}"
    if chunk_path.exists():
        # 已上传过，跳过（幂等）
        return {"ok": True, "chunk_index": chunk_index, "skipped": True}

    content = await chunk.read()
    chunk_path.write_bytes(content)
    return {"ok": True, "chunk_index": chunk_index, "size": len(content)}


@app.post("/api/upload/complete")
async def upload_complete(
    upload_id: str = Form(...),
):
    """所有分片上传完成，合并文件并注册到视频库。"""
    session_dir = CHUNK_UPLOAD_DIR / upload_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"error": "upload session not found"})

    meta = json.loads((session_dir / "meta.json").read_text())
    safe_name = meta["filename"]
    total_size = meta["total_size"]

    # 合并分片
    chunk_files = sorted(
        [f for f in session_dir.iterdir() if f.name.startswith("chunk_")],
        key=lambda f: int(f.name.split("_")[1]),
    )

    video_path = VIDEO_LIBRARY_DIR / safe_name
    if video_path.exists():
        prefix = uuid.uuid4().hex[:8]
        video_path = VIDEO_LIBRARY_DIR / f"{prefix}_{safe_name}"

    merged_size = 0
    with open(video_path, "wb") as out:
        for cf in chunk_files:
            data = cf.read_bytes()
            out.write(data)
            merged_size += len(data)

    # 清理分片
    import shutil
    shutil.rmtree(session_dir, ignore_errors=True)

    if merged_size != total_size:
        logger.warning(f"分片合并大小不匹配: {merged_size} != {total_size}")

    video_info = _probe_video(str(video_path))
    video_id = db.stable_video_id(video_path.name, merged_size)
    record = db.upsert_video(video_id, safe_name, str(video_path), merged_size, video_info)
    _video_registry[video_id] = record
    logger.info(f"分片上传完成: {video_id} -> {video_path} ({merged_size} bytes)")

    return {"video_id": video_id, "filename": safe_name, "size": merged_size, "info": video_info}


@app.delete("/api/upload/abort/{upload_id}")
async def upload_abort(upload_id: str):
    """取消上传，清理分片。"""
    import shutil
    session_dir = CHUNK_UPLOAD_DIR / upload_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
    return {"ok": True}


@app.get("/api/library")
async def list_library():
    """列出视频库中所有可用视频"""
    # 刷新扫描，注册新加入的文件
    _scan_library()

    videos = []
    for vid, v in _video_registry.items():
        path = Path(v["path"])
        if not path.exists():
            continue
        info = v.get("info", {})
        if not info:
            info = _probe_video(str(path))
            v["info"] = info
        videos.append({
            "video_id": vid,
            "filename": v["filename"],
            "size": path.stat().st_size,
            "info": info,
        })

    # 按文件名排序
    videos.sort(key=lambda x: x["filename"])
    return {"total": len(videos), "videos": videos}


@app.get("/api/videos/{video_id}/preview")
async def video_preview(video_id: str):
    """获取视频第一帧并检测人物，返回JPEG图片+人物边界框"""
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})

    video_path = _video_registry[video_id]["path"]
    info = _video_registry[video_id].get("info", {})

    try:
        import av
        import cv2
        container = av.open(video_path)
        frame = next(container.decode(video=0))
        img = frame.to_ndarray(format="bgr24")
        container.close()

        # 缩放到960宽
        h, w = img.shape[:2]
        scale = 1.0
        target_width = 960
        if w > target_width:
            scale = target_width / w
            img = cv2.resize(img, (target_width, int(h * scale)))

        # 检测人物
        persons = []
        try:
            from .models import YOLOPoseDetector
            from .utils.device_manager import DeviceManager
            device_info = DeviceManager.detect(get_config().device_mode)
            yolo_cfg = get_config().get("models", "yolo_pose", default={})
            detector = YOLOPoseDetector(device_info, yolo_cfg)
            detector.load()
            detected = detector.extract_keypoints(img)
            detector.unload()
            for i, p in enumerate(detected):
                bbox = p.get("bbox")
                if bbox:
                    persons.append({
                        "id": i,
                        "bbox": [round(b, 1) for b in bbox],
                        "conf": round(p.get("conf", 0), 2),
                    })
        except Exception as e:
            logger.warning(f"预览检测失败: {e}")

        # 编码JPEG
        import base64
        _, jpeg_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")

        return JSONResponse({
            "image": "data:image/jpeg;base64," + img_b64,
            "width": img.shape[1],
            "height": img.shape[0],
            "scale": scale,
            "persons": persons,
        })
    except Exception as e:
        logger.error(f"预览失败: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/videos/{video_id}/info")
async def video_info(video_id: str):
    """获取视频元数据"""
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})
    return _video_registry[video_id].get("info", {})


@app.get("/api/videos/{video_id}/download")
async def download_video(video_id: str):
    """下载原始视频文件"""
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})
    v = _video_registry[video_id]
    path = Path(v["path"])
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "file not found"})
    return FileResponse(str(path), filename=v["filename"])


@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: str):
    """删除视频及其关联的任务记录和磁盘文件。"""
    # 先从内存获取路径
    v = _video_registry.get(video_id)
    storage_path = v["path"] if v else None
    filename = v["filename"] if v else None

    # 删除数据库记录（含关联任务）
    deleted = db.delete_video(video_id)
    if not deleted and not v:
        return JSONResponse(status_code=404, content={"detail": "视频不存在"})

    # 清理内存
    _video_registry.pop(video_id, None)

    # 清理该视频相关的内存任务结果
    with _task_results_lock:
        task_ids_to_clean = [
            tid for tid, res in _task_results.items()
            if res.get("video_id") == video_id
        ]
        for tid in task_ids_to_clean:
            _task_results.pop(tid, None)
            _clip_registry.pop(tid, None)
            _cancel_events.pop(tid, None)

    # 删除磁盘文件
    if storage_path:
        try:
            Path(storage_path).unlink(missing_ok=True)
        except Exception:
            pass

    # 删除关联的 clip 文件
    clip_dir = OUTPUT_DIR
    if clip_dir.exists():
        for clip_file in clip_dir.glob(f"*{video_id}*"):
            try:
                clip_file.unlink(missing_ok=True)
            except Exception:
                pass

    return {"detail": f"视频 {filename or video_id} 已删除"}


@app.get("/api/videos/{video_id}/table-calibration")
async def get_table_calibration(video_id: str):
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})
    try:
        calibration = _table_calibration_store().load(video_id)
    except ValueError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return {
        "video_id": video_id,
        "calibration": calibration.to_dict() if calibration else None,
    }


@app.post("/api/videos/{video_id}/table-calibration")
async def save_table_calibration(video_id: str, payload=Body(...)):
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})
    try:
        calibration = _parse_table_calibration(video_id, payload)
        _table_calibration_store().save(video_id, calibration)
    except (TypeError, ValueError, KeyError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return {"video_id": video_id, "calibration": calibration.to_dict()}


@app.delete("/api/videos/{video_id}/table-calibration")
async def delete_table_calibration(video_id: str):
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})
    _table_calibration_store().delete(video_id)
    return {"video_id": video_id, "deleted": True}


@app.get("/api/videos/{video_id}/frame")
async def video_frame(video_id: str, frame_index: int = Query(..., ge=0)):
    """Return one source frame as JPEG for manual ball annotation."""
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})
    info = _video_registry[video_id].get("info", {})
    total_frames = int(info.get("total_frames", 0))
    if total_frames and frame_index >= total_frames:
        return JSONResponse(status_code=400, content={"error": "frame_index out of range"})

    import cv2

    cap = cv2.VideoCapture(_video_registry[video_id]["path"])
    if not cap.isOpened():
        return JSONResponse(status_code=500, content={"error": "video cannot be opened"})
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
    finally:
        cap.release()
    if not success:
        return JSONResponse(status_code=400, content={"error": "frame cannot be decoded"})
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return JSONResponse(status_code=500, content={"error": "frame encoding failed"})
    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Frame-Index": str(frame_index),
            "X-Video-Width": str(frame.shape[1]),
            "X-Video-Height": str(frame.shape[0]),
        },
    )


@app.get("/api/videos/{video_id}/tracknet/annotations")
async def get_tracknet_annotations(video_id: str):
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})
    try:
        annotations = _annotation_store().load(video_id)
    except ValueError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return {
        "video_id": video_id,
        "annotations": [annotation.to_dict() for annotation in annotations],
    }


@app.post("/api/videos/{video_id}/tracknet/annotations")
async def save_tracknet_annotations(video_id: str, payload=Body(...)):
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})
    try:
        annotations = _parse_tracknet_annotations(video_id, payload)
        _annotation_store().save(video_id, annotations)
    except (TypeError, ValueError, KeyError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return {
        "video_id": video_id,
        "annotations": [annotation.to_dict() for annotation in annotations],
        "count": len(annotations),
    }


@app.delete("/api/videos/{video_id}/tracknet/annotations")
async def delete_tracknet_annotations(video_id: str):
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})
    _annotation_store().delete(video_id)
    return {"video_id": video_id, "deleted": True}


@app.get("/api/tracknet/models")
async def list_tracknet_models():
    registry = _tracknet_registry()
    models = []
    for record in registry.list_models():
        path = Path(record.checkpoint_path)
        models.append({
            **record.to_dict(),
            "available": path.is_file() and record.status == "available",
        })
    return {"models": models, "default_model_id": "base"}


@app.post("/api/videos/{video_id}/tracknet/train")
async def start_tracknet_training(video_id: str, payload=Body(default={} )):
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})
    payload = payload if isinstance(payload, dict) else {}
    try:
        annotations = _parse_tracknet_annotations(
            video_id,
            {"annotations": [item.to_dict() for item in _annotation_store().load(video_id)]},
        )
        config = _tracknet_training_config()
        config.update({key: payload[key] for key in (
            "epochs", "batch_size", "learning_rate", "min_positive_annotations"
        ) if key in payload})
        tuner = TrackNetFineTuner(config)
        tuner._validate_annotations(annotations)
        base_model_id = str(payload.get("model_id", "base"))
        base_path = _resolve_tracknet_model(base_model_id)
        registry = _tracknet_registry()
        run = registry.create_temp_run({
            "video_id": video_id,
            "base_model_id": base_model_id,
            "annotation_count": len(annotations),
        })
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    except LookupError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    except (TypeError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    job_id = f"train_{uuid.uuid4().hex[:8]}"
    # 持久化到数据库，使其在工作台历史列表中可见
    db.create_task(job_id, mode="tracknet_train", video_id=video_id)
    db.update_task(job_id, status="processing", progress=0, message="训练任务已排队")
    job = {
        "job_id": job_id,
        "video_id": video_id,
        "run_id": run.run_id,
        "status": "queued",
        "progress": 0,
        "message": "训练任务已排队",
        "base_model_id": base_model_id,
        "annotation_count": len(annotations),
        "cancel_event": threading.Event(),
    }
    with _training_lock:
        _training_jobs[job_id] = job
    thread = threading.Thread(
        target=_run_tracknet_training_task,
        args=(job_id, run.output_dir, base_path, video_id, annotations, config),
        name=f"tracknet-{job_id}",
        daemon=True,
    )
    thread.start()
    return {
        "job_id": job_id,
        "run_id": run.run_id,
        "status": "queued",
        "annotation_count": len(annotations),
    }


def _update_training_job(job_id: str, **values) -> None:
    with _training_lock:
        if job_id in _training_jobs:
            _training_jobs[job_id].update(values)


def _run_tracknet_training_task(
    job_id: str,
    output_dir: Path,
    base_path: Path,
    video_id: str,
    annotations: list[BallAnnotation],
    config: dict,
):
    try:
        _update_training_job(job_id, status="running", progress=1, message="正在准备训练设备")
        db.update_task(job_id, status="processing", progress=1, message="正在准备训练设备")
        device_info = DeviceManager.detect(get_config().device_mode)
        epoch_history: list[dict] = []

        def on_progress(payload: dict):
            epoch = int(payload.get("epoch", 0))
            epochs = max(1, int(payload.get("epochs", config.get("epochs", 5))))
            progress = min(99, int(epoch / epochs * 100))
            msg = f"训练第 {epoch}/{epochs} 轮"
            # 累积每轮指标
            if epoch > 0 and payload.get("status") == "running":
                epoch_entry = {
                    "epoch": epoch,
                    "train_loss": payload.get("train_loss"),
                    "validation_loss": payload.get("validation_loss"),
                    "positive_detection_rate": payload.get("positive_detection_rate"),
                    "mean_pixel_error": payload.get("mean_pixel_error"),
                    "learning_rate": float(config.get("learning_rate", 1e-4)),
                }
                # 去重：同 epoch 覆盖
                epoch_history[:] = [e for e in epoch_history if e["epoch"] != epoch]
                epoch_history.append(epoch_entry)
            _update_training_job(
                job_id,
                status=payload.get("status", "running"),
                progress=progress,
                message=msg,
                metrics=payload,
                epoch_history=list(epoch_history),
            )
            db.update_task(
                job_id,
                status="processing",
                progress=progress,
                message=msg,
                result={"epoch_history": list(epoch_history), "metrics": payload},
            )

        tuner = TrackNetFineTuner(config)
        with _analysis_lock:
            result = tuner.train(
                base_model_path=base_path,
                video_path=_video_registry[video_id]["path"],
                annotations=annotations,
                output_dir=output_dir,
                device_info=device_info,
                progress_callback=on_progress,
                cancel_event=_training_jobs[job_id]["cancel_event"],
            )
        if result.get("status") == "cancelled":
            _tracknet_registry().discard_run(_training_jobs[job_id]["run_id"])
            _update_training_job(job_id, status="cancelled", progress=0, message="训练已取消", result=result)
            db.update_task(job_id, status="cancelled", progress=0, message="训练已取消")
            return
        result["epoch_history"] = list(epoch_history)
        _update_training_job(
            job_id,
            status="completed",
            progress=100,
            message="训练完成，等待保存或放弃",
            result=result,
            checkpoint_path=result.get("checkpoint_path"),
            epoch_history=list(epoch_history),
        )
        db.update_task(
            job_id,
            status="completed",
            progress=100,
            message="训练完成",
            result={"epoch_history": list(epoch_history), "metrics": result.get("metrics", {}), "checkpoint_path": result.get("checkpoint_path")},
        )
    except Exception as exc:
        logger.exception(f"TrackNet 训练任务 {job_id} 失败")
        try:
            _tracknet_registry().discard_run(_training_jobs[job_id]["run_id"])
        except Exception:
            logger.exception(f"TrackNet 临时任务清理失败: {job_id}")
        _update_training_job(job_id, status="failed", progress=0, message=str(exc), error=str(exc))
        db.update_task(job_id, status="failed", progress=0, message=str(exc))


@app.get("/api/tracknet/train/{job_id}")
async def get_tracknet_training(job_id: str):
    with _training_lock:
        job = _training_jobs.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "training job not found"})
        data = {key: value for key, value in job.items() if key != "cancel_event"}
    return data


@app.post("/api/tracknet/train/{job_id}/cancel")
async def cancel_tracknet_training(job_id: str):
    with _training_lock:
        job = _training_jobs.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "training job not found"})
        if job["status"] in {"completed", "failed", "cancelled", "saved", "discarded"}:
            return JSONResponse(status_code=409, content={"error": "training job is no longer running"})
        job["cancel_event"].set()
        job["message"] = "正在取消训练"
    return {"job_id": job_id, "status": "cancelling"}


@app.post("/api/tracknet/train/{job_id}/save")
async def save_tracknet_training(job_id: str, payload=Body(default={} )):
    payload = payload if isinstance(payload, dict) else {}
    with _training_lock:
        job = _training_jobs.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "training job not found"})
        if job.get("status") != "completed":
            return JSONResponse(status_code=409, content={"error": "training is not completed"})
        checkpoint_path = job.get("checkpoint_path")
        run_id = job["run_id"]
    try:
        metadata = dict(job.get("result", {}))
        if payload.get("display_name"):
            metadata["display_name"] = str(payload["display_name"])
        record = _tracknet_registry().save_run(run_id, checkpoint_path, metadata)
    except (OSError, ValueError, FileNotFoundError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    _update_training_job(job_id, status="saved", progress=100, model_id=record.model_id, message="模型已保存")
    db.update_task(job_id, status="completed", progress=100, message="模型已保存")
    return {"job_id": job_id, "model": record.to_dict()}


@app.delete("/api/tracknet/train/{job_id}")
async def discard_tracknet_training(job_id: str):
    with _training_lock:
        job = _training_jobs.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "training job not found"})
        if job.get("status") == "running":
            job["cancel_event"].set()
        run_id = job["run_id"]
    try:
        _tracknet_registry().discard_run(run_id)
    except OSError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    _update_training_job(job_id, status="discarded", progress=0, message="训练结果已放弃")
    db.update_task(job_id, status="cancelled", progress=0, message="训练结果已放弃")
    return {"job_id": job_id, "status": "discarded"}


@app.post("/api/tracknet/models/{model_id}/disable")
async def disable_tracknet_model(model_id: str):
    try:
        _tracknet_registry().disable(model_id)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except KeyError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    return {"model_id": model_id, "status": "disabled"}


@app.get("/api/stream/{video_id}")
async def stream_tracking(
    video_id: str,
    max_frames: int = Query(-1, description="最大帧数, -1=全部"),
    width: int = Query(960, description="输出宽度"),
    mode: str = Query("rally", description="rally 或 action"),
    model_id: str = Query("base", description="手动选择的 TrackNet 模型"),
    no_crossing_timeout_seconds: float = Query(4.0, ge=0.5, le=10.0),
):
    """MJPEG 实时追踪流 (无元数据, 仅视频)"""
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})
    if mode not in {"rally", "action"}:
        return JSONResponse(status_code=400, content={"error": "invalid mode"})

    model_path = None
    table_calibration = None
    if mode == "rally":
        try:
            table_calibration = _required_table_calibration(video_id)
        except LookupError as exc:
            return JSONResponse(status_code=409, content={"error": str(exc)})
        try:
            model_path = str(_resolve_tracknet_model(model_id))
        except RuntimeError as exc:
            return JSONResponse(status_code=409, content={"error": str(exc)})
        except LookupError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})

    video_path = _video_registry[video_id]["path"]

    def generate():
        boundary = "ppframe"
        for frame_bytes in generate_mjpeg_stream(
            video_path,
            max_frames=max_frames,
            target_width=width,
            mode=mode,
            tracknet_model_path=model_path,
            table_calibration=table_calibration,
            no_crossing_timeout_seconds=no_crossing_timeout_seconds,
        ):
            yield (
                f"--{boundary}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(frame_bytes)}\r\n\r\n"
            ).encode()
            yield frame_bytes
            yield b"\r\n"
        yield f"--{boundary}--\r\n".encode()

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=ppframe",
    )


@app.websocket("/api/ws/{video_id}")
async def tracking_ws(websocket: WebSocket, video_id: str):
    """
    WebSocket 实时追踪: 视频帧 + 元数据

    消息协议:
    - text: JSON 元数据 {"type": "meta", "frame": N, "ball_speed": S, ...}
    - text: JSON 状态 {"type": "status", "status": "loading"|"completed"|"error"}
    - binary: JPEG 帧字节 (紧跟在 meta 消息后)

    Query 参数:
    - person_filter: 选中的bbox列表，格式 "x1,y1,x2,y2;x1,y1,x2,y2"
    - playback_speed: 初始播放速度，按视频原始FPS发送，默认1.0
    """
    # WebSocket 认证：从 query param 验证 token
    from urllib.parse import parse_qs
    query = parse_qs(websocket.url.query)
    ws_token = query.get("token", [None])[0]
    if not ws_token or not verify_token(ws_token):
        await websocket.accept()
        await websocket.send_json({"type": "status", "status": "error", "error": "未认证或认证已过期"})
        await websocket.close(code=1008)
        return

    await websocket.accept()

    # 从query string读取模式和人物过滤
    mode = query.get("mode", ["rally"])[0]
    playback_speed = _normalize_playback_speed(query.get("playback_speed", [1.0])[0])
    crossing_timeout = _normalize_crossing_timeout(
        query.get("no_crossing_timeout_seconds", [4.0])[0]
    )
    if mode not in {"rally", "action"}:
        await websocket.send_json({"type": "status", "status": "error", "error": "invalid mode"})
        await websocket.close()
        return
    person_filter_raw = query.get("person_filter", [None])[0]
    model_id = query.get("model_id", ["base"])[0]

    person_boxes = []
    if person_filter_raw:
        for part in person_filter_raw.split(";"):
            coords = part.strip().split(",")
            if len(coords) == 4:
                person_boxes.append([float(c) for c in coords])

    if video_id not in _video_registry:
        await websocket.send_json({"type": "status", "status": "error", "error": "video not found"})
        await websocket.close()
        return

    video_path = _video_registry[video_id]["path"]
    video_info = _video_registry[video_id].get("info", {})

    model_path = None
    table_calibration = None
    if mode == "rally":
        try:
            table_calibration = _required_table_calibration(video_id)
        except LookupError as exc:
            await websocket.send_json({"type": "status", "status": "error", "error": str(exc)})
            await websocket.close()
            return
        try:
            model_path = str(_resolve_tracknet_model(model_id))
        except RuntimeError as exc:
            await websocket.send_json({"type": "status", "status": "error", "error": str(exc)})
            await websocket.close()
            return
        except LookupError as exc:
            await websocket.send_json({"type": "status", "status": "error", "error": str(exc)})
            await websocket.close()
            return

    await websocket.send_json({"type": "status", "status": "loading", "percent": 0, "message": "正在初始化..."})

    visualizer = TrackingVisualizer(
        mode=mode,
        tracknet_model_path=model_path,
        table_calibration=table_calibration,
        no_crossing_timeout_seconds=crossing_timeout,
    )
    if person_boxes:
        visualizer.set_person_filter(person_boxes)
    try:
        loop = asyncio.get_event_loop()

        def on_progress(pct, msg):
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({"type": "status", "status": "loading", "percent": pct, "message": msg}),
                loop,
            )

        await loop.run_in_executor(None, lambda: visualizer.load_models(on_progress))
        await websocket.send_json({
            "type": "status",
            "status": "preparing",
            "message": "正在准备视频背景...",
        })
        await loop.run_in_executor(None, lambda: visualizer.prepare_background(video_path))
        await websocket.send_json({
            "type": "status",
            "status": "tracking",
            "warning": visualizer.tracking_warning,
            "playback_speed": playback_speed,
            "no_crossing_timeout_seconds": crossing_timeout,
        })

        import av
        import cv2
        import numpy as np

        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        total_frames = video_info.get("total_frames", stream.frames or 0)
        target_width = 960
        next_frame_at = time.monotonic()

        for frame in container.decode(video=0):
            try:
                # 检查是否收到停止指令 (非阻塞)
                try:
                    msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                    if msg == "stop":
                        break
                    if msg.startswith("speed:"):
                        playback_speed = _normalize_playback_speed(msg.split(":", 1)[1])
                        next_frame_at = time.monotonic()
                        await websocket.send_json({
                            "type": "status",
                            "status": "speed",
                            "playback_speed": playback_speed,
                        })
                except asyncio.TimeoutError:
                    pass

                # Keep source resolution during inference so persisted calibration
                # points remain in the same coordinate system as the ball detector.
                img = frame.to_ndarray(format="bgr24")
                annotated, meta = visualizer.process_frame(img, fps)
                h, w = annotated.shape[:2]
                if w > target_width:
                    scale = target_width / w
                    annotated = cv2.resize(annotated, (target_width, int(h * scale)))

                # 先发 JSON 元数据
                await websocket.send_text(json.dumps({
                    "type": "meta",
                    "frame": meta["frame"],
                    "timestamp": meta["timestamp"],
                    "total_frames": total_frames,
                    "ball_pos": meta["ball_pos"],
                    "ball_speed": meta["ball_speed"],
                    "ball_speed_kmh": meta.get("ball_speed_kmh"),
                    "calibrated": meta.get("calibrated", False),
                    "persons": meta["persons"],
                    "rally_count": meta["rally_count"],
                    "board_count": meta["board_count"],
                    "rally_state": meta["rally_state"],
                    "arm_angles": meta["arm_angles"],
                    "mode": meta["mode"],
                    "ball_tracking": meta["ball_tracking"],
                    "warning": meta["warning"],
                    "playback_speed": playback_speed,
                }))

                # 再发 JPEG 二进制帧
                _, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                await websocket.send_bytes(jpeg_buf.tobytes())

                # 以视频 FPS 为基准节流。处理速度达不到目标时直接进入下一帧，
                # 不累计睡眠债务，避免后续突然连发造成“播放过快”。
                next_frame_at += 1.0 / max(fps * playback_speed, 0.01)
                delay = next_frame_at - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_frame_at = time.monotonic()

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WS帧处理错误: {e}")
                break

        await websocket.send_json({"type": "status", "status": "completed"})
        container.close()
    except WebSocketDisconnect:
        logger.info(f"WS断开: {video_id}")
    except Exception as e:
        logger.error(f"WS错误: {e}")
        try:
            await websocket.send_json({"type": "status", "status": "error", "error": str(e)})
        except:
            pass
    finally:
        visualizer.unload_models()
        try:
            await websocket.close()
        except:
            pass


@app.post("/api/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    min_boards: int = 4,
    max_frames: int = -1,
    model_id: str = "base",
    no_crossing_timeout_seconds: float = Query(4.0, ge=0.5, le=10.0),
    max_retries: int = Query(5, ge=0, le=20),
):
    """上传视频并执行纯球回合分析 (异步)。"""
    task_id = str(uuid.uuid4())[:8]
    safe_name = file.filename.replace(" ", "_")
    video_path = VIDEO_LIBRARY_DIR / safe_name

    # 重名时加task_id前缀
    if video_path.exists():
        video_path = VIDEO_LIBRARY_DIR / f"{task_id}_{safe_name}"

    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

    video_info = _probe_video(str(video_path))
    video_id = db.stable_video_id(video_path.name, len(content))
    record = db.upsert_video(video_id, file.filename, str(video_path), len(content), video_info)
    _video_registry[video_id] = record
    logger.info(f"任务 {task_id}: 视频已上传 {video_path} (video_id={video_id})")

    try:
        model_path = str(_resolve_tracknet_model(model_id))
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    except LookupError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    _update_task_result(
        task_id,
        status="processing",
        mode="rally",
        progress=0,
        message="分析任务已排队",
    )
    background_tasks.add_task(
        _run_analysis_task,
        str(video_path),
        task_id,
        min_boards,
        max_frames,
        model_path,
        no_crossing_timeout_seconds,
        max_retries,
    )
    return {"task_id": task_id, "status": "processing", "message": "分析已开始"}


@app.post("/api/analyze/{video_id}")
async def analyze_existing(
    background_tasks: BackgroundTasks,
    video_id: str,
    min_boards: int = 4,
    max_frames: int = -1,
    model_id: str = "base",
    no_crossing_timeout_seconds: float = Query(4.0, ge=0.5, le=10.0),
    max_retries: int = Query(5, ge=0, le=20),
):
    """分析已上传的视频，只生成回合结果和剪辑。"""
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})

    task_id = video_id
    video_path = _video_registry[video_id]["path"]
    try:
        _required_table_calibration(video_id)
    except LookupError as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    try:
        model_path = str(_resolve_tracknet_model(model_id))
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    except LookupError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    db.create_task(task_id, mode="rally", video_id=video_id)
    _update_task_result(
        task_id,
        status="processing",
        mode="rally",
        progress=0,
        message="分析任务已排队",
    )
    background_tasks.add_task(
        _run_analysis_task,
        video_path,
        task_id,
        min_boards,
        max_frames,
        model_path,
        no_crossing_timeout_seconds,
        max_retries,
    )
    return {"task_id": task_id, "status": "processing"}


def _run_analysis_task(
    video_path: str,
    task_id: str,
    min_boards: int,
    max_frames: int,
    model_path: str | None = None,
    no_crossing_timeout_seconds: float = 4.0,
    max_retries: int = 5,
):
    """后台执行分析任务"""
    cancel_event = _cancel_events.setdefault(task_id, threading.Event())
    try:
        # T4 只有16GB显存, 串行分析可避免多个后台任务同时加载三套模型。
        _init_steps(task_id, _RALLY_STEPS)
        _update_step(task_id, "load_model", "processing")
        _update_task_result(
            task_id,
            status="processing",
            mode="rally",
            progress=2,
            message="正在加载 TrackNet 模型...",
        )
        with _analysis_lock:
            if cancel_event.is_set():
                _update_task_result(task_id, status="cancelled", progress=0, message="任务已取消")
                db.update_task(task_id, status="cancelled", progress=0, message="任务已取消")
                return
            table_calibration = _table_calibration_store().load(task_id)
            analyzer = get_analyzer(
                tracknet_model_path=model_path,
                table_calibration=table_calibration,
                no_crossing_timeout_seconds=no_crossing_timeout_seconds,
            )
            analyzer.rally_detector.min_boards = min_boards
            _update_step(task_id, "load_model", "completed", 100)

            if cancel_event.is_set():
                _update_task_result(task_id, status="cancelled", progress=0, message="任务已取消")
                db.update_task(task_id, status="cancelled", progress=0, message="任务已取消")
                return

            _update_step(task_id, "sample_bg", "processing")
            _update_task_result(
                task_id,
                status="processing",
                mode="rally",
                progress=5,
                message="正在采样视频背景...",
            )

            def on_progress(ratio: float) -> None:
                ratio = min(max(float(ratio), 0.0), 1.0)
                pct = int(ratio * 100)
                _update_step(task_id, "analyze_frames", "processing", pct)
                _update_step(task_id, "sample_bg", "completed", 100)
                _update_task_result(
                    task_id,
                    status="processing",
                    mode="rally",
                    progress=10 + int(ratio * 75),
                    message=f"正在分析视频帧 · {pct}%",
                )

            # 带断点续传的重试循环
            checkpoint_frame = 0
            for attempt in range(max_retries + 1):
                if cancel_event.is_set():
                    break
                try:
                    segments = analyzer.analyze(
                        video_path,
                        max_frames=max_frames,
                        progress_callback=on_progress,
                        cancel_event=cancel_event,
                        start_frame=checkpoint_frame,
                        resume=attempt > 0,
                    )
                    break
                except PipelineStuckError as e:
                    if attempt < max_retries:
                        checkpoint_frame = getattr(analyzer, '_checkpoint_frame_idx', checkpoint_frame)
                        logger.warning(
                            f"任务 {task_id}: 管线卡死 (attempt {attempt+1}/{max_retries}), "
                            f"从帧 {checkpoint_frame} 重试: {e}"
                        )
                        _update_task_result(
                            task_id,
                            status="processing",
                            mode="rally",
                            progress=10 + int(checkpoint_frame / max(1, max_frames or 999999) * 75),
                            message=f"管线卡死，第 {attempt+1}/{max_retries} 次重试（从帧 {checkpoint_frame} 继续）...",
                        )
                        continue
                    raise
            else:
                raise RuntimeError(f"任务 {task_id}: 达到最大重试次数 {max_retries}，仍然失败")
            _update_step(task_id, "analyze_frames", "completed", 100)
            _update_step(task_id, "detect_rallies", "completed", 100)
            _update_step(task_id, "export_clips", "processing")
            _update_task_result(
                task_id,
                status="processing",
                mode="rally",
                progress=88,
                message="正在生成得分段剪辑...",
            )
            outputs = analyzer.export_clips(video_path, segments)
            _clip_registry[task_id] = [Path(path).name for path in outputs]
            _update_step(task_id, "export_clips", "completed", 100)
            _update_step(task_id, "save_results", "processing")

            # 0 回合时不创建编辑项目
            if not segments:
                project_id = None
                _update_step(task_id, "save_results", "completed", 100)
                _update_task_result(
                    task_id,
                    status="completed",
                    mode="rally",
                    progress=100,
                    message="分析完成，未检测到符合阈值的得分段",
                    steps=[
                        {"key": k, "name": n, "status": "completed", "progress": 100}
                        for k, n in _RALLY_STEPS
                    ],
                    total_rallies=0,
                    rallies=[],
                    clips=[],
                )
                db.update_task(
                    task_id,
                    status="completed",
                    progress=100,
                    message="分析完成，未检测到符合阈值的得分段",
                    project_id=None,
                    clips=[],
                    result={"total_rallies": 0, "rallies": []},
                )
                logger.info(f"任务 {task_id}: 完成, 0 个回合（未检测到符合阈值的得分段）")
                return

            project = create_edit_project_from_segments(
                task_id, video_path, segments, outputs
            )
            project_id = project["project_id"]

        _update_step(task_id, "save_results", "completed", 100)
        _update_task_result(
            task_id,
            status="completed",
            mode="rally",
            progress=100,
            message="分析完成",
            project_id=project_id,
            steps=[
                {"key": k, "name": n, "status": "completed", "progress": 100}
                for k, n in _RALLY_STEPS
            ],
            total_rallies=len(segments),
            rallies=_serialize_rallies(segments),
            clips=[Path(p).name for p in outputs],
        )
        db.update_task(
            task_id,
            status="completed",
            progress=100,
            message="分析完成",
            project_id=project_id,
            clips=[Path(p).name for p in outputs],
            result={
                "total_rallies": len(segments),
                "rallies": _serialize_rallies(segments),
            },
        )
        logger.info(f"任务 {task_id}: 完成, {len(segments)} 个回合, {len(outputs)} 个片段")
    except Exception as e:
        if cancel_event.is_set():
            _update_task_result(task_id, status="cancelled", progress=0, message="任务已取消")
            db.update_task(task_id, status="cancelled", progress=0, message="任务已取消")
            logger.info(f"任务 {task_id}: 已取消")
        else:
            logger.error(f"任务 {task_id}: 失败 - {e}")
            _update_task_result(
                task_id,
                status="failed",
                mode="rally",
                progress=0,
                message="分析失败",
                error=str(e),
            )
            db.update_task(task_id, status="failed", progress=0, message="分析失败")
    finally:
        _cancel_events.pop(task_id, None)


@app.post("/api/edit-projects/from-analysis/{task_id}")
async def create_edit_project_endpoint(task_id: str):
    """Create an editor project for a completed rally result, including legacy results."""
    try:
        project = create_edit_project_from_analysis(task_id)
    except ProjectNotFoundError:
        return JSONResponse(status_code=404, content={"error": "edit project not found"})
    except (LookupError, ProjectValidationError, ValueError) as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    return project


@app.get("/api/edit-projects")
async def list_edit_projects(video_id: str | None = None):
    try:
        projects = _edit_project_store().list(video_id=video_id)
    except ProjectValidationError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return {"projects": projects}


@app.get("/api/edit-projects/{project_id}")
async def get_edit_project(project_id: str):
    try:
        return _edit_project_store().get(project_id)
    except ProjectNotFoundError:
        return JSONResponse(status_code=404, content={"error": "edit project not found"})
    except ProjectValidationError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.patch("/api/edit-projects/{project_id}")
async def patch_edit_project(project_id: str, payload=Body(...)):
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        return JSONResponse(
            status_code=400,
            content={"error": "segments must be an array"},
        )
    try:
        return _edit_project_store().update_segments(project_id, payload["segments"])
    except ProjectNotFoundError:
        return JSONResponse(status_code=404, content={"error": "edit project not found"})
    except ProjectValidationError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


def _composition_renderer() -> CompositionRenderer:
    config = get_config()
    return CompositionRenderer(
        output_dir=COMPOSITION_OUTPUT_DIR,
        encoder=config.get("composition", "encoder", default="auto"),
        preset=config.get("composition", "preset", default="fast"),
        crf=config.get("composition", "crf", default=21),
    )


_RENDER_MAX_RETRIES = 5


def _run_edit_render_task(project_id: str, task_id: str) -> None:
    """Render one project with auto-retry (up to 5 attempts)."""
    store = _edit_project_store()

    def _update_db(status: str, progress: int, message: str, **kwargs):
        """Update both edit-project render state and DB task."""
        try:
            store.update_render(
                project_id,
                status=status,
                progress=progress,
                message=message,
                task_id=task_id,
                **kwargs,
            )
        except Exception:
            pass
        _update_task_result(
            task_id,
            mode="render",
            status=status,
            progress=progress,
            message=message,
            **{k: v for k, v in kwargs.items() if k in ("error",)},
        )
        try:
            db.update_task(task_id, status=status, progress=progress, message=message)
        except Exception:
            pass

    for attempt in range(1, _RENDER_MAX_RETRIES + 1):
        with _render_lock:
            try:
                project = store.get(project_id)
                if project["render"]["status"] not in ("rendering", "retrying"):
                    return
                source_path = _resolve_project_source(project)
                selected = sorted(
                    [segment for segment in project["segments"] if segment["selected"]],
                    key=lambda segment: segment["order"],
                )
                if not selected:
                    raise CompositionError("至少需要选择一个精彩分段")

                output_path = COMPOSITION_OUTPUT_DIR / f"{project_id}.mp4"

                retry_msg = f"（第 {attempt}/{_RENDER_MAX_RETRIES} 次）" if attempt > 1 else ""
                _update_db("rendering", 0, f"正在渲染{retry_msg}")

                def on_progress(ratio: float, message: str) -> None:
                    _update_db(
                        "rendering",
                        round(min(max(float(ratio), 0.0), 1.0) * 100, 1),
                        message or "正在渲染",
                    )

                result = _composition_renderer().render(
                    source_path,
                    selected,
                    output_path,
                    progress_callback=on_progress,
                )
                output_size = output_path.stat().st_size if output_path.is_file() else None
                _update_db(
                    "completed", 100, "渲染完成",
                    output_filename=Path(result.get("output_filename", output_path.name)).name,
                    output_size_bytes=output_size,
                    duration=float(result.get("duration", 0.0)),
                    error=None,
                )
                # 更新 DB result
                _update_task_result(
                    task_id, mode="render", status="completed", progress=100,
                    message="渲染完成",
                )
                logger.info("成片项目 {} 渲染完成", project_id)
                return
            except Exception as exc:
                logger.error("成片项目 {} 渲染失败 (attempt {}/{}): {}", project_id, attempt, _RENDER_MAX_RETRIES, exc)
                if attempt < _RENDER_MAX_RETRIES:
                    _update_db("retrying", 0, f"渲染失败，准备重试（第 {attempt+1}/{_RENDER_MAX_RETRIES} 次）", error=str(exc))
                    _update_task_result(
                        task_id, mode="render", status="processing", progress=0,
                        message=f"渲染失败，准备重试（{attempt+1}/{_RENDER_MAX_RETRIES}）",
                        error=str(exc),
                    )
                else:
                    _update_db("failed", 0, f"渲染失败（已重试 {_RENDER_MAX_RETRIES} 次）", error=str(exc))
                    _update_task_result(
                        task_id, mode="render", status="failed", progress=0,
                        message=f"渲染失败（已重试 {_RENDER_MAX_RETRIES} 次）",
                        error=str(exc),
                    )
                    return


@app.post("/api/edit-projects/{project_id}/render")
async def render_edit_project(project_id: str, background_tasks: BackgroundTasks):
    store = _edit_project_store()
    with _render_lock:
        try:
            project = store.get(project_id)
        except ProjectNotFoundError:
            return JSONResponse(status_code=404, content={"error": "edit project not found"})
        except ProjectValidationError as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

        render = project["render"]
        if render["status"] == "rendering":
            return JSONResponse(status_code=409, content=_render_status_payload(project))
        selected = [segment for segment in project["segments"] if segment["selected"]]
        if not selected:
            return JSONResponse(status_code=409, content={"error": "至少需要选择一个精彩分段"})
        try:
            _resolve_project_source(project)
        except FileNotFoundError as exc:
            return JSONResponse(status_code=409, content={"error": str(exc)})

        task_id = f"render_{uuid.uuid4().hex[:10]}"
        project = store.update_render(
            project_id,
            status="rendering",
            progress=0,
            message="等待渲染",
            task_id=task_id,
            output_filename=None,
            output_size_bytes=None,
            duration=None,
            error=None,
        )

        # 创建 DB 任务记录，让渲染在任务列表中可见
        try:
            video_id = project.get("video_id", "")
            db.create_task(
                task_id=task_id,
                video_id=video_id,
                mode="render",
            )
            db.update_task(
                task_id,
                status="processing",
                progress=0,
                message="等待渲染",
                project_id=project_id,
            )
        except Exception as exc:
            logger.warning("无法为渲染任务创建 DB 记录: {}", exc)

        _update_task_result(
            task_id, mode="render", status="processing", progress=0, message="等待渲染",
        )

        background_tasks.add_task(_run_edit_render_task, project_id, task_id)
    return _render_status_payload(project)


@app.get("/api/edit-projects/{project_id}/render")
async def get_edit_render(project_id: str):
    try:
        project = _edit_project_store().get(project_id)
    except ProjectNotFoundError:
        return JSONResponse(status_code=404, content={"error": "edit project not found"})
    except ProjectValidationError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return _render_status_payload(project)


@app.get("/api/edit-projects/{project_id}/export")
async def export_edit_project(project_id: str):
    try:
        project = _edit_project_store().get(project_id)
    except ProjectNotFoundError:
        return JSONResponse(status_code=404, content={"error": "edit project not found"})
    except ProjectValidationError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})

    render = project["render"]
    if render["status"] != "completed":
        return JSONResponse(status_code=409, content={"error": "成片尚未渲染完成"})
    output_filename = render.get("output_filename")
    if not output_filename:
        return JSONResponse(status_code=404, content={"error": "成片文件不存在"})
    output_root = COMPOSITION_OUTPUT_DIR.resolve()
    output_path = (COMPOSITION_OUTPUT_DIR / output_filename).resolve()
    if output_root not in output_path.parents or not output_path.is_file():
        return JSONResponse(status_code=404, content={"error": "成片文件不存在"})
    return FileResponse(str(output_path), media_type="video/mp4", filename=output_filename)


@app.get("/editor/{project_id}")
async def editor_page(project_id: str):
    """Serve the independent editor only for an existing project."""
    try:
        _edit_project_store().get(project_id)
    except ProjectNotFoundError:
        return JSONResponse(status_code=404, content={"error": "edit project not found"})
    except ProjectValidationError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    editor_path = STATIC_DIR / "editor.html"
    if not editor_path.is_file():
        return JSONResponse(status_code=404, content={"error": "editor frontend not found"})
    return FileResponse(str(editor_path))


def _resolve_clip_path(filename: str) -> Path:
    """解析回合片段，并阻止通过文件名越界访问其他路径。"""
    output_root = OUTPUT_DIR.resolve()
    clip_path = (OUTPUT_DIR / filename).resolve()
    if output_root not in clip_path.parents or not clip_path.is_file():
        raise FileNotFoundError(f"clip not found: {filename}")
    return clip_path


@app.post("/api/action/analyze/{video_id}")
async def analyze_action(
    background_tasks: BackgroundTasks,
    video_id: str,
    clip_filename: str | None = Query(None, description="单个回合片段文件名"),
    all_clips: bool | None = Query(None, description="是否分析该视频生成的全部回合片段"),
    source: str | None = Query(None, description="clips 或 original；默认优先得分段"),
    include_frames: bool = Query(False, description="调试：是否返回精简逐帧数据"),
    max_frames: int = Query(-1, description="每个输入最多处理的帧数"),
):
    """独立动作分析入口。默认分析已生成的得分段，而不是整段原片。"""
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})
    if clip_filename and all_clips:
        return JSONResponse(status_code=400, content={"error": "clip_filename and all_clips are mutually exclusive"})
    source_mode = (source or "").strip().lower()
    if source_mode and source_mode not in {"clips", "original"}:
        return JSONResponse(status_code=400, content={"error": "source 只能是 clips 或 original"})

    rally_payload = _rally_payload_for_video(video_id)
    clip_names = rally_payload["clips"]
    rallies = rally_payload["rallies"]
    want_original = source_mode == "original" or all_clips is False
    want_clips = (
        bool(clip_filename)
        or all_clips is True
        or source_mode == "clips"
        or (not want_original and not clip_filename)
    )

    source_paths: list[str] = []
    hit_groups: list[list[dict]] = []
    buffer_before, _buffer_after = _clip_buffer_values()

    if clip_filename:
        try:
            source_paths = [str(_resolve_clip_path(clip_filename))]
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        rally = None
        clip_stem = Path(clip_filename).name
        for index, name in enumerate(clip_names):
            if name == clip_stem and index < len(rallies):
                rally = rallies[index]
                break
        hit_groups = [_clip_relative_hits(rally, buffer_before)]
    elif want_clips and not want_original:
        if not clip_names:
            return JSONResponse(
                status_code=409,
                content={"error": "请先完成得分段剪辑，或使用 source=original 分析原始视频"},
            )
        for index, name in enumerate(clip_names):
            try:
                source_paths.append(str(_resolve_clip_path(name)))
            except FileNotFoundError:
                continue
        if not source_paths:
            return JSONResponse(status_code=404, content={"error": "回合片段文件不存在"})
        for index, path in enumerate(source_paths):
            rally = rallies[index] if index < len(rallies) else None
            hit_groups.append(_clip_relative_hits(rally, buffer_before))
    else:
        source_paths = [_video_registry[video_id]["path"]]
        original_hits = []
        for rally in rallies:
            for hit in rally.get("hit_events") or []:
                item = hit.to_dict() if hasattr(hit, "to_dict") else dict(hit)
                original_hits.append(item)
        hit_groups = [original_hits]

    task_id = f"action_{str(uuid.uuid4())[:8]}"
    source_names = [Path(path).name for path in source_paths]
    db.create_task(task_id, mode="action", video_id=video_id, source_names=", ".join(source_names))
    _task_results[task_id] = {
        "status": "processing",
        "mode": "action",
        "sources": source_names,
        "progress": 0,
        "message": "准备动作分析...",
    }
    background_tasks.add_task(
        _run_action_analysis_task,
        source_paths,
        task_id,
        max_frames,
        hit_groups,
        include_frames,
    )
    return {
        "task_id": task_id,
        "status": "processing",
        "mode": "action",
        "sources": source_names,
        "progress": 0,
        "message": "准备动作分析...",
    }


def _action_source_frame_total(video_path: str, max_frames: int) -> int:
    """Estimate the amount of work for one action-analysis source."""
    total_frames = int(_probe_video(video_path).get("total_frames", 0) or 0)
    if max_frames > 0:
        total_frames = min(total_frames, max_frames) if total_frames else max_frames
    return max(1, total_frames)


def _run_action_analysis_task(
    source_paths: list[str],
    task_id: str,
    max_frames: int,
    hit_groups: list[list[dict]] | None = None,
    include_frames: bool = False,
):
    """后台执行动作分析，模型生命周期和回合任务完全分离。"""
    cancel_event = _cancel_events.setdefault(task_id, threading.Event())
    hit_groups = hit_groups or [[] for _ in source_paths]
    try:
        with _analysis_lock:
            if cancel_event.is_set():
                _update_task_result(task_id, status="cancelled", progress=0, message="任务已取消")
                db.update_task(task_id, status="cancelled", progress=0, message="任务已取消")
                return
            analyzer = ActionAnalyzer()
            try:
                source_totals = [
                    _action_source_frame_total(path, max_frames)
                    for path in source_paths
                ]
                total_work = max(1, sum(source_totals))
                completed_work = 0
                analyses = []
                _init_steps(task_id, _ACTION_STEPS)
                _update_step(task_id, "load_model", "processing")
                _update_task_result(
                    task_id,
                    status="processing",
                    mode="action",
                    progress=0,
                    message=f"正在加载姿态模型...",
                )
                _update_step(task_id, "load_model", "completed", 100)
                _update_step(task_id, "analyze_clips", "processing")
                _update_task_result(
                    task_id,
                    progress=0,
                    message=f"正在分析第 1/{len(source_paths)} 个片段 · 0%",
                )
                for source_index, path in enumerate(source_paths):
                    if cancel_event.is_set():
                        break
                    source_total = source_totals[source_index]

                    def report_progress(processed_frames, estimated_frames):
                        current_total = max(1, estimated_frames or source_total)
                        bounded_frames = min(max(0, processed_frames), current_total)
                        progress = round(
                            (completed_work + bounded_frames) / total_work * 100
                        )
                        progress = min(99, max(0, progress))
                        _update_step(task_id, "analyze_clips", "processing", progress)
                        _update_task_result(
                            task_id,
                            progress=progress,
                            message=(
                                f"正在分析第 {source_index + 1}/{len(source_paths)} 个片段"
                                f" · {progress}%"
                            ),
                        )

                    hits = hit_groups[source_index] if source_index < len(hit_groups) else []
                    analysis = analyzer.analyze(
                        path,
                        max_frames=max_frames,
                        progress_callback=report_progress,
                        cancel_event=cancel_event,
                        include_frames=include_frames,
                        hit_events=hits,
                    )
                    analyses.append(analysis)
                    completed_work += source_total
                    completed_progress = round(completed_work / total_work * 100)
                    _update_task_result(
                        task_id,
                        progress=min(99, completed_progress),
                        message=(
                            f"已完成第 {source_index + 1}/{len(source_paths)} 个片段"
                            f" · {min(99, completed_progress)}%"
                        ),
                    )
            finally:
                analyzer.unload_models()

        if cancel_event.is_set():
            _update_task_result(task_id, status="cancelled", progress=0, message="任务已取消")
            db.update_task(task_id, status="cancelled", progress=0, message="任务已取消")
            return

        _update_step(task_id, "analyze_clips", "completed", 100)
        _update_step(task_id, "save_results", "completed", 100)
        _task_results[task_id] = {
            "status": "completed",
            "mode": "action",
            "sources": [Path(path).name for path in source_paths],
            "progress": 100,
            "message": "动作分析完成",
            "analyses": analyses,
            "steps": [
                {"key": k, "name": n, "status": "completed", "progress": 100}
                for k, n in _ACTION_STEPS
            ],
        }
        db.update_task(
            task_id,
            status="completed",
            progress=100,
            message="动作分析完成",
            result={
                "analyses": analyses,
                "sources": [Path(path).name for path in source_paths],
            },
        )
    except Exception as e:
        if cancel_event.is_set():
            _update_task_result(task_id, status="cancelled", progress=0, message="任务已取消")
            db.update_task(task_id, status="cancelled", progress=0, message="任务已取消")
        else:
            logger.error(f"动作分析任务 {task_id}: 失败 - {e}")
            _task_results[task_id] = {
                "status": "failed",
                "mode": "action",
                "progress": 0,
                "error": str(e),
            }
            db.update_task(task_id, status="failed", progress=0, message="动作分析失败")
    finally:
        _cancel_events.pop(task_id, None)


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """查询任务结果"""
    with _task_results_lock:
        result = dict(_task_results.get(task_id, {}))
    if not result:
        # 内存中没有（服务重启或任务已完成被清理），回退到数据库
        task = db.get_task(task_id)
        if not task:
            return JSONResponse(
                status_code=404,
                content={"task_id": task_id, "status": "not_found"},
            )
        result = {
            "task_id": task["task_id"],
            "video_id": task.get("video_id"),
            "mode": task.get("mode"),
            "status": task["status"],
            "progress": task.get("progress", 0),
            "message": task.get("message", ""),
            "project_id": task.get("project_id"),
            "video_filename": task.get("video_filename"),
        }
        # 统一与内存路径一致的数据结构：rallies/total_rallies 在顶层
        task_result = task.get("result") or {}
        result["total_rallies"] = task_result.get("total_rallies", 0)
        result["rallies"] = task_result.get("rallies", [])
        result["clips"] = task.get("clips") if task.get("clips") is not None else []
        if task.get("mode") == "action":
            result["analyses"] = task_result.get("analyses", [])
            result["sources"] = task_result.get("sources", [])
        # 训练任务的指标
        if task.get("mode") == "tracknet_train":
            result["epoch_history"] = task_result.get("epoch_history", [])
            result["metrics"] = task_result.get("metrics", {})
            result["checkpoint_path"] = task_result.get("checkpoint_path")
    return {"task_id": task_id, **result}


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消正在运行的分析任务。"""
    event = _cancel_events.get(task_id)
    if event is not None:
        event.set()
        logger.info(f"任务 {task_id}: 收到取消信号")
        return {"task_id": task_id, "status": "cancelling"}

    # event 不存在但数据库里还是 processing → 任务卡住了，强制标记为已取消
    task = db.get_task(task_id)
    if task and task["status"] == "processing":
        _update_task_result(task_id, status="cancelled", progress=0, message="任务已强制取消")
        db.update_task(task_id, status="cancelled", progress=0, message="任务已强制取消")
        logger.info(f"任务 {task_id}: 强制取消（卡住的任务）")
        return {"task_id": task_id, "status": "cancelled"}

    return JSONResponse(
        status_code=404,
        content={"detail": "任务不存在或已完成"},
    )


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """删除历史任务记录。"""
    # 先获取任务信息（用于重置渲染状态）
    task = db.get_task(task_id)

    deleted = db.delete_task(task_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"detail": "任务不存在"})
    # 同时清理内存中的结果
    with _task_results_lock:
        _task_results.pop(task_id, None)
    _clip_registry.pop(task_id, None)
    _cancel_events.pop(task_id, None)

    # 如果是渲染任务，重置 edit project 的 render 状态
    if task and task.get("mode") == "render" and task.get("project_id"):
        try:
            store = _edit_project_store()
            project = store.get(task["project_id"])
            if project["render"]["status"] in ("rendering", "retrying"):
                store.update_render(
                    task["project_id"],
                    status="failed",
                    progress=0,
                    message="任务已删除",
                    error="用户删除了渲染任务",
                )
        except Exception as exc:
            logger.warning("重置渲染状态失败: {}", exc)

    return {"detail": "任务已删除"}


@app.get("/api/clips/{filename}")
async def download_clip(filename: str, download: bool = False):
    """下载/在线播放剪辑片段"""
    try:
        clip_path = _resolve_clip_path(filename)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "clip not found"})
    # 默认 inline 模式让浏览器 <video> 流式播放；download=true 时触发下载
    if download:
        return FileResponse(str(clip_path), media_type="video/mp4", filename=filename)
    return FileResponse(str(clip_path), media_type="video/mp4")


@app.get("/api/history")
async def list_history(
    mode: str | None = Query(None, description="按模式过滤: rally/action"),
    status: str | None = Query(None, description="按状态过滤: processing/completed/failed/cancelled"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """分页返回历史分析任务记录。"""
    tasks = db.list_tasks(mode=mode, status=status, limit=limit, offset=offset)
    total = db.count_tasks(mode=mode, status=status)
    return {"total": total, "tasks": tasks}


# ---------- 前端静态文件 ----------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def workbench():
    """工作台主页"""
    wb_path = STATIC_DIR / "workbench.html"
    if wb_path.exists():
        return FileResponse(str(wb_path))
    return JSONResponse({"error": "frontend not built"}, status_code=404)


@app.get("/analyze")
async def analyze_page():
    """分析页面"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"error": "frontend not built"}, status_code=404)


@app.get("/login")
async def login_page():
    """登录页面"""
    login_path = STATIC_DIR / "login.html"
    if login_path.exists():
        return FileResponse(str(login_path))
    return JSONResponse({"error": "frontend not built"}, status_code=404)


def main():
    parser = argparse.ArgumentParser(description="PingPong AI Analyst API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8077, help="监听端口")
    parser.add_argument("--config", default=None, help="配置文件路径")
    args = parser.parse_args()

    get_config(args.config)

    # 清理上次服务中断留下的僵尸任务
    try:
        zombie_count = db.cleanup_zombie_tasks()
        if zombie_count > 0:
            logger.info(f"启动时清理了 {zombie_count} 个僵尸任务")
    except Exception as exc:
        logger.warning(f"清理僵尸任务失败: {exc}")

    import uvicorn
    logger.info(f"启动 API 服务: {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
