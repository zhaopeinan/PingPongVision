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
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Query, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .core import ActionAnalyzer, VideoAnalyzer
from .core.tracknet_annotations import BallAnnotation, TrackNetAnnotationStore
from .models.tracknet_finetune import TrackNetFineTuner
from .models.tracknet_registry import TrackNetModelRegistry
from .utils.config import get_config, resolve_project_path
from .utils.device_manager import DeviceManager
from .tracking_visualizer import TrackingVisualizer, generate_mjpeg_stream


app = FastAPI(title="PingPong AI Analyst", version="0.2.0")

VIDEO_LIBRARY_DIR = Path("data")
OUTPUT_DIR = Path("output/clips")
STATIC_DIR = Path(__file__).parent / "static"
VIDEO_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

_video_registry: dict[str, dict] = {}
_task_results: dict[str, dict] = {}
_clip_registry: dict[str, list[str]] = {}
_analysis_lock = threading.Lock()
_training_jobs: dict[str, dict] = {}
_training_lock = threading.Lock()

# These defaults are replaceable in tests and keep all adaptation data outside
# source videos and the immutable base checkpoint.
ANNOTATION_DIR = Path("data/tracknet_annotations")
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

# 启动时扫描视频库，自动注册已有视频
def _scan_library():
    """扫描视频库目录，注册所有视频文件"""
    for f in VIDEO_LIBRARY_DIR.rglob("*"):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            # 用文件名（不含路径）作为key检查是否已注册
            rel_name = f.name
            # 检查是否已有注册项指向这个文件
            already_registered = any(
                v["path"] == str(f) for v in _video_registry.values()
            )
            if not already_registered:
                vid = str(uuid.uuid4())[:8]
                info = _probe_video(str(f))
                _video_registry[vid] = {
                    "path": str(f),
                    "filename": rel_name,
                    "size": f.stat().st_size,
                    "info": info,
                }


def get_analyzer(tracknet_model_path: str | None = None) -> VideoAnalyzer:
    """创建一次性分析器, 避免跨任务共享时序状态和模型对象。"""
    return VideoAnalyzer(tracknet_model_path=tracknet_model_path)


def _annotation_store() -> TrackNetAnnotationStore:
    return TrackNetAnnotationStore(ANNOTATION_DIR)


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
    video_id = str(uuid.uuid4())[:8]
    safe_name = file.filename.replace(" ", "_")
    video_path = VIDEO_LIBRARY_DIR / safe_name

    # 重名时加video_id前缀
    if video_path.exists():
        video_path = VIDEO_LIBRARY_DIR / f"{video_id}_{safe_name}"

    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

    video_info = _probe_video(str(video_path))
    _video_registry[video_id] = {
        "path": str(video_path),
        # 展示名保持用户上传的文件名，磁盘重名只影响内部存储路径。
        "filename": safe_name,
        "size": len(content),
        "info": video_info,
    }
    logger.info(f"视频上传: {video_id} -> {video_path} ({len(content)} bytes)")

    return {"video_id": video_id, "filename": safe_name, "size": len(content), "info": video_info}


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
        device_info = DeviceManager.detect(get_config().device_mode)

        def on_progress(payload: dict):
            epoch = int(payload.get("epoch", 0))
            epochs = max(1, int(payload.get("epochs", config.get("epochs", 5))))
            progress = min(99, int(epoch / epochs * 100))
            _update_training_job(
                job_id,
                status=payload.get("status", "running"),
                progress=progress,
                message=f"训练第 {epoch}/{epochs} 轮",
                metrics=payload,
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
            return
        _update_training_job(
            job_id,
            status="completed",
            progress=100,
            message="训练完成，等待保存或放弃",
            result=result,
            checkpoint_path=result.get("checkpoint_path"),
        )
    except Exception as exc:
        logger.exception(f"TrackNet 训练任务 {job_id} 失败")
        try:
            _tracknet_registry().discard_run(_training_jobs[job_id]["run_id"])
        except Exception:
            logger.exception(f"TrackNet 临时任务清理失败: {job_id}")
        _update_training_job(job_id, status="failed", progress=0, message=str(exc), error=str(exc))


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
):
    """MJPEG 实时追踪流 (无元数据, 仅视频)"""
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})
    if mode not in {"rally", "action"}:
        return JSONResponse(status_code=400, content={"error": "invalid mode"})

    model_path = None
    if mode == "rally":
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
    await websocket.accept()

    # 从query string读取模式和人物过滤
    from urllib.parse import parse_qs
    query = parse_qs(websocket.url.query)
    mode = query.get("mode", ["rally"])[0]
    playback_speed = _normalize_playback_speed(query.get("playback_speed", [1.0])[0])
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
    if mode == "rally":
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

    visualizer = TrackingVisualizer(mode=mode, tracknet_model_path=model_path)
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
        await loop.run_in_executor(None, lambda: visualizer.prepare_background(video_path))
        await websocket.send_json({
            "type": "status",
            "status": "tracking",
            "warning": visualizer.tracking_warning,
            "playback_speed": playback_speed,
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

                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]
                if w > target_width:
                    scale = target_width / w
                    img = cv2.resize(img, (target_width, int(h * scale)))

                annotated, meta = visualizer.process_frame(img, fps)

                # 先发 JSON 元数据
                await websocket.send_text(json.dumps({
                    "type": "meta",
                    "frame": meta["frame"],
                    "timestamp": meta["timestamp"],
                    "total_frames": total_frames,
                    "ball_pos": meta["ball_pos"],
                    "ball_speed": meta["ball_speed"],
                    "persons": meta["persons"],
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
    min_boards: int = 6,
    max_frames: int = -1,
    model_id: str = "base",
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
    _video_registry[task_id] = {
        "path": str(video_path),
        "filename": file.filename,
        "size": len(content),
        "info": video_info,
    }
    logger.info(f"任务 {task_id}: 视频已上传 {video_path}")

    try:
        model_path = str(_resolve_tracknet_model(model_id))
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    except LookupError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    background_tasks.add_task(
        _run_analysis_task,
        str(video_path),
        task_id,
        min_boards,
        max_frames,
        model_path,
    )
    return {"task_id": task_id, "status": "processing", "message": "分析已开始"}


@app.post("/api/analyze/{video_id}")
async def analyze_existing(
    background_tasks: BackgroundTasks,
    video_id: str,
    min_boards: int = 6,
    max_frames: int = -1,
    model_id: str = "base",
):
    """分析已上传的视频，只生成回合结果和剪辑。"""
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})

    task_id = video_id
    video_path = _video_registry[video_id]["path"]
    try:
        model_path = str(_resolve_tracknet_model(model_id))
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    except LookupError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    background_tasks.add_task(
        _run_analysis_task,
        video_path,
        task_id,
        min_boards,
        max_frames,
        model_path,
    )
    return {"task_id": task_id, "status": "processing"}


def _run_analysis_task(
    video_path: str,
    task_id: str,
    min_boards: int,
    max_frames: int,
    model_path: str | None = None,
):
    """后台执行分析任务"""
    try:
        # T4 只有16GB显存, 串行分析可避免多个后台任务同时加载三套模型。
        with _analysis_lock:
            analyzer = get_analyzer(tracknet_model_path=model_path)
            analyzer.rally_detector.min_boards = min_boards
            segments = analyzer.analyze(video_path, max_frames=max_frames)
            outputs = analyzer.export_clips(video_path, segments)
            _clip_registry[task_id] = [Path(path).name for path in outputs]

        _task_results[task_id] = {
            "status": "completed",
            "mode": "rally",
            "total_rallies": len(segments),
            "rallies": [
                {
                    "index": i + 1,
                    "start_time": round(s.start_time, 2),
                    "end_time": round(s.end_time, 2),
                    "board_count": s.board_count,
                    "duration": round(s.duration, 2),
                    # 回合模式不运行人体模型，因此不返回击球者/动作事件。
                    "hit_events": [],
                }
                for i, s in enumerate(segments)
            ],
            "clips": [Path(p).name for p in outputs],
        }
        logger.info(f"任务 {task_id}: 完成, {len(segments)} 个回合, {len(outputs)} 个片段")
    except Exception as e:
        logger.error(f"任务 {task_id}: 失败 - {e}")
        _task_results[task_id] = {"status": "failed", "error": str(e)}


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
    all_clips: bool = Query(False, description="是否分析该视频生成的全部回合片段"),
    max_frames: int = Query(-1, description="每个输入最多处理的帧数"),
):
    """独立动作分析入口，可分析原始视频、单个回合或全部回合片段。"""
    if video_id not in _video_registry:
        return JSONResponse(status_code=404, content={"error": "video not found"})
    if clip_filename and all_clips:
        return JSONResponse(status_code=400, content={"error": "clip_filename and all_clips are mutually exclusive"})

    source_paths: list[str] = []
    if clip_filename:
        try:
            source_paths = [str(_resolve_clip_path(clip_filename))]
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
    elif all_clips:
        clip_names = _clip_registry.get(video_id, [])
        if not clip_names:
            return JSONResponse(status_code=400, content={"error": "该视频还没有可分析的回合片段"})
        for name in clip_names:
            try:
                source_paths.append(str(_resolve_clip_path(name)))
            except FileNotFoundError:
                continue
        if not source_paths:
            return JSONResponse(status_code=404, content={"error": "回合片段文件不存在"})
    else:
        source_paths = [_video_registry[video_id]["path"]]

    task_id = f"action_{str(uuid.uuid4())[:8]}"
    _task_results[task_id] = {
        "status": "processing",
        "mode": "action",
        "sources": [Path(path).name for path in source_paths],
    }
    background_tasks.add_task(_run_action_analysis_task, source_paths, task_id, max_frames)
    return {
        "task_id": task_id,
        "status": "processing",
        "mode": "action",
        "sources": [Path(path).name for path in source_paths],
    }


def _run_action_analysis_task(source_paths: list[str], task_id: str, max_frames: int):
    """后台执行动作分析，模型生命周期和回合任务完全分离。"""
    try:
        with _analysis_lock:
            analyzer = ActionAnalyzer()
            try:
                analyses = [
                    analyzer.analyze(path, max_frames=max_frames)
                    for path in source_paths
                ]
            finally:
                analyzer.unload_models()
        _task_results[task_id] = {
            "status": "completed",
            "mode": "action",
            "sources": [Path(path).name for path in source_paths],
            "analyses": analyses,
        }
    except Exception as e:
        logger.error(f"动作分析任务 {task_id}: 失败 - {e}")
        _task_results[task_id] = {
            "status": "failed",
            "mode": "action",
            "error": str(e),
        }


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """查询任务结果"""
    if task_id not in _task_results:
        return JSONResponse(
            status_code=404,
            content={"task_id": task_id, "status": "not_found_or_processing"},
        )
    return {"task_id": task_id, **_task_results[task_id]}


@app.get("/api/clips/{filename}")
async def download_clip(filename: str):
    """下载/在线播放剪辑片段"""
    try:
        clip_path = _resolve_clip_path(filename)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "clip not found"})
    return FileResponse(str(clip_path), media_type="video/mp4", filename=filename)


# ---------- 前端静态文件 ----------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    """主页"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"error": "frontend not built"}, status_code=404)


def main():
    parser = argparse.ArgumentParser(description="PingPong AI Analyst API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8077, help="监听端口")
    parser.add_argument("--config", default=None, help="配置文件路径")
    args = parser.parse_args()

    get_config(args.config)

    import uvicorn
    logger.info(f"启动 API 服务: {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
