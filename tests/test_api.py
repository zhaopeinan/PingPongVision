"""
API 接口测试
测试所有 HTTP 和 WebSocket 端点
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import cv2
import numpy as np

import pingpong_analyst.api as api_module
from pingpong_analyst.api import app, _clip_registry, _task_results, _training_jobs, _video_registry


@pytest.fixture
def client(tmp_path, monkeypatch):
    """将 API 测试文件隔离，避免污染正式 data/ 视频库。"""
    test_library = tmp_path / "videos"
    test_library.mkdir()
    test_output = tmp_path / "clips"
    test_output.mkdir()
    monkeypatch.setattr(api_module, "VIDEO_LIBRARY_DIR", test_library)
    monkeypatch.setattr(api_module, "OUTPUT_DIR", test_output)
    monkeypatch.setattr(api_module, "ANNOTATION_DIR", tmp_path / "annotations")
    monkeypatch.setattr(api_module, "MODEL_REGISTRY_DIR", tmp_path / "models" / "runs")
    monkeypatch.setattr(
        api_module,
        "TRACKNET_BASE_PATH",
        Path(__file__).parent.parent / "models" / "TrackNet_best.pt",
    )
    _video_registry.clear()
    _task_results.clear()
    _clip_registry.clear()
    _training_jobs.clear()
    with TestClient(app) as test_client:
        yield test_client
    _video_registry.clear()
    _task_results.clear()
    _clip_registry.clear()
    _training_jobs.clear()


@pytest.fixture
def test_video(tmp_path):
    """生成测试视频"""
    video_path = str(tmp_path / "test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30, (640, 360))
    for i in range(60):
        frame = np.full((360, 640, 3), 40, dtype=np.uint8)
        x = 100 + (i % 30) * 14
        cv2.circle(frame, (x, 180), 12, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return video_path


# ========== 基础接口 ==========

def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_device_info(client):
    res = client.get("/api/device")
    assert res.status_code == 200
    data = res.json()
    assert "device_type" in data
    assert "device_name" in data
    assert "torch_device" in data


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")


# ========== 上传接口 ==========

def test_upload_video(client, test_video):
    with open(test_video, "rb") as f:
        res = client.post("/api/upload", files={"file": ("test.mp4", f, "video/mp4")})

    assert res.status_code == 200
    data = res.json()
    assert "video_id" in data
    assert data["filename"] == "test.mp4"
    assert data["size"] > 0
    assert "info" in data
    assert data["info"]["width"] == 640
    assert data["info"]["height"] == 360
    assert data["info"]["fps"] == 30.0


def test_video_info(client, test_video):
    # 先上传
    with open(test_video, "rb") as f:
        res = client.post("/api/upload", files={"file": ("test.mp4", f, "video/mp4")})
    video_id = res.json()["video_id"]

    # 查询信息
    res = client.get(f"/api/videos/{video_id}/info")
    assert res.status_code == 200
    data = res.json()
    assert data["width"] == 640
    assert data["height"] == 360


def test_video_info_not_found(client):
    res = client.get("/api/videos/nonexistent/info")
    assert res.status_code == 404


def test_tracknet_annotations_and_frame_endpoint(client, test_video):
    with open(test_video, "rb") as f:
        upload = client.post("/api/upload", files={"file": ("annotate.mp4", f, "video/mp4")})
    video_id = upload.json()["video_id"]

    invalid = client.post(
        f"/api/videos/{video_id}/tracknet/annotations",
        json={"annotations": [{"frame_index": 2, "label": "ball", "x": 700, "y": 2}]},
    )
    assert invalid.status_code == 400

    saved = client.post(
        f"/api/videos/{video_id}/tracknet/annotations",
        json={
            "annotations": [
                {"frame_index": 2, "label": "ball", "x": 120, "y": 180},
                {"frame_index": 4, "label": "absent"},
            ]
        },
    )
    assert saved.status_code == 200
    assert saved.json()["count"] == 2

    loaded = client.get(f"/api/videos/{video_id}/tracknet/annotations")
    assert loaded.status_code == 200
    assert loaded.json()["annotations"][0]["label"] == "ball"

    frame = client.get(f"/api/videos/{video_id}/frame?frame_index=2")
    assert frame.status_code == 200
    assert frame.headers["content-type"].startswith("image/jpeg")
    assert frame.content[:2] == b"\xff\xd8"

    out_of_range = client.get(f"/api/videos/{video_id}/frame?frame_index=999")
    assert out_of_range.status_code == 400


def test_table_calibration_lifecycle(client, test_video):
    with open(test_video, "rb") as f:
        upload = client.post("/api/upload", files={"file": ("calibrate.mp4", f, "video/mp4")})
    video_id = upload.json()["video_id"]

    empty = client.get(f"/api/videos/{video_id}/table-calibration")
    assert empty.status_code == 200
    assert empty.json()["calibration"] is None

    saved = client.post(
        f"/api/videos/{video_id}/table-calibration",
        json={
            "frame_index": 5,
            "corners": [[20, 30], [620, 30], [610, 330], [30, 330]],
            "net_points": [[320, 30], [320, 330]],
        },
    )
    assert saved.status_code == 200
    assert saved.json()["calibration"]["frame_index"] == 5

    loaded = client.get(f"/api/videos/{video_id}/table-calibration")
    assert loaded.json()["calibration"]["corners"][0] == [20.0, 30.0]

    invalid = client.post(
        f"/api/videos/{video_id}/table-calibration",
        json={
            "frame_index": 5,
            "corners": [[-1, 30], [620, 30], [610, 330], [30, 330]],
        },
    )
    assert invalid.status_code == 400

    deleted = client.delete(f"/api/videos/{video_id}/table-calibration")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True


def test_tracknet_models_and_selected_model_propagation(client, test_video, monkeypatch):
    with open(test_video, "rb") as f:
        upload = client.post("/api/upload", files={"file": ("model.mp4", f, "video/mp4")})
    video_id = upload.json()["video_id"]

    models = client.get("/api/tracknet/models")
    assert models.status_code == 200
    assert models.json()["models"][0]["model_id"] == "base"

    called = {}

    def fake_analysis(*args):
        called["args"] = args

    monkeypatch.setattr(api_module, "_run_analysis_task", fake_analysis)
    result = client.post(f"/api/analyze/{video_id}?model_id=base&max_frames=1")
    assert result.status_code == 200
    assert called["args"][-1].endswith("models/TrackNet_best.pt")

    missing = client.post(f"/api/analyze/{video_id}?model_id=missing")
    assert missing.status_code == 404


def test_tracknet_training_returns_job_id_without_blocking(client, test_video, monkeypatch):
    with open(test_video, "rb") as f:
        upload = client.post("/api/upload", files={"file": ("train.mp4", f, "video/mp4")})
    video_id = upload.json()["video_id"]
    labels = [
        {"frame_index": index, "label": "ball", "x": 100 + index, "y": 180}
        for index in range(12)
    ]
    assert client.post(
        f"/api/videos/{video_id}/tracknet/annotations",
        json={"annotations": labels},
    ).status_code == 200

    def fake_training(*args):
        return None

    monkeypatch.setattr(api_module, "_run_tracknet_training_task", fake_training)
    response = client.post(f"/api/videos/{video_id}/tracknet/train", json={"model_id": "base"})

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"].startswith("train_")
    status = client.get(f"/api/tracknet/train/{data['job_id']}")
    assert status.status_code == 200
    assert status.json()["status"] in {"queued", "running"}


# ========== 分析接口 ==========

def test_analyze_existing(client, test_video):
    # 上传
    with open(test_video, "rb") as f:
        res = client.post("/api/upload", files={"file": ("test.mp4", f, "video/mp4")})
    video_id = res.json()["video_id"]

    # 分析 (小帧数)
    res = client.post(f"/api/analyze/{video_id}?min_boards=1&max_frames=30")
    assert res.status_code == 200
    assert res.json()["status"] == "processing"

    status = client.get(f"/api/result/{video_id}")
    assert status.status_code == 200
    assert 0 <= status.json()["progress"] <= 100


def test_analyze_not_found(client):
    res = client.post("/api/analyze/nonexistent")
    assert res.status_code == 404


def test_action_analyze_not_found(client):
    res = client.post("/api/action/analyze/nonexistent")
    assert res.status_code == 404


def test_action_analyze_all_requires_rally_clips(client, test_video):
    with open(test_video, "rb") as f:
        res = client.post("/api/upload", files={"file": ("action.mp4", f, "video/mp4")})
    video_id = res.json()["video_id"]

    res = client.post(f"/api/action/analyze/{video_id}?all_clips=true")
    assert res.status_code == 400
    assert "回合片段" in res.json()["error"]


def test_result_not_found(client):
    res = client.get("/api/result/nonexistent")
    assert res.status_code == 404


# ========== 剪辑接口 ==========

def test_clip_not_found(client):
    res = client.get("/api/clips/nonexistent.mp4")
    assert res.status_code == 404


# ========== MJPEG 流接口 ==========

def test_stream_not_found(client):
    res = client.get("/api/stream/nonexistent")
    assert res.status_code == 404


def test_stream_valid(client, test_video):
    """测试 MJPEG 流返回正确 content-type"""
    with open(test_video, "rb") as f:
        res = client.post("/api/upload", files={"file": ("test.mp4", f, "video/mp4")})
    video_id = res.json()["video_id"]

    # 请求流 (只取头部)
    with client.stream("GET", f"/api/stream/{video_id}?max_frames=5&width=320") as response:
        assert response.status_code == 200
        assert "multipart/x-mixed-replace" in response.headers.get("content-type", "")


# ========== WebSocket 测试 ==========

def test_websocket_tracking(client, test_video):
    """测试 WebSocket 追踪流"""
    import json

    # 先上传
    with open(test_video, "rb") as f:
        res = client.post("/api/upload", files={"file": ("test.mp4", f, "video/mp4")})
    video_id = res.json()["video_id"]

    # 连接 WebSocket
    with client.websocket_connect(f"/api/ws/{video_id}?playback_speed=0.5") as ws:
        # 收到 loading 状态
        msg1 = ws.receive_json()
        assert msg1["type"] == "status"
        assert msg1["status"] == "loading"

        # 模型加载期间会发送多条进度消息，直到进入 tracking。
        msg2 = ws.receive_json()
        while msg2.get("status") in {"loading", "preparing"}:
            msg2 = ws.receive_json()
        assert msg2["type"] == "status"
        assert msg2["status"] == "tracking"
        assert msg2["playback_speed"] == 0.5
        tracknet_available = (Path(__file__).parent.parent / "models" / "TrackNet_best.pt").exists()
        assert (msg2["warning"] is None) == tracknet_available

        # 收到元数据
        msg3 = ws.receive_json()
        assert msg3["type"] == "meta"
        assert "frame" in msg3
        assert "timestamp" in msg3
        assert "ball_speed" in msg3
        assert "board_count" in msg3
        assert "rally_count" in msg3
        assert "rally_state" in msg3
        assert msg3["mode"] == "rally"
        assert msg3["persons"] == 0
        assert msg3["arm_angles"] is None
        assert msg3["board_count"] == 0
        assert msg3["ball_tracking"] == ("tracknet" if tracknet_available else "unavailable")

        # 收到二进制帧 (JPEG)
        frame_data = ws.receive_bytes()
        assert len(frame_data) > 0
        # JPEG 魔数
        assert frame_data[0] == 0xFF
        assert frame_data[1] == 0xD8

        # 发送停止
        ws.send_text("stop")


def test_websocket_not_found(client):
    """测试不存在的视频"""
    with client.websocket_connect("/api/ws/nonexistent") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "status"
        assert msg["status"] == "error"
