"""database 持久化层测试。"""

from pathlib import Path

import pytest

from pingpong_analyst.core import database as db


@pytest.fixture
def db_instance(tmp_path):
    """每个测试使用独立的临时数据库。"""
    db.init_db(tmp_path / "test.db")
    yield db
    # 重置全局连接，避免影响后续测试
    db._CONN = None
    db._DB_PATH = Path("data/pingpong.db")


def test_stable_video_id_deterministic(db_instance):
    """相同文件名+大小应生成相同 ID。"""
    id1 = db_instance.stable_video_id("test.mp4", 1024)
    id2 = db_instance.stable_video_id("test.mp4", 1024)
    assert id1 == id2
    assert len(id1) == 8


def test_stable_video_id_differs(db_instance):
    """不同文件名或大小应生成不同 ID。"""
    id1 = db_instance.stable_video_id("test.mp4", 1024)
    id2 = db_instance.stable_video_id("test.mp4", 2048)
    id3 = db_instance.stable_video_id("other.mp4", 1024)
    assert id1 != id2
    assert id1 != id3


def test_upsert_and_get_video(db_instance):
    """插入视频后应能通过 ID 和路径查回。"""
    info = {"width": 1920, "height": 1080, "fps": 30.0, "total_frames": 900, "duration": 30.0}
    record = db_instance.upsert_video("vid1", "test.mp4", "/data/test.mp4", 1024, info)
    assert record["video_id"] == "vid1"
    assert record["filename"] == "test.mp4"
    assert record["info"]["width"] == 1920

    got = db_instance.get_video("vid1")
    assert got is not None
    assert got["filename"] == "test.mp4"

    by_path = db_instance.get_video_by_path("/data/test.mp4")
    assert by_path is not None
    assert by_path["video_id"] == "vid1"


def test_upsert_video_updates_existing(db_instance):
    """重复插入相同 ID 应更新而非报错。"""
    db_instance.upsert_video("vid1", "old.mp4", "/data/old.mp4", 100, {"width": 640})
    db_instance.upsert_video("vid1", "new.mp4", "/data/new.mp4", 200, {"width": 1920})
    got = db_instance.get_video("vid1")
    assert got["filename"] == "new.mp4"
    assert got["size"] == 200
    assert got["info"]["width"] == 1920


def test_list_videos(db_instance):
    """应返回所有视频并按文件名排序。"""
    db_instance.upsert_video("vid_b", "b.mp4", "/data/b.mp4", 100)
    db_instance.upsert_video("vid_a", "a.mp4", "/data/a.mp4", 100)
    videos = db_instance.list_videos()
    assert len(videos) == 2
    assert videos[0]["filename"] == "a.mp4"


def test_create_and_get_task(db_instance):
    """创建任务后应能通过 task_id 查回。"""
    db_instance.upsert_video("vid1", "test.mp4", "/data/test.mp4", 100)
    db_instance.create_task("task1", mode="rally", video_id="vid1")
    task = db_instance.get_task("task1")
    assert task is not None
    assert task["mode"] == "rally"
    assert task["status"] == "processing"
    assert task["progress"] == 0


def test_update_task(db_instance):
    """应能更新任务状态和结果。"""
    db_instance.upsert_video("vid1", "test.mp4", "/data/test.mp4", 100)
    db_instance.create_task("task1", mode="rally", video_id="vid1")
    db_instance.update_task(
        "task1",
        status="completed",
        progress=100,
        message="完成",
        project_id="edit_abc",
        clips=["rally_001.mp4"],
        result={"total_rallies": 3},
    )
    task = db_instance.get_task("task1")
    assert task["status"] == "completed"
    assert task["progress"] == 100
    assert task["project_id"] == "edit_abc"
    assert task["clips"] == ["rally_001.mp4"]
    assert task["result"]["total_rallies"] == 3


def test_list_tasks(db_instance):
    """应能按模式和状态过滤，按创建时间倒序。"""
    db_instance.upsert_video("vid1", "test.mp4", "/data/test.mp4", 100)
    db_instance.upsert_video("vid2", "other.mp4", "/data/other.mp4", 100)
    db_instance.create_task("task1", mode="rally", video_id="vid1")
    db_instance.create_task("task2", mode="action", video_id="vid1")
    db_instance.create_task("task3", mode="rally", video_id="vid2")
    db_instance.update_task("task1", status="completed")

    all_tasks = db_instance.list_tasks()
    assert len(all_tasks) == 3

    rally_tasks = db_instance.list_tasks(mode="rally")
    assert len(rally_tasks) == 2

    completed = db_instance.list_tasks(status="completed")
    assert len(completed) == 1
    assert completed[0]["task_id"] == "task1"


def test_count_tasks(db_instance):
    """应正确计数任务。"""
    db_instance.upsert_video("vid1", "test.mp4", "/data/test.mp4", 100)
    db_instance.create_task("task1", mode="rally", video_id="vid1")
    db_instance.create_task("task2", mode="action", video_id="vid1")
    db_instance.create_task("task3", mode="rally", video_id="vid1")

    assert db_instance.count_tasks() == 3
    assert db_instance.count_tasks(mode="rally") == 2
    assert db_instance.count_tasks(mode="action") == 1
