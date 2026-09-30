"""SQLite 持久化层，让视频和分析任务在服务重启后仍可恢复。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

_LOCK = threading.Lock()
_CONN: sqlite3.Connection | None = None
_DB_PATH: Path = Path("data/pingpong.db")


def init_db(db_path: Path | str | None = None) -> None:
    """初始化数据库连接并创建表结构。"""
    global _CONN, _DB_PATH
    if db_path is not None:
        _DB_PATH = Path(db_path)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONN = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    _CONN.row_factory = sqlite3.Row
    _CONN.execute("PRAGMA journal_mode=WAL")
    _CONN.execute("PRAGMA foreign_keys=ON")
    _create_tables()
    logger.info(f"数据库已初始化: {_DB_PATH}")


def _conn() -> sqlite3.Connection:
    if _CONN is None:
        init_db()
    assert _CONN is not None
    return _CONN


def _create_tables() -> None:
    conn = _conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS videos (
        video_id     TEXT PRIMARY KEY,
        filename     TEXT NOT NULL,
        storage_path TEXT NOT NULL,
        size         INTEGER NOT NULL,
        width        INTEGER DEFAULT 0,
        height       INTEGER DEFAULT 0,
        fps          REAL DEFAULT 0,
        total_frames INTEGER DEFAULT 0,
        duration     REAL DEFAULT 0,
        created_at   TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS analysis_tasks (
        task_id      TEXT PRIMARY KEY,
        video_id     TEXT,
        mode         TEXT NOT NULL,
        status       TEXT NOT NULL,
        progress     INTEGER DEFAULT 0,
        message      TEXT DEFAULT '',
        result_json  TEXT,
        clips_json   TEXT,
        project_id   TEXT,
        source_names TEXT,
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL,
        FOREIGN KEY (video_id) REFERENCES videos(video_id)
    );

    CREATE INDEX IF NOT EXISTS idx_tasks_video ON analysis_tasks(video_id);
    CREATE INDEX IF NOT EXISTS idx_tasks_status ON analysis_tasks(status);
    CREATE INDEX IF NOT EXISTS idx_tasks_created ON analysis_tasks(created_at DESC);

    CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        password_salt TEXT NOT NULL,
        role          TEXT NOT NULL DEFAULT 'analyst',
        active        INTEGER NOT NULL DEFAULT 1,
        created_at    TEXT NOT NULL,
        last_login    TEXT
    );
    """)


# ---------- 视频 ----------

def stable_video_id(filename: str, size: int) -> str:
    """根据文件名和大小生成稳定的 video_id，重启后不变。"""
    raw = f"{filename}:{size}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def upsert_video(
    video_id: str,
    filename: str,
    storage_path: str,
    size: int,
    info: dict | None = None,
) -> dict:
    """插入或更新视频记录，返回完整的视频信息字典。"""
    info = info or {}
    now = _now()
    with _LOCK:
        _conn().execute(
            """INSERT INTO videos (video_id, filename, storage_path, size,
               width, height, fps, total_frames, duration, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(video_id) DO UPDATE SET
                 filename=excluded.filename,
                 storage_path=excluded.storage_path,
                 size=excluded.size,
                 width=excluded.width,
                 height=excluded.height,
                 fps=excluded.fps,
                 total_frames=excluded.total_frames,
                 duration=excluded.duration""",
            (
                video_id, filename, storage_path, size,
                info.get("width", 0), info.get("height", 0),
                info.get("fps", 0), info.get("total_frames", 0),
                info.get("duration", 0), now,
            ),
        )
        _conn().commit()
    return get_video(video_id) or {}


def get_video(video_id: str) -> dict | None:
    with _LOCK:
        row = _conn().execute(
            "SELECT * FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
    return _row_to_video(row) if row else None


def get_video_by_path(path: str) -> dict | None:
    with _LOCK:
        row = _conn().execute(
            "SELECT * FROM videos WHERE storage_path = ?", (path,)
        ).fetchone()
    return _row_to_video(row) if row else None


def list_videos() -> list[dict]:
    with _LOCK:
        rows = _conn().execute(
            "SELECT * FROM videos ORDER BY filename"
        ).fetchall()
    return [_row_to_video(r) for r in rows]


def delete_video(video_id: str) -> dict | None:
    """删除视频记录及关联任务，返回被删除的视频信息（含 storage_path）。"""
    with _LOCK:
        row = _conn().execute(
            "SELECT * FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        if not row:
            return None
        _conn().execute("DELETE FROM analysis_tasks WHERE video_id = ?", (video_id,))
        _conn().execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        _conn().commit()
    return _row_to_video(row)


def _row_to_video(row: sqlite3.Row) -> dict:
    return {
        "video_id": row["video_id"],
        "filename": row["filename"],
        "path": row["storage_path"],
        "size": row["size"],
        "info": {
            "width": row["width"],
            "height": row["height"],
            "fps": row["fps"],
            "total_frames": row["total_frames"],
            "duration": row["duration"],
        },
    }


# ---------- 分析任务 ----------

def create_task(
    task_id: str,
    mode: str,
    video_id: str | None = None,
    source_names: str | None = None,
) -> None:
    """创建一条新的任务记录。"""
    now = _now()
    with _LOCK:
        _conn().execute(
            """INSERT INTO analysis_tasks
               (task_id, video_id, mode, status, progress, message,
                result_json, clips_json, project_id, source_names,
                created_at, updated_at)
               VALUES (?, ?, ?, 'processing', 0, '', NULL, NULL, NULL, ?, ?, ?)""",
            (task_id, video_id, mode, source_names, now, now),
        )
        _conn().commit()


def update_task(task_id: str, **fields: Any) -> None:
    """更新任务字段，支持 status/progress/message/result/clips/project_id。"""
    if not fields:
        return
    sets: list[str] = []
    vals: list[Any] = []
    mapping = {
        "status": "status",
        "progress": "progress",
        "message": "message",
        "project_id": "project_id",
    }
    for key, col in mapping.items():
        if key in fields:
            sets.append(f"{col} = ?")
            vals.append(fields[key])
    if "result" in fields:
        sets.append("result_json = ?")
        vals.append(json.dumps(fields["result"], ensure_ascii=False))
    if "clips" in fields:
        sets.append("clips_json = ?")
        vals.append(json.dumps(fields["clips"], ensure_ascii=False))
    sets.append("updated_at = ?")
    vals.append(_now())
    vals.append(task_id)
    with _LOCK:
        _conn().execute(
            f"UPDATE analysis_tasks SET {', '.join(sets)} WHERE task_id = ?",
            vals,
        )
        _conn().commit()


def get_task(task_id: str) -> dict | None:
    with _LOCK:
        row = _conn().execute(
            """SELECT t.*, v.duration AS video_duration, v.filename AS video_filename
               FROM analysis_tasks t
               LEFT JOIN videos v ON t.video_id = v.video_id
               WHERE t.task_id = ?""",
            (task_id,),
        ).fetchone()
    return _row_to_task(row) if row else None


def list_tasks(
    mode: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    sql = """
        SELECT t.*, v.duration AS video_duration, v.filename AS video_filename
        FROM analysis_tasks t
        LEFT JOIN videos v ON t.video_id = v.video_id
    """
    clauses: list[str] = []
    vals: list[Any] = []
    if mode:
        clauses.append("t.mode = ?")
        vals.append(mode)
    if status:
        clauses.append("t.status = ?")
        vals.append(status)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY t.created_at DESC LIMIT ? OFFSET ?"
    vals.extend([limit, offset])
    with _LOCK:
        rows = _conn().execute(sql, vals).fetchall()
    return [_row_to_task(r) for r in rows]


def count_tasks(mode: str | None = None, status: str | None = None) -> int:
    sql = "SELECT COUNT(*) as n FROM analysis_tasks"
    clauses: list[str] = []
    vals: list[Any] = []
    if mode:
        clauses.append("mode = ?")
        vals.append(mode)
    if status:
        clauses.append("status = ?")
        vals.append(status)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    with _LOCK:
        row = _conn().execute(sql, vals).fetchone()
    return row["n"] if row else 0


def delete_task(task_id: str) -> bool:
    """删除任务记录，返回是否删除成功。"""
    with _LOCK:
        cur = _conn().execute("DELETE FROM analysis_tasks WHERE task_id = ?", (task_id,))
        _conn().commit()
    return cur.rowcount > 0


def cleanup_zombie_tasks() -> int:
    """将所有 processing 状态的任务标记为 failed（服务重启时调用）。"""
    with _LOCK:
        cur = _conn().execute(
            "UPDATE analysis_tasks SET status='failed', message='服务重启后任务中断', updated_at=? "
            "WHERE status='processing'",
            (_now(),),
        )
        _conn().commit()
    return cur.rowcount


def _row_to_task(row: sqlite3.Row) -> dict:
    result: dict[str, Any] = {
        "task_id": row["task_id"],
        "video_id": row["video_id"],
        "mode": row["mode"],
        "status": row["status"],
        "progress": row["progress"],
        "message": row["message"],
        "project_id": row["project_id"],
        "source_names": row["source_names"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "video_duration": row["video_duration"] if "video_duration" in row.keys() else None,
        "video_filename": row["video_filename"] if "video_filename" in row.keys() else None,
    }
    if row["result_json"]:
        result["result"] = json.loads(row["result_json"])
    if row["clips_json"]:
        result["clips"] = json.loads(row["clips_json"])
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- 用户 ----------

def create_user(username: str, password_hash: str, password_salt: str, role: str = "analyst") -> dict:
    """创建用户，返回用户字典。"""
    now = _now()
    with _LOCK:
        _conn().execute(
            """INSERT INTO users (username, password_hash, password_salt, role, active, created_at)
               VALUES (?, ?, ?, ?, 1, ?)""",
            (username, password_hash, password_salt, role, now),
        )
        _conn().commit()
    return get_user_by_username(username) or {}


def get_user_by_username(username: str) -> dict | None:
    with _LOCK:
        row = _conn().execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return _row_to_user(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with _LOCK:
        row = _conn().execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return _row_to_user(row) if row else None


def list_users() -> list[dict]:
    with _LOCK:
        rows = _conn().execute(
            "SELECT * FROM users ORDER BY created_at"
        ).fetchall()
    return [_row_to_user(r) for r in rows]


def update_user(user_id: int, **fields: Any) -> None:
    sets: list[str] = []
    vals: list[Any] = []
    for key in ("password_hash", "password_salt", "role", "active", "last_login"):
        if key in fields:
            sets.append(f"{key} = ?")
            vals.append(fields[key])
    if not sets:
        return
    vals.append(user_id)
    with _LOCK:
        _conn().execute(
            f"UPDATE users SET {', '.join(sets)} WHERE id = ?", vals
        )
        _conn().commit()


def delete_user(user_id: int) -> None:
    with _LOCK:
        _conn().execute("DELETE FROM users WHERE id = ?", (user_id,))
        _conn().commit()


def count_users() -> int:
    with _LOCK:
        row = _conn().execute("SELECT COUNT(*) as n FROM users").fetchone()
    return row["n"] if row else 0


def _row_to_user(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "username": row["username"],
        "password_hash": row["password_hash"],
        "password_salt": row["password_salt"],
        "role": row["role"],
        "active": bool(row["active"]),
        "created_at": row["created_at"],
        "last_login": row["last_login"],
    }
