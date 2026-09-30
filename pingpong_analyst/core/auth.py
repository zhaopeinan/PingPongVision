"""认证模块：JWT 令牌 + PBKDF2 密码哈希，仅使用 Python 内置库。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from loguru import logger

from . import database as db

# ---------- 配置 ----------

_TOKEN_EXPIRY = 86400  # 24 小时（秒）
_PBKDF2_ITERATIONS = 100_000
_PBKDF2_ALGO = "sha256"


def _load_or_create_secret() -> str:
    """从数据目录加载持久化的 token secret，不存在则生成并保存。"""
    from pathlib import Path
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    secret_file = data_dir / ".token_secret"
    if secret_file.exists():
        return secret_file.read_text().strip()
    secret = secrets.token_urlsafe(32)
    secret_file.write_text(secret)
    secret_file.chmod(0o600)
    logger.info("已生成新的 token 签名密钥并持久化")
    return secret


_TOKEN_SECRET = _load_or_create_secret()

_security = HTTPBearer(auto_error=False)


def set_token_secret(secret: str) -> None:
    """覆盖令牌签名密钥（用于测试或配置注入）。"""
    global _TOKEN_SECRET
    _TOKEN_SECRET = secret


# ---------- 密码哈希 ----------

def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """返回 (password_hash, password_salt)。"""
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(_PBKDF2_ALGO, password.encode(), salt.encode(), _PBKDF2_ITERATIONS)
    return dk.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """验证密码是否匹配。"""
    dk = hashlib.pbkdf2_hmac(_PBKDF2_ALGO, password.encode(), salt.encode(), _PBKDF2_ITERATIONS)
    return hmac.compare_digest(dk.hex(), password_hash)


# ---------- JWT 令牌 ----------

def create_token(user: dict) -> str:
    """为用户生成 JWT 令牌。"""
    payload = {
        "user_id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "exp": int(time.time()) + _TOKEN_EXPIRY,
        "iat": int(time.time()),
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
    sig = hmac.new(_TOKEN_SECRET.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def verify_token(token: str) -> dict | None:
    """验证令牌，返回 payload 或 None。"""
    try:
        payload_b64, sig = token.rsplit(".", 1)
    except ValueError:
        return None
    expected_sig = hmac.new(_TOKEN_SECRET.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None
    try:
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload


# ---------- 首次启动初始化 ----------

def ensure_initial_admin() -> None:
    """如果没有任何用户，创建默认管理员 admin / admin123。"""
    if db.count_users() > 0:
        return
    password_hash, password_salt = hash_password("admin123")
    db.create_user("admin", password_hash, password_salt, role="admin")
    logger.warning("已创建默认管理员 admin / admin123，请尽快修改密码！")


# ---------- FastAPI 依赖 ----------

async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> dict:
    """从请求头提取并验证 JWT 令牌，返回用户信息。"""
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="未提供认证令牌")
    payload = verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="令牌无效或已过期")
    user = db.get_user_by_id(payload["user_id"])
    if user is None or not user["active"]:
        raise HTTPException(status_code=401, detail="用户不存在或已被禁用")
    return user


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """要求当前用户是管理员。"""
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


def update_last_login(user_id: int) -> None:
    """更新用户最后登录时间。"""
    db.update_user(user_id, last_login=datetime.now(timezone.utc).isoformat())
