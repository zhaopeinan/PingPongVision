"""验证码生成与登录限流。"""

from __future__ import annotations

import base64
import random
import string
import threading
import time
from typing import Any

# ---------- 验证码存储 ----------

_CAPTCHA_TTL = 300  # 5 分钟
_captcha_store: dict[str, dict[str, Any]] = {}
_captcha_lock = threading.Lock()


def generate_captcha() -> dict[str, str]:
    """生成一个验证码，返回 {captcha_id, image}。image 是 data URI。"""
    # 清理过期验证码
    now = time.time()
    with _captcha_lock:
        expired = [k for k, v in _captcha_store.items() if v["expires"] < now]
        for k in expired:
            del _captcha_store[k]

    # 生成验证码文本
    chars = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    # 排除容易混淆的字符
    chars = chars.replace("O", "A").replace("0", "3").replace("I", "T").replace("1", "7")

    captcha_id = base64.urlsafe_b64encode(random.randbytes(16)).rstrip(b"=").decode()

    with _captcha_lock:
        _captcha_store[captcha_id] = {
            "text": chars,
            "expires": now + _CAPTCHA_TTL,
        }

    # 生成 SVG 图片
    image = _render_svg(chars)
    return {"captcha_id": captcha_id, "image": image}


def verify_captcha(captcha_id: str, text: str) -> bool:
    """验证验证码，验证后立即删除（一次性使用）。"""
    now = time.time()
    with _captcha_lock:
        entry = _captcha_store.get(captcha_id)
        if entry is None:
            return False
        # 一次性使用
        del _captcha_store[captcha_id]
        if entry["expires"] < now:
            return False
        return entry["text"].upper() == text.strip().upper()


def _render_svg(text: str) -> str:
    """渲染验证码 SVG，返回 base64 data URI。"""
    colors = ["#e8833a", "#3acbe8", "#3ae872", "#e8c83a", "#e83a8a"]
    font_sizes = [28, 30, 32]
    x = 15
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="160" height="50" viewBox="0 0 160 50">']
    # 背景渐变
    parts.append('<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
                 '<stop offset="0" stop-color="#2a2520"/><stop offset="1" stop-color="#1e1a16"/>'
                 '</linearGradient></defs>')
    parts.append('<rect width="160" height="50" fill="url(#bg)"/>')
    # 噪声线
    for _ in range(6):
        x1, y1 = random.randint(0, 160), random.randint(0, 50)
        x2, y2 = random.randint(0, 160), random.randint(0, 50)
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                     f'stroke="{random.choice(colors)}" stroke-width="1" opacity="0.3"/>')
    # 字符
    for i, ch in enumerate(text):
        color = colors[i % len(colors)]
        size = random.choice(font_sizes)
        rot = random.randint(-20, 20)
        y = 35 + random.randint(-3, 3)
        parts.append(
            f'<text x="{x}" y="{y}" font-family="monospace" font-size="{size}" '
            f'font-weight="bold" fill="{color}" '
            f'transform="rotate({rot} {x} {y})">{ch}</text>'
        )
        x += 35
    parts.append('</svg>')
    svg = "".join(parts)
    b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"


# ---------- 登录限流 ----------

_MAX_FAILURES = 5
_LOCK_DURATION = 900  # 15 分钟
_login_attempts: dict[str, dict[str, Any]] = {}
_login_lock = threading.Lock()


def check_rate_limit(ip: str, username: str) -> tuple[bool, int]:
    """检查是否被限流。返回 (allowed, retry_after_seconds)。"""
    key = f"{ip}:{username}"
    now = time.time()
    with _login_lock:
        entry = _login_attempts.get(key)
        if entry and entry.get("locked_until", 0) > now:
            retry = int(entry["locked_until"] - now)
            return False, retry
        return True, 0


def record_login_failure(ip: str, username: str) -> int:
    """记录登录失败，返回剩余尝试次数。"""
    key = f"{ip}:{username}"
    now = time.time()
    with _login_lock:
        entry = _login_attempts.get(key, {"failures": 0})
        # 如果锁定已过期，重置计数
        locked_until = entry.get("locked_until", 0)
        if locked_until and locked_until < now:
            entry = {"failures": 0}
        entry["failures"] = entry.get("failures", 0) + 1
        if entry["failures"] >= _MAX_FAILURES:
            entry["locked_until"] = now + _LOCK_DURATION
        _login_attempts[key] = entry
        remaining = _MAX_FAILURES - entry["failures"]
        return max(0, remaining)


def clear_login_failures(ip: str, username: str) -> None:
    """登录成功后清除失败记录。"""
    key = f"{ip}:{username}"
    with _login_lock:
        _login_attempts.pop(key, None)
