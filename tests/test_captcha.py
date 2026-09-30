"""captcha 验证码与限流模块测试。"""

import pytest

from pingpong_analyst.core.captcha import (
    _captcha_store,
    _login_attempts,
    check_rate_limit,
    clear_login_failures,
    generate_captcha,
    record_login_failure,
    verify_captcha,
)


@pytest.fixture(autouse=True)
def clean_state():
    """每个测试前清理验证码和限流状态。"""
    _captcha_store.clear()
    _login_attempts.clear()
    yield
    _captcha_store.clear()
    _login_attempts.clear()


def test_generate_captcha():
    """生成验证码应返回 captcha_id 和 image。"""
    data = generate_captcha()
    assert "captcha_id" in data
    assert "image" in data
    assert len(data["captcha_id"]) > 0
    assert data["image"].startswith("data:image/svg+xml;base64,")


def test_verify_captcha_correct():
    """正确验证码应通过验证。"""
    # 生成验证码后，从内部存储中读取文本
    from pingpong_analyst.core.captcha import _captcha_store
    data = generate_captcha()
    captcha_id = data["captcha_id"]
    text = _captcha_store[captcha_id]["text"]
    assert verify_captcha(captcha_id, text) is True


def test_verify_captcha_wrong():
    """错误验证码应失败。"""
    data = generate_captcha()
    assert verify_captcha(data["captcha_id"], "WRONG") is False


def test_verify_captcha_one_time():
    """验证码一次性使用，第二次验证应失败。"""
    from pingpong_analyst.core.captcha import _captcha_store
    data = generate_captcha()
    captcha_id = data["captcha_id"]
    text = _captcha_store[captcha_id]["text"]
    assert verify_captcha(captcha_id, text) is True
    # 第二次使用同一个 captcha_id
    assert verify_captcha(captcha_id, text) is False


def test_verify_captcha_nonexistent():
    """不存在的 captcha_id 应失败。"""
    assert verify_captcha("nonexistent", "ABCD") is False


def test_rate_limit_allows_initial():
    """初始状态不应限流。"""
    allowed, retry = check_rate_limit("1.2.3.4", "user")
    assert allowed is True
    assert retry == 0


def test_rate_limit_after_failures():
    """失败 5 次后应被限流。"""
    ip, username = "1.2.3.5", "testuser"
    for _ in range(5):
        record_login_failure(ip, username)
    allowed, retry = check_rate_limit(ip, username)
    assert allowed is False
    assert retry > 0


def test_rate_limit_clears_on_success():
    """成功后清除失败记录。"""
    ip, username = "1.2.3.6", "testuser2"
    for _ in range(3):
        record_login_failure(ip, username)
    clear_login_failures(ip, username)
    allowed, retry = check_rate_limit(ip, username)
    assert allowed is True


def test_rate_limit_separate_users():
    """不同用户的限流应独立。"""
    ip = "1.2.3.7"
    for _ in range(5):
        record_login_failure(ip, "user_a")
    # user_b 不应被限流
    allowed, _ = check_rate_limit(ip, "user_b")
    assert allowed is True
