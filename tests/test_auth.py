"""auth 认证模块测试。"""

import pytest

from pingpong_analyst.core import database as db
from pingpong_analyst.core.auth import (
    create_token,
    hash_password,
    verify_password,
    verify_token,
    ensure_initial_admin,
)


@pytest.fixture
def db_instance(tmp_path):
    db.init_db(tmp_path / "test.db")
    yield db
    db._CONN = None
    db._DB_PATH = __import__("pathlib").Path("data/pingpong.db")


def test_hash_and_verify_password(db_instance):
    """密码哈希和验证。"""
    pw_hash, pw_salt = hash_password("mypassword")
    assert pw_hash != "mypassword"
    assert len(pw_salt) > 0
    assert verify_password("mypassword", pw_hash, pw_salt)
    assert not verify_password("wrong", pw_hash, pw_salt)


def test_hash_password_with_fixed_salt(db_instance):
    """使用固定 salt 时，相同密码生成相同哈希。"""
    salt = "abcd1234"
    h1, _ = hash_password("test", salt=salt)
    h2, _ = hash_password("test", salt=salt)
    assert h1 == h2


def test_create_and_verify_token(db_instance):
    """令牌生成和验证。"""
    pw_hash, pw_salt = hash_password("pass")
    user = db.create_user("testuser", pw_hash, pw_salt, role="admin")
    token = create_token(user)
    assert isinstance(token, str)
    assert "." in token

    payload = verify_token(token)
    assert payload is not None
    assert payload["username"] == "testuser"
    assert payload["role"] == "admin"
    assert "exp" in payload


def test_verify_token_invalid(db_instance):
    """无效令牌应返回 None。"""
    assert verify_token("invalid") is None
    assert verify_token("a.b.c") is None
    assert verify_token("") is None


def test_ensure_initial_admin(db_instance):
    """首次启动应创建默认管理员。"""
    assert db.count_users() == 0
    ensure_initial_admin()
    assert db.count_users() == 1
    admin = db.get_user_by_username("admin")
    assert admin is not None
    assert admin["role"] == "admin"
    assert verify_password("admin123", admin["password_hash"], admin["password_salt"])

    # 再次调用不应创建新用户
    ensure_initial_admin()
    assert db.count_users() == 1
