from __future__ import annotations

import importlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone


def test_register_new_account_succeeds(monkeypatch, tmp_path):
    """Registration: New accounts work."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    client = TestClient(main.app)

    res = client.post("/auth/register", json={"username": "newuser", "password": "securepass123"})
    assert res.status_code == 201, res.text
    data = res.json()
    assert data["username"] == "newuser"
    assert "id" in data
    assert "message" in data


def test_register_duplicate_username_rejected(monkeypatch, tmp_path):
    """Registration: Duplicate usernames are rejected."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    client = TestClient(main.app)

    # First registration
    res1 = client.post("/auth/register", json={"username": "alice", "password": "pass123456"})
    assert res1.status_code == 201

    # Duplicate registration
    res2 = client.post("/auth/register", json={"username": "alice", "password": "pass123456"})
    assert res2.status_code == 409, res2.text
    assert "already exists" in res2.json()["detail"]


def test_register_username_too_short_rejected(monkeypatch, tmp_path):
    """Registration: Username must be at least 3 characters."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    client = TestClient(main.app)

    res = client.post("/auth/register", json={"username": "ab", "password": "pass123456"})
    assert res.status_code == 422


def test_register_password_too_short_rejected(monkeypatch, tmp_path):
    """Registration: Password must be at least 8 characters."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    client = TestClient(main.app)

    res = client.post("/auth/register", json={"username": "validuser", "password": "short"})
    assert res.status_code == 422


def test_login_correct_credentials_work(monkeypatch, tmp_path):
    """Login: Correct credentials work."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    client = TestClient(main.app)

    # Register
    client.post("/auth/register", json={"username": "bob", "password": "correctpass123"})

    # Login with correct credentials
    res = client.post("/auth/login", json={"username": "bob", "password": "correctpass123"})
    assert res.status_code == 200, res.text
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "expires_at" in data
    assert data["user"]["username"] == "bob"


def test_login_incorrect_password_rejected(monkeypatch, tmp_path):
    """Login: Incorrect password is rejected."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    client = TestClient(main.app)

    # Register
    client.post("/auth/register", json={"username": "charlie", "password": "correctpass123"})

    # Login with wrong password
    res = client.post("/auth/login", json={"username": "charlie", "password": "wrongpass"})
    assert res.status_code == 401, res.text
    assert "Invalid username or password" in res.json()["detail"]


def test_login_nonexistent_user_rejected(monkeypatch, tmp_path):
    """Login: Non-existent user is rejected."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    client = TestClient(main.app)

    res = client.post("/auth/login", json={"username": "doesnotexist", "password": "anypass123"})
    assert res.status_code == 401


def test_logout_revoked_token_cannot_authenticate(monkeypatch, tmp_path):
    """Logout: A revoked token can no longer authenticate."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    client = TestClient(main.app)

    # Register and login
    client.post("/auth/register", json={"username": "dave", "password": "securepass123"})
    login_res = client.post("/auth/login", json={"username": "dave", "password": "securepass123"})
    token = login_res.json()["access_token"]

    # Verify token works
    auth_header = {"Authorization": f"Bearer {token}"}
    me_res = client.get("/auth/me", headers=auth_header)
    assert me_res.status_code == 200

    # Logout
    logout_res = client.post("/auth/logout", headers=auth_header)
    assert logout_res.status_code == 200

    # Token should no longer work
    revoked_res = client.get("/auth/me", headers=auth_header)
    assert revoked_res.status_code == 401, revoked_res.text
    assert "Invalid session" in revoked_res.json()["detail"]


def test_session_expired_tokens_rejected(monkeypatch, tmp_path):
    """Sessions: Expired sessions are rejected."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))
    monkeypatch.setenv("SESSION_DAYS", "0")  # Sessions expire immediately

    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    client = TestClient(main.app)

    # Register and login
    client.post("/auth/register", json={"username": "eve", "password": "securepass123"})
    login_res = client.post("/auth/login", json={"username": "eve", "password": "securepass123"})
    token = login_res.json()["access_token"]

    # Try to use expired token
    auth_header = {"Authorization": f"Bearer {token}"}
    res = client.get("/auth/me", headers=auth_header)
    assert res.status_code == 401
    assert "expired" in res.json()["detail"].lower() or "Invalid session" in res.json()["detail"]


def test_no_token_rejects_protected_endpoint(monkeypatch, tmp_path):
    """Sessions: Protected endpoints require token."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    client = TestClient(main.app)

    # Try to access /auth/me without token
    res = client.get("/auth/me")
    assert res.status_code == 403  # Forbidden or 401

    # Try to access /experiments without token
    res = client.get("/experiments")
    assert res.status_code in [401, 403]
