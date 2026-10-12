from __future__ import annotations

import importlib
import sqlite3


def test_register_login_logout_and_me(monkeypatch, tmp_path):
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main

    importlib.reload(main)
    from fastapi.testclient import TestClient

    client = TestClient(main.app)

    res = client.post("/auth/register", json={"username": "alice", "password": "strongpass123"})
    assert res.status_code == 201, res.text
    assert res.json()["username"] == "alice"

    login = client.post("/auth/login", json={"username": "alice", "password": "strongpass123"})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    assert token

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200, me.text
    assert me.json()["username"] == "alice"

    logout = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout.status_code == 200, logout.text
    assert logout.json()["message"] == "Logged out"

    revoked = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert revoked.status_code == 401, revoked.text


def test_password_hashing_and_verification(monkeypatch, tmp_path):
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main

    importlib.reload(main)
    password = "SuperSecretPassword!99"

    hashed = main.hash_password(password)
    assert hashed.startswith("pbkdf2_sha256$")
    assert main.verify_password(password, hashed) is True
    assert main.verify_password("wrong-password", hashed) is False


def test_db_migration_adds_expected_columns(monkeypatch, tmp_path):
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main

    importlib.reload(main)

    conn = sqlite3.connect(str(sqlite_path))
    columns = {row[1] for row in conn.execute("PRAGMA table_info(experiments)")}
    conn.close()

    assert {"user_id", "seed", "reward_history", "q_states_learned"}.issubset(columns)
