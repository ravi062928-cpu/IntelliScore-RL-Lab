from __future__ import annotations

import importlib
import json
import sqlite3


def test_q_learning_result_shape(monkeypatch, tmp_path):
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)

    req = main.QLearningRequest(episodes=20, grid_size=5, alpha=0.25, gamma=0.9, epsilon=0.2, max_steps=30, seed=11)
    result = main.train_q_learning(req, "user-test")

    assert result["algorithm"] == "q_learning"
    assert result["episodes"] == req.episodes
    assert result["grid_size"] == req.grid_size
    assert len(result["reward_history"]) == req.episodes
    assert result["q_states_learned"] > 0
    assert 0.0 <= result["success_rate"] <= 1.0


def test_q_learning_repeated_seed_is_consistent(monkeypatch, tmp_path):
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)

    req = main.QLearningRequest(episodes=18, grid_size=4, alpha=0.15, gamma=0.95, epsilon=0.1, max_steps=25, seed=123)
    first = main.train_q_learning(req, "user-a")
    second = main.train_q_learning(req, "user-a")

    assert first["reward_history"] == second["reward_history"]
    assert first["mean_steps"] == second["mean_steps"]


def test_q_learning_exposes_mean_steps_alias(monkeypatch, tmp_path):
    """Regression: both mean_steps and steps_mean should exist for compatibility."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)

    req = main.QLearningRequest(episodes=12, grid_size=5, alpha=0.2, gamma=0.9, epsilon=0.15, max_steps=20, seed=9)
    result = main.train_q_learning(req, "compat-user")

    assert "mean_steps" in result
    assert "steps_mean" in result
    assert result["mean_steps"] == result["steps_mean"]


def test_experiment_data_isolation(monkeypatch, tmp_path):
    """Users cannot access or compare another user's experiments."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)

    # Create two users and one experiment for user-a
    req1 = main.QLearningRequest(episodes=5, grid_size=3, alpha=0.2, gamma=0.9, epsilon=0.1, max_steps=10, seed=1)
    exp1 = main.train_q_learning(req1, "user-a")

    # Create user-b and ensure experiment is not visible
    req2 = main.QLearningRequest(episodes=6, grid_size=3, alpha=0.2, gamma=0.9, epsilon=0.1, max_steps=10, seed=2)
    exp2 = main.train_q_learning(req2, "user-b")

    # User B cannot read user A's experiment id
    assert exp1["experiment_id"] != exp2["experiment_id"]

    # The endpoint check enforces ownership
    from fastapi.testclient import TestClient
    client = TestClient(main.app)

    # Register both users and get tokens
    res_a = client.post("/auth/register", json={"username": "ownerA", "password": "Password123!"})
    res_b = client.post("/auth/register", json={"username": "ownerB", "password": "Password456!"})
    assert res_a.status_code == 201
    assert res_b.status_code == 201

    token_a = client.post("/auth/login", json={"username": "ownerA", "password": "Password123!"}).json()["access_token"]
    token_b = client.post("/auth/login", json={"username": "ownerB", "password": "Password456!"}).json()["access_token"]

    # Create an experiment as A
    exp_as_a = client.post(
        "/experiments/q-learning",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"episodes": 5, "grid_size": 3, "alpha": 0.2, "gamma": 0.9, "epsilon": 0.1, "max_steps": 10, "seed": 7},
    )
    assert exp_as_a.status_code == 200
    owned_id = exp_as_a.json()["experiment_id"]

    # B attempts to fetch A's experiment -> forbidden/404
    res = client.get(f"/experiments/{owned_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert res.status_code == 404

    # B cannot compare against A's experiment ids
    compare = client.post(
        "/experiments/compare",
        headers={"Authorization": f"Bearer {token_b}"},
        json={"experiment_ids": [owned_id, owned_id]},
    )
    assert compare.status_code in [404, 422]


def test_report_exports_include_expected_data(monkeypatch, tmp_path):
    """Reports: JSON and CSV exports contain the expected data."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    client = TestClient(main.app)
    client.post("/auth/register", json={"username": "reporter", "password": "Password123!"})
    token = client.post("/auth/login", json={"username": "reporter", "password": "Password123!"}).json()["access_token"]

    req = {"episodes": 4, "grid_size": 3, "alpha": 0.2, "gamma": 0.9, "epsilon": 0.1, "max_steps": 8, "seed": 99}
    exp = client.post("/experiments/q-learning", headers={"Authorization": f"Bearer {token}"}, json=req)
    assert exp.status_code == 200
    experiment_id = exp.json()["experiment_id"]

    json_resp = client.get(f"/experiments/{experiment_id}/report.json", headers={"Authorization": f"Bearer {token}"})
    assert json_resp.status_code == 200
    json_payload = json_resp.json()
    assert json_payload["algorithm"] == "q_learning"
    assert json_payload["experiment_id"] == experiment_id
    assert "reward_history" in json_payload

    csv_resp = client.get(f"/experiments/{experiment_id}/report.csv", headers={"Authorization": f"Bearer {token}"})
    assert csv_resp.status_code == 200
    csv_text = csv_resp.text
    assert "metric,value" in csv_text
    assert "algorithm" in csv_text
    assert "reward_history" in csv_text


def test_dqn_optional_dependency_behavior(monkeypatch, tmp_path):
    """DQN: optional dependency behavior is graceful when TensorFlow is unavailable."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import builtins
    import main
    importlib.reload(main)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("tensorflow"):
            raise ImportError("No module named tensorflow")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        req = main.DQNRequest(episodes=2, grid_size=3, seed=7)
        try:
            main.train_dqn(req, "user-dqn")
        except Exception as exc:
            assert isinstance(exc, main.HTTPException)
            assert exc.status_code == 503
            assert "TensorFlow" in str(exc.detail) or "tensorflow" in str(exc.detail)
    finally:
        builtins.__import__ = original_import


def test_db_migration_adds_expected_columns(monkeypatch, tmp_path):
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)

    conn = sqlite3.connect(str(sqlite_path))
    columns = {row[1] for row in conn.execute("PRAGMA table_info(experiments)")}
    conn.close()

    assert {"user_id", "seed", "reward_history", "q_states_learned"}.issubset(columns)


def test_db_migration_legacy_schema_support(monkeypatch, tmp_path):
    """Database: fresh setup and legacy schema migration both work."""
    sqlite_path = tmp_path / "legacy.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    conn = sqlite3.connect(str(sqlite_path))
    conn.execute(
        """
        CREATE TABLE experiments (
            id TEXT PRIMARY KEY,
            algorithm TEXT NOT NULL,
            episodes INTEGER NOT NULL,
            grid_size INTEGER NOT NULL,
            goal_row INTEGER NOT NULL,
            goal_col INTEGER NOT NULL,
            mean_reward REAL NOT NULL,
            success_rate REAL NOT NULL,
            steps_mean REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.close()

    import main
    importlib.reload(main)

    conn = sqlite3.connect(str(sqlite_path))
    columns = {row[1] for row in conn.execute("PRAGMA table_info(experiments)")}
    conn.close()

    assert {"user_id", "seed", "reward_history", "q_states_learned"}.issubset(columns)


def test_session_and_user_isolation_with_multiple_accounts(monkeypatch, tmp_path):
    """Security: user data isolation and session control across multiple accounts."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))

    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    client = TestClient(main.app)

    # Register user-a and user-b
    client.post("/auth/register", json={"username": "usera", "password": "Password123!"})
    client.post("/auth/register", json={"username": "userb", "password": "Password456!"})

    token_a = client.post("/auth/login", json={"username": "usera", "password": "Password123!"}).json()["access_token"]
    token_b = client.post("/auth/login", json={"username": "userb", "password": "Password456!"}).json()["access_token"]

    # Both can fetch their own /auth/me
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token_a}"}).status_code == 200
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token_b}"}).status_code == 200

    # Each user should only see their own experiments
    exp_a = client.post(
        "/experiments/q-learning",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"episodes": 3, "grid_size": 3, "alpha": 0.2, "gamma": 0.9, "epsilon": 0.1, "max_steps": 10, "seed": 1},
    )
    exp_b = client.post(
        "/experiments/q-learning",
        headers={"Authorization": f"Bearer {token_b}"},
        json={"episodes": 4, "grid_size": 3, "alpha": 0.2, "gamma": 0.9, "epsilon": 0.1, "max_steps": 10, "seed": 2},
    )

    exp_a_id = exp_a.json()["experiment_id"]
    exp_b_id = exp_b.json()["experiment_id"]

    assert client.get(f"/experiments/{exp_a_id}", headers={"Authorization": f"Bearer {token_a}"}).status_code == 200
    assert client.get(f"/experiments/{exp_b_id}", headers={"Authorization": f"Bearer {token_b}"}).status_code == 200
    assert client.get(f"/experiments/{exp_a_id}", headers={"Authorization": f"Bearer {token_b}"}).status_code == 404
    assert client.get(f"/experiments/{exp_b_id}", headers={"Authorization": f"Bearer {token_a}"}).status_code == 404


def test_expired_session_is_deleted_from_db(monkeypatch, tmp_path):
    """Security: expired sessions are removed from DB when checked."""
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))
    monkeypatch.setenv("SESSION_DAYS", "1")

    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    client = TestClient(main.app)
    client.post("/auth/register", json={"username": "expuser", "password": "Password123!"})
    token = client.post("/auth/login", json={"username": "expuser", "password": "Password123!"}).json()["access_token"]

    with sqlite3.connect(str(sqlite_path)) as conn:
        conn.execute(
            "UPDATE sessions SET expires_at = ? WHERE token_hash = ?",
            ((datetime.now(timezone.utc) - timedelta(days=2)).isoformat(), main.token_digest(token)),
        )
        conn.commit()

    res = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 401

    with sqlite3.connect(str(sqlite_path)) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM sessions WHERE token_hash = ?", (main.token_digest(token),)).fetchone()
    assert rows[0] == 0
