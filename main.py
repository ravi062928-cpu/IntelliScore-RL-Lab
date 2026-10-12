from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
import random
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

APP_NAME = "IntelliScore RL Lab"
APP_VERSION = "1.3.0"
DB_PATH = os.getenv("DB_PATH", "intelliscore.db")
SESSION_DAYS = int(os.getenv("SESSION_DAYS", "7"))
PASSWORD_ITERATIONS = 600_000

if os.path.dirname(DB_PATH):
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_NAME, version=APP_VERSION)
bearer = HTTPBearer(auto_error=False)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with db_connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
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

            columns = {row["name"] for row in conn.execute("PRAGMA table_info(experiments)")}
            additions = {
                "user_id": "TEXT REFERENCES users(id) ON DELETE CASCADE",
                "seed": "INTEGER NOT NULL DEFAULT 42",
                "reward_history": "TEXT NOT NULL DEFAULT '[]'",
                "q_states_learned": "INTEGER NOT NULL DEFAULT 0",
            }
            for name, definition in additions.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE experiments ADD COLUMN {name} {definition}")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_user_created ON experiments(user_id, created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
            conn.commit()
        except Exception:
            conn.rollback()
            raise


init_db()


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_hex, digest_hex = stored.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(actual.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=1, max_length=128)


class QLearningRequest(BaseModel):
    episodes: int = Field(default=300, ge=1, le=5000)
    grid_size: int = Field(default=6, ge=2, le=30)
    alpha: float = Field(default=0.1, gt=0, le=1)
    gamma: float = Field(default=0.95, ge=0, le=1)
    epsilon: float = Field(default=0.2, ge=0, le=1)
    max_steps: int = Field(default=100, ge=1, le=1000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")

    digest = token_digest(credentials.credentials)
    with db_connect() as conn:
        row = conn.execute(
            """
            SELECT users.id, users.username, users.created_at, sessions.expires_at
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ?
            """,
            (digest,),
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
        if datetime.fromisoformat(row["expires_at"]) <= utc_now():
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (digest,))
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired; please log in again")

    return {"id": row["id"], "username": row["username"], "created_at": row["created_at"]}


class GridWorld:
    def __init__(self, grid_size: int = 6, goal: tuple[int, int] | None = None):
        if grid_size < 2 or grid_size > 30:
            raise ValueError("grid_size must be between 2 and 30")
        self.grid_size = grid_size
        self.goal = goal or (grid_size - 1, grid_size - 1)
        if not all(0 <= value < grid_size for value in self.goal):
            raise ValueError("goal must be inside the grid")
        self.position = (0, 0)

    def reset(self) -> tuple[int, int]:
        self.position = (0, 0)
        return self.position

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        if action not in (0, 1, 2, 3):
            raise ValueError("action must be 0, 1, 2, or 3")
        row, col = self.position
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            col = min(self.grid_size - 1, col + 1)
        elif action == 2:
            row = min(self.grid_size - 1, row + 1)
        else:
            col = max(0, col - 1)
        self.position = (row, col)
        done = self.position == self.goal
        return self.position, (10.0 if done else -1.0), done


def train_q_learning(req: QLearningRequest, user_id: str) -> dict[str, Any]:
    rng = random.Random(req.seed)
    env = GridWorld(req.grid_size)
    q_table: dict[tuple[int, int], np.ndarray] = {}
    rewards: list[float] = []
    steps_per_episode: list[int] = []
    successes = 0

    def row_for(state: tuple[int, int]) -> np.ndarray:
        if state not in q_table:
            q_table[state] = np.zeros(4, dtype=np.float64)
        return q_table[state]

    for _ in range(req.episodes):
        state = env.reset()
        total_reward = 0.0
        reached_goal = False
        for step_no in range(1, req.max_steps + 1):
            if rng.random() < req.epsilon:
                action = rng.randrange(4)
            else:
                values = row_for(state)
                best = np.flatnonzero(values == np.max(values))
                action = int(rng.choice(best.tolist()))

            next_state, reward, done = env.step(action)
            current = row_for(state)
            future = 0.0 if done else float(np.max(row_for(next_state)))
            current[action] += req.alpha * (reward + req.gamma * future - current[action])
            total_reward += reward
            state = next_state
            if done:
                reached_goal = True
                successes += 1
                steps_per_episode.append(step_no)
                break

        if not reached_goal:
            steps_per_episode.append(req.max_steps)
        rewards.append(total_reward)

    result = {
        "algorithm": "q_learning",
        "episodes": req.episodes,
        "grid_size": req.grid_size,
        "goal": list(env.goal),
        "mean_reward": round(float(np.mean(rewards)), 4),
        "success_rate": round(successes / req.episodes, 4),
        "mean_steps": round(float(np.mean(steps_per_episode)), 4),
        "q_states_learned": len(q_table),
        "reward_history": [round(x, 4) for x in rewards],
        "seed": req.seed,
    }

    experiment_id = str(uuid.uuid4())
    created_at = utc_now().isoformat()
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO experiments
            (id, user_id, algorithm, episodes, grid_size, goal_row, goal_col,
             mean_reward, success_rate, steps_mean, created_at, seed,
             reward_history, q_states_learned)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                user_id,
                "q_learning",
                req.episodes,
                req.grid_size,
                env.goal[0],
                env.goal[1],
                result["mean_reward"],
                result["success_rate"],
                result["mean_steps"],
                created_at,
                req.seed,
                json.dumps(result["reward_history"]),
                result["q_states_learned"],
            ),
        )

    result.update({"experiment_id": experiment_id, "created_at": created_at})
    return result


@app.post("/auth/register", status_code=201)
def register(req: RegisterRequest):
    username = req.username.strip()
    if len(username) < 3:
        raise HTTPException(status_code=422, detail="Username must contain at least 3 characters")
    user_id = str(uuid.uuid4())
    try:
        with db_connect() as conn:
            conn.execute(
                "INSERT INTO users(id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (user_id, username, hash_password(req.password), utc_now().isoformat()),
            )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Username already exists") from exc
    return {"id": user_id, "username": username, "message": "Account created. Log in to continue."}


@app.post("/auth/login")
def login(req: LoginRequest):
    with db_connect() as conn:
        user = conn.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ? COLLATE NOCASE",
            (req.username.strip(),),
        ).fetchone()

    if user is None or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = secrets.token_urlsafe(32)
    created = utc_now()
    expires = created + timedelta(days=max(1, min(SESSION_DAYS, 30)))

    with db_connect() as conn:
        conn.execute(
            "INSERT INTO sessions(token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (token_digest(token), user["id"], expires.isoformat(), created.isoformat()),
        )

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_at": expires.isoformat(),
        "user": {"id": user["id"], "username": user["username"]},
    }


@app.post("/auth/logout")
def logout(user: dict[str, Any] = Depends(current_user), credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")
    with db_connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_digest(credentials.credentials),))
    return {"message": "Logged out"}


@app.get("/auth/me")
def me(user: dict[str, Any] = Depends(current_user)):
    return user


@app.get("/")
def home():
    return {
        "name": APP_NAME,
        "version": app.version,
        "status": "running",
        "docs": "/docs",
        "auth": "/auth/register and /auth/login",
    }


@app.get("/health")
def health():
    try:
        with db_connect() as conn:
            conn.execute("SELECT 1")
        return {"status": "ok", "database": "sqlite"}
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc


@app.post("/experiments/q-learning")
def run_q_learning(req: QLearningRequest, user: dict[str, Any] = Depends(current_user)):
    try:
        return train_q_learning(req, user["id"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/experiments")
def list_experiments(limit: int = 20, user: dict[str, Any] = Depends(current_user)):
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=400, detail="limit must be 1..100")
    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, algorithm, episodes, grid_size, goal_row, goal_col,
                   mean_reward, success_rate, steps_mean, created_at, seed,
                   reward_history, q_states_learned
            FROM experiments
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user["id"], limit),
        ).fetchall()

    items = []
    for row in rows:
        item = dict(row)
        item["reward_history"] = json.loads(item["reward_history"])
        items.append(item)
    return {"items": items, "count": len(items)}


@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str, user: dict[str, Any] = Depends(current_user)):
    with db_connect() as conn:
        row = conn.execute(
            """
            SELECT id, algorithm, episodes, grid_size, goal_row, goal_col,
                   mean_reward, success_rate, steps_mean, created_at, seed,
                   reward_history, q_states_learned
            FROM experiments
            WHERE id = ? AND user_id = ?
            """,
            (experiment_id, user["id"]),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    result = dict(row)
    result["reward_history"] = json.loads(result["reward_history"])
    return result


class DQNRequest(BaseModel):
    episodes: int = Field(default=100, ge=1, le=1000)
    grid_size: int = Field(default=5, ge=2, le=12)
    gamma: float = Field(default=0.95, ge=0, le=1)
    epsilon: float = Field(default=1.0, ge=0, le=1)
    epsilon_min: float = Field(default=0.05, ge=0, le=1)
    epsilon_decay: float = Field(default=0.99, gt=0, le=1)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    batch_size: int = Field(default=32, ge=4, le=256)
    max_steps: int = Field(default=80, ge=1, le=300)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


def train_dqn(req: DQNRequest, user_id: str) -> dict[str, Any]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="DQN is optional and unavailable: install tensorflow before requesting this route.",
        ) from exc

    tf.keras.utils.set_random_seed(req.seed)
    tf.random.set_seed(req.seed)
    rng = random.Random(req.seed)
    state_count = req.grid_size * req.grid_size
    goal = (req.grid_size - 1, req.grid_size - 1)

    def encode(position: tuple[int, int]) -> np.ndarray:
        out = np.zeros(state_count, dtype=np.float32)
        out[position[0] * req.grid_size + position[1]] = 1.0
        return out

    def move(position: tuple[int, int], action: int):
        row, col = position
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            col = min(req.grid_size - 1, col + 1)
        elif action == 2:
            row = min(req.grid_size - 1, row + 1)
        else:
            col = max(0, col - 1)
        nxt = (row, col)
        return nxt, (10.0 if nxt == goal else -1.0), nxt == goal

    def make_model():
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(state_count,)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(4, activation="linear"),
            ]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(req.learning_rate), loss="mse")
        return model

    model = make_model()
    target = make_model()
    target.set_weights(model.get_weights())
    replay: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
    rewards: list[float] = []
    steps_history: list[int] = []
    successes = 0
    epsilon = req.epsilon
    for episode in range(req.episodes):
        position = (0, 0)
        state = encode(position)
        total_reward = 0.0
        for step_no in range(1, req.max_steps + 1):
            if rng.random() < epsilon:
                action = rng.randrange(4)
            else:
                action = int(np.argmax(model(state[None, :], training=False).numpy()[0]))
            nxt, reward, done = move(position, action)
            next_state = encode(nxt)
            replay.append((state, action, reward, next_state, done))
            if len(replay) > 5000:
                replay.pop(0)
            if len(replay) >= req.batch_size:
                batch = rng.sample(replay, req.batch_size)
                states = np.stack([x[0] for x in batch])
                actions = np.array([x[1] for x in batch], dtype=np.int32)
                batch_rewards = np.array([x[2] for x in batch], dtype=np.float32)
                next_states = np.stack([x[3] for x in batch])
                dones = np.array([x[4] for x in batch], dtype=np.float32)
                targets = model(states, training=False).numpy()
                next_q = target(next_states, training=False).numpy().max(axis=1)
                targets[np.arange(req.batch_size), actions] = batch_rewards + req.gamma * next_q * (1.0 - dones)
                model.train_on_batch(states, targets)
            position = nxt
            state = next_state
            total_reward += reward
            if done:
                successes += 1
                break

        rewards.append(total_reward)
        steps_history.append(step_no)
        epsilon = max(req.epsilon_min, epsilon * req.epsilon_decay)
        if (episode + 1) % 10 == 0:
            target.set_weights(model.get_weights())

    created_at = utc_now().isoformat()
    experiment_id = str(uuid.uuid4())
    result = {
        "algorithm": "dqn",
        "episodes": req.episodes,
        "grid_size": req.grid_size,
        "goal": list(goal),
        "mean_reward": round(float(np.mean(rewards)), 4),
        "success_rate": round(successes / req.episodes, 4),
        "mean_steps": round(float(np.mean(steps_history)), 4),
        "q_states_learned": state_count,
        "reward_history": [round(x, 4) for x in rewards],
        "seed": req.seed,
        "experiment_id": experiment_id,
        "created_at": created_at,
    }

    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO experiments
            (id, user_id, algorithm, episodes, grid_size, goal_row, goal_col,
             mean_reward, success_rate, steps_mean, created_at, seed,
             reward_history, q_states_learned)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                user_id,
                "dqn",
                req.episodes,
                req.grid_size,
                goal[0],
                goal[1],
                result["mean_reward"],
                result["success_rate"],
                result["mean_steps"],
                created_at,
                req.seed,
                json.dumps(result["reward_history"]),
                state_count,
            ),
        )

    return result


@app.post("/experiments/dqn")
def run_dqn(req: DQNRequest, user: dict[str, Any] = Depends(current_user)):
    return train_dqn(req, user["id"])


class CompareRequest(BaseModel):
    experiment_ids: list[str] = Field(min_length=2, max_length=10)


def _owned_experiments(user_id: str, experiment_ids: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with db_connect() as conn:
        for experiment_id in dict.fromkeys(experiment_ids):
            row = conn.execute(
                """
                SELECT id, algorithm, episodes, grid_size, goal_row, goal_col,
                       mean_reward, success_rate, steps_mean, created_at, seed,
                       reward_history, q_states_learned
                FROM experiments
                WHERE id = ? AND user_id = ?
                """,
                (experiment_id, user_id),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Experiment not found")
            item = dict(row)
            item["reward_history"] = json.loads(item["reward_history"])
            items.append(item)
    if len(items) < 2:
        raise HTTPException(status_code=422, detail="Provide at least two distinct experiment IDs")
    return items


@app.post("/experiments/compare")
def compare_experiments(req: CompareRequest, user: dict[str, Any] = Depends(current_user)):
    items = _owned_experiments(user["id"], req.experiment_ids)
    metrics = ("mean_reward", "success_rate", "steps_mean", "episodes", "seed", "q_states_learned")
    return {
        "count": len(items),
        "experiments": [
            {
                "id": item["id"],
                "algorithm": item["algorithm"],
                "created_at": item["created_at"],
                **{metric: item[metric] for metric in metrics},
            }
            for item in items
        ],
        "best_mean_reward_id": max(items, key=lambda x: x["mean_reward"])["id"],
        "best_success_rate_id": max(items, key=lambda x: x["success_rate"])["id"],
    }


@app.get("/experiments/{experiment_id}/report.json")
def download_experiment_json(experiment_id: str, user: dict[str, Any] = Depends(current_user)):
    result = get_experiment(experiment_id, user)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    return StreamingResponse(
        io.BytesIO(payload.encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="intelliscore-{experiment_id}.json"'},
    )


@app.get("/experiments/{experiment_id}/report.csv")
def download_experiment_csv(experiment_id: str, user: dict[str, Any] = Depends(current_user)):
    result = get_experiment(experiment_id, user)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["metric", "value"])
    for key, value in result.items():
        writer.writerow([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])
    data = output.getvalue().encode("utf-8")
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="intelliscore-{experiment_id}.csv"'},
    )


@app.get("/stage2/status")
def stage2_status(user: dict[str, Any] = Depends(current_user)):
    return {
        "version": APP_VERSION,
        "q_learning": "available",
        "experiment_comparison": "available",
        "json_csv_reports": "available",
        "dqn": "optional_tensorflow_dependency_required",
        "background_workers": "not_implemented",
        "postgresql": "not_integrated",
        "billing": "not_implemented",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "app",
    "db_connect",
    "hash_password",
    "verify_password",
    "token_digest",
    "GridWorld",
    "QLearningRequest",
    "DQNRequest",
    "train_q_learning",
    "train_dqn",
    "current_user",
    "init_db",
]
