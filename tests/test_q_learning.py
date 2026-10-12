from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "intelliscore_test.db"
    monkeypatch.setenv("DB_PATH", str(sqlite_path))
    import main

    importlib.reload(main)
    return main


def test_q_learning_result_shape(app_module):
    req = app_module.QLearningRequest(episodes=20, grid_size=5, alpha=0.25, gamma=0.9, epsilon=0.2, max_steps=30, seed=11)
    result = app_module.train_q_learning(req, "user-test")

    assert result["algorithm"] == "q_learning"
    assert result["episodes"] == req.episodes
    assert result["grid_size"] == req.grid_size
    assert len(result["reward_history"]) == req.episodes
    assert result["q_states_learned"] > 0
    assert 0.0 <= result["success_rate"] <= 1.0


def test_q_learning_repeated_seed_is_consistent(app_module):
    req = app_module.QLearningRequest(episodes=18, grid_size=4, alpha=0.15, gamma=0.95, epsilon=0.1, max_steps=25, seed=123)
    first = app_module.train_q_learning(req, "user-a")
    second = app_module.train_q_learning(req, "user-a")

    assert first["reward_history"] == second["reward_history"]
    assert first["mean_steps"] == second["mean_steps"]
