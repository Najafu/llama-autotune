"""Shared test setup."""

import pytest

import llama_autotune.database as database


@pytest.fixture(autouse=True)
def _private_database(monkeypatch, tmp_path):
    """Keep every test away from the user's real ~/.llama-autotune database."""
    monkeypatch.setattr(database, "get_db_path", lambda: str(tmp_path / "benchmarks.db"))
