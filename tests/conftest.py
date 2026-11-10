"""Shared test setup."""

import pytest

import llama_autotune.compat as compat
import llama_autotune.database as database
import llama_autotune.memory as memory


@pytest.fixture(autouse=True)
def _private_storage(monkeypatch, tmp_path):
    """Keep every test away from the user's real ~/.llama-autotune database and cache."""
    monkeypatch.setattr(database, "get_db_path", lambda: str(tmp_path / "benchmarks.db"))
    monkeypatch.setattr(compat, "cache_dir", lambda: tmp_path / "cache")
    memory._compute_at_4096.cache_clear()
