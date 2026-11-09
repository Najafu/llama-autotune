"""SQLite storage for benchmark history, tuned settings and launch profiles.

Uses the standard-library ``sqlite3`` module. The schema is versioned
with ``PRAGMA user_version`` and upgrades databases written by v0.3.0 in
place, so existing history is kept.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .models import BenchmarkEntry, LaunchProfile, SearchConfig

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY,
    hardware_id TEXT,
    model_id TEXT,
    config_json TEXT,
    prompt_tps REAL,
    generation_tps REAL,
    startup_time REAL,
    memory_usage REAL,
    success INTEGER,
    objective TEXT,
    timestamp TEXT
);
CREATE TABLE IF NOT EXISTS launch_profiles (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    args_json TEXT,
    model_path TEXT,
    hardware TEXT,
    created TEXT,
    score REAL
);
CREATE TABLE IF NOT EXISTS tuned_settings (
    id INTEGER PRIMARY KEY,
    hardware_id TEXT,
    model_id TEXT,
    llama_build INTEGER,
    config_json TEXT,
    generation_tps REAL,
    prompt_tps REAL,
    baseline_tps REAL,
    ctx INTEGER,
    efficiency REAL,
    bandwidth_gbs REAL,
    model_path TEXT,
    timestamp TEXT
);
"""

# Columns added after v0.3.0: (table, column, type).
_ADDED_COLUMNS = [
    ("benchmarks", "llama_build", "INTEGER DEFAULT 0"),
    ("benchmarks", "model_path", "TEXT DEFAULT ''"),
    ("benchmarks", "min_available_mb", "REAL DEFAULT 0"),
    ("benchmarks", "error", "TEXT DEFAULT ''"),
    ("launch_profiles", "llama_build", "INTEGER DEFAULT 0"),
]


def get_db_path() -> str:
    """Return the database path (``~/.llama-autotune/benchmarks.db``).

    The parent directory is created if it does not exist.
    """
    data_dir = os.path.join(Path.home(), ".llama-autotune")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "benchmarks.db")


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """Open the database, creating or upgrading the schema as needed.

    Args:
        db_path: Database file; defaults to :func:`get_db_path`.

    Returns:
        An open connection whose rows can be read by column name.
    """
    conn = sqlite3.connect(db_path or get_db_path())
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
        for table, column, ctype in _ADDED_COLUMNS:
            existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ctype}")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    return conn


get_session = connect


def save_benchmark(conn: sqlite3.Connection, entry: BenchmarkEntry) -> None:
    """Store one benchmark result."""
    conn.execute(
        """INSERT INTO benchmarks (hardware_id, model_id, config_json, prompt_tps, generation_tps,
               startup_time, memory_usage, success, objective, timestamp, llama_build, model_path,
               min_available_mb, error)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            entry.hardware_id, entry.model_id, entry.config.model_dump_json(exclude_none=True),
            entry.result.prompt_tps, entry.result.generation_tps, entry.result.startup_time,
            entry.result.memory_usage, int(entry.result.success), entry.objective.value,
            entry.timestamp or datetime.now(timezone.utc).isoformat(), entry.llama_build,
            entry.model_path, entry.result.min_available_mb, entry.result.error,
        ),
    )
    conn.commit()


def get_best_benchmark(
    conn: sqlite3.Connection,
    model_id: str,
    objective: str,
    hardware_id: str | None = None,
    llama_build: int | None = None,
) -> sqlite3.Row | None:
    """Return the fastest successful result for a model and objective.

    Args:
        conn: Open database connection.
        model_id: Model fingerprint (or legacy id).
        objective: Optimization objective value.
        hardware_id: Restrict to one machine when given.
        llama_build: Restrict to one llama.cpp build when given.

    Returns:
        The best row by generation speed, or None.
    """
    sql = "SELECT * FROM benchmarks WHERE model_id = ? AND objective = ? AND success = 1"
    params: list[object] = [model_id, objective]
    if hardware_id is not None:
        sql += " AND hardware_id = ?"
        params.append(hardware_id)
    if llama_build is not None:
        sql += " AND llama_build = ?"
        params.append(llama_build)
    return conn.execute(sql + " ORDER BY generation_tps DESC LIMIT 1", params).fetchone()


def save_launch_profile(conn: sqlite3.Connection, profile: LaunchProfile) -> None:
    """Insert or update a launch profile by name."""
    conn.execute(
        """INSERT INTO launch_profiles (name, args_json, model_path, hardware, created, score, llama_build)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET args_json = excluded.args_json,
               model_path = excluded.model_path, hardware = excluded.hardware,
               score = excluded.score, llama_build = excluded.llama_build""",
        (profile.name, json.dumps(profile.args), profile.model_path, profile.hardware,
         profile.created, profile.score, profile.llama_build),
    )
    conn.commit()


def get_launch_profile(conn: sqlite3.Connection, name: str) -> LaunchProfile | None:
    """Load a launch profile by name, or None if it does not exist."""
    row = conn.execute("SELECT * FROM launch_profiles WHERE name = ?", (name,)).fetchone()
    if row is None:
        return None
    return LaunchProfile(
        name=row["name"], args=json.loads(row["args_json"] or "[]"), model_path=row["model_path"] or "",
        hardware=row["hardware"] or "", created=row["created"] or "", score=row["score"] or 0.0,
        llama_build=row["llama_build"] or 0,
    )


@dataclass
class TunedSettings:
    """Settings found by ``tune`` for one model on one machine."""

    config: SearchConfig
    llama_build: int = 0
    generation_tps: float = 0.0
    prompt_tps: float = 0.0
    baseline_tps: float = 0.0
    ctx: int = 0
    efficiency: float = 0.0
    bandwidth_gbs: float = 0.0
    model_path: str = ""
    timestamp: str = ""


def save_tuned(conn: sqlite3.Connection, hardware_id: str, model_id: str, tuned: TunedSettings) -> None:
    """Store the result of a tuning run; the newest one per machine and model is used."""
    conn.execute(
        """INSERT INTO tuned_settings (hardware_id, model_id, llama_build, config_json, generation_tps,
               prompt_tps, baseline_tps, ctx, efficiency, bandwidth_gbs, model_path, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (hardware_id, model_id, tuned.llama_build, tuned.config.model_dump_json(exclude_none=True),
         tuned.generation_tps, tuned.prompt_tps, tuned.baseline_tps, tuned.ctx, tuned.efficiency,
         tuned.bandwidth_gbs, tuned.model_path, tuned.timestamp or datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def best_bandwidth(conn: sqlite3.Connection, hardware_id: str) -> float:
    """Highest memory bandwidth (GB/s) ever measured on this machine, or 0.

    Other programs can only make the copy test slower, never faster, so the
    best result is the machine's real bandwidth.
    """
    row = conn.execute("SELECT MAX(bandwidth_gbs) FROM tuned_settings WHERE hardware_id = ?",
                       (hardware_id,)).fetchone()
    return float(row[0] or 0.0)


def get_tuned(conn: sqlite3.Connection, hardware_id: str, model_id: str) -> TunedSettings | None:
    """Return the newest tuning result for a machine and model, or None."""
    row = conn.execute(
        "SELECT * FROM tuned_settings WHERE hardware_id = ? AND model_id = ? ORDER BY id DESC LIMIT 1",
        (hardware_id, model_id),
    ).fetchone()
    if row is None:
        return None
    return TunedSettings(
        config=SearchConfig.model_validate_json(row["config_json"] or "{}"), llama_build=row["llama_build"] or 0,
        generation_tps=row["generation_tps"] or 0.0, prompt_tps=row["prompt_tps"] or 0.0,
        baseline_tps=row["baseline_tps"] or 0.0, ctx=row["ctx"] or 0, efficiency=row["efficiency"] or 0.0,
        bandwidth_gbs=row["bandwidth_gbs"] or 0.0, model_path=row["model_path"] or "",
        timestamp=row["timestamp"] or "",
    )
