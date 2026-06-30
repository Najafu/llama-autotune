"""SQLAlchemy ORM models and persistence helpers for llama-autotune.

Defines the database schema (``HardwareProfileModel``, ``ModelProfileModel``,
``BenchmarkModel``, ``LaunchProfileModel``) and convenience functions for
storing and retrieving benchmark results and launch profiles.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Column, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

from .models import BenchmarkEntry, BenchmarkResult, LaunchProfile, SearchConfig


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""
    pass


class HardwareProfileModel(Base):
    """Persisted hardware profile entry.

    Stores CPU/GPU topology, memory sizes, and backend type for a given
    machine. Corresponds to the ``hardware_profiles`` table.
    """
    __tablename__ = "hardware_profiles"
    id = Column(Integer, primary_key=True)
    cpu_name = Column(String)
    physical_cores = Column(Integer)
    logical_cores = Column(Integer)
    ram_gb = Column(Float)
    gpu_count = Column(Integer)
    gpu_vendor = Column(String)
    gpu_models = Column(Text)
    vram_per_gpu = Column(Text)
    backend = Column(String)


class ModelProfileModel(Base):
    """Persisted model profile entry.

    Stores model architecture metadata such as parameter count,
    quantization, layer/head counts, and file size. Corresponds to the
    ``model_profiles`` table.
    """
    __tablename__ = "model_profiles"
    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True)
    architecture = Column(String)
    parameters = Column(Integer)
    quantization = Column(String)
    n_layers = Column(Integer)
    n_heads = Column(Integer)
    training_context = Column(Integer)
    is_moe = Column(Integer)
    active_parameters = Column(Integer)
    file_size_gb = Column(Float)


class BenchmarkModel(Base):
    """Persisted benchmark result entry.

    Records a single benchmark run including throughput metrics, memory
    usage, the configuration JSON, and the optimisation objective.
    Corresponds to the ``benchmarks`` table.
    """
    __tablename__ = "benchmarks"
    id = Column(Integer, primary_key=True)
    hardware_id = Column(String)
    model_id = Column(String)
    config_json = Column(Text)
    prompt_tps = Column(Float)
    generation_tps = Column(Float)
    startup_time = Column(Float)
    memory_usage = Column(Float)
    success = Column(Integer)
    objective = Column(String)
    timestamp = Column(String)


class LaunchProfileModel(Base):
    """Persisted launch-profile entry.

    Stores a named profile with its CLI arguments, model path, hardware
    description, and score. Corresponds to the ``launch_profiles`` table.
    """
    __tablename__ = "launch_profiles"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    args_json = Column(Text)
    model_path = Column(String)
    hardware = Column(String)
    created = Column(String)
    score = Column(Float)


def get_db_path() -> str:
    """Return the default path for the SQLite benchmark database.

    The database is stored at ``~/.llama-autotune/benchmarks.db``. The
    parent directory is created if it does not exist.

    Returns:
        Absolute path to the database file.
    """
    data_dir = os.path.join(Path.home(), ".llama-autotune")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "benchmarks.db")


def get_session(db_path: str | None = None) -> Session:
    """Create and return a new SQLAlchemy session.

    Ensures all tables exist before returning the session.

    Args:
        db_path: Path to the SQLite database. Defaults to ``get_db_path()``.

    Returns:
        An open SQLAlchemy ``Session``.
    """
    if db_path is None:
        db_path = get_db_path()
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return Session(engine)


def save_benchmark(session: Session, entry: BenchmarkEntry) -> None:
    """Persist a benchmark entry to the database.

    Args:
        session: An active SQLAlchemy session.
        entry: The benchmark entry to save.
    """
    bm = BenchmarkModel(
        hardware_id=entry.hardware_id,
        model_id=entry.model_id,
        config_json=entry.config.model_dump_json(),
        prompt_tps=entry.result.prompt_tps,
        generation_tps=entry.result.generation_tps,
        startup_time=entry.result.startup_time,
        memory_usage=entry.result.memory_usage,
        success=1 if entry.result.success else 0,
        objective=entry.objective.value,
        timestamp=entry.timestamp or datetime.now(timezone.utc).isoformat(),
    )
    session.add(bm)
    session.commit()


def save_launch_profile(session: Session, profile: LaunchProfile) -> None:
    """Upsert a launch profile into the database.

    If a profile with the same name already exists its fields are updated;
    otherwise a new row is inserted.

    Args:
        session: An active SQLAlchemy session.
        profile: The launch profile to persist.
    """
    existing = (
        session.query(LaunchProfileModel)
        .filter(LaunchProfileModel.name == profile.name)
        .first()
    )
    if existing:
        existing.args_json = json.dumps(profile.args)
        existing.model_path = profile.model_path
        existing.hardware = profile.hardware
        existing.score = profile.score
    else:
        model = LaunchProfileModel(
            name=profile.name,
            args_json=json.dumps(profile.args),
            model_path=profile.model_path,
            hardware=profile.hardware,
            created=profile.created,
            score=profile.score,
        )
        session.add(model)
    session.commit()


def get_best_benchmark(
    session: Session, model_id: str, objective: str
) -> BenchmarkModel | None:
    """Retrieve the highest-scoring successful benchmark for a model/objective pair.

    Results are ordered by generation tokens-per-second descending.

    Args:
        session: An active SQLAlchemy session.
        model_id: The model identifier to match.
        objective: The optimisation objective to match.

    Returns:
        The best ``BenchmarkModel`` row, or ``None`` if no matching entry exists.
    """
    col = BenchmarkModel.generation_tps
    return (
        session.query(BenchmarkModel)
        .filter(
            BenchmarkModel.model_id == model_id,
            BenchmarkModel.objective == objective,
            BenchmarkModel.success == 1,
        )
        .order_by(col.desc())
        .first()
    )
