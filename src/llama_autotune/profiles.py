from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .models import LaunchProfile


def get_profiles_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "profiles")


def export_profile(profile: LaunchProfile, path: str | None = None) -> str:
    if path is None:
        profiles_dir = get_profiles_dir()
        os.makedirs(profiles_dir, exist_ok=True)
        path = os.path.join(profiles_dir, f"{profile.name}.json")

    with open(path, "w") as f:
        json.dump(profile.model_dump(), f, indent=2)
    return path


def import_profile(path: str) -> LaunchProfile:
    with open(path) as f:
        data = json.load(f)
    return LaunchProfile(**data)


def create_profile(
    name: str,
    args: list[str],
    model_path: str,
    hardware: str,
    score: float = 0.0,
) -> LaunchProfile:
    return LaunchProfile(
        name=name,
        args=args,
        model_path=model_path,
        hardware=hardware,
        created=datetime.now(timezone.utc).isoformat(),
        score=score,
    )
