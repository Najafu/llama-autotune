"""Launch-profile import/export utilities.

Provides functions to read and write ``LaunchProfile`` objects as JSON
files in a dedicated profiles directory, as well as a factory function
for creating new profiles.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .models import LaunchProfile


def get_profiles_dir() -> str:
    """Return the absolute path to the profiles storage directory.

    The directory is located at ``~/.llama-autotune/profiles/``.

    Returns:
        Absolute directory path as a string.
    """
    return os.path.join(Path.home(), ".llama-autotune", "profiles")


def export_profile(profile: LaunchProfile, path: str | None = None) -> str:
    """Write a launch profile to a JSON file.

    If *path* is ``None`` the profile is written to
    ``<profiles_dir>/<name>.json``.

    Args:
        profile: The profile to export.
        path: Optional explicit file path.

    Returns:
        The path that the profile was written to.
    """
    if path is None:
        profiles_dir = get_profiles_dir()
        os.makedirs(profiles_dir, exist_ok=True)
        path = os.path.join(profiles_dir, f"{profile.name}.json")

    with open(path, "w") as f:
        json.dump(profile.model_dump(), f, indent=2)
    return path


def import_profile(path: str) -> LaunchProfile:
    """Load a launch profile from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The deserialised ``LaunchProfile``.
    """
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
    """Create a new ``LaunchProfile`` with the current timestamp.

    Args:
        name: Human-readable profile name.
        args: CLI argument list for llama.cpp.
        model_path: Path to the model file.
        hardware: Human-readable hardware description.
        score: Optional quality score (default 0.0).

    Returns:
        A new ``LaunchProfile`` instance.
    """
    return LaunchProfile(
        name=name,
        args=args,
        model_path=model_path,
        hardware=hardware,
        created=datetime.now(timezone.utc).isoformat(),
        score=score,
    )
