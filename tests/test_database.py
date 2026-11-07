import sqlite3

from llama_autotune.database import (
    SCHEMA_VERSION,
    connect,
    get_best_benchmark,
    get_launch_profile,
    save_benchmark,
    save_launch_profile,
)
from llama_autotune.models import (
    BenchmarkEntry,
    BenchmarkResult,
    LaunchProfile,
    OptimizeObjective,
    SearchConfig,
)


def _entry(gen: float, build: int = 11347, hw: str = "hw1") -> BenchmarkEntry:
    return BenchmarkEntry(
        hardware_id=hw,
        model_id="model1",
        config=SearchConfig(threads=3, batch_size=2048),
        result=BenchmarkResult(prompt_tps=60.0, generation_tps=gen, success=True),
        objective=OptimizeObjective.BALANCED,
        llama_build=build,
    )


def test_save_and_get_best_benchmark(tmp_path):
    conn = connect(str(tmp_path / "db.sqlite"))
    for gen in (10.0, 12.5, 11.0):
        save_benchmark(conn, _entry(gen))
    best = get_best_benchmark(conn, "model1", "balanced")
    assert best["generation_tps"] == 12.5
    assert best["llama_build"] == 11347
    conn.close()


def test_best_benchmark_filters_by_build_and_hardware(tmp_path):
    conn = connect(str(tmp_path / "db.sqlite"))
    save_benchmark(conn, _entry(20.0, build=100))
    save_benchmark(conn, _entry(12.0, build=11347))
    save_benchmark(conn, _entry(30.0, build=11347, hw="other"))
    best = get_best_benchmark(conn, "model1", "balanced", hardware_id="hw1", llama_build=11347)
    assert best["generation_tps"] == 12.0
    conn.close()


def test_failed_benchmarks_are_not_best(tmp_path):
    conn = connect(str(tmp_path / "db.sqlite"))
    entry = _entry(99.0)
    entry.result.success = False
    entry.result.error = "memory_guard"
    save_benchmark(conn, entry)
    assert get_best_benchmark(conn, "model1", "balanced") is None
    conn.close()


def test_launch_profile_roundtrip_and_upsert(tmp_path):
    conn = connect(str(tmp_path / "db.sqlite"))
    profile = LaunchProfile(name="p", args=["-t", "3"], model_path="m.gguf", hardware="i3",
                            score=12.0, llama_build=11347)
    save_launch_profile(conn, profile)
    profile.args = ["-t", "4"]
    save_launch_profile(conn, profile)
    loaded = get_launch_profile(conn, "p")
    assert loaded.args == ["-t", "4"]
    assert loaded.llama_build == 11347
    assert get_launch_profile(conn, "missing") is None
    conn.close()


def test_upgrades_v03_database_in_place(tmp_path):
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE benchmarks (id INTEGER PRIMARY KEY, hardware_id TEXT, model_id TEXT, config_json TEXT,"
        " prompt_tps REAL, generation_tps REAL, startup_time REAL, memory_usage REAL, success INTEGER,"
        " objective TEXT, timestamp TEXT);"
        "INSERT INTO benchmarks (model_id, objective, generation_tps, success) VALUES ('model1', 'balanced', 7.5, 1);"
    )
    old.commit()
    old.close()

    conn = connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    best = get_best_benchmark(conn, "model1", "balanced")
    assert best["generation_tps"] == 7.5
    assert best["llama_build"] == 0
    conn.close()
