"""跨运行汇总脚本的端到端测试（P1-10）。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _write_run(
    root: Path, name: str, *, seed: int, status: str, f1: float,
    mode: str = "train_from_archive_features", primary_metric: str = "f1", config: dict | None = None,
):
    directory = root / name
    (directory / "predictions").mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": name,
        "status": status,
        "experiment": "rf_baseline",
        "variant": "baseline",
        "seed": seed,
        "execution_mode": mode,
        "components": {
            "model": "rf",
            "tuner": "none",
            "primary_metric": primary_metric,
        },
        "config": config if config is not None else {
            "experiment": {"name": "rf_baseline", "seed": seed, "output_dir": str(directory)},
            "model": {"name": "rf", "params": {"n_estimators": 10}},
            "validation": {"primary_metric": primary_metric, "metrics": ["f1", "roc_auc"]},
            "dataset": {"archive_dir": "/data/archive"},
        },
        "failure": {"exception_type": "ValueError", "message": "boom"} if status == "failed" else None,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "metrics.json").write_text(
        json.dumps({"mode": mode, "primary_metric": primary_metric, "f1": f1, "roc_auc": f1 + 0.1, "accuracy": f1 - 0.1}),
        encoding="utf-8",
    )


def test_aggregate_runs_writes_runs_summary_and_failures(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_run(runs_dir, "run_a", seed=1, status="completed", f1=0.8)
    _write_run(runs_dir, "run_b", seed=2, status="completed", f1=0.6)
    _write_run(runs_dir, "run_c", seed=3, status="failed", f1=0.0)
    _write_run(runs_dir, "run_replay", seed=4, status="completed", f1=0.9, mode="archive_replay")

    out = tmp_path / "summary"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "aggregate_runs.py"),
         "--runs-dir", str(runs_dir), "--output", str(out)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    runs = pd.read_csv(out / "runs.csv")
    assert len(runs) == 4

    summary = pd.read_csv(out / "summary.csv")
    # 排除 failed 与 archive_replay：仅 run_a/run_b 参与均值。
    assert len(summary) == 1
    assert summary.iloc[0]["n_runs"] == 2
    assert abs(summary.iloc[0]["f1_mean"] - 0.7) < 1e-9
    assert abs(summary.iloc[0]["f1_std"] - (0.8 - 0.6) / (2 ** 0.5)) < 1e-9

    failures = pd.read_csv(out / "failures.csv")
    assert len(failures) == 1
    assert failures.iloc[0]["run_id"] == "run_c"
    assert failures.iloc[0]["failure_type"] == "ValueError"


def test_aggregate_does_not_mix_different_primary_metrics(tmp_path):
    """P1-06: 同名但 primary_metric 不同的运行不得混算为同一均值。"""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_run(runs_dir, "run_a", seed=1, status="completed", f1=0.2, primary_metric="f1")
    _write_run(runs_dir, "run_b", seed=2, status="completed", f1=0.9, primary_metric="roc_auc")

    out = tmp_path / "summary"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "aggregate_runs.py"),
         "--runs-dir", str(runs_dir), "--output", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    summary = pd.read_csv(out / "summary.csv")
    # 两个不同 primary_metric 的运行应拆分为两组，而非混成 mean=0.55。
    assert len(summary) == 2
    assert set(summary["primary_metric"]) == {"f1", "roc_auc"}


def test_aggregate_dedupes_same_directory(tmp_path):
    """P1-06: 同一目录经父目录扫描与 --run 同时提供时只计一次。"""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_run(runs_dir, "run_a", seed=1, status="completed", f1=0.8)

    out = tmp_path / "summary"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "aggregate_runs.py"),
         "--runs-dir", str(runs_dir), "--run", str(runs_dir / "run_a"),
         "--output", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    runs = pd.read_csv(out / "runs.csv")
    assert len(runs) == 1


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        from pathlib import Path

        test_aggregate_runs_writes_runs_summary_and_failures(Path(temporary))
    print("聚合脚本测试通过")
