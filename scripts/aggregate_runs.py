#!/usr/bin/env python3
"""跨运行汇总：扫描多个实验输出目录，生成 runs/summary/failures 三个 CSV。

P1-10 的最小闭环：多种子/多配置实验结束后，用本脚本把逐 run 的
``manifest.json`` + ``metrics.json`` 合并为可比较的批量结果索引，
并对标量指标按 (experiment, variant, mode, model, tuner) 分组计算 mean±std。
失败运行进入 failures.csv，归档重放（archive_replay）不参与均值统计。

用法:
    # 汇总某个父目录下的所有 run 子目录
    python scripts/aggregate_runs.py --runs-dir outputs/ --output outputs/_summary

    # 显式指定若干运行目录（可重复）
    python scripts/aggregate_runs.py --run outputs/run_a --run outputs/run_b
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _canonical_config(manifest: dict) -> dict:
    """把 manifest 的 config 归一化为实验指纹输入，剔除运行特定/允许变化的字段。

    剔除 ``experiment.output_dir``（运行路径）与 ``experiment.seed``（组内允许
    变化的种子）；保留 dataset、model.params、preprocess、validation、tuning、
    label_refinement、knowledge、predicates 等实验定义字段（P1-06）。
    """
    config = manifest.get("config")
    if not isinstance(config, dict):
        return {}
    cfg = copy.deepcopy(config)
    experiment = cfg.get("experiment")
    if isinstance(experiment, dict):
        experiment.pop("output_dir", None)
        experiment.pop("seed", None)
    return cfg


def _fingerprint(manifest: dict) -> str:
    """生成实验指纹（不含 run_id/输出路径/时间/种子）。"""
    payload = json.dumps(_canonical_config(manifest), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _scalar_metrics(metrics: dict) -> dict[str, float]:
    """仅保留数值型标量指标，排除字符串/列表/嵌套 dict（如 confusion_matrix、independent_cv）。"""
    result = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = float(value)
    return result


def _discover_run_dirs(runs_dir: Path | None, run_dirs: list[str]) -> list[Path]:
    """按规范路径去重：同一目录同时通过父目录扫描与 --run 提供时只计一次。"""
    dirs: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            dirs.append(path)

    if runs_dir is not None:
        root = Path(runs_dir)
        for p in sorted(root.iterdir()):
            if (p / "manifest.json").is_file():
                add(p)
    for p in run_dirs:
        add(Path(p))
    return dirs


def _collect(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for directory in run_dirs:
        manifest = _load_json(directory / "manifest.json")
        if manifest is None:
            continue
        metrics = _load_json(directory / "metrics.json") or {}
        components = manifest.get("components") or {}
        scalars = _scalar_metrics(metrics)
        primary_metric = components.get("primary_metric") or metrics.get("primary_metric")
        row = {
            "run_id": manifest.get("run_id"),
            "output_dir": str(directory),
            "experiment": manifest.get("experiment"),
            "variant": manifest.get("variant", "baseline"),
            "execution_mode": manifest.get("execution_mode"),
            "model": components.get("model"),
            "tuner": components.get("tuner"),
            "seed": manifest.get("seed"),
            "status": manifest.get("status"),
            "primary_metric": primary_metric,
            "primary_value": scalars.get(primary_metric) if primary_metric else None,
            "fingerprint": _fingerprint(manifest),
        }
        row.update(scalars)
        failure = manifest.get("failure")
        if failure:
            row["failure_type"] = failure.get("exception_type")
            row["failure_message"] = failure.get("message")
        rows.append(row)
    return rows


def _summarize(rows: list[dict]) -> pd.DataFrame:
    completed = [
        row for row in rows
        if row.get("status") == "completed" and row.get("execution_mode") != "archive_replay"
    ]
    if not completed:
        return pd.DataFrame()

    frame = pd.DataFrame(completed)
    # 分组键加入实验指纹与 primary_metric：不同数据/标签/特征/划分/模型参数/
    # 评分协议的同名运行不再混组；仅允许 seed 变化（指纹已剔除 seed）在同一组。
    group_keys = ["experiment", "variant", "execution_mode", "model", "tuner",
                  "primary_metric", "fingerprint"]
    # 每个分组内至少 2 个值才统计 mean±std。
    numeric_cols = [
        col for col in frame.columns
        if col not in group_keys + ["run_id", "output_dir", "status",
                                    "failure_type", "failure_message", "seed"]
    ]
    summaries = []
    for keys, group in frame.groupby(group_keys, dropna=False):
        base = dict(zip(group_keys, keys))
        base["n_runs"] = int(len(group))
        base["n_seeds"] = int(group["seed"].nunique())
        for col in numeric_cols:
            series = group[col].dropna()
            if len(series) >= 2:
                base[f"{col}_mean"] = float(series.mean())
                base[f"{col}_std"] = float(series.std(ddof=1))
            elif len(series) == 1:
                base[f"{col}_mean"] = float(series.iloc[0])
                base[f"{col}_std"] = None
        summaries.append(base)
    return pd.DataFrame(summaries)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", help="父目录，扫描其下含 manifest.json 的子目录")
    parser.add_argument("--run", action="append", default=[], help="显式运行目录（可重复）")
    parser.add_argument("--output", default=".", help="汇总 CSV 输出目录")
    args = parser.parse_args()

    if args.runs_dir is None and not args.run:
        parser.error("provide --runs-dir or at least one --run")

    run_dirs = _discover_run_dirs(args.runs_dir, args.run)
    if not run_dirs:
        print("No run directories found.", file=sys.stderr)
        return 1

    rows = _collect(run_dirs)
    if not rows:
        print("No manifest.json read successfully.", file=sys.stderr)
        return 1

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    runs = pd.DataFrame(rows)
    summary = _summarize(rows)
    failures = runs[runs.get("status") == "failed"] if "status" in runs else pd.DataFrame()

    runs.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    if not failures.empty:
        failures.to_csv(out / "failures.csv", index=False)

    print(f"runs:      {len(runs)} -> {out / 'runs.csv'}")
    print(f"summary:   {len(summary)} groups -> {out / 'summary.csv'}")
    print(f"failures:  {len(failures)} -> {out / 'failures.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
