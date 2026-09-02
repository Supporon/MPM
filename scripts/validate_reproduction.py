#!/usr/bin/env python3
"""
复现验证脚本：在 MPM_codex_phase2 框架中复现 EarthByte-MPM_Lachlan_Porphyry 的实验结果。

该脚本系统地验证 phase2 框架对原始 EarthByte 实验结果的复现能力，
覆盖三种执行模式（archive_replay / train_from_archive_features / raw_gis）
以及所有已注册的实验配置。

用法:
    # 完整验证（推荐）
    python scripts/validate_reproduction.py --all

    # 仅验证配置
    python scripts/validate_reproduction.py --validate-only

    # 仅运行 archive_replay 实验
    python scripts/validate_reproduction.py --mode archive_replay

    # 仅运行 train_from_archive_features 实验
    python scripts/validate_reproduction.py --mode train_from_archive_features

    # 运行单个实验
    python scripts/validate_reproduction.py --experiment lachlan_rf_baseline

    # 生成 HTML 报告
    python scripts/validate_reproduction.py --all --report report.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bootstrap import load_builtin_components
from src.core.config import ConfigError, load_config
from src.core.experiment import Experiment

# ---------------------------------------------------------------------------
# 实验定义
# ---------------------------------------------------------------------------

# 所有已注册实验配置，按执行模式和预期结果分组
EXPERIMENT_MANIFEST = {
    "archive_replay": [
        {
            "name": "lachlan_rf_baseline",
            "config": "configs/experiments/lachlan_rf_baseline.yaml",
            "description": "Lachlan RF 基线 — 归档回放（含 PUB 标签细化 + 贝叶斯调参）",
            "expected": {
                "feature_count": 138,
                "train_rows": 359,
                "test_rows": 120,
                "model_file": "model_rf.pkl",
                "prediction_files": ["target_probs.csv", "probability_map.tif"],
            },
        },
        {
            "name": "nsw_rf_baseline",
            "config": "configs/experiments/nsw_rf_baseline.yaml",
            "description": "NSW RF 基线 — 归档回放",
            "expected": {
                "model_file": "model_rf.pkl",
                "prediction_files": ["target_probs.csv", "probability_map.tif"],
            },
        },
    ],
    "train_from_archive_features": [
        {
            "name": "lachlan_rf_train",
            "config": "configs/experiments/lachlan_rf_baseline.yaml",
            "description": "Lachlan RF — 从归档特征重新训练（无 PUB，无调参）",
            "overrides": {
                "experiment": {
                    "name": "lachlan_rf_train",
                    "output_dir": "outputs/lachlan_rf_train_validation",
                    "execution_mode": "train_from_archive_features",
                },
                "tuning": {"name": "none", "params": {}},
                "label_refinement": {"enabled": False},
                "knowledge": {"enabled": False},
                "predicates": {"enabled": False},
                "prediction": {"export_geotiff": False},
            },
            "expected": {
                "model_file": "model_rf.pkl",
                "prediction_files": ["target_probs.csv"],
            },
        },
        {
            "name": "lachlan_rf_simple_train",
            "config": "configs/experiments/lachlan_rf_phase2.yaml",
            "description": "Lachlan RF Phase2 配置 — 从归档特征重新训练（无 PUB，无调参）",
            "overrides": {
                "experiment": {
                    "name": "lachlan_rf_simple_train",
                    "output_dir": "outputs/lachlan_rf_simple_train_validation",
                    "execution_mode": "train_from_archive_features",
                },
                "tuning": {"name": "none", "params": {}},
                "label_refinement": {"enabled": False},
                "knowledge": {"enabled": False},
                "predicates": {"enabled": False},
                "prediction": {"export_geotiff": False},
            },
            "expected": {
                "model_file": "model_rf.pkl",
                "prediction_files": ["target_probs.csv"],
            },
        },
    ],
    # raw_gis 模式需要完整 GIS 数据和 geopandas/rasterio 等依赖，
    # 默认跳过，仅当 --include-raw-gis 时尝试运行
    "raw_gis": [
        {
            "name": "lachlan_rf_raw_gis",
            "config": "configs/experiments/lachlan_rf_raw_gis.yaml",
            "description": "Lachlan RF — 从原始 GIS 数据构建特征并训练",
            "overrides": {
                "experiment": {
                    "output_dir": "outputs/lachlan_rf_raw_gis_validation",
                },
            },
            "expected": {
                "model_file": "model_rf.pkl",
                "prediction_files": ["target_probs.csv"],
            },
            "requires_gis": True,
        },
    ],
}

# EarthByte 原始归档中已知的参考指标（来自 model_comparison_results.csv）
# 用于 train_from_archive_features 模式的指标对比
EARTHBYTE_REFERENCE_METRICS = {
    "RF (Pre-PUB)": {
        "Accuracy": 0.9770,
        "Precision": 1.0,
        "Recall": 0.875,
        "F1-Score": 0.9333,
        "ROC_AUC": 0.9761,
    },
    "RF (Post-PUB)": {
        "Accuracy": 0.9770,
        "Precision": 1.0,
        "Recall": 0.875,
        "F1-Score": 0.9333,
        "ROC_AUC": 0.9846,
    },
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """单个实验的验证结果。"""

    experiment_name: str
    mode: str
    config_path: str
    status: str = "pending"  # "passed", "failed", "skipped", "error"
    duration_seconds: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, bool] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reference_comparison: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """计算文件的 SHA-256 摘要。"""
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典。"""
    result = {}
    for key in set(base) | set(override):
        if key in override and key in base and isinstance(base[key], dict) and isinstance(override[key], dict):
            result[key] = deep_merge(base[key], override[key])
        elif key in override:
            result[key] = override[key]
        else:
            result[key] = base[key]
    return result


# ---------------------------------------------------------------------------
# 验证步骤
# ---------------------------------------------------------------------------

def step_validate_configs() -> dict[str, Any]:
    """验证所有实验配置文件是否可正确加载。"""
    results = {"total": 0, "passed": 0, "failed": 0, "details": []}
    all_configs: list[tuple[str, str]] = []
    for mode, experiments in EXPERIMENT_MANIFEST.items():
        for exp in experiments:
            config_path = ROOT / exp["config"]
            if not config_path.is_file():
                continue
            all_configs.append((exp["name"], str(config_path)))

    # 去重
    seen = set()
    unique: list[tuple[str, str]] = []
    for name, path in all_configs:
        if path not in seen:
            seen.add(path)
            unique.append((name, path))

    for name, config_path in unique:
        results["total"] += 1
        try:
            load_builtin_components()
            config = load_config(config_path)
            spec = config.spec
            detail = {
                "name": name,
                "config": config_path,
                "experiment": spec.name,
                "execution_mode": spec.execution_mode,
                "task": spec.task.name,
                "model": spec.model.name,
                "tuner": spec.tuner.name,
                "feature_operators": [op.name for op in spec.feature_operators],
            }
            results["passed"] += 1
            results["details"].append({"status": "passed", **detail})
        except (ConfigError, FileNotFoundError, Exception) as exc:
            results["failed"] += 1
            results["details"].append(
                {"status": "failed", "name": name, "config": config_path, "error": str(exc)}
            )

    return results


def step_run_experiment(
    exp_def: dict,
    mode: str,
    *,
    include_raw_gis: bool = False,
) -> ValidationResult:
    """运行单个实验并验证结果。"""
    exp_name = exp_def["name"]
    config_path = ROOT / exp_def["config"]

    if exp_def.get("requires_gis") and not include_raw_gis:
        return ValidationResult(
            experiment_name=exp_name,
            mode=mode,
            status="skipped",
            config_path=str(config_path),
            warnings=["raw_gis 模式需要 --include-raw-gis 标志"],
        )

    result = ValidationResult(
        experiment_name=exp_name,
        mode=mode,
        config_path=str(config_path),
    )

    try:
        # 加载配置
        load_builtin_components()
        config = load_config(config_path)

        # 应用覆盖
        overrides = exp_def.get("overrides", {})
        if overrides:
            import copy
            values = deep_merge(copy.deepcopy(config.values), copy.deepcopy(overrides))
            from src.core.config import ExperimentConfig, validate_config
            # 确保 prediction.export_geotiff 在非 raw_gis 模式下设为 false（避免 GDAL 依赖）
            if mode != "raw_gis":
                values.setdefault("prediction", {})
                values["prediction"]["export_geotiff"] = False
            validate_config(values)
            config = ExperimentConfig(values=values, source_path=config.source_path)

        t_start = time.monotonic()

        # 运行实验
        experiment = Experiment(config)
        manifest = experiment.run()

        result.duration_seconds = time.monotonic() - t_start
        result.manifest = manifest

        # 验证产物
        output_dir = experiment.output_dir
        expected = exp_def.get("expected", {})

        # 检查模型文件
        if "model_file" in expected:
            model_path = output_dir / "models" / expected["model_file"]
            result.artifacts["model"] = model_path.is_file()

        # 检查预测文件
        for pred_file in expected.get("prediction_files", []):
            pred_path = output_dir / "predictions" / pred_file
            result.artifacts[pred_file] = pred_path.is_file()

        # 检查必需文件
        for required in ["config_resolved.yaml", "manifest.json", "metrics.json"]:
            path = output_dir / required
            result.artifacts[required] = path.is_file()

        # 读取指标
        metrics_path = output_dir / "metrics.json"
        if metrics_path.is_file():
            result.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        # 检查特征数量
        if "feature_count" in expected:
            actual_count = manifest.get("feature_count", 0)
            if actual_count != expected["feature_count"]:
                result.warnings.append(
                    f"特征数量不匹配: 期望 {expected['feature_count']}, 实际 {actual_count}"
                )

        # 检查训练/测试行数
        if "train_rows" in expected and result.metrics.get("train_rows"):
            actual = result.metrics["train_rows"]
            if actual != expected["train_rows"]:
                result.warnings.append(f"训练行数不匹配: 期望 {expected['train_rows']}, 实际 {actual}")

        if "test_rows" in expected and result.metrics.get("test_rows"):
            actual = result.metrics["test_rows"]
            if actual != expected["test_rows"]:
                result.warnings.append(f"测试行数不匹配: 期望 {expected['test_rows']}, 实际 {actual}")

        # 对于 train_from_archive_features 模式，检查指标是否合理
        if mode == "train_from_archive_features" and result.metrics:
            for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                if metric_name in result.metrics:
                    val = result.metrics[metric_name]
                    if val is not None and not (0 <= val <= 1):
                        result.warnings.append(f"{metric_name} 超出合理范围: {val}")

            # 与 EarthByte 参考指标对比
            if result.metrics.get("f1") is not None:
                ref = EARTHBYTE_REFERENCE_METRICS.get("RF (Pre-PUB)", {})
                for metric_name, ref_val in ref.items():
                    mapped = {
                        "Accuracy": "accuracy",
                        "Precision": "precision",
                        "Recall": "recall",
                        "F1-Score": "f1",
                        "ROC_AUC": "roc_auc",
                    }.get(metric_name)
                    if mapped and mapped in result.metrics and result.metrics[mapped] is not None:
                        actual = result.metrics[mapped]
                        diff = abs(actual - ref_val)
                        result.reference_comparison[mapped] = {
                            "earthbyte": ref_val,
                            "phase2": actual,
                            "diff": diff,
                            "within_10pct": diff < 0.1,
                        }

        # 综合判断
        if result.errors:
            result.status = "error"
        elif all(result.artifacts.values()) if result.artifacts else True:
            result.status = "passed"
        else:
            missing = [k for k, v in result.artifacts.items() if not v]
            result.status = "failed"
            result.errors.append(f"缺失产物: {missing}")

    except ConfigError as exc:
        result.status = "error"
        result.errors.append(f"配置错误: {exc}")
    except FileNotFoundError as exc:
        result.status = "error"
        result.errors.append(f"文件未找到: {exc}")
    except ImportError as exc:
        result.status = "skipped"
        result.warnings.append(f"缺少依赖: {exc}")
    except Exception as exc:
        result.status = "error"
        result.errors.append(f"{type(exc).__name__}: {exc}")

    return result


def step_compare_archive_artifacts(baseline_result: ValidationResult) -> dict[str, Any]:
    """对比归档回放产物与原始 EarthByte 归档文件是否一致。"""
    comparison = {"status": "unknown", "details": {}}

    if baseline_result.status != "passed":
        comparison["status"] = "skipped"
        comparison["details"]["reason"] = "基线实验未通过"
        return comparison

    manifest = baseline_result.manifest
    input_paths = manifest.get("input_paths", {})
    archive_artifacts = input_paths.get("archive_artifacts", {})

    output_dir = Path(manifest.get("config", {}).get("experiment", {}).get("output_dir", ""))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    checks = []
    for artifact_name, artifact_info in archive_artifacts.items():
        source_path = Path(artifact_info.get("path", ""))
        if artifact_name in ("Xy_train.csv", "Xy_train_new.csv", "Xy_rf_train.csv", "Xy_rf_test.csv",
                            "target_features.csv", "target_coords_purged.csv", "target_mask.csv"):
            replayed = output_dir / "intermediate" / artifact_name
        elif artifact_name.endswith(".pkl"):
            replayed = output_dir / "models" / artifact_name
        elif artifact_name.endswith(".csv") or artifact_name.endswith(".tif"):
            replayed = output_dir / "predictions" / artifact_name
        else:
            continue

        source_hash = artifact_info.get("sha256", "")
        replayed_hash = sha256_file(replayed) if replayed.is_file() else ""

        checks.append({
            "artifact": artifact_name,
            "source_exists": source_path.is_file(),
            "replayed_exists": replayed.is_file(),
            "hash_match": source_hash == replayed_hash if source_hash and replayed_hash else None,
        })

    comparison["details"]["checks"] = checks
    all_match = all(c.get("hash_match") in (True, None) and c["replayed_exists"] for c in checks)
    comparison["status"] = "passed" if all_match else "warning"
    if not all_match:
        mismatches = [c["artifact"] for c in checks if c.get("hash_match") is False]
        comparison["details"]["mismatches"] = mismatches

    return comparison


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def generate_report(
    config_results: dict,
    experiment_results: list[ValidationResult],
    archive_comparisons: dict,
    output_path: Path | None = None,
) -> str:
    """生成纯文本或 HTML 验证报告。"""
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("  MPM_codex_phase2 复现验证报告")
    lines.append("=" * 72)
    lines.append(f"  生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  项目根目录: {ROOT}")
    lines.append("")

    # 第一节：配置验证
    lines.append("-" * 72)
    lines.append("  1. 配置验证")
    lines.append("-" * 72)
    lines.append(f"  总计: {config_results['total']}, "
                 f"通过: {config_results['passed']}, "
                 f"失败: {config_results['failed']}")
    for detail in config_results["details"]:
        status_icon = "✓" if detail["status"] == "passed" else "✗"
        lines.append(f"    {status_icon} {detail.get('name', detail.get('config', 'unknown'))}")
        if detail["status"] == "failed":
            lines.append(f"      错误: {detail['error']}")
    lines.append("")

    # 第二节：实验运行
    lines.append("-" * 72)
    lines.append("  2. 实验运行")
    lines.append("-" * 72)

    by_mode: dict[str, list[ValidationResult]] = {}
    for r in experiment_results:
        by_mode.setdefault(r.mode, []).append(r)

    for mode, results in by_mode.items():
        lines.append(f"  [{mode}]")
        for r in results:
            icon = {"passed": "✓", "failed": "✗", "skipped": "○", "error": "⚠"}.get(r.status, "?")
            lines.append(f"    {icon} {r.experiment_name} ({r.status}) — {r.duration_seconds:.1f}s")
            if r.errors:
                for err in r.errors:
                    lines.append(f"      错误: {err}")
            if r.warnings:
                for warn in r.warnings[:3]:
                    lines.append(f"      警告: {warn}")
            if r.metrics:
                metrics_str = ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in r.metrics.items()
                    if k in ("accuracy", "precision", "recall", "f1", "roc_auc", "mode")
                )
                if metrics_str:
                    lines.append(f"      指标: {metrics_str}")
            if r.reference_comparison:
                lines.append("      与 EarthByte 参考值对比:")
                for metric_name, comp in r.reference_comparison.items():
                    match_icon = "✓" if comp["within_10pct"] else "⚠"
                    lines.append(
                        f"        {match_icon} {metric_name}: "
                        f"EarthByte={comp['earthbyte']:.4f}, "
                        f"Phase2={comp['phase2']:.4f}, "
                        f"diff={comp['diff']:.4f}"
                    )
        lines.append("")

    # 第三节：归档产物对比
    lines.append("-" * 72)
    lines.append("  3. 归档产物一致性")
    lines.append("-" * 72)
    for exp_name, comp in archive_comparisons.items():
        status_icon = {"passed": "✓", "warning": "⚠", "skipped": "○"}.get(comp["status"], "?")
        lines.append(f"  {status_icon} {exp_name}: {comp['status']}")
        if comp.get("details", {}).get("mismatches"):
            lines.append(f"    不匹配: {comp['details']['mismatches']}")
    lines.append("")

    # 总结
    lines.append("=" * 72)
    runs_executed = len(experiment_results)
    passed_count = sum(1 for r in experiment_results if r.status == "passed")
    failed_count = sum(1 for r in experiment_results if r.status == "failed")
    error_count = sum(1 for r in experiment_results if r.status == "error")
    skipped_count = sum(1 for r in experiment_results if r.status == "skipped")
    lines.append(f"  总结: {passed_count} 通过, {failed_count} 失败, "
                 f"{error_count} 错误, {skipped_count} 跳过")
    if runs_executed == 0:
        lines.append("  ○ 配置验证完成，但未运行任何实验（未验证复现）")
    elif config_results["failed"] == 0 and failed_count == 0 and error_count == 0 and skipped_count == 0:
        lines.append("  ✓ 所有验证通过 — 框架可正确复现 EarthByte 实验结果")
    else:
        lines.append("  ⚠ 存在未通过或未完成的验证项，请检查上述错误")
    lines.append("=" * 72)

    report_text = "\n".join(lines)

    if output_path and output_path.suffix == ".html":
        html = _render_html(report_text, experiment_results, archive_comparisons)
        output_path.write_text(html, encoding="utf-8")
        return html

    return report_text


def _render_html(
    text_report: str,
    experiment_results: list[ValidationResult],
    archive_comparisons: dict,
) -> str:
    """将验证结果渲染为 HTML 报告。"""
    rows_html = ""
    for r in experiment_results:
        icon = {"passed": "✅", "failed": "❌", "skipped": "⬜", "error": "⚠️"}.get(r.status, "❓")
        metrics_str = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in r.metrics.items()
            if k in ("accuracy", "precision", "recall", "f1", "roc_auc")
        )
        errors_str = "<br>".join(r.errors) if r.errors else "—"
        warnings_str = "<br>".join(r.warnings[:5]) if r.warnings else "—"
        rows_html += f"""
        <tr>
            <td>{icon}</td>
            <td>{r.experiment_name}</td>
            <td><code>{r.mode}</code></td>
            <td>{r.status}</td>
            <td>{r.duration_seconds:.1f}s</td>
            <td style="font-size:0.9em">{metrics_str}</td>
            <td style="font-size:0.85em;color:#c00">{errors_str}</td>
            <td style="font-size:0.85em;color:#960">{warnings_str}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>MPM Phase2 复现验证报告</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f5f5f5; }}
  tr:hover {{ background: #fafafa; }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }}
  .summary {{ font-size: 1.1em; padding: 1em; background: #f8f8f8; border-radius: 6px; }}
  .passed {{ color: #1a7a1a; }}
  .failed {{ color: #c00; }}
  pre {{ background: #f5f5f5; padding: 1em; border-radius: 4px; overflow-x: auto; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>MPM_codex_phase2 复现验证报告</h1>
<p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="summary">
  <strong>通过:</strong> {sum(1 for r in experiment_results if r.status == 'passed')}
  &nbsp;|&nbsp; <strong>失败:</strong> {sum(1 for r in experiment_results if r.status == 'failed')}
  &nbsp;|&nbsp; <strong>错误:</strong> {sum(1 for r in experiment_results if r.status == 'error')}
  &nbsp;|&nbsp; <strong>跳过:</strong> {sum(1 for r in experiment_results if r.status == 'skipped')}
</div>

<h2>实验详情</h2>
<table>
<thead>
<tr>
  <th></th><th>实验</th><th>模式</th><th>状态</th><th>耗时</th><th>指标</th><th>错误</th><th>警告</th>
</tr>
</thead>
<tbody>{rows_html}
</tbody>
</table>

<h2>完整报告</h2>
<pre>{text_report}</pre>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="复现验证：在 MPM_codex_phase2 框架中验证 EarthByte 实验结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --all                       # 运行所有验证
  %(prog)s --validate-only             # 仅验证配置
  %(prog)s --mode archive_replay       # 仅运行归档回放
  %(prog)s --experiment lachlan_rf_baseline  # 运行单个实验
  %(prog)s --all --report report.html  # 生成 HTML 报告
        """,
    )
    parser.add_argument("--all", action="store_true", help="运行所有验证（含训练模式）")
    parser.add_argument("--validate-only", action="store_true", help="仅验证配置，不运行实验")
    parser.add_argument(
        "--mode",
        choices=["archive_replay", "train_from_archive_features", "raw_gis"],
        help="指定运行模式",
    )
    parser.add_argument("--experiment", help="运行指定实验名称")
    parser.add_argument("--include-raw-gis", action="store_true", help="包含 raw_gis 模式（需要 GIS 依赖）")
    parser.add_argument("--report", type=Path, help="输出报告文件路径（.html 生成 HTML 报告）")
    parser.add_argument("--quiet", action="store_true", help="静默模式，仅输出最终报告")

    args = parser.parse_args()

    if not any([args.all, args.validate_only, args.mode, args.experiment]):
        parser.print_help()
        return 1

    log = lambda msg: None if args.quiet else print(msg)

    load_builtin_components()

    experiment_results: list[ValidationResult] = []
    archive_comparisons: dict[str, Any] = {}

    # 第一步：配置验证
    log("🔍 验证实验配置...")
    config_results = step_validate_configs()
    if config_results["failed"] > 0:
        log(f"  ⚠ {config_results['failed']} 个配置验证失败")
    else:
        log(f"  ✓ 所有 {config_results['total']} 个配置验证通过")

    if args.validate_only:
        report = generate_report(config_results, experiment_results, archive_comparisons, args.report)
        if args.report:
            print(f"报告已写入: {args.report}")
        else:
            print(report)
        return 0 if config_results["failed"] == 0 else 1

    # 第二步：确定要运行的实验
    to_run: list[tuple[str, dict]] = []

    if args.experiment:
        for mode, experiments in EXPERIMENT_MANIFEST.items():
            for exp in experiments:
                if exp["name"] == args.experiment:
                    to_run.append((mode, exp))
                    break
        if not to_run:
            print(f"未找到实验: {args.experiment}")
            return 1
    elif args.mode:
        for exp in EXPERIMENT_MANIFEST.get(args.mode, []):
            to_run.append((args.mode, exp))
    elif args.all:
        for mode in ["archive_replay", "train_from_archive_features"]:
            for exp in EXPERIMENT_MANIFEST.get(mode, []):
                to_run.append((mode, exp))
        if args.include_raw_gis:
            for exp in EXPERIMENT_MANIFEST.get("raw_gis", []):
                to_run.append(("raw_gis", exp))

    if not to_run:
        print("未选择任何实验，请使用 --all / --mode / --experiment")
        return 1

    # 第三步：运行实验
    log(f"\n🚀 运行 {len(to_run)} 个实验...")
    for i, (mode, exp_def) in enumerate(to_run):
        log(f"  [{i+1}/{len(to_run)}] {exp_def['name']} ({mode})...")
        result = step_run_experiment(exp_def, mode, include_raw_gis=args.include_raw_gis)
        experiment_results.append(result)

        status_icon = {"passed": "✓", "failed": "✗", "skipped": "○", "error": "⚠"}.get(result.status, "?")
        log(f"    {status_icon} {result.status} ({result.duration_seconds:.1f}s)")

        if result.errors:
            for err in result.errors:
                log(f"      错误: {err}")

        # 第四步：对 archive_replay 实验对比归档产物
        if mode == "archive_replay" and result.status == "passed":
            log(f"    🔍 对比归档产物...")
            comp = step_compare_archive_artifacts(result)
            archive_comparisons[exp_def["name"]] = comp
            comp_icon = {"passed": "✓", "warning": "⚠"}.get(comp["status"], "?")
            log(f"    {comp_icon} 归档一致性: {comp['status']}")

    # 第五步：生成报告
    log(f"\n📊 生成验证报告...")
    report = generate_report(config_results, experiment_results, archive_comparisons, args.report)

    if args.report:
        print(f"报告已写入: {args.report}")
    else:
        print(report)

    # 返回码
    has_failures = any(r.status in ("failed", "error") for r in experiment_results)
    return 1 if has_failures or config_results["failed"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())