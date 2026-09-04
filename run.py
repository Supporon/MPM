#!/usr/bin/env python3
"""运行或校验一个已配置的 MPM 实验。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.core.config import ConfigError, load_config
from src.core.experiment import Experiment
from src.utils.logging import get_logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a configured MPM experiment.")
    parser.add_argument("--config", default="./configs/experiments/lachlan_rf_phase2.yaml", help="YAML experiment configuration")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the resolved component graph without touching dataset files.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values (e.g. --set model.name=spe tuning.params.n_iter=100)",
    )
    args = parser.parse_args()
    log = get_logger("run")
    try:
        log.info("Loading config from %s", args.config)
        config = load_config(args.config)
        if args.set:
            from src.core.config import apply_cli_overrides
            config = apply_cli_overrides(config, args.set)
            log.info("Applied %d CLI override(s)", len(args.set))
        if args.validate_only:
            spec = config.spec
            print(
                json.dumps(
                    {
                        "experiment": spec.name,
                        "execution_mode": spec.execution_mode,
                        "task": spec.task.name,
                        "feature_operators": [item.name for item in spec.feature_operators],
                        "model": spec.model.name,
                        "label_refinement": spec.label_refinement.name if spec.label_refinement else None,
                        "knowledge": [item.name for item in spec.knowledge],
                        "predicates": [item.name for item in spec.predicates],
                        "holdout": spec.holdout.name,
                        "cross_validation": spec.cross_validation.name,
                        "metrics": list(spec.metrics),
                        "primary_metric": spec.primary_metric,
                        "tuner": spec.tuner.name,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            log.info("Config validation passed")
            return 0
        manifest = Experiment(config).run()
        # 运行完成后自动绘图（绘图失败不影响实验本身）
        try:
            from src.plotting.plot import plot_run
            out = plot_run(config.output_dir)
            if out is not None:
                log.info("Plot written to %s", out)
        except Exception as error:  # noqa: BLE001 - 绘图是尽力而为
            log.warning("Automatic plotting failed (experiment results unaffected): %s", error)
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError, NotImplementedError) as error:
        log.error("Experiment failed: %s", error)
        parser.error(str(error))
    print(f"Completed {manifest['experiment']} -> {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
