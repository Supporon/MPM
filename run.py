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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a configured MPM experiment.")
    parser.add_argument("--config", default="./configs/experiments/lachlan_rf_phase2.yaml", help="YAML experiment configuration")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the resolved component graph without touching dataset files.",
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config)
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
            return 0
        manifest = Experiment(config).run()
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError, NotImplementedError) as error:
        parser.error(str(error))
    print(f"Completed {manifest['experiment']} -> {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
