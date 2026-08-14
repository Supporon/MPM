#!/usr/bin/env python3
"""列出二阶段框架中当前已注册的组件名称。"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bootstrap import load_builtin_components
from src.core.config import load_config
from src.knowledge.registry import KNOWLEDGE_REGISTRY
from src.models.registry import LABEL_REFINER_REGISTRY, MODEL_REGISTRY
from src.operators.features.registry import FEATURE_OPERATOR_REGISTRY
from src.predicates.registry import PREDICATE_REGISTRY
from src.tasks.registry import TASK_REGISTRY
from src.tuning.registry import TUNER_REGISTRY
from src.validation.registry import METRIC_REGISTRY, SPLITTER_REGISTRY


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Optional YAML whose plugin modules should be loaded")
    args = parser.parse_args()
    load_builtin_components()
    if args.config:
        load_config(args.config)
    print(
        json.dumps(
            {
                "tasks": TASK_REGISTRY.names(),
                "models": MODEL_REGISTRY.names(),
                "label_refiners": LABEL_REFINER_REGISTRY.names(),
                "feature_operators": FEATURE_OPERATOR_REGISTRY.names(),
                "knowledge_providers": KNOWLEDGE_REGISTRY.names(),
                "predicates": PREDICATE_REGISTRY.names(),
                "splitters": SPLITTER_REGISTRY.names(),
                "metrics": METRIC_REGISTRY.names(),
                "tuners": TUNER_REGISTRY.names(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
