"""知识「被访问 vs 被实际使用」审计测试（P1-03）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bootstrap import load_builtin_components
from src.core.contracts import TrainingData
from src.core.experiment import _KnowledgeAccessTracker
from src.predicates.registry import PREDICATE_REGISTRY


def _data(n: int = 20) -> TrainingData:
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "X": np.linspace(140.0, 155.0, n),
            "Y": np.linspace(-40.0, -25.0, n),
        }
    )
    return TrainingData(
        features=features,
        labels=pd.Series(np.tile([1, 0], n // 2)),
        sample_weight=pd.Series(np.ones(n)),
    )


def _knowledge():
    return {
        "spatial_extent": {
            "bounds": {"min_x": 140.0, "min_y": -40.0, "max_x": 155.0, "max_y": -25.0},
            "center": {"x": 147.5, "y": -32.5},
            "span": {"x": 15.0, "y": 15.0},
            "n_units": 20,
            "source": "research_unit_metadata",
            "crs": None,
        }
    }


def test_manual_box_does_not_mark_knowledge_consumed():
    """手动指定中心/边长时，spatial_box 不读取知识，不应记为已消费。"""
    load_builtin_components()
    tracked = _KnowledgeAccessTracker(_knowledge())
    context = {"knowledge": tracked}

    predicate = PREDICATE_REGISTRY.create(
        "spatial_box",
        {"auto_center": False, "auto_side": False,
         "center_x": 147.5, "center_y": -32.5, "side_length": 5.0},
    )
    predicate.apply(_data(), context)

    assert "spatial_extent" not in tracked.accessed


def test_auto_box_marks_knowledge_consumed():
    """自动计算中心/边长时，spatial_box 读取知识，应记为已消费。"""
    load_builtin_components()
    tracked = _KnowledgeAccessTracker(_knowledge())
    context = {"knowledge": tracked}

    predicate = PREDICATE_REGISTRY.create(
        "spatial_box", {"auto_center": True, "auto_side": True},
    )
    result = predicate.apply(_data(), context)

    assert "spatial_extent" in tracked.accessed
    # 使用知识的中心/跨度（而非仅从坐标重估）。
    assert result.metadata["spatial_box"]["center_x"] == 147.5
    assert result.metadata["spatial_box"]["center_y"] == -32.5
