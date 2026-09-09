"""TrainingData 主训练前契约检查（P1-05）测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.contracts import TrainingData, validate_training_data


def _data(labels, weights=None, features=None):
    n = len(labels)
    feats = features if features is not None else np.arange(n * 2, dtype=float).reshape(n, 2)
    return TrainingData(
        features=pd.DataFrame(feats, columns=["a", "b"]),
        labels=pd.Series(labels, dtype=float),
        sample_weight=pd.Series(weights if weights is not None else np.ones(n)),
    )


def test_accepts_two_class_finite_data():
    validate_training_data(_data([0, 1, 0, 1]))  # 不抛异常


def test_rejects_single_class_training():
    with pytest.raises(ValueError, match="single-class"):
        validate_training_data(_data([0, 0, 0, 0]))


def test_rejects_non_finite_features():
    feats = np.array([[0.0, 1.0], [np.nan, 2.0], [3.0, 4.0], [5.0, 6.0]])
    with pytest.raises(ValueError, match="non-finite"):
        validate_training_data(_data([0, 1, 0, 1], features=feats))


def test_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        validate_training_data(_data([0, 1, 0, 1], weights=[1.0, -0.5, 1.0, 1.0]))


def test_rejects_zero_total_weight():
    with pytest.raises(ValueError, match="sum to zero"):
        validate_training_data(_data([0, 1, 0, 1], weights=[0.0, 0.0, 0.0, 0.0]))
