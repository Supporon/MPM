"""CNN2D 适配器的配置校验与训练循环保护测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bootstrap import load_builtin_components
from src.models.registry import MODEL_REGISTRY


def _adapter():
    load_builtin_components()
    return MODEL_REGISTRY.create("cnn2d")


def test_validate_config_rejects_zero_epochs():
    """P0-03: 恢复 n_epochs 正整数校验，0 轮配置被拒绝。"""
    adapter = _adapter()
    with pytest.raises(ValueError):
        adapter.validate_config({"n_epochs": 0})
    with pytest.raises(ValueError):
        adapter.validate_config({"n_epochs": -3})
    # 正常配置通过。
    adapter.validate_config({"n_epochs": 5, "batch_size": 8})


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_fit_zero_epochs_raises_even_bypassing_validation():
    """绕过配置校验直接构建估计器时，零优化步训练仍被拒绝。"""
    adapter = _adapter()
    model = adapter.build({"n_epochs": 0, "batch_size": 4, "patch": 5, "device": "cpu"}, seed=0)
    grid = np.random.default_rng(0).normal(size=(8, 8, 3)).astype(np.float32)
    inside = np.ones((8, 8), dtype=bool)
    cells = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    y = np.array([1, 0, 1, 0, 1, 0], dtype=np.float32)
    with pytest.raises(RuntimeError):
        model.fit(None, y, grid=grid, inside=inside, train_cells=cells)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_fit_minibatches_and_predicts():
    """batch_size 真正参与分批训练；正常正轮数训练后可预测。"""
    adapter = _adapter()
    model = adapter.build({"n_epochs": 2, "batch_size": 4, "patch": 5, "device": "cpu"}, seed=0)
    grid = np.random.default_rng(0).normal(size=(8, 8, 3)).astype(np.float32)
    inside = np.ones((8, 8), dtype=bool)
    cells = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    y = np.array([1, 0, 1, 0, 1, 0], dtype=np.float32)
    model.fit(None, y, grid=grid, inside=inside, train_cells=cells)
    proba = model.predict_proba(None)
    assert proba.shape == (64, 2)  # 8×8 有效单元
    assert np.all((proba >= 0) & (proba <= 1))
