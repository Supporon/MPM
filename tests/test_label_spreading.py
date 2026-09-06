"""LabelSpreading 估计器契约（fit 返回 self / 预测状态 / 校准折数）测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.label_spreading import LabelSpreadingClassifier


def _small_grid():
    rng = np.random.default_rng(0)
    grid = rng.normal(size=(6, 6, 2)).astype(float)
    inside = np.ones((6, 6), dtype=bool)
    return grid, inside


def test_fit_returns_self_and_predict_proba_works_without_set_predict_cells():
    """P1-07: fit 返回 self，且未调用 set_predict_cells 也能预测。"""
    grid, inside = _small_grid()
    X = np.random.default_rng(0).normal(size=(4, 2))
    y = np.array([1, 0, 1, 0])
    cells = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

    model = LabelSpreadingClassifier(calibrate=True, cv_folds=5, random_state=0)
    result = model.fit(X, y, grid=grid, inside=inside, train_cells=cells)
    assert result is model
    proba = model.predict_proba(None)  # 不传 X，走 _predict_cells is None 分支
    assert proba.shape == (36, 2)  # 6×6 有效单元
    assert np.all(np.isfinite(proba))


def test_calibration_folds_adapt_to_minority_class():
    """少数类样本少于 cv_folds 时自动降折，不因分层折数超类别成员数而崩溃。"""
    grid, inside = _small_grid()
    X = np.random.default_rng(1).normal(size=(6, 2))
    y = np.array([1, 0, 0, 0, 0, 0])  # 仅 1 个正例
    cells = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])

    model = LabelSpreadingClassifier(calibrate=True, cv_folds=5, random_state=0)
    model.fit(X, y, grid=grid, inside=inside, train_cells=cells)
    # 正例仅 1 个，无法分层切分，校准折数降为 0（跳过校准，返回原软分数）。
    assert model.cv_folds_ == 0
    assert model.predict_proba(None).shape == (36, 2)
