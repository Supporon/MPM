#!/usr/bin/env python3
"""转导式模型：标签传播（LabelSpreading）+ 概率校准（Platt scaling）。

转导学习直接利用未标注的待预测数据参与训练：把训练点（矿点/背景，有标签）与
全区预测网格单元（无标签）拼成特征相似图，用 sklearn 的 LabelSpreading 传播标签。

LabelSpreading 输出的 ``label_distributions_`` 是图传播的软分数，不是校准概率
（分数刻度可能整体偏移，0.5 阈值不靠谱）。因此默认做 **Platt scaling**：

    P(y=1) = sigmoid(a · f + b)

其中 a、b 用**交叉验证 out-of-fold 分数**拟合（验证折训练点 snap 到网格读软分数，
该分数未被它自己的标签锚定过，无泄漏）。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ..core.contracts import OptionalDependencyError, TrainingData
from .registry import MODEL_REGISTRY


class LabelSpreadingClassifier(BaseEstimator, ClassifierMixin):
    """sklearn 兼容的转导式标签传播分类器（grid 型），带概率校准。"""

    def __init__(
        self,
        kernel: str = "knn",
        n_neighbors: int = 7,
        alpha: float = 0.2,
        max_iter: int = 30,
        n_jobs: int = -1,
        calibrate: bool = True,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.kernel = kernel
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.calibrate = calibrate
        self.cv_folds = cv_folds
        self.random_state = random_state
        self._predict_cells = None

    def _run_ls(self, X, y, grid_arr, inside_2d):
        """跑一次 LabelSpreading，返回 (H, W) 的网格软分数图（外部 NaN）。"""
        from sklearn.semi_supervised import LabelSpreading

        X_unlabeled = grid_arr[inside_2d]
        X_all = np.vstack([X, X_unlabeled])
        y_all = np.concatenate([y, -np.ones(len(X_unlabeled), dtype=int)])
        ls = LabelSpreading(
            kernel=self.kernel, n_neighbors=self.n_neighbors,
            alpha=self.alpha, max_iter=self.max_iter, n_jobs=self.n_jobs,
        )
        ls.fit(X_all, y_all)
        probs = np.asarray(ls.label_distributions_[len(X):, 1], dtype=float)
        H, W = inside_2d.shape
        grid_map = np.full((H, W), np.nan, dtype=float)
        grid_map[inside_2d] = probs
        return grid_map

    def _fit_platt(self, scores, y):
        """用 (scores, y) 拟合 Platt 校准器：sigmoid(a·score + b)。"""
        from sklearn.linear_model import LogisticRegression

        mask = ~np.isnan(scores)
        if mask.sum() < 2:
            return None
        self.platt_ = LogisticRegression(C=1.0, solver="lbfgs")
        self.platt_.fit(scores[mask].reshape(-1, 1), y[mask])
        return self.platt_

    def _calibrate(self, scores):
        """把软分数映射到校准概率。"""
        if self.platt_ is None:
            return scores
        return self.platt_.predict_proba(np.asarray(scores, dtype=float).reshape(-1, 1))[:, 1]

    def fit(self, X, y, sample_weight=None, grid=None, inside=None, train_cells=None):
        if grid is None or inside is None:
            raise ValueError("label_spreading requires grid/inside injected via fit_params")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        grid_arr = np.asarray(grid, dtype=float)
        inside_2d = np.asarray(inside, dtype=bool)
        train_cells = np.asarray(train_cells, dtype=int) if train_cells is not None else None

        # 全量 LabelSpreading → 网格软分数
        grid_map_raw = self._run_ls(X, y, grid_arr, inside_2d)

        self.platt_ = None
        if self.calibrate:
            # CV out-of-fold 分数拟合 Platt
            oof = self._out_of_fold_scores(X, y, grid_arr, inside_2d, train_cells)
            self._fit_platt(oof, y)

        # 应用校准（若未校准则用原软分数）
        raw_vals = grid_map_raw[inside_2d]
        calibrated_vals = self._calibrate(raw_vals)
        self.grid_prob_map_ = np.full(inside_2d.shape, np.nan, dtype=float)
        self.grid_prob_map_[inside_2d] = calibrated_vals

        self.inside_ = inside_2d
        self.classes_ = np.array([0, 1])
        # 未调用 set_predict_cells 时，predict_proba 默认在整幅有效单元上返回。
        self._predict_cells = None
        # 记录校准实际使用的折数（小样本时可能小于配置的 cv_folds）。
        self.cv_folds_ = self._resolve_cv_folds(y)
        return self

    def _resolve_cv_folds(self, y) -> int:
        """返回实际可用的校准折数（分层折数不能超过任一类样本数）。"""
        counts = np.bincount(np.asarray(y, dtype=int).ravel(), minlength=2)
        if counts.min() < 2:
            return 0
        return max(2, min(int(self.cv_folds), int(counts.min())))

    def _out_of_fold_scores(self, X, y, grid_arr, inside_2d, train_cells):
        """对训练点做分层 CV，返回每个训练点的 OOF 软分数。

        折数取 ``cv_folds``，但受少数类样本数约束（少正例时自动降折，避免
        ``StratifiedKFold`` 因折数超过类别成员数而直接失败）。
        """
        from sklearn.model_selection import StratifiedKFold

        oof = np.full(len(X), np.nan, dtype=float)
        if train_cells is None or len(np.unique(y)) < 2:
            return oof
        n_splits = self._resolve_cv_folds(y)
        if n_splits < 2:
            return oof
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for tr_idx, va_idx in skf.split(X, y):
            map_fold = self._run_ls(X[tr_idx], y[tr_idx], grid_arr, inside_2d)
            cells_va = train_cells[va_idx]
            oof[va_idx] = map_fold[cells_va[:, 0], cells_va[:, 1]]
        return oof

    def set_predict_cells(self, cells):
        self._predict_cells = cells

    def predict_proba(self, X):
        if self._predict_cells is not None:
            cells = self._predict_cells
            p = self.grid_prob_map_[cells[:, 0], cells[:, 1]]
        else:
            p = self.grid_prob_map_[self.inside_]
        p = np.nan_to_num(p, nan=0.0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {
            "kernel": self.kernel, "n_neighbors": self.n_neighbors,
            "alpha": self.alpha, "max_iter": self.max_iter, "n_jobs": self.n_jobs,
            "calibrate": self.calibrate, "cv_folds": self.cv_folds,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


@MODEL_REGISTRY.decorator("label_spreading")
class LabelSpreadingAdapter:
    """转导式标签传播适配器。grid_model=True 表示需要网格数据参与训练。"""

    name = "label_spreading"
    artifact_filename = "model_label_spreading.pkl"
    supports_constraints = False
    grid_model = True

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        if int(params.get("n_neighbors", 7)) < 1:
            raise ValueError("label_spreading n_neighbors must be positive")
        if int(params.get("cv_folds", 5)) < 2:
            raise ValueError("label_spreading cv_folds must be at least 2")

    def build(self, params: Mapping[str, Any], seed: int) -> LabelSpreadingClassifier:
        return LabelSpreadingClassifier(
            kernel=str(params.get("kernel", "knn")),
            n_neighbors=int(params.get("n_neighbors", 7)),
            alpha=float(params.get("alpha", 0.2)),
            max_iter=int(params.get("max_iter", 30)),
            n_jobs=int(params.get("n_jobs", -1)),
            calibrate=bool(params.get("calibrate", True)),
            cv_folds=int(params.get("cv_folds", 5)),
            random_state=seed,
        )

    def fit_params(self, data: TrainingData) -> Mapping[str, Any]:
        return {
            "sample_weight": data.sample_weight,
            "grid": data.metadata["grid"],
            "inside": data.metadata["inside"],
            "train_cells": data.metadata["train_cells"],
        }
