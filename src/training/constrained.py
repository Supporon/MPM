"""约束重加权：给非梯度（sklearn 兼容）模型加谓词约束的通用训练组件。

把 LUSI 的约束项 ``τ·(1/N)·(φ̃ᵀe)²`` 翻译成"样本权重"，从而让无梯度的树集成/集成
模型也能满足谓词约束（方案 B）：

    e_i = p_i - y_i（训练残差），约束目标 g = Σ φ_i e_i / Σ φ_i  →  0

每轮：训练基模型 → 算组内平均残差 g → 若 |g| 超容差，按 g 的方向上调组内
"欠预测"那一类的样本权重 → 重训，逐步逼近 g≈0。

可包装任意 sklearn 兼容分类器（RandomForest / SPE / XGBoost 等），只要它实现
``fit(X, y, sample_weight=...)``、``predict_proba`` 与 ``get_params``。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def extract_phi(constraints: Mapping[str, Any] | None, n: int) -> np.ndarray | None:
    """从 constraints 提取合并的 φ 向量（支持 phi_vector / phi_vectors）。"""
    if not constraints:
        return None
    if "phi_vector" in constraints:
        return np.asarray(constraints["phi_vector"], dtype=float).ravel()
    if "phi_vectors" in constraints:
        vecs = [np.asarray(v["vector"], dtype=float).ravel() for v in constraints["phi_vectors"]]
        if vecs:
            return np.sum(vecs, axis=0)
    return None


class ConstrainedReweighting(BaseEstimator, ClassifierMixin):
    """约束重加权包装器：给任意 sklearn 兼容分类器加谓词约束。"""

    def __init__(
        self,
        base_estimator=None,
        n_iter: int = 5,
        eta: float = 0.5,
        tol: float = 1e-3,
    ):
        self.base_estimator = base_estimator
        self.n_iter = n_iter
        self.eta = eta
        self.tol = tol

    def fit(self, X, y, sample_weight=None, constraints=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)
        phi = extract_phi(constraints, n)
        w = np.asarray(sample_weight, dtype=float).copy() if sample_weight is not None else np.ones(n)

        estimator = None
        for _ in range(self.n_iter):
            estimator = clone(self.base_estimator)
            estimator.fit(X, y, sample_weight=w)
            if phi is None or phi.sum() <= 0:
                break
            p = estimator.predict_proba(X)[:, 1]
            e = p - y
            g = float((phi * e).sum() / phi.sum())
            if abs(g) < self.tol:
                break
            correction = (1.0 - y) if g > 0 else y  # 上调组内欠预测类
            w = w * (1.0 + self.eta * phi * correction)
            w = w / w.mean()
        self.estimator_ = estimator
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)

    def get_params(self, deep=True):
        params = {"n_iter": self.n_iter, "eta": self.eta, "tol": self.tol}
        if deep and self.base_estimator is not None:
            for key, val in self.base_estimator.get_params(deep=True).items():
                params[f"base_estimator__{key}"] = val
        else:
            params["base_estimator"] = self.base_estimator
        return params

    def set_params(self, **params):
        own = {}
        base = {}
        for key, val in params.items():
            if key.startswith("base_estimator__"):
                base[key[len("base_estimator__"):]] = val
            elif key == "base_estimator":
                self.base_estimator = val
            else:
                own[key] = val
        for key, val in own.items():
            setattr(self, key, val)
        if base and self.base_estimator is not None:
            self.base_estimator.set_params(**base)
        return self
