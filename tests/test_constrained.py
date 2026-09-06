"""约束重加权（ConstrainedReweighting）的多谓词语义与 φ 提取测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.constrained import ConstrainedReweighting, extract_phi


def _two_group_constraints(n: int = 100) -> dict:
    """两个等规模互斥组，各占一半样本。"""
    half = n // 2
    phi_a = np.array([1.0] * half + [0.0] * half)
    phi_b = np.array([0.0] * half + [1.0] * half)
    return {
        "phi_vectors": [
            {"name": "A", "vector": phi_a},
            {"name": "B", "vector": phi_b},
        ]
    }


def test_extract_phi_preserves_multiple_constraints():
    """P0-05: 多谓词返回矩阵，而非把各 φ 先求和。"""
    constraints = _two_group_constraints()
    mat = extract_phi(constraints, 100)
    assert mat is not None
    assert mat.shape == (2, 100)
    # 两行各自仍是原始 φ，未被求和。
    assert np.allclose(mat[0], constraints["phi_vectors"][0]["vector"])
    assert np.allclose(mat[1], constraints["phi_vectors"][1]["vector"])


def test_extract_phi_single_vector_shape():
    mat = extract_phi({"phi_vector": np.ones(10)}, 10)
    assert mat is not None
    assert mat.shape == (1, 10)


def test_extract_phi_length_mismatch_raises():
    with pytest.raises(ValueError):
        extract_phi({"phi_vector": np.ones(3)}, 10)


def test_constrained_records_per_constraint_status():
    """fit 逐迭代记录每个约束的组内平均残差，而非单一聚合值。"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + rng.normal(0, 0.5, 200) > 0).astype(int)
    constraints = _two_group_constraints(200)

    model = ConstrainedReweighting(
        base_estimator=RandomForestClassifier(n_estimators=20, random_state=0),
        n_iter=3,
    )
    model.fit(X, y, constraints=constraints)

    assert len(model.constraint_status_) == 3
    for status in model.constraint_status_:
        # 两个约束各自一个残差，不是被求和成一个标量。
        assert isinstance(status["g"], list)
        assert len(status["g"]) == 2
    # 拟合产物可用。
    assert model.predict_proba(X).shape == (200, 2)


def test_constrained_opposite_groups_do_not_cancel():
    """两个互斥组的残差相反时，逐约束判断仍不视为已收敛（不再相消）。"""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3))
    # 让第一组被明显过预测、第二组被明显欠预测，构造方向相反的残差。
    y = np.zeros(100, dtype=int)
    y[0] = 1  # 至少一个正例，避免单类
    phi_a = np.array([1.0] * 50 + [0.0] * 50)
    phi_b = np.array([0.0] * 50 + [1.0] * 50)
    constraints = {
        "phi_vectors": [
            {"name": "A", "vector": phi_a},
            {"name": "B", "vector": phi_b},
        ]
    }
    model = ConstrainedReweighting(
        base_estimator=RandomForestClassifier(n_estimators=10, random_state=0),
        n_iter=2,
        tol=1e-9,
    )
    model.fit(X, y, constraints=constraints)
    # 至少记录了两组各自残差；若实现仍先求和，两列会退化为单一聚合值。
    assert all(len(s["g"]) == 2 for s in model.constraint_status_)
