"""P1-08: 归档坐标匹配容差/歧义与网格吸附越界检测。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.archive_coords import match_rows, snap_cells


def test_match_rows_exact_and_tolerant():
    X_source = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    frame = pd.DataFrame([[3.0, 4.0], [1.0, 2.0]])
    idx = match_rows(X_source, frame)
    assert idx.tolist() == [1, 0]


def test_match_rows_rejects_over_tolerance():
    X_source = np.array([[1.0, 2.0], [3.0, 4.0]])
    frame = pd.DataFrame([[1.0, 2.5]])  # Chebyshev error 0.5 > 0.01
    with pytest.raises(ValueError, match="max abs error"):
        match_rows(X_source, frame, max_error=0.01)


def test_match_rows_rejects_ambiguous_match():
    # 两行源特征相同 → 任何一行都歧义
    X_source = np.array([[1.0, 2.0], [1.0, 2.0], [9.0, 9.0]])
    frame = pd.DataFrame([[1.0, 2.0]])
    with pytest.raises(ValueError, match="ambiguous"):
        match_rows(X_source, frame, max_error=1e-3)
    # 关闭歧义检查后返回最近邻（第一个）
    idx = match_rows(X_source, frame, max_error=1e-3, require_unique=False)
    assert idx[0] in (0, 1)


def test_snap_cells_snaps_and_warns_out_of_bounds():
    x_axis = np.array([0.0, 1.0, 2.0])
    y_axis = np.array([0.0, 1.0, 2.0])
    coords = pd.DataFrame({"X": [0.0, 1.0, 10.0], "Y": [0.0, 2.0, 20.0]})
    with pytest.warns(UserWarning, match="outside the grid extent"):
        cells = snap_cells(coords, x_axis, y_axis)
    # 远在格网之外的点被吸附到最近边缘单元，不报错
    assert cells.shape == (3, 2)
    assert cells[0].tolist() == [0, 0]
    assert cells[1].tolist() == [2, 1]


def test_snap_cells_no_warning_when_inside():
    x_axis = np.array([0.0, 1.0, 2.0])
    y_axis = np.array([0.0, 1.0, 2.0])
    coords = pd.DataFrame({"X": [0.0, 1.0, 2.0], "Y": [0.0, 1.0, 2.0]})
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 无越界 → 不应有警告
        cells = snap_cells(coords, x_axis, y_axis)
    assert cells.shape == (3, 2)
