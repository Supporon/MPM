"""从归档重建训练点坐标，供空间谓词使用。

``train_from_archive_features`` 模式下，归档特征（Xy_rf_train/Xy_rf_test）本身
不含 X/Y 坐标，而空间谓词（spatial_box / spatial_distance）需要坐标。本模块用归档的
StandardScaler + OneHotEncoder 从原始 806 列重建 549 个源点（277 矿点 + 272 背景）
的 138 维标准化特征，再与归档训练/测试特征逐行匹配，得到每行的坐标。

背景点坐标的精确对齐：``training_data_unlab_grids.csv``（dropna 前）含 5 个
NaN 行，正是这 5 行在后续被 dropna 丢弃；据此从 277 个背景坐标里剔除同样 5 行，
即可与 272 行背景特征一一对应。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CAT_COLS = [
    "MetamorphicFacies_Benambran",
    "MetamorphicFacies_KanimblanTablelands",
    "MetamorphicFacies_Tabberabberan",
    "Intrusions_Tabberabberan",
    "RockUnits_LAO",
]


def _manual_onehot(cat_df: pd.DataFrame, enc) -> np.ndarray:
    """旧版 OneHotEncoder 在新 sklearn 下无法 transform，按 categories_ 手动重建。"""
    cols = []
    for i, cats in enumerate(enc.categories_):
        vals = cat_df.iloc[:, i].astype(str).to_numpy()
        for cat in cats:
            cols.append((vals == str(cat)).astype(np.float32))
    return np.column_stack(cols)


def reconstruct_archive_points(archive_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """重建 549 个源点（277 矿点 + 272 背景）的坐标与标准化特征。

    Returns:
        coords: DataFrame [X, Y]，549 行（矿点在前，背景在后）。
        X: (549, 138) 标准化特征矩阵，列序与归档 Xy_train 一致。
    """
    with open(archive_dir / "st_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(archive_dir / "encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    deposit = pd.read_csv(archive_dir / "training_data_deposit.csv")
    unlab = pd.read_csv(archive_dir / "training_data_unlab.csv")
    num_cols = list(scaler.feature_names_in_)

    def reconstruct(df: pd.DataFrame) -> np.ndarray:
        num = scaler.transform(df[num_cols].to_numpy())
        onehot = _manual_onehot(df[CAT_COLS], encoder)
        return np.hstack([num, onehot])

    X = np.vstack([reconstruct(deposit), reconstruct(unlab)])

    deposit_coords = pd.read_csv(archive_dir / "training_data_deposit_coords.csv")[["X", "Y"]]
    unlab_coords = pd.read_csv(archive_dir / "training_data_unlab_coords.csv")[["X", "Y"]]

    # 背景点：277 坐标 → 去掉 dropna 前含 NaN 的 5 行 → 272，与 unlab 特征对齐
    unlab_grids = pd.read_csv(archive_dir / "training_data_unlab_grids.csv")
    nan_mask = unlab_grids.isna().any(axis=1).to_numpy()
    unlab_coords_aligned = unlab_coords.loc[~nan_mask].reset_index(drop=True)

    coords = pd.concat([deposit_coords, unlab_coords_aligned], ignore_index=True)
    return coords, X


def match_rows(X_source: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """把 frame 每一行（138 维特征）匹配到 X_source 的行号（最近邻）。"""
    idx = np.empty(len(frame), dtype=int)
    for i, row in enumerate(frame.to_numpy(dtype=float)):
        diff = np.abs(X_source - row).max(axis=1)
        idx[i] = int(np.argmin(diff))
    return idx


def build_grid(archive_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """重建 (H, W, C) 特征张量、有效掩膜与坐标轴（供 grid 型模型使用）。"""
    archive_dir = Path(archive_dir)
    mask = pd.read_csv(archive_dir / "target_mask.csv", header=None)
    inside = mask[2].astype(str).str.lower().isin(["1", "true", "1.0"]).to_numpy()
    feats = pd.read_csv(archive_dir / "target_features.csv").to_numpy(dtype=np.float32)
    x_axis = np.unique(mask[0].to_numpy(dtype=float))
    y_axis = np.unique(mask[1].to_numpy(dtype=float))
    col = np.searchsorted(x_axis, mask[0].to_numpy(dtype=float))
    row = np.searchsorted(y_axis, mask[1].to_numpy(dtype=float))
    H, W = len(y_axis), len(x_axis)
    tensor = np.zeros((H, W, feats.shape[1]), dtype=np.float32)
    tensor[row[inside], col[inside]] = feats
    inside_2d = np.zeros((H, W), dtype=bool)
    inside_2d[row[inside], col[inside]] = True
    return tensor, inside_2d, x_axis, y_axis


def snap_cells(coords: pd.DataFrame, x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    """把坐标点吸附到最近格网单元，返回 (N, 2) 的 (row, col)。"""
    c = np.argmin(np.abs(x_axis[None, :] - coords["X"].to_numpy(dtype=float)[:, None]), axis=1)
    r = np.argmin(np.abs(y_axis[None, :] - coords["Y"].to_numpy(dtype=float)[:, None]), axis=1)
    return np.column_stack([r, c])
