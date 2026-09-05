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
import warnings
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


def match_rows(
    X_source: np.ndarray,
    frame: pd.DataFrame,
    *,
    max_error: float = 1e-3,
    require_unique: bool = True,
) -> np.ndarray:
    """把 frame 每一行（特征）匹配到 X_source 的行号（最近邻）。

    Args:
        X_source: shape ``(M, C)`` 的源特征矩阵。
        frame: shape ``(N, C)`` 的待匹配特征（列序须与 ``X_source`` 一致）。
        max_error: 允许的最大切比雪夫误差。重建特征与归档特征在浮点精度内
            应完全一致（实测 ~1e-15）；超过阈值说明特征 schema 不匹配。
        require_unique: True 时，若某行有多个源行落在容差内（歧义），报错。

    Returns:
        shape ``(N,)`` 的整数行号，``frame`` 每行对应 ``X_source`` 的最近邻。

    Raises:
        ValueError: 最近邻误差超过 ``max_error``，或存在歧义匹配。
    """
    idx = np.empty(len(frame), dtype=int)
    for i, row in enumerate(frame.to_numpy(dtype=float)):
        diff = np.abs(X_source - row).max(axis=1)
        nearest = int(np.argmin(diff))
        error = float(diff[nearest])
        if error > max_error:
            raise ValueError(
                f"archive coordinate match failed at row {i}: nearest match has "
                f"max abs error {error:.3e} > {max_error:.3e}. The reconstructed "
                "feature schema likely does not match the archive features."
            )
        if require_unique:
            within = np.flatnonzero(diff <= max_error)
            if len(within) > 1:
                raise ValueError(
                    f"archive coordinate match ambiguous at row {i}: {len(within)} "
                    f"source rows within tolerance (error {error:.3e} <= "
                    f"{max_error:.3e}). Refine the reconstruction or relax "
                    "max_error/require_unique."
                )
        idx[i] = nearest
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
    """把坐标点吸附到最近格网单元，返回 (N, 2) 的 (row, col)。

    坐标落在格网范围之外（超过半个格网间距）时发出警告：训练点可能覆盖比
    目标格网更大的区域（如整个州域 vs 靶区），此时吸附到边缘单元是既有
    行为；警告用于提示坐标 CRS 或格网范围是否与预期一致，而非静默忽略。
    """
    x = coords["X"].to_numpy(dtype=float)
    y = coords["Y"].to_numpy(dtype=float)
    x_step = float(np.diff(x_axis).min()) if len(x_axis) > 1 else np.inf
    y_step = float(np.diff(y_axis).min()) if len(y_axis) > 1 else np.inf
    x_lo = float(x_axis.min()) - x_step / 2
    x_hi = float(x_axis.max()) + x_step / 2
    y_lo = float(y_axis.min()) - y_step / 2
    y_hi = float(y_axis.max()) + y_step / 2
    outside = int(((x < x_lo) | (x > x_hi) | (y < y_lo) | (y > y_hi)).sum())
    if outside:
        warnings.warn(
            f"snap_cells: {outside}/{len(coords)} coordinate(s) fall outside the "
            f"grid extent (X∈[{x_lo:.3g}, {x_hi:.3g}], Y∈[{y_lo:.3g}, {y_hi:.3g}]). "
            "Verify the coordinate CRS matches the grid axes; points are snapped "
            "to the nearest edge cell.",
            UserWarning,
            stacklevel=2,
        )
    c = np.argmin(np.abs(x_axis[None, :] - x[:, None]), axis=1)
    r = np.argmin(np.abs(y_axis[None, :] - y[:, None]), axis=1)
    return np.column_stack([r, c])
