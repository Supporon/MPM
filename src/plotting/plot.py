"""结果绘图：训练点概率 + 全区栅格预测（红=高概率，蓝=低概率）。

对单个运行目录生成 1x2 图：
  (c) 训练点概率：边界 + 训练背景点/训练矿点（按预测概率着色）
      + 测试集已知矿点（红色描边）+ 概率 colorbar
  (d) (c) 叠加全区连续概率栅格

红高蓝低的发散色带（RdYlBu_r）符合靶区预测论文常见呈现：高概率区=红/暖色。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import numpy.ma as ma
import pandas as pd

for _fp in font_manager.findSystemFonts(fontext="ttf"):
    if "CJK" in _fp:
        try:
            font_manager.fontManager.addfont(_fp)
        except Exception:
            pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

try:
    import geopandas as gpd
    _HAS_GEOPANDAS = True
except ImportError:
    _HAS_GEOPANDAS = False

from ..data.archive_coords import match_rows, reconstruct_archive_points

CMAP = "RdYlBu_r"

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = ROOT.parent / "EarthByte-MPM_Lachlan_Porphyry" / "Datasets" / "Outputs_Cu_NSW_v1.6"
DEFAULT_BOUNDARY = (
    ROOT.parent / "EarthByte-MPM_Lachlan_Porphyry" / "Datasets" / "Frames"
    / "NSW_Boundary" / "NSW_STATE_POLYGON_shp_GDA94_NoIsland_ACT.shp"
)


def _load_boundary(path: Path):
    if _HAS_GEOPANDAS and path.is_file():
        return gpd.read_file(path)
    return None


def point_kind_labels(archive_dir: Path) -> np.ndarray:
    """从归档的矿点/背景点文件行数推导每个重建点的 kind，避免硬编码 277/272。"""
    n_deposit = len(pd.read_csv(archive_dir / "training_data_deposit.csv"))
    n_unlab = len(pd.read_csv(archive_dir / "training_data_unlab.csv"))
    return np.array(["deposit"] * n_deposit + ["unlab"] * n_unlab)


def build_point_table(archive_dir: Path):
    """返回 (tbl, X549)：tbl 含 X/Y/kind/split，X549 为 N×F 标准化特征。"""
    coords, X549 = reconstruct_archive_points(archive_dir)
    xy_rf_train = pd.read_csv(archive_dir / "Xy_rf_train.csv")
    xy_rf_test = pd.read_csv(archive_dir / "Xy_rf_test.csv")
    fcols = [c for c in xy_rf_train.columns if c not in ("sample_weight", "label")]
    train_idx = set(match_rows(X549, xy_rf_train[fcols]).tolist())
    test_idx = set(match_rows(X549, xy_rf_test[fcols]).tolist())
    kind = point_kind_labels(archive_dir)
    split = np.array(["train" if i in train_idx else "test" if i in test_idx else "?" for i in range(len(X549))])
    tbl = coords.copy()
    tbl["kind"] = kind
    tbl["split"] = split
    return tbl, X549


def reconstruct_raster(archive_dir: Path, score: np.ndarray):
    """把 7583 个有效单元分数重建为 (H,W) 栅格，返回 arr/extent/x_axis/y_axis。"""
    mask = pd.read_csv(archive_dir / "target_mask.csv", header=None)
    inside = mask[2].astype(str).str.lower().isin(["1", "true", "1.0"]).to_numpy()
    x_axis = np.unique(mask[0].to_numpy(dtype=float))
    y_axis = np.unique(mask[1].to_numpy(dtype=float))
    col = np.searchsorted(x_axis, mask[0].to_numpy(dtype=float))
    row = np.searchsorted(y_axis, mask[1].to_numpy(dtype=float))
    H, W = len(y_axis), len(x_axis)
    arr = np.full((H, W), np.nan, dtype=float)
    arr[row[inside], col[inside]] = score
    dx = float(np.diff(x_axis).min())
    dy = float(np.diff(y_axis).min())
    extent = [float(x_axis.min()) - dx / 2, float(x_axis.max()) + dx / 2,
              float(y_axis.min()) - dy / 2, float(y_axis.max()) + dy / 2]
    return arr, extent, x_axis, y_axis


def _resolve_archive_from_manifest(run_dir: Path) -> Path | None:
    """从运行 manifest 读取归档目录，使绘图不依赖硬编码的 NSW 归档路径。"""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    archive_dir = (manifest.get("input_paths") or {}).get("archive_dir")
    return Path(archive_dir) if archive_dir else None


def plot_run(
    run_dir: Path,
    archive_dir: Path | None = None,
    boundary_shp: Path = DEFAULT_BOUNDARY,
) -> Path | None:
    """为单个运行目录绘图，保存到 run_dir/figures/overview_2panel.png。

    archive_dir 缺省时优先从 run_dir/manifest.json 的 input_paths.archive_dir 读取，
    回退到 NSW 归档默认路径（保持旧调用兼容）。
    """
    run_dir = Path(run_dir)
    if archive_dir is None:
        archive_dir = _resolve_archive_from_manifest(run_dir) or DEFAULT_ARCHIVE
    archive_dir = Path(archive_dir)
    probs_path = run_dir / "predictions" / "target_probs.csv"
    model_path = run_dir / "models" / next(
        (p.name for p in (run_dir / "models").glob("model_*.pkl")), "model_rf.pkl"
    )
    if not probs_path.is_file() or not model_path.is_file():
        return None

    boundary = _load_boundary(boundary_shp)
    tbl, X549 = build_point_table(archive_dir)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    grid = pd.read_csv(probs_path)
    score_col = next(c for c in grid.columns if c in {"prob", "raw_score", "relative_score"})
    # 分数语义决定 colorbar 标签与归一化范围；只有 probability 固定 [0,1]。
    if score_col == "prob":
        score_label = "预测概率"
        score_vmin, score_vmax = 0.0, 1.0
    elif score_col == "raw_score":
        score_label = "原始决策分数"
        score_vmin = float(grid[score_col].min())
        score_vmax = float(grid[score_col].max())
    else:  # relative_score
        score_label = "相对分数"
        score_vmin = float(grid[score_col].min())
        score_vmax = float(grid[score_col].max())
    arr, extent, x_axis, y_axis = reconstruct_raster(archive_dir, grid[score_col].to_numpy(dtype=float))

    # grid 型模型（cnn2d / label_spreading）predict_proba 只在网格上滑动 → 取最近网格单元的栅格值
    if hasattr(model, "inside_"):
        c_idx = np.argmin(np.abs(x_axis[None, :] - tbl.X.to_numpy()[:, None]), axis=1)
        r_idx = np.argmin(np.abs(y_axis[None, :] - tbl.Y.to_numpy()[:, None]), axis=1)
        point_prob = np.nan_to_num(arr[r_idx, c_idx], nan=0.0)
    else:
        point_prob = model.predict_proba(X549)[:, 1]

    m_train_dep = (tbl.kind == "deposit") & (tbl.split == "train")
    m_train_unl = (tbl.kind == "unlab") & (tbl.split == "train")
    m_test_dep = (tbl.kind == "deposit") & (tbl.split == "test")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    # (c) 训练点概率
    ax = axes[0]
    if boundary is not None:
        boundary.boundary.plot(ax=ax, color="0.25", linewidth=0.8)
        boundary.plot(ax=ax, color="0.90", alpha=0.35, zorder=0)
    sc1 = ax.scatter(
        tbl.X[m_train_unl], tbl.Y[m_train_unl], c=point_prob[m_train_unl],
        cmap=CMAP, vmin=score_vmin, vmax=score_vmax, s=14, marker="o", edgecolors="0.4", linewidths=0.3,
        label=f"训练背景点 ({m_train_unl.sum()})",
    )
    ax.scatter(
        tbl.X[m_train_dep], tbl.Y[m_train_dep], c=point_prob[m_train_dep],
        cmap=CMAP, vmin=score_vmin, vmax=score_vmax, s=40, marker="*", edgecolors="k", linewidths=0.4,
        label=f"训练矿点 ({m_train_dep.sum()})",
    )
    ax.scatter(
        tbl.X[m_test_dep], tbl.Y[m_test_dep], s=60, c="none", marker="*",
        edgecolors="crimson", linewidths=0.9, label=f"测试集已知矿点 ({m_test_dep.sum()})",
    )
    ax.set_title(f"(c) 训练点 + 测试矿点（点色={score_label}）")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(loc="lower right", fontsize=7)
    fig.colorbar(sc1, ax=ax, fraction=0.046, pad=0.04, label=score_label)

    # (d) 全区栅格 + 训练点
    ax = axes[1]
    cmap_bad = plt.get_cmap(CMAP).copy()
    cmap_bad.set_bad(color="white", alpha=0.0)
    im = ax.imshow(ma.masked_invalid(arr), cmap=cmap_bad, extent=extent, origin="lower",
                   vmin=score_vmin, vmax=score_vmax, interpolation="nearest", zorder=1)
    if boundary is not None:
        boundary.plot(ax=ax, color="0.90", alpha=0.5, zorder=0)
        boundary.boundary.plot(ax=ax, color="0.25", linewidth=0.8, zorder=2)
    ax.scatter(tbl.X[m_train_unl], tbl.Y[m_train_unl], c=point_prob[m_train_unl],
               cmap=CMAP, vmin=score_vmin, vmax=score_vmax, s=14, marker="o", edgecolors="white",
               linewidths=0.4, zorder=3, label=f"训练背景点 ({m_train_unl.sum()})")
    ax.scatter(tbl.X[m_train_dep], tbl.Y[m_train_dep], c=point_prob[m_train_dep],
               cmap=CMAP, vmin=score_vmin, vmax=score_vmax, s=42, marker="*", edgecolors="k",
               linewidths=0.4, zorder=4, label=f"训练矿点 ({m_train_dep.sum()})")
    ax.scatter(tbl.X[m_test_dep], tbl.Y[m_test_dep], s=60, c="none", marker="*",
               edgecolors="crimson", linewidths=0.9, zorder=5, label=f"测试集已知矿点 ({m_test_dep.sum()})")
    ax.set_title(f"(d) 训练点/测试矿点 + 全区{score_label}")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(loc="lower right", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=score_label)

    fig.suptitle(f"{run_dir.name}", fontsize=13, fontweight="bold")
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "overview_2panel.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
