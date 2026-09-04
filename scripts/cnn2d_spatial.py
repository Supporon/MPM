#!/usr/bin/env python3
"""空间二维卷积（2D CNN）找矿预测实验。

将整个 NSW 研究区当作一张"图像"：每个 0.1° 格网单元是一个像素，
138 个特征作为通道（channel）。用全卷积 Conv2d 网络学习"矿点/非矿点"
的空间分布，输出整幅找矿远景区概率图。

与现有 1D CNN 的区别：
  - 1D CNN 把 138 个特征当作 1D 信号（丢弃空间位置）；
  - 本 2D CNN 显式保留 2D 空间结构，卷积核在相邻格网上滑动，
    捕捉"相邻单元共享地质/地球物理背景"的空间自相关。

评估：按矿点做 70%/30% 留出，报告留出矿点在概率图中的
AUC 与"前 10%/20% 高概率区捕获率"。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

for _fp in font_manager.findSystemFonts(fontext="ttf"):
    if "CJK" in _fp:
        try:
            font_manager.fontManager.addfont(_fp)
        except Exception:
            pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "EarthByte-MPM_Lachlan_Porphyry" / "Datasets" / "Outputs_Cu_NSW_v1.6"
BOUNDARY_SHP = (
    ROOT.parent / "EarthByte-MPM_Lachlan_Porphyry" / "Datasets" / "Frames"
    / "NSW_Boundary" / "NSW_STATE_POLYGON_shp_GDA94_NoIsland_ACT.shp"
)
OUT = ROOT / "outputs" / "cnn2d_spatial"

SEED = 42
EPOCHS = 250


def set_seed(s: int):
    torch.manual_seed(s)
    np.random.seed(s)


class SpatialCNN2D(nn.Module):
    """全卷积 2D 网络：输入 (C, H, W) → 输出 (1, H, W) 概率图。"""

    def __init__(self, in_ch: int, base: int = 48):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base * 2, base, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base, 1, 1),
        )

    def forward(self, x):
        return self.enc(x)


def build_grid():
    """重建 H×W 格网，返回 (H, W, C) 特征张量、有效掩膜与坐标轴。"""
    mask = pd.read_csv(ARCHIVE / "target_mask.csv", header=None)
    inside = mask[2].astype(str).str.lower().isin(["1", "true", "1.0"]).to_numpy()
    feats = pd.read_csv(ARCHIVE / "target_features.csv").to_numpy(dtype=np.float32)
    x_axis = np.unique(mask[0].to_numpy(dtype=float))
    y_axis = np.unique(mask[1].to_numpy(dtype=float))
    col = np.searchsorted(x_axis, mask[0].to_numpy(dtype=float))
    row = np.searchsorted(y_axis, mask[1].to_numpy(dtype=float))
    H, W = len(y_axis), len(x_axis)
    tensor = np.zeros((H, W, feats.shape[1]), dtype=np.float32)
    tensor[row[inside], col[inside]] = feats  # 外部单元保持 0
    inside_2d = np.zeros((H, W), dtype=bool)
    inside_2d[row[inside], col[inside]] = True
    return tensor, inside_2d, x_axis, y_axis


def snap(coords: pd.DataFrame, x_axis, y_axis):
    """把坐标点吸附到最近格网单元。"""
    x = coords["X"].to_numpy(dtype=float)
    y = coords["Y"].to_numpy(dtype=float)
    c = np.argmin(np.abs(x_axis[None, :] - x[:, None]), axis=1)
    r = np.argmin(np.abs(y_axis[None, :] - y[:, None]), axis=1)
    return r, c


def train(model, tensor, inside, lab, epochs=EPOCHS, lr=1e-3):
    X = torch.from_numpy(tensor.transpose(2, 0, 1))[None]  # (1, C, H, W)
    inside_t = torch.from_numpy(inside)
    target = torch.from_numpy(lab)
    pos = lab[inside].sum()
    neg = inside.sum() - pos
    pos_weight = float(max(1.0, neg / max(pos, 1.0)))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight), reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        opt.zero_grad()
        logit = model(X)[0, 0]
        loss = bce(logit, target)[inside_t].mean()
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1}/{epochs} loss={loss.item():.4f}")
    return pos_weight


@torch.no_grad()
def predict_map(model, tensor):
    X = torch.from_numpy(tensor.transpose(2, 0, 1))[None]
    return torch.sigmoid(model(X)[0, 0]).numpy()


def main():
    set_seed(SEED)
    tensor, inside, x_axis, y_axis = build_grid()
    H, W = inside.shape
    dep = pd.read_csv(ARCHIVE / "training_data_deposit_coords.csv")
    r, c = snap(dep, x_axis, y_axis)
    deposit_cells = np.zeros((H, W), dtype=bool)
    deposit_cells[r, c] = True
    print(f"格网 {H}x{W}，有效单元 {int(inside.sum())}，矿点单元 {int(deposit_cells.sum())}")

    # 矿点 70%/30% 留出（用二维行列索引）
    dep_rc = np.argwhere(deposit_cells)  # (N, 2) = (row, col)
    rng = np.random.RandomState(SEED)
    rng.shuffle(dep_rc)
    n_train = int(0.7 * len(dep_rc))
    train_rc = dep_rc[:n_train]
    hold_rc = dep_rc[n_train:]
    print(f"训练矿点 {len(train_rc)}，留出矿点 {len(hold_rc)}")

    train_lab = np.zeros((H, W), dtype=np.float32)
    train_lab[train_rc[:, 0], train_rc[:, 1]] = 1.0

    model = SpatialCNN2D(in_ch=tensor.shape[2])
    print("训练 2D CNN ...")
    pos_weight = train(model, tensor, inside, train_lab)

    prob = predict_map(model, tensor)

    # 留出评估：留出矿点单元概率 vs 其它非矿点有效单元概率
    other_cells = inside & (~deposit_cells)
    pos_scores = prob[hold_rc[:, 0], hold_rc[:, 1]]
    neg_scores = prob[other_cells]
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    s = np.concatenate([pos_scores, neg_scores])
    auc = roc_auc_score(y, s)

    inside_flat = prob[inside]
    def capture(frac):
        thr = np.quantile(inside_flat, 1 - frac)
        return float((pos_scores >= thr).mean())

    cap10 = capture(0.10)
    cap20 = capture(0.20)

    print(f"\n=== 2D CNN 留出评估（pos_weight={pos_weight:.1f}）===")
    print(f"AUC (留出矿点 vs 非矿点单元): {auc:.4f}")
    print(f"前 10% 高概率区捕获率: {cap10:.3f}")
    print(f"前 20% 高概率区捕获率: {cap20:.3f}")

    # 保存
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "prob_map.npy", prob)
    np.save(OUT / "label_map.npy", train_lab)
    summary = {
        "grid": [H, W], "inside": int(inside.sum()), "deposit_cells": int(deposit_cells.sum()),
        "train_dep": len(train_rc), "hold_dep": len(hold_rc),
        "pos_weight": pos_weight, "auc": auc, "capture_top10": cap10, "capture_top20": cap20,
    }
    import json
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 绘图
    extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
    hold_y = y_axis[hold_rc[:, 0]]
    hold_x = x_axis[hold_rc[:, 1]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    im0 = axes[0].imshow(train_lab, cmap="Reds", extent=extent, origin="lower")
    axes[0].set_title(f"(a) 训练矿点标签 (n={len(train_rc)})")
    im1 = axes[1].imshow(prob, cmap="viridis", extent=extent, origin="lower", vmin=0, vmax=1)
    axes[1].scatter(hold_x, hold_y, s=18, c="none", edgecolors="red", linewidths=1.0,
                    label=f"留出矿点 (n={len(hold_rc)})")
    axes[1].set_title("(b) 2D CNN 找矿远景区概率")
    axes[1].legend(loc="lower right", fontsize=7)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label="概率")
    axes[2].hist(prob[inside].ravel(), bins=60, color="0.6", alpha=0.7, label="全部有效单元")
    axes[2].hist(pos_scores, bins=60, color="crimson", alpha=0.7, label="留出矿点")
    axes[2].set_title(f"(c) 概率分布\nAUC={auc:.3f}, 前10%捕获={cap10:.2f}")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("经度 (°E)")
        ax.set_ylabel("纬度 (°N)")
    fig.savefig(OUT / "cnn2d_result.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"结果与图已写入 {OUT}")


if __name__ == "__main__":
    main()
