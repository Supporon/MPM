#!/usr/bin/env python3
"""基于 patch 的 2D CNN 找矿预测（解决全图 N=1 问题）。

之前的全图 2D CNN 只有"一张图"作为样本（N=1），本质是空间插值。本脚本改为
**patch 训练**：围绕每个训练点（矿点/背景点）取一个 P×P 的局部空间窗口作为样本，
用卷积网络判断窗口中心是否矿化。这样样本量 = 训练点数（359），是真正的监督学习，
且每个样本都携带了空间邻域（地质/地球物理背景）。

训练/测试沿用归档的 Xy_rf_train / Xy_rf_test 划分（359/120）。
评估：测试集 patch 的 AUC/F1 等，以及整幅滑动得到的找矿远景区图上
留出矿点的"前 10%/20% 捕获率"。
"""

from __future__ import annotations

import sys
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

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.archive_coords import match_rows, reconstruct_archive_points

ARCHIVE = ROOT.parent / "EarthByte-MPM_Lachlan_Porphyry" / "Datasets" / "Outputs_Cu_NSW_v1.6"
OUT = ROOT / "outputs" / "cnn2d_patch"

SEED = 42
PATCH = 15          # 15×15 单元 ≈ 165 km 空间上下文
EPOCHS = 150
LR = 1e-3


def set_seed(s: int):
    torch.manual_seed(s)
    np.random.seed(s)


class PatchCNN(nn.Module):
    """输入 (B, C, P, P) → 输出 (B,) 二分类 logit。"""

    def __init__(self, in_ch: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        h = self.features(x).flatten(1)
        return self.classifier(h).squeeze(-1)


def build_grid():
    """重建 (H, W, C) 特征张量与有效掩膜、坐标轴。"""
    mask = pd.read_csv(ARCHIVE / "target_mask.csv", header=None)
    inside = mask[2].astype(str).str.lower().isin(["1", "true", "1.0"]).to_numpy()
    feats = pd.read_csv(ARCHIVE / "target_features.csv").to_numpy(dtype=np.float32)
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


def snap(X, Y, x_axis, y_axis):
    c = np.argmin(np.abs(x_axis[None, :] - np.asarray(X)[:, None]), axis=1)
    r = np.argmin(np.abs(y_axis[None, :] - np.asarray(Y)[:, None]), axis=1)
    return r, c


def extract_patches(grid, cells, patch=PATCH):
    """cells: [(r, c)] → 返回 (N, C, P, P)。reflect 填充边缘。"""
    C, H, W = grid.shape
    pad = patch // 2
    padded = np.pad(grid, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    patches = np.stack([padded[:, r:r + patch, c:c + patch] for r, c in cells], axis=0)
    return patches


def train(model, X, y, epochs=EPOCHS, lr=LR):
    pos = int(y.sum())
    neg = len(y) - pos
    pos_weight = torch.tensor(neg / max(pos, 1))
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        # 简单数据增强：随机水平/垂直翻转
        x = Xt
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[3])
        loss = bce(model(x), yt)
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1}/{epochs} loss={loss.item():.4f}")
    return pos_weight.item()


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = []
    for i in range(0, len(X), 256):
        out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + 256]).float())).numpy())
    return np.concatenate(out)


def main():
    set_seed(SEED)
    grid, inside, x_axis, y_axis = build_grid()
    grid_t = grid.transpose(2, 0, 1)  # (C, H, W)

    coords, X549 = reconstruct_archive_points(ARCHIVE)
    xy_rf_train = pd.read_csv(ARCHIVE / "Xy_rf_train.csv")
    xy_rf_test = pd.read_csv(ARCHIVE / "Xy_rf_test.csv")
    fcols = [c for c in xy_rf_train.columns if c not in ("sample_weight", "label")]
    train_idx = match_rows(X549, xy_rf_train[fcols])
    test_idx = match_rows(X549, xy_rf_test[fcols])

    kind = np.array(["deposit"] * 277 + ["unlab"] * 272)
    y_full = (kind == "deposit").astype(np.float32)
    r, c = snap(coords["X"], coords["Y"], x_axis, y_axis)
    cells = list(zip(r.tolist(), c.tolist()))

    print(f"训练点 {len(train_idx)}，测试点 {len(test_idx)}，patch {PATCH}×{PATCH}")
    X_train = extract_patches(grid_t, [cells[i] for i in train_idx])
    y_train = y_full[train_idx]
    X_test = extract_patches(grid_t, [cells[i] for i in test_idx])
    y_test = y_full[test_idx]

    model = PatchCNN(in_ch=grid_t.shape[0])
    print("训练 patch CNN ...")
    pos_weight = train(model, X_train, y_train)

    # 测试集 patch 评估
    p_test = predict(model, X_test)
    thr = 0.5
    pred = (p_test >= thr).astype(int)
    auc = roc_auc_score(y_test, p_test)
    f1 = f1_score(y_test, pred)
    print(f"\n=== patch CNN 测试集 ({len(y_test)} 点) ===")
    print(f"AUC={auc:.4f}  F1={f1:.4f}  Precision={precision_score(y_test, pred):.4f}  "
          f"Recall={recall_score(y_test, pred):.4f}  Acc={accuracy_score(y_test, pred):.4f}")

    # 整幅滑动 → 找矿远景区图 → 留出矿点捕获率
    inside_cells = list(zip(*np.where(inside)))
    X_map = extract_patches(grid_t, inside_cells)
    p_map = predict(model, X_map)
    prob_map = np.full(inside.shape, np.nan, dtype=np.float32)
    prob_map[inside] = p_map

    test_dep_cells = [cells[i] for i in test_idx if kind[i] == "deposit"]
    pos_scores = np.array([prob_map[rr, cc] for rr, cc in test_dep_cells])
    inside_flat = prob_map[inside]
    def capture(frac):
        thr = np.quantile(inside_flat, 1 - frac)
        return float((pos_scores >= thr).mean())
    print(f"留出矿点 {len(pos_scores)} 个：前10%捕获率={capture(0.10):.3f}，前20%捕获率={capture(0.20):.3f}")

    # 保存
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "prob_map.npy", prob_map)
    import json
    (OUT / "metrics.json").write_text(json.dumps({
        "patch": PATCH, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "pos_weight": pos_weight, "auc": auc, "f1": f1,
        "capture_top10": capture(0.10), "capture_top20": capture(0.20),
    }, indent=2), encoding="utf-8")

    # 图
    extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
    hold_rc = np.array(test_dep_cells)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    im = axes[0].imshow(prob_map, cmap="viridis", extent=extent, origin="lower", vmin=0, vmax=1)
    if len(hold_rc):
        axes[0].scatter(x_axis[hold_rc[:, 1]], y_axis[hold_rc[:, 0]], s=50, c="none",
                        edgecolors="red", linewidths=1.0, label=f"留出矿点 ({len(hold_rc)})")
    axes[0].set_title("patch CNN 找矿远景区概率")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="概率")
    axes[1].hist(inside_flat, bins=60, color="0.6", alpha=0.7, label="全部有效单元")
    axes[1].hist(pos_scores, bins=60, color="crimson", alpha=0.7, label="留出矿点")
    axes[1].set_title(f"概率分布\nAUC={auc:.3f} 前10%捕获={capture(0.10):.2f}")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    fig.savefig(OUT / "cnn2d_patch_result.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"结果与图已写入 {OUT}")


if __name__ == "__main__":
    main()
