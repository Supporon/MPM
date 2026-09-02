"""确定性的随机种子派生，避免各组件共用同一个随机种子。

同一个实验用不同随机种子重复运行时，采样、划分、调参、模型、数据加载
各阶段应互不干扰：改变某阶段的种子不应改变其它阶段。这里用稳定哈希
从基础种子为各用途派生独立种子，保证同 base_seed 的派生结果完全一致。
"""

from __future__ import annotations

import hashlib

# 参与派生的用途。顺序固定，保证派生结果稳定。
_SEED_PURPOSES: tuple[str, ...] = (
    "sampling",
    "split",
    "tuning",
    "model",
    "dataloader",
)


def derive_seed(base_seed: int, purpose: str) -> int:
    """从基础种子为单个用途派生一个非负 32 位种子。"""
    digest = hashlib.sha256(f"{base_seed}:{purpose}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def derive_seeds(base_seed: int) -> dict[str, int]:
    """从基础种子派生各用途种子，返回 ``{"base_seed": ..., "<purpose>_seed": ...}``。"""
    seeds = {"base_seed": int(base_seed)}
    for purpose in _SEED_PURPOSES:
        seeds[f"{purpose}_seed"] = derive_seed(base_seed, purpose)
    return seeds
