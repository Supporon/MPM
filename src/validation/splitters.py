"""通过统一 Splitter 注册表选择留出法与交叉验证策略。"""

from __future__ import annotations

import warnings
from inspect import signature
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

from ..core.contracts import SplitData, TrainingData
from .registry import SPLITTER_REGISTRY


@SPLITTER_REGISTRY.decorator("none")
class NoCrossValidationSplitter:
    """禁用独立评价 CV 的占位 splitter。

    ``Experiment._run_independent_cv`` 在 ``cross_validation.name == "none"`` 时
    直接跳过，本 splitter 仅用于让 ``validate_config`` 的 ``require`` 通过，避免
    文档承诺的关闭选项在配置层被拒绝（第 7 节局部功能缺口）。任何需要真实 CV
    对象的调用（如 ``bayes`` 调参器）都应在配置层与 ``none`` 组合被拒绝，或在
    此处得到明确错误。
    """

    kind = "cross_validation"

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        pass

    def build_cv(self, params: Mapping[str, Any], seed: int, data: TrainingData | None = None):
        raise ValueError(
            "cross_validation.name='none' disables cross-validation; no CV splitter "
            "can be built. Do not combine it with a CV-based tuner."
        )


@SPLITTER_REGISTRY.decorator("random_holdout")
class RandomHoldoutSplitter:
    kind = "holdout"

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        test_size = float(params.get("test_size", 0.25))
        if not 0 < test_size < 1:
            raise ValueError("random_holdout test_size must be in (0, 1)")

    def split(self, data: TrainingData, params: Mapping[str, Any], seed: int) -> SplitData:
        warnings.warn(
            "Baseline compatibility: using a random point-level train/test split; spatial leakage remains possible.",
            UserWarning,
            stacklevel=2,
        )
        indices = list(range(len(data.features)))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=float(params.get("test_size", 0.25)),
            random_state=seed,
            stratify=None,
        )
        return SplitData(_take(data, train_idx), _take(data, test_idx))


@SPLITTER_REGISTRY.decorator("stratified_kfold")
class StratifiedKFoldSplitter:
    kind = "cross_validation"

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        if int(params.get("n_splits", 5)) < 2:
            raise ValueError("stratified_kfold n_splits must be at least 2")

    def build_cv(self, params: Mapping[str, Any], seed: int, data: TrainingData | None = None):
        shuffle = bool(params.get("shuffle", False))
        kwargs = {
            "n_splits": int(params.get("n_splits", 5)),
            "shuffle": shuffle,
        }
        if shuffle:
            kwargs["random_state"] = seed
        return StratifiedKFold(**kwargs)


# ============================================================================
# 空间交叉验证 — 空间块 K-Fold
# ============================================================================


def _extract_coords_from_data(data: TrainingData) -> np.ndarray:
    """从 TrainingData 的 metadata["units"] 中提取 X/Y 坐标。

    Returns:
        shape (N, 2) 的坐标数组 [X, Y]。
    Raises:
        ValueError: 如果缺少坐标元数据。
    """
    units = data.metadata.get("units") if data.metadata else None
    if units is None:
        raise ValueError(
            "Spatial splitter requires coordinate metadata. "
            "Ensure TrainingData.metadata['units'] contains X/Y columns."
        )
    if isinstance(units, pd.DataFrame):
        x_col = next((c for c in units.columns if c.upper() == "X"), None)
        y_col = next((c for c in units.columns if c.upper() == "Y"), None)
        if x_col and y_col:
            return np.column_stack([units[x_col].to_numpy(), units[y_col].to_numpy()])
    raise ValueError(
        "Spatial splitter requires X/Y columns in the units metadata DataFrame."
    )


def _looks_like_degrees(x: np.ndarray, y: np.ndarray) -> bool:
    """启发式判断坐标是否为十进制经纬度（度）。

    投影坐标（如 UTM，单位米）通常 x 可达数十万/数百万、或超出
    [-180, 180]，而经纬度 x∈[-180,180]、y∈[-90,90]。用于在
    空间分块前给出明确错误，避免以米为单位的 block_size 把全部
    经纬度样本归入同一个块。
    """
    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    if len(finite_x) == 0 or len(finite_y) == 0:
        return False
    return bool(
        finite_x.min() >= -180
        and finite_x.max() <= 180
        and finite_y.min() >= -90
        and finite_y.max() <= 90
    )


def _assign_spatial_blocks(
    coords: np.ndarray, block_size_m: float
) -> np.ndarray:
    """将样本按空间坐标分配到网格块中。

    每个样本被分配到一个整数 block_id，同一 block_id 的样本
    在空间上属于同一个网格块。

    Args:
        coords: shape (N, 2) 的坐标数组 [X, Y]。
        block_size_m: 网格块边长（米）。

    Returns:
        shape (N,) 的整数 block_id 数组。
    """
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    if _looks_like_degrees(x_coords, y_coords):
        raise ValueError(
            "spatial block splitting requires projected coordinates (meters), "
            "but X/Y appear to be in decimal degrees. Reproject the data to a "
            "metric CRS (e.g. UTM) before using spatial_block_kfold."
        )

    # 计算每个样本所属的网格块索引
    x_block = np.floor((x_coords - x_coords.min()) / block_size_m).astype(int)
    y_block = np.floor((y_coords - y_coords.min()) / block_size_m).astype(int)

    # 将二维块索引编码为一维 block_id
    max_x = x_block.max() + 1
    block_ids = y_block * max_x + x_block
    return block_ids


class _GroupKFoldWithGroups:
    """GroupKFold 包装器，把预计算的 group 数组固化。

    调用方（如 BayesSearchCV）通常只调用 ``split(X, y)`` 而不传
    ``groups``，本包装器在内部注入固化的 group 数组，避免
    ``The 'groups' parameter should not be None`` 错误。
    """

    def __init__(self, n_splits: int, groups: np.ndarray, shuffle: bool, random_state: int | None):
        self.n_splits = n_splits
        self.groups = np.asarray(groups)
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """生成 (train_idx, test_idx) 迭代器。"""
        gkf = GroupKFold(n_splits=self.n_splits)
        resolved = self.groups
        # 如果 shuffle，先对块进行随机排列
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            unique = np.unique(self.groups)
            shuffled = rng.permutation(unique)
            mapping = {old: new for new, old in enumerate(shuffled)}
            resolved = np.array([mapping[g] for g in self.groups])
        yield from gkf.split(X, y, groups=resolved)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


@SPLITTER_REGISTRY.decorator("spatial_block_kfold")
class SpatialBlockKFoldSplitter:
    """空间块 K-Fold 交叉验证。

    基于空间坐标将样本分配到网格块中，使用 GroupKFold 按块划分，
    确保每个 fold 的 train/test 在空间上分离。

    参数:
        n_splits: fold 数量，默认 5。
        block_size_m: 空间块边长（米），默认 50000。
        shuffle: 是否在划分前随机排列块，默认 True。
    """

    kind = "cross_validation"

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        if int(params.get("n_splits", 5)) < 2:
            raise ValueError("spatial_block_kfold n_splits must be at least 2")
        if float(params.get("block_size_m", 50000)) <= 0:
            raise ValueError("spatial_block_kfold block_size_m must be positive")

    def build_cv(self, params: Mapping[str, Any], seed: int, data: TrainingData | None = None):
        if data is None:
            raise ValueError(
                "spatial_block_kfold requires TrainingData to extract spatial coordinates. "
                "Pass data= to build_cv()."
            )
        n_splits = int(params.get("n_splits", 5))
        block_size_m = float(params.get("block_size_m", 50000))
        shuffle = bool(params.get("shuffle", True))

        coords = _extract_coords_from_data(data)
        block_ids = _assign_spatial_blocks(coords, block_size_m)

        unique_blocks = len(np.unique(block_ids))
        if unique_blocks < 2:
            raise ValueError(
                f"spatial_block_kfold found only {unique_blocks} spatial block. "
                "Reduce block_size_m, or ensure coordinates are in a projected "
                "(metric) CRS so samples span more than one block."
            )
        if unique_blocks < n_splits:
            warnings.warn(
                f"spatial_block_kfold: only {unique_blocks} spatial blocks found, "
                f"but n_splits={n_splits}. Reducing n_splits to {unique_blocks}.",
                UserWarning,
                stacklevel=2,
            )
            n_splits = unique_blocks

        return _GroupKFoldWithGroups(
            n_splits=n_splits,
            groups=block_ids,
            shuffle=shuffle,
            random_state=seed,
        )


# ============================================================================
# 空间分组 K-Fold — 基于已有的 group_id
# ============================================================================


@SPLITTER_REGISTRY.decorator("spatial_group_kfold")
class SpatialGroupKFoldSplitter:
    """基于已有分组 ID 的空间交叉验证。

    使用 TrainingData.metadata 中已有的 group_id（如矿集区/地质域）
    作为分组依据，通过 GroupKFold 进行划分。

    参数:
        n_splits: fold 数量，默认 5。
        group_column: metadata 中分组列的列名，默认 "group_id"。
    """

    kind = "cross_validation"

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        if int(params.get("n_splits", 5)) < 2:
            raise ValueError("spatial_group_kfold n_splits must be at least 2")

    def build_cv(self, params: Mapping[str, Any], seed: int, data: TrainingData | None = None):
        if data is None:
            raise ValueError(
                "spatial_group_kfold requires TrainingData with group metadata. "
                "Pass data= to build_cv()."
            )
        n_splits = int(params.get("n_splits", 5))
        group_column = str(params.get("group_column", "group_id"))

        units = data.metadata.get("units") if data.metadata else None
        if units is None or group_column not in units.columns:
            raise ValueError(
                f"spatial_group_kfold requires '{group_column}' column in "
                "TrainingData.metadata['units'] DataFrame."
            )

        groups = units[group_column].to_numpy()
        unique_groups = len(np.unique(groups))
        if unique_groups < 2:
            raise ValueError(
                f"spatial_group_kfold found only {unique_groups} group(s) in "
                f"column '{group_column}'; need at least 2 groups to split."
            )
        if unique_groups < n_splits:
            warnings.warn(
                f"spatial_group_kfold: only {unique_groups} groups found, "
                f"but n_splits={n_splits}. Reducing n_splits to {unique_groups}.",
                UserWarning,
                stacklevel=2,
            )
            n_splits = unique_groups

        return _GroupKFoldWithGroups(
            n_splits=n_splits,
            groups=groups,
            shuffle=False,
            random_state=seed,
        )


# ============================================================================
# 空间块留出法
# ============================================================================


@SPLITTER_REGISTRY.decorator("spatial_block_holdout")
class SpatialBlockHoldoutSplitter:
    """基于空间块的留出法划分。

    将样本按空间坐标分配到网格块中，按块进行留出法划分，
    确保测试集在空间上与训练集分离。

    参数:
        test_size: 测试集所占块的比例，默认 0.25。
        block_size_m: 空间块边长（米），默认 50000。
    """

    kind = "holdout"

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        test_size = float(params.get("test_size", 0.25))
        if not 0 < test_size < 1:
            raise ValueError("spatial_block_holdout test_size must be in (0, 1)")
        if float(params.get("block_size_m", 50000)) <= 0:
            raise ValueError("spatial_block_holdout block_size_m must be positive")

    def split(self, data: TrainingData, params: Mapping[str, Any], seed: int) -> SplitData:
        test_size = float(params.get("test_size", 0.25))
        block_size_m = float(params.get("block_size_m", 50000))

        coords = _extract_coords_from_data(data)
        block_ids = _assign_spatial_blocks(coords, block_size_m)

        unique_blocks = np.unique(block_ids)
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_blocks)

        n_test_blocks = max(1, int(len(unique_blocks) * test_size))
        test_blocks = set(unique_blocks[:n_test_blocks])

        test_idx = [i for i, b in enumerate(block_ids) if b in test_blocks]
        train_idx = [i for i, b in enumerate(block_ids) if b not in test_blocks]

        if len(test_idx) == 0:
            raise ValueError(
                "spatial_block_holdout: no samples in test set. "
                "Try reducing block_size_m or test_size."
            )
        if len(train_idx) == 0:
            raise ValueError(
                "spatial_block_holdout: no samples in train set. "
                "Try reducing block_size_m or test_size."
            )

        return SplitData(_take(data, train_idx), _take(data, test_idx))


def _take(data: TrainingData, indices: list[int]) -> TrainingData:
    return data.take(indices)


def create_holdout(name: str, params: Mapping[str, Any], seed: int):
    splitter = SPLITTER_REGISTRY.create(name)
    if getattr(splitter, "kind", None) != "holdout":
        raise ValueError(f"Splitter '{name}' is not a holdout splitter")
    return splitter


def build_cv(name: str, params: Mapping[str, Any], seed: int, data: TrainingData | None = None):
    splitter = SPLITTER_REGISTRY.create(name)
    if getattr(splitter, "kind", None) != "cross_validation":
        raise ValueError(f"Splitter '{name}' is not a cross-validation splitter")
    build_parameters = signature(splitter.build_cv).parameters
    if "data" in build_parameters or len(build_parameters) >= 3:
        return splitter.build_cv(params, seed, data)
    return splitter.build_cv(params, seed)


def random_split(features, labels, sample_weight, test_size: float, seed: int):
    """向后兼容一期接口的辅助类，对外保留旧字段名。"""
    data = TrainingData(
        features.reset_index(drop=True), labels.reset_index(drop=True), sample_weight.reset_index(drop=True)
    )
    split = RandomHoldoutSplitter().split(data, {"test_size": test_size}, seed)

    class LegacySplit:
        x_train = split.train.features
        x_test = split.test.features
        y_train = split.train.labels
        y_test = split.test.labels
        weight_train = split.train.sample_weight
        weight_test = split.test.sample_weight

    return LegacySplit()
