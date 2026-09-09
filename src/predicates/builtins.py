"""LUSI 风格的内置 Predicate 实现。

谓词按 kind 分为两类：

A. **constraint 谓词**（kind="constraint"）— 生成 φ 向量用于 LUSI 加权损失
   1. **all_ones** (Φ_ones 控制谓词):
      - 为所有样本生成 φ = 1 的向量
      - 对应 LUSI 中的全 1 控制谓词
      - 用于验证加权损失框架的正确性

   2. **spatial_box** (空间选框谓词):
      - 在当前研究区域内框选一个正方形区域
      - 将正方形内的样本 φ 值设置为指定值（默认 1.0）
      - 其余样本 φ 值设置为 0
      - 用于捕获空间局域内的统计不变量

   3. **spatial_distance** (空间距离谓词):
      - 基于样本到参考点的距离构造 φ 向量
      - 使用高斯核将距离映射到 (0, 1] 区间

   4. **combined** (组合谓词):
      - 将多个子谓词的 φ 向量拼接为多维 φ 矩阵

B. **data_transform 谓词**（kind="data_transform"）— 修改标签/权重
   当前内置暂无 data_transform 谓词。
   示例：positive_constraint（筛选特定地质特征）、weight_adjustment（调整权重）

管线执行顺序：data_transform 先于 constraint，
确保约束基于最终变换后的数据生成。

数学原理（参考 Vapnik & Izmailov 的 LUSI；实现参考 TSIL）:
    φ 向量 → L2 归一化 φ̃ = φ / ||φ||
    P = φ̃ φ̃ᵀ（投影矩阵）
    loss = τ̂ · MSE + τ · (1/N) · ||φ̃ᵀ e||²

P 矩阵将具有相似谓词特征的样本聚合在一起，在损失计算中引入
样本间的统计不变量。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from ..core.contracts import TrainingData
from .registry import PREDICATE_REGISTRY


# ============================================================================
# 辅助函数
# ============================================================================


def _get_coordinates(data: TrainingData) -> np.ndarray | None:
    """从 TrainingData 中提取 X/Y 坐标。

    优先从 metadata 中的 units DataFrame 提取，否则从 features 列名中推断。

    Returns:
        shape (N, 2) 的坐标数组 [X, Y]，或 None
    """
    # 方法 1: 从 metadata 的 units 中提取
    units = data.metadata.get("units")
    if units is not None:
        import pandas as pd

        if isinstance(units, pd.DataFrame):
            x_col = next((c for c in units.columns if c.upper() == "X"), None)
            y_col = next((c for c in units.columns if c.upper() == "Y"), None)
            if x_col and y_col:
                return np.column_stack(
                    [units[x_col].to_numpy(), units[y_col].to_numpy()]
                )

    # 方法 2: 从 features 列名中推断坐标
    features = data.features
    x_col = next((c for c in features.columns if c.upper() == "X"), None)
    y_col = next((c for c in features.columns if c.upper() == "Y"), None)
    if x_col and y_col:
        return np.column_stack(
            [features[x_col].to_numpy(), features[y_col].to_numpy()]
        )

    return None


# ============================================================================
# 1. All-Ones Predicate (Φ_ones)
# ============================================================================


@PREDICATE_REGISTRY.decorator("all_ones")
class AllOnesPredicate:
    """全 1 谓词（Φ_ones 控制谓词）。

    为所有样本生成 φ = 1 的向量。这是 LUSI 中的控制谓词，
    用于验证加权损失框架的正确性。

    当 φ = [1, 1, ..., 1]ᵀ 时：
        φ̃ = 1/√N · [1, 1, ..., 1]ᵀ
        P = φ̃ φ̃ᵀ = (1/N) · 1_{N×N}（全 1 矩阵）
        P_loss = (1/N) · ||φ̃ᵀ e||² = (1/N²) · (Σ e_i)²

    这意味着 P 项惩罚的是所有样本残差之和的平方，
    即鼓励模型在整个批次上保持无偏预测。

    Parameters
    ----------
    params : Mapping[str, Any]
        可选的谓词参数，当前为空。
    """

    name = "all_ones"
    kind = "constraint"

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)

    def apply(self, data: TrainingData, context: Mapping[str, Any]) -> TrainingData:
        n_samples = len(data.features)
        phi_vector = np.ones(n_samples, dtype=np.float32)

        return data.with_constraints(
            {"phi_vector": phi_vector},
            predicates_applied=list(
                data.metadata.get("predicates_applied", [])
            ) + [self.name],
        )


# ============================================================================
# 2. Spatial Box Predicate (空间选框谓词)
# ============================================================================


@PREDICATE_REGISTRY.decorator("spatial_box")
class SpatialBoxPredicate:
    """空间选框谓词。

    在当前研究区域内框选一个正方形区域，将正方形内的样本 φ 值
    设置为指定值（默认 1.0），其余样本设置为 0。

    可用于：
    - 捕获空间局域内的统计不变量
    - 将空间上相近的样本聚合在一起
    - 验证模型对空间结构的敏感性

    Parameters
    ----------
    params : Mapping[str, Any]
        谓词参数，支持以下键：
        - center_x : float, 选框中心 X 坐标（默认：数据范围中心）
        - center_y : float, 选框中心 Y 坐标（默认：数据范围中心）
        - side_length : float, 选框边长（默认：数据范围的一半）
        - inner_value : float, 选框内样本的 φ 值（默认 1.0）
        - outer_value : float, 选框外样本的 φ 值（默认 0.0）
        - auto_center : bool, 是否自动计算中心（默认 True）
        - auto_side : bool, 是否自动计算边长（默认 True）
    """

    name = "spatial_box"
    kind = "constraint"

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)
        self.center_x = float(self.params.get("center_x", 0.0))
        self.center_y = float(self.params.get("center_y", 0.0))
        self.side_length = float(self.params.get("side_length", 0.0))
        self.inner_value = float(self.params.get("inner_value", 1.0))
        self.outer_value = float(self.params.get("outer_value", 0.0))
        self.auto_center = bool(self.params.get("auto_center", True))
        self.auto_side = bool(self.params.get("auto_side", True))

    def apply(self, data: TrainingData, context: Mapping[str, Any]) -> TrainingData:
        coords = _get_coordinates(data)
        if coords is None:
            raise ValueError(
                "spatial_box predicate requires X/Y coordinates in the data. "
                "Ensure the TrainingData has 'units' metadata or X/Y feature columns."
            )

        n_samples = len(data.features)
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]

        # 优先从知识上下文消费 spatial_extent 的研究区域先验范围，
        # 否则回退到从当前数据坐标重新估计。仅在需要自动计算中心/边长时读取
        # 知识：手动指定 auto_center=False 且 auto_side=False 时，知识不影响 phi，
        # 不应把 knowledge 记为已消费（P1-03：accessed 与 used 语义区分）。
        extent = None
        if self.auto_center or self.auto_side:
            knowledge = context.get("knowledge", {})
            spatial_knowledge = knowledge.get("spatial_extent", {})
            if isinstance(spatial_knowledge, Mapping) and "bounds" in spatial_knowledge:
                extent = spatial_knowledge

        # 自动计算中心和边长
        if self.auto_center:
            if extent is not None and "center" in extent:
                center_x = float(extent["center"]["x"])
                center_y = float(extent["center"]["y"])
            else:
                center_x = float(np.median(x_coords))
                center_y = float(np.median(y_coords))
        else:
            center_x = self.center_x
            center_y = self.center_y

        if self.auto_side:
            if extent is not None and "bounds" in extent:
                bounds = extent["bounds"]
                x_range = bounds["max_x"] - bounds["min_x"]
                y_range = bounds["max_y"] - bounds["min_y"]
                side_length = float(min(x_range, y_range) * 0.5)
            else:
                x_range = x_coords.max() - x_coords.min()
                y_range = y_coords.max() - y_coords.min()
                side_length = float(min(x_range, y_range) * 0.5)
        else:
            side_length = self.side_length

        if side_length <= 0:
            raise ValueError(
                f"spatial_box side_length must be positive, got {side_length}"
            )

        # 计算选框边界
        half_side = side_length / 2.0
        x_min = center_x - half_side
        x_max = center_x + half_side
        y_min = center_y - half_side
        y_max = center_y + half_side

        # 判断每个样本是否在选框内
        in_box = (x_coords >= x_min) & (x_coords <= x_max) & (
            y_coords >= y_min
        ) & (y_coords <= y_max)

        phi_vector = np.where(in_box, self.inner_value, self.outer_value).astype(
            np.float32
        )

        # 记录谓词元数据
        predicate_meta = {
            "center_x": center_x,
            "center_y": center_y,
            "side_length": side_length,
            "inner_value": self.inner_value,
            "outer_value": self.outer_value,
            "n_in_box": int(in_box.sum()),
            "n_total": n_samples,
            "fraction_in_box": float(in_box.sum() / n_samples),
        }

        return data.with_constraints(
            {"phi_vector": phi_vector},
            predicates_applied=list(
                data.metadata.get("predicates_applied", [])
            ) + [self.name],
            spatial_box=predicate_meta,
        )


# ============================================================================
# 3. Spatial Distance Predicate (空间距离谓词)
# ============================================================================


@PREDICATE_REGISTRY.decorator("spatial_distance")
class SpatialDistancePredicate:
    """空间距离谓词。

    基于样本到参考点的距离构造 φ 向量。距离越近的样本 φ 值越大。
    使用高斯核将距离映射到 (0, 1] 区间。

    可用于：
    - 捕获空间自相关效应
    - 将空间上邻近的样本聚合在一起

    Parameters
    ----------
    params : Mapping[str, Any]
        - reference_x : float, 参考点 X 坐标
        - reference_y : float, 参考点 Y 坐标
        - sigma : float, 高斯核带宽（默认：数据范围的 1/10）
        - auto_reference : bool, 是否使用数据中心作为参考点（默认 True）
    """

    name = "spatial_distance"
    kind = "constraint"

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)
        self.reference_x = float(self.params.get("reference_x", 0.0))
        self.reference_y = float(self.params.get("reference_y", 0.0))
        self.sigma = float(self.params.get("sigma", 0.0))
        self.auto_reference = bool(self.params.get("auto_reference", True))

    def apply(self, data: TrainingData, context: Mapping[str, Any]) -> TrainingData:
        coords = _get_coordinates(data)
        if coords is None:
            raise ValueError(
                "spatial_distance predicate requires X/Y coordinates in the data."
            )

        x_coords = coords[:, 0]
        y_coords = coords[:, 1]

        if self.auto_reference:
            ref_x = float(np.median(x_coords))
            ref_y = float(np.median(y_coords))
        else:
            ref_x = self.reference_x
            ref_y = self.reference_y

        if self.sigma <= 0:
            x_range = x_coords.max() - x_coords.min()
            y_range = y_coords.max() - y_coords.min()
            sigma = float(max(x_range, y_range) * 0.1)
        else:
            sigma = self.sigma

        # 欧氏距离
        dist = np.sqrt((x_coords - ref_x) ** 2 + (y_coords - ref_y) ** 2)

        # 高斯核: φ = exp(-d² / (2σ²))
        phi_vector = np.exp(-(dist**2) / (2 * sigma**2)).astype(np.float32)

        return data.with_constraints(
            {"phi_vector": phi_vector},
            predicates_applied=list(
                data.metadata.get("predicates_applied", [])
            ) + [self.name],
            spatial_distance={
                "reference_x": ref_x,
                "reference_y": ref_y,
                "sigma": sigma,
            },
        )


# ============================================================================
# 4. Combined Predicate (组合谓词)
# ============================================================================


@PREDICATE_REGISTRY.decorator("combined")
class CombinedPredicate:
    """组合多个 φ 向量的谓词。

    将多个子谓词的 φ 向量拼接为多维 φ 矩阵 [N, D]，
    每个子谓词生成一个维度。在损失计算中，所有维度同时参与
    φ̃ᵀ e 的计算，即 P_loss = (1/N) · Σ_d (φ̃_{:,d} · e)²。

    Parameters
    ----------
    params : Mapping[str, Any]
        - predicates : list[dict], 子谓词配置列表，
          每个元素包含 name 和 params
    """

    name = "combined"
    kind = "constraint"

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)
        self.sub_predicates: list[dict] = list(
            self.params.get("predicates", [])
        )

    def apply(self, data: TrainingData, context: Mapping[str, Any]) -> TrainingData:
        if not self.sub_predicates:
            # 无子谓词时，默认使用全 1 谓词
            phi_vector = np.ones(len(data.features), dtype=np.float32)
            return data.with_constraints(
                {"phi_vectors": [{"name": self.name, "vector": phi_vector}]},
                predicates_applied=list(
                    data.metadata.get("predicates_applied", [])
                ) + [self.name],
            )

        phi_vectors = []
        # 子谓词必须在去除历史 phi 载荷的干净副本上独立计算，避免把
        # 上一级已写入的 phi_vector/phi_vectors 当作子谓词自身的输出收集。
        clean_constraints = {
            key: value
            for key, value in data.constraints.items()
            if key not in {"phi_vector", "phi_vectors"}
        }
        clean_base = replace(data, constraints=clean_constraints)
        for sub_spec in self.sub_predicates:
            sub_name = sub_spec["name"]
            sub_params = sub_spec.get("params", {})
            sub_predicate = PREDICATE_REGISTRY.create(sub_name, sub_params)
            temp = sub_predicate.apply(clean_base, context)
            if "phi_vector" in temp.constraints:
                vec = np.asarray(temp.constraints["phi_vector"], dtype=np.float32)
                phi_vectors.append(
                    {"name": sub_name, "vector": vec}
                )
            elif "phi_vectors" in temp.constraints:
                phi_vectors.extend(temp.constraints["phi_vectors"])

        if not phi_vectors:
            phi_vector = np.ones(len(data.features), dtype=np.float32)
            return data.with_constraints(
                {"phi_vectors": [{"name": self.name, "vector": phi_vector}]},
                predicates_applied=list(
                    data.metadata.get("predicates_applied", [])
                ) + [self.name],
            )

        return data.with_constraints(
            {"phi_vectors": phi_vectors},
            predicates_applied=list(
                data.metadata.get("predicates_applied", [])
            ) + [self.name],
        )