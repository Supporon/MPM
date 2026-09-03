"""研究变量注册表：研究单元、背景采样、标签策略、样本权重。

P1-1 注册化：将研究单元类型、未标注采样方式、标签策略和权重策略从
硬编码枚举改为可插拔注册表。Task 只声明所需维度、空间支撑和输出类型，
通过注册表调用具体实现，不再直接 import 模块级函数。
"""

from __future__ import annotations

from ..core.registry import ComponentRegistry

# 研究单元类型（如 point_local_environment、regular_grid）
RESEARCH_UNIT_REGISTRY = ComponentRegistry("research_unit")

# 未标注/背景采样方式（如 random_points_in_nsw_boundary、archive_fixed）
BACKGROUND_SAMPLER_REGISTRY = ComponentRegistry("background_sampler")

# 标签策略（如 positive_unlabeled_as_zero、occurrence_binary）
LABEL_STRATEGY_REGISTRY = ComponentRegistry("label_strategy")

# 样本权重策略（如 size_code、uniform）
WEIGHT_STRATEGY_REGISTRY = ComponentRegistry("weight_strategy")
