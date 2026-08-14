"""在三维数据契约建立之前，为深边部预测保留的显式占位实现。"""

from __future__ import annotations

from typing import Mapping, Any

from .registry import TASK_REGISTRY
from .target_area import TaskCapabilityError


@TASK_REGISTRY.decorator("deep_edge_prediction")
class DeepEdgePredictionTask:
    """明确报告三维数据契约尚未实现的深边部预测占位 Task。"""

    def __init__(self, task_config: Mapping[str, Any], prediction_config: Mapping[str, Any]):
        raise TaskCapabilityError(
            "deep_edge_prediction is registered as a capability placeholder but is not implemented: "
            "3D/voxel research units, drillhole/depth labels, 3D feature operators, and output contracts are still required."
        )
