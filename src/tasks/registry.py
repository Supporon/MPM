"""Task 注册表与构造辅助函数。"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.registry import ComponentRegistry

TASK_REGISTRY = ComponentRegistry("task")


def create_task(name: str, task_config: Mapping[str, Any], prediction_config: Mapping[str, Any]):
    """通过注册表实例化已配置的 Task。"""
    return TASK_REGISTRY.create(name, task_config, prediction_config)
