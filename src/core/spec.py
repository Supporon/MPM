"""可配置实验组件图的类型化视图。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ComponentSpec":
        return cls(name=str(value["name"]), params=dict(value.get("params", {})))


@dataclass(frozen=True)
class ExperimentSpec:
    """供 ``Experiment`` 使用的已解析组件选择，替代与具体实现绑定的配置键。"""

    name: str
    seed: int
    output_dir: Path
    execution_mode: str
    task: ComponentSpec
    feature_operators: tuple[ComponentSpec, ...]
    model: ComponentSpec
    label_refinement: ComponentSpec | None
    knowledge: tuple[ComponentSpec, ...]
    predicates: tuple[ComponentSpec, ...]
    holdout: ComponentSpec
    cross_validation: ComponentSpec
    metrics: tuple[str, ...]
    primary_metric: str
    tuner: ComponentSpec

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExperimentSpec":
        refinement = values.get("label_refinement", {})
        knowledge = values.get("knowledge", {})
        predicates = values.get("predicates", {})
        return cls(
            name=str(values["experiment"]["name"]),
            seed=int(values["experiment"]["seed"]),
            output_dir=Path(values["experiment"]["output_dir"]),
            execution_mode=str(values["experiment"]["execution_mode"]),
            task=ComponentSpec.from_mapping(values["task"]),
            feature_operators=tuple(
                ComponentSpec.from_mapping(item) for item in values["features"].get("operators", [])
            ),
            model=ComponentSpec.from_mapping(values["model"]),
            label_refinement=(
                ComponentSpec.from_mapping(refinement)
                if refinement.get("enabled", False)
                else None
            ),
            knowledge=tuple(
                ComponentSpec.from_mapping(item)
                for item in knowledge.get("items", [])
                if knowledge.get("enabled", False)
            ),
            predicates=tuple(
                ComponentSpec.from_mapping(item)
                for item in predicates.get("items", [])
                if predicates.get("enabled", False)
            ),
            holdout=ComponentSpec.from_mapping(values["validation"]["holdout"]),
            cross_validation=ComponentSpec.from_mapping(values["validation"]["cross_validation"]),
            metrics=tuple(values["validation"]["metrics"]),
            primary_metric=str(values["validation"]["primary_metric"]),
            tuner=ComponentSpec.from_mapping(values["tuning"]),
        )
