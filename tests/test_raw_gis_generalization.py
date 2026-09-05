"""raw_gis 字段可配置化与按算子收紧输入校验的单元测试（P1-09）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bootstrap import load_builtin_components
from src.tasks.target_area import TargetAreaPredictionTask
from src.data.weight_strategies import SizeCodeWeightStrategy
from src.operators.features.registry import FEATURE_OPERATOR_REGISTRY


def _base_task_args():
    research_unit = {"type": "point_local_environment", "prediction_grid_size": 0.1}
    prediction = {"score_type": "relative_score", "normalization": "minmax", "export_geotiff": False}
    return (
        {"name": "target_area_prediction"},
        research_unit,
        prediction,
    )


def _full_dataset():
    return {
        "occurrence": "occ.shp",
        "boundary": "bnd.shp",
        "training_boundary": "trn.shp",
        "geology": {
            "line_files": ["faults.shp"],
            "metamorphic_facies": "met.shp",
            "intrusions": "int.shp",
            "rock_units": "rock.shp",
        },
        "magnetic": "mag/",
        "gravity": "grav/",
        "radiometric": "rad/",
        "remote_sensing": "rs/",
        "elevation": "dem.tif",
        "seismic": "seis.shp",
    }


def test_operator_required_keys_are_declared():
    load_builtin_components()
    assert set(FEATURE_OPERATOR_REGISTRY.get("categorical_geology").REQUIRED_DATASET_KEYS) == {"geology"}
    assert set(FEATURE_OPERATOR_REGISTRY.get("categorical_geology").REQUIRED_GEOLOGY_KEYS) == {
        "metamorphic_facies", "intrusions", "rock_units"
    }
    assert set(FEATURE_OPERATOR_REGISTRY.get("elevation_gradient").REQUIRED_DATASET_KEYS) == {"elevation"}
    assert set(FEATURE_OPERATOR_REGISTRY.get("raster_statistics").REQUIRED_DATASET_KEYS) == {
        "magnetic", "gravity", "radiometric", "remote_sensing"
    }


def test_validate_config_requires_only_selected_operator_inputs():
    load_builtin_components()
    task_args = _base_task_args()

    # 只选 categorical_geology：不应再要求 magnetic/gravity/radiometric/remote_sensing/elevation/seismic。
    dataset = {
        "occurrence": "occ.shp",
        "boundary": "bnd.shp",
        "training_boundary": "trn.shp",
        "geology": {
            "metamorphic_facies": "met.shp",
            "intrusions": "int.shp",
            "rock_units": "rock.shp",
        },
    }
    TargetAreaPredictionTask.validate_config(
        *task_args, dataset, "raw_gis", [{"name": "categorical_geology", "params": {}}]
    )

    # 只选 elevation_gradient：不应要求 geology 及其子字段。
    dataset_elev = {
        "occurrence": "occ.shp",
        "boundary": "bnd.shp",
        "training_boundary": "trn.shp",
        "elevation": "dem.tif",
    }
    TargetAreaPredictionTask.validate_config(
        *task_args, dataset_elev, "raw_gis", [{"name": "elevation_gradient", "params": {}}]
    )


def test_validate_config_still_rejects_missing_selected_operator_inputs():
    load_builtin_components()
    task_args = _base_task_args()

    # 选了 categorical_geology 但缺 geology → 应报错。
    dataset = {"occurrence": "occ.shp", "boundary": "bnd.shp", "training_boundary": "trn.shp"}
    try:
        TargetAreaPredictionTask.validate_config(
            *task_args, dataset, "raw_gis", [{"name": "categorical_geology", "params": {}}]
        )
    except ValueError as error:
        assert "geology" in str(error)
    else:
        raise AssertionError("expected ValueError for missing geology")


def test_validate_config_no_operators_falls_back_to_full_nsw_requirement():
    load_builtin_components()
    task_args = _base_task_args()
    # 未提供算子列表 → 保持整套 NSW 字段要求。
    partial = _full_dataset()
    del partial["seismic"]
    try:
        TargetAreaPredictionTask.validate_config(*task_args, partial, "raw_gis", None)
    except ValueError as error:
        assert "seismic" in str(error)
    else:
        raise AssertionError("expected ValueError for missing seismic in full-NSW fallback")


def test_size_code_weight_strategy_column_is_configurable():
    load_builtin_components()
    df = pd.DataFrame({"SCALE": ["L", "M", "L"]})
    strategy = SizeCodeWeightStrategy({"column": "SCALE"})
    weights = strategy.compute(df, {"L": 1.0, "M": 2.0})
    assert list(weights) == [1.0, 2.0, 1.0]

    # 缺省仍为 SIZE_CODE，缺列时报错并提示列名。
    df_default = pd.DataFrame({"SIZE_CODE": ["L"]})
    assert list(SizeCodeWeightStrategy({}).compute(df_default, {"L": 3.0})) == [3.0]
    try:
        SizeCodeWeightStrategy({}).compute(df, {"L": 1.0})
    except ValueError as error:
        assert "SIZE_CODE" in str(error)
    else:
        raise AssertionError("expected ValueError for missing default SIZE_CODE column")


if __name__ == "__main__":
    test_operator_required_keys_are_declared()
    test_validate_config_requires_only_selected_operator_inputs()
    test_validate_config_still_rejects_missing_selected_operator_inputs()
    test_validate_config_no_operators_falls_back_to_full_nsw_requirement()
    test_size_code_weight_strategy_column_is_configurable()
    print("raw_gis 泛化测试全部通过")
