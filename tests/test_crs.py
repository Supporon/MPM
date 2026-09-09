"""P0-6: CRS 解析、一致性校验与导出校验测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pyproj = pytest.importorskip("pyproj")

from src.utils.crs import (
    crs_identifier,
    require_consistent_crs,
    resolve_crs_identifier,
)


def _crs(code: str):
    return pyproj.CRS.from_user_input(code)


def test_crs_identifier_prefers_epsg():
    assert crs_identifier(_crs("EPSG:4283")) == "EPSG:4283"
    assert crs_identifier(_crs("EPSG:3857")) == "EPSG:3857"
    assert crs_identifier(None) is None


def test_crs_identifier_falls_back_to_wkt():
    custom = pyproj.CRS.from_proj4(
        "+proj=longlat +datum=WGS84 +no_defs"
    )
    ident = crs_identifier(custom)
    assert ident is not None
    # 无权威 EPSG 时返回 WKT（或至少不为 None）
    assert ident.startswith("EPSG:") or "GEOGCS" in ident or "PROJCS" in ident


def test_require_consistent_crs_accepts_same():
    ident = require_consistent_crs(
        [("occurrence", _crs("EPSG:4283")), ("boundary", _crs("EPSG:4283"))],
        "test",
    )
    assert ident == "EPSG:4283"


def test_require_consistent_crs_rejects_mismatch():
    with pytest.raises(ValueError, match="CRS mismatch"):
        require_consistent_crs(
            [("occurrence", _crs("EPSG:4283")), ("boundary", _crs("EPSG:3857"))],
            "test",
        )


def test_require_consistent_crs_ignores_missing():
    ident = require_consistent_crs(
        [("occurrence", None), ("boundary", _crs("EPSG:4283"))],
        "test",
    )
    assert ident == "EPSG:4283"
    assert require_consistent_crs([("a", None), ("b", None)], "test") is None


def test_resolve_crs_identifier_epsg_int_and_string():
    assert resolve_crs_identifier(4283) == "EPSG:4283"
    assert resolve_crs_identifier("EPSG:4283") == "EPSG:4283"


def test_task_records_and_rejects_crs_mismatch():
    from src.tasks.target_area import TargetAreaPredictionTask

    task = TargetAreaPredictionTask(
        {}, {"score_type": "probability", "normalization": "none", "export_geotiff": False}
    )
    task._record_crs("occurrence", _crs("EPSG:4283"))
    assert task._research_crs_id == "EPSG:4283"
    with pytest.raises(ValueError, match="CRS mismatch"):
        task._record_crs("boundary", _crs("EPSG:3857"))


def test_export_geotiff_rejects_mismatched_target_crs(tmp_path):
    """导出时 target_crs 与已记录研究单元 CRS 不一致应明确失败。"""
    try:
        from osgeo import gdal
    except ImportError:  # pragma: no cover
        pytest.skip("GDAL python bindings not installed")

    from src.tasks.target_area import TargetAreaPredictionTask

    task = TargetAreaPredictionTask(
        {}, {"score_type": "probability", "normalization": "none", "export_geotiff": True}
    )
    task._record_crs("occurrence", _crs("EPSG:4283"))
    # target_crs 错标为 3857 → 拒绝
    task.prediction_config["target_crs"] = 3857
    import numpy as np
    import pandas as pd

    probabilities = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "prob": [0.1, 0.2]})
    target_mask = pd.DataFrame(
        [[0.0, 0.0, True], [1.0, 1.0, True], [0.0, 1.0, False], [1.0, 0.0, False]]
    )
    with pytest.raises(ValueError, match="does not match"):
        task.export_geotiff(probabilities, target_mask, tmp_path / "out.tif")


def test_feature_operators_propagate_research_unit_crs():
    """P0-04: 矢量算子把研究单元 CRS 显式传给底层 input_crs，而非退回默认经纬度。

    研究单元标记 EPSG:32755 时，line_distance 与 categorical_geology 调用必须
    收到 input_crs='EPSG:32755'，否则底层会把投影坐标当经纬度（提取错误特征）。
    """
    import pandas as pd
    from src.operators.features.context import LegacyFeatureContext
    from src.operators.features.builtins import (
        CategoricalGeologyOperator,
        LineDistanceOperator,
    )

    captured: dict[str, object] = {}

    class RecordingLegacy:
        def get_dist_line(self, xs, ys, line_files, distance_type="euclidean", input_crs=None):
            captured["line_input_crs"] = input_crs
            return pd.DataFrame({"distance": [0.0] * len(xs)})

        def get_cat_data(self, xs, ys, polygon_files, field, input_crs=None):
            captured[f"cat_input_crs_{field}"] = input_crs
            return pd.DataFrame({field: ["x"] * len(xs)})

    dataset_config = {
        "seismic": "seismic.shp",
        "geology": {
            "line_files": ["l1.shp"],
            "metamorphic_facies": "meta.shp",
            "intrusions": "intr.shp",
            "rock_units": "rock.shp",
        },
    }
    context = LegacyFeatureContext(dataset_config)
    context._legacy = RecordingLegacy()

    units = pd.DataFrame({"X": [500000.0], "Y": [6000000.0]})
    units.attrs["crs"] = _crs("EPSG:32755")

    LineDistanceOperator(context, {}).extract(units)
    CategoricalGeologyOperator(context, {}).extract(units)

    assert captured["line_input_crs"] == "EPSG:32755"
    assert captured["cat_input_crs_MetFacies"] == "EPSG:32755"
    assert captured["cat_input_crs_Dominant_L"] == "EPSG:32755"


def test_feature_operators_fall_back_without_crs():
    """无 CRS 记录时算子不传 input_crs，保留旧版经纬度默认行为。"""
    import pandas as pd
    from src.operators.features.context import LegacyFeatureContext
    from src.operators.features.builtins import LineDistanceOperator

    captured: dict[str, object] = {}

    class RecordingLegacy:
        def get_dist_line(self, xs, ys, line_files, distance_type="euclidean", input_crs=None):
            captured["input_crs"] = input_crs
            return pd.DataFrame({"distance": [0.0] * len(xs)})

    dataset_config = {
        "seismic": "seismic.shp",
        "geology": {"line_files": ["l1.shp"]},
    }
    context = LegacyFeatureContext(dataset_config)
    context._legacy = RecordingLegacy()

    units = pd.DataFrame({"X": [147.0], "Y": [-35.0]})  # 无 attrs.crs
    LineDistanceOperator(context, {}).extract(units)
    assert captured["input_crs"] is None


def test_build_spatial_reference_uses_set_from_user_input(monkeypatch):
    """P1-08: 导出用 SetFromUserInput 统一解析 EPSG/WKT/PROJ 并校验返回码。"""
    import sys
    import types

    calls: list[tuple] = []

    class FakeSRS:
        def SetFromUserInput(self, value):
            calls.append(value)
            return 0 if "not a crs" not in value else 1

        def ImportFromEPSG(self, value):  # 旧实现用，不应被调用
            raise AssertionError("ImportFromEPSG should not be used")

        def ImportFromWkt(self, value):  # 旧实现用，不应被调用
            raise AssertionError("ImportFromWkt should not be used")

    fake_osr = types.ModuleType("osgeo.osr")
    fake_osr.SpatialReference = FakeSRS
    fake_osgeo = types.ModuleType("osgeo")
    fake_osgeo.osr = fake_osr
    monkeypatch.setitem(sys.modules, "osgeo", fake_osgeo)
    monkeypatch.setitem(sys.modules, "osgeo.osr", fake_osr)

    from src.tasks.target_area import _build_spatial_reference

    _build_spatial_reference(4283)
    _build_spatial_reference("EPSG:4283")
    _build_spatial_reference("+proj=longlat +datum=WGS84 +no_defs")

    assert calls == [
        "EPSG:4283",
        "EPSG:4283",
        "+proj=longlat +datum=WGS84 +no_defs",
    ]

    with pytest.raises(ValueError, match="could not be parsed"):
        _build_spatial_reference("not a crs")
