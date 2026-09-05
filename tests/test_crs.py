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
