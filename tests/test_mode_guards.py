"""P1-06/P1-09: grid 模型与执行模式/调参组合的配置层守卫。"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bootstrap import load_builtin_components
from src.core.config import ConfigError, load_config, validate_config

RAW_GIS = ROOT / "configs" / "experiments" / "lachlan_rf_raw_gis.yaml"
ARCHIVE = ROOT / "configs" / "experiments" / "lachlan_rf_phase2.yaml"


def test_rejects_grid_model_in_raw_gis():
    load_builtin_components()
    bad = copy.deepcopy(load_config(RAW_GIS).values)
    bad["model"] = {"name": "cnn2d", "params": {}}
    with pytest.raises(ConfigError, match="grid model"):
        validate_config(bad)


def test_rejects_grid_model_with_cv_tuner():
    load_builtin_components()
    bad = copy.deepcopy(load_config(ARCHIVE).values)
    bad["model"] = {"name": "cnn2d", "params": {}}
    bad["tuning"] = {
        "name": "bayes",
        "params": {
            "n_iter": 2,
            "search_space": {"patch": {"type": "integer", "low": 5, "high": 9}},
        },
    }
    with pytest.raises(ConfigError, match="grid model"):
        validate_config(bad)


def test_allows_grid_model_with_none_tuner_in_archive():
    load_builtin_components()
    ok = copy.deepcopy(load_config(ARCHIVE).values)
    ok["model"] = {"name": "cnn2d", "params": {}}
    ok["tuning"] = {"name": "none", "params": {}}
    validate_config(ok)  # 不抛异常
