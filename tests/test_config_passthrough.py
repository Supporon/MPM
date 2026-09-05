"""P1-02: 研究单元/标签/权重配置透传与权重策略可配置。"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bootstrap import load_builtin_components
from src.core.config import ConfigError, load_config, validate_config
from src.data.label_strategies import PositiveUnlabeledAsZeroStrategy

SAMPLE_WEIGHT = {"VLG": 0.5, "LGE": 0.4, "MED": 0.3, "SML": 0.2, "OCC": 0.1}


def test_label_strategy_defaults_to_size_code_weight():
    load_builtin_components()
    strat = PositiveUnlabeledAsZeroStrategy({})
    occ = pd.DataFrame({"SIZE_CODE": ["VLG", "OCC", "MED"]})
    out = strat.apply_positive(occ, {"positive_value": 1, "sample_weight": SAMPLE_WEIGHT})
    assert out["label"].tolist() == [1, 1, 1]
    assert out["sample_weight"].tolist() == [0.5, 0.1, 0.3]


def test_label_strategy_uses_configured_uniform_weight():
    load_builtin_components()
    strat = PositiveUnlabeledAsZeroStrategy({})
    occ = pd.DataFrame({"SIZE_CODE": ["VLG", "OCC", "MED"]})
    label_config = {
        "positive_value": 1,
        "sample_weight": SAMPLE_WEIGHT,
        "weight_strategy": {"name": "uniform", "params": {"value": 1.0}},
    }
    out = strat.apply_positive(occ, label_config)
    assert (out["sample_weight"] == 1.0).all()


def test_config_validates_weight_strategy_name():
    load_builtin_components()
    gis = load_config(ROOT / "configs" / "experiments" / "lachlan_rf_raw_gis.yaml")

    valid = copy.deepcopy(gis.values)
    valid["label"]["weight_strategy"] = {"name": "uniform", "params": {"value": 1.0}}
    validate_config(valid)

    invalid = copy.deepcopy(gis.values)
    invalid["label"]["weight_strategy"] = {"name": "not_a_weight", "params": {}}
    with pytest.raises(ConfigError, match="weight_strategy"):
        validate_config(invalid)


def test_config_accepts_params_passthrough_keys():
    load_builtin_components()
    gis = load_config(ROOT / "configs" / "experiments" / "lachlan_rf_raw_gis.yaml")
    valid = copy.deepcopy(gis.values)
    valid["research_unit"]["params"] = {"foo": 1}
    valid["label"]["params"] = {"bar": 2}
    validate_config(valid)

    bad = copy.deepcopy(gis.values)
    bad["research_unit"]["params"] = ["not", "a", "mapping"]
    with pytest.raises(ConfigError, match="research_unit.params"):
        validate_config(bad)
