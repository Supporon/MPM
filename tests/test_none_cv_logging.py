"""cross_validation=none 与日志生命周期测试。"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bootstrap import load_builtin_components
from src.core.config import ConfigError, load_config, validate_config
from src.utils import logging as logging_mod
from src.utils.logging import close_logging, setup_logging


def test_config_accepts_cross_validation_none_with_none_tuner():
    load_builtin_components()
    loaded = load_config(ROOT / "configs" / "experiments" / "lachlan_rf_baseline.yaml")
    values = copy.deepcopy(loaded.values)
    values["tuning"] = {"name": "none", "params": {}}
    values["validation"]["cross_validation"] = {"name": "none", "params": {}}
    validate_config(values)  # 不抛异常


def test_config_rejects_cross_validation_none_with_cv_tuner():
    load_builtin_components()
    loaded = load_config(ROOT / "configs" / "experiments" / "lachlan_rf_baseline.yaml")
    values = copy.deepcopy(loaded.values)
    values["tuning"] = {"name": "bayes", "params": {"n_iter": 2, "search_space": {}}}
    values["validation"]["cross_validation"] = {"name": "none", "params": {}}
    with pytest.raises(ConfigError, match="disables cross-validation"):
        validate_config(values)


def test_none_splitter_build_cv_raises():
    load_builtin_components()
    from src.validation.splitters import build_cv

    with pytest.raises(ValueError, match="disables cross-validation"):
        build_cv("none", {}, 42)


def test_close_logging_releases_handlers(tmp_path):
    """setup_logging 后 close_logging 应清空管理的 handler 并释放文件句柄。"""
    setup_logging(tmp_path)
    logger = logging.getLogger("mpm")
    assert logger.handlers, "setup_logging should add handlers"
    assert logging_mod._managed_handlers, "managed handlers should be tracked"

    close_logging()

    assert logging_mod._managed_handlers == []
    assert logging_mod._initialized == set()
    assert logger.handlers == []

    # 再次 setup 应能重新建立 handler（关闭不破坏后续运行）。
    setup_logging(tmp_path)
    assert logger.handlers
    close_logging()
