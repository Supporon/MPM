"""绘图辅助函数的轻量单元测试（不触发 matplotlib 渲染或归档重建）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting.plot import point_kind_labels, _resolve_archive_from_manifest


def _write_kind_csv(archive_dir: Path, name: str, n_rows: int):
    pd.DataFrame({"x": np.arange(n_rows), "y": np.zeros(n_rows)}).to_csv(
        archive_dir / name, index=False
    )


def test_point_kind_labels_derives_counts_from_files(tmp_path):
    # 用非 277/272 的行数验证 kind 标签来自文件而非硬编码常量。
    _write_kind_csv(tmp_path, "training_data_deposit.csv", 5)
    _write_kind_csv(tmp_path, "training_data_unlab.csv", 3)
    kind = point_kind_labels(tmp_path)
    assert list(kind) == ["deposit"] * 5 + ["unlab"] * 3


def test_resolve_archive_from_manifest_reads_input_paths(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"input_paths": {"archive_dir": str(tmp_path / "arch")}}),
        encoding="utf-8",
    )
    assert _resolve_archive_from_manifest(tmp_path) == tmp_path / "arch"


def test_resolve_archive_from_manifest_returns_none_when_missing(tmp_path):
    assert _resolve_archive_from_manifest(tmp_path) is None
