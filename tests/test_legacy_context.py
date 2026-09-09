"""LegacyFeatureContext 的 lib_mpm 加载与隔离测试（P1-07）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.operators.features.context import LegacyFeatureContext


def _write_lib_mpm(root: Path, marker: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source = (
        f"MARKER = {marker!r}\n\n"
        "def get_dist_line(xs, ys, line_files, distance_type='euclidean', input_crs=None):\n"
        "    import pandas as pd\n"
        f"    return pd.DataFrame({{'distance': [{marker!r}] * len(xs)}})\n"
    )
    (root / "lib_mpm.py").write_text(source, encoding="utf-8")
    return root / "lib_mpm.py"


def test_two_roots_load_distinct_modules(tmp_path):
    """同一进程先后使用两个 root，分别返回各自的 lib_mpm，而非复用首个模块。"""
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    _write_lib_mpm(root_a, "A")
    _write_lib_mpm(root_b, "B")

    ctx_a = LegacyFeatureContext({"root": str(root_a)})
    ctx_b = LegacyFeatureContext({"root": str(root_b)})

    mod_a = ctx_a.legacy()
    mod_b = ctx_b.legacy()

    # 两个 root 的 lib_mpm 是不同模块对象（不再命中 sys.modules 缓存串用）。
    assert mod_a is not mod_b
    assert mod_a.MARKER == "A"
    assert mod_b.MARKER == "B"

    # 各自的算子返回各自 root 的特征。
    import pandas as pd

    xs = pd.Series([1.0, 2.0])
    assert mod_a.get_dist_line(xs, xs, [])["distance"].tolist() == ["A", "A"]
    assert mod_b.get_dist_line(xs, xs, [])["distance"].tolist() == ["B", "B"]


def test_missing_root_raises():
    ctx = LegacyFeatureContext({"root": str(Path("/nonexistent/root"))})
    with pytest.raises(FileNotFoundError):
        ctx.legacy()
