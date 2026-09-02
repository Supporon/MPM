"""种子派生（SeedContext）单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.seeds import derive_seed, derive_seeds


def test_derive_seeds_is_deterministic():
    assert derive_seeds(42) == derive_seeds(42)


def test_derive_seeds_distinct_per_purpose():
    seeds = derive_seeds(42)
    purposes = [k for k in seeds if k.endswith("_seed")]
    values = [seeds[k] for k in purposes]
    assert len(values) == len(set(values)), f"各用途种子应互不相同: {seeds}"


def test_derive_seeds_changes_with_base():
    assert derive_seeds(42) != derive_seeds(43)


def test_derive_seed_is_stable():
    assert derive_seed(42, "split") == derive_seed(42, "split")
    assert derive_seed(42, "split") != derive_seed(42, "sampling")
    assert 0 <= derive_seed(42, "split") < 2**31


if __name__ == "__main__":
    test_derive_seeds_is_deterministic()
    test_derive_seeds_distinct_per_purpose()
    test_derive_seeds_changes_with_base()
    test_derive_seed_is_stable()
    print("种子派生测试全部通过")
