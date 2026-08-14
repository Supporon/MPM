#!/usr/bin/env python3
"""为原始 MPM 仓库产物构建只读清单。"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import struct
import sys
from collections import Counter
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CODEX_ROOT = THIS_DIR.parent
sys.path.insert(0, str(CODEX_ROOT))

from baseline_tools import (  # noqa: E402
    OUTPUT_DIRECTORY_NAMES,
    artifact_metadata,
    sha256_file,
    tiff_metadata,
)


def relative(path: Path, root: Path) -> str:
    """使用 POSIX 分隔符返回相对于清单根目录的路径。"""
    return path.relative_to(root).as_posix()


def source_files(original_root: Path) -> list[dict[str, object]]:
    """清点已知的源代码、环境和模型比较文件。"""
    paths = [
        original_root / "MPM_Porphyry_Lachlan.ipynb",
        original_root / "MPM_Porphyry_NSW.ipynb",
        original_root / "lib_mpm.py",
        original_root / "README.md",
        original_root / "env.yml",
        original_root / "env_loose.yml",
        original_root / "env_minimal.yml",
    ]
    paths.extend(sorted((original_root / "model_comparison").glob("*.py")))
    return [
        {
            "path": relative(path, original_root),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
        if path.is_file()
    ]


def raster_inventory(datasets_root: Path) -> list[dict[str, object]]:
    """检查数据集根目录下不属于输出的 TIFF 输入。"""
    inventory: list[dict[str, object]] = []
    for path in sorted(datasets_root.rglob("*.tif")):
        if any(part in OUTPUT_DIRECTORY_NAMES for part in path.parts):
            continue
        entry: dict[str, object] = {
            "path": relative(path, datasets_root),
            "size_bytes": path.stat().st_size,
        }
        try:
            entry.update(tiff_metadata(path))
        except (OSError, ValueError, struct.error) as error:  # type: ignore[name-defined]
            entry["read_error"] = f"{type(error).__name__}: {error}"
        inventory.append(entry)
    return inventory


def vector_inventory(datasets_root: Path) -> list[dict[str, object]]:
    """清点数据集根目录下 Shapefile 的路径和字节大小。"""
    return [
        {"path": relative(path, datasets_root), "size_bytes": path.stat().st_size}
        for path in sorted(datasets_root.rglob("*.shp"))
    ]


def output_artifacts(datasets_root: Path) -> dict[str, dict[str, object]]:
    """安全收集已识别基线输出集内文件的元数据。"""
    results: dict[str, dict[str, object]] = {}
    for output_name in sorted(OUTPUT_DIRECTORY_NAMES):
        directory = datasets_root / output_name
        if not directory.is_dir():
            continue
        artifacts: dict[str, object] = {}
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            rel = relative(path, datasets_root)
            try:
                artifacts[rel] = artifact_metadata(path)
            except (OSError, UnicodeDecodeError, ValueError, struct.error) as error:  # type: ignore[name-defined]
                artifacts[rel] = {
                    "kind": "unreadable",
                    "size_bytes": path.stat().st_size,
                    "read_error": f"{type(error).__name__}: {error}",
                }
        results[output_name] = {
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        }
    return results


def count_by_parent(paths: list[dict[str, object]]) -> dict[str, int]:
    """按相对路径的第一个组成部分统计清单条目。"""
    counts: Counter[str] = Counter()
    for entry in paths:
        counts[str(Path(str(entry["path"])).parts[0])] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    """解析清单选项、清点源目录树并写入 JSON。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-root",
        type=Path,
        default=CODEX_ROOT.parent / "EarthByte-MPM_Lachlan_Porphyry",
        help="Original repository root (read-only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CODEX_ROOT / "docs" / "baseline_manifest.json",
        help="Manifest to write.",
    )
    args = parser.parse_args()
    original_root = args.original_root.resolve()
    datasets_root = original_root / "Datasets"
    if not datasets_root.is_dir():
        parser.error(f"Datasets directory not found: {datasets_root}")

    rasters = raster_inventory(datasets_root)
    vectors = vector_inventory(datasets_root)
    manifest = {
        "manifest_version": 1,
        "purpose": "Pre-refactor behavioral and artifact baseline; no training performed.",
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "original_repository": {
            "path_hint": "../EarthByte-MPM_Lachlan_Porphyry",
            "datasets_path_hint": "Datasets",
        },
        "source_files": source_files(original_root),
        "input_inventory": {
            "raster_count": len(rasters),
            "raster_count_by_top_level_directory": count_by_parent(rasters),
            "rasters": rasters,
            "vector_shapefile_count": len(vectors),
            "vectors": vectors,
        },
        "output_sets": output_artifacts(datasets_root),
        "contract_output_sets": [
            "Outputs_Cu_Lachlan_v1.6",
            "Outputs_Cu_NSW_v1.6",
        ],
        "non_contract_snapshots": {
            "Outputs": "Incomplete early intermediates.",
            "Outputs_test": "Notebook cache snapshot with known internal path/cache inconsistency.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} with {len(rasters)} input rasters and "
          f"{sum(item['artifact_count'] for item in manifest['output_sets'].values())} output artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
