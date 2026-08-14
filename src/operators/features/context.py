"""旧版 GIS 特征算子共享的上下文。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Mapping


class LegacyFeatureContext:
    """延迟加载 ``lib_mpm``，并仅解析一次已配置的栅格输入。"""

    def __init__(self, dataset_config: Mapping[str, Any]):
        self.dataset_config = dataset_config
        self._legacy = None
        self._raster_files_cache: list[str] | None = None

    def legacy(self):
        if self._legacy is None:
            legacy_root = Path(self.dataset_config["root"])
            if not legacy_root.is_dir():
                raise FileNotFoundError(f"Original repository root not found: {legacy_root}")
            root_string = str(legacy_root)
            if root_string not in sys.path:
                sys.path.insert(0, root_string)
            try:
                self._legacy = importlib.import_module("lib_mpm")
            except ImportError as error:
                raise RuntimeError(
                    "Legacy GIS feature operators require lib_mpm.py and its geospatial dependencies."
                ) from error
        return self._legacy

    def raster_files(self) -> list[str]:
        if self._raster_files_cache is None:
            files: list[str] = []
            for key in ("magnetic", "gravity", "radiometric", "remote_sensing"):
                directory = Path(self.dataset_config[key])
                if not directory.is_dir():
                    raise FileNotFoundError(f"Raster directory not found: {directory}")
                files.extend(
                    str(path)
                    for path in sorted(directory.rglob("*"))
                    if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
                )
            if not files:
                raise FileNotFoundError("No raster inputs found")
            self._raster_files_cache = files
        return list(self._raster_files_cache)
