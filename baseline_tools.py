"""用于采集和检查重构前基线的只读工具。

本模块有意避免导入原始项目。具体而言，它不会反序列化模型产物，
也不依赖 GDAL 或 rasterio，因此可以在最小 Python 环境中安全地
执行基线检查。
"""

from __future__ import annotations

import csv
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


OUTPUT_DIRECTORY_NAMES = {
    "Outputs",
    "Outputs_test",
    "Outputs_Cu_Lachlan_v1.6",
    "Outputs_Cu_NSW_v1.6",
}


def sha256_file(path: Path) -> str:
    """通过限定大小的分块读取计算文件的 SHA-256 摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_digest(columns: Iterable[str]) -> str:
    """使用稳定的 JSON 编码计算有序列名序列的哈希。"""
    payload = json.dumps(list(columns), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def csv_metadata(path: Path) -> dict[str, Any]:
    """在不使用 pandas 的情况下返回 schema、行数和标签分布。"""
    headerless = path.name == "target_mask.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
        if first_row is None:
            return {
                "kind": "csv",
                "has_header": not headerless,
                "columns": [],
                "column_count": 0,
                "row_count": 0,
                "schema_sha256": schema_digest([]),
            }

        if headerless:
            columns = [f"column_{index}" for index in range(len(first_row))]
            rows: Iterable[list[str]] = _prepend(first_row, reader)
        else:
            columns = first_row
            rows = reader

        label_index = columns.index("label") if "label" in columns else None
        label_counts: Counter[str] = Counter()
        row_count = 0
        for row in rows:
            if not row:
                continue
            row_count += 1
            if label_index is not None and label_index < len(row):
                label_counts[row[label_index]] += 1

    result: dict[str, Any] = {
        "kind": "csv",
        "has_header": not headerless,
        "columns": columns,
        "column_count": len(columns),
        "row_count": row_count,
        "schema_sha256": schema_digest(columns),
    }
    if label_index is not None:
        result["label_distribution"] = dict(sorted(label_counts.items()))
        result["feature_count"] = len(columns) - int("sample_weight" in columns) - 1
    return result


def _prepend(first_row: list[str], iterator: Iterable[list[str]]) -> Iterable[list[str]]:
    """先生成已读取的首行，再生成迭代器中的其余行。"""
    yield first_row
    yield from iterator


def _read_at(handle: Any, offset: int, length: int) -> bytes:
    """从可定位的二进制句柄中精确读取 ``length`` 个字节。"""
    handle.seek(offset)
    data = handle.read(length)
    if len(data) != length:
        raise ValueError("unexpected end of TIFF")
    return data


def _unpack_values(
    handle: Any,
    raw_value: bytes,
    value_offset: int,
    field_type: int,
    count: int,
    endian: str,
    inline_size: int,
) -> list[Any]:
    """在不使用外部库的情况下解码一个 classic TIFF 或 BigTIFF 字段。"""
    type_info = {
        1: ("B", 1),
        2: ("c", 1),
        3: ("H", 2),
        4: ("I", 4),
        5: ("II", 8),
        6: ("b", 1),
        7: ("B", 1),
        8: ("h", 2),
        9: ("i", 4),
        10: ("ii", 8),
        11: ("f", 4),
        12: ("d", 8),
        16: ("Q", 8),
        17: ("q", 8),
        18: ("Q", 8),
    }
    if field_type not in type_info:
        return []
    fmt, byte_size = type_info[field_type]
    total_size = byte_size * count
    data = raw_value[:total_size] if total_size <= inline_size else _read_at(handle, value_offset, total_size)
    if field_type == 2:
        return [data.rstrip(b"\x00").decode("ascii", errors="replace")]
    if field_type in {5, 10}:
        values = struct.unpack(endian + fmt * count, data)
        return [numerator / denominator if denominator else None for numerator, denominator in zip(values[::2], values[1::2])]
    return list(struct.unpack(endian + fmt * count, data))


def tiff_metadata(path: Path) -> dict[str, Any]:
    """在不使用 GDAL 的情况下读取基线所需的首幅 TIFF 图像字段。"""
    with path.open("rb") as handle:
        byte_order = _read_at(handle, 0, 2)
        if byte_order == b"II":
            endian = "<"
        elif byte_order == b"MM":
            endian = ">"
        else:
            raise ValueError("not a TIFF byte-order marker")
        magic = struct.unpack(endian + "H", _read_at(handle, 2, 2))[0]
        if magic == 42:
            offset_size, count_size, entry_size = 4, 2, 12
            first_ifd = struct.unpack(endian + "I", _read_at(handle, 4, 4))[0]
            count_format = "H"
            entry_format = "HHII"
        elif magic == 43:
            offset_size = struct.unpack(endian + "H", _read_at(handle, 4, 2))[0]
            if offset_size != 8:
                raise ValueError("unsupported BigTIFF offset size")
            count_size, entry_size = 8, 20
            first_ifd = struct.unpack(endian + "Q", _read_at(handle, 8, 8))[0]
            count_format = "Q"
            entry_format = "HHQQ"
        else:
            raise ValueError("unrecognised TIFF magic")

        entry_count = struct.unpack(endian + count_format, _read_at(handle, first_ifd, count_size))[0]
        entries: dict[int, list[Any]] = {}
        for index in range(entry_count):
            entry = _read_at(handle, first_ifd + count_size + index * entry_size, entry_size)
            tag, field_type, count, value_offset = struct.unpack(endian + entry_format, entry)
            raw_value = entry[-offset_size:]
            entries[tag] = _unpack_values(
                handle, raw_value, value_offset, field_type, count, endian, offset_size
            )

    def scalar(tag: int, default: Any = None) -> Any:
        """返回 TIFF 标签的第一个解码值；不存在时返回默认值。"""
        values = entries.get(tag, [])
        return values[0] if values else default

    sample_format = scalar(339, 1)
    sample_format_names = {1: "uint", 2: "int", 3: "float"}
    bits = scalar(258)
    dtype = None
    if bits is not None:
        dtype = f"{sample_format_names.get(sample_format, 'unknown')}{bits}"
    result = {
        "kind": "geotiff",
        "shape": [scalar(257), scalar(256)],
        "band_count": scalar(277, 1),
        "bits_per_sample": bits,
        "sample_format": sample_format_names.get(sample_format, str(sample_format)),
        "dtype": dtype,
        "nodata": scalar(42113),
    }
    if 33550 in entries:
        result["pixel_scale"] = entries[33550]
    if 33922 in entries:
        result["tiepoint"] = entries[33922]
    return result


def pickle_model_type(path: Path) -> str:
    """在不执行 pickle 载荷的情况下识别已知模型或缩放器类型。"""
    payload = path.read_bytes()
    for marker in (
        b"BaggingPuClassifier",
        b"RandomForestClassifier",
        b"GradientBoostingClassifier",
        b"LGBMClassifier",
        b"LogisticRegression",
        b"XGBClassifier",
        b"SVC",
        b"OneHotEncoder",
        b"StandardScaler",
    ):
        if marker in payload:
            return marker.decode("ascii")
    return "unknown_pickle_type"


def artifact_metadata(path: Path) -> dict[str, Any]:
    """根据后缀检查基线产物，同时避免执行 pickle 数据。"""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        result = csv_metadata(path)
    elif suffix in {".tif", ".tiff"}:
        result = tiff_metadata(path)
    elif suffix == ".pkl":
        result = {"kind": "pickle", "model_type": pickle_model_type(path)}
    else:
        result = {"kind": suffix.lstrip(".") or "file"}
    result["size_bytes"] = path.stat().st_size
    return result
