"""不包含实验特定策略的文件系统辅助函数。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_timestamp() -> str:
    """返回不含小数秒的当前 UTC 时间戳。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    """通过定长分块读取计算文件的 SHA-256 摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """创建父目录并写入带缩进、兼容 ASCII 的 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_yaml(path: Path, value: Any) -> None:
    """创建父目录并写入保留键顺序的 YAML。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def git_commit(path: Path) -> str | None:
    """如果给定检出目录是 Git 工作树，则返回其提交哈希。"""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def git_dirty(path: Path) -> bool | None:
    """返回工作树是否有未提交改动；非 Git 工作树返回 None。"""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def environment_summary() -> dict[str, Any]:
    """收集 Python 与核心依赖版本，用于复现清单。"""
    import platform
    import sys

    summary: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for module_name in ("numpy", "pandas", "sklearn", "skopt", "geopandas", "torch"):
        try:
            module = __import__(module_name)
            summary[module_name] = getattr(module, "__version__", "unknown")
        except Exception:  # pragma: no cover - 依赖可选
            summary[module_name] = None
    return summary
