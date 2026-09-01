"""结构化日志配置，用于实验运行追踪。

为每个实验提供控制台和文件双通道日志，日志文件写入实验输出目录。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized: set[str] = set()


def setup_logging(output_dir: Path, *, console_level: int = logging.INFO) -> logging.Logger:
    """为一次实验运行配置控制台和文件日志。

    控制台输出到 stderr，文件输出到 ``<output_dir>/experiment.log``（追加模式）。
    返回包级 logger ``"mpm"`` 供框架各处使用。重复调用不会重复添加 handler。

    Parameters
    ----------
    output_dir : Path
        实验输出目录，日志文件将写入此目录下的 ``experiment.log``。
    console_level : int, optional
        控制台日志级别，默认 ``logging.INFO``。

    Returns
    -------
    logging.Logger
        包级 logger ``"mpm"``。
    """
    logger = logging.getLogger("mpm")

    output_dir.mkdir(parents=True, exist_ok=True)
    key = str(output_dir.resolve())
    if key in _initialized:
        return logger
    _initialized.add(key)

    logger.setLevel(logging.DEBUG)

    # --- 控制台 handler (stderr) ---
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console_handler)

    # --- 文件 handler (完整日志写入实验输出目录) ---
    file_path = output_dir / "experiment.log"
    file_handler = logging.FileHandler(str(file_path), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(file_handler)

    # 防止第三方库的 "No handlers could be found" 警告
    logging.getLogger().addHandler(logging.NullHandler())

    logger.info("Logging initialized; file output: %s", file_path)
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取 ``mpm`` 命名空间下的子 logger。

    Parameters
    ----------
    name : str
        子 logger 名称，如 ``"experiment"``、``"cnn"``。

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(f"mpm.{name}")