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
_managed_handlers: list[logging.Handler] = []


def setup_logging(output_dir: Path, *, console_level: int = logging.INFO) -> logging.Logger:
    """为一次实验运行配置控制台和文件日志。

    控制台输出到 stderr，文件输出到 ``<output_dir>/experiment.log``（追加模式）。
    返回包级 logger ``"mpm"`` 供框架各处使用。重复调用不会重复添加 handler；
    切换到新的输出目录时会关闭并移除上一轮本框架添加的 handler，避免日志
    累积并重复写入此前的运行目录。

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

    # 关闭并移除上一轮由本框架添加的 handler，避免跨运行目录累积
    for handler in _managed_handlers:
        logger.removeHandler(handler)
        handler.close()
    _managed_handlers.clear()

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

    _managed_handlers.extend([console_handler, file_handler])

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


def close_logging() -> None:
    """关闭并移除由 :func:`setup_logging` 管理的 handler，释放文件句柄。

    实验结束（含失败路径）后应调用，否则 FileHandler 会持有输出目录下的
    ``experiment.log`` 文件句柄，导致 Windows 下 TemporaryDirectory 清理时
    出现 WinError 32（第 7 节日志生命周期问题）。同一 logger 的 handler 边界
    已由 ``_managed_handlers`` 跟踪，这里只清理框架自身添加的 handler。
    """
    logger = logging.getLogger("mpm")
    for handler in _managed_handlers:
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover - 关闭失败不阻断收尾
            pass
    _managed_handlers.clear()
    _initialized.clear()