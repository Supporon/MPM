#!/usr/bin/env python3
"""批量绘制所有 NSW 运行的结果（封装 src.plotting.plot_run）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting.plot import plot_run


def main() -> None:
    runs = sorted(ROOT.glob("outputs/nsw_*"))
    if not runs:
        print("未找到 nsw_* 运行目录", file=sys.stderr)
        return
    for run in runs:
        out = plot_run(run)
        if out is None:
            print(f"跳过 {run.name}（缺 target_probs.csv 或模型）", file=sys.stderr)
        else:
            print(f"写入 {out}")
    print("完成。")


if __name__ == "__main__":
    main()
