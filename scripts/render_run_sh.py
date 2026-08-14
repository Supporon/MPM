#!/usr/bin/env python3
"""校验实验 YAML，并生成用于执行该实验的确定性 Shell 命令。"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="Optional .sh path; stdout is used when omitted")
    args = parser.parse_args()
    config = load_config(args.config)
    relative_config = Path(args.config)
    command = f"python run.py --config {shlex.quote(str(relative_config))}"
    content = "#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        output.chmod(output.stat().st_mode | 0o111)
        print(f"Wrote {output} for experiment {config.name}")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
