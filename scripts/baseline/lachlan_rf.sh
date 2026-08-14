#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python run.py --config configs/experiments/lachlan_rf_baseline.yaml "$@"
