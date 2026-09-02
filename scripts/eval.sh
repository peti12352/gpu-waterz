#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
"$PY" src/segment.py data/ws_bounty/cremiA_val/affinity.h5 --out-dir data/ws_bounty --thresholds 0.2 0.3 0.4 0.5
"$PY" data/ws_bounty/baseline/run_baseline.py --candidate \
  data/ws_bounty/mine_thr0.2.h5 data/ws_bounty/mine_thr0.3.h5 \
  data/ws_bounty/mine_thr0.4.h5 data/ws_bounty/mine_thr0.5.h5
"$PY" scripts/g5_det.py
"$PY" scripts/bench.py
