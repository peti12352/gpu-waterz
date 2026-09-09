#!/bin/bash
# N18 full ladder on greengoblin after freeing GPU.
# Not a 2 Gvox/s claim. Not 3090 Ti.
# Experiment failures must not abort the ladder.
set -uo pipefail
cd /home/v/proj/petya/waterz
bash scripts/n18_free_gpu.sh || true
PY=.venv/bin/python

run_step() {
  local title="$1"; shift
  echo "=== ${title} ==="
  bash scripts/n18_free_gpu.sh || true
  set +e
  "$@" 2>&1 | tee "/tmp/n18_${title}.log"
  local rc=${PIPESTATUS[0]}
  echo "=== ${title} rc=${rc} ==="
  return 0
}

run_step A1 $PY -u scripts/n18_run.py a1
run_step A2A5 $PY -u scripts/n18_run.py a
run_step B1 $PY -u scripts/n18_b1_binq.py
run_step B2 $PY -u scripts/n18_b2_eps.py
run_step B3 $PY -u scripts/n18_b3_inner.py
run_step C $PY -u scripts/n18_3090_grade.py

echo "=== DONE ==="
ls -la notes/N18_*.md data/cache/n18*.json data/cache/N18_*.json 2>/dev/null | head -40
