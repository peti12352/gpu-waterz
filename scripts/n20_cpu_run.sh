#!/usr/bin/env bash
# N20 CPU campaign. Wall vs 1679.9 only after n20_res says idle.
# Correctness jobs may run dirty; timing jobs wait.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export N20_GPU_RUN=0
PY="${ROOT}/.venv/bin/python"
LOG="${ROOT}/notes/N20_CPU_RUN.log"
mkdir -p notes data/cache

wait_idle() {
  echo "wait idle before $1 $(date -Is)" | tee -a "$LOG"
  "$PY" -u scripts/n20_res.py wait | tee -a "$LOG"
}

{
  echo "host=$(hostname) date=$(date -Is)"
  echo "N20_GPU_RUN=${N20_GPU_RUN}"
  "$PY" -u scripts/n20_res.py snap
  "$PY" -u scripts/n20_unit.py
  echo "n20_unit exit=$?"
  "$PY" -u scripts/n20_d1.py
  wait_idle N20_D2
  taskset -c 2 "$PY" -u scripts/n20_d2.py
  echo "n20_d2 exit=$?"
  wait_idle N20_X4
  taskset -c 2 "$PY" -u scripts/n20_x.py x4
  echo "n20_x4 exit=$? (1 expected if complete-link FAILs VOI)"
  wait_idle N20_X5
  taskset -c 2 "$PY" -u scripts/n20_x.py x5
  echo "n20_x5 exit=$? (1 expected if WPGMA FAILs VOI)"
  wait_idle N20_D3
  taskset -c 2 "$PY" -u scripts/n20_d3.py
  echo "n20_d3 exit=$?"
  "$PY" -u scripts/n20_rnn.py
  echo "n20_rnn exit=$?"
  wait_idle N20_LU2
  taskset -c 2 "$PY" -u scripts/n20_lu2.py
  echo "n20_lu2 exit=$?"
  "$PY" -u scripts/n20_gpu_prep.py
  echo "n20_gpu_prep exit=$? (must not launch)"
  echo "done $(date -Is)"
} 2>&1 | tee -a "$LOG"
