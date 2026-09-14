#!/bin/bash
# N19 ladder. Experiment failures must not abort.
# Not a 2 Gvox/s claim. Not 3090 Ti.
set -uo pipefail
cd /home/v/proj/petya/waterz
PY=.venv/bin/python
free_gpu() { bash scripts/n18_free_gpu.sh || true; }

run() {
  local title="$1"; shift
  echo "=== ${title} ==="
  free_gpu
  set +e
  "$@" 2>&1 | tee "/tmp/n19_${title}.log"
  echo "=== ${title} rc=${PIPESTATUS[0]} ==="
}

# Watchdog: kill vllm only
( while true; do
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
      cmd=$(ps -p "$p" -o args= 2>/dev/null || true)
      case "$cmd" in *vllm*|*VLLM*|*EngineCore*) kill -9 "$p" 2>/dev/null ;; esac
    done
    sleep 8
  done ) &
WDOG=$!

free_gpu
run I0_NSYS $PY -u scripts/n19_nsys.py
run I0_REPRO $PY -u scripts/n19_fast_kill.py --exp-id N19_I0_REPRO --mode voi_only

# W track
run W1 $PY -u scripts/n19_fast_kill.py --exp-id N19_W1 --mode voi_only \
  --extra-json '{"WATERZ_PLAYNE_HOOK":"1"}' --owner-substr hook --kill-reason "four-T FAIL or WS cut"
run W3 $PY -u scripts/n19_fast_kill.py --exp-id N19_W3 --mode voi_only \
  --extra-json '{"WATERZ_VCOUNT_PRIV":"1"}' --owner-substr count_v --kill-reason "four-T FAIL or WS cut"

# H track
run H1 $PY -u scripts/n19_h1_binladder.py
run H2 $PY -u scripts/n19_h2_nng.py
run H3 $PY -u scripts/n19_h3_rnn.py
run H4 $PY -u scripts/n19_h4_chunk.py

# X track
run X1 $PY -u scripts/n19_x1_pruf.py
run X2 $PY -u scripts/n19_x2_gasp.py
run X3 $PY -u scripts/n19_x3_rama.py

# C
run C $PY -u scripts/n19_3090_grade.py

# Stop notes
$PY - <<'PY'
import json
from pathlib import Path
CACHE = Path("data/cache")
NOTE = Path("notes")
CLAIM = "not a 2 Gvox/s number; not 3090 Ti"
ws_ok = []
for p in CACHE.glob("N19_W*.json"):
    d = json.loads(p.read_text())
    if (d.get("four") or {}).get("four_pass") and (d.get("ws_ms") or 0) <= 900:
        ws_ok.append((p.name, d.get("ws_ms")))
if not ws_ok:
    (NOTE / "N19_W_STOP.md").write_text(
        f"# N19 W-stop\n\n{CLAIM}.\n\nNo WS≤900 with four-T PASS. Freeze WS.\n"
    )
agg_ok = []
for p in list(CACHE.glob("N19_H*.json")):
    d = json.loads(p.read_text())
    if not d.get("kill") and (d.get("agg_ms") or 0) and d.get("agg_ms") <= 1000:
        agg_ok.append(p.name)
if not agg_ok:
    (NOTE / "N19_H_STOP.md").write_text(
        f"# N19 H-stop\n\n{CLAIM}.\n\nMEAN-exact local maximum; no agg≤1000 with four-T.\n"
        f"Remaining: WS~1315 agg~1683 e2e~3104; need ~2.9x to 1080 ms.\n"
    )
(NOTE / "N19_SUMMARY.md").write_text(
    f"# N19 summary\n\n{CLAIM}.\n\nSee N19_W_STOP / N19_H_STOP / LOG.\n"
)
print("stop notes written", "ws_ok", ws_ok, "agg_ok", agg_ok)
PY

kill $WDOG 2>/dev/null || true
echo "=== DONE ==="
ls notes/N19_*.md | wc -l
