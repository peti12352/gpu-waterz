#!/usr/bin/env bash
# Sample CPU while a named process pattern is alive. JSONL, one object per line.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PAT="${1:-scripts/n20_d2.py}"
OUT="${2:-data/cache/N20_RES.jsonl}"
PY="${ROOT}/.venv/bin/python"
while pgrep -f "$PAT" >/dev/null; do
  "$PY" -u scripts/n20_res.py jsonl "$OUT" "$PAT"
  sleep 10
done
