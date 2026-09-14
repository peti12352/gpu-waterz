#!/usr/bin/env bash
# Product gate on the public API: four-T VOI + T=0.3 identity.
# Optional --216 times [3,375,2400,2400] if that HDF5 is present.
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --216) ARGS+=(--216); shift ;;
    -h|--help)
      echo "Usage: $0 [--216]"
      echo "  default: CREMI-A val four-T VOI + two-run identity at T=0.3"
      echo "  --216: also time [3,375,2400,2400] if present"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

export WATERZ_UF_ALGO="${WATERZ_UF_ALGO:-3}"
export WATERZ_HOST_PARK="${WATERZ_HOST_PARK:-0}"
export WATERZ_AFF_PARK="${WATERZ_AFF_PARK:-0}"
export WATERZ_AGG_LEVERS="${WATERZ_AGG_LEVERS:-15}"
export WATERZ_FOLD_FLATTEN=1
export WATERZ_SHARE_OFF=1
export WATERZ_HOOK_ROOT=1
export WATERZ_FUSE_DIRTY=1
export WATERZ_NLIVE_ARITH=1
export WATERZ_EMIT_HOLES=1

echo "eval: product env (docs/porting.md); dual-eps four-T=0.08 / T=0.3=0.40"
exec "$PY" -u scripts/check.py "${ARGS[@]}"
