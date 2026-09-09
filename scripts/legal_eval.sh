#!/usr/bin/env bash
# Legal-stack capture: N17 env + EMIT_HOLES, parks off, dual-ε.
# Default: identity diagnostic + T=0.3 VOI 2-run + four-T ε=0.08 (no 2.16).
# Optional: --216 for official make_big volume timing (not a 3090 Ti grade).
#
# Not a 2 Gvox/s claim. Not 3090 Ti.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

RUN_216=0
FORCE=0
NAME="LEGAL_FOUR"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --216) RUN_216=1; shift ;;
    --force) FORCE=1; shift ;;
    --name) NAME="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--216] [--force] [--name TAG]"
      echo "  default: voi_only without 2.16 (four-T + T=0.3 VOI + identity diag)"
      echo "  --216: also time official [3,375,2400,2400] if present"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Legal floor (process env; C++ product defaults stay 0)
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
# Dual-ε: four-T always 0.08 in n15_four.py; T=0.3 speed path defaults ε=0.40
# unless WATERZ_AGG_EPS overrides.

ARGS=( -u scripts/n19_voi_gate.py --mode voi_only --name "$NAME" --exp-id "$NAME" )
if [[ "$RUN_216" -eq 0 ]]; then
  ARGS+=( --no-216 )
fi
if [[ "$FORCE" -eq 1 ]]; then
  ARGS+=( --force )
fi

echo "legal_eval: claim=not a 2 Gvox/s number; not 3090 Ti"
echo "legal_eval: env FOLD+SHARE_OFF+HOOK+FUSE_DIRTY+NLIVE_ARITH+EMIT_HOLES parks=0 UF=3"
echo "legal_eval: dual-ε four-T=0.08 / T=0.3 speed=0.40"
exec "$PY" "${ARGS[@]}"
