#!/usr/bin/env bash
# Compatibility name. Prefer scripts/eval.sh.
exec "$(cd "$(dirname "$0")" && pwd)/eval.sh" "$@"
