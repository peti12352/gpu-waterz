#!/usr/bin/env python3
"""N17 T2: EMIT_HOLES on N16 deep stack. Skip vacate+fill O(nscan)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from n17_gate import CACHE, DEEP, run_stack  # noqa: E402

NOTE = ROOT / "notes/N17_T2.md"
OUT = CACHE / "n17_t2.json"
EXTRA = {**DEEP, "WATERZ_EMIT_HOLES": "1"}


def main():
    rc, doc = run_stack("T2 emit_holes+deep", EXTRA, "n17_t2_four")
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    m1 = (doc.get("voi") or {}).get("m1") or {}
    ident = doc.get("ident") or {}
    four = doc.get("four") or {}
    run216 = doc.get("run216") or {}
    NOTE.write_text(
        "# N17 T2 emit into rewrite holes\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. "
        "WATERZ_EMIT_HOLES=1 + deep. Replaces k_vacate_kept + k_fill_holes.\n\n"
        f"- ident={ident.get('identity')} run2={ident.get('run2_array_equal')} "
        f"ndiff={ident.get('ndiff_raw')}\n"
        f"- voi ok={m1.get('ok')} split={m1.get('split')} merge={m1.get('merge')}\n"
        f"- four_pass={four.get('four_pass')}\n"
        f"- 2.16 WS={doc['ws_ms']:.1f} agg={doc['agg_ms']:.1f} "
        f"cut_agg={doc['cut_agg']:.3f} nlab={run216.get('nlab')} "
        f"vs deep agg {doc['deep_agg']:.1f} gate {doc['agg_gate']:.1f}\n"
        f"- keep_default={doc['keep_default']}\n"
    )
    print(f"N17 T2 keep={doc['keep_default']} agg={doc['agg_ms']:.1f} -> {OUT}",
          flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
