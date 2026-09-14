#!/usr/bin/env python3
"""Optional: 8 unique official make_big.mirror flip triples. nfrag + fingerprint.

Explains TASK 26.02 M vs 12x2.175 M. Not a speed path. Idle-5090. Val tiles
only: no fused 2.16 allocation.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from p1_make_big_indep import (  # noqa: E402
    AFF, CACHE, GpuWs, card_busy, fingerprint, gpu_state, load_official_mirror,
    nfrag_bg,
)
from task_gate import FRAGMENTS_VAL  # noqa: E402

OUT = CACHE / "b_8flip.json"


def main():
    print("B 8 official mirror triples. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"B REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    mirror = load_official_mirror()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    ws = GpuWs()
    rows = []
    for flips in itertools.product((False, True), repeat=3):
        tile = aff if flips == (False, False, False) else mirror(aff, flips)
        seg, meta = ws.run(tile)
        nfrag, bg = nfrag_bg(seg)
        fp, nsz, _ = fingerprint(seg)
        row = {
            "flips": list(flips),
            "nfrag": nfrag,
            "bg": bg,
            "fingerprint": fp,
            "n_sizes": nsz,
            "peak_bytes": int(meta["peak_bytes"]),
        }
        rows.append(row)
        print(f"B 8flip {flips} nfrag={nfrag} bg={bg} fp={fp}", flush=True)
        del seg, tile
    nfrags = [r["nfrag"] for r in rows]
    fps = [r["fingerprint"] for r in rows]
    doc = {
        "claim": "not a 2 Gvox/s number",
        "n_unique_nfrag": len(set(nfrags)),
        "n_unique_fp": len(set(fps)),
        "identity_nfrag": nfrags[0],
        "fragments_val": FRAGMENTS_VAL,
        "sum_8_nfrag": int(sum(nfrags)),
        "times_12_identity": int(nfrags[0] * 12),
        "official_fused_nfrag": 26_023_852,
        "rows": rows,
        "gpu": gpu_state(),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"B 8flip unique_nfrag={doc['n_unique_nfrag']} unique_fp={doc['n_unique_fp']} "
        f"sum8={doc['sum_8_nfrag']} 12x_id={doc['times_12_identity']} "
        f"official={doc['official_fused_nfrag']} -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
