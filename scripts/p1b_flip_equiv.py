#!/usr/bin/env python3
"""P1b: is WS(official mirror(aff)) a spatial flip of WS(aff)?

P1 fingerprints already differ, so the partitions are not the same even up
to relabel. This measures where they disagree: a seam-halo-only miss still
leaves a cheap repair; a volume-wide miss kills 1x-val + stamp.

Greengoblin. Not a 2 Gvox/s claim. Do not copy volumes off this machine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from p1_make_big_indep import (  # noqa: E402
    AFF, CACHE, GpuWs, card_busy, fingerprint, gpu_state,
    load_official_mirror, nfrag_bg, same_partition,
)

OUT = CACHE / "p1b_flip_equiv.json"


def dump(doc):
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(OUT)


def plane_disagree(a, b, axis):
    """Fraction of voxels that differ after id-remap, per plane along axis."""
    a = np.ascontiguousarray(a, dtype=np.uint32)
    b = np.ascontiguousarray(b, dtype=np.uint32)
    ar, br = a.ravel(), b.ravel()
    am = int(ar.max()) + 1
    to_b = np.zeros(am, dtype=np.int64)
    to_b[ar] = br
    mapped = to_b[a]
    # If remap is inconsistent, mapped!=b on those voxels too.
    diff = mapped != b
    n = a.shape[axis]
    rates = []
    for i in range(n):
        sl = [slice(None)] * 3
        sl[axis] = i
        plane = diff[tuple(sl)]
        rates.append(float(plane.mean()) if plane.size else 0.0)
    return {
        "n_disagree": int(diff.sum()),
        "frac": float(diff.mean()),
        "first_nonzero_plane": next((i for i, r in enumerate(rates) if r > 0), None),
        "last_nonzero_plane": next(
            (n - 1 - i for i, r in enumerate(reversed(rates)) if r > 0), None
        ),
        "max_plane_frac": float(max(rates) if rates else 0.0),
        "n_planes_touched": int(sum(1 for r in rates if r > 0)),
        "halo1_frac": float(
            (rates[0] + rates[-1]) / 2.0 if n >= 2 else rates[0] if rates else 0.0
        ),
        "interior_frac": float(np.mean(rates[1:-1]) if n > 2 else 0.0),
    }


def main():
    print("P1b flip-equivariance. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"P1b REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    mirror = load_official_mirror()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    ws = GpuWs()
    val, _ = ws.run(aff)
    fp0, _, _ = fingerprint(val)
    doc = {
        "claim": "not a 2 Gvox/s number",
        "val_fingerprint": fp0,
        "flip_preserves_histogram": True,
        "axes": {},
        "pass_1x_stamp": False,
        "halo_only": False,
    }
    dump(doc)

    halo_only_all = True
    for axis, name in enumerate("zyx"):
        flips = [False, False, False]
        flips[axis] = True
        mir_aff = mirror(aff, tuple(flips))
        mir, meta = ws.run(mir_aff)
        flipped = np.ascontiguousarray(np.flip(val, axis=axis))
        fp_mir, _, _ = fingerprint(mir)
        fp_flip, _, _ = fingerprint(flipped)
        eq = same_partition(flipped, mir)
        n0, _ = nfrag_bg(val)
        n1, _ = nfrag_bg(mir)
        spat = plane_disagree(flipped, mir, axis)
        # Halo-only: interior planes have zero disagreement after remap.
        halo = bool(spat["interior_frac"] == 0.0 and spat["n_disagree"] > 0)
        halo_only_all = halo_only_all and halo
        row = {
            "axis": name,
            "mir_nfrag": n1,
            "val_nfrag": n0,
            "same_partition_flip_vs_mir": eq,
            "mir_fingerprint": fp_mir,
            "flip_fingerprint": fp_flip,
            "fingerprints_equal": fp_mir == fp_flip,
            "ws": meta,
            "spatial": spat,
            "halo_only": halo,
        }
        doc["axes"][name] = row
        dump(doc)
        print(
            f"  {name}: flip==mir {eq} fp_eq={fp_mir == fp_flip} "
            f"disagree={spat['n_disagree']} "
            f"interior_frac={spat['interior_frac']:.4e} "
            f"planes={spat['n_planes_touched']}/{val.shape[axis]} "
            f"halo_only={halo}",
            flush=True,
        )
        del mir_aff, mir, flipped

    doc["halo_only"] = bool(halo_only_all)
    doc["pass_1x_stamp"] = False
    doc["gpu_final"] = gpu_state()
    dump(doc)
    print(
        f"P1b 1x-stamp illegal (histograms). halo_only={halo_only_all} -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
