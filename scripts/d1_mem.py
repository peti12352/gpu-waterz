#!/usr/bin/env python3
"""D1: where the VRAM goes, measured, and what that implies at 2.16 Gvox.

ws.cu and parhac_d.cu already wrap cudaMalloc in a tracker (ws_mem_peak,
agg_mem_peak). Nothing read them. This does, at two crop sizes, and fits a
straight line in voxel count so the 2.16 Gvox figure is an extrapolation of
measurements rather than a hand-count of 148 allocation sites that would drift
the moment anyone adds one.

Both peaks are lower bounds: thrust allocates its own scratch for sort_by_key
and reduce_by_key outside the tracked path, as the note in parhac_d.cu says.
The RAG hash table is computed instead of measured because it is not tracked,
and it is exactly determined by max_edges: next_pow2(2*max_edges) slots of 16 B.

Two crops rather than one because the agglomeration allocates per node and per
edge as well as per voxel, and only a fit separates the slope from the fixed
floor that _max_edges imposes below ~360 Mvox.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
CACHE = ROOT / "data/cache"
GIB = 1024 ** 3
TARGETS = [("graded 2.16 Gvox", 2_160_000_000), ("scaling 1.44 Gvox", 1_440_000_000)]
CARD_GIB = {"3090 Ti": 24.0, "5090": 32.0}


def next_pow2(x):
    p = 1
    while p < x:
        p <<= 1
    return p


def rag_table_bytes(nvox, S):
    """Slot is {uint64 key, uint32 isum, uint32 n} = 16 B, plus the uint32
    occupancy flags and the prefix sum over them."""
    cap = max(1024, next_pow2(2 * S._max_edges(nvox)))
    return cap * 16 + cap * 4


def edge_array_bytes(nvox, S, wide=True):
    """u, v, sm, ct as handed from the RAG to the agglomeration."""
    per = 4 + 4 + (8 + 8 if wide else 4 + 4)
    return S._max_edges(nvox) * per


def measure(S, aff_path, crop, thr):
    libw = ctypes.CDLL(str(S._WS))
    libp = ctypes.CDLL(str(S._PARHAC_D))
    for lib, name in ((libw, "ws"), (libp, "agg")):
        for fn in (f"{name}_mem_reset", f"{name}_mem_peak"):
            getattr(lib, fn).restype = ctypes.c_size_t
    cz, cy, cx = crop
    with h5py.File(aff_path, "r") as f:
        aff = f["affinity"][:, :cz, :cy, :cx]
    aff_u8 = np.ascontiguousarray(S._as_u8(aff))
    nvox = int(aff_u8[0].size)

    libw.ws_mem_reset()
    libp.agg_mem_reset()
    aff_d = S.DevBuf.from_host(aff_u8)
    out = S.segment_d(aff_d, [thr], return_device=True)
    for b in out:
        b.free()
    aff_d.free()

    def by_line(lib, fn):
        f = getattr(lib, fn)
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.POINTER(ctypes.c_int),
                      ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
        cap = 256
        lines = (ctypes.c_int * cap)()
        byts = (ctypes.c_size_t * cap)()
        got = min(cap, f(lines, byts, cap))
        return sorted(((int(byts[i]), int(lines[i])) for i in range(got)),
                      reverse=True)

    breakdown = by_line(libp, "agg_mem_peak_lines")
    ws_breakdown = by_line(libw, "ws_mem_peak_lines")
    return {
        "crop": [cz, cy, cx],
        "nvox": nvox,
        "nedge": int(S.LAST_NEDGE),
        "ws_peak": int(libw.ws_mem_peak()),
        "agg_peak": int(libp.agg_mem_peak()),
        "rag_table": rag_table_bytes(nvox, S),
        "edges": edge_array_bytes(nvox, S),
        "agg_peak_by_line": [[ln, b] for b, ln in breakdown],
        "ws_peak_by_line": [[ln, b] for b, ln in ws_breakdown],
    }


def fit(xs, ys):
    """Least squares slope and intercept, so the extrapolation shows both the
    per-voxel cost and the fixed floor."""
    a, b = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    return float(a), float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aff", default=str(AFF))
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--crops", nargs="+", default=["24,512,512", "48,768,768"])
    args = ap.parse_args()

    import segment as S

    runs = []
    for c in args.crops:
        crop = [int(v) for v in c.split(",")]
        r = measure(S, args.aff, crop, args.threshold)
        runs.append(r)
        print(f"D1 {r['crop']} {r['nvox'] / 1e6:.1f} Mvox  "
              f"ws={r['ws_peak'] / GIB:.3f} GiB ({r['ws_peak'] / r['nvox']:.2f} B/vox)  "
              f"agg={r['agg_peak'] / GIB:.3f} GiB ({r['agg_peak'] / r['nvox']:.2f} B/vox)  "
              f"rag_table={r['rag_table'] / GIB:.3f} GiB", flush=True)

    # Attribute each peak to source lines, on the largest crop, so effort goes
    # where the bytes are.
    big = runs[-1]
    for what, srcfile, key, unit, denom in (
        ("agg", "csrc/parhac_d.cu", "agg_peak_by_line", "B/edge", big["nedge"]),
        ("ws", "csrc/ws.cu", "ws_peak_by_line", "B/vox", big["nvox"]),
    ):
        src = (ROOT / srcfile).read_text().splitlines()
        total = big[f"{what}_peak"]
        print(f"D1 {what} peak {total / GIB:.3f} GiB by allocation site at "
              f"{big['nvox'] / 1e6:.1f} Mvox:", flush=True)
        for ln, b in big[key][:12]:
            text = src[ln - 1].strip() if 0 < ln <= len(src) else "?"
            print(f"D1   {b / total * 100:5.1f}%  {b / 1e6:8.1f} MB  "
                  f"{b / denom:6.1f} {unit}  L{ln}  {text}", flush=True)

    xs = [r["nvox"] for r in runs]
    ws_a, ws_b = fit(xs, [r["ws_peak"] for r in runs])
    ag_a, ag_b = fit(xs, [r["agg_peak"] for r in runs])
    ed_a, ed_b = fit(xs, [r["nedge"] for r in runs])
    print(f"D1 fit ws  = {ws_a:.2f} B/vox + {ws_b / 1e6:.1f} MB\n"
          f"D1 fit agg = {ag_a:.2f} B/vox + {ag_b / 1e6:.1f} MB\n"
          f"D1 fit edges = {ed_a:.4f} per vox + {ed_b / 1e6:.2f} M", flush=True)

    report = {"runs": runs, "ws_b_per_vox": ws_a, "agg_b_per_vox": ag_a,
              "edges_per_vox": ed_a, "targets": {}}
    for name, nvox in TARGETS:
        aff_b = 3 * nvox
        seg_b = 4 * nvox
        ws_b_t = ws_a * nvox + ws_b
        agg_b_t = ag_a * nvox + ag_b
        rag_b = rag_table_bytes(nvox, S)
        wide = edge_array_bytes(nvox, S, wide=True)
        narrow = edge_array_bytes(nvox, S, wide=False)
        nedge = ed_a * nvox + ed_b

        # HSlot is narrowed already and the entry scratch is gone, so what is
        # left inside is four 8-byte payload temporaries: tsm, tct, csm, cct.
        # 4 B/edge each. dkey and dkeyo are also 8 B/edge but are packed 64-bit
        # sort keys, not payloads, so they cannot narrow.
        agg_narrow_save = nedge * 16

        # A peak is a stage maximum, not a sum. The fragment/label buffer is the
        # only thing that spans all stages; the affinity dies with the RAG, and
        # the watershed scratch is gone before the RAG table exists.
        peak_ws = aff_b + seg_b + ws_b_t
        peak_rag = aff_b + seg_b + rag_b + wide
        peak_agg = seg_b + wide + agg_b_t
        peak_agg_thin = seg_b + narrow + agg_b_t - agg_narrow_save
        t = {
            "nvox": nvox, "nedge": nedge,
            "aff_gib": aff_b / GIB, "seg_gib": seg_b / GIB,
            "ws_scratch_gib": ws_b_t / GIB,
            "rag_table_gib": rag_b / GIB,
            "edges_wide_gib": wide / GIB, "edges_narrow_gib": narrow / GIB,
            "agg_gib": agg_b_t / GIB,
            "agg_narrow_save_gib": agg_narrow_save / GIB,
            "peak_ws_gib": peak_ws / GIB,
            "peak_rag_gib": peak_rag / GIB,
            "peak_agg_gib": peak_agg / GIB,
            "peak_agg_thin_gib": peak_agg_thin / GIB,
        }
        report["targets"][name] = t
        print(f"\nD1 {name} ({nvox / 1e9:.2f} Gvox, {nedge / 1e6:.0f} M edges)\n"
              f"D1   affinity            {aff_b / GIB:7.2f} GiB  freed after rag\n"
              f"D1   fragments/labels    {seg_b / GIB:7.2f} GiB  one buffer, spans all stages\n"
              f"D1   ws scratch          {ws_b_t / GIB:7.2f} GiB\n"
              f"D1   rag table           {rag_b / GIB:7.2f} GiB\n"
              f"D1   edge arrays wide    {wide / GIB:7.2f} GiB\n"
              f"D1   edge arrays narrow  {narrow / GIB:7.2f} GiB\n"
              f"D1   agg tracked peak    {agg_b_t / GIB:7.2f} GiB  lower bound\n"
              f"D1   agg narrowing saves {agg_narrow_save / GIB:7.2f} GiB\n"
              f"D1   stage peak ws       {peak_ws / GIB:7.2f} GiB\n"
              f"D1   stage peak rag      {peak_rag / GIB:7.2f} GiB\n"
              f"D1   stage peak agg      {peak_agg / GIB:7.2f} GiB\n"
              f"D1   stage peak agg thin {peak_agg_thin / GIB:7.2f} GiB",
              flush=True)
        for card, gib in CARD_GIB.items():
            for tag, agg in (("wide", peak_agg), ("narrow", peak_agg_thin)):
                worst = max(peak_ws, peak_rag, agg) / GIB
                binding = max((peak_ws, "ws"), (peak_rag, "rag"), (agg, "agg"))[1]
                print(f"D1   {card} {gib:.0f} GiB {tag}: worst {worst:.2f} GiB "
                      f"({binding}) {'FITS' if worst < gib else 'OVER'}; "
                      f"slabbed ws, worst "
                      f"{max(peak_rag, agg) / GIB:.2f} GiB "
                      f"{'FITS' if max(peak_rag, agg) / GIB < gib else 'OVER'}",
                      flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / "d1_mem.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nD1 wrote {dest.name}", flush=True)
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
