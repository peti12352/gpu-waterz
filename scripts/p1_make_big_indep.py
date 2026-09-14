#!/usr/bin/env python3
"""P1: official make_big 2-tile independence. Greengoblin only. Not a 2 Gvox/s claim.

Imports official data/ws_bounty/make_big.py::mirror: not the N4 reconstruction.
For each axis, builds a 2-tile concat and checks:

  1. seam plane of the join channel is identically zero
  2. GPU WS: no fragment id on both sides of the seam (except 0)
  3. per-tile nfrag of the mirrored tile equals val; concat nfrag == n0+n1
  4. tile slices of concat WS == standalone WS up to id remap
  5. every cross-seam face has aff==0, so T=0.3 cannot merge across tiles

(5) is the agglomeration-independence claim without running ParHAC on 360 Mvox:
a cross-seam RAG edge is composed only of those faces.

Refuses if the 5090 is busy. Writes data/cache/p1_make_big_indep.json incrementally.
Do not copy volumes off this machine.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from task_gate import AFF_HIGH, AFF_LOW, BG_VAL_MEASURED, FRAGMENTS_VAL  # noqa: E402

CACHE = ROOT / "data/cache"
AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
MAKE_BIG = ROOT / "data/ws_bounty/make_big.py"
WS_SO = ROOT / "src/libws_gpu.so"
OUT = CACHE / "p1_make_big_indep.json"
GIB = 1024.0 ** 3
T_MERGE = 0.3


def load_official_mirror():
    if not MAKE_BIG.is_file():
        raise FileNotFoundError(
            f"official make_big.py missing at {MAKE_BIG}; run this on greengoblin"
        )
    spec = importlib.util.spec_from_file_location("ws_bounty_make_big", MAKE_BIG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.mirror


def card_busy():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return [f"nvidia-smi unavailable: {e}"]
    if r.returncode != 0:
        return [f"nvidia-smi rc={r.returncode}"]
    me = str(os.getpid())
    return [ln for ln in r.stdout.splitlines()
            if ln.strip() and ln.split(",")[0].strip() != me]


def gpu_state():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def dump(doc):
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(OUT)


def fingerprint(seg):
    counts = np.bincount(seg.ravel())
    sizes = np.sort(counts[1:][counts[1:] > 0]).astype(np.int64)
    h = hashlib.sha256()
    h.update(sizes.tobytes())
    return h.hexdigest()[:32], int(sizes.size), int(counts[0]) if counts.size else 0


def same_partition(a, b):
    """True iff a and b are the same partition (background 0 fixed)."""
    if a.shape != b.shape:
        return False
    a = np.ascontiguousarray(a, dtype=np.uint32).ravel()
    b = np.ascontiguousarray(b, dtype=np.uint32).ravel()
    if not np.array_equal(a == 0, b == 0):
        return False
    am = int(a.max()) + 1
    bm = int(b.max()) + 1
    to_b = np.zeros(am, dtype=np.int64)
    to_a = np.zeros(bm, dtype=np.int64)
    to_b[a] = b
    to_a[b] = a
    return bool(np.array_equal(to_b[a], b) and np.array_equal(to_a[b], a))


def nfrag_bg(seg):
    counts = np.bincount(seg.ravel())
    nfrag = int((counts[1:] > 0).sum()) if counts.size > 1 else 0
    bg = int(counts[0]) if counts.size else 0
    return nfrag, bg


def spanning_ids(lab, axis, mid):
    sl0 = [slice(None)] * 3
    sl1 = [slice(None)] * 3
    sl0[axis] = slice(0, mid)
    sl1[axis] = slice(mid, None)
    left = lab[tuple(sl0)]
    right = lab[tuple(sl1)]
    mx = int(max(int(left.max()), int(right.max()))) + 1
    cl = np.bincount(left.ravel(), minlength=mx) > 0
    cr = np.bincount(right.ravel(), minlength=mx) > 0
    both = cl & cr
    both[0] = False
    ids = np.flatnonzero(both)
    return int(ids.size), [int(x) for x in ids[:8]]


def take_plane(arr, axis, index):
    sl = [slice(None)] * arr.ndim
    sl[axis if arr.ndim == 3 else axis + 1] = index
    return arr[tuple(sl)]


class GpuWs:
    def __init__(self):
        if not WS_SO.is_file():
            raise FileNotFoundError(f"missing {WS_SO}")
        lib = ctypes.CDLL(str(WS_SO))
        lib.watershed_gpu_e9.restype = ctypes.c_int
        lib.watershed_gpu_e9.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_uint32),
        ]
        for name in ("reset", "peak", "cur"):
            fn = getattr(lib, f"ws_mem_{name}", None)
            if fn is None:
                continue
            fn.restype = None if name == "reset" else ctypes.c_size_t
            fn.argtypes = []
        self.lib = lib

    def run(self, aff_u8):
        z, y, x = (int(v) for v in aff_u8.shape[1:])
        aff_u8 = np.ascontiguousarray(aff_u8, dtype=np.uint8)
        seg = np.zeros((z, y, x), dtype=np.uint32)
        if hasattr(self.lib, "ws_mem_reset"):
            self.lib.ws_mem_reset()
        t0 = time.perf_counter()
        nfrag_ret = self.lib.watershed_gpu_e9(
            aff_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            z, y, x, ctypes.c_float(AFF_LOW), ctypes.c_float(AFF_HIGH),
            seg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        )
        ms = (time.perf_counter() - t0) * 1e3
        # API returns nfrag (>=1). Negative/zero is a real failure.
        if nfrag_ret <= 0:
            raise RuntimeError(
                f"watershed_gpu_e9 nfrag={nfrag_ret} shape={aff_u8.shape}"
            )
        peak = int(self.lib.ws_mem_peak()) if hasattr(self.lib, "ws_mem_peak") else 0
        leaked = int(self.lib.ws_mem_cur()) if hasattr(self.lib, "ws_mem_cur") else 0
        return seg, {"ms": ms, "peak_bytes": peak, "leaked_bytes": leaked,
                     "peak_gib": peak / GIB, "nfrag_ret": int(nfrag_ret)}


def main():
    print("P1 official make_big independence. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"P1 REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)

    mirror = load_official_mirror()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    Z, Y, X = (int(v) for v in aff.shape[1:])
    print(f"loaded {AFF} {aff.shape} {aff.dtype} {aff.nbytes / GIB:.3f} GiB", flush=True)

    doc = {
        "claim": "not a 2 Gvox/s number; official 2-tile independence only",
        "gpu": gpu_state(),
        "aff_shape": [3, Z, Y, X],
        "mirror": "official data/ws_bounty/make_big.py::mirror",
        "val": {},
        "axes": {},
        "pass": False,
    }
    dump(doc)

    ws = GpuWs()
    print("GPU WS val...", flush=True)
    val_gpu, val_meta = ws.run(aff)
    n0, bg0 = nfrag_bg(val_gpu)
    fp0, nreg0, _ = fingerprint(val_gpu)
    val_row = {
        "nfrag": n0, "bg": bg0, "fingerprint": fp0, "nregions": nreg0,
        "nfrag_expected": FRAGMENTS_VAL, "bg_expected": BG_VAL_MEASURED,
        **val_meta,
    }
    op = CACHE / "wz_fragments.npy"
    if op.exists():
        ref = np.ascontiguousarray(np.load(op), dtype=np.uint32)
        oracle_eq = bool(ref.shape == val_gpu.shape and np.array_equal(ref, val_gpu))
        val_row["cpu_oracle_array_equal"] = oracle_eq
        print(f"CPU oracle wz_fragments.npy vs GPU val: array_equal={oracle_eq}",
              flush=True)
        del ref
    val_row["nfrag_match"] = n0 == FRAGMENTS_VAL
    doc["val"] = val_row
    dump(doc)
    print(f"val GPU nfrag={n0} bg={bg0} {val_meta}", flush=True)
    if n0 != FRAGMENTS_VAL:
        print("P1 FAIL: GPU val nfrag != TASK 2175400", flush=True)
        dump(doc)
        raise SystemExit(1)

    all_ok = True
    for axis, name, mid in ((0, "z", Z), (1, "y", Y), (2, "x", X)):
        print(f"\n=== axis {name} 2-tile ===", flush=True)
        flips = [False, False, False]
        flips[axis] = True
        mir = mirror(aff, tuple(flips))
        if mir.shape != aff.shape:
            raise RuntimeError(f"mirror shape {mir.shape} != {aff.shape}")

        sl = [slice(None)] * 4
        sl[0] = axis
        sl[axis + 1] = 0
        seam_tile = mir[tuple(sl)]
        seam_zero_tile = bool(np.all(seam_tile == 0))

        big = np.concatenate([aff, mir], axis=axis + 1)
        slb = [slice(None)] * 4
        slb[0] = axis
        slb[axis + 1] = mid
        seam_big = big[tuple(slb)]
        seam_max = int(seam_big.max())
        seam_zero = bool(np.all(seam_big == 0))
        print(f"  official mirror seam_tile_zero={seam_zero_tile} "
              f"concat_seam_zero={seam_zero} max={seam_max}", flush=True)

        print("  GPU WS mirrored tile...", flush=True)
        mir_lab, mir_meta = ws.run(mir)
        n1, bg1 = nfrag_bg(mir_lab)
        fp1, _, _ = fingerprint(mir_lab)
        print(f"  mir nfrag={n1} bg={bg1} vs val {n0} {mir_meta}", flush=True)

        print(f"  GPU WS concat {big.shape}...", flush=True)
        cat_lab, cat_meta = ws.run(big)
        nc, bgc = nfrag_bg(cat_lab)
        fpc, _, _ = fingerprint(cat_lab)
        nspan, span_ids = spanning_ids(cat_lab, axis, mid)
        print(f"  concat nfrag={nc} bg={bgc} span={nspan} {cat_meta}", flush=True)

        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[axis] = slice(0, mid)
        sl1[axis] = slice(mid, None)
        tile0 = cat_lab[tuple(sl0)]
        tile1 = cat_lab[tuple(sl1)]
        eq0 = same_partition(tile0, val_gpu)
        eq1 = same_partition(tile1, mir_lab)
        print(f"  tile0==val remap {eq0}; tile1==mir remap {eq1}", flush=True)

        la = take_plane(cat_lab, axis, mid - 1)
        lb = take_plane(cat_lab, axis, mid)
        face = take_plane(big[axis], axis, mid)
        labeled = (la > 0) & (lb > 0)
        diff = labeled & (la != lb)
        same = labeled & (la == lb)
        n_cross = int(diff.sum())
        n_same = int(same.sum())
        max_aff_cross = int(face[diff].max()) if n_cross else 0
        max_aff_same = int(face[same].max()) if n_same else 0
        n_aff_pos_cross = int((face[diff] > 0).sum()) if n_cross else 0
        rag_cross_dead = bool(n_aff_pos_cross == 0 and max_aff_cross == 0)
        print(f"  seam faces cross={n_cross} same_id={n_same} "
              f"max_aff_cross={max_aff_cross} rag_cross_dead={rag_cross_dead}",
              flush=True)

        nfrag_mir_match = n1 == n0
        nfrag_cat_match = nc == n0 + n1
        axis_ok = bool(
            seam_zero and seam_zero_tile
            and nspan == 0
            and nfrag_mir_match and nfrag_cat_match
            and eq0 and eq1
            and rag_cross_dead
        )
        all_ok = all_ok and axis_ok
        row = {
            "axis": name,
            "big_shape": list(big.shape),
            "seam_zero_tile": seam_zero_tile,
            "seam_zero_concat": seam_zero,
            "seam_max": seam_max,
            "mir_nfrag": n1,
            "mir_bg": bg1,
            "mir_fingerprint": fp1,
            "mir_nfrag_eq_val": nfrag_mir_match,
            "concat_nfrag": nc,
            "concat_bg": bgc,
            "concat_fingerprint": fpc,
            "concat_nfrag_eq_sum": nfrag_cat_match,
            "spanning_ids": nspan,
            "spanning_id_sample": span_ids,
            "tile0_eq_val_remap": eq0,
            "tile1_eq_mir_remap": eq1,
            "cross_faces": n_cross,
            "same_id_faces": n_same,
            "max_aff_cross": max_aff_cross,
            "max_aff_same_id": max_aff_same,
            "n_aff_pos_cross": n_aff_pos_cross,
            "rag_cross_dead_at_T": rag_cross_dead,
            "T": T_MERGE,
            "mir_ws": mir_meta,
            "concat_ws": cat_meta,
            "pass": axis_ok,
        }
        doc["axes"][name] = row
        doc["pass"] = False
        doc["gpu_after_" + name] = gpu_state()
        dump(doc)
        print(f"  axis {name} {'PASS' if axis_ok else 'FAIL'}", flush=True)
        del mir, big, mir_lab, cat_lab, tile0, tile1

    doc["pass"] = bool(all_ok and doc["val"].get("nfrag_match"))
    doc["gpu_final"] = gpu_state()
    dump(doc)
    print(f"\nP1 {'PASS' if doc['pass'] else 'FAIL'} -> {OUT}", flush=True)
    print("Independence of official 2-tile is "
          + ("proven." if doc["pass"] else "NOT proven. Keep N6.")
          + " Not a 2 Gvox/s claim.", flush=True)
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
