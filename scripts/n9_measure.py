#!/usr/bin/env python3
"""N9 Type D: W5+face-clear 2.16, component hist, dirty multiplicity, BFS vs UF.

Idle-5090. Not a 2 Gvox/s claim. Numbers only from this run.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b_dev_aff import bind, build  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402
from n8_run216 import NVOX, SHAPE, official_216  # noqa: E402
from p1_make_big_indep import AFF, CACHE, card_busy, gpu_state, nfrag_bg  # noqa: E402
from task_gate import AFF_HIGH, AFF_LOW, FRAGMENTS_VAL  # noqa: E402
import segment as S  # noqa: E402

OUT = CACHE / "n9_measure.json"
NOTE = ROOT / "notes/N9_MEASURE.md"
WS_STOP_MS = 4000.0
GIANT_FRAC = 0.10
WARP = 32


def ws_stats(lib):
    w5_calls = ctypes.c_int(0)
    nlist_sum = ctypes.c_int64(0)
    nlist_max = ctypes.c_int(0)
    nlist_last = ctypes.c_int(0)
    nvox_last = ctypes.c_int64(0)
    rounds_sum = ctypes.c_int(0)
    w5_ms = ctypes.c_float(0)
    bfs_ms = ctypes.c_float(0)
    bfs_calls = ctypes.c_int(0)
    lib.ws_n9_stats(
        ctypes.byref(w5_calls), ctypes.byref(nlist_sum), ctypes.byref(nlist_max),
        ctypes.byref(nlist_last), ctypes.byref(nvox_last), ctypes.byref(rounds_sum),
        ctypes.byref(w5_ms), ctypes.byref(bfs_ms), ctypes.byref(bfs_calls),
    )
    nv = int(nvox_last.value)
    nl = int(nlist_last.value)
    return {
        "w5_calls": int(w5_calls.value),
        "nlist_sum": int(nlist_sum.value),
        "nlist_max": int(nlist_max.value),
        "nlist_last": nl,
        "nvox_last": nv,
        "nlist_over_nvox": (nl / nv) if nv else None,
        "rounds_sum": int(rounds_sum.value),
        "w5_ms": float(w5_ms.value),
        "bfs_ms": float(bfs_ms.value),
        "bfs_calls": int(bfs_calls.value),
    }


def dirty_stats(libp):
    de = ctypes.c_ulonglong(0)
    du = ctypes.c_ulonglong(0)
    n = ctypes.c_int(0)
    de_max = ctypes.c_int(0)
    du_max = ctypes.c_int(0)
    libp.agg_n9_dirty(
        ctypes.byref(de), ctypes.byref(du), ctypes.byref(n),
        ctypes.byref(de_max), ctypes.byref(du_max),
    )
    e = int(de.value)
    u = int(du.value)
    nn = int(n.value)
    mean_mult = (e / u) if u else None
    mean_e = (e / nn) if nn else None
    return {
        "dirty_e_sum": e,
        "dirty_unique_sum": u,
        "calls": nn,
        "dirty_e_max": int(de_max.value),
        "dirty_unique_max": int(du_max.value),
        "mean_multiplicity": mean_mult,
        "mean_dirty_e": mean_e,
        "mean_mult_lt_warp": bool(mean_mult is not None and mean_mult < WARP),
    }


def bind_n9(libw, libp):
    libw.ws_n9_reset.restype = None
    libw.ws_n9_stats.restype = None
    libw.ws_n9_stats.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    libp.agg_n9_reset.restype = None
    libp.agg_n9_dirty.restype = None
    libp.agg_n9_dirty.argtypes = [
        ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]


def size_hist(lab, tag):
    flat = np.asarray(lab).ravel()
    nvox = int(flat.size)
    cnt = np.bincount(flat)
    if cnt.size > 0:
        cnt[0] = 0
    sizes = cnt[cnt > 0]
    if sizes.size == 0:
        return {
            "tag": tag, "nvox": nvox, "ncomp": 0, "max": 0,
            "max_frac_nvox": 0.0, "giant_gt_10pct": False,
        }
    mx = int(sizes.max())
    return {
        "tag": tag,
        "nvox": nvox,
        "ncomp": int(sizes.size),
        "max": mx,
        "p50": int(np.median(sizes)),
        "p99": int(np.percentile(sizes, 99)),
        "max_frac_nvox": mx / nvox,
        "giant_gt_10pct": bool(mx > GIANT_FRAC * nvox),
    }


def write_note(doc):
    val = doc["val"]
    big = doc.get("v216")
    basin = doc["basin_hist"]
    plat = doc.get("plateau_hist")
    dirty = doc["dirty"]
    lines = [
        "# N9 Type D measurements",
        "",
        "Not a 2 Gvox/s number. Not a 3090 Ti number. Idle 5090 only.",
        "Numbers below are from this run. No invented fill-ins.",
        "",
        f"GPU: `{doc['gpu']}`",
        "",
        "## Val W5 (`WATERZ_UF_ALGO=3`)",
        "",
        f"- identity vs `wz_fragments.npy`: {val['oracle_array_equal']}",
        f"- nfrag={val['nfrag']} bg={val['bg']}",
        f"- STAGE_MS ws={val['stages'].get('ws')} rag={val['stages'].get('rag')} "
        f"agg={val['stages'].get('agg')}",
        f"- w5_ms={val['ws_n9']['w5_ms']:.2f} (calls={val['ws_n9']['w5_calls']}) "
        f"bfs_ms={val['ws_n9']['bfs_ms']:.2f} (calls={val['ws_n9']['bfs_calls']})",
        f"- nlist_last={val['ws_n9']['nlist_last']} / nvox={val['ws_n9']['nvox_last']} "
        f"frac={val['ws_n9']['nlist_over_nvox']}",
        f"- stitch rounds_sum={val['ws_n9']['rounds_sum']}",
    ]
    ws = float(val["stages"].get("ws") or 0)
    bfs = float(val["ws_n9"]["bfs_ms"] or 0)
    bfs_frac = (bfs / ws) if ws else None
    lines += [
        f"- BFS / WS = {bfs_frac} (PRUF dead if <0.01)",
        "",
        "## Component sizes",
        "",
        f"- basin (fragments `{basin['tag']}`): ncomp={basin['ncomp']} "
        f"max={basin['max']} max/nvox={basin['max_frac_nvox']:.6f} "
        f"giant>10%={basin['giant_gt_10pct']}",
    ]
    if plat:
        lines.append(
            f"- plateau (`{plat['tag']}`): ncomp={plat['ncomp']} max={plat['max']} "
            f"max/nvox={plat['max_frac_nvox']:.6f} giant>10%={plat['giant_gt_10pct']}"
        )
    skip_connectit = (not basin["giant_gt_10pct"]) and (
        plat is None or not plat["giant_gt_10pct"]
    )
    lines += [
        f"- ConnectIt/Afforest: {'SKIP (no component >10% nvox)' if skip_connectit else 'gate open'}",
        "",
        "## Dirty-key multiplicity (`hash_combine_dirty`, val G15)",
        "",
        f"- calls={dirty['calls']} dirty_e_sum={dirty['dirty_e_sum']} "
        f"unique_sum={dirty['dirty_unique_sum']}",
        f"- mean multiplicity={dirty['mean_multiplicity']} "
        f"(max dirty_e={dirty['dirty_e_max']} max unique={dirty['dirty_unique_max']})",
        f"- mean_mult < warp {WARP}: {dirty['mean_mult_lt_warp']} "
        f"(hash-table PDFs stay dead if true)",
        "",
    ]
    if big:
        lines += [
            "## 2.16 W5 + z-slab face-clear (`WATERZ_UF_ALGO=3`)",
            "",
            f"- e2e_ms={big['e2e_ms']:.1f} gvox_s_5090={big['gvox_s_5090']:.3f} "
            "(not TASK-grade; wrong card)",
            f"- STAGE_MS ws={big['stages'].get('ws')} rag={big['stages'].get('rag')} "
            f"agg={big['stages'].get('agg')}",
            f"- nlab={big['nlab']} nfrag={big.get('nfrag')}",
            f"- w5_ms={big['ws_n9']['w5_ms']:.2f} bfs_ms={big['ws_n9']['bfs_ms']:.2f} "
            f"nlist_max={big['ws_n9']['nlist_max']} "
            f"nlist_last={big['ws_n9']['nlist_last']} / {big['ws_n9']['nvox_last']}",
            f"- WS > {WS_STOP_MS:.0f} ms: {big['ws_over_4s']} "
            f"{'STOP CUDA tracks A/B/C' if big['ws_over_4s'] else 'CUDA tracks may proceed'}",
            "",
        ]
    lines += [
        "## Verdicts from this run",
        "",
        f"- PRUF: {'dead (BFS <1% WS)' if bfs_frac is not None and bfs_frac < 0.01 else 'see BFS/WS'}",
        f"- ConnectIt/Afforest: {'do not implement' if skip_connectit else 'hist has a giant'}",
        f"- hash-table library swap: "
        f"{'do not implement (multiplicity << warp)' if dirty['mean_mult_lt_warp'] else 'multiplicity not << warp'}",
        f"- CUDA leftover tracks: "
        f"{'STOP (WS still >4s after W5)' if doc.get('stop_cuda') else 'open if identity gates pass'}",
        "",
    ]
    NOTE.write_text("\n".join(lines))


def main():
    print("N9 Type D. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N9 REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ["WATERZ_UF_ALGO"] = "3"
    os.environ["WATERZ_AGG_LEVERS"] = "15"
    os.environ["WATERZ_STAGE_MS"] = "1"
    os.environ.pop("WATERZ_AGG_EPS", None)
    if DSO.exists():
        DSO.unlink()
    build()
    compile_d()
    libw = ctypes.CDLL(str(ROOT / "src/libws_gpu.so"))
    bind(libw)
    libp = ctypes.CDLL(str(DSO))
    bind_n9(libw, libp)

    basin = None
    op = CACHE / "wz_fragments.npy"
    if op.exists():
        fr = np.ascontiguousarray(np.load(op), dtype=np.uint32)
        basin = size_hist(fr, "wz_fragments.npy")
        print(
            f"N9 basin hist ncomp={basin['ncomp']} max={basin['max']} "
            f"frac={basin['max_frac_nvox']:.6f} giant={basin['giant_gt_10pct']}",
            flush=True,
        )
        del fr

    plat = None

    libw.ws_n9_reset()
    libp.agg_n9_reset()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    aff_d = S.DevBuf.from_host(aff)
    out, e2e = S.cuda_event_time(
        lambda: S.segment_d(aff_d, [0.3], return_device=True))
    st = dict(S.STAGE_MS)
    nlab = 0
    if out:
        host = out[0].to_host()
        nlab = int(np.unique(host).size)
        for b in out:
            b.free()
        del host
    aff_d.free()
    # Identity is WS fragments, not agglomerated labels.
    bits_nfrag, bits_bg = None, None
    oracle = None
    libw.ws_n9_reset()
    from b_dev_aff import run_split  # noqa: E402
    seg, meta = run_split(libw, aff, park=True)
    bits_nfrag, bits_bg = nfrag_bg(seg)
    if op.exists():
        ref = np.ascontiguousarray(np.load(op), dtype=np.uint32)
        oracle = bool(ref.shape == seg.shape and np.array_equal(ref, seg))
        del ref
    if basin is None:
        basin = size_hist(seg, "gpu_val_w5_labels")
    del seg
    val = {
        "e2e_ms": float(e2e),
        "stages": {k: float(v) for k, v in st.items()},
        "nlab": nlab,
        "nfrag": bits_nfrag,
        "bg": bits_bg,
        "oracle_array_equal": oracle,
        "nfrag_ok": bits_nfrag == FRAGMENTS_VAL,
        "divide_ms": float(meta.get("divide_ms", 0)),
        "ws_n9": ws_stats(libw),
        "aff_parked": bool(S.LAST_AFF_PARKED),
    }
    dirty = dirty_stats(libp)
    print(
        f"N9 val identity={oracle} nfrag={bits_nfrag} "
        f"ws={st.get('ws')} w5={val['ws_n9']['w5_ms']:.2f} "
        f"bfs={val['ws_n9']['bfs_ms']:.2f} dirty_mult={dirty['mean_multiplicity']}",
        flush=True,
    )

    libw.ws_n9_reset()
    libp.agg_n9_reset()
    print("N9 2.16 W5+face-clear load", flush=True)
    aff216 = official_216()
    print(f"N9 2.16 aff {aff216.shape} {aff216.nbytes / 2**30:.2f} GiB", flush=True)
    aff_d = S.DevBuf.from_host(aff216)
    del aff216
    out, ms216 = S.cuda_event_time(
        lambda: S.segment_d(aff_d, [0.3], return_device=True))
    st216 = dict(S.STAGE_MS)
    nlab216 = 0
    slab_hist = None
    nfrag216 = None
    if out:
        host = out[0].to_host()
        nlab216 = int(np.unique(host).size)
        # One z-slab (Z=125) basin hist, not the full 8 GB unique.
        slab = host[:125]
        slab_hist = size_hist(slab, "2.16_slab0_z125")
        nfrag216 = int((host != 0).sum() and np.unique(host).size)
        for b in out:
            b.free()
        del host
    aff_d.free()
    ws216 = float(st216.get("ws") or 0)
    gvox = (NVOX / 1e9) / (float(ms216) / 1000.0) if ms216 else 0.0
    v216 = {
        "shape": list(SHAPE),
        "nvox": NVOX,
        "e2e_ms": float(ms216),
        "stages": {k: float(v) for k, v in st216.items()},
        "nlab": nlab216,
        "nfrag": nfrag216,
        "gvox_s_5090": gvox,
        "ws_n9": ws_stats(libw),
        "ws_over_4s": bool(ws216 > WS_STOP_MS),
        "slab_hist": slab_hist,
        "aff_parked": bool(S.LAST_AFF_PARKED),
    }
    print(
        f"N9 2.16 e2e={ms216:.1f} ws={ws216:.1f} "
        f"gvox/s={gvox:.3f} over4s={v216['ws_over_4s']}",
        flush=True,
    )

    plat = None
    e9a_src = ROOT / "src/e9a_hist.cpp"
    e9a_so = ROOT / "src/libe9a_hist.so"
    if e9a_src.is_file():
        import subprocess
        subprocess.check_call([
            "g++", "-O3", "-DNDEBUG", "-shared", "-fPIC",
            "-o", str(e9a_so), str(e9a_src),
        ])
        with h5py.File(AFF, "r") as f:
            aff_e9a = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
        z, y, x = (int(v) for v in aff_e9a.shape[1:])
        stats = np.zeros(9, dtype=np.int64)
        libe = ctypes.CDLL(str(e9a_so))
        libe.e9a_plateau_hist.restype = ctypes.c_int
        libe.e9a_plateau_hist.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_int64),
        ]
        libe.e9a_plateau_hist(
            aff_e9a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            z, y, x, ctypes.c_float(AFF_LOW), ctypes.c_float(AFF_HIGH),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )
        n_fg = int(stats[0])
        n_plat = int(stats[3])
        mx = int(stats[6])
        nvox = z * y * x
        plat = {
            "tag": "e9a_plateau_hist",
            "nvox": nvox,
            "n_fg": n_fg,
            "ncomp": n_plat,
            "max": mx,
            "p50": int(stats[7]),
            "p99": int(stats[8]),
            "max_frac_nvox": mx / nvox if nvox else 0.0,
            "giant_gt_10pct": bool(mx > GIANT_FRAC * nvox),
        }
        print(
            f"N9 plateau hist ncomp={n_plat} max={mx} "
            f"frac={plat['max_frac_nvox']:.6f} giant={plat['giant_gt_10pct']}",
            flush=True,
        )
        del aff_e9a

    stop_cuda = bool(v216["ws_over_4s"])
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "val": val,
        "v216": v216,
        "basin_hist": basin,
        "plateau_hist": plat,
        "dirty": dirty,
        "stop_cuda": stop_cuda,
        "skip_connectit": bool(
            basin and not basin["giant_gt_10pct"]
            and (plat is None or not plat["giant_gt_10pct"])
        ),
        "skip_hashtable": bool(dirty.get("mean_mult_lt_warp")),
        "bfs_frac_ws": (
            (val["ws_n9"]["bfs_ms"] / val["stages"]["ws"])
            if val["stages"].get("ws") else None
        ),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    write_note(doc)
    print(f"N9 measure -> {OUT} {NOTE} stop_cuda={stop_cuda}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
