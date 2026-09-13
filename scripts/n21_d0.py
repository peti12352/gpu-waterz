#!/usr/bin/env python3
"""N21 Round 0: unique owners (file), dirty inners, compress hops.

Legal stack env. Val only. No 2.16. No StarMerge. No RNN GPU.
Not a 2 Gvox/s number; not 3090 Ti.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from e6r_parhac import DSO, compile_d  # noqa: E402
from _agg_common import load_rag  # noqa: E402
from t3_memsafe import grade_t3, voi_parent_mmap  # noqa: E402
from b_dev_aff import bind, build, run_split  # noqa: E402
from p1_make_big_indep import AFF, CACHE, card_busy, gpu_state, nfrag_bg  # noqa: E402
from n20_res import attach  # noqa: E402
from task_gate import FRAGMENTS_VAL  # noqa: E402

CLAIM = "N21 diagnostic; not a throughput claim; not a 2 Gvox/s number; not 3090 Ti"
PIN_AGG = 1679.9063761718571
GO_RATIO = 0.25
BG_VAL = 506_568
GOLD = CACHE / "wz_fragments.npy"


def legal_env():
    os.environ.setdefault("WATERZ_UF_ALGO", "3")
    os.environ["WATERZ_HOST_PARK"] = "0"
    os.environ["WATERZ_AFF_PARK"] = "0"
    os.environ.setdefault("WATERZ_AGG_LEVERS", "15")
    os.environ["WATERZ_FOLD_FLATTEN"] = "1"
    os.environ["WATERZ_SHARE_OFF"] = "1"
    os.environ["WATERZ_HOOK_ROOT"] = "1"
    os.environ["WATERZ_FUSE_DIRTY"] = "1"
    os.environ["WATERZ_NLIVE_ARITH"] = "1"
    os.environ["WATERZ_EMIT_HOLES"] = "1"


def pct(xs, q):
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def run_agg_inners():
    compile_d()
    lib = ctypes.CDLL(str(DSO))
    lib.parhac_paper_d.restype = ctypes.c_int
    lib.parhac_paper_d.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.agg_n21_reset.restype = None
    lib.agg_n21_count.restype = ctypes.c_int
    lib.agg_n21_get.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    u, v, sm, ct, _fr, max_id = load_rag()
    thrs = np.asarray([0.3], dtype=np.float64)
    parents = np.empty((1, max_id + 1), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
    lib.agg_n21_reset()
    t0 = time.perf_counter()
    rc = lib.parhac_paper_d(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(1),
        ctypes.c_double(0.40),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    split, merge, nseg = voi_parent_mmap(parents[0])
    ok, sl, ml = grade_t3(split, merge)
    n = int(lib.agg_n21_count())
    rows = []
    for i in range(n):
        a = ctypes.c_int(); b = ctypes.c_int(); c = ctypes.c_int()
        d = ctypes.c_int(); e = ctypes.c_int(); f = ctypes.c_int()
        lib.agg_n21_get(i, ctypes.byref(a), ctypes.byref(b), ctypes.byref(c),
                        ctypes.byref(d), ctypes.byref(e), ctypes.byref(f))
        ns, nl, nd, nu, na, nt = (
            int(a.value), int(b.value), int(c.value),
            int(d.value), int(e.value), int(f.value),
        )
        rows.append({
            "nscan": ns, "nlive": nl, "ndirty": nd, "nuniq": nu,
            "nact": na, "ntab": nt,
            "ndirty_over_nscan": (nd / ns) if ns else None,
            "nact_over_nscan": (na / ns) if ns else None,
        })
    dirty_r = [r["ndirty_over_nscan"] for r in rows if r["nscan"] > 0]
    act_r = [r["nact_over_nscan"] for r in rows if r["nscan"] > 0]
    med_d = pct(dirty_r, 50)
    med_a = pct(act_r, 50)
    return {
        "rc": int(rc),
        "wall_ms": wall_ms,
        "split": split,
        "merge": merge,
        "nseg": nseg,
        "ok": bool(ok),
        "limit_split": sl,
        "limit_merge": ml,
        "inner": int(stats[0, 2]),
        "merges": int(stats[0, 1]),
        "n_records": n,
        "nscan_p50": pct([r["nscan"] for r in rows], 50),
        "nscan_p99": pct([r["nscan"] for r in rows], 99),
        "nlive_p50": pct([r["nlive"] for r in rows], 50),
        "ndirty_p50": pct([r["ndirty"] for r in rows], 50),
        "ndirty_p99": pct([r["ndirty"] for r in rows], 99),
        "ndirty_max": max((r["ndirty"] for r in rows), default=0),
        "nact_p50": pct([r["nact"] for r in rows], 50),
        "nact_max": max((r["nact"] for r in rows), default=0),
        "ndirty_over_nscan_p50": med_d,
        "ndirty_over_nscan_p99": pct(dirty_r, 99),
        "nact_over_nscan_p50": med_a,
        "nact_over_nscan_p99": pct(act_r, 99),
        "go_listed_insert": bool(med_d is not None and med_d < GO_RATIO),
        "go_listed_rebuild": bool(
            (med_a is not None and med_a < GO_RATIO)
            or (med_d is not None and med_d < GO_RATIO)
        ),
        "rows_head": rows[:8],
        "rows_tail": rows[-4:] if len(rows) > 8 else [],
    }


def run_ws_hops():
    os.environ["WATERZ_N21_HOPS"] = "1"
    build()
    import h5py
    lib = ctypes.CDLL(str(ROOT / "src/libws_gpu.so"))
    bind(lib)
    lib.ws_n9_reset.restype = None
    lib.ws_n21_hop_count.restype = ctypes.c_int
    lib.ws_n21_hop_get.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint), ctypes.c_char_p, ctypes.c_int,
    ]
    lib.ws_n9_reset()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    t0 = time.perf_counter()
    seg, meta = run_split(lib, aff, park=False)
    ws_ms = (time.perf_counter() - t0) * 1000.0
    nfrag, bg = nfrag_bg(seg)
    gold = np.load(GOLD) if GOLD.is_file() else None
    eq = bool(gold is not None and np.array_equal(seg, gold))
    nh = int(lib.ws_n21_hop_count())
    hops = []
    buf = ctypes.create_string_buffer(16)
    for i in range(nh):
        nlist = ctypes.c_int()
        mean = ctypes.c_double()
        rf = ctypes.c_double()
        hmax = ctypes.c_uint()
        lib.ws_n21_hop_get(
            i, ctypes.byref(nlist), ctypes.byref(mean), ctypes.byref(rf),
            ctypes.byref(hmax), buf, 16,
        )
        hops.append({
            "nlist": int(nlist.value),
            "mean_hops": float(mean.value),
            "root_frac": float(rf.value),
            "hop_max": int(hmax.value),
            "site": buf.value.decode("ascii", "replace"),
        })
    mean_h = pct([h["mean_hops"] for h in hops], 50) if hops else None
    root_p = pct([h["root_frac"] for h in hops], 50) if hops else None
    return {
        "ws_ms": ws_ms,
        "nfrag": nfrag,
        "bg": bg,
        "identity": bool(eq and nfrag == FRAGMENTS_VAL and bg == BG_VAL),
        "array_equal": eq,
        "n_probes": nh,
        "hops": hops,
        "mean_hops_p50": mean_h,
        "root_frac_p50": root_p,
        "go_compress_subset": bool(
            mean_h is not None and mean_h < 1.5 and root_p is not None and root_p >= 0.5
        ),
        "go_list_jump": bool(mean_h is not None and mean_h >= 1.5),
        "meta": meta,
    }


def main():
    legal_env()
    busy = card_busy()
    if busy:
        print("N21_D0 REFUSE card_busy", busy, flush=True)
        return 2
    print("N21_D0 gpu", gpu_state(), flush=True)
    dirty = run_agg_inners()
    hops = run_ws_hops()
    owners = json.loads((CACHE / "N21_D0_OWNERS.json").read_text())
    doc = {
        "claim": CLAIM,
        "pin_agg_ms": PIN_AGG,
        "owners": owners["floors_vs_1679.9"],
        "dirty": dirty,
        "hops": hops,
        "go_listed_insert": dirty["go_listed_insert"],
        "go_listed_rebuild": dirty["go_listed_rebuild"],
        "go_compress_subset": hops["go_compress_subset"],
        "go_list_jump": hops["go_list_jump"],
        "ran_216": False,
    }
    doc = attach(doc, "N21_D0")
    (CACHE / "N21_D0.json").write_text(json.dumps(doc, indent=2) + "\n")
    lines = [
        "# N21 D0 dirty + hops",
        "",
        CLAIM + ".",
        "",
        f"- T=0.3 VOI ok={dirty['ok']} split={dirty['split']:.6f} merge={dirty['merge']:.6f} "
        f"nseg={dirty['nseg']} inner={dirty['inner']} merges={dirty['merges']} wall_ms={dirty['wall_ms']:.1f}",
        f"- n_records={dirty['n_records']} nscan_p50={dirty['nscan_p50']} ndirty_p50={dirty['ndirty_p50']} "
        f"nact_p50={dirty['nact_p50']}",
        f"- ndirty/nscan p50={dirty['ndirty_over_nscan_p50']} p99={dirty['ndirty_over_nscan_p99']} "
        f"go_listed_insert={dirty['go_listed_insert']}",
        f"- nact/nscan p50={dirty['nact_over_nscan_p50']} go_listed_rebuild={dirty['go_listed_rebuild']}",
        f"- hops n={hops['n_probes']} mean_hops_p50={hops['mean_hops_p50']} "
        f"root_frac_p50={hops['root_frac_p50']} identity={hops['identity']}",
        f"- go_compress_subset={hops['go_compress_subset']} go_list_jump={hops['go_list_jump']}",
        f"- timing_usable_vs_1679={doc.get('timing_usable_vs_1679')}",
        "- no 2.16",
    ]
    (ROOT / "notes" / "N21_D0.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: doc[k] for k in (
        "go_listed_insert", "go_listed_rebuild", "go_compress_subset",
        "go_list_jump", "timing_usable_vs_1679",
    )}, indent=2), flush=True)
    return 0 if dirty["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
