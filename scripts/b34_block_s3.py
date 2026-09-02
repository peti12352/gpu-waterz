#!/usr/bin/env python3
"""B34: naive intra-tile Kruskal then residual S4; Lu-exact only if naive is close."""
from __future__ import annotations

import argparse
import ctypes
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import (  # noqa: E402
    AFF_THRESHOLDS, SO, compile_so, frag_sizes, grade_parents, load_rag, stamp,
)
from _kruskal_probe import parse_voi  # noqa: E402
from p0_leftover_block import fragment_centroids, tile_ids  # noqa: E402
from task_gate import BASE_MERGE, BASE_SPLIT, SLACK, print_contract  # noqa: E402


def miss_both_halves(text: str) -> bool:
    rows = parse_voi(text)
    if 0.2 not in rows:
        return True
    s, m = rows[0.2]
    return (s > BASE_SPLIT[0.2] + SLACK + 0.05) and (m > BASE_MERGE[0.2] + SLACK + 0.05)


def close_enough(text: str) -> bool:
    rows = parse_voi(text)
    if 0.2 not in rows:
        return False
    s, m = rows[0.2]
    return (s <= BASE_SPLIT[0.2] + SLACK + 0.05) or (m <= BASE_MERGE[0.2] + SLACK + 0.05)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dz", type=int, default=16)
    p.add_argument("--dy", type=int, default=64)
    p.add_argument("--dx", type=int, default=64)
    p.add_argument("--lu", action="store_true")
    args = p.parse_args()
    print_contract()
    compile_so()
    u, v, sm, ct, fr, max_id = load_rag()
    sz = frag_sizes(fr, max_id)
    cz, cy, cx = fragment_centroids(fr, max_id)
    tid = np.ascontiguousarray(
        tile_ids(cz, cy, cx, args.dz, args.dy, args.dx, fr.shape), dtype=np.int32
    )
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    lib = ctypes.CDLL(str(SO))

    if not args.lu:
        fn = lib.block_s3_naive_cpu
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int64),
        ]
        print(f"B34 naive tile={args.dz}x{args.dy}x{args.dx}", flush=True)
        t0 = time.time()
        rc = fn(
            u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            sz.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            tid.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int64(len(u)),
            thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(len(thrs)),
            parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(max_id),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )
        wall = time.time() - t0
        nres = int(stats[:, 1].max())
        print(f"B34naive rc={rc} wall={wall:.3f} residual={list(map(int, stats[:, 1]))}", flush=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            ok = grade_parents(parents, fr, "B34naive", f"b34_naive_{args.dz}_{args.dy}_{args.dx}")
        text = buf.getvalue()
        sys.stdout.write(text)
        if ok:
            stamp("b34", True, f"LOCK naive depth=2 residual={nres} wall={wall:.3f}")
            print("B34 naive PASS LOCK", flush=True)
            return True
        if miss_both_halves(text):
            stamp("b34", False, "FAIL naive both-halves>0.05 skip Lu")
            print("B34 naive miss >0.05 both halves — skip Lu-exact", flush=True)
            return False
        if close_enough(text):
            print("B34 naive close — running Lu-exact", flush=True)
            args.lu = True
        else:
            stamp("b34", False, f"FAIL naive residual={nres}")
            print("B34 naive FAIL (not close enough for Lu)", flush=True)
            return False

    if args.lu:
        fn = lib.block_s3_lu_cpu
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int64),
        ]
        czc = np.ascontiguousarray(cz)
        cyc = np.ascontiguousarray(cy)
        cxc = np.ascontiguousarray(cx)
        print(f"B34 Lu tile0={args.dz}x{args.dy}x{args.dx}", flush=True)
        t0 = time.time()
        rc = fn(
            u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            sz.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            czc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cyc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cxc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(len(u)),
            ctypes.c_double(args.dz), ctypes.c_double(args.dy), ctypes.c_double(args.dx),
            thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(len(thrs)),
            parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(max_id),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )
        wall = time.time() - t0
        nlev = int(stats[:, 0].max())
        last = int(stats[:, 1].max())
        maxres = int(stats[:, 2].max())
        print(f"B34lu rc={rc} wall={wall:.3f} levels={list(map(int, stats[:, 0]))} last_res={list(map(int, stats[:, 1]))} max_res={maxres}", flush=True)
        if last >= 50_000 or maxres >= 50_000:
            stamp("b34", False, f"FAIL-depth max_residual={maxres} last={last} levels={nlev}")
            print("B34 Lu FAIL-depth residual>=50k", flush=True)
            return False
        buf = io.StringIO()
        with redirect_stdout(buf):
            ok = grade_parents(parents, fr, "B34lu", f"b34_lu_{args.dz}_{args.dy}_{args.dx}")
        sys.stdout.write(buf.getvalue())
        if ok and nlev <= 30:
            stamp("b34", True, f"LOCK Lu depth={nlev} residual={last} wall={wall:.3f}")
            print("B34 Lu PASS LOCK", flush=True)
            return True
        stamp("b34", False, f"{'PASS' if ok else 'FAIL'} Lu depth={nlev} residual={last}")
        print(f"B34 Lu {'PASS' if ok else 'FAIL'}", flush=True)
        return False
    return False


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
