#!/usr/bin/env python3
"""N20 unit tests on a 3-node RAG. CPU only. No GPU.

Edges (contact mean):
  1-2 mean 0.90,  2-3 mean 0.40,  1-3 mean 0.50
After merging 1-2:
  S3     (4+5)/(10+10) = 0.45
  complete min(0.40, 0.50) = 0.40
  WPGMA  0.5*(0.40+0.50) = 0.45
At T=0.42: S3 and WPGMA merge all; complete keeps 3 separate.
"""
from __future__ import annotations

import ctypes
import json
import socket
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from n20_lib import load_n20  # noqa: E402

CACHE = ROOT / "data/cache"
U32P = ctypes.POINTER(ctypes.c_uint32)
U8P = ctypes.POINTER(ctypes.c_uint8)
F64P = ctypes.POINTER(ctypes.c_double)
I64P = ctypes.POINTER(ctypes.c_int64)


def roots(parents):
    return tuple(int(parents[i]) for i in (1, 2, 3))


def same(a, b):
    return a == b


def run():
    lib = load_n20()
    u = np.array([1, 2, 1], dtype=np.uint32)
    v = np.array([2, 3, 3], dtype=np.uint32)
    sm = np.array([9.0, 4.0, 5.0], dtype=np.float64)
    ct = np.array([10, 10, 10], dtype=np.int64)
    max_id = 3
    thrs = np.array([0.42, 0.85], dtype=np.float64)
    checks = []

    def heap(linkage):
        parents = np.empty((2, max_id + 1), dtype=np.uint32)
        stats = np.zeros((2, 3), dtype=np.int64)
        rc = lib.n20_lw_heap_cpu(
            u.ctypes.data_as(U32P), v.ctypes.data_as(U32P),
            sm.ctypes.data_as(F64P), ct.ctypes.data_as(I64P),
            ctypes.c_int64(3), thrs.ctypes.data_as(F64P),
            ctypes.c_int(2), ctypes.c_int(linkage),
            parents.ctypes.data_as(U32P), ctypes.c_uint32(max_id),
            stats.ctypes.data_as(I64P),
        )
        assert rc == 1, linkage
        return parents, stats

    p0, s0 = heap(0)
    p1, s1 = heap(1)
    p2, s2 = heap(2)
    # T=0.85 is index 1: only 1-2 (mean 0.90)
    r85_s3 = roots(p0[1])
    checks.append(("s3_T085_12", same(r85_s3[0], r85_s3[1]) and not same(r85_s3[0], r85_s3[2])))
    # T=0.42 is index 0
    r42_s3 = roots(p0[0])
    r42_c = roots(p1[0])
    r42_w = roots(p2[0])
    checks.append(("s3_T042_all", same(r42_s3[0], r42_s3[1]) and same(r42_s3[0], r42_s3[2])))
    checks.append(("complete_T042_3_out", same(r42_c[0], r42_c[1]) and not same(r42_c[0], r42_c[2])))
    checks.append(("wpgma_T042_all", same(r42_w[0], r42_w[1]) and same(r42_w[0], r42_w[2])))
    checks.append(("s3_height_pos", int(s0[0, 1]) >= 1))

    # RNN vs S3 heap at both T
    pr = np.empty((2, max_id + 1), dtype=np.uint32)
    st = np.zeros((2, 3), dtype=np.int64)
    rc = lib.n20_rnn_s3_cpu(
        u.ctypes.data_as(U32P), v.ctypes.data_as(U32P),
        sm.ctypes.data_as(F64P), ct.ctypes.data_as(I64P),
        ctypes.c_int64(3), thrs.ctypes.data_as(F64P),
        ctypes.c_int(2), ctypes.c_int64(50),
        pr.ctypes.data_as(U32P), ctypes.c_uint32(max_id),
        st.ctypes.data_as(I64P),
    )
    checks.append(("rnn_rc", rc == 1))
    checks.append(("rnn_eq_s3_T085", np.array_equal(pr[1], p0[1])))
    checks.append(("rnn_eq_s3_T042", np.array_equal(pr[0], p0[0])))

    # LU2: freeze nobody -> should equal S3 heap
    F0 = np.zeros(max_id + 1, dtype=np.uint8)
    pl = np.empty((2, max_id + 1), dtype=np.uint32)
    sl = np.zeros((2, 3), dtype=np.int64)
    rc = lib.n20_lu_alg2_cpu(
        u.ctypes.data_as(U32P), v.ctypes.data_as(U32P),
        sm.ctypes.data_as(F64P), ct.ctypes.data_as(I64P),
        ctypes.c_int64(3), F0.ctypes.data_as(U8P),
        thrs.ctypes.data_as(F64P), ctypes.c_int(2),
        pl.ctypes.data_as(U32P), ctypes.c_uint32(max_id),
        sl.ctypes.data_as(I64P),
    )
    checks.append(("lu2_emptyB_rc", rc == 1))
    checks.append(("lu2_emptyB_eq_s3", np.array_equal(pl[0], p0[0]) and np.array_equal(pl[1], p0[1])))

    # LU2: freeze all nodes; residual is the full graph; still exact
    Fall = np.array([0, 1, 1, 1], dtype=np.uint8)
    pl2 = np.empty((2, max_id + 1), dtype=np.uint32)
    sl2 = np.zeros((2, 3), dtype=np.int64)
    rc = lib.n20_lu_alg2_cpu(
        u.ctypes.data_as(U32P), v.ctypes.data_as(U32P),
        sm.ctypes.data_as(F64P), ct.ctypes.data_as(I64P),
        ctypes.c_int64(3), Fall.ctypes.data_as(U8P),
        thrs.ctypes.data_as(F64P), ctypes.c_int(2),
        pl2.ctypes.data_as(U32P), ctypes.c_uint32(max_id),
        sl2.ctypes.data_as(I64P),
    )
    checks.append(("lu2_allB_rc", rc == 1))
    checks.append(("lu2_allB_eq_s3", np.array_equal(pl2[0], p0[0]) and np.array_equal(pl2[1], p0[1])))

    failed = [n for n, ok in checks if not ok]
    doc = {
        "claim": "N20 unit; not a throughput claim",
        "host": socket.gethostname(),
        "checks": {n: bool(ok) for n, ok in checks},
        "failed": failed,
        "roots": {
            "s3_T085": r85_s3, "s3_T042": r42_s3,
            "complete_T042": r42_c, "wpgma_T042": r42_w,
        },
        "ok": not failed,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "N20_UNIT.json").write_text(json.dumps(doc, indent=2) + "\n")
    (ROOT / "notes/N20_UNIT.md").write_text(
        "# N20 unit\n\nCPU 3-node RAG. No GPU.\n\n"
        + ("PASS\n" if doc["ok"] else "FAIL " + str(failed) + "\n")
        + json.dumps(doc["roots"]) + "\n"
    )
    print(json.dumps(doc, indent=2), flush=True)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
