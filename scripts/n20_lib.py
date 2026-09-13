"""N20 ctypes wrappers around rac_agg.cpp agglomerators."""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import SO, compile_so, load_rag  # noqa: E402
from task_gate import AFF_THRESHOLDS  # noqa: E402

U32P = ctypes.POINTER(ctypes.c_uint32)
U8P = ctypes.POINTER(ctypes.c_uint8)
F64P = ctypes.POINTER(ctypes.c_double)
I64P = ctypes.POINTER(ctypes.c_int64)


def load_n20():
    compile_so()
    lib = ctypes.CDLL(str(SO))
    lib.n20_lw_heap_cpu.restype = ctypes.c_int
    lib.n20_lw_heap_cpu.argtypes = [
        U32P, U32P, F64P, I64P, ctypes.c_int64, F64P, ctypes.c_int, ctypes.c_int,
        U32P, ctypes.c_uint32, I64P,
    ]
    lib.n20_rnn_s3_cpu.restype = ctypes.c_int
    lib.n20_rnn_s3_cpu.argtypes = [
        U32P, U32P, F64P, I64P, ctypes.c_int64, F64P, ctypes.c_int, ctypes.c_int64,
        U32P, ctypes.c_uint32, I64P,
    ]
    lib.n20_lu_alg2_cpu.restype = ctypes.c_int
    lib.n20_lu_alg2_cpu.argtypes = [
        U32P, U32P, F64P, I64P, ctypes.c_int64, U8P, F64P, ctypes.c_int,
        U32P, ctypes.c_uint32, I64P,
    ]
    lib.s4_fast_cpu.restype = ctypes.c_int
    lib.s4_fast_cpu.argtypes = [
        U32P, U32P, F64P, I64P, ctypes.c_int64, F64P, ctypes.c_int,
        U32P, ctypes.c_uint32,
    ]
    return lib


def rag_ptrs(u, v, sm, ct, thrs, parents, stats=None, extra=None):
    args = [
        u.ctypes.data_as(U32P),
        v.ctypes.data_as(U32P),
        sm.ctypes.data_as(F64P),
        ct.ctypes.data_as(I64P),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(F64P),
        ctypes.c_int(len(thrs)),
    ]
    if extra:
        args.extend(extra)
    args.extend([
        parents.ctypes.data_as(U32P),
        ctypes.c_uint32(parents.shape[1] - 1 if parents.ndim == 2 else parents.size - 1),
    ])
    if stats is not None:
        args.append(stats.ctypes.data_as(I64P))
    return args
