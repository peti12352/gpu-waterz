#!/usr/bin/env python3
"""N11 B: count n_rep vs n_face vs W5 n_list after tile-local. Count-only.

Val, then one 720 Mvox z-slab if the val gate passes. Recip as e9b.
Kill E4 if n_rep/n_face >= 0.25 or n_rep/nvox >= 0.10.
Idle-5090. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b_dev_aff import bind, build, run_split  # noqa: E402
from n8_run216 import official_216  # noqa: E402
from p1_make_big_indep import AFF, CACHE, card_busy, gpu_state  # noqa: E402
from task_gate import AFF_HIGH, AFF_LOW, FRAGMENTS_VAL  # noqa: E402
import segment as S  # noqa: E402

OUT = CACHE / "n11_nrep.json"
NOTE = ROOT / "notes/N11_NREP.md"
GATE_FACE = 0.25
GATE_NVOX = 0.10
SLAB_Z = 125


def bind_nrep(lib):
    bind(lib)
    lib.ws_n11_nrep.restype = ctypes.c_int
    lib.ws_n11_nrep.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong),
    ]


def count_nrep(lib, bits_d, z, y, x):
    n_face = ctypes.c_ulonglong(0)
    n_list = ctypes.c_ulonglong(0)
    n_rep = ctypes.c_ulonglong(0)
    n_cross = ctypes.c_ulonglong(0)
    rc = lib.ws_n11_nrep(
        ctypes.c_void_p(bits_d.ptr), z, y, x,
        ctypes.byref(n_face), ctypes.byref(n_list),
        ctypes.byref(n_rep), ctypes.byref(n_cross),
    )
    nvox = z * y * x
    nf = int(n_face.value)
    nl = int(n_list.value)
    nr = int(n_rep.value)
    nc = int(n_cross.value)
    face_ratio = (nr / nf) if nf else None
    nvox_ratio = (nr / nvox) if nvox else None
    e4_ok = (
        nf > 0
        and face_ratio is not None
        and face_ratio < GATE_FACE
        and nvox_ratio < GATE_NVOX
    )
    return {
        "rc": int(rc),
        "shape": [z, y, x],
        "nvox": nvox,
        "n_face": nf,
        "n_list": nl,
        "n_rep": nr,
        "n_cross": nc,
        "n_list_over_nvox": (nl / nvox) if nvox else None,
        "n_rep_over_n_face": face_ratio,
        "n_rep_over_nvox": nvox_ratio,
        "e4_gate": e4_ok,
        "kill": (not e4_ok) or rc != 0,
    }


def flow_bits(lib, aff_u8):
    z, y, x = (int(v) for v in aff_u8.shape[1:])
    aff_d = S.DevBuf.from_host(aff_u8)
    bits_d = S.DevBuf((z, y, x), np.uint8)
    lib.ws_flow_d(
        ctypes.c_void_p(aff_d.ptr), z, y, x,
        ctypes.c_float(AFF_LOW), ctypes.c_float(AFF_HIGH),
        ctypes.c_void_p(bits_d.ptr),
    )
    aff_d.free()
    return bits_d, z, y, x


def load_val_aff():
    with h5py.File(AFF, "r") as f:
        return np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)


def load_slab_aff():
    # Slice only the first official z-tile. Do not load the full 6 GiB.
    candidates = [
        ROOT / "data/ws_bounty/big/affinity.h5",
        ROOT / "data/cache/big_216.h5",
        Path("/tmp/wz_big_216.h5"),
    ]
    for p in candidates:
        if p.is_file():
            print(f"N11 nrep slab load {p} z=:{SLAB_Z}", flush=True)
            with h5py.File(p, "r") as f:
                return np.ascontiguousarray(
                    f["affinity"][:, :SLAB_Z, :, :], dtype=np.uint8
                )
    print("N11 nrep slab: 2.16 h5 missing, building via official_216", flush=True)
    aff = official_216()
    return np.ascontiguousarray(aff[:, :SLAB_Z, :, :])


def write_note(doc):
    val = doc["val"]
    slab = doc.get("slab")
    lines = [
        "# N11 B: n_rep after tile-local",
        "",
        "Count-only. Recip as e9b. No stitch. Not a 2 Gvox/s number. Not 3090 Ti.",
        "",
        f"Gate: n_rep/n_face < {GATE_FACE} and n_rep/nvox < {GATE_NVOX}.",
        "",
        "## Val",
        "",
        f"- nvox={val['nvox']} n_face={val['n_face']} n_list={val['n_list']} "
        f"n_rep={val['n_rep']} n_cross={val['n_cross']}",
        f"- n_rep/n_face={val['n_rep_over_n_face']}",
        f"- n_rep/nvox={val['n_rep_over_nvox']}",
        f"- n_list/nvox={val['n_list_over_nvox']}",
        f"- identity vs wz_fragments.npy: {doc['identity']}",
        f"- e4_gate={val['e4_gate']} kill={val['kill']}",
        "",
    ]
    if slab:
        lines += [
            "## 720 Mvox z-slab",
            "",
            f"- nvox={slab['nvox']} n_face={slab['n_face']} n_list={slab['n_list']} "
            f"n_rep={slab['n_rep']} n_cross={slab['n_cross']}",
            f"- n_rep/n_face={slab['n_rep_over_n_face']}",
            f"- n_rep/nvox={slab['n_rep_over_nvox']}",
            f"- e4_gate={slab['e4_gate']} kill={slab['kill']}",
            "",
        ]
    else:
        lines += ["## 720 Mvox z-slab", "", "Skipped (val gate failed).", "",]
    e4 = doc["e4_alive"]
    lines += [
        "## Verdict",
        "",
        "E4 alive." if e4 else "E4 dead. Skip D.",
        "",
    ]
    NOTE.write_text("\n".join(lines) + "\n")


def main():
    print("N11 n_rep count. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N11 REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    build()
    lib = ctypes.CDLL(str(ROOT / "src/libws_gpu.so"))
    bind_nrep(lib)

    aff = load_val_aff()
    bits_d, z, y, x = flow_bits(lib, aff)
    val = count_nrep(lib, bits_d, z, y, x)
    bits_d.free()
    print(
        f"N11 val n_face={val['n_face']} n_list={val['n_list']} "
        f"n_rep={val['n_rep']} n_cross={val['n_cross']} "
        f"n_rep/n_face={val['n_rep_over_n_face']} "
        f"n_rep/nvox={val['n_rep_over_nvox']} kill={val['kill']}",
        flush=True,
    )

    gold = np.load(CACHE / "wz_fragments.npy")
    seg, meta = run_split(lib, aff, park=True)
    identity = bool(np.array_equal(seg, gold))
    nfrag, bg = int((np.unique(seg) != 0).sum()), int((seg == 0).sum())
    print(
        f"N11 val identity={identity} nfrag={nfrag} bg={bg} "
        f"expect nfrag={FRAGMENTS_VAL} meta={meta}",
        flush=True,
    )
    del aff, seg

    slab = None
    if (not val["kill"]) and identity:
        aff_s = load_slab_aff()
        bits_s, zs, ys, xs = flow_bits(lib, aff_s)
        slab = count_nrep(lib, bits_s, zs, ys, xs)
        bits_s.free()
        del aff_s
        print(
            f"N11 slab n_face={slab['n_face']} n_list={slab['n_list']} "
            f"n_rep={slab['n_rep']} n_cross={slab['n_cross']} "
            f"n_rep/n_face={slab['n_rep_over_n_face']} "
            f"n_rep/nvox={slab['n_rep_over_nvox']} kill={slab['kill']}",
            flush=True,
        )

    e4_alive = (
        (not val["kill"])
        and identity
        and (slab is not None)
        and (not slab["kill"])
    )
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "identity": identity,
        "val": val,
        "slab": slab,
        "e4_alive": e4_alive,
        "gate_face": GATE_FACE,
        "gate_nvox": GATE_NVOX,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    write_note(doc)
    print(f"N11 nrep e4_alive={e4_alive} -> {OUT} {NOTE}", flush=True)
    return 0 if identity else 1


if __name__ == "__main__":
    raise SystemExit(main())
