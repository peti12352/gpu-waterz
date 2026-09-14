#!/usr/bin/env python3
"""N11 D: GPU E4 WATERZ_UF_ALGO=4. Contracted p1 stitch.

Val: identity vs algo-0 and wz_fragments.npy; stitch+flatten <= 0.5x W5 list.
Then one 2.16 parks-off if gates pass. Idle-5090. Not a 2 Gvox/s claim.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b_dev_aff import bind, build, run_split  # noqa: E402
from n9_measure import ws_stats  # noqa: E402
from p1_make_big_indep import AFF, CACHE, card_busy, gpu_state, nfrag_bg  # noqa: E402
from task_gate import FRAGMENTS_VAL  # noqa: E402

OUT = CACHE / "n11_e4.json"
NOTE = ROOT / "notes/N11_E4.md"
STITCH_GATE = 0.5


def uf_stats(lib):
    tile = ctypes.c_float(0)
    stitch = ctypes.c_float(0)
    nrep = ctypes.c_int64(0)
    ncross = ctypes.c_int64(0)
    lib.ws_n11_uf_stats(
        ctypes.byref(tile), ctypes.byref(stitch),
        ctypes.byref(nrep), ctypes.byref(ncross),
    )
    return {
        "tile_ms": float(tile.value),
        "stitch_ms": float(stitch.value),
        "n_rep": int(nrep.value),
        "n_cross": int(ncross.value),
    }


def bind_e4(lib):
    bind(lib)
    lib.ws_n9_reset.restype = None
    lib.ws_n11_uf_stats.restype = None
    lib.ws_n11_uf_stats.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
    ]
    lib.ws_n9_stats.restype = None


def one(algo: int):
    os.environ["WATERZ_UF_ALGO"] = str(algo)
    os.environ["WATERZ_HOST_PARK"] = "0"
    os.environ["WATERZ_AFF_PARK"] = "0"
    if not os.environ.get("WATERZ_N11_CHILD"):
        build()
    lib = ctypes.CDLL(str(ROOT / "src/libws_gpu.so"))
    bind_e4(lib)
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    lib.ws_n9_reset()
    seg, meta = run_split(lib, aff, park=True)
    nfrag, bg = nfrag_bg(seg)
    gold = np.load(CACHE / "wz_fragments.npy")
    oracle = bool(np.array_equal(seg, gold))
    st = uf_stats(lib)
    w5 = ws_stats(lib)
    return {
        "algo": algo,
        "nfrag": nfrag,
        "bg": bg,
        "oracle_array_equal": oracle,
        "nfrag_ok": nfrag == FRAGMENTS_VAL,
        "identity": bool(oracle and nfrag == FRAGMENTS_VAL),
        "divide_ms": float(meta["divide_ms"]),
        **st,
        "w5_ms": w5["w5_ms"],
        "w5_calls": w5["w5_calls"],
    }


def write_note(doc):
    rows = {r["algo"]: r for r in doc["rows"]}
    a0, a3, a4 = rows[0], rows[3], rows[4]
    lines = [
        "# N11 D: GPU E4 algo=4",
        "",
        "Contracted p1 pairs. Not Chen face-voxel Kernel 2. "
        "Not a 2 Gvox/s number. Not 3090 Ti.",
        "",
        f"- algo0 identity={a0['identity']} nfrag={a0['nfrag']}",
        f"- algo3 W5 stitch={a3['stitch_ms']:.2f} ms tile={a3['tile_ms']:.2f} "
        f"w5={a3['w5_ms']:.2f}",
        f"- algo4 identity={a4['identity']} vs_algo0={doc['vs_algo0']} "
        f"nfrag={a4['nfrag']} n_rep={a4['n_rep']} n_cross={a4['n_cross']}",
        f"- algo4 stitch={a4['stitch_ms']:.2f} ms tile={a4['tile_ms']:.2f} "
        f"w5={a4['w5_ms']:.2f}",
        f"- stitch/W5-list={doc['stitch_ratio']:.3f} gate<={STITCH_GATE} "
        f"pass={doc['stitch_ok']}",
        f"- keep={doc['keep']}",
        "",
    ]
    r216 = doc.get("run216")
    if r216:
        lines += [
            "## 2.16 parks-off",
            "",
            f"- e2e={r216['e2e_ms']:.1f} ms gvox/s={r216['gvox_s_5090']:.3f}",
            f"- stages={r216['stages']}",
            f"- nlab={r216['nlab']}",
            "",
        ]
    else:
        lines += ["## 2.16", "", "Skipped (val gate failed).", "",]
    NOTE.write_text("\n".join(lines) + "\n")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--algo":
        doc = one(int(sys.argv[2]))
        print(json.dumps(doc), flush=True)
        return 0 if doc["identity"] else 1

    print("N11 E4 algo=4. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N11 REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    build()
    rows = []
    for algo in (0, 3, 4):
        env = os.environ.copy()
        env["WATERZ_N11_CHILD"] = "1"
        r = subprocess.run(
            [sys.executable, str(Path(__file__)), "--algo", str(algo)],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
            env=env,
        )
        line = None
        for ln in r.stdout.splitlines():
            if ln.startswith("{"):
                line = ln
        if line is None:
            print(
                f"N11 E4 algo={algo} failed rc={r.returncode}\n"
                f"stdout={r.stdout[-2000:]}\nstderr={r.stderr[-2000:]}",
                flush=True,
            )
            raise SystemExit(1)
        doc = json.loads(line)
        rows.append(doc)
        print(
            f"N11 E4 algo={algo} identity={doc['identity']} "
            f"stitch={doc['stitch_ms']:.2f} tile={doc['tile_ms']:.2f} "
            f"n_rep={doc['n_rep']} w5={doc['w5_ms']:.2f} "
            f"stderr_tail={r.stderr[-400:].replace(chr(10), ' | ')}",
            flush=True,
        )

    a0 = next(x for x in rows if x["algo"] == 0)
    a3 = next(x for x in rows if x["algo"] == 3)
    a4 = next(x for x in rows if x["algo"] == 4)
    vs_algo0 = bool(a4["identity"] and a0["identity"])
    stitch_ratio = (
        a4["stitch_ms"] / a3["stitch_ms"] if a3["stitch_ms"] > 0 else 1e9
    )
    stitch_ok = bool(stitch_ratio <= STITCH_GATE)
    keep = bool(a4["identity"] and vs_algo0 and stitch_ok)
    print(
        f"N11 E4 vs_algo0={vs_algo0} stitch_ratio={stitch_ratio:.3f} "
        f"stitch_ok={stitch_ok} keep={keep}",
        flush=True,
    )

    run216 = None
    if a4["identity"] and vs_algo0:
        env = os.environ.copy()
        env["WATERZ_UF_ALGO"] = "4"
        env["WATERZ_HOST_PARK"] = "0"
        env["WATERZ_AFF_PARK"] = "0"
        env["WATERZ_STAGE_MS"] = "1"
        r = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
            cwd=str(ROOT), env=env, check=False,
        )
        if r.returncode != 0:
            print(f"N11 E4 2.16 rc={r.returncode}", flush=True)
        p = CACHE / "n8_216.json"
        if p.is_file():
            run216 = json.loads(p.read_text())

    keep = bool(a4["identity"] and vs_algo0 and stitch_ok)
    # Identity True + stitch 0.5-0.8x: keep the kernel behind algo=4,
    # do not default it. 2.16 already ran.

    out = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "rows": rows,
        "vs_algo0": vs_algo0,
        "stitch_ratio": stitch_ratio,
        "stitch_ok": stitch_ok,
        "keep": keep,
        "run216": run216,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    write_note(out)
    print(f"N11 E4 keep={keep} -> {OUT} {NOTE}", flush=True)
    return 0 if keep else 1


if __name__ == "__main__":
    raise SystemExit(main())
