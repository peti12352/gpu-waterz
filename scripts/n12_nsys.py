#!/usr/bin/env python3
"""N12: NVTX nsys val then one 2.16 parks-off UF=3. Parse kernel % of 2582/2229.

Not a 2 Gvox/s claim. Not 3090 Ti. Idle-5090 only.
"""
from __future__ import annotations

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
from e6r_parhac import compile_d  # noqa: E402
from p1_make_big_indep import AFF, CACHE, card_busy, gpu_state, nfrag_bg  # noqa: E402
from task_gate import FRAGMENTS_VAL  # noqa: E402

NOTE = ROOT / "notes/N12_NSYS.md"
OUT = CACHE / "n12_nsys.json"
NSYS = "nsys"
VAL_REP = Path("/tmp/n12_val")
BIG_REP = Path("/tmp/n12_216")


def parks_env(env=None):
    e = os.environ.copy() if env is None else env
    e["WATERZ_UF_ALGO"] = "3"
    e["WATERZ_HOST_PARK"] = "0"
    e["WATERZ_AFF_PARK"] = "0"
    e["WATERZ_STAGE_MS"] = "1"
    e["WATERZ_AGG_LEVERS"] = "15"
    e.pop("WATERZ_AGG_EPS", None)
    return e


def ident():
    os.environ.update({
        "WATERZ_UF_ALGO": "3",
        "WATERZ_HOST_PARK": "0",
        "WATERZ_AFF_PARK": "0",
    })
    lib = __import__("ctypes").CDLL(str(ROOT / "src/libws_gpu.so"))
    bind(lib)
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    seg, _ = run_split(lib, aff, park=True)
    nfrag, bg = nfrag_bg(seg)
    gold = np.load(CACHE / "wz_fragments.npy")
    eq = bool(np.array_equal(seg, gold))
    return {
        "identity": eq and nfrag == FRAGMENTS_VAL,
        "array_equal": eq,
        "nfrag": nfrag,
        "bg": bg,
    }


def val_e2e():
    import segment as S
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    aff_d = S.DevBuf.from_host(aff)
    del aff
    out, ms = S.cuda_event_time(
        lambda: S.segment_d(aff_d, [0.3], return_device=True))
    st = dict(S.STAGE_MS)
    nlab = 0
    if out:
        nlab = int(np.unique(out[0].to_host()).size)
        for b in out:
            b.free()
    aff_d.free()
    print(f"N12 val e2e={ms:.1f} ms stages={st} nlab={nlab}", flush=True)


def nsys_profile(out_prefix: Path, argv: list[str], env: dict):
    cmd = [
        NSYS, "profile", "-t", "cuda,nvtx", "--stats=true",
        "-o", str(out_prefix), "--force-overwrite", "true",
        *argv,
    ]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return r.returncode


def nsys_dump(rep: Path) -> str:
    src = Path(str(rep) + ".nsys-rep")
    if not src.is_file():
        src = Path(str(rep) + ".qdrep")
    if not src.is_file():
        return f"missing {rep}.*"
    chunks = []
    for report in ("cuda_gpu_kern_sum", "nvtx_sum", "cuda_api_sum"):
        p = subprocess.run(
            [NSYS, "stats", "--report", report, str(src)],
            capture_output=True, text=True, check=False,
        )
        chunks.append(f"## {report}\n{p.stdout}\n{p.stderr}")
    return "\n".join(chunks)


def top_kernels(blob: str, n=20):
    rows = []
    for ln in blob.splitlines():
        if "k_" in ln or "cub::" in ln or "Device" in ln:
            rows.append(ln.strip())
    return rows[:n]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--ident":
        print(json.dumps(ident()), flush=True)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--val-e2e":
        val_e2e()
        return 0

    print("Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N12 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)

    env = parks_env()
    build()
    compile_d()

    idr = subprocess.run(
        [sys.executable, str(Path(__file__)), "--ident"],
        cwd=str(ROOT), env=parks_env(), capture_output=True, text=True,
    )
    ident_doc = {}
    for ln in idr.stdout.splitlines():
        if ln.startswith("{"):
            ident_doc = json.loads(ln)
    print(f"N12 ident {ident_doc}", flush=True)

    rc_val = nsys_profile(
        VAL_REP,
        [sys.executable, "-u", str(Path(__file__)), "--val-e2e"],
        parks_env({**env, "WATERZ_SKIP_BUILD": "1"}),
    )
    val_stats = nsys_dump(VAL_REP)

    rc_216 = nsys_profile(
        BIG_REP,
        [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
        parks_env({**env, "WATERZ_SKIP_BUILD": "1"}),
    )
    big_stats = nsys_dump(BIG_REP)
    stages = {}
    p216 = CACHE / "n8_216.json"
    if p216.is_file():
        stages = json.loads(p216.read_text())

    lines = [
        "# N12 nsys",
        "",
        "Not a 2 Gvox/s claim. Not 3090 Ti. Parks-off UF=3.",
        "",
        f"- ident={ident_doc}",
        f"- nsys val rc={rc_val} 2.16 rc={rc_216}",
        f"- 2.16 stages={stages.get('stages')} e2e={stages.get('e2e_ms')}",
        "",
        "## 2.16 top kernel lines",
        "",
    ]
    lines += [f"- `{x}`" for x in top_kernels(big_stats)]
    lines += ["", "## val NVTX/kernel dump", "", "```", val_stats[:12000], "```",
              "", "## 2.16 NVTX/kernel dump", "", "```", big_stats[:20000], "```", ""]
    NOTE.write_text("\n".join(lines))
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "ident": ident_doc,
        "rc_val": rc_val,
        "rc_216": rc_216,
        "run216": stages,
        "val_top": top_kernels(val_stats),
        "big_top": top_kernels(big_stats),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"N12 nsys -> {OUT} {NOTE}", flush=True)
    return 0 if ident_doc.get("identity") else 1


if __name__ == "__main__":
    raise SystemExit(main())
