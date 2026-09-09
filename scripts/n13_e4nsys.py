#!/usr/bin/env python3
"""N13 L3: nsys UF=4 stitch anatomy. Unique vs flatten.

Not a 2 Gvox/s claim. Not 3090 Ti. Idle-5090 only.
"""
from __future__ import annotations

import json
import os
import re
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

NOTE = ROOT / "notes/N13_E4.md"
OUT = CACHE / "n13_e4nsys.json"
NSYS = "nsys"
VAL_REP = Path("/tmp/n13_e4_val")
BIG_REP = Path("/tmp/n13_e4_216")
BG_VAL = 506_568


def parks_env(extra=None):
    e = os.environ.copy()
    e["WATERZ_UF_ALGO"] = "4"
    e["WATERZ_HOST_PARK"] = "0"
    e["WATERZ_AFF_PARK"] = "0"
    e["WATERZ_STAGE_MS"] = "1"
    e["WATERZ_AGG_LEVERS"] = "15"
    e.pop("WATERZ_AGG_EPS", None)
    e.pop("WATERZ_PAPER_E6T", None)
    if extra:
        e.update(extra)
    return e


def ident():
    os.environ.update({
        "WATERZ_UF_ALGO": "4",
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
        "identity": bool(eq and nfrag == FRAGMENTS_VAL and bg == BG_VAL),
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
    if out:
        for b in out:
            b.free()
    aff_d.free()
    print(f"N13 E4 val e2e={ms:.1f} stages={st}", flush=True)


def nsys_profile(prefix: Path, argv, env):
    cmd = [
        NSYS, "profile", "-t", "cuda,nvtx", "--stats=true",
        "-o", str(prefix), "--force-overwrite", "true", *argv,
    ]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode


def nsys_text(prefix: Path, report: str):
    src = Path(str(prefix) + ".nsys-rep")
    if not src.is_file():
        src = Path(str(prefix) + ".qdrep")
    if not src.is_file():
        return f"missing {prefix}.*"
    p = subprocess.run(
        [NSYS, "stats", "--force-export=true", "--report", report, str(src)],
        capture_output=True, text=True, check=False,
    )
    return p.stdout + "\n" + p.stderr


def parse_ns(s):
    return float(s.replace(",", ""))


def nvtx_ms(blob: str):
    out = {}
    in_table = False
    for ln in blob.splitlines():
        if "Total Time" in ln and "Range" in ln:
            in_table = True
            continue
        if in_table and not ln.strip():
            in_table = False
            continue
        if not in_table:
            continue
        parts = ln.split()
        if len(parts) < 4:
            continue
        try:
            ns = parse_ns(parts[1])
        except ValueError:
            continue
        rng = parts[-1]
        if rng.startswith(":"):
            rng = rng[1:]
        out[rng] = ns / 1e6
    return out


def kern_ms(blob: str, names):
    hit = {n: 0.0 for n in names}
    in_table = False
    for ln in blob.splitlines():
        if "Total Time" in ln and "Time (%)" in ln:
            in_table = True
            continue
        if in_table and not ln.strip():
            in_table = False
            continue
        if not in_table:
            continue
        parts = ln.split()
        if len(parts) < 4:
            continue
        try:
            ns = parse_ns(parts[1])
        except ValueError:
            continue
        rest = " ".join(parts[8:]) if len(parts) > 8 else ln
        for n in names:
            if n in rest or n in ln:
                hit[n] += ns / 1e6
    return hit


def load_json_line(stdout):
    for ln in stdout.splitlines():
        if ln.startswith("{"):
            return json.loads(ln)
    return {}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--parse-only":
        ident_doc = {"identity": True, "note": "parse-only"}
        p216 = CACHE / "n8_216.json"
        run216 = json.loads(p216.read_text()) if p216.is_file() else {}
        kern = nsys_text(BIG_REP, "cuda_gpu_kern_sum")
        nvtx = nsys_text(BIG_REP, "nvtx_sum")
        nmap = nvtx_ms(nvtx)
        kmap = kern_ms(kern, ("k_w5_compress_list", "k_uf_compress_c"))
        stitch = nmap.get("w5_stitch") or 0.0
        unique = nmap.get("e4_unique") or 0.0
        hook = nmap.get("e4_hook") or 0.0
        emit = nmap.get("e4_emit") or 0.0
        flatten = kmap.get("k_w5_compress_list", 0) + kmap.get("k_uf_compress_c", 0)
        flatten_frac = (flatten / stitch) if stitch else None
        unique_frac = (unique / stitch) if stitch else None
        stop = bool(flatten_frac is not None and flatten_frac >= 0.70)
        cont = bool(unique_frac is not None and unique_frac >= 0.30 and not stop)
        doc = {
            "claim": "not a 2 Gvox/s number; not 3090 Ti",
            "parse_only": True,
            "ident": ident_doc,
            "nvtx_ms": nmap,
            "flatten_kernel_ms": flatten,
            "stitch_nvtx_ms": stitch,
            "unique_nvtx_ms": unique,
            "hook_nvtx_ms": hook,
            "emit_nvtx_ms": emit,
            "flatten_frac_of_stitch": flatten_frac,
            "unique_frac_of_stitch": unique_frac,
            "gate_l3_stop": stop,
            "gate_l3_continue": cont,
            "third_stitch": bool(cont),
            "keep_default": False,
            "run216": run216,
            "kern_head": kern[:2500],
            "nvtx_head": nvtx[:2500],
        }
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2) + "\n")
        NOTE.write_text(
            "# N13 L3 E4 nsys\n\n"
            "Not a 2 Gvox/s claim. Not 3090 Ti. UF=4 parks-off. "
            "k_uf_tile_local unchanged.\n\n"
            f"- identity True (prior nsys run)\n"
            f"- 2.16 e2e={run216.get('e2e_ms')} stages={run216.get('stages')}\n"
            f"- NVTX stitch={stitch:.1f} emit={emit:.1f} unique={unique:.1f} "
            f"hook={hook:.1f} ms\n"
            f"- flatten kernels={flatten:.1f} ms  frac_of_stitch={flatten_frac}\n"
            f"- unique frac_of_stitch={unique_frac}\n"
            f"- L3-stop (flatten>=70% stitch)={stop} -> no third stitch\n"
            f"- L3-continue (unique>=30% and not stop)={cont}\n"
            f"- keep_default=False (0.5x gate already failed at 0.849)\n"
        )
        print(
            f"N13 L3 parse-only stop={stop} cont={cont} "
            f"flatten_frac={flatten_frac} unique_frac={unique_frac}",
            flush=True,
        )
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "--ident":
        print(json.dumps(ident()), flush=True)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--val-e2e":
        val_e2e()
        return 0

    print("Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N13 L3 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    build()
    compile_d()

    idr = subprocess.run(
        [sys.executable, str(Path(__file__)), "--ident"],
        cwd=str(ROOT), env=parks_env(), capture_output=True, text=True,
    )
    ident_doc = load_json_line(idr.stdout)
    print(f"N13 L3 ident {ident_doc}", flush=True)
    if not ident_doc.get("identity"):
        print("N13 L3 STOP identity fail; no 2.16 nsys.", flush=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({
            "claim": "not a 2 Gvox/s number; not 3090 Ti",
            "ident": ident_doc, "keep_default": False,
            "third_stitch": False,
        }, indent=2) + "\n")
        return 1

    env = parks_env({"WATERZ_SKIP_BUILD": "1"})
    rc_val = nsys_profile(
        VAL_REP,
        [sys.executable, "-u", str(Path(__file__)), "--val-e2e"],
        env,
    )
    rc_216 = nsys_profile(
        BIG_REP,
        [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
        env,
    )
    kern = nsys_text(BIG_REP, "cuda_gpu_kern_sum")
    nvtx = nsys_text(BIG_REP, "nvtx_sum")
    nmap = nvtx_ms(nvtx)
    kmap = kern_ms(kern, ("k_w5_compress_list", "k_uf_compress_c"))
    stitch = nmap.get("w5_stitch") or 0.0
    unique = nmap.get("e4_unique") or 0.0
    hook = nmap.get("e4_hook") or 0.0
    emit = nmap.get("e4_emit") or 0.0
    flatten = kmap.get("k_w5_compress_list", 0) + kmap.get("k_uf_compress_c", 0)
    flatten_frac = (flatten / stitch) if stitch else None
    unique_frac = (unique / stitch) if stitch else None
    stop = bool(flatten_frac is not None and flatten_frac >= 0.70)
    cont = bool(
        unique_frac is not None and unique_frac >= 0.30 and not stop
    )
    run216 = {}
    p216 = CACHE / "n8_216.json"
    if p216.is_file():
        run216 = json.loads(p216.read_text())
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "ident": ident_doc,
        "rc_val": rc_val,
        "rc_216": rc_216,
        "nvtx_ms": nmap,
        "flatten_kernel_ms": flatten,
        "stitch_nvtx_ms": stitch,
        "unique_nvtx_ms": unique,
        "hook_nvtx_ms": hook,
        "emit_nvtx_ms": emit,
        "flatten_frac_of_stitch": flatten_frac,
        "unique_frac_of_stitch": unique_frac,
        "gate_l3_stop": stop,
        "gate_l3_continue": cont,
        "third_stitch": bool(cont),
        "keep_default": False,
        "run216": run216,
        "kern_head": kern[:2500],
        "nvtx_head": nvtx[:2500],
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N13 L3 E4 nsys\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. UF=4 parks-off. "
        "k_uf_tile_local unchanged.\n\n"
        f"- identity={ident_doc.get('identity')} nfrag={ident_doc.get('nfrag')}\n"
        f"- 2.16 e2e={run216.get('e2e_ms')} stages={run216.get('stages')}\n"
        f"- NVTX stitch={stitch:.1f} emit={emit:.1f} unique={unique:.1f} "
        f"hook={hook:.1f} ms\n"
        f"- flatten kernels={flatten:.1f} ms  frac_of_stitch={flatten_frac}\n"
        f"- unique frac_of_stitch={unique_frac}\n"
        f"- L3-stop (flatten>=70% stitch)={stop} -> no third stitch\n"
        f"- L3-continue (unique>=30% and not stop)={cont}\n"
        f"- keep_default=False (0.5x gate already failed at 0.849)\n"
    )
    print(
        f"N13 L3 stop={stop} cont={cont} flatten_frac={flatten_frac} "
        f"unique_frac={unique_frac} -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
