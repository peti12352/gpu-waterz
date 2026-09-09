#!/usr/bin/env python3
"""N13 L1: nsys kernel-sum vs cudaMemcpy API vs e2e. Do not assume artifact.

Not a 2 Gvox/s claim. Not 3090 Ti. Idle-5090 only.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b_dev_aff import build  # noqa: E402
from e6r_parhac import compile_d  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402

NOTE = ROOT / "notes/N13_D2H.md"
OUT = CACHE / "n13_d2h.json"
NSYS = "nsys"
BIG_REP = Path("/tmp/n12_216")
N12_NSYS_E2E = 4943.3
N12_FATBIN_E2E = 4923.9
REAL_MS = 500.0


def parks_env():
    e = os.environ.copy()
    e["WATERZ_UF_ALGO"] = "3"
    e["WATERZ_HOST_PARK"] = "0"
    e["WATERZ_AFF_PARK"] = "0"
    e["WATERZ_STAGE_MS"] = "1"
    e["WATERZ_AGG_LEVERS"] = "15"
    e.pop("WATERZ_AGG_EPS", None)
    e.pop("WATERZ_PAPER_E6T", None)
    return e


def find_rep():
    for p in (
        Path("/tmp/n12_216.nsys-rep"),
        Path("/tmp/n12_216.qdrep"),
        Path("/tmp/n13_216.nsys-rep"),
    ):
        if p.is_file():
            return p
    return None


def nsys_text(rep: Path, report: str) -> str:
    p = subprocess.run(
        [NSYS, "stats", "--report", report, str(rep)],
        capture_output=True, text=True, check=False,
    )
    return p.stdout + "\n" + p.stderr


def parse_ns(cell: str):
    return float(cell.replace(",", "").replace(" ", "") or 0)


def parse_table_sum(blob: str, name_substr=None):
    """Sum Total Time (ns) from nsys stats tables. Optional name filter."""
    total = 0.0
    memcpy = 0.0
    memcpy_rows = []
    in_table = False
    for ln in blob.splitlines():
        if "Total Time" in ln and "Time (%)" in ln:
            in_table = True
            continue
        if in_table and set(ln.strip()) <= set("- "):
            continue
        if in_table and not ln.strip():
            in_table = False
            continue
        if not in_table:
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            ns = parse_ns(parts[1])
        except ValueError:
            continue
        name = " ".join(parts[8:]) if len(parts) > 8 else " ".join(parts[3:])
        if name_substr is None:
            total += ns
        if name_substr and name_substr.lower() in name.lower():
            memcpy += ns
            memcpy_rows.append({"name": name, "ns": ns})
        if "cudaMemcpy" in name:
            memcpy += ns if name_substr is None else 0
            if name_substr is None:
                memcpy_rows.append({"name": name, "ns": ns})
    if name_substr:
        return memcpy, memcpy_rows
    return total, memcpy_rows


def parse_kern_sum(blob: str):
    total = 0.0
    rows = []
    in_table = False
    for ln in blob.splitlines():
        if "Total Time" in ln and "Time (%)" in ln:
            in_table = True
            continue
        if in_table and (not ln.strip() or set(ln.strip()) <= set("- ")):
            if ln.strip() and set(ln.strip()) <= set("- "):
                continue
            if not ln.strip():
                in_table = False
            continue
        if not in_table:
            continue
        m = re.match(
            r"\s*([\d.]+)\s+([\d,]+)\s+(\d+)\s+",
            ln,
        )
        if not m:
            continue
        ns = parse_ns(m.group(2))
        name = ln[m.end():].strip()
        total += ns
        rows.append({"name": name, "ns": ns, "pct": float(m.group(1))})
    return total, rows


def parse_api_memcpy(blob: str):
    total_api = 0.0
    memcpy = 0.0
    rows = []
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
            pct = float(parts[0])
            ns = parse_ns(parts[1])
        except ValueError:
            continue
        name = parts[-1]
        total_api += ns
        if name == "cudaMemcpy":
            memcpy += ns
            rows.append({"name": name, "ns": ns, "pct": pct})
    return total_api, memcpy, rows


def main():
    print("Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N13 L1 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)

    rep = find_rep()
    reran = False
    if rep is None:
        print("N13 L1 no nsys rep; profiling 2.16 UF=3 parks-off once.", flush=True)
        build()
        compile_d()
        cmd = [
            NSYS, "profile", "-t", "cuda,nvtx", "--stats=true",
            "-o", "/tmp/n13_216", "--force-overwrite", "true",
            sys.executable, "-u", str(ROOT / "scripts/n8_run216.py"),
        ]
        env = parks_env()
        env["WATERZ_SKIP_BUILD"] = "1"
        r = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if r.returncode != 0:
            print(f"N13 L1 nsys rc={r.returncode}", flush=True)
            return 1
        rep = Path("/tmp/n13_216.nsys-rep")
        reran = True
        p216 = CACHE / "n8_216.json"
        e2e = float(json.loads(p216.read_text())["e2e_ms"]) if p216.is_file() else N12_NSYS_E2E
    else:
        base = CACHE / "n13_baseline.json"
        e2e = N12_NSYS_E2E
        if base.is_file():
            e2e = float(json.loads(base.read_text()).get("e2e_ms") or N12_NSYS_E2E)
        print(f"N13 L1 parse {rep}", flush=True)

    kern = nsys_text(rep, "cuda_gpu_kern_sum")
    api = nsys_text(rep, "cuda_api_sum")
    k_ns, k_rows = parse_kern_sum(kern)
    api_ns, m_ns, m_rows = parse_api_memcpy(api)
    k_ms = k_ns / 1e6
    m_ms = m_ns / 1e6
    api_ms = api_ns / 1e6
    leftover = e2e - k_ms
    artifact = bool(m_ns >= 0.9 * k_ns and leftover < REAL_MS)
    real = bool(leftover >= REAL_MS)
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "rep": str(rep),
        "reran_nsys": reran,
        "K_kernel_ms": k_ms,
        "M_memcpy_api_ms": m_ms,
        "api_total_ms": api_ms,
        "E_e2e_ms": e2e,
        "E_minus_K_ms": leftover,
        "M_ge_0p9_K": bool(m_ns >= 0.9 * k_ns),
        "gate_artifact": artifact,
        "gate_real": real,
        "memcpy_rows": m_rows[:12],
        "top_kernels": k_rows[:8],
        "do_l1b": real,
        "keep_default": False,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N13 L1 D2H accounting\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti.\n\n"
        f"- rep={rep} reran={reran}\n"
        f"- K kernel GPU={k_ms:.1f} ms\n"
        f"- M cudaMemcpy API={m_ms:.1f} ms ({(100.0 * m_ns / api_ns) if api_ns else 0:.1f}% of API)\n"
        f"- E e2e={e2e:.1f} ms  E-K={leftover:.1f} ms\n"
        f"- M>=0.9K={doc['M_ge_0p9_K']} leftover<500={leftover < REAL_MS}\n"
        f"- gate_artifact={artifact} (stop D2H campaign)\n"
        f"- gate_real={real} (pinned nprop/nmerge only if true)\n"
        f"- L1b executed={False}\n"
        f"- keep_default=False\n"
    )
    print(
        f"N13 L1 K={k_ms:.1f} M={m_ms:.1f} E={e2e:.1f} leftover={leftover:.1f} "
        f"artifact={artifact} real={real} -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
