#!/usr/bin/env python3
"""N19 nsys with --force-export=true → n19_owners.json.

Kill proposed micro-opts whose max kernel <200 ms of e2e.
Not a substitute for a GPU timed gate. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from n13_baseline import parks_env  # noqa: E402
from n17_gate import DEEP  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from b_dev_aff import build  # noqa: E402
from e6r_parhac import compile_d  # noqa: E402

NSYS = "nsys"
VAL_REP = Path("/tmp/n19_n17_val")
BIG_REP = Path("/tmp/n19_n17_216")
OUT = CACHE / "n19_nsys.json"
OWNERS = CACHE / "n19_owners.json"
NOTE = ROOT / "notes/N19_I0_NSYS.md"
CLAIM = "not a 2 Gvox/s number; not 3090 Ti"
KILL_THRESH_MS = 200.0


def nsys_profile(out_prefix: Path, argv: list[str], env: dict):
    cmd = [
        NSYS, "profile", "-t", "cuda,nvtx", "--stats=true",
        "-o", str(out_prefix), "--force-overwrite", "true",
        *argv,
    ]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode


def nsys_dump(rep: Path) -> str:
    src = Path(str(rep) + ".nsys-rep")
    if not src.is_file():
        src = Path(str(rep) + ".qdrep")
    if not src.is_file():
        return f"missing {rep}.*"
    chunks = []
    for report in ("cuda_gpu_kern_sum", "nvtx_sum"):
        p = subprocess.run(
            [NSYS, "stats", "--force-export=true", "--report", report, str(src)],
            capture_output=True, text=True, check=False,
        )
        chunks.append(f"## {report}\n{p.stdout}\n{p.stderr}")
    return "\n".join(chunks)


def parse_kernel_ms(blob: str, n=40):
    """Parse nsys cuda_gpu_kern_sum rows → [{name, total_ms, pct}, ...]."""
    rows = []
    for ln in blob.splitlines():
        if "Time (%)" in ln or not ln.strip() or ln.strip().startswith("-"):
            continue
        # Typical: pct total_ns count avg ... name
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            pct = float(parts[0])
            total_ns = float(parts[1].replace(",", ""))
        except ValueError:
            continue
        name = parts[-1]
        if not any(x in name for x in ("k_", "cub::", "Device", "parhac", "hash")):
            if "k_" not in ln and "cub::" not in ln:
                continue
            # name may have spaces — take last token with k_ or cub
            for p in reversed(parts):
                if "k_" in p or "cub::" in p or "Device" in p:
                    name = p
                    break
        rows.append({
            "name": name,
            "total_ms": total_ns / 1e6,
            "pct": pct,
        })
    rows.sort(key=lambda r: -r["total_ms"])
    return rows[:n]


def parse_nvtx_ms(blob: str):
    rows = []
    in_nvtx = False
    for ln in blob.splitlines():
        if "## nvtx_sum" in ln:
            in_nvtx = True
            continue
        if in_nvtx and ln.startswith("## "):
            break
        if not in_nvtx:
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            # various nsys formats; look for range name at end
            total_ns = None
            for i, p in enumerate(parts):
                try:
                    v = float(p.replace(",", ""))
                    if v > 1000:  # ns-ish
                        total_ns = v
                        break
                except ValueError:
                    continue
            if total_ns is None:
                continue
            name = parts[-1].strip(":")
            if name in ("Name", "Range", "-----"):
                continue
            rows.append({"name": name, "total_ms": total_ns / 1e6})
        except (ValueError, IndexError):
            continue
    rows.sort(key=lambda r: -r["total_ms"])
    return rows[:30]


def owner_gate(owners: dict, kernel_substr: str) -> dict:
    """Return whether a micro-opt targeting kernel_substr clears ≥200 ms."""
    e2e = float(owners.get("e2e_ms") or 0)
    hits = [
        k for k in (owners.get("kernels") or [])
        if kernel_substr in k.get("name", "")
    ]
    max_ms = max((k["total_ms"] for k in hits), default=0.0)
    return {
        "kernel_substr": kernel_substr,
        "max_ms": max_ms,
        "ok_to_try": max_ms >= KILL_THRESH_MS,
        "kill_thresh_ms": KILL_THRESH_MS,
        "e2e_ms": e2e,
    }


def main():
    print(f"N19 I0 nsys. {CLAIM}.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N19 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)

    extra = {**DEEP, "WATERZ_EMIT_HOLES": "1", "WATERZ_NVTX": "1"}
    build()
    compile_d()

    rc_val = nsys_profile(
        VAL_REP,
        [sys.executable, "-u", str(ROOT / "scripts/n12_nsys.py"), "--val-e2e"],
        parks_env({**extra, "WATERZ_SKIP_BUILD": "1"}),
    )
    val_stats = nsys_dump(VAL_REP)

    rc_216 = nsys_profile(
        BIG_REP,
        [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
        parks_env({**extra, "WATERZ_SKIP_BUILD": "1"}),
    )
    big_stats = nsys_dump(BIG_REP)
    stages = {}
    p216 = CACHE / "n8_216.json"
    if p216.is_file():
        stages = json.loads(p216.read_text())
    e2e = float(stages.get("e2e_ms") or 0)

    kern = parse_kernel_ms(big_stats)
    nvtx = parse_nvtx_ms(big_stats)
    owners = {
        "claim": CLAIM,
        "e2e_ms": e2e,
        "stages": stages.get("stages"),
        "kernels": kern,
        "nvtx": nvtx,
        "kill_thresh_ms": KILL_THRESH_MS,
        "gates": {
            "vcount": owner_gate({"kernels": kern, "e2e_ms": e2e}, "count_v"),
            "sort": owner_gate({"kernels": kern, "e2e_ms": e2e}, "RadixSort"),
            "hook": owner_gate({"kernels": kern, "e2e_ms": e2e}, "hook"),
            "hash": owner_gate({"kernels": kern, "e2e_ms": e2e}, "hash"),
            "stitch": owner_gate({"kernels": kern, "e2e_ms": e2e}, "stitch"),
        },
        "gpu": gpu_state(),
        "rc_val": rc_val,
        "rc_216": rc_216,
    }
    # Also match NVTX range names
    for key in ("vcount", "sort", "hash_insert", "propose", "rebuild"):
        hits = [n for n in nvtx if key in n["name"].lower()]
        owners["gates"][f"nvtx_{key}"] = {
            "max_ms": max((h["total_ms"] for h in hits), default=0.0),
            "ok_to_try": max((h["total_ms"] for h in hits), default=0.0) >= KILL_THRESH_MS,
        }

    CACHE.mkdir(parents=True, exist_ok=True)
    OWNERS.write_text(json.dumps(owners, indent=2) + "\n")
    OUT.write_text(json.dumps({
        "claim": CLAIM, "owners_path": str(OWNERS),
        "val_top": kern[:15], "big_top": kern[:20],
        "run216": stages, "rc_val": rc_val, "rc_216": rc_216,
    }, indent=2) + "\n")

    lines = [
        "# N19 I0 nsys",
        "",
        f"{CLAIM}. Parks off. Information only.",
        "",
        f"- e2e={e2e} stages={stages.get('stages')}",
        f"- kill_thresh={KILL_THRESH_MS} ms",
        f"- gates={json.dumps(owners['gates'], indent=2)}",
        "",
        "## top kernels",
        "",
    ]
    for k in kern[:20]:
        lines.append(f"- {k['total_ms']:.1f} ms ({k['pct']:.1f}%) `{k['name']}`")
    lines += ["", "## nvtx", ""]
    for n in nvtx[:15]:
        lines.append(f"- {n['total_ms']:.1f} ms `{n['name']}`")
    lines += ["", "```", big_stats[:16000], "```", ""]
    NOTE.write_text("\n".join(lines))
    print(f"N19 nsys -> {OWNERS} {NOTE}", flush=True)
    print(json.dumps({"e2e": e2e, "n_kern": len(kern), "gates": owners["gates"]}),
          flush=True)
    return 0 if rc_216 == 0 and len(kern) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
