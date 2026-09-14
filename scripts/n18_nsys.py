#!/usr/bin/env python3
"""N18 A0: nsys on N17 env stack (information only).

Kill proposed micro-opts whose max kernel <200 ms of e2e.
Do NOT treat nsys as a substitute for a GPU timed gate.

Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from n13_baseline import parks_env  # noqa: E402
from n17_gate import DEEP  # noqa: E402
from n18_dead import refuse_or_ok  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from b_dev_aff import build  # noqa: E402
from e6r_parhac import compile_d  # noqa: E402

NSYS = "nsys"
VAL_REP = Path("/tmp/n18_n17_val")
BIG_REP = Path("/tmp/n18_n17_216")
OUT = CACHE / "n18_nsys.json"
NOTE = ROOT / "notes/N18_A0_NSYS.md"
CLAIM = "not a 2 Gvox/s number; not 3090 Ti"


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
    for report in ("cuda_gpu_kern_sum", "nvtx_sum"):
        p = subprocess.run(
            [NSYS, "stats", "--report", report, str(src)],
            capture_output=True, text=True, check=False,
        )
        chunks.append(f"## {report}\n{p.stdout}\n{p.stderr}")
    return "\n".join(chunks)


def top_kernels(blob: str, n=30):
    rows = []
    for ln in blob.splitlines():
        if "k_" in ln or "cub::" in ln or "Device" in ln or "parhac" in ln.lower():
            rows.append(ln.strip())
    return rows[:n]


def main():
    if "--force" not in sys.argv:
        msg = refuse_or_ok("N18_A0_NSYS", force=False)
        # nsys baseline is always allowed even if stamped later
        _ = msg

    print(f"N18 A0 nsys N17 stack. {CLAIM}.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N18 A0 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)

    extra = {**DEEP, "WATERZ_EMIT_HOLES": "1", "WATERZ_NVTX": "1"}
    build()
    compile_d()
    env = parks_env(extra)

    # Val e2e via n12_nsys --val-e2e pattern: reuse n8 is heavy; profile val segment
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
    kill_thresh_ms = 200.0
    lines = [
        "# N18 A0 nsys (N17 env stack)",
        "",
        f"{CLAIM}. Parks off. Information only: not a timed gate substitute.",
        "",
        f"- gpu: `{gpu_state()}`",
        f"- nsys val rc={rc_val} 2.16 rc={rc_216}",
        f"- 2.16 stages={stages.get('stages')} e2e={e2e}",
        f"- micro-opt kill rule: max kernel <{kill_thresh_ms} ms of e2e -> skip as closer",
        "",
        "## 2.16 top kernel lines",
        "",
    ]
    tops = top_kernels(big_stats)
    lines += [f"- `{x}`" for x in tops]
    lines += [
        "",
        "## val dump (trim)",
        "",
        "```",
        val_stats[:12000],
        "```",
        "",
        "## 2.16 dump (trim)",
        "",
        "```",
        big_stats[:20000],
        "```",
        "",
    ]
    NOTE.write_text("\n".join(lines))
    doc = {
        "claim": CLAIM,
        "gpu": gpu_state(),
        "rc_val": rc_val,
        "rc_216": rc_216,
        "run216": stages,
        "val_top": top_kernels(val_stats),
        "big_top": tops,
        "kill_thresh_ms": kill_thresh_ms,
        "env": extra,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"N18 A0 nsys -> {OUT} {NOTE}", flush=True)
    return 0 if rc_216 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
