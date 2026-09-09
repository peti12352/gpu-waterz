#!/usr/bin/env python3
"""N18 C: 3090 Ti grade path — median-of-5 e2e + peak.

Refuse unless nvidia-smi reports a real RTX 3090 Ti.
Do not flip C++ defaults without median≥claim and 24 GB peak fit.
Also writes 1.44 Gvox report (TASK also-report).

Not a 2 Gvox/s claim from 5090. Not runnable on greengoblin 5090.
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
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402

CLAIM = "not a 2 Gvox/s number; not 3090 Ti (this host)"
OUT = CACHE / "n18_3090_grade.json"
NOTE = ROOT / "notes/N18_C_3090.md"


def gpu_name() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        return (r.stdout or "").strip().splitlines()[0] if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def is_3090_ti(name: str) -> bool:
    n = name.lower().replace(" ", "")
    return "3090ti" in n or "3090 ti" in name.lower()


def main():
    name = gpu_name()
    print(f"N18 C grade. GPU={name!r}. {CLAIM}.", flush=True)
    doc = {
        "claim": "TASK grade requires real 3090 Ti; do not claim 2 Gvox/s from 5090",
        "gpu_name": name,
        "is_3090_ti": is_3090_ti(name),
        "flip_cpp_defaults": False,
        "reason": "",
    }
    if not is_3090_ti(name):
        doc["reason"] = (
            f"refuse: not RTX 3090 Ti (got {name!r}). "
            "Block until real 3090 Ti. No default flip."
        )
        print(f"N18 C REFUSE: {doc['reason']}", flush=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2) + "\n")
        NOTE.write_text(
            f"# N18 C 3090 Ti grade\n\n"
            f"- GPU: `{name}`\n"
            f"- REFUSE: script blocks until real 3090 Ti\n"
            f"- No C++ default flip without median≥2 Gvox/s AND peak≤24 GB\n"
            f"- Never claim 2 Gvox/s from 5090\n"
            f"- When on 3090 Ti: warmup + median/min/max of 5 CUDA-event e2e "
            f"on official 2.16 @ T=0.3; also report 1.44 Gvox; parks off; "
            f"full N17 env; ws_mem_peak\n"
        )
        return 2

    busy = card_busy()
    if busy:
        print(f"N18 C REFUSE card busy: {busy}", flush=True)
        return 2

    # Real 3090 Ti path: 5-run median
    extra = {**DEEP, "WATERZ_EMIT_HOLES": "1"}
    env = parks_env(extra)
    runs = []
    for i in range(5):
        r = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
            cwd=str(ROOT), env=env,
        )
        p = CACHE / "n8_216.json"
        if p.is_file():
            runs.append(json.loads(p.read_text()))
        print(f"N18 C run {i+1}/5 rc={r.returncode}", flush=True)

    e2es = sorted(float(x.get("e2e_ms") or 0) for x in runs)
    median = e2es[len(e2es) // 2] if e2es else 0
    gvox = (2.16 / (median / 1000.0)) if median else 0
    peaks = [float((x.get("ws_mem_peak") or x.get("peak_giB") or 0)) for x in runs]
    peak = max(peaks) if peaks else 0
    # SHARE_OFF headroom: +4 B/vox ≈ +8 GB at 2.16 — check peak + headroom ≤ 24
    share_off_headroom_gib = 8.0
    fits = (peak + share_off_headroom_gib) <= 24.0
    claim_ok = gvox >= 2.0 and fits
    doc.update({
        "runs": runs,
        "e2e_ms": {"min": min(e2es) if e2es else 0, "median": median,
                   "max": max(e2es) if e2es else 0},
        "gvox_s_median": gvox,
        "ws_mem_peak_gib": peak,
        "fits_24gb_with_share_off_headroom": fits,
        "flip_cpp_defaults": bool(claim_ok),
        "reason": "flip ok" if claim_ok else "median or peak gate failed",
        "also_report_144": "run make_big 2x2x2 / 1.44 when available",
        "gpu": gpu_state(),
    })
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        f"# N18 C 3090 Ti grade\n\n"
        f"- GPU: `{name}`\n"
        f"- median e2e={median:.1f} ms → {gvox:.3f} Gvox/s\n"
        f"- peak={peak:.2f} GiB fits24={fits} flip_defaults={claim_ok}\n"
    )
    print(json.dumps({"median_ms": median, "gvox_s": gvox, "flip": claim_ok}),
          flush=True)
    return 0 if claim_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
