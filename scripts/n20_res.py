#!/usr/bin/env python3
"""CPU resource gate for N20. Timing is invalid if another project is on core.

Correctness (parents, height, VOI, merge counts) does not need an idle box.
Wall-clock vs 1679.9 ms does. If this module says contaminated, redo timing.
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
SELF_MARKERS = (
    "n20_", "n21_", "n22_", "n23_", "librac_agg", "n20_cpu_run",
    "g0_agg_ref",
)
FOREIGN_MARKERS = (
    "leocad", "xvfb", "p1_real_comparison", "sft_train", "vllm",
    "sft_pilot", "generate_and_filter", "elm_mutation",
)
IGNORE_MARKERS = ("btop", "htop", " top", "/usr/bin/top", "gnome-shell", "Xorg")


def _loadavg():
    return tuple(float(x) for x in open("/proc/loadavg").read().split()[:3])


def _nproc():
    return os.cpu_count() or 1


def _swap():
    out = {"total": 0, "used": 0}
    for ln in open("/proc/meminfo"):
        if ln.startswith("SwapTotal:"):
            out["total"] = int(ln.split()[1])
        elif ln.startswith("SwapFree:"):
            out["free"] = int(ln.split()[1])
    out["used"] = out["total"] - out.get("free", 0)
    return out


def _ps():
    import subprocess
    r = subprocess.run(
        ["ps", "-eo", "pid,pcpu,pmem,etime,cmd", "--sort=-pcpu"],
        capture_output=True, text=True, check=False,
    )
    rows = []
    for ln in (r.stdout or "").splitlines()[1:]:
        parts = ln.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            rows.append({
                "pid": int(parts[0]),
                "pcpu": float(parts[1]),
                "pmem": float(parts[2]),
                "etime": parts[3],
                "cmd": parts[4],
            })
        except ValueError:
            continue
    return rows


def _is_self(cmd: str) -> bool:
    return any(m in cmd for m in SELF_MARKERS)


def _is_ignore(cmd: str) -> bool:
    c = cmd.lower()
    return any(m.lower() in c for m in IGNORE_MARKERS)


def _is_foreign(cmd: str) -> bool:
    c = cmd.lower()
    return any(m in c for m in FOREIGN_MARKERS)


def snapshot(tag: str = "") -> dict:
    rows = _ps()
    foreign = [
        r for r in rows
        if r["pcpu"] >= 5.0 and not _is_self(r["cmd"]) and not _is_ignore(r["cmd"])
    ]
    hogs = [r for r in foreign if _is_foreign(r["cmd"]) or r["pcpu"] >= 15.0]
    selfs = [r for r in rows if _is_self(r["cmd"])]
    load = _loadavg()
    swap = _swap()
    nproc = _nproc()
    self_cpu = sum(r["pcpu"] for r in selfs) / 100.0
    other_load = load[0] - min(self_cpu, load[0])
    # Decaying 1-minute loadavg after our own 100% heap is not a foreign hog.
    dirty = bool(hogs)
    doc = {
        "tag": tag,
        "host": socket.gethostname(),
        "ts": time.time(),
        "loadavg": load,
        "nproc": nproc,
        "swap_kib": swap,
        "self": selfs[:8],
        "foreign_ge5": foreign[:12],
        "hogs": hogs[:12],
        "other_load": other_load,
        "timing_contaminated": dirty,
        "reason": (
            "foreign CPU hog or extra loadavg"
            if dirty else "idle enough for single-thread wall"
        ),
    }
    return doc


def refuse_timing(tag: str = "") -> str | None:
    """None if wall-clock may be compared to 1679.9 ms. Else a reason."""
    s = snapshot(tag)
    if s["timing_contaminated"]:
        return (
            f"N20 REFUSE timing tag={tag} host={s['host']} "
            f"load={s['loadavg']} other_load={s['other_load']:.2f} "
            f"hogs={[(h['pid'], h['pcpu'], h['cmd'][:80]) for h in s['hogs']]}"
        )
    return None


def attach(doc: dict, tag: str) -> dict:
    s = snapshot(tag)
    doc = dict(doc)
    doc["res"] = s
    if s["timing_contaminated"]:
        doc["timing_contaminated"] = True
        doc["timing_usable_vs_1679"] = False
    else:
        doc["timing_contaminated"] = False
        doc["timing_usable_vs_1679"] = True
    return doc


def wait_idle(tag: str, timeout_sec: int = 3600, poll: float = 15.0) -> dict:
    t0 = time.time()
    last = snapshot(tag)
    while time.time() - t0 < timeout_sec:
        last = snapshot(tag)
        if not last["timing_contaminated"]:
            return last
        print(f"N20 wait idle: {refuse_timing(tag)}", flush=True)
        time.sleep(poll)
    last["wait_timeout"] = True
    return last


def dump(path, tag: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(snapshot(tag)) + "\n")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "snap"
    if cmd == "wait":
        print(json.dumps(wait_idle("cli"), indent=2))
    elif cmd == "jsonl":
        dump(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cli")
    else:
        print(json.dumps(snapshot("cli"), indent=2))
