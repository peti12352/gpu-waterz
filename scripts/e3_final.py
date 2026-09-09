#!/usr/bin/env python3
"""E3: run every gate and report, in one command.

Written so that E1 - putting this on a rented 3090 Ti - is one invocation and
not an afternoon of remembering which script gates what. It sequences the
existing gates rather than reimplementing them, because a gate that lives in
two places drifts.

Refuses to report a speed number when another process holds memory on the card.
That is not caution, it is the lesson: five runs of one fixed workload on the
shared development card spread 1633.9 to 5018.4 ms.

  --quick     correctness only, crop-scale, safe on a shared card
  --full      adds the val-scale gates and the benches; needs the card to itself
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_VENV = ROOT / ".venv/bin/python"
PY = _VENV if _VENV.is_file() else Path(sys.executable)
CACHE = ROOT / "data/cache"

# name, argv, needs_idle_card, what it proves
GATES = [
    ("d3_device_path", ["scripts/d3_dev_check.py"], False,
     "device-resident path identical to host, all four thresholds"),
    ("d1_hslot_narrow", ["scripts/d1_narrow_gate.py"], False,
     "narrowed dedup slot identical to the wide reference"),
    ("d2_buffer_share", ["scripts/d2_share_gate.py"], False,
     "watershed borrowing the label buffer identical to its own"),
    ("d1_memory", ["scripts/d1_mem.py"], False,
     "measured per-stage peaks and the 2.16 Gvox extrapolation"),
    ("voi_four_threshold", ["scripts/a1_e6t_voi.py", "--e6s"], False,
     "four-threshold VOI PASS and the locked partition fingerprint"),
    ("rag_determinism", ["scripts/b1_rag_determinism.py"], False,
     "RAG edge set deterministic and equal to the CPU oracle"),
    ("ws_invariants", ["scripts/c2_ws_invariants.py"], True,
     "val watershed vs CPU oracle, nfrag and size-histogram fingerprint"),
    ("bench_host", ["scripts/d_bench.py"], True,
     "median-of-5 through segment(), host round trip included"),
    ("bench_device", ["scripts/d_bench.py", "--device"], True,
     "median-of-5 through segment_d(), CUDA events, the graded shape"),
]


def card_state():
    """('idle', []), ('busy', lines), or ('none', reason).

    Missing nvidia-smi is not a busy card. The previous form returned
    'nvidia-smi unavailable' from card_busy() and --gpu-window treated
    that as refuse-the-whole-window, so the one command that is supposed
    to be the idle-card entry point refused on a machine with no driver.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as e:
        return "none", [f"nvidia-smi unavailable: {e}"]
    if r.returncode != 0:
        return "none", [f"nvidia-smi rc={r.returncode}"]
    me = str(os.getpid())
    busy = [ln for ln in r.stdout.splitlines()
            if ln.strip() and ln.split(",")[0].strip() != me]
    return ("busy", busy) if busy else ("idle", [])


def card_busy():
    kind, lines = card_state()
    return lines if kind != "idle" else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="include val-scale gates and the benches")
    ap.add_argument("--gpu-window", action="store_true",
                    help="refuse if the card is busy, then run nvcheck, "
                         "g_levers, w_levers, determinism, VOI, d_bench")
    args = ap.parse_args()

    if args.gpu_window:
        kind, info = card_state()
        if kind == "busy":
            print("E3 GPU-WINDOW refused; card busy:\n" + "\n".join(info),
                  flush=True)
            (CACHE / "e3_gpu_window.json").write_text(
                json.dumps({"status": "refused", "card_busy": info},
                           indent=2) + "\n")
            return False
        cpu = [
            ("nvcheck", ["scripts/nvcheck.py"],
             "clang syntax of parhac_d.cu, ws.cu, rag.cu", None),
            ("w_levers", ["scripts/w_levers.py"],
             "WATERZ_UF_ALGO 0/1/2/3 and W3 CPU identity", None),
            ("r1_rag_tile", ["scripts/r1_rag_tile.py"],
             "R1 tiled RAG edge-set identity", None),
        ]
        # Full-val V sweep is ~40 min of host work. Do it when there is no
        # card; do not sit on an idle GPU doing it.
        if kind == "none":
            vj = CACHE / "v_levers.json"
            have_v = False
            if vj.is_file():
                try:
                    have_v = json.loads(vj.read_text()).get("sub", 1) == 0
                except (OSError, json.JSONDecodeError):
                    have_v = False
            if have_v:
                print("E3 skip v_levers; data/cache/v_levers.json is full-val",
                      flush=True)
            else:
                cpu.append(("v_levers", ["scripts/v_levers.py"],
                            "V1/V2 full-val work factors", None))
        gpu = [
            ("g_levers", ["scripts/g_levers.py"],
             "WATERZ_AGG_LEVERS bits 1/2/4/8 bit-identity + phase timing",
             None),
            ("rag_determinism", ["scripts/b1_rag_determinism.py"],
             "RAG edge set deterministic", None),
            ("ws_invariants", ["scripts/c2_ws_invariants.py"],
             "val watershed vs CPU oracle", None),
            ("voi_four_threshold", ["scripts/a1_e6t_voi.py", "--e6s"],
             "four-threshold VOI PASS at locked eps=0.08", None),
            ("v1_voi", ["scripts/a1_e6t_voi.py", "--e6s"],
             "V1 WATERZ_AGG_EPS=0.16 four-threshold VOI",
             {"WATERZ_AGG_EPS": "0.16"}),
            ("v2_voi", ["scripts/a1_e6t_voi.py", "--e6s"],
             "V2 WATERZ_SIZE_ASYM=0 four-threshold VOI",
             {"WATERZ_SIZE_ASYM": "0"}),
            ("bench_device", ["scripts/d_bench.py", "--device"],
             "median-of-5 segment_d(), the graded shape", None),
        ]
        window = cpu if kind == "none" else cpu + gpu
        if kind == "none":
            print("E3 GPU-WINDOW no device; CPU gates only:\n"
                  + "\n".join(info), flush=True)
        results = {}
        for name, argv, claim, extra_env in window:
            print(f"\nE3 === {name}: {claim}", flush=True)
            t0 = time.perf_counter()
            env = os.environ.copy()
            if extra_env:
                env.update(extra_env)
            rc = subprocess.run([str(PY), *argv], cwd=str(ROOT),
                                env=env).returncode
            dt = time.perf_counter() - t0
            results[name] = {"status": "PASS" if rc == 0 else "FAIL",
                             "rc": rc, "seconds": round(dt, 1),
                             "claim": claim}
            print(f"E3 {name} {results[name]['status']} in {dt:.1f}s",
                  flush=True)
        failed = [n for n, r in results.items() if r["status"] == "FAIL"]
        blocked = [n for n, _, _, _ in gpu] if kind == "none" else []
        (CACHE / "e3_gpu_window.json").write_text(
            json.dumps({"status": "cpu_only" if kind == "none" else "ran",
                        "card": kind, "results": results,
                        "failed": failed, "gpu_blocked": blocked},
                       indent=2) + "\n")
        print(f"E3 gpu-window {len(results) - len(failed)}/{len(results)} "
              f"passed; gpu_blocked={blocked}", flush=True)
        return not failed

    busy = card_busy()
    print(f"E3 card busy: {busy if busy else 'no, idle'}", flush=True)
    if args.full and busy:
        print("E3 WARNING --full on a shared card: the val gates may OOM a "
              "co-tenant and the benches will not be gradeable", flush=True)

    results = {}
    for name, argv, needs_idle, claim in GATES:
        if needs_idle and not args.full:
            print(f"E3 SKIP {name} (needs --full)", flush=True)
            results[name] = {"status": "skipped", "claim": claim}
            continue
        print(f"\nE3 === {name}: {claim}", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.run([str(PY), *argv], cwd=str(ROOT)).returncode
        dt = time.perf_counter() - t0
        results[name] = {
            "status": "PASS" if rc == 0 else "FAIL",
            "rc": rc, "seconds": round(dt, 1), "claim": claim,
        }
        print(f"E3 {name} {'PASS' if rc == 0 else 'FAIL'} in {dt:.1f}s",
              flush=True)

    print("\nE3 summary")
    for name, r in results.items():
        print(f"E3   {r['status']:8s} {name}", flush=True)

    ran = [r for r in results.values() if r["status"] != "skipped"]
    failed = [n for n, r in results.items() if r["status"] == "FAIL"]
    gradeable = bool(args.full and not busy and not card_busy())
    print(f"E3 {len(ran) - len(failed)}/{len(ran)} passed; "
          f"speed numbers gradeable: {gradeable}", flush=True)
    if not gradeable:
        print("E3 no speed claim is made from this run", flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "e3_final.json").write_text(
        json.dumps({"results": results, "gradeable": gradeable,
                    "card_busy": busy}, indent=2) + "\n")
    return not failed


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
