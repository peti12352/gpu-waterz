#!/usr/bin/env python3
"""N18 experiment runner: A1–A5, B0–B3, C scaffolding.

Not a 2 Gvox/s claim. Not 3090 Ti. Parks off. card_busy refuse.
Subprocess per env. Abort timers via n18_voi_gate.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from n17_gate import DEEP  # noqa: E402
from n18_dead import refuse_or_ok, stamp  # noqa: E402
from n18_voi_gate import CLAIM, run_gate  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402

NOTE_DIR = ROOT / "notes"
LOG = NOTE_DIR / "LOG.md"


def note(path: Path, body: str):
    path.write_text(body if body.endswith("\n") else body + "\n")


def log_block(title: str, lines: list[str]):
    block = f"\n## {title}\n" + "\n".join(lines) + "\n"
    with LOG.open("a") as f:
        f.write(block)


def gate(name, extra, four_tag, mode="voi_only", run_216=True):
    os.environ["WATERZ_N18_EXP"] = name
    msg = refuse_or_ok(name, force="--force" in sys.argv)
    if msg:
        print(msg, flush=True)
        return 3, {"refuse": msg}
    base = {**DEEP, "WATERZ_EMIT_HOLES": "1"}
    base.update(extra)
    return run_gate(name, base, four_tag, mode=mode, run_216=run_216)


def write_result(name: str, rc: int, doc: dict, md_extra: str = ""):
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{name}.json").write_text(json.dumps(doc, indent=2) + "\n")
    ws = doc.get("ws_ms") or 0
    agg = doc.get("agg_ms") or 0
    e2e = doc.get("e2e_ms") or 0
    body = (
        f"# {name}\n\n"
        f"{CLAIM}. Parks off.\n\n"
        f"- rc={rc} keep_default={doc.get('keep_default')} hang={doc.get('hang')}\n"
        f"- mode={doc.get('mode')} four_pass={((doc.get('four') or {}).get('four_pass'))}\n"
        f"- voi_ok={((doc.get('voi') or {}).get('ok'))} "
        f"ident_diag_eq={((doc.get('ident_diag') or {}).get('array_equal') or (doc.get('ident_diag') or {}).get('identity'))}\n"
        f"- ws_ms={ws} agg_ms={agg} e2e_ms={e2e} ws_le_900={doc.get('ws_le_900')}\n"
        f"- env={doc.get('env')}\n"
        f"{md_extra}\n"
    )
    note(NOTE_DIR / f"{name}.md", body)
    log_block(name, [
        f"claim: {CLAIM}",
        f"rc={rc} keep={doc.get('keep_default')} hang={doc.get('hang')}",
        f"ws={ws} agg={agg} e2e={e2e}",
        f"four={((doc.get('four') or {}).get('four_pass'))}",
    ])
    return rc, doc


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"N18 runner stage={stage}. {CLAIM}.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N18 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU: {gpu_state()}", flush=True)

    results = {}

    if stage in ("all", "a1"):
        # A1: prove voi_only on N17 stack; identity diagnostic only
        rc, doc = gate("N18_A1", {}, "n18_a1", mode="voi_only", run_216=True)
        write_result("N18_A1", rc, doc, md_extra="- unlock: harness voi_only on N17 DEEP+EMIT_HOLES\n")
        results["A1"] = {"rc": rc, "ws": doc.get("ws_ms"), "four": (doc.get("four") or {}).get("four_pass")}
        if rc == 2:
            return 2

    ws_stop = False
    if stage in ("all", "a2", "a"):
        rc, doc = gate(
            "N18_A2", {"WATERZ_VCOUNT_COMPACT": "1"}, "n18_a2", mode="voi_only")
        cut = (1315.0 / doc["ws_ms"]) if doc.get("ws_ms") else 0
        kill = False
        reason = ""
        if doc.get("hang") or not (doc.get("voi") or {}).get("ok"):
            kill, reason = True, "voi/hang fail"
        elif not (doc.get("four") or {}).get("four_pass"):
            kill, reason = True, "four-T FAIL"
        elif not doc.get("ws_ms"):
            kill, reason = True, "no 2.16 ws"
        elif cut < 1.05:
            kill, reason = True, f"WS cut {cut:.3f}x <1.05 vs 1315"
        write_result("N18_A2", rc, doc, md_extra=f"- cut_vs_1315={cut:.3f} kill={kill} {reason}\n")
        if kill:
            stamp("N18_A2", reason, {"ws_ms": doc.get("ws_ms"), "cut": cut}, "notes/N18_A2.md")
        results["A2"] = {"rc": rc, "ws": doc.get("ws_ms"), "kill": kill}
        if rc == 2:
            return 2

    if stage in ("all", "a3", "a"):
        rc, doc = gate(
            "N18_A3", {"WATERZ_TIE_FLIP": "1"}, "n18_a3", mode="voi_only")
        kill = (not (doc.get("four") or {}).get("four_pass")
                or not doc.get("ws_ms")
                or (doc.get("ws_ms") or 9999) > 1100)
        reason = "four-T FAIL or WS>1100 or no 2.16"
        write_result("N18_A3", rc, doc, md_extra=f"- kill={kill} {reason}\n")
        if kill:
            stamp("N18_A3", reason, {"ws_ms": doc.get("ws_ms")}, "notes/N18_A3.md")
        results["A3"] = {"rc": rc, "ws": doc.get("ws_ms"), "kill": kill}
        if rc == 2:
            return 2

    if stage in ("all", "a4", "a"):
        rc, doc = gate(
            "N18_A4",
            {"WATERZ_COARSE_DELTA": "8"},
            "n18_a4", mode="voi_only")
        kill = not (doc.get("four") or {}).get("four_pass")
        reason = "four-T FAIL"
        write_result("N18_A4", rc, doc, md_extra=f"- coarse_delta=8 kill={kill}\n")
        if kill:
            stamp("N18_A4", reason, {"ws_ms": doc.get("ws_ms")}, "notes/N18_A4.md")
        results["A4"] = {"rc": rc, "ws": doc.get("ws_ms"), "kill": kill}
        if rc == 2:
            return 2

    if stage in ("all", "a5", "a"):
        rc, doc = gate(
            "N18_A5", {"WATERZ_BLOCK_VOI": "1"}, "n18_a5", mode="voi_only")
        kill = (doc.get("hang") or not (doc.get("four") or {}).get("four_pass"))
        reason = "four-T FAIL / hang / OOM"
        write_result("N18_A5", rc, doc, md_extra=f"- block_voi skip face clear; kill={kill}\n")
        if kill:
            stamp("N18_A5", reason, {"ws_ms": doc.get("ws_ms"), "hang": doc.get("hang")},
                  "notes/N18_A5.md")
        results["A5"] = {"rc": rc, "ws": doc.get("ws_ms"), "kill": kill}
        if rc == 2:
            return 2

    # A-track stop rule
    if stage in ("all", "a"):
        ws_vals = [
            (results[k].get("ws") or 9e9)
            for k in ("A2", "A3", "A4", "A5") if k in results and not results[k].get("kill")
        ]
        best_ws = min(ws_vals) if ws_vals else 9e9
        if best_ws > 900:
            ws_stop = True
            note(NOTE_DIR / "N18_A_STOP.md", (
                f"# N18 A-track stop\n\n{CLAIM}.\n\n"
                f"After A2–A5 no experiment achieved WS≤900 with four-T PASS. "
                f"best_legal_ws={best_ws}. Freeze WS; dump EV into Track B.\n"
            ))
            log_block("N18_A_STOP", [f"best_legal_ws={best_ws} freeze WS track"])

    if stage in ("all", "b0", "b"):
        # B0: use existing nsys script / parse; if missing run n18_nsys
        out = CACHE / "n18_nsys.json"
        if not out.is_file():
            r = subprocess.run(
                [sys.executable, "-u", str(ROOT / "scripts/n18_nsys.py")],
                cwd=str(ROOT),
            )
            results["B0"] = {"rc": r.returncode}
        else:
            results["B0"] = {"rc": 0, "cached": True}
        note(NOTE_DIR / "N18_B0.md", (
            f"# N18 B0 nsys agg owners\n\n{CLAIM}.\n\n"
            f"See notes/N18_A0_NSYS.md / data/cache/n18_nsys.json.\n"
        ))

    if stage in ("all", "b1", "b"):
        r = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n18_b1_binq.py")],
            cwd=str(ROOT),
        )
        results["B1"] = {"rc": r.returncode}

    if stage in ("all", "b2", "b"):
        r = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n18_b2_eps.py")],
            cwd=str(ROOT),
        )
        results["B2"] = {"rc": r.returncode}

    if stage in ("all", "b3", "b"):
        r = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n18_b3_inner.py")],
            cwd=str(ROOT),
        )
        results["B3"] = {"rc": r.returncode}

    if stage in ("all", "c"):
        r = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n18_3090_grade.py")],
            cwd=str(ROOT),
        )
        results["C"] = {"rc": r.returncode}

    (CACHE / "n18_results.json").write_text(json.dumps({
        "claim": CLAIM, "results": results, "ws_stop": ws_stop, "gpu": gpu_state(),
    }, indent=2) + "\n")
    print(json.dumps({"claim": CLAIM, "results": results}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
