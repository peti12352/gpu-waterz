#!/usr/bin/env python3
"""N21 W2: list-only J-jump in k_w5_compress_list.

Ident is blocking. Not LIST_HALVING, JUMP_FLATTEN, PIN_CHANGED, Playne, E1.
No 2.16 unless ident+four-T PASS and val WS cut >=100 ms.
Not a 2 Gvox/s number; not 3090 Ti.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from n13_baseline import parks_env  # noqa: E402
from n15_agg_gate import AGG_BASE  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from b_dev_aff import build  # noqa: E402
from n20_res import attach  # noqa: E402
from n21_dead import stamp  # noqa: E402

CLAIM = "N21 W2 list jump J=4; not a throughput claim; not a 2 Gvox/s number; not 3090 Ti"
NOISE_MS = 50.0
CUT_216_MS = 100.0
J = 4


def legal_extra(jump: bool) -> dict:
    e = {
        "WATERZ_UF_ALGO": "3",
        "WATERZ_HOST_PARK": "0",
        "WATERZ_AFF_PARK": "0",
        "WATERZ_AGG_LEVERS": "15",
        "WATERZ_FOLD_FLATTEN": "1",
        "WATERZ_SHARE_OFF": "1",
        "WATERZ_HOOK_ROOT": "1",
        "WATERZ_FUSE_DIRTY": "1",
        "WATERZ_NLIVE_ARITH": "1",
        "WATERZ_EMIT_HOLES": "1",
        "WATERZ_LIST_JUMP": str(J) if jump else "0",
        "WATERZ_SKIP_BUILD": "1",
        "WATERZ_FOUR_TAG": "N21_W2",
    }
    return e


def run_json(script: str, extra: dict, timeout: int):
    env = parks_env(extra)
    for k in (
        "WATERZ_LIST_HALVING", "WATERZ_JUMP_FLATTEN", "WATERZ_PIN_CHANGED",
        "WATERZ_HASH_INSERT_ONLY", "WATERZ_FUSE_PACK", "WATERZ_N21_HOPS",
        "WATERZ_PLAYNE_HOOK",
    ):
        env.pop(k, None)
    r = subprocess.run(
        [sys.executable, "-u", str(ROOT / script)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout,
    )
    doc = {}
    if r.stdout.strip():
        try:
            doc = json.loads(r.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            doc = {"stdout_tail": r.stdout[-2000:]}
    doc["rc"] = r.returncode
    if r.returncode != 0:
        doc["stderr_tail"] = (r.stderr or "")[-2000:]
    return doc


def main():
    banned = [
        os.environ.get("WATERZ_LIST_HALVING"),
        os.environ.get("WATERZ_JUMP_FLATTEN"),
        os.environ.get("WATERZ_PIN_CHANGED"),
        os.environ.get("WATERZ_PLAYNE_HOOK"),
    ]
    if any(x and str(x) != "0" for x in banned):
        print("N21_W2 REFUSE dead WS lever in env", banned, flush=True)
        return 2
    busy = card_busy()
    if busy:
        print("N21_W2 REFUSE card_busy", busy, flush=True)
        return 2
    print("N21_W2 gpu", gpu_state(), flush=True)
    os.environ.update(legal_extra(True))
    os.environ.pop("WATERZ_SKIP_BUILD", None)
    build()

    ident = run_json("scripts/n15_gate.py", legal_extra(True), 180)
    ident_ok = bool(ident.get("identity") and ident.get("run2_array_equal"))
    ws_on = run_json("scripts/n19_val_ws.py", legal_extra(True), 180)
    ws_off = run_json("scripts/n19_val_ws.py", legal_extra(False), 180)
    ws_cut = None
    if ws_off.get("ws_ms") is not None and ws_on.get("ws_ms") is not None:
        ws_cut = float(ws_off["ws_ms"]) - float(ws_on["ws_ms"])

    doc = {
        "claim": CLAIM,
        "J": J,
        "LIST_HALVING": False,
        "JUMP_FLATTEN": False,
        "PIN_CHANGED": False,
        "ident": ident,
        "ident_ok": ident_ok,
        "ws_on": ws_on,
        "ws_off": ws_off,
        "val_ws_cut_ms": ws_cut,
        "ran_216": False,
        "keep_default": False,
    }

    four = {}
    voi = {}
    stamp_reason = None
    closer = None
    if not ident_ok:
        stamp_reason = "ident_fail"
        closer = False
        stamp("N21_W2", "ident_fail", {
            "nfrag": ident.get("nfrag"), "identity": ident.get("identity"),
        }, note="list jump J=4")
    else:
        voi = run_json("scripts/n15_agg_gate.py", legal_extra(True), 180)
        voi_ok = bool(voi.get("ok") and voi.get("run2_array_equal"))
        if not voi_ok:
            stamp_reason = "voi_fail"
            closer = False
            stamp("N21_W2", "voi_fail", {"voi": voi}, note="list jump J=4")
        else:
            four = run_json("scripts/n15_four.py", legal_extra(True), 300)
            four_ok = bool(four.get("four_pass"))
            if not four_ok:
                stamp_reason = "four_t_fail"
                closer = False
                stamp("N21_W2", "four_t_fail", {"four": four}, note="list jump J=4")
            elif ws_cut is None or ws_cut < NOISE_MS:
                stamp_reason = "noise_ws_cut"
                closer = False
                stamp("N21_W2", "noise_ws_cut", {
                    "val_ws_cut_ms": ws_cut,
                    "ws_on": ws_on.get("ws_ms"), "ws_off": ws_off.get("ws_ms"),
                }, note="ident+four PASS; freeze WS; nlist is the owner not hops")
            else:
                closer = True
                stamp_reason = "val_ws_cut_ge_noise"
                if ws_cut >= CUT_216_MS:
                    stamp_reason = "would_216"

    doc["voi"] = voi
    doc["four"] = four
    doc["closer"] = closer
    doc["stamp_reason"] = stamp_reason
    doc = attach(doc, "N21_W2")
    (CACHE / "N21_W2.json").write_text(json.dumps(doc, indent=2) + "\n")
    lines = [
        "# N21 W2 list-only jump J=4",
        "",
        CLAIM + ".",
        "",
        "- not LIST_HALVING, not JUMP_FLATTEN, not PIN_CHANGED, not Playne",
        f"- ident_ok={ident_ok} nfrag={ident.get('nfrag')} "
        f"array_equal={ident.get('array_equal')} run2={ident.get('run2_array_equal')}",
        f"- val ws on={ws_on.get('ws_ms')} off={ws_off.get('ws_ms')} cut={ws_cut}",
        f"- voi_ok={voi.get('ok')} four_pass={four.get('four_pass')}",
        f"- ran_216=False closer={closer} stamp={stamp_reason}",
        f"- timing_usable_vs_1679={doc.get('timing_usable_vs_1679')}",
        "- D0: hop_max=8-9; nlist 71-105M is the owner",
    ]
    (ROOT / "notes" / "N21_W2.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "ident_ok": ident_ok,
        "val_ws_cut_ms": ws_cut,
        "voi_ok": voi.get("ok"),
        "four_pass": four.get("four_pass"),
        "closer": closer,
        "stamp_reason": stamp_reason,
        "timing_usable_vs_1679": doc.get("timing_usable_vs_1679"),
        "ran_216": False,
    }, indent=2), flush=True)
    rc = 0 if ident_ok and (not voi or voi.get("ok")) and (
        not four or four.get("four_pass") or stamp_reason == "ident_fail"
    ) else 1
    if ident_ok and voi.get("ok") and four.get("four_pass"):
        return 0
    if stamp_reason == "ident_fail":
        return 1
    if stamp_reason in ("voi_fail", "four_t_fail"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
