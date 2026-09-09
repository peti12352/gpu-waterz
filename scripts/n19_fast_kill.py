#!/usr/bin/env python3
"""N19 single-exp fast kill wrapper.

dead-check → free_gpu → gate → notes/LOG/dead. Claim string mandatory.
Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from n17_gate import DEEP  # noqa: E402
from n19_dead import refuse_or_ok, stamp  # noqa: E402
from n19_voi_gate import CLAIM, run_gate  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402

NOTE_DIR = ROOT / "notes"
LOG = NOTE_DIR / "LOG.md"
OWNERS = CACHE / "n19_owners.json"
KILL_THRESH = 200.0


def free_gpu():
    script = ROOT / "scripts/n18_free_gpu.sh"
    if script.is_file():
        subprocess.run(["bash", str(script)], check=False)


def owner_ok(substr: str) -> tuple[bool, dict]:
    if not OWNERS.is_file():
        return True, {"warn": "no n19_owners.json; allowing"}
    owners = json.loads(OWNERS.read_text())
    hits = [k for k in (owners.get("kernels") or []) if substr in k.get("name", "")]
    # also nvtx gates
    gates = owners.get("gates") or {}
    for gname, g in gates.items():
        if substr.lower() in gname.lower() and g.get("ok_to_try"):
            return True, g
    max_ms = max((k["total_ms"] for k in hits), default=0.0)
    # NVTX fallback
    for n in owners.get("nvtx") or []:
        if substr.lower() in n.get("name", "").lower():
            max_ms = max(max_ms, n["total_ms"])
    info = {"substr": substr, "max_ms": max_ms, "thresh": KILL_THRESH}
    return max_ms >= KILL_THRESH, info


def write_note(name: str, rc: int, doc: dict, extra: str = ""):
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    ws = doc.get("ws_ms") or 0
    agg = doc.get("agg_ms") or 0
    e2e = doc.get("e2e_ms") or 0
    body = (
        f"# {name}\n\n"
        f"{CLAIM}. Parks off.\n\n"
        f"- rc={rc} keep_default={doc.get('keep_default')} hang={doc.get('hang')}\n"
        f"- mode={doc.get('mode')} four={((doc.get('four') or {}).get('four_pass'))}\n"
        f"- voi={((doc.get('voi') or {}).get('ok'))}\n"
        f"- ws_ms={ws} agg_ms={agg} e2e_ms={e2e} ws_le_900={doc.get('ws_le_900')}\n"
        f"- env={doc.get('env')}\n"
        f"{extra}\n"
    )
    (NOTE_DIR / f"{name}.md").write_text(body)
    with LOG.open("a") as f:
        f.write(
            f"\n## {name}\nclaim: {CLAIM}\n"
            f"rc={rc} keep={doc.get('keep_default')} "
            f"ws={ws} agg={agg} e2e={e2e}\n"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--mode", default="voi_only",
                    choices=("voi_only", "val_ws", "ident"))
    ap.add_argument("--extra-json", default="{}")
    ap.add_argument("--owner-substr", default="",
                    help="If set, refuse micro-opt unless owner ≥200 ms")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-216", action="store_true")
    ap.add_argument("--kill-reason", default="")
    args = ap.parse_args()

    print(f"N19 fast_kill {args.exp_id}. {CLAIM}.", flush=True)
    msg = refuse_or_ok(args.exp_id, force=args.force)
    if msg:
        print(msg, flush=True)
        return 3

    if args.owner_substr:
        ok, info = owner_ok(args.owner_substr)
        print(f"N19 owner_gate {info}", flush=True)
        if not ok:
            stamp(args.exp_id, f"owner <{KILL_THRESH} ms", info, "n19_fast_kill")
            write_note(args.exp_id, 4, {"claim": CLAIM, "hang": False},
                       extra=f"- SKIP micro-opt owner_gate={info}\n")
            print(f"N19 SKIP {args.exp_id} owner too small", flush=True)
            return 4

    free_gpu()
    busy = card_busy()
    if busy:
        print(f"N19 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU: {gpu_state()}", flush=True)

    os.environ["WATERZ_N19_EXP"] = args.exp_id
    extra = {**DEEP, "WATERZ_EMIT_HOLES": "1"}
    extra.update(json.loads(args.extra_json))

    rc, doc = run_gate(
        args.exp_id, extra, f"n19_{args.exp_id.lower()}",
        mode=args.mode, run_216=not args.no_216 and args.mode != "val_ws",
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{args.exp_id}.json").write_text(json.dumps(doc, indent=2) + "\n")

    kill = False
    reason = args.kill_reason
    if doc.get("hang"):
        kill, reason = True, "hang"
    elif args.mode == "voi_only":
        if not (doc.get("four") or {}).get("four_pass"):
            kill, reason = True, reason or "four-T FAIL"
        elif doc.get("ws_ms") == 0 and not args.no_216:
            kill, reason = True, reason or "no 2.16"
    write_note(args.exp_id, rc, doc, extra=f"- kill={kill} reason={reason}\n")
    if kill:
        stamp(args.exp_id, reason or "kill", {
            "ws_ms": doc.get("ws_ms"), "agg_ms": doc.get("agg_ms"),
            "e2e_ms": doc.get("e2e_ms"),
        }, f"notes/{args.exp_id}.md")
    print(json.dumps({
        "exp_id": args.exp_id, "rc": rc, "kill": kill, "reason": reason,
        "ws_ms": doc.get("ws_ms"), "agg_ms": doc.get("agg_ms"),
        "e2e_ms": doc.get("e2e_ms"), "claim": CLAIM,
    }), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
