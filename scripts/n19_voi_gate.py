#!/usr/bin/env python3
"""N19 VOI-first gate. Modes: voi_only (default), val_ws, ident.

Order: card_busy -> build -> (val_ws | VOI 2-run -> four-T -> optional 2.16).
Identity diagnostic only in voi_only. Unlink n8_216.json before 2.16.
WATERZ_ABORT_SEC (60 val_ws / 120 val / 600 2.16).

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
from n13_baseline import load_json_line, parks_env  # noqa: E402
from n15_agg_gate import AGG_BASE, AGG_GATE  # noqa: E402
from n15_gate import WS_BASE, WS_GATE  # noqa: E402
from n17_gate import DEEP  # noqa: E402
from n19_dead import refuse_or_ok, stamp  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from b_dev_aff import build  # noqa: E402
from e6r_parhac import compile_d  # noqa: E402

CLAIM = "not a 2 Gvox/s number; not 3090 Ti"
N18_WS = 1315.0
N18_AGG = 1683.0
N18_E2E = 3104.0


def abort_sec(kind: str) -> int:
    env = os.environ.get("WATERZ_ABORT_SEC")
    if env is not None and env.strip() != "":
        return int(env)
    if kind == "216":
        return 600
    if kind == "val_ws":
        return 60
    return 120


def run_cap(argv, env, timeout, label: str):
    try:
        r = subprocess.run(
            argv, cwd=str(ROOT), env=env, capture_output=True, text=True,
            timeout=timeout,
        )
        doc = load_json_line(r.stdout)
        if not doc and r.stdout.strip():
            doc = {"stdout_tail": r.stdout[-2000:], "stderr_tail": r.stderr[-2000:]}
        return doc, False, r.returncode
    except subprocess.TimeoutExpired as e:
        stamp(
            os.environ.get("WATERZ_N19_EXP", "unknown"),
            "hang",
            {"label": label, "timeout_sec": timeout},
            note="n19_voi_gate",
        )
        out = e.stdout or b""
        err = e.stderr or b""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return {
            "hang": True, "label": label, "timeout_sec": timeout,
            "stdout_tail": out[-2000:], "stderr_tail": err[-2000:],
        }, True, -9


def run_val_ws(extra: dict):
    """Timed val WS + identity diagnostic. No agg/four/2.16."""
    script = ROOT / "scripts/n19_val_ws.py"
    env = parks_env(extra)
    return run_cap(
        [sys.executable, str(script)], env, abort_sec("val_ws"), "val_ws",
    )


def run_gate(
    name: str,
    extra: dict,
    four_tag: str,
    mode: str = "voi_only",
    run_216: bool = True,
    need_build: bool = True,
):
    print(
        f"N19 {name} mode={mode}. {CLAIM}. Parks off. card_busy refuse.",
        flush=True,
    )
    busy = card_busy()
    if busy:
        print(f"N19 {name} REFUSE card busy: {busy}", flush=True)
        return 2, {"refuse": "card_busy", "busy": busy, "claim": CLAIM}

    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ.update(extra)
    compile_d()
    if need_build:
        build()

    env = parks_env(extra)
    hang = False

    if mode == "val_ws":
        val_ws, h, rc = run_val_ws(extra)
        hang = hang or h
        print(f"N19 {name} val_ws {val_ws}", flush=True)
        doc = {
            "claim": CLAIM, "mode": mode, "name": name,
            "val_ws": val_ws, "hang": hang, "env": extra, "gpu": gpu_state(),
            "keep_default": False,
        }
        return (0 if (not hang and val_ws.get("ok")) else 1), doc

    ident, ident_ok = {}, True
    if mode == "ident" or os.environ.get("WATERZ_N19_IDENT_DIAG", "1") == "1":
        ident, h, _ = run_cap(
            [sys.executable, str(ROOT / "scripts/n15_gate.py")],
            env, abort_sec("val"), "ident",
        )
        hang = hang or h
        print(f"N19 {name} ident_diag {ident}", flush=True)
        if mode == "ident":
            ident_ok = bool(
                not h and ident.get("identity") and ident.get("run2_array_equal")
            )
        else:
            ident_ok = not h

    voi, voi_ok = {}, False
    if ident_ok and not hang:
        voi, h, _ = run_cap(
            [sys.executable, str(ROOT / "scripts/n15_agg_gate.py")],
            env, abort_sec("val"), "voi",
        )
        hang = hang or h
        print(f"N19 {name} voi {voi}", flush=True)
        voi_ok = bool(not h and voi.get("ok") and voi.get("run2_array_equal"))

    four, four_ok = {}, False
    if voi_ok and not hang:
        four, h, _ = run_cap(
            [sys.executable, str(ROOT / "scripts/n15_four.py")],
            parks_env({**extra, "WATERZ_FOUR_TAG": four_tag, "WATERZ_SKIP_BUILD": "1"}),
            abort_sec("val"), "four",
        )
        hang = hang or h
        print(f"N19 {name} four {four}", flush=True)
        four_ok = bool(not h and four.get("four_pass"))

    run216, agg, ws, e2e, rc216 = {}, 0.0, 0.0, 0.0, None
    if run_216 and four_ok and not hang:
        p216 = CACHE / "n8_216.json"
        if p216.is_file():
            p216.unlink()
        try:
            r = subprocess.run(
                [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
                cwd=str(ROOT),
                env=parks_env({**extra, "WATERZ_SKIP_BUILD": "1"}),
                timeout=abort_sec("216"),
            )
            rc216 = r.returncode
        except subprocess.TimeoutExpired:
            hang = True
            rc216 = -9
            stamp(
                os.environ.get("WATERZ_N19_EXP", name),
                "hang",
                {"label": "216", "timeout_sec": abort_sec("216")},
                note="n19_voi_gate",
            )
            run216 = {"hang": True}
        if p216.is_file() and not hang and rc216 == 0:
            run216 = json.loads(p216.read_text())
        elif rc216 not in (0, None) and not hang:
            run216 = {"failed": True, "rc216": rc216}
        st = run216.get("stages") or {}
        agg = float(st.get("agg") or 0)
        ws = float(st.get("ws") or 0)
        e2e = float(run216.get("e2e_ms") or 0)

    cut_agg = (AGG_BASE / agg) if agg else 0.0
    cut_ws = (WS_BASE / ws) if ws else 0.0
    keep = bool(
        not hang and voi_ok and four_ok and rc216 == 0
        and (not run_216 or (agg > 0 and agg <= AGG_GATE
                             and ws > 0 and ws <= WS_GATE))
    )
    if mode == "ident":
        keep = keep and bool(ident.get("identity") and ident.get("run2_array_equal"))

    doc = {
        "claim": CLAIM, "mode": mode, "name": name,
        "ident_diag": ident, "ident_blocking": mode == "ident",
        "voi": voi, "four": four, "run216": run216,
        "agg_ms": agg, "ws_ms": ws, "e2e_ms": e2e,
        "cut_agg": cut_agg, "cut_ws": cut_ws,
        "agg_gate": AGG_GATE, "ws_gate": WS_GATE,
        "n18_ws": N18_WS, "n18_agg": N18_AGG, "n18_e2e": N18_E2E,
        "ws_le_900": bool(ws > 0 and ws <= 900.0),
        "e2e_vs_n18": (N18_E2E / e2e) if e2e else 0.0,
        "hang": hang, "keep_default": keep, "rc216": rc216,
        "env": extra, "gpu": gpu_state(),
    }
    return (0 if (voi_ok and four_ok and not hang) else 1), doc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("voi_only", "ident", "val_ws"),
                    default="voi_only")
    ap.add_argument("--name", default="n19_gate")
    ap.add_argument("--four-tag", default="n19_four")
    ap.add_argument("--exp-id", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-216", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--extra-json", default="")
    args = ap.parse_args()

    exp_id = args.exp_id or args.name
    os.environ["WATERZ_N19_EXP"] = exp_id
    msg = refuse_or_ok(exp_id, force=args.force)
    if msg:
        print(msg, flush=True)
        return 3

    extra = dict(DEEP)
    extra["WATERZ_EMIT_HOLES"] = "1"
    if args.extra_json:
        extra.update(json.loads(args.extra_json))

    rc, doc = run_gate(
        args.name, extra, args.four_tag, mode=args.mode,
        run_216=not args.no_216 and args.mode != "val_ws",
    )
    out = Path(args.out) if args.out else CACHE / f"{args.name}.json"
    CACHE.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({
        "rc": rc, "out": str(out), "keep_default": doc.get("keep_default"),
        "ws_ms": doc.get("ws_ms"), "agg_ms": doc.get("agg_ms"),
        "e2e_ms": doc.get("e2e_ms"), "hang": doc.get("hang"), "claim": CLAIM,
    }), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
