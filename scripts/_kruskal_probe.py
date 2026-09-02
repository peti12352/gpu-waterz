"""Run one Kruskal predicate config, grade, optionally batched regrade."""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import CACHE, grade_parents, run_kruskal, stamp  # noqa: E402
from task_gate import BASE_MERGE, BASE_SPLIT, SLACK, print_contract  # noqa: E402


def parse_voi(text: str):
    rows = {}
    for m in re.finditer(
        r"aff_thr=(0\.[2345])\s+split\s+([0-9.]+).*?\bmerge\s+([0-9.]+)",
        text,
    ):
        rows[float(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    if rows:
        return rows
    for m in re.finditer(
        r"(0\.[2345])\s+split\s+([0-9.]+)\s+merge\s+([0-9.]+)",
        text,
    ):
        rows[float(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return rows


def kill_first_t(text: str) -> bool:
    rows = parse_voi(text)
    if 0.2 not in rows:
        return False
    s, m = rows[0.2]
    return (s > BASE_SPLIT[0.2] + SLACK + 0.05) and (m > BASE_MERGE[0.2] + SLACK + 0.05)


def probe(name, pred, n_bins, p0, p1, dest):
    print_contract()
    print(f"{name} pred={pred} bins={n_bins} p0={p0} p1={p1}", flush=True)
    t0 = time.time()
    rc, parents, fr, stats, _ = run_kruskal(pred, n_bins, p0, p1)
    wall = time.time() - t0
    rounds = int(stats[:, 0].max())
    print(f"{name} rc={rc} wall={wall:.3f} rounds={rounds}", flush=True)
    if rc != 1:
        stamp(name, False, f"rc={rc}")
        return False, False, rounds, wall
    import io
    from contextlib import redirect_stdout, redirect_stderr

    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = grade_parents(parents, fr, name, dest)
    text = buf.getvalue()
    sys.stdout.write(text)
    stamp(name, ok, f"rounds={rounds} bins={n_bins} p0={p0} p1={p1} wall={wall:.3f}")
    return ok, kill_first_t(text), rounds, wall


def probe_and_batch(name, pred, p0, p1, dest_base):
    ok, kill, rounds, wall = probe(name, pred, 1, p0, p1, dest_base)
    if kill:
        print(f"{name} KILL first-T both-halves miss >0.05", flush=True)
        return False
    if not ok:
        print(f"{name} serial FAIL", flush=True)
        return False
    locked = False
    for B in (8, 16, 32):
        bok, _, br, bw = probe(f"{name}_b{B}", pred, B, p0, p1, f"{dest_base}_b{B}")
        if bok and br <= 32:
            stamp(name, True, f"LOCK batched B={B} rounds={br} wall={bw:.3f}")
            locked = True
            break
        print(f"{name} batched B={B} {'PASS-serial-class' if bok else 'FAIL'}", flush=True)
    if not locked:
        stamp(name, False, f"PASS-serial / FAIL-batched wall={wall:.3f}")
        print(f"{name} PASS-serial / FAIL-batched (Q20-class, not a lock)", flush=True)
    return locked
