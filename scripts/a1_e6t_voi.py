#!/usr/bin/env python3
"""A1: grade E6t/StarMerge on its own at the four TASK thresholds.

Track A tunes the StarMerge path, so its correctness has to be established
first, tuning a wrong implementation is worthless. This gate answers only
"is E6t VOI-legal", never "is E6t fast", and it writes no `*_pass.txt` stamp,
so `segment()` path selection is untouched.

E6t is selected by `WATERZ_PAPER_E6T` in csrc/parhac_d.cu; without it
`parhac_paper_d_timed` runs E6s. The reported inner/merge counts identify
which path actually ran (E6s reference: inner=1930 merges=1853410;
E6t reference: inner=1562 merges=1853416).

Any timing printed here is PROVISIONAL: the dev GPU is shared, so device_ms
is inflated by whatever else is resident. Correctness is contention-immune.
"""
from __future__ import annotations

import ctypes
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from _agg_common import CACHE, grade_parents, load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402
from task_gate import AFF_THRESHOLDS, print_contract  # noqa: E402

# Reference stats from notes/LOG.md, used to prove which path executed.
E6S_REF = {"inner": 1930, "merges": 1853410}
E6T_REF = {"inner": 1562, "merges": 1853416}


def gpu_state():
    """Record contention so any timing in the JSON is self-documenting."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return out


def bind(lib):
    lib.parhac_paper_d_timed.restype = ctypes.c_int
    lib.parhac_paper_d_timed.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double),
    ]


def _agg_eps(default=0.08):
    """Same override as src/segment.py. a1 used to hardcode 0.08, so
    e3's v1_voi env (WATERZ_AGG_EPS=0.16) never reached the kernel."""
    s = os.environ.get("WATERZ_AGG_EPS")
    return float(s) if s else default


def run(lib, u, v, sm, ct, thrs, max_id, eps):
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    device_ms = ctypes.c_double(0.0)
    t0 = time.perf_counter()
    rc = lib.parhac_paper_d_timed(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        ctypes.c_double(eps),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(device_ms),
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return rc, parents, stats, float(device_ms.value), wall_ms


def capture_stderr(fn):
    """Run fn with the C library's fd-2 output captured.

    parhac_d.cu announces its path as "E6s T=..." / "E6t T=..." on stderr.
    Reading that marker is the only airtight way to know which path executed:
    inferring it from inner/merge counts is unreliable, because in a
    four-threshold run the per-threshold counts are marginal and much smaller
    than the single-threshold reference figures.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    try:
        with tempfile.TemporaryFile(mode="w+b") as tf:
            os.dup2(tf.fileno(), 2)
            try:
                out = fn()
            finally:
                sys.stderr.flush()
                os.dup2(saved, 2)
            tf.seek(0)
            txt = tf.read().decode("utf-8", "replace")
    finally:
        os.close(saved)
    return out, txt


def main():
    # --nograde skips the ~2 min grading pass and reports stats only. Track A
    # uses it as a fast regression oracle: any change to the proposal sort
    # order shows up as a drift in per-threshold merge counts long before it
    # shows up in VOI, so exact-match on merges is the tighter gate.
    nograde = "--nograde" in sys.argv
    # --e6s grades the deterministic compact-every-inner path instead. Needed
    # whenever a shared kernel (propose priority, contact-sum scaling) changes,
    # since those affect both paths.
    want_e6s = "--e6s" in sys.argv
    tag = "E6s" if want_e6s else "E6t"
    print_contract()
    compile_d()
    if want_e6s:
        os.environ.pop("WATERZ_PAPER_E6T", None)
    else:
        os.environ["WATERZ_PAPER_E6T"] = "1"
    before = gpu_state()
    print(f"A1 {tag} four-T VOI. GPU at start: {before}", flush=True)
    print("A1 timing is PROVISIONAL (shared GPU); this gate gates VOI only.",
          flush=True)

    u, v, sm, ct, fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    bind(lib)
    eps = _agg_eps()
    print(f"A1 eps={eps} (WATERZ_AGG_EPS or locked 0.08)", flush=True)

    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    (rc, parents, stats, device_ms, wall_ms), errtxt = capture_stderr(
        lambda: run(lib, u, v, sm, ct, thrs, max_id, eps))
    sys.stderr.write(errtxt)
    inner = [int(x) for x in stats[:, 2]]
    merges = [int(x) for x in stats[:, 1]]
    outer = [int(x) for x in stats[:, 0]]
    print(
        f"A1 rc={rc} device_ms={device_ms:.2f} wall_ms={wall_ms:.2f}\n"
        f"   outer={outer}\n   inner={inner}\n   merges={merges}",
        flush=True,
    )

    # Prove which path ran, so a silently-defaulted E6s cannot pass as E6t.
    saw_e6s = "E6s T=" in errtxt
    saw_e6t = "E6t T=" in errtxt
    ran = "E6s" if (saw_e6s and not saw_e6t) else (
        "E6t" if (saw_e6t and not saw_e6s) else "ambiguous")
    path_ok = ran == tag
    print(
        f"A1 path check (from library stderr marker): ran={ran} wanted={tag}"
        f" -> {'OK' if path_ok else 'WRONG PATH'}",
        flush=True,
    )

    result = {
        "rc": int(rc),
        "device_ms_provisional": device_ms,
        "wall_ms_provisional": wall_ms,
        "outer": outer,
        "inner": inner,
        "merges": merges,
        "eps": eps,
        "thresholds": list(AFF_THRESHOLDS),
        "path_ran": ran,
        "wanted_path": tag,
        "path_ok": bool(path_ok),
        "gpu_at_start": before,
        "gpu_at_end": gpu_state(),
        "timing_is_graded": False,
    }

    # nseg is a cheap, exact partition fingerprint; combined with merge counts
    # it detects any reordering of the proposal sort without paying for VOI.
    nseg = [int(np.unique(parents[i]).size) for i in range(len(thrs))]
    result["nseg_unique_parents"] = nseg
    print(f"A1 unique-parent fingerprint={nseg}", flush=True)

    ok = False
    if rc != 1:
        print(f"A1 FAIL rc={rc}", flush=True)
    elif not path_ok:
        print(f"A1 FAIL: wanted {tag} but the other path ran.", flush=True)
    elif nograde:
        print("A1 --nograde: stats only, VOI not evaluated", flush=True)
    else:
        buf = io.StringIO()
        with redirect_stdout(buf):
            ok = grade_parents(parents, fr, f"A1-{tag}", f"a1_{tag.lower()}_voi")
        sys.stdout.write(buf.getvalue())

    result["voi_pass"] = bool(ok)
    result["graded"] = not nograde
    CACHE.mkdir(parents=True, exist_ok=True)
    stem = f"a1_{tag.lower()}"
    name = f"{stem}_stats.json" if nograde else f"{stem}_voi.json"
    (CACHE / name).write_text(json.dumps(result, indent=2) + "\n")
    if nograde:
        return rc == 1 and took_e6t
    print(
        f"A1 {'PASS' if ok else 'FAIL'} (VOI only; no stamp written; "
        "no speed claim)",
        flush=True,
    )
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
