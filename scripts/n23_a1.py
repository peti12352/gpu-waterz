#!/usr/bin/env python3
"""N23 A1: WATERZ_CSR_REWRITE incidence-list dirty fuse on E6s.

Requires N23_D0 PASS. Identity+four then 2.16 even if val cut < 100 ms
(CSR exception). A5 flags stay off. Default CSR off. Idle RTX 5090.
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from e6r_parhac import DSO, compile_d  # noqa: E402
from _agg_common import load_rag  # noqa: E402
from t3_memsafe import grade_t3, voi_parent_mmap  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from n13_baseline import parks_env  # noqa: E402
from n20_res import attach  # noqa: E402
from n23_dead import stamp, release_gpu  # noqa: E402

CLAIM = "N23_A1 CSR rewrite listed fuse; idle RTX 5090; flags stay default off"
PIN_AGG_216 = 1679.9063761718571
PIN_E2E_216 = 3093.4
VAL_PIN_MS = 456.7932924255729
PIN_REWRITE_216_MS = 273.0
PIN_INNER = 538
PIN_MERGES = 1853545
PIN_NSEG = 321855
NOISE_MS = 50.0
TAG = "N23_A1"


def legal_env():
    os.environ.setdefault("WATERZ_UF_ALGO", "3")
    os.environ["WATERZ_HOST_PARK"] = "0"
    os.environ["WATERZ_AFF_PARK"] = "0"
    os.environ.setdefault("WATERZ_AGG_LEVERS", "15")
    os.environ["WATERZ_FOLD_FLATTEN"] = "1"
    os.environ["WATERZ_SHARE_OFF"] = "1"
    os.environ["WATERZ_HOOK_ROOT"] = "1"
    os.environ["WATERZ_FUSE_DIRTY"] = "1"
    os.environ["WATERZ_NLIVE_ARITH"] = "1"
    os.environ["WATERZ_EMIT_HOLES"] = "1"
    os.environ.pop("WATERZ_HASH_INSERT_ONLY", None)
    os.environ.pop("WATERZ_PAPER_E6T", None)
    os.environ.pop("WATERZ_FUSE_PACK", None)
    os.environ.pop("WATERZ_SLOT_EMIT", None)
    os.environ.pop("WATERZ_LISTED_INSERT", None)
    os.environ.pop("WATERZ_LISTED_REBUILD", None)


def extra_env(csr: bool, four_tag: str, skip_build: bool):
    e = {
        "WATERZ_CSR_REWRITE": "1" if csr else "0",
        "WATERZ_SLOT_EMIT": "0",
        "WATERZ_LISTED_INSERT": "0",
        "WATERZ_LISTED_REBUILD": "0",
        "WATERZ_FOLD_FLATTEN": "1",
        "WATERZ_SHARE_OFF": "1",
        "WATERZ_HOOK_ROOT": "1",
        "WATERZ_FUSE_DIRTY": "1",
        "WATERZ_NLIVE_ARITH": "1",
        "WATERZ_EMIT_HOLES": "1",
        "WATERZ_FOUR_TAG": four_tag,
    }
    if skip_build:
        e["WATERZ_SKIP_BUILD"] = "1"
    return e


def bind_parhac(lib):
    lib.parhac_paper_d.restype = ctypes.c_int
    lib.parhac_paper_d.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    return lib


def run_once(lib, u, v, sm, ct, max_id, eps=0.40):
    thrs = np.asarray([0.3], dtype=np.float64)
    parents = np.empty((1, max_id + 1), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
    t0 = time.perf_counter()
    rc = lib.parhac_paper_d(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(1),
        ctypes.c_double(eps),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    split, merge, nseg = voi_parent_mmap(parents[0])
    ok, sl, ml = grade_t3(split, merge)
    return parents[0].copy(), {
        "rc": int(rc),
        "split": split,
        "merge": merge,
        "nseg": nseg,
        "ok": bool(ok),
        "limit_split": sl,
        "limit_merge": ml,
        "inner": int(stats[0, 2]),
        "merges": int(stats[0, 1]),
        "wall_ms": wall_ms,
        "nouter": int(stats[0, 0]),
    }


def control_dump(path: Path):
    legal_env()
    os.environ["WATERZ_CSR_REWRITE"] = "0"
    os.environ["WATERZ_SKIP_BUILD"] = "1"
    compile_d()
    lib = bind_parhac(ctypes.CDLL(str(DSO)))
    u, v, sm, ct, _fr, max_id = load_rag()
    p, meta = run_once(lib, u, v, sm, ct, max_id)
    np.save(path, p)
    path.with_suffix(".json").write_text(json.dumps(meta) + "\n")
    return 0


def run_four():
    env = parks_env(extra_env(True, TAG, True))
    env.pop("WATERZ_HASH_INSERT_ONLY", None)
    env.pop("WATERZ_FUSE_PACK", None)
    r = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts/n15_four.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300,
    )
    doc = {}
    if r.stdout.strip():
        try:
            doc = json.loads(r.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            doc = {"stdout_tail": r.stdout[-2000:]}
    doc["rc"] = r.returncode
    if r.returncode != 0 and "four_pass" not in doc:
        doc["stderr_tail"] = (r.stderr or "")[-2000:]
    return doc


def run_216():
    release_gpu()
    p216 = CACHE / "n8_216.json"
    if p216.is_file():
        p216.unlink()
    env = parks_env(extra_env(True, TAG, True))
    env["WATERZ_STAGE_MS"] = "1"
    env.pop("WATERZ_HASH_INSERT_ONLY", None)
    env.pop("WATERZ_FUSE_PACK", None)
    r = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
        cwd=str(ROOT), env=env, timeout=900,
    )
    doc = {"rc216": r.returncode}
    if p216.is_file() and r.returncode == 0:
        doc.update(json.loads(p216.read_text()))
    return doc


def try_nsys_val():
    nsys = shutil.which("nsys")
    if not nsys:
        return {"skipped": True, "reason": "nsys_missing"}
    release_gpu()
    out = CACHE / "N23_A1_nsys"
    env = parks_env(extra_env(True, TAG, True))
    py = (
        "import ctypes,sys; sys.path[:0]=['src','scripts']; "
        "from e6r_parhac import compile_d,DSO; from n23_a1 import bind_parhac,run_once; "
        "from _agg_common import load_rag; compile_d(); "
        "lib=bind_parhac(ctypes.CDLL(str(DSO))); "
        "u,v,sm,ct,_fr,max_id=load_rag(); run_once(lib,u,v,sm,ct,max_id)"
    )
    cmd = [
        nsys, "profile", "--stats=true", "--force-overwrite=true",
        "-o", str(out),
        sys.executable, "-c", py,
    ]
    try:
        r = subprocess.run(
            cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
            timeout=240,
        )
    except subprocess.TimeoutExpired:
        return {"skipped": True, "reason": "nsys_timeout"}
    text = (r.stdout or "") + "\n" + (r.stderr or "")
    hits = []
    for ln in text.splitlines():
        if "k_rewrite" in ln or "k_csr_" in ln:
            hits.append(ln.strip())
    return {
        "skipped": False,
        "rc": r.returncode,
        "kernel_lines": hits[:40],
        "pin_rewrite_216_ms": PIN_REWRITE_216_MS,
        "note": "val nsys; 273.0 ms pin is 2.16 k_rewrite_dirty_fuse",
    }


def write_notes(doc: dict):
    m1 = doc["trial"]["m1"]
    lines = [
        f"# {TAG} CSR rewrite listed fuse",
        "",
        CLAIM + ".",
        "",
        "- env WATERZ_CSR_REWRITE=1; A5 flags off; product default still off",
        f"- trial T=0.3 wall_ms={m1['wall_ms']:.1f} "
        f"run2={doc['trial']['m2']['wall_ms']:.1f}",
        f"- VOI ok={doc['trial']['ok']} split={m1['split']:.6f} "
        f"merge={m1['merge']:.6f} nseg={m1['nseg']}",
        f"- inner={m1['inner']} merges={m1['merges']}",
        f"- run2_array_equal={doc['trial']['run2_array_equal']}",
        f"- vs_i0_counts={doc['vs_i0_counts']} vs_control_parents={doc['vs_control_parents']}",
        f"- identity={doc['identity']} voi_ok={doc['voi_ok']}",
        f"- val_cut_vs_d0_ms={doc.get('val_cut_vs_d0_ms')}",
        f"- four_pass={doc.get('four', {}).get('four_pass')} four_ms={doc.get('four', {}).get('ms')}",
        f"- ran_216={doc.get('ran_216')} keep_default=False",
        f"- csr_exception_100ms={doc.get('csr_exception_100ms')}",
        f"- agg_216_ms={doc.get('agg_216_ms')} pin={PIN_AGG_216}",
        f"- e2e_216_ms={doc.get('e2e_216_ms')} pin={PIN_E2E_216}",
        f"- closer={doc.get('closer')} stamp={doc.get('stamp_reason')}",
        f"- nsys={doc.get('nsys', {}).get('skipped')}",
    ]
    (ROOT / "notes" / f"{TAG}.md").write_text("\n".join(lines) + "\n")


def main():
    if "--control-dump" in sys.argv:
        return control_dump(CACHE / f"{TAG}_control_parent.npy")

    d0p = CACHE / "N23_D0.json"
    if not d0p.is_file():
        print(f"{TAG} REFUSE missing N23_D0.json", flush=True)
        return 2
    d0 = json.loads(d0p.read_text())
    if not d0.get("go_gpu_csr"):
        print(f"{TAG} REFUSE D0 did not PASS", flush=True)
        stamp(TAG, "d0_block", d0, note="do not write CUDA run")
        return 2

    legal_env()
    if os.environ.get("WATERZ_HASH_INSERT_ONLY") or os.environ.get("WATERZ_FUSE_PACK"):
        print(f"{TAG} REFUSE HASH_INSERT_ONLY/FUSE_PACK", flush=True)
        return 2
    if os.environ.get("WATERZ_PAPER_E6T"):
        print(f"{TAG} REFUSE WATERZ_PAPER_E6T", flush=True)
        return 2
    busy = card_busy()
    if busy:
        print(f"{TAG} REFUSE card_busy", busy, flush=True)
        return 2
    print(f"{TAG} gpu", gpu_state(), flush=True)

    os.environ["WATERZ_CSR_REWRITE"] = "1"
    os.environ["WATERZ_SLOT_EMIT"] = "0"
    os.environ["WATERZ_LISTED_INSERT"] = "0"
    os.environ["WATERZ_LISTED_REBUILD"] = "0"
    compile_d()
    lib = bind_parhac(ctypes.CDLL(str(DSO)))
    u, v, sm, ct, _fr, max_id = load_rag()
    p1, m1 = run_once(lib, u, v, sm, ct, max_id)
    p2, m2 = run_once(lib, u, v, sm, ct, max_id)
    trial = {
        "ok": bool(m1["ok"] and m2["ok"]),
        "run2_array_equal": bool(np.array_equal(p1, p2)),
        "m1": m1,
        "m2": m2,
        "env": "WATERZ_CSR_REWRITE=1",
    }
    vs_i0 = bool(
        m1["inner"] == PIN_INNER and m2["inner"] == PIN_INNER
        and m1["merges"] == PIN_MERGES and m2["merges"] == PIN_MERGES
        and m1["nseg"] == PIN_NSEG and m2["nseg"] == PIN_NSEG
    )

    ctrl_npy = CACHE / f"{TAG}_control_parent.npy"
    if ctrl_npy.is_file():
        ctrl_npy.unlink()
    env = parks_env(extra_env(False, TAG, True))
    cr = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts/n23_a1.py"), "--control-dump"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180,
    )
    vs_ctrl = False
    ctrl_meta = {"rc": cr.returncode}
    if cr.returncode == 0 and ctrl_npy.is_file():
        ctrl = np.load(ctrl_npy)
        vs_ctrl = bool(np.array_equal(p1, ctrl) and np.array_equal(p2, ctrl))
        cj = ctrl_npy.with_suffix(".json")
        if cj.is_file():
            ctrl_meta = json.loads(cj.read_text())
            ctrl_meta["rc"] = cr.returncode
    else:
        ctrl_meta["stderr_tail"] = (cr.stderr or "")[-2000:]
        ctrl_meta["stdout_tail"] = (cr.stdout or "")[-2000:]

    identity = bool(trial["run2_array_equal"] and vs_i0 and vs_ctrl)
    voi_ok = bool(trial["ok"])
    wall = float(m1["wall_ms"])
    val_cut = VAL_PIN_MS - wall

    doc = {
        "claim": CLAIM,
        "tag": TAG,
        "pin_agg_216_ms": PIN_AGG_216,
        "val_pin_ms": VAL_PIN_MS,
        "pin_rewrite_216_ms": PIN_REWRITE_216_MS,
        "d0": {"pass": d0.get("pass"), "scan_over_csr_lookup": d0.get("scan_over_csr_lookup")},
        "force_216_after_ident_four": True,
        "trial": trial,
        "control": ctrl_meta,
        "vs_i0_counts": vs_i0,
        "vs_control_parents": vs_ctrl,
        "identity": identity,
        "voi_ok": voi_ok,
        "val_cut_vs_d0_ms": val_cut,
        "ran_216": False,
        "keep_default": False,
        "csr_exception_100ms": "A1 does 2.16 after identity+four even if val cut < 100 ms",
    }

    four = {}
    run216 = {}
    nsys = {}
    stamp_reason = None
    closer = False
    if not identity:
        stamp_reason = "identity_fail"
        stamp(TAG, "identity_fail", {
            "vs_i0_counts": vs_i0, "vs_control_parents": vs_ctrl,
            "inner": m1["inner"], "merges": m1["merges"], "nseg": m1["nseg"],
        }, note="CSR rewrite; no 2.16")
    elif not voi_ok:
        stamp_reason = "voi_fail"
        stamp(TAG, "voi_fail", {"split": m1["split"], "merge": m1["merge"]})
    else:
        four = run_four()
        if not bool(four.get("four_pass")):
            stamp_reason = "four_t_fail"
            stamp(TAG, "four_t_fail", {"four": four})
        else:
            doc = attach(doc, TAG)
            if not doc.get("timing_usable_vs_1679"):
                stamp_reason = "timing_contaminated"
                stamp(TAG, "timing_contaminated", {"val_cut_vs_d0_ms": val_cut})
            else:
                run216 = run_216()
                doc["ran_216"] = True
                st = run216.get("stages") or {}
                agg = float(st.get("agg") or 0)
                e2e = float(run216.get("e2e_ms") or 0)
                doc["agg_216_ms"] = agg
                doc["e2e_216_ms"] = e2e
                doc["nlab_216"] = run216.get("nlab")
                nsys = try_nsys_val()
                if run216.get("rc216") not in (0, None) or agg <= 0:
                    stamp_reason = "216_refused"
                    stamp(TAG, "216_refused", {
                        "rc216": run216.get("rc216"), "agg_216_ms": agg,
                    }, note="n8_run216 refused or missing stages.agg")
                elif agg > 0 and agg < PIN_AGG_216 - NOISE_MS:
                    stamp_reason = "216_cut"
                    closer = True
                    run2162 = run_216()
                    doc["run216_2"] = run2162
                    st2 = run2162.get("stages") or {}
                    agg2 = float(st2.get("agg") or 0)
                    doc["agg_216_ms_2"] = agg2
                    if agg2 <= 0 or agg2 >= PIN_AGG_216 - NOISE_MS:
                        closer = False
                        stamp_reason = "216_cut_not_repro"
                        stamp(TAG, "216_cut_not_repro", {
                            "agg_216_ms": agg, "agg_216_ms_2": agg2,
                            "pin": PIN_AGG_216,
                        }, note="CSR rewrite; second 2.16 did not hold 50 ms")
                    else:
                        stamp(TAG, "216_cut", {
                            "agg_216_ms": agg, "agg_216_ms_2": agg2,
                            "pin": PIN_AGG_216, "e2e_ms": e2e,
                        }, note="CSR rewrite; keep_default false")
                elif agg > PIN_AGG_216 + NOISE_MS:
                    stamp_reason = "216_slower"
                    stamp(TAG, "216_slower", {
                        "agg_216_ms": agg, "pin": PIN_AGG_216, "e2e_ms": e2e,
                    })
                else:
                    stamp_reason = "216_no_cut"
                    stamp(TAG, "216_no_cut", {
                        "agg_216_ms": agg, "pin": PIN_AGG_216, "e2e_ms": e2e,
                    }, note="CSR rewrite forced 2.16; not a closer")

    doc["four"] = four
    doc["run216"] = run216
    doc["nsys"] = nsys
    doc["closer"] = closer
    doc["stamp_reason"] = stamp_reason
    if "res" not in doc:
        doc = attach(doc, TAG)
    (CACHE / f"{TAG}.json").write_text(json.dumps(doc, indent=2) + "\n")
    write_notes(doc)
    print(json.dumps({
        "tag": TAG,
        "identity": identity,
        "voi_ok": voi_ok,
        "four_pass": four.get("four_pass"),
        "val_cut_vs_d0_ms": val_cut,
        "closer": closer,
        "stamp_reason": stamp_reason,
        "ran_216": doc.get("ran_216"),
        "agg_216_ms": doc.get("agg_216_ms"),
        "e2e_216_ms": doc.get("e2e_216_ms"),
    }, indent=2), flush=True)
    return 0 if identity and voi_ok and four.get("four_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
