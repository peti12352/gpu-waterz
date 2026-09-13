#!/usr/bin/env python3
"""N21 A2 listed rebuild (holes ∪ old alist). Optional --stack with A1 insert.

Identity vs E6s control + T=0.3 VOI, then four-T. Not FUSE_PACK.
No 2.16 unless four-T PASS and val agg cut >=100 ms vs D0 val pin.
Not a 2 Gvox/s number; not 3090 Ti.
"""
from __future__ import annotations

import ctypes
import json
import os
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
from n21_dead import stamp  # noqa: E402

PIN_AGG_216 = 1679.9063761718571
VAL_PIN_MS = 456.7932924255729
PIN_INNER = 538
PIN_MERGES = 1853545
PIN_NSEG = 321855
NOISE_MS = 50.0
CUT_216_MS = 100.0


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


def extra_env(insert: bool, rebuild: bool, four_tag: str, skip_build: bool):
    e = {
        "WATERZ_LISTED_INSERT": "1" if insert else "0",
        "WATERZ_LISTED_REBUILD": "1" if rebuild else "0",
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


def control_dump(path: Path):
    legal_env()
    os.environ["WATERZ_LISTED_INSERT"] = "0"
    os.environ["WATERZ_LISTED_REBUILD"] = "0"
    os.environ["WATERZ_SKIP_BUILD"] = "1"
    compile_d()
    lib = bind_parhac(ctypes.CDLL(str(DSO)))
    u, v, sm, ct, _fr, max_id = load_rag()
    p, meta = run_once(lib, u, v, sm, ct, max_id)
    np.save(path, p)
    path.with_suffix(".json").write_text(json.dumps(meta) + "\n")
    return 0


def run_four(tag: str, insert: bool, rebuild: bool):
    env = parks_env(extra_env(insert, rebuild, tag, True))
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


def run_216(tag: str, insert: bool, rebuild: bool):
    p216 = CACHE / "n8_216.json"
    if p216.is_file():
        p216.unlink()
    env = parks_env(extra_env(insert, rebuild, tag, True))
    env["WATERZ_STAGE_MS"] = "1"
    env.pop("WATERZ_HASH_INSERT_ONLY", None)
    env.pop("WATERZ_FUSE_PACK", None)
    r = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
        cwd=str(ROOT), env=env, timeout=600,
    )
    doc = {"rc216": r.returncode}
    if p216.is_file() and r.returncode == 0:
        doc.update(json.loads(p216.read_text()))
    return doc


def write_notes(tag: str, claim: str, doc: dict, insert: bool, rebuild: bool):
    lines = [
        f"# {tag} listed rebuild" + (" + insert" if insert else ""),
        "",
        claim + ".",
        "",
        f"- env LISTED_REBUILD={int(rebuild)} LISTED_INSERT={int(insert)}; "
        "FUSE_PACK unset; HASH_INSERT_ONLY unset; product default still off",
        f"- trial T=0.3 wall_ms={doc['trial']['m1']['wall_ms']:.1f} "
        f"run2={doc['trial']['m2']['wall_ms']:.1f} "
        f"(cold vs D0 pin; run2 is warmup, not the cut)",
        f"- VOI ok={doc['trial']['ok']} split={doc['trial']['m1']['split']:.6f} "
        f"merge={doc['trial']['m1']['merge']:.6f} nseg={doc['trial']['m1']['nseg']}",
        f"- inner={doc['trial']['m1']['inner']} merges={doc['trial']['m1']['merges']}",
        f"- run2_array_equal={doc['trial']['run2_array_equal']}",
        f"- vs_i0_counts={doc['vs_i0_counts']} vs_control_parents={doc['vs_control_parents']}",
        f"- identity={doc['identity']} voi_ok={doc['voi_ok']}",
        f"- val_cut_vs_d0_ms={doc.get('val_cut_vs_d0_ms')} "
        f"(D0 val pin {VAL_PIN_MS:.1f} ms, not the 2.16 pin {PIN_AGG_216:.1f})",
        f"- four_pass={doc.get('four', {}).get('four_pass')} four_ms={doc.get('four', {}).get('ms')}",
        f"- ran_216={doc.get('ran_216')} keep_default={doc.get('keep_default')}",
        f"- timing_usable_vs_1679={doc.get('timing_usable_vs_1679')}",
        f"- closer={doc.get('closer')} stamp={doc.get('stamp_reason')}",
    ]
    (ROOT / "notes" / f"{tag}.md").write_text("\n".join(lines) + "\n")


def main():
    stack = "--stack" in sys.argv
    if "--control-dump" in sys.argv:
        tag = "N21_A4" if stack else "N21_A2"
        return control_dump(CACHE / f"{tag}_control_parent.npy")

    tag = "N21_A4" if stack else "N21_A2"
    insert = bool(stack)
    rebuild = True
    claim = (
        f"{tag} listed rebuild"
        + (" + insert" if insert else "")
        + "; not a throughput claim; not a 2 Gvox/s number; not 3090 Ti"
    )

    legal_env()
    if os.environ.get("WATERZ_HASH_INSERT_ONLY") or os.environ.get("WATERZ_FUSE_PACK"):
        print(f"{tag} REFUSE HASH_INSERT_ONLY/FUSE_PACK", flush=True)
        return 2
    busy = card_busy()
    if busy:
        print(f"{tag} REFUSE card_busy", busy, flush=True)
        return 2
    print(f"{tag} gpu", gpu_state(), flush=True)

    os.environ["WATERZ_LISTED_INSERT"] = "1" if insert else "0"
    os.environ["WATERZ_LISTED_REBUILD"] = "1"
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
        "env": f"LISTED_REBUILD=1 LISTED_INSERT={int(insert)}",
    }
    vs_i0 = bool(
        m1["inner"] == PIN_INNER and m2["inner"] == PIN_INNER
        and m1["merges"] == PIN_MERGES and m2["merges"] == PIN_MERGES
        and m1["nseg"] == PIN_NSEG and m2["nseg"] == PIN_NSEG
    )

    ctrl_npy = CACHE / f"{tag}_control_parent.npy"
    if ctrl_npy.is_file():
        ctrl_npy.unlink()
    env = parks_env(extra_env(False, False, tag, True))
    env.pop("WATERZ_HASH_INSERT_ONLY", None)
    env.pop("WATERZ_FUSE_PACK", None)
    argv = [sys.executable, "-u", str(ROOT / "scripts/n21_a2.py"), "--control-dump"]
    if stack:
        argv.append("--stack")
    cr = subprocess.run(
        argv, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180,
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
        "claim": claim,
        "tag": tag,
        "pin_agg_216_ms": PIN_AGG_216,
        "val_pin_ms": VAL_PIN_MS,
        "trial": trial,
        "control": ctrl_meta,
        "vs_i0_counts": vs_i0,
        "vs_control_parents": vs_ctrl,
        "identity": identity,
        "voi_ok": voi_ok,
        "val_cut_vs_d0_ms": val_cut,
        "ran_216": False,
        "keep_default": False,
        "FUSE_PACK": False,
        "HASH_INSERT_ONLY": False,
    }

    four = {}
    run216 = {}
    stamp_reason = None
    closer = None
    note = "listed rebuild" + (" + insert" if insert else "")
    if not identity:
        stamp_reason = "identity_fail"
        closer = False
        stamp(tag, "identity_fail", {
            "vs_i0_counts": vs_i0, "vs_control_parents": vs_ctrl,
            "run2_array_equal": trial["run2_array_equal"],
            "inner": m1["inner"], "merges": m1["merges"], "nseg": m1["nseg"],
        }, note=note)
    elif not voi_ok:
        stamp_reason = "voi_fail"
        closer = False
        stamp(tag, "voi_fail", {
            "split": m1["split"], "merge": m1["merge"],
        }, note=note)
    else:
        four = run_four(tag, insert, rebuild)
        four_ok = bool(four.get("four_pass"))
        if not four_ok:
            stamp_reason = "four_t_fail"
            closer = False
            stamp(tag, "four_t_fail", {"four": four}, note=note)
        elif val_cut < NOISE_MS:
            stamp_reason = "noise_val_cut"
            closer = False
            stamp(tag, "noise_val_cut", {
                "val_cut_vs_d0_ms": val_cut, "wall_ms": wall,
                "val_pin_ms": VAL_PIN_MS,
            }, note=note + "; identity+four PASS; not a closer")
        else:
            closer = True
            stamp_reason = "val_cut_ge_noise"
            doc = attach(doc, tag)
            if val_cut >= CUT_216_MS and doc.get("timing_usable_vs_1679"):
                run216 = run_216(tag, insert, rebuild)
                doc["ran_216"] = True
                st = run216.get("stages") or {}
                agg = float(st.get("agg") or 0)
                doc["agg_216_ms"] = agg
                if agg > 0 and agg < PIN_AGG_216 - NOISE_MS:
                    stamp_reason = "216_cut"
                else:
                    stamp_reason = "216_no_cut"
                    closer = False
                    stamp(tag, "216_no_cut", {
                        "agg_216_ms": agg, "pin": PIN_AGG_216,
                    }, note=note)

    doc["four"] = four
    doc["run216"] = run216
    doc["closer"] = closer
    doc["stamp_reason"] = stamp_reason
    if "res" not in doc:
        doc = attach(doc, tag)
    (CACHE / f"{tag}.json").write_text(json.dumps(doc, indent=2) + "\n")
    write_notes(tag, claim, doc, insert, rebuild)
    print(json.dumps({
        "tag": tag,
        "identity": identity,
        "voi_ok": voi_ok,
        "four_pass": four.get("four_pass"),
        "val_cut_vs_d0_ms": val_cut,
        "closer": closer,
        "stamp_reason": stamp_reason,
        "timing_usable_vs_1679": doc.get("timing_usable_vs_1679"),
        "ran_216": doc.get("ran_216"),
        "agg_216_ms": doc.get("agg_216_ms"),
    }, indent=2), flush=True)
    return 0 if identity and voi_ok and four.get("four_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
