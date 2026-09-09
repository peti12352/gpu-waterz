#!/usr/bin/env python3
"""N13 L5: build RAMA, then grade T=0.3 VOI. Never default on FAIL. No cuSLINK.

Not a 2 Gvox/s claim. Not 3090 Ti. Idle-5090 only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import load_rag  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from t3_memsafe import grade_t3, voi_parent_mmap  # noqa: E402

NOTE = ROOT / "notes/N13_RAMA.md"
OUT = CACHE / "n13_rama.json"
SRC = ROOT / "papers/repos/RAMA"
BUILD = Path("/tmp/n13_rama_build")
INST = Path("/tmp/n13_rama.mc")
SOL = Path("/tmp/n13_rama_sol.txt")
T = 0.3


def emit_multicut():
    u, v, sm, ct, fr, max_id = load_rag()
    mean = sm / np.maximum(ct.astype(np.float64), 1.0)
    # skip background endpoints
    mask = (u != 0) & (v != 0)
    uu = u[mask].astype(np.int64)
    vv = v[mask].astype(np.int64)
    cost = (mean[mask] - T).astype(np.float64)
    nodes = np.unique(np.concatenate([uu, vv]))
    remap = {int(n): i for i, n in enumerate(nodes.tolist())}
    lines = ["MULTICUT\n"]
    for a, b, c in zip(uu.tolist(), vv.tolist(), cost.tolist()):
        lines.append(f"{remap[int(a)]} {remap[int(b)]} {c:.8g}\n")
    INST.write_text("".join(lines))
    return {
        "n_edges": int(mask.sum()),
        "n_nodes": int(nodes.size),
        "max_id": int(max_id),
        "nodes": nodes.astype(np.int64),
        "fr_max": int(fr.max()),
    }


def parents_from_sol(meta):
    labs = []
    with SOL.open() as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                labs.append(int(ln))
    nodes = meta["nodes"]
    if len(labs) != len(nodes):
        raise RuntimeError(f"sol len {len(labs)} != nodes {len(nodes)}")
    parent = np.arange(meta["max_id"] + 1, dtype=np.uint32)
    # cluster ids from RAMA are 0-based; shift so 0 stays background
    for n, lab in zip(nodes.tolist(), labs):
        parent[int(n)] = np.uint32(lab + 1)
    parent[0] = 0
    return parent


def main():
    print("Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N13 L5 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "src": str(SRC),
        "built": False,
        "task_legal": False,
        "keep_default": False,
    }
    if not SRC.is_dir():
        doc["error"] = "RAMA tree missing"
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2) + "\n")
        NOTE.write_text(
            "# N13 L5 RAMA\n\nNot a 2 Gvox/s claim. Not 3090 Ti.\n\n"
            "- built=False tree missing. not a closer.\n"
        )
        print("N13 L5 tree missing", flush=True)
        return 0

    BUILD.mkdir(parents=True, exist_ok=True)
    nvcc = "/usr/local/cuda-12.8/bin/nvcc"
    cm_cmd = [
        "cmake", "-S", str(SRC), "-B", str(BUILD),
        "-DWITH_CUDA=ON", "-DWITH_TORCH=OFF",
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        "-DCMAKE_CUDA_ARCHITECTURES=86;120",
    ]
    cm = subprocess.run(
        cm_cmd,
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "CUDACXX": nvcc,
             "PATH": "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")},
    )
    doc["cmake_rc"] = cm.returncode
    doc["cmake_tail"] = (cm.stdout + cm.stderr)[-2500:]
    print(f"N13 L5 cmake rc={cm.returncode}", flush=True)
    if cm.returncode != 0:
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2) + "\n")
        NOTE.write_text(
            "# N13 L5 RAMA\n\nNot a 2 Gvox/s claim. Not 3090 Ti.\n\n"
            f"- cmake_rc={cm.returncode} built=False. not a closer.\n"
            f"```\n{doc['cmake_tail']}\n```\n"
        )
        print("N13 L5 cmake fail", flush=True)
        return 0

    mk_env = {**os.environ, "CUDACXX": nvcc,
              "PATH": "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")}
    mk = subprocess.run(
        ["cmake", "--build", str(BUILD), "-j", "8",
         "--target", "rama_text_input_gpu"],
        capture_output=True, text=True, timeout=1200,
        env=mk_env,
    )
    doc["build_rc"] = mk.returncode
    doc["build_tail"] = (mk.stdout + mk.stderr)[-2500:]
    print(f"N13 L5 build rc={mk.returncode}", flush=True)
    bin_path = BUILD / "src" / "rama_text_input_gpu"
    if not bin_path.is_file():
        alt = list(BUILD.rglob("rama_text_input_gpu"))
        bin_path = alt[0] if alt else bin_path
    doc["binary"] = str(bin_path)
    doc["built"] = bool(bin_path.is_file() and mk.returncode == 0)
    if not doc["built"]:
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2) + "\n")
        NOTE.write_text(
            "# N13 L5 RAMA\n\nNot a 2 Gvox/s claim. Not 3090 Ti.\n\n"
            f"- built=False build_rc={mk.returncode}. not a closer. "
            "Did not vendor cudaMST.\n"
            f"```\n{doc['build_tail']}\n```\n"
        )
        print("N13 L5 build fail", flush=True)
        return 0

    meta = emit_multicut()
    doc["graph"] = {k: (int(v) if k != "nodes" else int(v.size))
                    for k, v in meta.items()}
    try:
        run = subprocess.run(
            [str(bin_path), "-f", str(INST), "-o", str(SOL)],
            capture_output=True, text=True, timeout=120,
        )
        doc["run_rc"] = run.returncode
        doc["run_tail"] = (run.stdout + run.stderr)[-2000:]
        doc["run_timeout"] = False
    except subprocess.TimeoutExpired as e:
        doc["run_rc"] = None
        doc["run_timeout"] = True
        doc["run_tail"] = ((e.stdout or b"") + (e.stderr or b"")).decode(
            "utf-8", "replace")[-2000:] if isinstance(e.stdout, (bytes, type(None))) else str(e)[-500:]
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2) + "\n")
        NOTE.write_text(
            "# N13 L5 RAMA\n\nNot a 2 Gvox/s claim. Not 3090 Ti.\n\n"
            f"- built=True binary={bin_path}\n"
            "- GPU solver timed out at 120s on val RAG. No labels. "
            "Prior hang was 600s. not a closer. keep_default=False.\n"
            f"- graph={doc.get('graph')}\n"
        )
        print("N13 L5 solver timeout. not graded. keep_default=False", flush=True)
        return 0
    print(f"N13 L5 run rc={run.returncode}\n{doc['run_tail']}", flush=True)
    if run.returncode != 0 or not SOL.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2) + "\n")
        NOTE.write_text(
            "# N13 L5 RAMA\n\nNot a 2 Gvox/s claim. Not 3090 Ti.\n\n"
            f"- built=True run_rc={run.returncode} no sol. not graded.\n"
            f"```\n{doc['run_tail']}\n```\n"
        )
        return 0

    parent = parents_from_sol(meta)
    split, merge, nseg = voi_parent_mmap(parent)
    ok, sl, ml = grade_t3(split, merge)
    doc["split"] = split
    doc["merge"] = merge
    doc["nseg"] = nseg
    doc["ok"] = bool(ok)
    doc["task_legal"] = bool(ok)
    doc["keep_default"] = False
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N13 L5 RAMA\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. Signed multicut, "
        "cost=mean_aff-0.3. No cuSLINK.\n\n"
        f"- built=True binary={bin_path}\n"
        f"- T=0.3 split={split:.4f} merge={merge:.4f} nseg={nseg} "
        f"limit {sl}/{ml} ok={ok}\n"
        f"- task_legal={ok} keep_default=False\n"
        f"- gpu_time_line={doc['run_tail'][:400]}\n"
    )
    print(f"N13 L5 VOI ok={ok} split={split:.4f} merge={merge:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
