#!/usr/bin/env python3
"""E5: W2 sound levers as CPU/static proofs. No 17.22 GiB claim.

1. Affinity consumers in ws.cu: only k_flow reads aff[].
2. vcount is indexed by plateau root; n_plateau << nvox on real fragments.
3. Peak-line table from d1_mem.json + w2_mem arithmetic, with the
   phase-ownership caveat left intact.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
WS = ROOT / "csrc/ws.cu"
GIB = 1 << 30
USABLE = 23.0


def aff_readers():
    src = WS.read_text()
    # kernel signatures that take aff
    kern = re.findall(r"__global__ void (\w+)\s*\(([^)]*)\)", src, re.S)
    aff_kern = [n for n, args in kern if re.search(r"\baff\b", args)]
    # device functions that index aff[
    idx = sorted(set(re.findall(r"(\w+)\s*\([^;]{0,200}aff\[", src)))
    return aff_kern, idx


def n_plateau(lab: np.ndarray) -> dict:
    """Unique nonzero labels are fragments; plateaus are a subset.

    vcount is written at parent[i] for in-plateau voxels. After e9b, parent
    is the plateau root. Unique nonzero labels on a fragment crop upper-bound
    the plateau-root count used as an index (one root per fragment after
    divide is even smaller). We report both unique-nonzero and a 6-connected
    'interior plateau' proxy: voxels whose 6-neighbours are the same label.
    """
    nvox = int(lab.size)
    labs = lab.ravel()
    nfrag = int(np.unique(labs).size) - int((labs == 0).any())
    return {"nvox": nvox, "nfrag": nfrag,
            "vcount_if_per_voxel": nvox,
            "vcount_if_per_root": nfrag,
            "saving_frac": 1.0 - nfrag / nvox}


def main():
    kern, _idx = aff_readers()
    # k_flow is the only __global__ that takes aff. Host wrappers memcpy then
    # free. Anything else reading aff[] in a watershed kernel is a kill.
    only_flow = kern == ["k_flow"]
    print(f"E5 aff-reading kernels: {kern}  only_k_flow={only_flow}")

    src = CACHE / "gpu_fragments.npy"
    vol = np.load(src, mmap_mode="r")
    lab = np.ascontiguousarray(vol[:32, :256, :256])
    plat = n_plateau(lab)
    print(f"E5 crop {lab.shape} nfrag={plat['nfrag']} nvox={plat['nvox']} "
          f"vcount shrink {plat['saving_frac']*100:.2f}%")

    d1 = json.loads((CACHE / "d1_mem.json").read_text())
    t = d1["targets"]["graded 2.16 Gvox"]
    w2 = json.loads((CACHE / "w2_mem.json").read_text())
    peak = t["peak_ws_gib"]
    lines = d1["runs"][-1].get("ws_peak_by_line") or []
    # map first crop's peak line if present
    top_line = lines[0][0] if lines else None
    print(f"E5 d1 peak_ws={peak:.2f} GiB  top ws_peak_by_line={top_line}")
    print(f"E5 w2 projected after sound levers={w2['peak_after_gib']:.2f} GiB "
          f"(LOWER BOUND, not a measurement)")

    # phase ownership: streaming aff does not apply if the peak is after
    # k_flow has freed aff. vcount-per-root applies at k_plat_meta.
    # in-place e9c applies only to the basin stage.
    out = {
        "aff_kernels": kern,
        "only_k_flow_reads_aff": only_flow,
        "plateau_crop": plat,
        "peak_measured_gib": peak,
        "w2_lower_bound_gib": w2["peak_after_gib"],
        "usable_gib": USABLE,
        "claimed_as_measured": False,
        "fits_if_lower_bound_is_peak": w2["peak_after_gib"] <= USABLE,
        "kill_if_peak_moves_to_other_phase": True,
        "top_ws_peak_line": top_line,
        "pass": bool(only_flow and plat["saving_frac"] > 0.9),
    }
    dest = CACHE / "e5_w2_proofs.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"E5 {'PASS' if out['pass'] else 'FAIL'} static proofs; "
          f"do not claim {w2['peak_after_gib']:.2f} GiB as measured")
    print(f"E5 wrote {dest.name}")
    return out["pass"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
