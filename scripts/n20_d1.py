#!/usr/bin/env python3
"""N20_D1: per-inner merge histogram. StarMerge EV.

ParHAC NeurIPS 2022 p.6 (papers/parhac_dhulipala2022.txt): full-graph update
is wasteful when rounds 'only merge a small number of vertices'.
Device fingerprint data/cache/p0aa_e6s.json is eps=0.08 on this RAG.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from n19_dead import stamp  # noqa: E402

CACHE = ROOT / "data/cache"
NOTE = ROOT / "notes/N20_D1.md"
OUT = CACHE / "N20_D1.json"
CLAIM = "N20 diagnostic; not a throughput claim"


def summarize(inner, layer_merges, nmerge, nlive0, tag):
    inner = np.asarray(inner, dtype=np.int64)
    pos = inner[inner > 0]
    med = float(np.median(pos)) if pos.size else 0.0
    p90 = float(np.percentile(pos, 90)) if pos.size else 0.0
    frac0 = float(layer_merges[0]) / max(nmerge, 1)
    small = int((pos < 0.01 * nlive0).sum()) if pos.size else 0
    return {
        "tag": tag,
        "n_inner": int(inner.size),
        "n_inner_pos": int(pos.size),
        "median_hm": med,
        "p90_hm": p90,
        "max_hm": int(pos.max()) if pos.size else 0,
        "layer0_merges": int(layer_merges[0]) if layer_merges else 0,
        "layer0_frac": frac0,
        "n_inner_lt_1pct_nlive": small,
        "star_ev": "dead" if frac0 >= 0.5 else "maybe",
    }


def main():
    pin = json.loads((CACHE / "p0aa_e6s.json").read_text())
    # Device E6s at eps=0.08. inner_merges not recorded on device; layer
    # totals are. 64 outers in layer 0, 1.32M merges.
    nlive0 = 1_853_545  # N19 I0 T=0.3 merges ~ remaining clusters from 2.17M
    layer_m = pin["layer_merges"]
    nmerge = int(pin["nmerge"])
    nouter = int(pin["nouter"])
    ninner = int(pin["ninner"])
    per_outer_l0 = layer_m[0] / max(pin["layer_outers"][0], 1)
    d008 = {
        "source": "p0aa_e6s.json device fingerprint",
        "eps": 0.08,
        "ninner": ninner,
        "nouter": nouter,
        "nmerge": nmerge,
        "layer_merges": layer_m,
        "layer_outers": pin["layer_outers"],
        "layer0_merges": layer_m[0],
        "layer0_frac": layer_m[0] / nmerge,
        "layer0_merges_per_outer": per_outer_l0,
        "mean_inners_per_outer": ninner / max(nouter, 1),
        "star_ev": "dead",
        "reason": (
            "layer 0 merges %.1f%% of all merges (%.0f / %d) over %d outers; "
            "not the paper 'small number of vertices' regime"
            % (100.0 * layer_m[0] / nmerge, layer_m[0], nmerge,
               pin["layer_outers"][0])
        ),
    }
    # eps=0.40 is the speed path (N19 I0). More merges per round, worse for
    # StarMerge. Reuse I0 merge count as a lower bound on aggressiveness.
    i0 = json.loads((CACHE / "N19_I0_REPRO.json").read_text())
    d040 = {
        "source": "N19_I0_REPRO.json",
        "eps": 0.40,
        "inner": int(i0["voi"]["m1"]["inner"]),
        "merges": int(i0["voi"]["m1"]["merges"]),
        "star_ev": "dead",
        "reason": (
            "speed-path eps=0.40 does %d merges in %d inners; denser than "
            "eps=0.08 layer-0 already-not-small"
            % (i0["voi"]["m1"]["merges"], i0["voi"]["m1"]["inner"])
        ),
    }
    go_star = False
    doc = {
        "claim": CLAIM,
        "d008": d008,
        "d040": d040,
        "N20_STAR": "EV-dead",
        "go_star": go_star,
        "nlive0_note": nlive0,
        "optimistic_agg_floor_ms": 840,
        "best_agg_ms": 1679.9,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N20 D1 StarMerge EV\n\n"
        f"{CLAIM}.\n\n"
        f"- eps=0.08 device: layer0 {layer_m[0]} / {nmerge} "
        f"({100.0 * layer_m[0] / nmerge:.1f}%) over "
        f"{pin['layer_outers'][0]} outers "
        f"({per_outer_l0:.0f} merges/outer).\n"
        f"- eps=0.40 I0: {d040['merges']} merges, {d040['inner']} inners.\n"
        "- N20_STAR: EV-dead. Do not rewrite CUDA. Paper p.6 does not apply.\n"
        "- Optimistic floor if hash/rebuild/fuse went to 0: ~840 ms agg "
        "(ATLAS), still not a new HAC.\n"
    )
    stamp("N20_D1", "StarMerge EV-dead: layer0 not small-merge", doc, "notes/N20_D1.md")
    stamp("N20_STAR", "EV-dead gated by N20_D1", {"go_star": False}, "notes/N20_D1.md")
    print(json.dumps(doc, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
