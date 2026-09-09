#!/usr/bin/env python3
"""M1: a byte-level cost model for the pipeline, calibrated against the one
timing measurement in this repo that is not contention noise.

Why this exists. Every speed conclusion in notes/LOG.md was drawn from wall
clock taken on a shared card, where a4_sync_cost.json measures the round-trip
quantum at 883 us contended against 2.55 us idle -- a 346x distortion that is
not uniform across phases, because it scales with the number of launches and
syncs a phase performs rather than with the work it does. Ranking phases by
those numbers ranks them by launch count. That is how "compact 1760 ms, sort
1378 ms, memset 1375 ms" happened: the memset phase moves 12 bytes per inner
iteration.

So this does not use the phase timings as costs at all. It builds the cost from
array widths and launch counts read out of csrc/parhac_d.cu, sums the bytes,
and divides by one free parameter -- achieved bandwidth -- which is then fixed
by requiring the model to reproduce a4_sync_cost.json's 667.58 ms. If the
implied efficiency lands at a plausible fraction of the 5090's 1792 GB/s the
model is trustworthy; if it lands above 100% or at 3% the model is wrong and
says so.

The projection to the graded volume rests on one structural fact: make_big.py
mirror-tiles 3x2x2, so [3,375,2400,2400] is twelve statistically identical
copies of the validation volume. TASK.md's own fragment and edge counts confirm
it (2175400 -> 26023852 is 11.96x, 7505458 -> 90323139 is 12.03x). Byte terms
therefore scale 12x while launch counts stay put, which is why the two are
tracked separately throughout.

Nothing here needs a GPU. Everything is derived from data/cache/*.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
GIB = 1024 ** 3
GB = 1e9

# Peak theoretical DRAM bandwidth. The 5090 figure is what the calibration is
# expressed against; the 3090 Ti figure is what TASK.md grades on.
BW_PEAK = {"5090": 1792.0e9, "3090Ti": 1008.0e9}

# A 4-byte random gather still moves a 32-byte sector. Every random-access term
# below is multiplied by this, and it is the single largest reason the union-find
# kernels are far off roofline.
SECTOR_AMP = 32.0 / 4.0

# Array widths, read off the declarations in csrc/parhac_d.cu.
W_UV = 4 + 4          # du, dv        uint32 pair
W_SM = 8              # dsm           double
W_CT = 8              # dct           int64
W_EDGE = W_UV + W_SM + W_CT
W_PARENT = 4          # dparent       uint32
W_SZ = 4              # dsz, dsz0     uint32
W_COLOR = 1           # dcolor, dfrozen uint8
W_PROP = 8            # dprop         unsigned long long
W_HSLOT = 16          # HSlot         {uint64 key, uint32 sm, uint32 ct}
W_SORTKEY = 8 + 8     # packed proposal key + payload

# TASK.md, "Hard numbers". Not derived, not rounded.
TASK = {
    "budget_s": 1.08,
    "nvox_graded": 2_160_000_000,
    "nvox_scaling": 1_440_000_000,
    "nvox_val": 180_000_000,
    "nfrag_val": 2_175_400,
    "nedge_val": 7_505_458,
    "nfrag_graded": 26_023_852,
    "nedge_graded": 90_323_139,
}


def load(name):
    p = CACHE / name
    if not p.is_file():
        raise SystemExit(f"M1 missing {p}; cannot build the model without it")
    return json.loads(p.read_text())


def counts():
    """Iteration counts and graph sizes for the single graded threshold 0.3.

    ninner/nouter/n_layer come from p0aa_e6s.json, which is the locked E6s path
    (a2_determinism.json corroborates ninner=1405 across two runs). The other
    T=0.3 files report 1840 and 1930 because they instrument different branches.
    """
    aa = load("p0aa_e6s.json")
    a3 = load("a3_work_accounting.json")
    pz = load("p0z_starmarge.json")
    ninner, nouter = aa["ninner"], aa["nouter"]
    return {
        "nnode": a3["nnode"],
        "nedge": a3["nedge"],
        "ninner": ninner,
        "nouter": nouter,
        "nlayer": aa["n_layer"],
        "nmerge": aa["nmerge"],
        "sum_nlive": aa["sum_nlive"],
        "mean_nlive": aa["sum_nlive"] / ninner,
        "above_frac": aa["above_frac"],
        "layer_outers": aa["layer_outers"],
        "layer_merges": aa["layer_merges"],
        # Every outer terminates on an inner that proposes nothing or merges
        # nothing and breaks, so that many inners never reach the compaction.
        "ncompact": ninner - nouter,
        "nprop_mean": pz["nprop_mean"],
        # The dirty-edge instrumentation that already exists under zprof:
        # k_mark_acc_blue marks merged blues and their receiving reds,
        # k_count_dirty_edges counts the live edges incident to either.
        "ndirty_mean": pz["ndirty_mean"],
        "ndirty_total": pz["ndirty_total"],
        "pz_ninner": pz["ninner"],
        # Active roots, i.e. what a root list would iterate instead of nnode.
        "nact_mean": pz["nact_mean"],
        "nact_p50": a3["nact_p50"],
        "listing_ceiling": a3["listing_speedup_ceiling"],
    }


def agg_terms(c, nnode, nedge, mean_nlive):
    """Bytes moved per launch, by kernel, then multiplied by launch count.

    Split into 'bytes' (scales with volume) and 'launch' (does not). Random
    terms carry SECTOR_AMP. Each entry is (label, total_bytes, nlaunch, scope).
    """
    ninner, nouter, nlayer = c["ninner"], c["nouter"], c["nlayer"]
    ncompact = c["ncompact"]
    above = c["above_frac"]
    nprop = c["nprop_mean"]
    # Path length after a k_compress is short; B3 in the log measured chains at
    # 1-2 hops, so one random hop per find is the right charge.
    chase = 1.0 * W_PARENT * SECTOR_AMP

    t = []

    # --- per inner ---------------------------------------------------------
    # k_propose streams the whole live edge array; only the above-TL fraction
    # pays for two union-find finds.
    t.append(("k_propose", (W_EDGE + 2 * chase * above) * mean_nlive, ninner, "inner"))
    # k_pack_prop_fused scans all nnode dprop slots to collect ~nprop winners.
    t.append(("k_pack_prop_fused", (W_PROP + W_SZ) * nnode, ninner, "inner"))
    # k_compress: coalesced read+write plus one random chase per node.
    t.append(("k_compress/inner", (2 * W_PARENT + chase) * nnode, ninner, "inner"))
    # k_freeze reads dcolor for every node and only chases for reds.
    t.append(("k_freeze", (W_COLOR + chase * nprop / nnode) * nnode, ninner, "inner"))
    # The proposal sort and its unpack are tiny in bytes and launch-bound.
    t.append(("sort+unpack", W_SORTKEY * 2 * nprop, ninner, "inner"))
    t.append(("k_accept_reds", (W_PARENT + W_SZ) * 2 * nprop, ninner, "inner"))

    # hash_combine_live, only on inners that actually merged. ntab_use is
    # next_pow2(2n+1024), so ~2n slots get scanned and the occupied ones
    # zeroed; emit writes one full edge record per surviving unique edge.
    ntab_use = 2.0 * mean_nlive
    combine = (
        (W_EDGE + W_UV + 1) * mean_nlive            # k_rewrite read + write
        + W_HSLOT * SECTOR_AMP * mean_nlive         # k_hash_insert random CAS
        + W_HSLOT * ntab_use                        # k_hash_emit table scan
        + W_EDGE * mean_nlive                       # k_hash_emit output
    )
    t.append(("hash_combine_live", combine, ncompact, "inner"))

    # --- per outer ---------------------------------------------------------
    t.append(("k_compress/outer", (2 * W_PARENT + chase) * nnode, nouter, "outer"))
    t.append(("k_zero_sz", W_SZ * nnode, nouter, "outer"))
    # k_rebuild_sz reads parent then hits a random atomicAdd on sz.
    t.append(("k_rebuild_sz", (W_PARENT + W_SZ * SECTOR_AMP) * nnode, nouter, "outer"))
    t.append(("memset dcolor+dfrozen", 2 * W_COLOR * nnode, nouter, "outer"))
    t.append(("k_color", (W_PARENT + W_COLOR) * nnode, nouter, "outer"))
    t.append(("k_copy_sz", 2 * W_SZ * nnode, nouter, "outer"))

    # --- per layer ---------------------------------------------------------
    t.append(("k_wmax_live", W_EDGE * nedge, nlayer, "layer"))
    # compact_radix: a CUB radix sort of 64-bit key/payload pairs is roughly
    # four read+write passes, plus the select and reduce-by-key around it.
    t.append(("compact_radix", 4 * 2 * W_SORTKEY * nedge + 3 * W_EDGE * nedge,
              nlayer, "layer"))

    return [(lbl, b * n, n, scope) for lbl, b, n, scope in t]


def launch_cost(c, us_per_launch, us_per_sync):
    """Kernels launched and syncs taken, neither of which scales with volume."""
    ninner, nouter, nlayer = c["ninner"], c["nouter"], c["nlayer"]
    # Counted off the launch sites in parhac_e6s_dev: 6 unconditional per inner
    # plus 3 more on the inners that merge, 7 per outer, 3 per layer.
    nlaunch = 6 * ninner + 3 * c["ncompact"] + 7 * nouter + 3 * nlayer
    # One D2H for nprop every inner, one for nmerge and one inside the combine
    # on the inners that get that far, two per layer inside compact_radix.
    nsync = ninner + 2 * c["ncompact"] + 2 * nlayer
    return nlaunch, nsync, (nlaunch * us_per_launch + nsync * us_per_sync) / 1e3


def calibrate(c, args):
    """Fix achieved bandwidth by making the model reproduce the idle figure."""
    a4 = load("a4_sync_cost.json")
    target_ms = a4["device_ms_median_by_probe"]["0"]
    terms = agg_terms(c, c["nnode"], c["nedge"], c["mean_nlive"])
    total_bytes = sum(b for _, b, _, _ in terms)
    nlaunch, nsync, ctl_ms = launch_cost(c, args.us_launch, args.us_sync)
    bytes_ms = target_ms - ctl_ms
    if bytes_ms <= 0:
        raise SystemExit("M1 launch overhead alone exceeds the measured time; "
                         "lower --us-launch/--us-sync")
    bw = total_bytes / (bytes_ms / 1e3)
    return {
        "target_ms": target_ms,
        "target_src": "a4_sync_cost.json device_ms_median_by_probe[0], T=0.3, val",
        "total_bytes": total_bytes,
        "nlaunch": nlaunch,
        "nsync": nsync,
        "control_ms": ctl_ms,
        "bytes_ms": bytes_ms,
        "achieved_bw": bw,
        "efficiency_5090": bw / BW_PEAK["5090"],
        "terms": terms,
    }


def project(c, cal, nvox, card):
    """Scale byte terms by the volume ratio, hold launch counts fixed."""
    r = nvox / TASK["nvox_val"]
    bw = BW_PEAK[card] * cal["efficiency_5090"]
    scaled = [(lbl, b * r, n, scope) for lbl, b, n, scope in cal["terms"]]
    bytes_ms = sum(b for _, b, _, _ in scaled) / bw * 1e3
    return {
        "ratio": r,
        "card": card,
        "bw": bw,
        "bytes_ms": bytes_ms,
        "control_ms": cal["control_ms"],
        "total_ms": bytes_ms + cal["control_ms"],
        "terms": sorted(scaled, key=lambda x: -x[1]),
    }


def stages(cal, c, card="3090Ti"):
    """The other three stages, from their own measurements.

    Watershed and extract have no byte model here; they are scaled from the
    idle val figures recorded in notes/LOG.md, which is weaker evidence than
    the agglomeration model and is flagged as such. RAG comes from
    c1_rag_ab.json, which is a CUDA-event median around the kernels.
    """
    ratio = TASK["nvox_graded"] / TASK["nvox_val"]
    bwr = BW_PEAK["5090"] / BW_PEAK[card]
    rag_val = load("c1_rag_ab.json")["work"]["median_ms"]
    return {
        # LOG.md, idle 5090 val: ws 536 ms (of which UF 225), extract ~175 ms
        # via the host stage timer; the isolated extract kernel measured 3.4 ms,
        # so the stage figure is dominated by allocation and is not scaled.
        "watershed": {"val_ms": 536.0, "graded_ms": 536.0 * ratio * bwr,
                      "src": "LOG.md idle 5090 val, weak", "budget_ms": 350.0},
        "rag": {"val_ms": rag_val, "graded_ms": rag_val * ratio * bwr,
                "src": "c1_rag_ab.json CUDA-event median", "budget_ms": 180.0},
        "agglomeration": {"val_ms": cal["target_ms"], "graded_ms": None,
                          "src": cal["target_src"], "budget_ms": 400.0},
        "extract": {"val_ms": 3.434, "graded_ms": 3.434 * ratio * bwr,
                    "src": "LOG.md extract_gpu device_ms", "budget_ms": 70.0},
    }


def roofline(card="3090Ti"):
    """Essential traffic at 2.16 Gvox: what the work cannot be done without.

    The point of this is to separate "too slow because physics" from "too slow
    because the code does 2058 full passes over 26M-element arrays". Only the
    second kind is fixable, so it matters a great deal which one this is.
    """
    n = TASK["nvox_graded"]
    nedge, nfrag = TASK["nedge_graded"], TASK["nfrag_graded"]
    items = [
        ("affinity read, perfect reuse", 3 * n),
        ("direction field, k_flow write + ~15 later passes", 16 * 1 * n),
        # The plan costed this line as block labels at 0.5 B/vox. W1 is
        # unsound (see w0_ws_ref.py --blocks and notes/LOG.md), so the
        # union-find keeps a 4 B/vox parent array and this is the honest
        # figure. It is what makes the watershed the dominant essential term.
        ("union-find parent 4 B/vox, 14 rounds r+w", 14 * 2 * 4 * n),
        ("label compaction scan + uint32 label write", 4 * n + 4 * n),
        ("RAG: seg reads + affinity + hash", 1.5 * 4 * n + 3 * n + 64 * nedge),
        ("agglomeration: 17 layer passes over the edge list", 17 * 16 * nedge),
        ("agglomeration: active/dirty inner work", 15e9),
        ("extract: seg read + label write", 4 * n + 4 * n),
    ]
    total = sum(b for _, b in items)
    budget = TASK["budget_s"] * BW_PEAK[card] * 0.84
    return {"items": items, "total": total, "budget": budget,
            "frac": total / budget, "card": card}


def ws_model(card="3090Ti", plateau_rounds=7, basin_rounds=7,
             plateau_frac=0.356):
    """A byte model for the watershed, which had none.

    The watershed is 90% of the projected runtime, so scaling one idle val
    number by 12x and a bandwidth ratio is not good enough to decide
    feasibility on. This costs the stage the same way the agglomeration model
    costs its kernels: essential traffic per pass, times the number of passes
    the algorithm actually performs.

    The decisive term is the union-find. Each round of k_hook_bidir /
    k_hook_remain reads the direction byte and its own parent slot, then
    gathers parent[j] for up to six neighbours; k_uf_compress_c then walks and
    rewrites chains. Whether those neighbour gathers hit cache is the whole
    question, and it is a function of tiling, not of the label representation:

      untiled   a parent z-plane at 2.16 Gvox is 4 B x 2400 x 2400 = 23.04 MB,
                so on a 3090 Ti's 6 MB L2 the +/-z gathers always miss, and
                every round pays a cold 4 B read per neighbour direction.
      xy-tiled  a 512x512 patch over the 3-plane window the gathers need is
                512*512*3*4 = 3.15 MB and fits in 6 MB. Then only the tile
                halo crosses L2 and the neighbour gathers are hits.

    Note what that means for W1. The plan justified block labels as the
    prerequisite for L2 residency, claiming a block-label z-plane-pair is
    5.76 MB and fits. It does not: the hook kernels need three planes, which is
    17.28 MB even with block labels. Tiling is what buys residency, and tiling
    works on the 4 B/vox array directly. W1 was never load-bearing for L2.
    """
    n = TASK["nvox_graded"]
    bw = BW_PEAK[card] * 0.84
    rounds = plateau_rounds + basin_rounds

    def ms(b):
        return b / bw * 1e3

    # Per voxel per round, in bytes.
    hook_untiled = 1 + 4 + 6 * 4 + 4      # bits, own parent, 6 gathers, write
    hook_tiled = 1 + 4 + 4                # gathers become L2 hits
    comp = 4 + 4                          # read chain head, write root

    terms_untiled = [
        ("k_flow: affinity read + bits write", 3 * n + 1 * n, 1),
        ("hook rounds, cold neighbour gathers", hook_untiled * n, rounds),
        ("compress rounds", comp * n, rounds),
        ("divide: corner flag, vcount, BFS", (1 + 4 + 4 + 8) * n, 1),
        ("label: root flag, scan, write", (4 + 4 + 4) * n, 1),
    ]
    # W3 restricted to what is provably the same kernel. k_hook_bidir skips a
    # voxel with no direction bits and only hooks reciprocal edges, so a voxel
    # with no reciprocal edge issues no hook at all: launching over the
    # compacted list of voxels that have one is definitionally identical, the
    # same argument that makes g3's active-edge list exact. That is 35.6% of
    # the volume on val. It applies to the plateau half only -- after the
    # divide almost every voxel carries a bit, so the basin hook has no such
    # list to restrict to.
    terms_tiled = [
        ("k_flow: affinity read + bits write", 3 * n + 1 * n, 1),
        ("plateau hook, tiled + plateau list (W3+W4)",
         hook_tiled * n * plateau_frac, plateau_rounds),
        ("basin hook, tiled (W4)", hook_tiled * n, basin_rounds),
        ("compress rounds", comp * n, rounds),
        ("divide: corner flag, vcount over plateau roots (W2/W3)",
         (1 + 4 + 8) * n * plateau_frac, 1),
        # flag as a bitmask rather than a 4 B/vox array: k_root_flag writes
        # 1 bit/vox, the scan runs over per-word popcounts (n/32 entries), and
        # k_write_labels recovers psum[i] from the word offset plus a popcount
        # of the bits below i. Same three touches, 32x less traffic on two of
        # them.
        ("label: bitmask flag + popcount scan + write (W3)",
         (1 / 8 + 2 * 4 / 32 + 1 / 8) * n + 4 * n, 1),
    ]
    # W5, measured on the real fragment crop (scripts/w5_tile_uf.py,
    # 8.19 Mvox, tree graph, 8x16x32 tile): stitch rounds stay at 5 against
    # a baseline of 5. The win is the domain, not the count. Phase 1 is one
    # shared-memory pass that writes a parent per voxel; the stitch hook
    # covers face voxels (0.38 of the volume); the stitch compress covers
    # faces plus phase-1 roots (0.46); one full-volume flatten finishes.
    # Two union-finds (plateau, basin). Compress traffic drops to
    # 0.46 * 5 + 1 = 3.3 volume-passes against the current 14.
    face_frac, comp_frac, stitch_r = 0.382, 0.456, 5
    terms_w5 = [
        ("k_flow: affinity read + bits write", 3 * n + 1 * n, 1),
        ("W5 phase-1 tile-local UF, two halves", 5 * n, 2),
        ("W5 stitch hook over faces", hook_tiled * n * face_frac, stitch_r * 2),
        ("W5 stitch compress over face+roots", comp * n * comp_frac, stitch_r * 2),
        ("W5 final flatten", comp * n, 2),
        ("divide: corner flag, vcount over plateau roots (W2/W3)",
         (1 + 4 + 8) * n * plateau_frac, 1),
        ("label: bitmask flag + popcount scan + write (W3)",
         (1 / 8 + 2 * 4 / 32 + 1 / 8) * n + 4 * n, 1),
    ]

    def total(terms):
        return sum(b * k for _, b, k in terms)

    tu, tt, tw = total(terms_untiled), total(terms_tiled), total(terms_w5)
    return {"card": card, "rounds": rounds,
            "untiled_bytes": tu, "tiled_bytes": tt, "w5_bytes": tw,
            "untiled_ms": ms(tu), "tiled_ms": ms(tt), "w5_ms": ms(tw),
            "terms_untiled": terms_untiled, "terms_tiled": terms_tiled,
            "terms_w5": terms_w5,
            "budget_ms": 350.0}


def levers(c, cal, nvox, card="3090Ti"):
    """What each planned change removes, expressed against the projection.

    Every reduction factor here is a ratio of logical work counters measured by
    scripts/g0_agg_ref.py, which runs a CPU replica of this exact loop on the
    real val RAG and checks the restructured version against it for bit
    equality. Nothing in this table is a guess about how much work a list
    saves; it is the count of visits each version actually performed.

    That matters because the first version of this table used p0z_starmarge's
    ndirty_mean and claimed 74x for g2. The measured figure is ~8x. The gap is
    that p0z reports the blue-side dirty count in ndirty and the red-side count
    in nstar, and the red side is nine times larger -- a receiving red's
    incidence row has to be rescanned too, and rows grow as clusters merge.
    """
    base = project(c, cal, nvox, card)
    by = {lbl: b for lbl, b, _, _ in base["terms"]}
    bw = base["bw"]
    ms = lambda b: b / bw * 1e3
    nnode, nprop = c["nnode"], c["nprop_mean"]
    nedge = c["nedge"]
    r = nvox / TASK["nvox_val"]
    W = measured_work()

    def ratio(key, fallback):
        """Fraction of the base work the fast path still does."""
        if not W:
            return fallback, "estimated"
        b, f = W["base"][key], W["fast"][key]
        return (f / b if b else fallback), f"measured {b / 1e6:.0f}M -> {f / 1e6:.1f}M"

    out = []
    # g1: k_freeze over nnode -> k_freeze_reds over the unique reds, which also
    # retires k_color and the dcolor clear because nothing else reads them.
    keep, src = ratio("freeze_node_visits", nprop / nnode)
    out.append(("g1 freeze_reds (+color)",
                (ms(by["k_freeze"]) + ms(by["k_color"])
                 + 0.5 * ms(by["memset dcolor+dfrozen"])) * (1 - keep),
                f"{src}; k_propose ignores dcolor, so k_color dies with it"))

    # g2: hash_combine_live over every live edge -> over the incidence rows of
    # the merged blues and their receiving reds. Two residual costs are charged
    # back or the lever is a fiction: the rows still have to be rewritten and
    # merged, and the CSR index has to be built once per layer, both directions.
    keep, src = ratio("compact_edge_visits", 0.12)
    dirty_work = (W["fast"]["ndirty_total"] if W else 0) \
        * (W_EDGE + W_HSLOT * SECTOR_AMP) * r
    # The CUDA G2 path is a scan of current endpoints (k_rewrite_dirty),
    # not a CSR. A CSR built at layer entry misses edges rewritten onto a
    # dirty root in an earlier inner unless it is spliced; the scan does
    # not have that bug. Charge the scan as residual, not a transpose sort
    # the implementation does not run.
    scan_vis = (W["fast"]["dirty_scan_visits"] if W
                else c["ncompact"] * c["mean_nlive"])
    scan = scan_vis * (W_UV + W_CT + 2) * r
    g2_save = ms(by["hash_combine_live"]) - ms(dirty_work) - ms(scan)
    out.append(("g2 dirty-set compaction",
                g2_save,
                f"{src} = {1 / keep:.1f}x; residual scan {ms(scan):.0f} ms "
                f"+ dirty hash {ms(dirty_work):.0f} ms"))

    # g3: propose over the above-TL list, refreshed once per layer; pack over
    # the candidate blues instead of all nnode dprop slots.
    keep, src = ratio("propose_edge_visits", c["above_frac"])
    out.append(("g3 active edge list", ms(by["k_propose"]) * (1 - keep), src))
    keep, src = ratio("pack_node_visits", nprop / nnode)
    out.append(("g3 candidate blue list",
                ms(by["k_pack_prop_fused"]) * (1 - keep), src))

    # g4: the seven per-outer full-nnode passes over a root list, and the
    # per-inner compress restricted to the accepted blues.
    outer_terms = ["k_compress/outer", "k_zero_sz", "k_rebuild_sz",
                   "k_copy_sz"]
    keep, src = ratio("outer_node_visits", 1 / c["listing_ceiling"])
    out.append(("g4 per-outer root list",
                sum(ms(by[k]) for k in outer_terms) * (1 - keep),
                f"{src}; a3 ceiling {c['listing_ceiling']:.2f}x"))
    keep, src = ratio("compress_node_visits", nprop / nnode)
    out.append(("g4 per-inner compress",
                ms(by["k_compress/inner"]) * (1 - keep), src))
    return base, out


def measured_work():
    """Logical work counters from the CPU replica, if it has been run.

    Refused unless the replica ran base and fast to convergence on the whole
    val graph and they agreed. A subgraph or a truncated run has different
    visit ratios -- the per-outer sweeps are O(nnode) while the proposals are
    not -- so using one here would quietly mis-size every lever.
    """
    p = CACHE / "g0_agg_ref.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    if not d.get("pass") or "base" not in d or "fast" not in d:
        return None
    # Validated against the device rather than against a metadata field: a
    # replica that reproduces p0aa's sum_nlive and merge count to the digit ran
    # the whole val graph to convergence at T=0.3, and one that does not is a
    # subgraph or a truncated run whose visit ratios would mis-size every lever.
    ref = load("p0aa_e6s.json")
    b = d["base"]
    if (b.get("sum_nlive") != ref["sum_nlive"]
            or b.get("nmerge") != ref["nmerge"]
            or b.get("ninner") != ref["ninner"]):
        print(f"M1   note: g0_agg_ref.json is not a converged full-val run "
              f"(sum_nlive={b.get('sum_nlive')} vs {ref['sum_nlive']}); "
              f"lever factors fall back to estimates")
        return None
    return {"base": b["work"], "fast": d["fast"]["work"]}


def eps_layers(c, eps_values):
    """Layers as a function of eps, calibrated so eps=0.08 reproduces 17.

    TL = wmax/(1+eps) clamped at T, so the layer count is
    ln(wmax0/T)/ln(1+eps) with wmax0 at the top of the affinity byte range.
    The measured 17 against a predicted 15.6 gives the calibration.
    """
    thr = 0.3
    span = math.log(1.0 / thr)
    pred = span / math.log(1.0 + 0.08)
    k = c["nlayer"] / pred
    per_layer_outers = c["nouter"] / c["nlayer"]
    return [{"eps": e,
             "layers": span / math.log(1.0 + e) * k,
             "outers": span / math.log(1.0 + e) * k * per_layer_outers}
            for e in eps_values], k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--us-launch", type=float, default=3.0,
                    help="kernel launch cost on an idle card, microseconds")
    ap.add_argument("--us-sync", type=float, default=6.0,
                    help="D2H round trip on an idle card, microseconds")
    ap.add_argument("--card", default="3090Ti", choices=sorted(BW_PEAK))
    args = ap.parse_args()

    c = counts()
    print("M1 graph and iteration counts, threshold 0.3, val 180 Mvox")
    print(f"M1   nnode={c['nnode']}  nedge={c['nedge']}  "
          f"merges={c['nmerge']}")
    print(f"M1   layers={c['nlayer']}  outers={c['nouter']}  "
          f"inners={c['ninner']}  compactions={c['ncompact']}")
    print(f"M1   mean nlive={c['mean_nlive']:.0f}  "
          f"above_frac={c['above_frac']:.4f}  "
          f"mean nprop={c['nprop_mean']:.0f}  mean ndirty={c['ndirty_mean']:.0f}")
    lm, lo = c["layer_merges"], c["layer_outers"]
    print(f"M1   layer 0 alone does {lm[0]} of {sum(lm)} merges "
          f"({lm[0] / sum(lm) * 100:.0f}%) in {lo[0]} outers (the cap)")

    cal = calibrate(c, args)
    print(f"\nM1 calibration against {cal['target_src']}")
    print(f"M1   measured               {cal['target_ms']:8.1f} ms")
    print(f"M1   model bytes            {cal['total_bytes'] / GB:8.1f} GB")
    print(f"M1   {cal['nlaunch']} launches + {cal['nsync']} syncs "
          f"= {cal['control_ms']:.1f} ms of control")
    print(f"M1   implied achieved bw    {cal['achieved_bw'] / GB:8.1f} GB/s "
          f"= {cal['efficiency_5090'] * 100:.1f}% of 5090 peak")
    eff = cal["efficiency_5090"]
    verdict = ("plausible" if 0.15 <= eff <= 0.75 else
               "IMPLAUSIBLE - model is wrong, do not trust what follows")
    print(f"M1   verdict                {verdict}")

    print(f"\nM1 agglomeration cost at val, by term "
          f"({cal['total_bytes'] / GB:.1f} GB total)")
    for lbl, b, n, scope in sorted(cal["terms"], key=lambda x: -x[1]):
        print(f"M1   {b / cal['total_bytes'] * 100:5.1f}%  "
              f"{b / GB:7.2f} GB  {b / cal['achieved_bw'] * 1e3:7.1f} ms  "
              f"x{n:<5d} {scope:5s} {lbl}")

    rf = roofline(args.card)
    print(f"\nM1 roofline at 2.16 Gvox on a {rf['card']}: is this physics or code?")
    for lbl, b in rf["items"]:
        print(f"M1   {b / GB:7.1f} GB  {lbl}")
    print(f"M1   {rf['total'] / GB:7.1f} GB essential total")
    print(f"M1   {rf['budget'] / GB:7.1f} GB budget "
          f"({TASK['budget_s']:.2f} s x 84% of peak)")
    print(f"M1   essential traffic is {rf['frac'] * 100:.0f}% of budget -> "
          f"{'NOT bandwidth-bound; the gap is work efficiency' if rf['frac'] < 0.6 else 'bandwidth-bound'}")

    st = stages(cal, c, args.card)
    for nvox, tag in ((TASK["nvox_graded"], "graded 2.16 Gvox"),
                      (TASK["nvox_scaling"], "scaling 1.44 Gvox")):
        pr = project(c, cal, nvox, args.card)
        st_here = dict(st)
        st_here["agglomeration"] = dict(st["agglomeration"],
                                        graded_ms=pr["total_ms"])
        print(f"\nM1 {tag} on a {args.card}, x{pr['ratio']:.2f} bytes, "
              f"launches held fixed")
        tot = 0.0
        for name in ("watershed", "rag", "agglomeration", "extract"):
            s = st_here[name]
            g = s["graded_ms"] * (nvox / TASK["nvox_graded"]) \
                if name != "agglomeration" else s["graded_ms"]
            tot += g
            gap = g / s["budget_ms"]
            print(f"M1   {name:14s} {g:9.1f} ms  budget {s['budget_ms']:6.1f} ms  "
                  f"{gap:6.1f}x   [{s['src']}]")
        print(f"M1   {'TOTAL':14s} {tot:9.1f} ms  "
              f"budget {TASK['budget_s'] * 1e3:6.1f} ms  "
              f"{tot / (TASK['budget_s'] * 1e3):6.1f}x")
        print(f"M1   throughput {nvox / (tot / 1e3) / 1e9:.3f} Gvox/s "
              f"vs 2.000 required")
        if nvox == TASK["nvox_graded"]:
            print(f"\nM1 agglomeration terms at {tag}, top 8")
            for lbl, b, n, scope in pr["terms"][:8]:
                print(f"M1   {b / pr['bw'] * 1e3:8.1f} ms  {b / GB:7.1f} GB  "
                      f"x{n:<5d} {scope:5s} {lbl}")

    base, lv = levers(c, cal, TASK["nvox_graded"], args.card)
    print(f"\nM1 bit-identical lever payoffs at 2.16 Gvox on a {args.card}")
    print(f"M1   agglomeration before {base['total_ms']:.0f} ms")
    saved = 0.0
    for name, s, why in lv:
        saved += s
        print(f"M1   -{s:8.1f} ms  {name:32s} {why}")
    after = base["total_ms"] - saved
    print(f"M1   agglomeration after  {after:.0f} ms "
          f"({base['total_ms'] / max(after, 1e-9):.1f}x) "
          f"vs 400 ms budget")

    # Bottom line, with the V-lever factor taken from the measured sweep if it
    # has been run rather than from the layer-count algebra below.
    vsweep = CACHE / "v_levers.json"
    if not vsweep.is_file():
        vsweep = CACHE / "v_levers_sub.json"
    vfac, vsrc = 1.0, "none"
    if vsweep.is_file():
        vr = json.loads(vsweep.read_text())["rows"]
        b0 = vr[0]["sum_nlive"]
        best = min(vr, key=lambda r: r["sum_nlive"])
        vfac = b0 / best["sum_nlive"] if best["sum_nlive"] else 1.0
        vsrc = (f"{vsweep.name} {best['tag']} eps={best['eps']:.2f}"
                f"{'' if best['size_asym'] else ' no-asym'}")
    wm = ws_model(args.card)
    print(f"\nM1 watershed byte model at 2.16 Gvox on a {args.card}, "
          f"{wm['rounds']} union-find rounds")
    for lbl, b, k in wm["terms_untiled"]:
        print(f"M1   {b * k / wm['untiled_bytes'] * wm['untiled_ms']:8.1f} ms  "
              f"{b * k / GB:7.1f} GB  x{k:<3d} {lbl}")
    print(f"M1   untiled total   {wm['untiled_ms']:8.1f} ms   "
          f"({wm['untiled_bytes'] / GB:.0f} GB)")
    print(f"M1   xy-tiled + W3   {wm['tiled_ms']:8.1f} ms   "
          f"({wm['tiled_bytes'] / GB:.0f} GB)  vs {wm['budget_ms']:.0f} ms budget")
    print(f"M1   the scaled-measurement figure was "
          f"{st['watershed']['graded_ms']:.0f} ms, so the byte model says the "
          f"stage is {st['watershed']['graded_ms'] / wm['untiled_ms']:.1f}x off "
          f"its own roofline even untiled")

    st2, agg_v = st, after / vfac
    tot2 = (st2["watershed"]["graded_ms"] + st2["rag"]["graded_ms"]
            + agg_v + st2["extract"]["graded_ms"])
    print(f"\nM1 end to end at 2.16 Gvox on a {args.card}, "
          f"all G levers plus the V levers")
    print(f"M1   V-lever work factor  {vfac:.2f}x  [{vsrc}]")
    print(f"M1   watershed      {st2['watershed']['graded_ms']:9.0f} ms  "
          f"{st2['watershed']['graded_ms'] / tot2 * 100:4.1f}% of total "
          f"-- as it stands today")
    print(f"M1   rag            {st2['rag']['graded_ms']:9.0f} ms")
    print(f"M1   agglomeration  {agg_v:9.0f} ms  "
          f"({base['total_ms'] / max(agg_v, 1e-9):.1f}x from {base['total_ms']:.0f})")
    print(f"M1   extract        {st2['extract']['graded_ms']:9.0f} ms")
    print(f"M1   TOTAL          {tot2:9.0f} ms = "
          f"{TASK['nvox_graded'] / (tot2 / 1e3) / 1e9:.3f} Gvox/s, "
          f"{tot2 / (TASK['budget_s'] * 1e3):.1f}x over budget")

    # The best case that every identified sound change can deliver: the
    # watershed at its own tiled essential traffic, the RAG at the r1 target,
    # agglomeration at the measured lever stack plus the V levers.
    rag_r1 = st2["rag"]["budget_ms"]
    ws_best = wm["w5_ms"]
    best = ws_best + rag_r1 + agg_v + st2["extract"]["graded_ms"]
    print(f"\nM1 best case if every sound change lands perfectly")
    print(f"M1   watershed      {ws_best:9.0f} ms  "
          f"W5 tile-local UF + stitch (was {wm['tiled_ms']:.0f} tiled-only)")
    print(f"M1   rag            {rag_r1:9.0f} ms  r1 target")
    print(f"M1   agglomeration  {agg_v:9.0f} ms  G levers + V levers")
    print(f"M1   extract        {st2['extract']['graded_ms']:9.0f} ms")
    print(f"M1   TOTAL          {best:9.0f} ms = "
          f"{TASK['nvox_graded'] / (best / 1e3) / 1e9:.3f} Gvox/s, "
          f"{best / (TASK['budget_s'] * 1e3):.2f}x of budget")
    # Where the residual actually sits, without leaning on the plan's own
    # per-stage budget split, which is an allocation rather than a constraint.
    budget = TASK["budget_s"] * 1e3
    gap = best / budget
    floor = ws_best + rag_r1 + st2["extract"]["graded_ms"]
    agg_allowed = budget - floor
    print(f"M1 verdict")
    print(f"M1   best case {best:.0f} ms against a {budget:.0f} ms budget, "
          f"{gap:.2f}x the budget at "
          f"{TASK['nvox_graded'] / (best / 1e3) / 1e9:.3f} Gvox/s")
    print(f"M1   W5 does not cut union-find rounds (measured 5 stitch vs 5 "
          f"baseline on the real fragment crop). It cuts the compress domain "
          f"to ~46% of the volume plus one flatten, which is why the "
          f"watershed moves from {wm['tiled_ms']:.0f} to {ws_best:.0f} ms.")
    print(f"M1   agglomeration is {agg_v:.0f} ms against the watershed's "
          f"{ws_best:.0f} ms.")
    if agg_allowed <= 0:
        print(f"M1   the watershed, RAG and extract alone come to "
              f"{floor:.0f} ms, over budget, so 2 Gvox/s is excluded even "
              f"with agglomeration free.")
    elif agg_v <= agg_allowed:
        print(f"M1   watershed + RAG + extract at their achievable figures is "
              f"{floor:.0f} ms, which leaves agglomeration {agg_allowed:.0f} "
              f"ms.")
        print(f"M1   the measured G-lever stack plus the V levers gets it to "
              f"{agg_v:.0f} ms, which fits. On this model the 2 Gvox/s gate "
              f"is reachable.")
        print(f"M1   that is not a claim the gate is hit: V changes nseg "
              f"(v1+v2 eps=0.32 is {vsrc}), so it still owes four-threshold "
              f"VOI; W5/R1 are byte models; no G lever has run on a device. "
              f"The next fact is an idle-card e3 --gpu-window.")
    else:
        print(f"M1   watershed + RAG + extract at their achievable figures is "
              f"{floor:.0f} ms, which leaves agglomeration {agg_allowed:.0f} "
              f"ms.")
        print(f"M1   the measured lever stack plus the V levers gets it to "
              f"{agg_v:.0f} ms, so the gate needs a further "
              f"{agg_v / agg_allowed:.2f}x in agglomeration that is not "
              f"identified, or fewer union-find rounds in the watershed.")
        print(f"M1   both are algorithm changes rather than better "
              f"implementations of what is there, which is why 2 Gvox/s is "
              f"not reachable from this design.")

    rows, k = eps_layers(c, [0.08, 0.12, 0.16, 0.24, 0.32])
    print(f"\nM1 v1 eps lever: layers = ln(1/0.3)/ln(1+eps) x {k:.3f} "
          f"(calibrated to the measured 17)")
    for r in rows:
        print(f"M1   eps={r['eps']:.2f}  layers={r['layers']:5.1f}  "
              f"outers={r['outers']:6.0f}  "
              f"work x{rows[0]['layers'] / r['layers']:.2f}")
    print("M1   caveat: layer 0 already does "
          f"{lm[0] / sum(lm) * 100:.0f}% of merges at the outer cap, so a wider "
          "first band may need more outers, not fewer. The layer-count cut is "
          "an upper bound on the benefit and v1 must measure outers, not assume.")

    report = {
        "counts": c,
        "calibration": {k2: v for k2, v in cal.items() if k2 != "terms"},
        "val_terms": [{"label": l, "bytes": b, "nlaunch": n, "scope": s}
                      for l, b, n, s in sorted(cal["terms"], key=lambda x: -x[1])],
        "roofline": {"items": [{"label": l, "bytes": b} for l, b in rf["items"]],
                     "total_bytes": rf["total"], "budget_bytes": rf["budget"],
                     "frac_of_budget": rf["frac"], "card": rf["card"]},
        "projection": {},
        "levers": [{"name": n, "saves_ms": s, "why": w} for n, s, w in lv],
        "agg_ms_after_levers": after,
        "eps_rows": rows,
    }
    for nvox, tag in ((TASK["nvox_graded"], "graded 2.16 Gvox"),
                      (TASK["nvox_scaling"], "scaling 1.44 Gvox")):
        pr = project(c, cal, nvox, args.card)
        report["projection"][tag] = {
            "nvox": nvox, "card": args.card, "ratio": pr["ratio"],
            "agg_ms": pr["total_ms"],
            "stage_ms": {n: (s["graded_ms"] * (nvox / TASK["nvox_graded"])
                                 if n != "agglomeration" else pr["total_ms"])
                         for n, s in stages(cal, c, args.card).items()
                         if s["graded_ms"] is not None or n == "agglomeration"},
        }
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / "m1_cost_model.json"
    dest.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\nM1 wrote {dest.name}")
    return 0.15 <= eff <= 0.75



if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
