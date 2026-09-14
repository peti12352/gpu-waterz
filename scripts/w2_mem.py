#!/usr/bin/env python3
"""W2: can the z-slab requirement actually be deleted?

The plan's claim is that three changes take the watershed peak from 42.22 GiB
to "~23 GiB, under the 24 GiB card", which would retire chunking entirely --
the one item the log calls mandatory but unfinished. Two things went wrong with
that claim and this script recomputes it from data/cache/d1_mem.json rather
than restating it.

First, the arithmetic. The plan's own three savings are block labels -3.5,
vcount -4 and a uint32 BFS queue -1.43, so -8.93 B/vox against a measured
13.988. That leaves 5.058 B/vox of scratch = 10.18 GiB at 2.16 Gvox, and the
peak is scratch plus the affinity input plus the label buffer:

    6.03 (aff) + 8.05 (seg) + 10.18 (scratch) = 24.26 GiB

which is over 24 GiB, not under it, before allowing any headroom for the
driver's own context. The plan's "~23 GiB" does not follow from its own list.

Second, and worse, the -3.5 B/vox from block labels is not available at all.
scripts/w0_ws_ref.py shows that contracting each 2x2x2 block to a single label
is unsound for this watershed, it collapses the whole volume into one or two
components in 40 of 40 checks, and the sound variant needs one slot per
intra-block connectivity class, which the same sweep measures at a mean of
2.3-5.6 and a maximum of 8. Eight slots of uint32 is 4 B/vox, exactly what the
per-voxel array already costs. So W1 yields no memory saving, and W2 has to
find its budget somewhere else.

What this script does is enumerate the levers that are actually sound, note
which phase each one applies to, and report whether 24 GiB is reachable.
"""
from __future__ import annotations

import json
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
GIB = 1 << 30
CARD_GIB = 24.0
# A 24 GiB card does not offer 24 GiB. Driver context plus CUDA's own
# allocator overhead is ~0.5-1.0 GiB, so anything above this is not a fit.
USABLE_GIB = 23.0


def main() -> int:
    d = json.loads((CACHE / "d1_mem.json").read_text())
    t = d["targets"]["graded 2.16 Gvox"]
    nvox = t["nvox"]
    bpv = d["ws_b_per_vox"]

    aff, seg, scratch = t["aff_gib"], t["seg_gib"], t["ws_scratch_gib"]
    peak = t["peak_ws_gib"]
    print(f"W2 measured at 2.16 Gvox, from d1_mem.json")
    print(f"W2   affinity input   {aff:6.2f} GiB   3 B/vox")
    print(f"W2   label buffer     {seg:6.2f} GiB   4 B/vox")
    print(f"W2   ws scratch       {scratch:6.2f} GiB   {bpv:.3f} B/vox")
    print(f"W2   peak             {peak:6.2f} GiB   vs {USABLE_GIB} usable "
          f"on a {CARD_GIB:.0f} GiB card")

    def gib(b_per_vox: float) -> float:
        return b_per_vox * nvox / GIB

    # Each lever: label, GiB saved, whether it is sound, and why.
    levers = [
        ("W1 block labels (-3.5 B/vox)", gib(3.5), False,
         "unsound: one label per 2x2x2 block over-merges in 40/40 w0 checks, "
         "and the sound k-slot form needs k up to 8 = 4 B/vox = no saving"),
        ("stream the affinity through k_flow", aff, True,
         "k_flow is the affinity's only consumer, so it never needs to be "
         "resident alongside the union-find scratch; upload it in z-slabs and "
         "the input stops counting toward the peak"),
        ("uint32 BFS queue (-1.43 B/vox)", gib(1.43), True,
         "q holds voxel indices and e9b_divide_d already refuses volumes "
         "above 2^32 voxels, so int64 is dead weight"),
        ("vcount over plateau roots (-4 B/vox)", gib(4.0), True,
         "vcount is only read as vcount[plat_root[p]] in k_plat_meta, so it "
         "needs one entry per plateau, not one per voxel"),
        ("in-place scan in e9c (-4 B/vox)", gib(4.0), True,
         "e9c holds flag and psum as separate per-voxel arrays; e9b already "
         "scans flag into itself and recovers the predicate from the scan's "
         "differences via k_scatter_idx_u32, so the same trick applies"),
    ]

    print(f"\nW2 levers")
    live = 0.0
    for name, saved, sound, why in levers:
        mark = "ok  " if sound else "DEAD"
        if sound:
            live += saved
        print(f"W2   {mark} {saved:6.2f} GiB  {name}")
        print(f"W2        {why}")

    after = peak - live
    print(f"\nW2 peak after the sound levers  {after:6.2f} GiB")
    verdict = "FITS" if after <= USABLE_GIB else "DOES NOT FIT"
    print(f"W2   {verdict} in {USABLE_GIB} GiB usable")
    print(f"W2   plan's own arithmetic, for comparison: "
          f"{aff + seg + gib(bpv - 8.93):.2f} GiB, which is already over 24")

    # The two levers above that touch different phases cannot both be assumed
    # to hit the peak. ws_mem_peak_lines exists to say which line owns it.
    print(f"\nW2 caveat that decides the verdict")
    print(f"W2   'stream the affinity' and 'in-place scan in e9c' apply to "
          f"different phases, and only the phase that owns the peak counts.")
    print(f"W2   ws_mem_peak_lines() in csrc/ws.cu attributes the peak to a "
          f"source line; scripts/d1_mem.py already reads it. Until that is")
    print(f"W2   run, treat {after:.2f} GiB as a lower bound on what the "
          f"levers achieve, not a measurement.")

    out = {"nvox": nvox, "peak_measured_gib": peak,
           "aff_gib": aff, "seg_gib": seg, "scratch_gib": scratch,
           "sound_saving_gib": live, "peak_after_gib": after,
           "usable_gib": USABLE_GIB, "fits": bool(after <= USABLE_GIB),
           "plan_own_arithmetic_gib": aff + seg + gib(bpv - 8.93),
           "levers": [{"name": n, "gib": s, "sound": q, "why": w}
                      for n, s, q, w in levers]}
    (CACHE / "w2_mem.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nW2 wrote w2_mem.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
