#!/usr/bin/env python3
"""W1: does 2x2x2 block labelling actually pay, measured on the real output?

Block-based labelling (Komura equivalence, BKE-3D) is the largest single item
in the watershed plan and the most expensive to write, so it is worth knowing
whether its premise holds before touching ws.cu. The premise is that most 2x2x2
blocks of voxels lie entirely inside one fragment, so one label per block
replaces eight and the union-find shrinks by roughly the same factor.

That is measurable without a GPU and without the affinity volume, because the
watershed output is cached: data/cache/gpu_fragments.npy is the 125x1200x1200
label volume from the device run. This counts, exactly:

 - the fraction of blocks that are internally uniform, which is the label
    reduction actually available;
 - the number of face-adjacent voxel pairs in the same fragment, which is what
    the current per-voxel union-find has to resolve, against the number of
    block-level unions that replace them;
 - the L2 residency arithmetic at 2.16 Gvox, which is the reason the plan
    wants this at all.

The last one is worth checking rather than repeating: a block-label z-plane at
2.16 Gvox is 5.76 MB, but a +/-z neighbour gather touches three planes, so
block labels alone do not fit a 6 MB L2 and the claim that they do is wrong.
What they do is make an xy-tiled sweep fit, which is a different design.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
GB = 1e9
MB = 1e6

# L2 per card. The 3090 Ti's 6 MB against the 5090's 128 MB is the whole reason
# the watershed is expected to scale worse than bandwidth alone predicts.
L2 = {"3090Ti": 6 * MB, "5090": 128 * MB}
TARGET_SHAPE = (375, 2400, 2400)     # 2.16 Gvox, make_big.py 3x2x2 of val


def block_stats(a, zchunk=8):
    """Uniformity of every full 2x2x2 block, and the label reduction.

    Done in z-chunks because materialising the 8-neighbour gather over 180 M
    voxels at once is 1.4 GB per copy.
    """
    nz, ny, nx = a.shape
    bz, by, bx = nz // 2, ny // 2, nx // 2
    n_uniform = 0
    n_block = 0
    distinct_hist = np.zeros(9, dtype=np.int64)
    for z0 in range(0, bz, zchunk):
        z1 = min(z0 + zchunk, bz)
        blk = np.asarray(a[2 * z0:2 * z1, :2 * by, :2 * bx], dtype=np.uint32)
        # (z,2,y,2,x,2) -> (blocks, 8)
        blk = blk.reshape(z1 - z0, 2, by, 2, bx, 2)
        blk = blk.transpose(0, 2, 4, 1, 3, 5).reshape(-1, 8)
        first = blk[:, :1]
        same = (blk == first).all(axis=1)
        n_uniform += int(same.sum())
        n_block += blk.shape[0]
        # How many distinct fragments a block spans, for the non-uniform tail.
        srt = np.sort(blk, axis=1)
        nd = 1 + (srt[:, 1:] != srt[:, :-1]).sum(axis=1)
        distinct_hist += np.bincount(nd, minlength=9)[:9]
    return {
        "n_block": n_block,
        "n_uniform": n_uniform,
        "frac_uniform": n_uniform / n_block,
        "distinct_hist": distinct_hist.tolist(),
        "mean_distinct": float((np.arange(9) * distinct_hist).sum()
                               / max(distinct_hist.sum(), 1)),
    }


def union_counts(a, zchunk=16):
    """Face-adjacent same-fragment pairs, per voxel and per block.

    The per-voxel figure is the number of union operations a naive 1-thread-
    per-voxel union-find has to perform along the three face directions. The
    per-block figure is what survives when the 2x2x2 interior is resolved in
    registers and only block faces reach the union-find.
    """
    nz, ny, nx = a.shape
    same_vox = 0
    for z0 in range(0, nz, zchunk):
        z1 = min(z0 + zchunk + 1, nz)
        c = np.asarray(a[z0:z1], dtype=np.uint32)
        if c.shape[0] > 1:
            same_vox += int((c[:-1] == c[1:]).sum())
        # y and x only for the rows owned by this chunk, to avoid double count
        own = c[:min(zchunk, c.shape[0])]
        same_vox += int((own[:, :-1] == own[:, 1:]).sum())
        same_vox += int((own[:, :, :-1] == own[:, :, 1:]).sum())

    # Block-level: one label per block, unions only across block faces.
    bz, by, bx = nz // 2, ny // 2, nx // 2
    # A block face carries 4 voxel adjacencies; the block union is attempted
    # once per face pair, so the count is the number of block face pairs.
    block_face_pairs = ((bz - 1) * by * bx + bz * (by - 1) * bx
                        + bz * by * (bx - 1))
    vox_face_pairs = ((nz - 1) * ny * nx + nz * (ny - 1) * nx
                      + nz * ny * (nx - 1))
    return {
        "vox_face_pairs": vox_face_pairs,
        "vox_same_fragment_pairs": same_vox,
        "block_face_pairs": block_face_pairs,
        "union_reduction": vox_face_pairs / block_face_pairs,
    }


def l2_arithmetic(shape, frac_uniform):
    """Residency of the label array's sliding window, voxel vs block."""
    nz, ny, nx = shape
    rows = []
    for tag, w, div in (("voxel labels, uint32", 4, 1),
                        ("block labels, uint32 per 2x2x2", 4, 2)):
        plane = (ny // div) * (nx // div) * w
        rows.append({
            "what": tag,
            "plane_mb": plane / MB,
            "window3_mb": 3 * plane / MB,
            "full_gb": (nz // div) * plane / GB,
        })
    # An xy-tiled sweep is what actually fits: a tile of TxT blocks with a
    # three-plane window in block space covers 2T x 2T x 6 voxels.
    tiles = []
    for t in (256, 512, 1024):
        tiles.append({"tile_blocks": t,
                      "window_mb": 3 * t * t * 4 / MB,
                      "voxels_covered": (2 * t) ** 2 * 6})
    return rows, tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frag", default=str(CACHE / "gpu_fragments.npy"))
    ap.add_argument("--zchunk", type=int, default=8)
    args = ap.parse_args()

    a = np.load(args.frag, mmap_mode="r")
    nvox = int(a.size)
    nfrag = int(np.unique(np.asarray(a[::7])).size)  # sampled, for a sanity line
    print(f"W1 {Path(args.frag).name} shape={a.shape} nvox={nvox / 1e6:.0f} M")

    bs = block_stats(a, args.zchunk)
    print(f"W1 2x2x2 blocks: {bs['n_block'] / 1e6:.1f} M, "
          f"{bs['frac_uniform'] * 100:.1f}% internally uniform, "
          f"mean {bs['mean_distinct']:.2f} distinct fragments per block")
    print("W1 distinct fragments per block:")
    for k, n in enumerate(bs["distinct_hist"]):
        if n:
            print(f"W1   {k} -> {n / 1e6:8.3f} M  "
                  f"{n / bs['n_block'] * 100:5.1f}%")

    uc = union_counts(a)
    print(f"W1 face-adjacent voxel pairs      {uc['vox_face_pairs'] / 1e6:9.1f} M")
    print(f"W1   of which same fragment       "
          f"{uc['vox_same_fragment_pairs'] / 1e6:9.1f} M "
          f"({uc['vox_same_fragment_pairs'] / uc['vox_face_pairs'] * 100:.1f}%) "
          f"-- these are the unions the current UF performs")
    print(f"W1 block face pairs               "
          f"{uc['block_face_pairs'] / 1e6:9.1f} M  "
          f"{uc['union_reduction']:.1f}x fewer union sites")

    rows, tiles = l2_arithmetic(TARGET_SHAPE, bs["frac_uniform"])
    print(f"\nW1 L2 residency at 2.16 Gvox {TARGET_SHAPE}, "
          f"3090 Ti L2 = {L2['3090Ti'] / MB:.0f} MB")
    for r in rows:
        fits = r["window3_mb"] < L2["3090Ti"] / MB
        print(f"W1   {r['what']:30s} plane {r['plane_mb']:7.2f} MB  "
              f"3-plane window {r['window3_mb']:7.2f} MB  "
              f"{'FITS' if fits else 'OVER'}  full {r['full_gb']:.2f} GB")
    print("W1   neither whole-plane form fits, so the plan's claim that block "
          "labels alone are L2-resident is wrong. What fits is an xy tile:")
    for t in tiles:
        fits = t["window_mb"] < L2["3090Ti"] / MB
        print(f"W1   {t['tile_blocks']:5d}^2 blocks  3-plane window "
              f"{t['window_mb']:6.2f} MB  {'FITS' if fits else 'OVER'}  "
              f"covers {t['voxels_covered'] / 1e6:.2f} M voxels")

    # How many labels a block actually needs.
    #
    # The plan costs this at "one uint32 per 8 voxels = 0.5 B/vox", which
    # assumes every block is internally uniform. 37.8% are not: a watershed
    # basin boundary runs through them, and a block straddling two basins needs
    # two labels, not one. So the real footprint is set by the distribution
    # above, and the design question is how many slots to reserve per block
    # before spilling to an overflow table.
    hist = np.asarray(bs["distinct_hist"], dtype=np.float64)
    tot = hist.sum()
    mean_lbl = bs["mean_distinct"]
    print(f"\nW1 labels per block: mean {mean_lbl:.2f}, so the union-find "
          f"carries {mean_lbl / 8 * 100:.0f}% of one label per voxel")
    print("W1 reserving k slots per block, uint32 each:")
    slots = []
    for k in (1, 2, 3, 4, 8):
        spill = float(hist[k + 1:].sum() / tot) if k < 8 else 0.0
        b_per_vox = k * 4 / 8
        slots.append({"k": k, "b_per_vox": b_per_vox, "spill_frac": spill})
        print(f"W1   k={k}  {b_per_vox:4.2f} B/vox  "
              f"spill {spill * 100:5.2f}% of blocks  "
              f"saves {4 - b_per_vox:4.2f} B/vox = "
              f"{(4 - b_per_vox) * 2.16e9 / (1024 ** 3):5.2f} GiB")
    print("W1   k=2 is the design point: 1 B/vox, 3 GiB saved at 2.16 Gvox, "
          f"and only {slots[1]['spill_frac'] * 100:.1f}% of blocks spill.")

    report = {"shape": list(a.shape), "nvox": nvox,
              "nfrag_sampled": nfrag, "blocks": bs, "unions": uc,
              "l2_rows": rows, "l2_tiles": tiles, "slot_options": slots,
              "target_shape": list(TARGET_SHAPE)}
    dest = CACHE / "w1_block_analysis.json"
    dest.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\nW1 wrote {dest.name}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
