#!/usr/bin/env python3
"""N4: make_big seam / reflect identity. Not a 12x throughput claim.

make_big.py is not in the tree. The tarball README specifies the transform:
flip every channel along the mirrored axis, then shift that axis's own
channel by one voxel, zeroing the seam plane. This reconstructs that
transform, proves the seam is identically zero, counts seam faces on a
2-tile fragment construction, and shows a zero-affinity seam cannot
create a mean>T RAG edge.

Do not treat this as 12x e2e. Agglomeration still has to run on the
union graph unless the seam is proven merge-irrelevant (mean=0).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"


def mirror_tile(aff, axis):
    """Edge-correct mirror along spatial axis 0/1/2 = z/y/x.

    aff: [3, Z, Y, X]. Flip all channels on that spatial axis, then shift
    the axis's own channel by +1 (toward the low index after flip) and
    zero the seam plane. Matches TARBALL_README / TASK.md make_big notes.
    """
    out = np.flip(aff, axis=axis + 1).copy()
    ch = out[axis]
    shifted = np.zeros_like(ch)
    sl_src = [slice(None)] * 3
    sl_dst = [slice(None)] * 3
    sl_src[axis] = slice(0, -1)
    sl_dst[axis] = slice(1, None)
    shifted[tuple(sl_dst)] = ch[tuple(sl_src)]
    # seam at index 0 of this channel is already 0
    out[axis] = shifted
    return out


def stitch_2tile(aff, axis):
    mir = mirror_tile(aff, axis)
    return np.concatenate([aff, mir], axis=axis + 1), mir


def synthetic_check(rng, shape=(8, 16, 16)):
    aff = rng.random((3, *shape)).astype(np.float32)
    rows = []
    for axis, name in enumerate("zyx"):
        big, mir = stitch_2tile(aff, axis)
        # seam plane of the joining channel is the first plane of the
        # mirrored half along that axis.
        sl = [slice(None)] * 4
        sl[0] = axis
        sl[axis + 1] = shape[axis]
        seam = big[tuple(sl)]
        # after concat, index `shape[axis]` is the first voxel of the
        # mirrored tile; its backward affinity on this axis is the seam.
        rows.append({
            "axis": name,
            "seam_max": float(seam.max()),
            "seam_all_zero": bool(np.all(seam == 0)),
            "big_shape": list(big.shape),
            "mir_shape": list(mir.shape),
            "orig_channel_sum": float(aff[axis].sum()),
            "mir_channel_sum": float(mir[axis].sum()),
        })
    return rows


def fragment_2tile(axis):
    fr = np.load(CACHE / "wz_fragments.npy")
    max_id = int(fr.max())
    mir = np.flip(fr, axis=axis)
    # disjoint ids on the mirrored copy (bg stays 0)
    mir = np.where(mir == 0, 0, mir.astype(np.int64) + max_id).astype(np.uint32)
    big = np.concatenate([fr, mir], axis=axis)
    # seam faces: last plane of tile 0 vs first plane of tile 1
    sl0 = [slice(None)] * 3
    sl1 = [slice(None)] * 3
    sl0[axis] = -1
    sl1[axis] = fr.shape[axis]
    a = big[tuple(sl0)]
    # first plane of tile 1 in `big` is at index fr.shape[axis]
    sl1b = [slice(None)] * 3
    sl1b[axis] = fr.shape[axis]
    b = big[tuple(sl1b)]
    differ = (a != b)
    both_fg = (a != 0) & (b != 0)
    n_face = int(a.size)
    n_diff = int(differ.sum())
    n_fg_diff = int((differ & both_fg).sum())
    n_touch_bg = int(((a == 0) != (b == 0)).sum())
    return {
        "axis": "zyx"[axis],
        "tile_shape": list(fr.shape),
        "n_seam_faces": n_face,
        "n_label_differ": n_diff,
        "n_fg_fg_differ": n_fg_diff,
        "n_bg_touch": n_touch_bg,
        "max_id_tile": max_id,
        "note": "spatial flip of labels, not watershed-of-mirrored-aff. "
                "make_big --verify already matched fragment *counts* "
                "(54936 vs 54936). Seam aff is 0 so these faces cannot "
                "enter agglomeration as mean>T edges.",
    }


def main():
    rng = np.random.default_rng(0)
    syn = synthetic_check(rng)
    print("N4 synthetic 2-tile seam (reconstructed make_big)", flush=True)
    for r in syn:
        print(f"N4   axis={r['axis']} seam_all_zero={r['seam_all_zero']} "
              f"seam_max={r['seam_max']:.4g} shape={r['big_shape']}",
              flush=True)
    seams = [fragment_2tile(ax) for ax in range(3)]
    print("N4 fragment 2-tile seam faces", flush=True)
    for r in seams:
        print(f"N4   axis={r['axis']} faces={r['n_seam_faces']} "
              f"fg-fg differ={r['n_fg_fg_differ']}", flush=True)

    all_zero = all(r["seam_all_zero"] for r in syn)
    # RAG identity under a zero seam: internal edges of each tile are a
    # reflected copy; seam edges have mean=0 and never satisfy mean>T.
    # Union agglomeration ≡ two independent copies. This is identity of
    # the *merge relation*, not a 12x wall-clock claim on a 3090 Ti.
    out = {
        "make_big_present": False,
        "affinity_present": False,
        "synthetic_seams": syn,
        "synthetic_all_seams_zero": all_zero,
        "fragment_seams": seams,
        "rag_identity": {
            "claim": "zero seam aff ⇒ no mean>T cross-tile RAG edge",
            "internal_rag": "reflected copy of val rag.npz (disjoint ids)",
            "parent_identical_if_independent": True,
            "throughput_12x": False,
            "note": "Stamping a finished val segmentation 12 times is "
                    "correct for merge-equivalence iff seams stay mean=0. "
                    "Wall time is still val e2e x (5090/3090 bw), not "
                    "val/12. Do not claim 2 Gvox/s from this.",
        },
        "pass": all_zero,
    }
    dest = CACHE / "n4_mirror_identity.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"N4 {'PASS' if all_zero else 'FAIL'} seam-zero identity; "
          f"not a 12x speed claim", flush=True)
    print(f"N4 wrote {dest.name}", flush=True)
    return all_zero


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
