#!/usr/bin/env python3
"""Mirror-tile CREMI-A val affinities into the 2.16 Gvox speed volume.

Same edge-aware mirror as the dataset tarball `make_big.py` (affinity is an
edge quantity: flip along axis a, roll channel a by +1, zero the seam).
Default 3x2x2 -> [3,375,2400,2400]. Needs h5py: `uv sync --extra eval`.

  uv run python scripts/make_big.py
  uv run python scripts/make_big.py --src data/cremiA_val/affinity.h5
  WATERZ_VAL_DIR=/path/to/cremiA_val uv run python scripts/make_big.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path(os.environ.get("WATERZ_VAL_DIR", ROOT / "data/cremiA_val")) / "affinity.h5"
DEFAULT_OUT = ROOT / "data/cremiA_216/affinity.h5"


def mirror(aff: np.ndarray, flips: tuple[bool, bool, bool]) -> np.ndarray:
    """aff: (3,Z,Y,X) uint8, channel order (z,y,x). flips: per spatial axis."""
    out = aff
    for a, do in enumerate(flips):
        if not do:
            continue
        out = np.flip(out, axis=a + 1)
        ch = np.roll(out[a], 1, axis=a)
        idx: list[slice | int] = [slice(None)] * 3
        idx[a] = 0
        ch[tuple(idx)] = 0
        out = out.copy()
        out[a] = ch
    return np.ascontiguousarray(out)


def build(src: Path, out_path: Path, factors: tuple[int, int, int]) -> int:
    import h5py

    fz, fy, fx = factors
    with h5py.File(src, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:])
    _, z, y, x = aff.shape
    shape = (3, z * fz, y * fy, x * fx)
    nvox = int(shape[1] * shape[2] * shape[3])
    print(
        f"source {aff.shape} -> {shape} = {nvox / 1e9:.2f} Gvox "
        f"({3 * nvox / 1e9:.2f} GB uncompressed)"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        d = f.create_dataset(
            "affinity",
            shape=shape,
            dtype=np.uint8,
            chunks=(3, min(64, z), 256, 256),
            compression="gzip",
            compression_opts=4,
        )
        for i in range(fz):
            for j in range(fy):
                for k in range(fx):
                    tile = mirror(aff, (i % 2 == 1, j % 2 == 1, k % 2 == 1))
                    d[:, i * z : (i + 1) * z, j * y : (j + 1) * y, k * x : (k + 1) * x] = tile
            print(f"  z-tile {i + 1}/{fz} written")
    print(f"[write] {out_path} {out_path.stat().st_size / 1e9:.2f} GB on disk")
    return nvox


def verify(src: Path) -> None:
    """Mirrored crop should match original watershed fragment count; naive flip should not."""
    import h5py
    import waterz

    with h5py.File(src, "r") as f:
        crop = np.ascontiguousarray(f["affinity"][:, 40:70, 300:700, 300:700])

    def nfrag(a: np.ndarray) -> int:
        a = np.ascontiguousarray(a.astype(np.float32) / 255.0)
        seg = next(iter(waterz.agglomerate(a, [0.0])))
        return len(np.unique(seg))

    base = nfrag(crop)
    for axis, name in ((0, "z"), (1, "y"), (2, "x")):
        flips = [False, False, False]
        flips[axis] = True
        m = nfrag(mirror(crop, tuple(flips)))
        naive = nfrag(np.ascontiguousarray(np.flip(crop, axis=axis + 1)))
        print(
            f"  mirror along {name}: fragments {m} vs original {base} "
            f"(rel {abs(m - base) / base:.4%})   naive np.flip: {naive}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--factors", nargs=3, type=int, default=[3, 2, 2], metavar=("FZ", "FY", "FX"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="waterz fragment-count check on a crop (needs pip install waterz)",
    )
    args = ap.parse_args()
    src = args.src.expanduser().resolve()
    if not src.is_file():
        print(f"missing {src}; unpack cremiA_val/affinity.h5 or set --src / WATERZ_VAL_DIR")
        return 1
    if args.verify:
        print("verifying mirror correctness...")
        verify(src)
    nvox = build(src, args.out.expanduser().resolve(), tuple(args.factors))
    if nvox < 1_000_000_000:
        print(f"WARNING: {nvox / 1e9:.2f} Gvox is below 1 Gvox")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
