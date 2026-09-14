#!/usr/bin/env python3
"""Type-check the CUDA sources without a GPU and without nvcc.

The 5090 box has the toolkit; the machine this was written on has neither a
device nor nvcc, and "I edited 900 lines of CUDA but could not compile it" is
not a state worth being in. clang has compiled CUDA for years and only needs
headers, so this pulls the headers out of the pip wheels NVIDIA publishes
(no driver, no device, no toolkit install) and runs clang -fsyntax-only.

What this catches: syntax, name lookup, overload resolution, kernel launch
argument types, template instantiation in CUB and Thrust. What it does not
catch: anything about how the code behaves. It is a compile gate, not a
correctness gate, scripts/g0_agg_ref.py is the correctness gate, and the
device fingerprint in a1_e6s_voi.json is the real one.

Header-only, so ptxas and libdevice are never invoked and the CUDA version
clang knows about does not have to match.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRCS = ["csrc/parhac_d.cu", "csrc/ws.cu", "csrc/rag.cu"]

# Header-only wheels. cuda_cccl carries cub and thrust; clang's CUDA wrapper
# includes curand unconditionally, hence the third.
WHEELS = [
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-cccl-cu12",
    "nvidia-curand-cu12",
    "nvidia-cuda-nvcc-cu12",
]
INCLUDE_DIRS = [
    "cuda_runtime/include",
    "cuda_nvcc/include",
    "cuda_cccl/include",
    "curand/include",
]


def build_cuda_home(root: Path, refresh: bool):
    """Assemble the include/nvvm/bin layout clang's CUDA detection expects."""
    pkg = root / "pkg"
    home = root / "home"
    if refresh and root.exists():
        shutil.rmtree(root)
    if not pkg.exists():
        pkg.mkdir(parents=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--quiet", "--target", str(pkg), *WHEELS])
    inc = home / "include"
    inc.mkdir(parents=True, exist_ok=True)
    for d in INCLUDE_DIRS:
        src = pkg / "nvidia" / d
        if not src.is_dir():
            continue
        for f in src.iterdir():
            link = inc / f.name
            if not link.exists():
                link.symlink_to(f)
    for name, target in (("nvvm", "cuda_nvcc/nvvm"), ("bin", "cuda_nvcc/bin")):
        link = home / name
        if not link.exists() and (pkg / "nvidia" / target).is_dir():
            link.symlink_to(pkg / "nvidia" / target)
    if not (inc / "cuda_runtime.h").is_file():
        raise SystemExit("NV missing cuda_runtime.h; wheel layout changed")
    return home


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sm_86",
                    help="sm_86 = 3090 Ti, sm_120 = 5090; syntax only, so it "
                         "mostly does not matter")
    ap.add_argument("--cache", default="/tmp/waterz_cuda",
                    help="where to keep the downloaded headers")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("srcs", nargs="*", default=None)
    args = ap.parse_args()

    srcs = args.srcs or SRCS
    rc = 0
    clang = shutil.which("clang++")
    nvcc = shutil.which("nvcc") or "/usr/local/cuda-12.8/bin/nvcc"
    if clang:
        home = build_cuda_home(Path(args.cache), args.refresh)
        compiler = "clang"
    elif Path(nvcc).is_file():
        home = None
        compiler = "nvcc"
    else:
        raise SystemExit("NV neither clang++ nor nvcc found")

    for s in srcs:
        p = ROOT / s if not Path(s).is_absolute() else Path(s)
        if compiler == "clang":
            cmd = ["clang++", "-x", "cuda", "-fsyntax-only",
                   f"--cuda-path={home}", f"--cuda-gpu-arch={args.arch}",
                   "-Wno-unknown-cuda-version",
                   "-Wno-deprecated-declarations",
                   "-std=c++17", str(p)]
        else:
            obj = Path("/tmp") / (p.stem + "_nvcheck.o")
            cmd = [nvcc, "-c", "-std=c++17", f"-arch={args.arch}",
                   "-o", str(obj), str(p)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        n_err = out.count("error:")
        print(f"NV {'ok  ' if r.returncode == 0 else 'FAIL'} {s}"
              f"{'' if r.returncode == 0 else f'  ({n_err} errors)'}")
        if r.returncode != 0:
            print(out[:8000])
            rc = 1
    print(f"NV {'PASS' if rc == 0 else 'FAIL'}")
    return rc == 0


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
