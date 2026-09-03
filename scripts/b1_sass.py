#!/usr/bin/env python3
"""B1: how much of each full-volume watershed kernel is index arithmetic?

Every full-volume kernel in ws.cu recovers (z,y,x) from a linear thread index:

    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;

NVIDIA GPUs have no integer division instruction, so each of those four
divisions expands into a long sequence of multiplies, shifts and adds. This
counts what the compiler actually emitted, per kernel, so the decision to
restructure the launches rests on the instruction mix rather than on a guess
about how expensive division is.

The interesting figure is not the total but the share of integer-arithmetic
opcodes with no memory op in sight, which is what a 3D grid launch removes.

Usage: b1_sass.py <libws.so> [--json out.json]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

CUOBJDUMP = "/usr/local/cuda-12.8/bin/cuobjdump"

# Opcodes that make up the div/mod expansion and the address math around it.
ARITH = {
    "IMAD", "IADD3", "SHF", "LOP3", "IABS", "ISETP", "SEL", "FLO", "POPC",
    "IMNMX", "MOV", "PRMT", "BFE", "LEA", "I2F", "F2I", "FFMA", "FADD",
    "FMUL", "MUFU", "FSETP", "FSEL",
}
MEM = {
    "LDG", "STG", "LD", "ST", "LDS", "STS", "LDC", "ATOM", "ATOMG", "ATOMS",
    "RED", "REDG", "LDGSTS",
}

# The full-volume kernels: one thread per voxel, so the prologue runs 2.16e9
# times on the graded volume.
HOT = [
    "k_flow", "k_hook_bidir", "k_uf_compress_c", "k_plat_meta",
    "k_hook_remain", "k_uf_compress", "k_root_flag", "k_write_labels",
    "k_parent_init", "k_divide_pass", "k_reach_pass", "k_rewrite_bits",
]


def kernels(sass: str):
    """Yield (demangled-ish name, [opcodes]) for each non-CUB kernel."""
    for block in re.split(r"\t\tFunction : ", sass)[1:]:
        name = block.split("\n", 1)[0].strip()
        if "cub" in name or "CUB_" in name:
            continue
        # SASS mnemonics carry dot-suffixes that select a variant: LDG.E,
        # IMAD.WIDE, SHF.R.S32.HI, ISETP.GE.AND. Classifying on the raw token
        # puts almost everything in "other" and made memory traffic look like
        # two instructions per kernel, so normalise to the base opcode.
        ops = [
            m.split(".")[0].rstrip(";")
            for m in re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?\w+\s+)?(\S+)", block)
        ]
        yield name, ops


def short(name: str) -> str:
    """Recover the source name from an Itanium-mangled symbol.

    Matching on `k_\\w+` does not work: `\\w+` swallows the mangled parameter
    suffix, so `_Z12k_hook_bidirPKhPjPilll` yields `k_hook_bidirPKhPjPilll`.
    Mangling length-prefixes each identifier, so looking for `12k_hook_bidir`
    pins the whole name and nothing else.
    """
    for k in HOT:
        if f"{len(k)}{k}" in name:
            return k
    m = re.search(r"(k_[a-z0-9_]+?)(?=[A-Z])", name)
    return m.group(1) if m else name[:32]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    so = Path(sys.argv[1])
    sass = subprocess.run(
        [CUOBJDUMP, "-sass", str(so)], capture_output=True, text=True, check=True
    ).stdout
    rows = []
    for name, ops in kernels(sass):
        s = short(name)
        if s not in HOT:
            continue
        c = Counter(ops)
        total = len(ops)
        arith = sum(n for op, n in c.items() if op in ARITH)
        mem = sum(n for op, n in c.items() if op in MEM)
        rows.append({
            "kernel": s,
            "total": total,
            "arith": arith,
            "mem": mem,
            "arith_frac": arith / total if total else 0.0,
            "imad": c["IMAD"],
            "shf": c["SHF"],
            "iadd3": c["IADD3"],
            # MUFU.RCP appears in the float-reciprocal step of the integer
            # division expansion, so it is a direct marker for how many
            # divisions the compiler had to emit.
            "mufu": c["MUFU"],
        })
    rows.sort(key=lambda r: -r["arith"])
    print(f"B1 SASS {so.name}  ({len(rows)} full-volume kernels)")
    print(f"{'kernel':22s} {'total':>6s} {'arith':>6s} {'mem':>5s} "
          f"{'arith%':>7s} {'IMAD':>5s} {'SHF':>5s} {'IADD3':>6s} {'MUFU':>5s}")
    for r in rows:
        print(f"{r['kernel']:22s} {r['total']:6d} {r['arith']:6d} {r['mem']:5d} "
              f"{r['arith_frac']*100:6.1f}% {r['imad']:5d} {r['shf']:5d} "
              f"{r['iadd3']:6d} {r['mufu']:5d}")
    tot = sum(r["total"] for r in rows)
    ar = sum(r["arith"] for r in rows)
    print(f"{'TOTAL':22s} {tot:6d} {ar:6d} {'':5s} "
          f"{ar/tot*100 if tot else 0:6.1f}%")
    for i, a in enumerate(sys.argv):
        if a == "--json" and i + 1 < len(sys.argv):
            Path(sys.argv[i + 1]).write_text(
                json.dumps({"so": so.name, "rows": rows}, indent=2)
            )
            print(f"B1 wrote {sys.argv[i + 1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
