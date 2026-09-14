#!/usr/bin/env python3
"""Rebuild docs/speed_216.png from data/cache/N19_I0_REPRO.json.

Bars are run216.stages. Title seconds are run216.e2e_ms / 1000.
No other source.

  uv run --with matplotlib python scripts/plot_speed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("uv run --with matplotlib python scripts/plot_speed.py")

PIN = ROOT / "data/cache/N19_I0_REPRO.json"
OUT_PNG = ROOT / "docs/speed_216.png"
OUT_SVG = ROOT / "docs/speed_216.svg"

# Display order: slowest first. Keys must match the pin JSON.
ORDER = (
    ("agg", "agglomeration"),
    ("ws", "watershed"),
    ("rag", "RAG"),
    ("extract", "extract"),
)


def main() -> None:
    pin = json.loads(PIN.read_text())
    run = pin["run216"]
    stages = run["stages"]
    e2e_ms = float(run["e2e_ms"])
    nvox = int(run["nvox"])
    assert nvox == 2160000000
    assert tuple(run["shape"]) == (3, 375, 2400, 2400)
    for k, _ in ORDER:
        if k not in stages:
            raise SystemExit(f"missing stage {k}")
    summed = sum(float(stages[k]) for k, _ in ORDER)
    if abs(summed - e2e_ms) > 2.0:
        raise SystemExit(f"stages sum {summed} != e2e {e2e_ms}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 11,
        "axes.linewidth": 0.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    labels = [name for _, name in ORDER]
    secs = [float(stages[k]) / 1000.0 for k, _ in ORDER]

    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    fig.subplots_adjust(left=0.28, right=0.82, top=0.78, bottom=0.18)
    y = list(range(len(labels)))[::-1]
    ax.barh(y, secs, height=0.62, color="#0c4a6e", edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("seconds")
    ax.set_xlim(0, 2.05)
    ax.tick_params(left=False, right=False, top=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="x", width=0.7, length=3.5)

    for yi, s in zip(y, secs):
        ax.text(s + 0.04, yi, f"{s:.2f} s", va="center", ha="left", fontsize=11)

    e2e_s = e2e_ms / 1000.0
    ax.set_title(f"{e2e_s:.1f} s   on 2.16 Gvox, idle RTX 5090", fontsize=13, pad=12)

    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight", pad_inches=0.12,
                facecolor="white")
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(ROOT)}")
    print(f"wrote {OUT_SVG.relative_to(ROOT)}")
    print(f"  e2e {e2e_s:.3f} s")
    for name, s in zip(labels, secs):
        print(f"  {name:16s} {s:.3f} s")


if __name__ == "__main__":
    main()
