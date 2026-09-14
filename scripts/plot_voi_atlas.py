#!/usr/bin/env python3
"""Rebuild docs/voi_t03.{png,svg} from data/cache/voi_atlas.csv.

Every marker is one CSV row. Empty split/merge is refused. The rectangle is
stock waterz +0.02 at T=0.3 from gpu_waterz.limits, not a fitted box.

RNN and WPGMA sit on the mutex point; GASP and Kruskal are off this window.
They stay in the CSV, not on the axes.

  uv run --with matplotlib python scripts/plot_voi_atlas.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpu_waterz.limits import BASE_MERGE, BASE_SPLIT, SLACK  # noqa: E402

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    sys.exit("uv run --with matplotlib python scripts/plot_voi_atlas.py")

CSV_PATH = ROOT / "data/cache/voi_atlas.csv"
OUT_PNG = ROOT / "docs/voi_t03.png"
OUT_SVG = ROOT / "docs/voi_t03.svg"

# Drawn. (algorithm_class, probe_id, T, legend)
ROWS = [
    ("parhac_legal", "N16_DEEP", "0.3", "gpu-waterz"),
    ("heap_s4", "E4", "0.3", "stock waterz"),
    ("mutex_absmax", "M16", "0.3", "mutex"),
    ("complete_link", "N20_X4", "0.3", "complete-link"),
]

# Loaded only to print; not drawn (would sit on mutex, or off-window).
OFF = [
    ("rnn_s3", "N20_RNN", "0.3", "RNN"),
    ("wpgma", "N20_X5", "0.3", "WPGMA"),
    ("kruskal_sdsl", "N12", "all", "Kruskal"),
    ("gasp_average", "N19_X2", "0.3", "GASP mean"),
    ("nng_filter", "N19_H2", "0.3", "NNG"),
]


def load_atlas() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def find_row(atlas: list[dict[str, str]], cls: str, probe: str, t: str) -> dict[str, str]:
    hits = [
        r for r in atlas
        if r["algorithm_class"] == cls and r["probe_id"] == probe and r["T"] == t
    ]
    if len(hits) != 1:
        raise SystemExit(f"need one row for {cls} {probe} T={t}, got {len(hits)}")
    return hits[0]


def xy(row: dict[str, str]) -> tuple[float, float]:
    s, m = row["voi_split"].strip(), row["voi_merge"].strip()
    if not s or not m:
        raise SystemExit(f"refusing empty VOI: {row}")
    return float(s), float(m)


def main() -> None:
    atlas = load_atlas()
    pts = {}
    for cls, probe, t, legend in ROWS:
        pts[legend] = xy(find_row(atlas, cls, probe, t))
    off = []
    for cls, probe, t, legend in OFF:
        off.append((legend, *xy(find_row(atlas, cls, probe, t))))

    lim_s = BASE_SPLIT[0.3] + SLACK
    lim_m = BASE_MERGE[0.3] + SLACK
    assert abs(lim_s - 0.4738) < 1e-9 and abs(lim_m - 0.2611) < 1e-9

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.86)

    ax.add_patch(Rectangle(
        (0.0, 0.0), lim_s, lim_m,
        facecolor="#dfe9dc", edgecolor="#2f5d38",
        linewidth=1.1, zorder=0,
    ))
    ax.text(
        0.04, 0.245, "PASS\nwaterz +0.02",
        fontsize=10, color="#2f5d38", va="top", ha="left",
        linespacing=1.25,
    )

    g_s, g_m = pts["gpu-waterz"]
    w_s, w_m = pts["stock waterz"]
    mu_s, mu_m = pts["mutex"]
    cl_s, cl_m = pts["complete-link"]

    ax.scatter([w_s], [w_m], s=70, facecolors="white", edgecolors="#111",
               linewidths=1.4, zorder=4)
    ax.scatter([g_s], [g_m], s=70, c="#0c4a6e", zorder=5)
    ax.scatter([mu_s, cl_s], [mu_m, cl_m], s=55, c="#5a5a5a", zorder=3)

    ax.annotate(
        "gpu-waterz\n= stock waterz",
        xy=(g_s, (g_m + w_m) / 2),
        xytext=(0.22, 0.12),
        fontsize=10, ha="center", va="center", color="#0c4a6e",
        arrowprops=dict(arrowstyle="-", color="#0c4a6e", lw=0.8),
    )
    ax.annotate(
        "mutex\n(too chopped)",
        xy=(mu_s, mu_m),
        xytext=(0.72, 0.08),
        fontsize=10, ha="center", va="center", color="#333",
        arrowprops=dict(arrowstyle="-", color="#888", lw=0.8),
    )
    ax.annotate(
        "complete-link\n(too chopped)",
        xy=(cl_s, cl_m),
        xytext=(1.22, 0.08),
        fontsize=10, ha="center", va="center", color="#333",
        arrowprops=dict(arrowstyle="-", color="#888", lw=0.8),
    )

    ax.set_xlim(0.0, 1.52)
    ax.set_ylim(0.0, 0.32)
    ax.set_xlabel("split error  ->  neurons chopped into pieces")
    ax.set_ylabel("merge error  ->  neurons fused together")
    ax.set_title(
        "CREMI-A, affinity 0.3.  Green box = same objects as waterz.",
        fontsize=11, pad=10,
    )
    ax.tick_params(top=True, right=True)

    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight", pad_inches=0.08,
                facecolor="white")
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(ROOT)}")
    print(f"wrote {OUT_SVG.relative_to(ROOT)}")
    for name, (s, m) in pts.items():
        print(f"  drawn  {name:16s}  split={s:.6f}  merge={m:.6f}")
    for name, s, m in off:
        print(f"  csv    {name:16s}  split={s:.6f}  merge={m:.6f}  (not drawn)")


if __name__ == "__main__":
    main()
