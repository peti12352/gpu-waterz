#!/usr/bin/env python3
"""Rebuild docs/voi_t03.{png,svg} from data/cache/voi_atlas.csv.

Every marker is one CSV row. Empty split/merge is refused. The rectangle is
stock waterz +0.02 at T=0.3 from gpu_waterz.limits, not a fitted box.

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
    from matplotlib.lines import Line2D
except ImportError:
    sys.exit("uv run --with matplotlib python scripts/plot_voi_atlas.py")

CSV_PATH = ROOT / "data/cache/voi_atlas.csv"
OUT_PNG = ROOT / "docs/voi_t03.png"
OUT_SVG = ROOT / "docs/voi_t03.svg"

# (algorithm_class, probe_id, T, legend)
ROWS = [
    ("parhac_legal", "N16_DEEP", "0.3", "gpu-waterz"),
    ("heap_s4", "E4", "0.3", "stock waterz"),
    ("mutex_absmax", "M16", "0.3", "mutex"),
    ("rnn_s3", "N20_RNN", "0.3", "RNN"),
    ("wpgma", "N20_X5", "0.3", "WPGMA"),
    ("complete_link", "N20_X4", "0.3", "complete-link"),
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


def style(name: str) -> dict:
    if name == "gpu-waterz":
        return dict(marker="o", s=48, c="#0c4a6e", zorder=5,
                    edgecolors="#0c4a6e", linewidths=0.6)
    if name == "stock waterz":
        return dict(marker="o", s=48, facecolors="white", edgecolors="#111111",
                    linewidths=1.15, zorder=4)
    return dict(marker="o", s=28, c="#6b6b6b", zorder=3,
                edgecolors="#6b6b6b", linewidths=0.4)


def main() -> None:
    atlas = load_atlas()
    pts = []
    for cls, probe, t, legend in ROWS:
        split, merge = xy(find_row(atlas, cls, probe, t))
        pts.append((legend, split, merge, cls, probe, t))

    lim_s = BASE_SPLIT[0.3] + SLACK
    lim_m = BASE_MERGE[0.3] + SLACK
    assert abs(lim_s - 0.4738) < 1e-9 and abs(lim_m - 0.2611) < 1e-9

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.15))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.28, top=0.90, wspace=0.32)
    fig.suptitle("CREMI-A val, affinity 0.3, same region graph",
                 fontsize=9, y=0.98)

    windows = (
        (0.0, 1.48, 0.0, 0.36),
        (0.0, 3.22, 0.0, 8.15),
    )
    notes = (
        [
            (0.90, 0.248, "mutex, RNN, WPGMA", "center"),
            (1.35, 0.230, "complete-link", "center"),
        ],
        [
            (0.40, 7.48, "GASP", "left"),
            (2.15, 0.38, "NNG", "center"),
            (2.94, 0.38, "Kruskal", "center"),
        ],
    )

    for ax, (x0, x1, y0, y1), letter, labels in zip(axes, windows, "ab", notes):
        ax.add_patch(Rectangle(
            (0.0, 0.0), lim_s, lim_m,
            facecolor="#e8efe6", edgecolor="#2f5d38",
            linewidth=0.75, zorder=0, joinstyle="miter",
        ))
        for legend, split, merge, *_ in pts:
            ax.scatter([split], [merge], **style(legend), clip_on=True)
        for x, y, s, ha in labels:
            ax.text(x, y, s, fontsize=7, color="#333333", ha=ha, va="bottom")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("VOI split")
        ax.tick_params(top=True, right=True)
        ax.text(
            0.03, 0.97, letter,
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            fontstyle="italic", va="top", ha="left",
        )

    axes[0].set_ylabel("VOI merge")

    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="#0c4a6e", markeredgecolor="#0c4a6e",
               markersize=6.5, label="gpu-waterz"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="white", markeredgecolor="#111111",
               markeredgewidth=1.15, markersize=6.5, label="stock waterz"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="#6b6b6b", markeredgecolor="#6b6b6b",
               markersize=5, label="other clustering"),
        Rectangle((0, 0), 1, 1, facecolor="#e8efe6", edgecolor="#2f5d38",
                  linewidth=0.75, label="waterz +0.02"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.0),
        ncol=4, frameon=False, handletextpad=0.45, columnspacing=1.4,
        fontsize=8,
    )

    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(ROOT)}")
    print(f"wrote {OUT_SVG.relative_to(ROOT)}")
    for legend, split, merge, cls, probe, t in pts:
        print(f"  {legend:16s}  split={split:.6f}  merge={merge:.6f}  "
              f"{cls}/{probe} T={t}")


if __name__ == "__main__":
    main()
