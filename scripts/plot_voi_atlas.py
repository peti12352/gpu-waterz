#!/usr/bin/env python3
"""Draw docs/voi_t03.svg from data/cache/voi_atlas.csv.

Every marker is a CSV row (affinity 0.3, or T=all when that is the
measurement). Empty split or merge is not plotted. The legal box is
gpu_waterz.limits at T=0.3 (stock waterz +0.02), not a fitted rectangle.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpu_waterz.limits import BASE_MERGE, BASE_SPLIT, SLACK  # noqa: E402

CSV_PATH = ROOT / "data/cache/voi_atlas.csv"
OUT = ROOT / "docs/voi_t03.svg"

# (csv algorithm_class, probe_id, T as in the file, legend, in_window)
# in_window False: print the numbers on the figure, do not invent a position.
ROWS = [
    ("parhac_legal", "N16_DEEP", "0.3", "gpu-waterz", True),
    ("heap_s4", "E4", "0.3", "stock waterz", True),
    ("mutex_absmax", "M16", "0.3", "mutex", True),
    ("rnn_s3", "N20_RNN", "0.3", "RNN", True),
    ("wpgma", "N20_X5", "0.3", "WPGMA", True),
    ("complete_link", "N20_X4", "0.3", "complete-link", True),
    ("kruskal_sdsl", "N12", "all", "Kruskal SDSL", False),
    ("gasp_average", "N19_X2", "0.3", "GASP mean", False),
    ("nng_filter", "N19_H2", "0.3", "NNG", False),
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
    pts = []
    for cls, probe, t, legend, in_window in ROWS:
        row = find_row(atlas, cls, probe, t)
        split, merge = xy(row)
        pts.append((legend, split, merge, in_window, cls, probe, t))

    lim_s = BASE_SPLIT[0.3] + SLACK
    lim_m = BASE_MERGE[0.3] + SLACK
    # 0.4538 + 0.02, 0.2411 + 0.02
    assert abs(lim_s - 0.4738) < 1e-9 and abs(lim_m - 0.2611) < 1e-9

    # Window chosen so the gate, stock, gpu-waterz, mutex, RNN, WPGMA,
    # and complete-link sit at true coordinates. Kruskal / GASP / NNG
    # do not fit; their CSV numbers are written in the margin.
    xmax, ymax = 1.55, 0.42
    W, H = 880, 520
    l, r, tpad, b = 92, 24, 36, 56
    pw, ph = W - l - r, H - tpad - b

    def sx(split: float) -> float:
        return l + pw * (split / xmax)

    def sy(merge: float) -> float:
        return tpad + ph * (1.0 - merge / ymax)

    def circ(x, y, rad, fill, stroke, sw=1.4) -> str:
        return (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def txt(x, y, s, *, fill="#1a1a1a", size=12, anchor="start", weight="400") -> str:
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'font-family="ui-sans-serif, Helvetica, Arial, sans-serif" '
            f'font-weight="{weight}" text-anchor="{anchor}">{s}</text>'
        )

    x0, y0 = sx(0), sy(0)
    x1, y1 = sx(xmax), sy(ymax)
    bx1, by1 = sx(lim_s), sy(lim_m)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-label="VOI split vs merge '
        f'at affinity 0.3 on the same CREMI-A RAG">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        txt(l, 22, "CREMI-A val, affinity 0.3, same contact-mean RAG",
            size=14, weight="600"),
        f'<rect x="{x0:.1f}" y="{y1:.1f}" width="{x1-x0:.1f}" height="{y0-y1:.1f}" '
        f'fill="#fff" stroke="#222" stroke-width="1.2"/>',
        f'<rect x="{x0:.1f}" y="{by1:.1f}" width="{bx1-x0:.1f}" height="{y0-by1:.1f}" '
        f'fill="#e4efe3" stroke="#2c6b3a" stroke-width="1.6"/>',
        txt(x0 + 8, by1 + 16, "waterz gate  (+0.02)", fill="#2c6b3a", size=11),
        txt((x0 + x1) / 2, H - 16, "VOI split  (chopped neurons)  ->",
            size=12, anchor="middle"),
        txt(16, (tpad + H - b) / 2, "VOI merge  (fused neurons)",
            size=12, anchor="middle"),
    ]
    # rotate merge label
    parts[-1] = (
        f'<text x="18" y="{(tpad + H - b) / 2:.1f}" fill="#1a1a1a" font-size="12" '
        f'font-family="ui-sans-serif, Helvetica, Arial, sans-serif" '
        f'text-anchor="middle" transform="rotate(-90 18 {(tpad + H - b) / 2:.1f})">'
        f"VOI merge  (fused neurons)  -&gt;</text>"
    )

    ticks_x = [0.0, 0.5, 1.0, 1.5]
    ticks_y = [0.0, 0.1, 0.2, 0.3, 0.4]
    for tv in ticks_x:
        x = sx(tv)
        parts.append(
            f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y0+5:.1f}" '
            f'stroke="#222" stroke-width="1"/>'
        )
        parts.append(txt(x, y0 + 20, f"{tv:g}", size=11, anchor="middle"))
    for tv in ticks_y:
        y = sy(tv)
        parts.append(
            f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0-5:.1f}" y2="{y:.1f}" '
            f'stroke="#222" stroke-width="1"/>'
        )
        parts.append(txt(x0 - 10, y + 4, f"{tv:g}", size=11, anchor="end"))

    # label offsets in pixels, chosen so they do not sit on the marker
    label_at = {
        "gpu-waterz": (10, -8),
        "stock waterz": (10, 18),
        "mutex": (8, 16),
        "RNN": (-8, -8),
        "WPGMA": (8, -10),
        "complete-link": (8, -8),
    }
    anchor_at = {
        "RNN": "end",
    }

    for legend, split, merge, in_window, cls, probe, t in pts:
        if not in_window:
            continue
        if not (0 <= split <= xmax and 0 <= merge <= ymax):
            raise SystemExit(f"{legend} ({split}, {merge}) outside window")
        x, y = sx(split), sy(merge)
        if legend == "gpu-waterz":
            parts.append(circ(x, y, 6.5, "#c23b22", "#1a1a1a", 1.6))
        elif legend == "stock waterz":
            parts.append(circ(x, y, 6.0, "#fff", "#1a1a1a", 1.8))
        else:
            parts.append(circ(x, y, 4.5, "#5c5c5c", "#1a1a1a", 1.1))
        dx, dy = label_at[legend]
        parts.append(txt(
            x + dx, y + dy, legend,
            size=12, weight="600" if legend.startswith("gpu") else "400",
            anchor=anchor_at.get(legend, "start"),
        ))

    off = [p for p in pts if not p[3]]
    lines = ["Same CSV, off this window:"]
    for legend, split, merge, _, cls, probe, t in off:
        lines.append(f"{legend}  split={split:.3f}  merge={merge:.3f}")
    ox, oy = l + 8, tpad + 22
    parts.append(
        f'<rect x="{ox-6:.1f}" y="{oy-14:.1f}" width="268" height="72" '
        f'fill="#fbfaf7" fill-opacity="0.92" stroke="#ccc" stroke-width="0.8"/>'
    )
    for i, line in enumerate(lines):
        parts.append(txt(ox, oy + i * 16, line, size=11, fill="#333"))

    parts.append(
        txt(l, H - 4,
            "Markers: voi_atlas.csv. Box: stock waterz +0.02 at T=0.3. "
            "Lower is better.",
            size=10, fill="#555")
    )
    parts.append("</svg>")
    OUT.write_text("\n".join(parts) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for legend, split, merge, in_window, cls, probe, t in pts:
        print(f"  {legend:16s}  split={split:.6f}  merge={merge:.6f}  "
              f"{cls}/{probe} T={t}  {'plot' if in_window else 'margin'}")


if __name__ == "__main__":
    main()
