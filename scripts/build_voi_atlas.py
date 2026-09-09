#!/usr/bin/env python3
"""Rebuild voi_atlas.csv + voi_atlas.json from pinned LOG/JSON/dead sources.

Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
FIELDS = [
    "algorithm_class",
    "probe_id",
    "param",
    "T",
    "voi_split",
    "voi_merge",
    "nseg",
    "wall_s",
    "task_legal",
    "fail_mode",
    "note_path",
]


def cell(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.15g}"
    if isinstance(x, bool):
        return "True" if x else "False"
    return str(x)


def row(
    algorithm_class,
    probe_id,
    param,
    T,
    voi_split,
    voi_merge,
    nseg,
    wall_s,
    task_legal,
    fail_mode,
    note_path,
):
    return {
        "algorithm_class": algorithm_class,
        "probe_id": probe_id,
        "param": cell(param),
        "T": cell(T),
        "voi_split": cell(voi_split),
        "voi_merge": cell(voi_merge),
        "nseg": cell(nseg),
        "wall_s": cell(wall_s),
        "task_legal": cell(bool(task_legal)),
        "fail_mode": fail_mode,
        "note_path": note_path,
    }


def main():
    h2 = json.loads((CACHE / "N19_H2.json").read_text())
    x2 = json.loads((CACHE / "N19_X2.json").read_text())
    i0 = json.loads((CACHE / "N19_I0_REPRO.json").read_text())
    dead = [
        json.loads(l)
        for l in (CACHE / "n19_dead.jsonl").read_text().splitlines()
        if l.strip()
    ]
    b2 = {}
    for d in dead:
        if d["exp_id"].startswith("N18_B2_eps_"):
            eps = d["exp_id"].split("_")[-1]
            b2[eps] = d["numbers"]["voi"]["m1"]

    rows = []

    # --- X0 frozen CC (LOG ## X0 FAIL) ---
    for T, merge, nseg in [
        (0.2, 7.78, 279468),
        (0.3, 7.55, 309454),
        (0.4, 6.43, 330578),
        (0.5, 4.61, 355268),
    ]:
        rows.append(
            row(
                "frozen_cc",
                "X0",
                "mean>T",
                T,
                None,
                merge,
                nseg,
                None,
                False,
                "giant",
                "notes/LOG.md#x0-fail",
            )
        )

    # --- W31 waterfall identical to X0 ---
    for T, merge, nseg in [
        (0.2, 7.78, 279468),
        (0.3, 7.55, 309454),
        (0.4, 6.43, 330578),
        (0.5, 4.61, 355268),
    ]:
        rows.append(
            row(
                "waterfall",
                "W31",
                "lowest-pass mean>T",
                T,
                None,
                merge,
                nseg,
                None,
                False,
                "giant",
                "notes/LOG.md#w31-fail",
            )
        )

    # --- X1 union-all-in-band ---
    for param, T, merge in [
        ("B=16", 0.3, 3.04),
        ("B=64", 0.3, 1.04),
        ("B=256", 0.3, 0.328),
        ("B=1024", 0.2, 0.498),
        ("B=1024", 0.3, 0.2724),
    ]:
        rows.append(
            row(
                "union_all_in_band",
                "X1",
                param,
                T,
                None,
                merge,
                None,
                None,
                False,
                "giant",
                "notes/LOG.md#x1-fail-all-b",
            )
        )

    rows.append(
        row(
            "mutual_in_bucket",
            "X1b",
            "B=16",
            "all",
            2.0,
            None,
            1050000,
            None,
            False,
            "under_merge",
            "notes/LOG.md#x1b-fail",
        )
    )

    # --- M16 mutex ---
    for T in (0.2, 0.3, 0.4, 0.5):
        rows.append(
            row(
                "mutex_absmax",
                "M16",
                "Wolf Alg2 / GASP AbsMax",
                T,
                0.9097,
                0.2033,
                421990 if T < 0.5 else 422041,
                13.081,
                False,
                "under_merge",
                "notes/LOG.md#m16-fail",
            )
        )
    rows.append(
        row(
            "mutex_hop",
            "M16b",
            "hop-k in {2,4,8}",
            "all",
            2.129,
            None,
            883000,
            153.241,
            False,
            "under_merge",
            "notes/LOG.md#m16b-fail",
        )
    )

    # --- N12 Kruskal / eps ---
    rows.append(
        row(
            "kruskal_sdsl",
            "N12",
            "frozen SDSL",
            "all",
            2.94,
            0.0,
            None,
            6.092,
            False,
            "under_merge",
            "notes/N12_OFF.md",
        )
    )
    for eps, split, merge, wall in [
        (0.5, 0.4345, 0.2712, 0.461),
        (0.8, 0.4323, 0.3058, 0.155),
        (1.0, 0.4225, 0.3760, 0.143),
        (2.0, 0.4490, 0.8735, 0.130),
    ]:
        rows.append(
            row(
                "parhac_eps",
                "N12",
                f"eps={eps}",
                0.3,
                split,
                merge,
                None,
                wall,
                False,
                "order_change",
                "notes/N12_OFF.md",
            )
        )

    # --- Z25 SDSL size-cap ---
    for s0, split, merge, mode in [
        (64, 1.616, 0.201, "under_merge"),
        (256, 1.050, 0.230, "under_merge"),
        (1024, 0.779, 0.252, "under_merge"),
        (4096, 0.663, 0.270, "both_halves"),
        (16384, 0.583, 0.302, "both_halves"),
    ]:
        rows.append(
            row(
                "sdsl_size_cap",
                "Z25",
                f"S0={s0}",
                0.2,
                split,
                merge,
                None,
                None,
                False,
                mode,
                "notes/LOG.md#z25-fail-all-s0",
            )
        )

    # --- F23 FH ---
    for k, split, merge, mode in [
        (50, 2.208, 0.191, "under_merge"),
        (200, 2.049, 0.213, "under_merge"),
        (2000, 1.521, 0.499, "both_halves"),
    ]:
        rows.append(
            row(
                "fh_mint",
                "F23",
                f"k={k}",
                0.2,
                split,
                merge,
                None,
                None,
                False,
                mode,
                "notes/LOG.md#f23-fail-all-k",
            )
        )

    # --- R24 SRM ---
    for q, split, merge in [(16, 1.148, 3.261), (256, 1.504, 1.623)]:
        rows.append(
            row(
                "srm",
                "R24",
                f"Q={q}",
                0.2,
                split,
                merge,
                None,
                None,
                False,
                "both_halves",
                "notes/LOG.md#r24-fail-all-q",
            )
        )

    # --- S26 Soille ---
    for w, split, merge, mode in [
        (0.05, 0.420, 2.876, "giant"),
        (0.10, 1.126, 0.359, "both_halves"),
    ]:
        rows.append(
            row(
                "soille",
                "S26",
                f"omega={w}",
                0.2,
                split,
                merge,
                None,
                None,
                False,
                mode,
                "notes/LOG.md#s26-fail-all-omega",
            )
        )

    # --- C28 ---
    for param, split, merge in [
        ("B=16 Smax=1024", 0.341, 3.109),
        ("B=32 Smax=4096", 0.437, 1.263),
    ]:
        rows.append(
            row(
                "union_all_size_cap",
                "C28",
                param,
                0.2,
                split,
                merge,
                None,
                None,
                False,
                "giant",
                "notes/LOG.md#c28-fail-all-bsmax",
            )
        )

    # --- R32 ---
    for T, split, merge, nseg in [
        (0.2, 0.4270, 0.3451, 280195),
        (0.3, 0.4618, 0.2792, 310243),
        (0.4, 0.4891, 0.2570, 331405),
        (0.5, 0.5304, 0.2350, 356087),
    ]:
        rows.append(
            row(
                "relative_contact",
                "R32",
                "gamma=0.10 alpha=0.67",
                T,
                split,
                merge,
                nseg,
                None,
                False,
                "both_halves",
                "notes/LOG.md#r32-fail",
            )
        )
    rows.append(
        row(
            "relative_contact",
            "R36",
            "gamma=0.05 alpha=0.67",
            0.2,
            0.3870,
            0.4421,
            None,
            None,
            False,
            "giant",
            "notes/LOG.md#r36-fail",
        )
    )

    # --- L33 / L36 ---
    for s0, split, merge, nseg in [
        (4096, 0.3076, 1.093, 279793),
        (1024, 0.2714, 1.880, None),
        (256, 0.1968, 4.455, None),
    ]:
        rows.append(
            row(
                "leftover_s4",
                "L33",
                f"S0={s0}",
                0.2,
                split,
                merge,
                nseg,
                None,
                False,
                "giant",
                "notes/LOG.md#l33-fail-all-s0",
            )
        )
    for g, split, merge in [(0.10, 0.3486, 0.5242), (0.05, 0.3443, 0.5089)]:
        rows.append(
            row(
                "leftover_s4",
                "L36",
                f"gamma={g} residual",
                0.2,
                split,
                merge,
                None,
                None,
                False,
                "giant",
                "notes/LOG.md#l36-fail-both-gamma",
            )
        )

    rows.append(
        row(
            "block_kruskal",
            "B34",
            "32x128x128 then residual",
            0.2,
            0.1272,
            7.718,
            None,
            None,
            False,
            "giant",
            "notes/LOG.md#b34-fail",
        )
    )
    rows.append(
        row(
            "lu_freeze_naive",
            "B34",
            "5-level freeze (not Alg2 paper)",
            0.2,
            0.120,
            7.778,
            None,
            None,
            False,
            "giant",
            "notes/LOG.md#b34-fail",
        )
    )
    rows.append(
        row(
            "hysteresis",
            "V35",
            "high=0.9 low=T",
            0.2,
            1.090,
            7.424,
            9420000,
            None,
            False,
            "both_halves",
            "notes/LOG.md#v35-fail",
        )
    )
    rows.append(
        row(
            "hem",
            "H30",
            "size-doubling 20 rounds",
            0.2,
            2.374,
            0.201,
            917000,
            82.5,
            False,
            "under_merge",
            "notes/LOG.md#h30-fail",
        )
    )

    # --- E5 Boruvka ---
    for T, merge, nseg in [
        (0.2, 0.3885, 282829),
        (0.3, 0.2779, 312939),
        (0.4, 0.2423, 334450),
    ]:
        rows.append(
            row(
                "boruvka_onesided",
                "E5",
                "PLAN 3.4",
                T,
                None,
                merge,
                nseg,
                None,
                False,
                "giant",
                "notes/LOG.md#e5-accuracy-gate-pass",
            )
        )

    # --- E1 extra plateau CC ---
    for T, split, merge, nseg, legal, mode in [
        (0.2, 0.3852, 0.4214, 294248, False, "not_S1_fragments"),
        (0.3, 0.4591, 0.2896, 322078, False, "not_S1_fragments"),
        (0.4, 0.5232, 0.2469, 345380, False, "not_S1_fragments"),
        (0.5, 0.6130, 0.2272, 380406, True, "pass"),
    ]:
        rows.append(
            row(
                "extra_plateau_cc",
                "E1",
                "GPU UF nfrag+1.6pct",
                T,
                split,
                merge,
                nseg,
                588,
                legal,
                mode,
                "notes/LOG.md#e1-fail-taskmd-voi",
            )
        )

    # --- E2 eps=1.0 FAIL ---
    rows.append(
        row(
            "parhac_eps",
            "E2",
            "eps=1.0 matching-only",
            0.2,
            0.3856,
            0.3729,
            None,
            278,
            False,
            "order_change",
            "notes/LOG.md#e2-paper-eps-on-y2-matching-cached-host-rag-7505458-edges",
        )
    )

    # --- N18 B2 full precision from dead ---
    for eps in sorted(b2.keys(), key=float):
        m = b2[eps]
        rows.append(
            row(
                "parhac_eps",
                "N18_B2",
                f"eps={eps}",
                0.3,
                float(m["split"]),
                float(m["merge"]),
                int(m["nseg"]),
                None,
                False,
                "order_change",
                "notes/N18_B2.md",
            )
        )

    # --- N19 probes from JSON ---
    rows.append(
        row(
            "nng_filter",
            "N19_H2",
            "7.5M->1.8M edges",
            0.3,
            float(h2["split"]),
            float(h2["merge"]),
            int(h2["nseg"]),
            float(h2["ms_t03"]) / 1000.0,
            False,
            "order_change",
            "notes/N19_H2.md",
        )
    )
    rows.append(
        row(
            "gasp_average",
            "N19_X2",
            "cached RAG",
            0.3,
            float(x2["split"]),
            float(x2["merge"]),
            int(x2["nseg"]),
            float(x2["ms"]) / 1000.0,
            False,
            "giant",
            "notes/N19_X2.md",
        )
    )
    rows.append(
        row(
            "bin_ladder",
            "N19_H1",
            "uint8 bins Luengo",
            0.3,
            None,
            None,
            None,
            5.841,
            False,
            "timeout",
            "notes/N19_H1.md",
        )
    )
    rows.append(
        row(
            "pruf_grayscale",
            "N12_OFF",
            "Meyer PRUF3D",
            "n/a",
            None,
            None,
            None,
            3.75,
            False,
            "not_S1_fragments",
            "notes/N12_OFF.md",
        )
    )
    rows.append(
        row(
            "pruf_affinity",
            "N19_X1",
            "no integration",
            "n/a",
            None,
            None,
            None,
            None,
            False,
            "not_S1_fragments",
            "notes/N19_X1.md",
        )
    )
    rows.append(
        row(
            "rama",
            "N13_L5",
            "signed multicut 600s",
            "n/a",
            None,
            None,
            None,
            600,
            False,
            "timeout",
            "notes/N13_RAMA.md",
        )
    )
    rows.append(
        row(
            "rama",
            "N19_X3",
            "60s abort",
            "n/a",
            None,
            None,
            None,
            60,
            False,
            "timeout",
            "notes/N19_X3.md",
        )
    )
    rows.append(
        row(
            "terahac",
            "T14",
            "cap=0 then 16384",
            "n/a",
            None,
            None,
            None,
            360,
            False,
            "timeout",
            "notes/LOG.md#t14-killed-y1-class",
        )
    )
    rows.append(
        row(
            "lu_chunk_simplified",
            "N19_H4",
            "parents vs ParHAC",
            "n/a",
            None,
            None,
            None,
            None,
            False,
            "order_change",
            "notes/N19_H4.md",
        )
    )

    # --- legal locks ---
    for T, nseg in [
        (0.2, 294231),
        (0.3, 322089),
        (0.4, 345229),
        (0.5, 379772),
    ]:
        rows.append(
            row(
                "parhac_legal",
                "Y2",
                "eps=0.01",
                T,
                None,
                None,
                nseg,
                501,
                True,
                "pass",
                "notes/LOG.md#y2-pass",
            )
        )
    for T, split, merge in [
        (0.2, 0.3817, 0.3381),
        (0.3, 0.4567, 0.2422),
        (0.4, 0.5180, 0.2185),
        (0.5, 0.6207, 0.2093),
    ]:
        rows.append(
            row(
                "parhac_legal",
                "E2",
                "eps=0.10 matching",
                T,
                split,
                merge,
                None,
                396,
                True,
                "pass",
                "notes/LOG.md#e2-paper-eps-on-y2-matching-cached-host-rag-7505458-edges",
            )
        )
    for T, split, merge in [
        (0.2, 0.370698, 0.335006),
        (0.3, 0.451219, 0.250531),
        (0.4, 0.516214, 0.226759),
        (0.5, 0.61294, 0.218363),
    ]:
        rows.append(
            row(
                "parhac_legal",
                "N16_DEEP",
                "eps=0.08 four-T",
                T,
                split,
                merge,
                None,
                None,
                True,
                "pass",
                "notes/N16_DEEP.md",
            )
        )
    m = i0["voi"]["m1"]
    rows.append(
        row(
            "parhac_legal",
            "N19_I0_REPRO",
            "eps=0.40 T=0.3 speed",
            0.3,
            float(m["split"]),
            float(m["merge"]),
            int(m["nseg"]),
            None,
            True,
            "pass",
            "data/cache/N19_I0_REPRO.json",
        )
    )
    rows.append(
        row(
            "rac",
            "Y1",
            "exact RAC",
            0.2,
            None,
            None,
            294231,
            536,
            True,
            "pass",
            "notes/LOG.md#y1-pass--too-serial",
        )
    )
    rows.append(
        row(
            "quantile_binqueue",
            "Q20",
            "Q=50 bins=256",
            0.2,
            None,
            None,
            294518,
            None,
            True,
            "pass",
            "notes/LOG.md#q20-pass-serial-not-a-g6-lock",
        )
    )
    for T, split, merge, nseg in [
        (0.2, 0.375447, 0.329890, 294518),
        (0.3, 0.452148, 0.242154, 322167),
        (0.4, 0.517995, 0.217998, 345376),
        (0.5, 0.610281, 0.209424, 379876),
    ]:
        rows.append(
            row(
                "heap_s4",
                "E4",
                "stock waterz S4",
                T,
                split,
                merge,
                nseg,
                162,
                True,
                "pass",
                "notes/LOG.md#e4-pass",
            )
        )

    # write CSV
    csv_path = CACHE / "voi_atlas.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="raise")
        w.writeheader()
        for r in rows:
            assert set(r.keys()) == set(FIELDS), r.keys()
            w.writerow(r)

    modes = sorted({r["fail_mode"] for r in rows})
    allowed = {
        "giant",
        "under_merge",
        "both_halves",
        "timeout",
        "order_change",
        "not_S1_fragments",
        "pass",
    }
    unknown = set(modes) - allowed
    if unknown:
        raise SystemExit(f"unknown fail_modes: {unknown}")

    meta = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "schema": FIELDS,
        "fail_modes": modes,
        "n_rows": len(rows),
        "n_task_legal_true": sum(1 for r in rows if r["task_legal"] == "True"),
        "n_task_legal_false": sum(1 for r in rows if r["task_legal"] == "False"),
        "pins": {
            "N19_H2": "data/cache/N19_H2.json",
            "N19_X2": "data/cache/N19_X2.json",
            "N19_I0_REPRO": "data/cache/N19_I0_REPRO.json",
            "N18_B2": "data/cache/n19_dead.jsonl",
            "LOG": "notes/LOG.md",
            "N12_OFF": "notes/N12_OFF.md",
            "N16_DEEP": "notes/N16_DEEP.md",
        },
        "rows": rows,
    }
    (CACHE / "voi_atlas.json").write_text(json.dumps(meta, indent=2) + "\n")

    # verify against pinned JSON
    by = {(r["probe_id"], r["param"], r["T"]): r for r in rows}
    r = by[("N19_H2", "7.5M->1.8M edges", "0.3")]
    assert abs(float(r["voi_split"]) - h2["split"]) < 1e-12
    assert abs(float(r["voi_merge"]) - h2["merge"]) < 1e-12
    assert int(r["nseg"]) == h2["nseg"]
    r = by[("N19_X2", "cached RAG", "0.3")]
    assert abs(float(r["voi_split"]) - x2["split"]) < 1e-12
    assert abs(float(r["voi_merge"]) - x2["merge"]) < 1e-12
    for eps, m in b2.items():
        r = by[("N18_B2", f"eps={eps}", "0.3")]
        assert abs(float(r["voi_split"]) - m["split"]) < 1e-12
        assert abs(float(r["voi_merge"]) - m["merge"]) < 1e-12
    r = by[("N19_I0_REPRO", "eps=0.40 T=0.3 speed", "0.3")]
    assert abs(float(r["voi_split"]) - float(i0["voi"]["m1"]["split"])) < 1e-12
    assert abs(float(r["voi_merge"]) - float(i0["voi"]["m1"]["merge"])) < 1e-12
    assert int(r["nseg"]) == int(i0["voi"]["m1"]["nseg"])
    print(
        f"OK n_rows={len(rows)} legal={meta['n_task_legal_true']} "
        f"illegal={meta['n_task_legal_false']} modes={modes}"
    )


if __name__ == "__main__":
    main()
