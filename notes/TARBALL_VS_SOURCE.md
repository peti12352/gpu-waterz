# Tarball README vs SOURCES S1-S5

Compared `data/ws_bounty/README.md` ("Exact semantics, verified against the waterz source")
to SOURCES.md S1-S5 / funkey/waterz a0184d2.

**No semantic disagreement.** TASK.md listing still owns gates.

| Topic | Tarball README | S1-S5 | Delta |
|---|---|---|---|
| 6-neigh, OOB=`low`, `m>low` flow | same | S1 | none |
| bits `aff==m \|\| aff>=high` | same | S1 | none |
| plateau BFS from corners | same | S1 | none |
| RAG 3 negative dirs, drop id1=0 | same; notes faces are accumulated then dropped | S2 | wording only |
| mean = running / area-weighted | same | S3 | none |
| heap, stale, keep cheaper | same | S4 | none |
| VOI `H(seg\|gt)` / `H(gt\|seg)`, skip `gt==0` | same; adds: pred label 0 is an ordinary label | S5 | extra note, not a conflict |
| score = 1-mean | same | S4 + THRESHOLD.md | none |

Extra files vs TASK.md listing (not a fail): `make_viz.py`; baseline labels also at aff 0.1/0.7/0.9.

`voi.csv` vs TASK.md 4-decimal table (rounded, match):

| aff | csv split | csv merge | TASK split | TASK merge |
|---|---|---|---|---|
| 0.2 | 0.37785188 | 0.33247380 | 0.3779 | 0.3325 |
| 0.3 | 0.45379016 | 0.24106186 | 0.4538 | 0.2411 |
| 0.4 | 0.51783094 | 0.21808035 | 0.5178 | 0.2181 |
| 0.5 | 0.61093103 | 0.20931783 | 0.6109 | 0.2093 |

Shapes: aff `[3,125,1200,1200]` uint8 dataset `affinity`; gt `[125,1200,1200]` uint32 dataset `gt`.
