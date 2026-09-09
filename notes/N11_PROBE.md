# N11 C: serial merge-cost probe

Val RAG. One device thread. Not a 2 Gvox/s number. Not 3090 Ti.
Kill if wall >= ParHAC val agg (202.0 ms).

## find+union (no neighbor rewrite)

- wall=2093.005 ms n_pop=4791399 n_merge=1853086
- ns/pop=436.82536951564146
- kill=True

## + neighbor walk

Skipped (find+union already >= ParHAC val agg).

## Verdict

GPU BinQueue dead. Skip E.
