# N20 STAR EV-dead

N20 diagnostic; not a throughput claim. Do not rewrite CUDA.

ParHAC NeurIPS 2022 p.6 / SOURCES S43: full-graph update is wasteful when
rounds "only merge a small number of vertices".

Measured (device fingerprint `data/cache/p0aa_e6s.json`, eps=0.08):

- layer 0 merges 1322268 / 1853427 (71.3%) over 64 outers
- ~20660 merges/outer in layer 0
- eps=0.40 speed path (N19_I0_REPRO): 1853545 merges in 538 inners

N20_STAR = EV-dead. Paper 7-11x is Affinity/SCC, not ParHAC-vs-ParHAC.
Do not reopen AGG_E6t_StarMerge_default. GPU prep script records this
and refuses launch.
