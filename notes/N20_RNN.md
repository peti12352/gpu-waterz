# N20 RNN

N20 legal S3 schedule; not a throughput claim. host=greengoblin.
Oracle is the S3 heap (`N20_D2_parents.npy`), not ParHAC.

- reused `N20_D3_parents.npy`; did not rerun the RNN
- heap_ms=2004580.3 rnn_ms=1116187.8 vs ParHAC 1679.9
- timing_usable_vs_1679=true (hogs empty); still not a GPU closer
- four_ok=False; parents_eq_heap=False all T
- T=0.2..0.5 split 0.856 / 0.863 / 0.879 / 0.929 (under-merge vs limits 0.40-0.63)
- rounds=5000 hit_cap all T (incomplete clustering, not a finished UPGMA)
- kill_as_closer=True; go_gpu=False; ran_216=False
