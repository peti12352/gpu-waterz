# N20 resource gate

Wall-clock vs 1679.9 ms is invalid if another project is on the CPU.
Parents / height / VOI / merge counts are still valid on a busy box.

Observed 2026-09-11 17:43 greengoblin during N20_D2:

- N20_D2 python 100% of one core, RSS 37.4 GiB
- lego `p1_real_comparison.py` spawning `leocad` + Xvfb bursts (~80-100% another core)
- vLLM Qwen3-VL-8B on GPU (26 GiB), ~2% CPU
- btop ~18% (ignored as a monitor)
- swap used ~405 MiB (lifetime; not used as an automatic kill)

Policy:

1. Snapshot `/proc/loadavg` + `ps` before and after every timed job (`scripts/n20_res.py`).
2. If `timing_contaminated`, keep correctness fields, set `timing_usable_vs_1679=false`, redo the wall later.
3. `taskset -c 2` for timed heaps so we do not share CPU0 with IRQs.
4. Do not kill the other project. Wait until `n20_res.wait_idle`.
5. GPU tests stay prep-only. vLLM currently owns the 5090 anyway.

N20_D2 wall_ms from the in-flight run is **contaminated**. Height and parents are kept.
