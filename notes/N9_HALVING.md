# N9 Track B: ECL-CC path-halving killed

Not a 2 Gvox/s number. Not implemented. Not `WATERZ_UF_ALGO=1` (already slower).

Type D stop: 2.16 W5 WS **6536 ms** > 4000 ms. Path-halving inside hook is a 1.2-2x compress bet on the UF slice. Val W5 UF was 176 ms of 557 ms WS; 2.16 W5 kernels summed to 2117 ms of 6536 ms WS. Even a 2x on that slice leaves WS multi-second.

Default `WATERZ_UF_ALGO` stays 0.
