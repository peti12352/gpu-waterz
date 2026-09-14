# N12 W5 tile A/B

Not a 2 Gvox/s claim. Not 3090 Ti. All identity True vs wz_fragments.

| tile | shmem | val stitch_ms | 2.16 WS ms | vs 2582 | keep default |
|---|---|---|---|---|---|
| 8x16x32 | 20 KB | 168.12 | 2576 | 0.998x | baseline |
| 16x16x32 | 40 KB | 154.38 | 2470 | 0.957x | no (need ≤0.85x) |
| 8x32x32 | 40 KB | 154.76 | 2486 | 0.963x | no |

16x16x32 cuts e9c list 0.396->0.293 as the comment predicted; tile_local occupancy tax (27 ms vs 15 ms on 2.16 e9b) eats the stitch win. Binaries kept as `src/libws_gpu_*`. Default stays **8x16x32**.
