# N18 A0 nsys (N17 env stack)

not a 2 Gvox/s number; not 3090 Ti. Parks off. Information only: not a timed gate substitute.

- gpu: `NVIDIA GeForce RTX 5090, 17928 MiB, 14182 MiB, 0 %`
- nsys val rc=0 2.16 rc=1
- 2.16 stages={'ws': 1311.435337178409, 'rag': 85.31448105350137, 'agg': 1933.3693240769207, 'extract': 19.338061101734638} e2e=3350.21533203125
- micro-opt kill rule: max kernel <200.0 ms of e2e -> skip as closer

## 2.16 top kernel lines


## val dump (trim)

```
## cuda_gpu_kern_sum


WARNING: Existing SQLite export found: /tmp/n18_n17_val.sqlite
         File is older than input file: /tmp/n18_n17_val.nsys-rep
         Use --force-export=true to update export file.

usage: nsys stats [<args>] <input-file>
Try 'nsys stats --help' for more information.

## nvtx_sum


WARNING: Existing SQLite export found: /tmp/n18_n17_val.sqlite
         File is older than input file: /tmp/n18_n17_val.nsys-rep
         Use --force-export=true to update export file.

usage: nsys stats [<args>] <input-file>
Try 'nsys stats --help' for more information.

```

## 2.16 dump (trim)

```
## cuda_gpu_kern_sum


WARNING: Existing SQLite export found: /tmp/n18_n17_216.sqlite
         File is older than input file: /tmp/n18_n17_216.nsys-rep
         Use --force-export=true to update export file.

usage: nsys stats [<args>] <input-file>
Try 'nsys stats --help' for more information.

## nvtx_sum


WARNING: Existing SQLite export found: /tmp/n18_n17_216.sqlite
         File is older than input file: /tmp/n18_n17_216.nsys-rep
         Use --force-export=true to update export file.

usage: nsys stats [<args>] <input-file>
Try 'nsys stats --help' for more information.

```
