# B: delete unused cu/cv/csm/cct

Date: 2026-09-06. Not a 2 Gvox/s claim.

`parhac_e6s_dev` malloc'd four per-edge arrays that were never read
(24 B/edge). Val rag.npz:

- before 1.124 GiB, after 0.956 GiB (saved 0.168 GiB = 7 505 458 x 24)
- parents byte-identical (`a52a70236a43f9e99a9546482041ac82`)

At official 90.3 M edges this is 2.02 GiB off the AGG tracked peak.
