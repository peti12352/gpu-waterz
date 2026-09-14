# B: device-path aff park after k_flow

Date: 2026-09-06. Not a 2 Gvox/s claim.

`ws_flow_d` / `ws_label_d` split. `segment_d` parks owned / caller `DevBuf`
aff after `k_flow` (E5: unread until RAG), restores before `rag_gpu_d`.
External tensors are left resident.

`data/cache/b_dev_aff.json` (idle 5090, val):

- nfrag 2175400, `wz_fragments.npy` **array_equal=True**
- held vs parked labels identical
- tracked peak **2.698 -> 2.195 GiB** (saved 0.503 GiB = val aff)
- same byte cut as host `e9c_watershed` free-aff

Fused 2.16 Gvox WS still needs remaining W2 levers; do not run 2.16.
