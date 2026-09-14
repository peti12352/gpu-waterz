# N19 I0 nsys

not a 2 Gvox/s number; not 3090 Ti. Parks off. Information only.

- e2e=3115.4560546875 stages={'ws': 1317.044184077531, 'rag': 85.5368827469647, 'agg': 1693.66952823475, 'extract': 18.53096066042781}
- kill_thresh=200.0 ms
- gates={
  "vcount": {
    "kernel_substr": "count_v",
    "max_ms": 148.142816,
    "ok_to_try": false,
    "kill_thresh_ms": 200.0,
    "e2e_ms": 3115.4560546875
  },
  "sort": {
    "kernel_substr": "RadixSort",
    "max_ms": 31.550445,
    "ok_to_try": false,
    "kill_thresh_ms": 200.0,
    "e2e_ms": 3115.4560546875
  },
  "hook": {
    "kernel_substr": "hook",
    "max_ms": 84.509482,
    "ok_to_try": false,
    "kill_thresh_ms": 200.0,
    "e2e_ms": 3115.4560546875
  },
  "hash": {
    "kernel_substr": "hash",
    "max_ms": 286.879773,
    "ok_to_try": true,
    "kill_thresh_ms": 200.0,
    "e2e_ms": 3115.4560546875
  },
  "stitch": {
    "kernel_substr": "stitch",
    "max_ms": 0.0,
    "ok_to_try": false,
    "kill_thresh_ms": 200.0,
    "e2e_ms": 3115.4560546875
  },
  "nvtx_vcount": {
    "max_ms": 177.533524,
    "ok_to_try": false
  },
  "nvtx_sort": {
    "max_ms": 35.524923,
    "ok_to_try": false
  },
  "nvtx_hash_insert": {
    "max_ms": 0.869406,
    "ok_to_try": false
  },
  "nvtx_propose": {
    "max_ms": 2.205588,
    "ok_to_try": false
  },
  "nvtx_rebuild": {
    "max_ms": 0.0,
    "ok_to_try": false
  }
}

## top kernels

- 508.0 ms (18.5%) `k_w5_compress_list(unsigned`
- 286.9 ms (14.9%) `:hash_rewrite`
- 278.4 ms (10.1%) `k_rebuild_active(const`
- 273.0 ms (9.9%) `k_rewrite_dirty_fuse(unsigned`
- 173.6 ms (6.3%) `k_hash_insert(const`
- 161.1 ms (5.9%) `k_hash_emit_holes(HSlot`
- 148.1 ms (5.4%) `k_count_v2(const`
- 146.8 ms (5.3%) `k_propose_listed(const`
- 114.5 ms (4.2%) `k_copy_sz_list(const`
- 100.2 ms (3.6%) `k_pack_amask(const`
- 84.5 ms (3.1%) `k_w5_hook_list<(bool)1>(const`
- 82.1 ms (3.0%) `k_w5_hook_list<(bool)0>(const`
- 77.4 ms (2.8%) `k_compact_roots(const`
- 52.6 ms (1.9%) `cub::CUB_200700_860_1200_NS::DeviceScanKernel<cub::CUB_200700_860_1200_NS::DeviceScanPolicy<un...`
- 47.9 ms (1.7%) `k_hash_faces(const`
- 46.9 ms (1.7%) `k_uf_tile_local<(bool)1>(const`
- 41.2 ms (1.5%) `k_uf_tile_local<(bool)0>(const`
- 37.7 ms (1.4%) `k_accept_reds(const`
- 31.6 ms (1.1%) `cub::CUB_200700_860_1200_NS::DeviceRadixSortOnesweepKernel<cub::CUB_200700_860_1200_NS::Device...`
- 25.1 ms (0.9%) `k_clear_frozen_list(unsigned`

## nvtx

- 835.3 ms `w5_stitch`
- 512.1 ms `e9c`
- 286.9 ms `hash_rewrite`
- 177.5 ms `vcount`
- 55.6 ms `compact_radix`
- 35.5 ms `sort`
- 6.0 ms `sort_prop`
- 4.9 ms `CCCL:cub::DeviceRadixSort`
- 4.6 ms `flow`
- 2.2 ms `propose`
- 1.2 ms `pack`
- 0.9 ms `hash_insert`
- 0.8 ms `accept`
- 0.3 ms `CCCL:cub::DeviceReduce::ReduceByKey`
- 0.2 ms `CCCL:cub::DeviceScan::ExclusiveSum`

```
## cuda_gpu_kern_sum
Generating SQLite file /tmp/n19_n17_216.sqlite from /tmp/n19_n17_216.nsys-rep
Processing [/tmp/n19_n17_216.sqlite] with [/opt/nvidia/nsight-systems/2024.6.2/host-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)    StdDev (ns)                                                   Name                                                
 --------  ---------------  ---------  ------------  ------------  ----------  -----------  ------------  ----------------------------------------------------------------------------------------------------
     18.5      508,013,208          6  84,668,868.0  84,469,978.5  69,124,113  100,337,441  16,822,075.2  k_w5_compress_list(unsigned int *, int *, const unsigned int *, int, int)                           
     10.1      278,440,219        438     635,708.3     313,614.0     126,815    1,828,628     518,663.2  k_rebuild_active(const unsigned int *, const unsigned int *, const double *, const long *, const un...
      9.9      272,950,646        438     623,175.0     383,342.0     148,991    4,653,217     537,663.6  k_rewrite_dirty_fuse(unsigned int *, unsigned int *, long *, const unsigned int *, const unsigned c...
      6.3      173,613,951        438     396,378.9     200,927.0      24,384   12,219,246     930,422.5  k_hash_insert(const unsigned int *, const unsigned int *, const double *, const long *, const unsig...
      5.9      161,122,414        438     367,859.4      42,272.0       1,408    6,765,011     672,688.6  k_hash_emit_holes(HSlot *, int, const unsigned int *, int, unsigned int *, unsigned int *, double *...
      5.4      148,142,816          3  49,380,938.7  49,060,087.0  48,853,017   50,229,712     742,315.0  k_count_v2(const unsigned char *, const unsigned int *, unsigned int *, long, long, long, int, int) 
      5.3      146,784,784        680     215,860.0       8,400.0       2,752    7,117,008     598,352.8  k_propose_listed(const unsigned int *, int, const unsigned int *, const unsigned int *, const doubl...
      4.2      114,511,686        246     465,494.7     478,445.0     174,943      543,708      61,544.5  k_copy_sz_list(const unsigned int *, unsigned int *, const unsigned int *, int)                     
      3.6      100,200,579        443     226,186.4      71,072.0      25,856    1,331,735     275,899.8  k_pack_amask(const unsigned char *, const long *, long, unsigned int *, int *)                      
      3.1       84,509,482         12   7,042,456.8   6,507,476.5   6,471,381    8,737,285     985,877.7  void k_w5_hook_list<(bool)1>(const unsigned char *, unsigned int *, int *, const unsigned int *, in...
      3.0       82,136,122         12   6,844,676.8   4,533,361.5   4,405,315   13,797,443   4,179,767.5  void k_w5_hook_list<(bool)0>(const unsigned char *, unsigned int *, int *, const unsigned int *, in...
      2.8       77,364,002        438     176,630.1     159,807.0     121,855      399,741      51,762.1  k_compact_roots(const unsigned int *, unsigned int *, const unsigned int *, int, int *)             
      1.9       52,554,496         18   2,919,694.2   3,771,927.0   1,144,312    3,780,999   1,245,147.6  void cub::CUB_200700_860_1200_NS::DeviceScanKernel<cub::CUB_200700_860_1200_NS::DeviceScanPolicy<un...
      1.7       47,879,648          1  47,879,648.0  47,879,648.0  47,879,648   47,879,648           0.0  k_hash_faces(const unsigned int *, const unsigned char *, long, long, long, Slot *, unsigned long)  
      1.7       46,881,989          3  15,627,329.7  15,626,039.0  15,623,351   15,632,599       4,757.2  void k_uf_tile_local<(bool)1>(const unsigned char *, unsigned int *, long, long, long)              
      1.5       41,150,541          3  13,716,847.0  13,724,324.0  13,695,205   13,731,012      19,038.5  void k_uf_tile_local<(bool)0>(const unsigned char *, unsigned int *, long, long, long)              
      1.4       37,670,835        438      86,006.5       4,912.0       1,984    1,344,919     212,908.1  k_accept_reds(const unsigned int *, const unsigned int *, const unsigned int *, int, unsigned int *...
      1.1       31,550,445         12   2,629,203.8   2,634,350.0   2,608,847    2,640,270      12,380.6  void cub::CUB_200700_860_1200_NS::DeviceRadixSortOnesweepKernel<cub::CUB_200700_860_1200_NS::Device...
      0.9       25,098,449        246     102,026.2      87,023.5      67,999      186,111      28,895.3  k_clear_frozen_list(unsigned char *, const unsigned int *, int)                                     
      0.9       23,878,722          9   2,653,191.3   2,579,534.0   2,471,856    2,906,636     195,108.2  k_scatter_idx_u32(const unsigned int *, unsigned int, unsigned int *, long)                         
      0.7       20,335,707         40     508,392.7     342,766.0     164,799    1,470,647     483,021.7  void cub::CUB_200700_860_1200_NS::DeviceRadixSortOnesweepKernel<cub::CUB_200700_860_1200_NS::Device...
      0.7       18,409,541          6   3,068,256.8   3,069,035.5   3,058,700    3,073,355       5,153.8  k_w5_list_flag(const unsigned int *, unsigned int *, long, long, long)                              
      0.7       17,921,194          5   3,584,238.8   1,641,494.0   1,054,201   11,692,498   4,554,316.4  k_gather_smct(const double *, const long *, const unsigned int *, double *, long *, long)           
      0.6       16,233,875          3   5,411,291.7   5,392,796.0   5,391,132    5,449,947      33,486.8  k_corner_flag(const unsigned char *, unsigned int *, long, long, long)                              
      0.6       15,660,536          3   5,220,178.7   5,217,118.0   5,216,605    5,226,813       5,751.2  k_write_labels(const unsigned char *, const unsigned int *, const unsigned int *, unsigned int *, l...
      0.6       15,194,459          3   5,064,819.7   5,064,670.0   5,063,582    5,066,207       1,318.9  k_root_flag(const unsigned char *, const unsigned int *, unsigned int *, long)                      
      0.5       14,223,937          3   4,741,312.3   4,722,561.0   4,716,608    4,784,768      37,751.2  k_indep_bfs(unsigned char *, const unsigned int *, const int *, unsigned int *, const unsigned int ...
      0.4       11,642,037          1  11,642,037.0  11,642,037.0  11,642,037   11,642,037           0.0  k_extract(const unsigned int *, const unsigned int *, unsigned int *, long)                         
      0.4       11,108,725          6   1,851,454.2   1,850,963.0   1,848,755    1,855,476       2,381.6  k_gather_u32(const unsigned int *, const unsigned int *, unsigned int *, int)                       
      0.4       11,108,408          6   1,851,401.3     982,906.0     559,709    5,070,494   1,792,544.2  k_wmax_live(const unsigned int *, const unsigned int *, const double *, const long *, unsigned int ...
      0.4       10,186,523          3   3,395,507.7   3,398,857.0   3,388,393    3,399,273       6,165.0  k_keys_from_parent_u32(const unsigned int *, const unsigned int *, unsigned int *, int, int)        
      0.4       10,109,248        968      10,443.4       6,432.0       4,192      169,215      17,201.2  void cub::CUB_200700_860_1200_NS::DeviceRadixSortOnesweepKernel<cub::CUB_200700_860_1200_NS::Device...
      0.4        9,856,958        684      14,410.8       1,504.0         864    1,130,808      80,585.1  k_pack_listed_fused(const unsigned int *, const int *, unsigned long long *, const unsigned int *, ...
      0.4        9,765,215          3   3,255,071.7   3,255,179.0   3,253,770    3,256,266       1,251.5  k_gather_vcount_u32(const unsigned int *, const unsigned int *, unsigned int *, int)                
      0.4        9,732,031          1   9,732,031.0   9,732,031.0   9,732,031    9,732,031           0.0  k_flow(const unsigned char *, long, long, long, float, float, unsigned char *, int, int)            
      0.3        9,417,282          5   1,883,456.4     783,611.0     454,589    5,675,066   2,188,784.7  k_rewrite(unsigned int *, unsigned int *, const double *, const long *, unsigned int *, long, unsig...
      0.3        8,853,477          2   4,426,738.5   4,426,738.5   4,406,499    4,446,978      28,623.0  k_offset_labels(unsigned int *, long, unsigned int)                                                 
      0.2        6,735,923          3   2,245,307.7   2,243,409.0   2,240,337    2,252,177       6,144.1  k_or40_u32(unsigned char *, const unsigned int *, int)                                              
      0.2        5,358,716          3   1,786,238.7   1,786,260.0   1,785,236    1,787,220         992.2  k_scatter_plat(const unsigned int *, const unsigned int *, int *, int)                              
      0.2        5,164,414          3   1,721,471.3   1,720,148.0   1,719,509    1,724,757       2,863.4  k_plat_meta(const int *, const unsigned int *, unsigned int *, int, int)                            
      0.2        4,751,321        312      15,228.6      15,040.0      14,720       17,920         540.8  void k_sort_prop_block<(int)256, (int)8>(const unsigned long long *, const unsigned long long *, un...
      0.2        4,692,160          1   4,692,160.0   4,692,160.0   4,692,160    4,692,160           0.0  k_scatter_edges(const Slot *, unsigned long, const unsigned int *, unsigned int *, unsigned int *, ...
      0.1        3,790,919          3   1,263,639.7   1,264,503.0   1,261,112    1,265,304       2,225.4  k_run_start(const unsigned int *, unsigned int *, int)                                              
      0.1        3,650,153          5     730,030.6     541,949.0     285,278    1,940,723     686,994.9  k_init_amask(const double *, const long *, long, double, unsigned char *)                           
      0.1        3,360,969          5     672,193.8     465,853.0     239,678    1,902,579     695,592.0  void cub::CUB_200700_860_1200_NS::DeviceReduceByKeyKernel<cub::CUB_200700_860_1200_NS::detail::devi...
      0.1        3,337,705          1   3,337,705.0   3,337,705.0   3,337,705    3,337,705           0.0  k_count_occ(const Slot *, unsigned long, unsigned int *)                                            
      0.1        3,293,611        438       7,519.7       1,152.0         992      314,206      30,254.2  k_compress_list(unsigned int *, const unsigned int *, int)                                          
      0.1        3,228,939          5     645,787.8     439,901.0     209,759    1,875,091     695,273.8  void cub::CUB_200700_860_1200_NS::DeviceReduceByKeyKernel<cub::CUB_200700_860_1200_NS::detail::devi...
      0.1        2,933,935        438       6,698.5       1,056.0         800      247,294      24,336.7  k_zero_hole_tail(const unsigned int *, int, int, long *)                                            
      0.1        2,806,099        438       6,406.6       1,856.0       1,440      185,023      17,061.2  k_mark_acc_blue(const unsigned int *, const unsigned int *, int, const unsigned int *, unsigned cha...
      0.1        2,554,799          1   2,554,799.0   2,554,799.0   2,554,799    2,554,799           0.0  k_hash_clear(HSlot *, int)                                                                          
      0.1        2,167,090          5     433,418.0     276,862.0     129,663    1,168,121     424,446.7  k_key_from_sel(const unsigned int *, const unsigned int *, const unsigned int *, unsigned long long...
      0.1        1,832,532          3     610,844.0     610,268.0     609,948      612,316       1,284.8  void cub::CUB_200700_860_1200_NS::DeviceRadixSortHistogramKernel<cub::CUB_200700_860_1200_NS::Devic...
      0.1        1,685,398          3     561,799.3     561,789.0     560,060      563,549       1,744.5  k_iota_u32(unsigned int *, int)                                                                     
      0.1        1,682,132          5     336,426.4     232,542.0     119,551      948,890     346,240.4  k_unpack_uvkey(const unsigned long long *, unsigned int *, unsigned int *, long)                    
      0.1        1,663,957          4     415,989.3     406,333.5     289,470      561,820     127,906.5  k_propose_bluelist(const unsigned int *, const unsigned int *, const double *, const long *, unsign...
      0.1        1,610,197         11     146,381.5     116,767.0      84,640      463,069     105,762.5  k_compress(unsigned int *, int)                                                                     
      0.1        1,403,256        438       3,203.8       1,472.0       1,056      140,511      10,392.1  k_freeze_reds(const unsigned int *, const int *, unsigned int *, const unsigned int *, unsigned cha...
      0.0        1,367,511          1   1,367,511.0   1,367,511.0   1,367,511    1,367,511           0.0  void cub::CUB_200700_1200_NS::DeviceScanKernel<cub::CUB_200700_1200_NS::DeviceScanPolicy<unsigned i...
      0.0          948,634          1     948,634.0     948,634.0     948,634      948,634           0.0  k_scale_sm_bytes(double *, long)                                                                    
      0.0          897,689        438       2,049.5       1,312.0         928      135,999       7,511.1  k_unpack_prop(const unsigned long long *, const unsigned long long *, unsigned int *, unsigned int ...
      0.0          760,381          5     152,076.2     101,632.0      42,336      458,461     173,467.5  void cub::CUB_200700_860_1200_NS::DeviceRadixSortHistogramKernel<cub::CUB_200700_860_1200_NS::Devic...
      0.0          646,300          1     646,300.0     646,300.0     646,300      646,300           0.0  void cub::CUB_200700_1200_NS::DeviceReduceKernel<cub::CUB_200700_1200_NS::DeviceReducePolicy<unsign...
      0.0          587,037          5     117,407.4      53,823.0      30,880      281,151     107,571.2  void cub::CUB_200700_860_1200_NS::DeviceSelectSweepKernel<cub::CUB_200700_860_1200_NS::detail::devi...
      0.0          574,363        121       4,746.8       2,464.0       2,336       42,815       6,047.7  void cub::CUB_200700_860_1200_NS::DeviceRadixSortHistogramKernel<cub::CUB_200700_860_1200_NS::Devic...
      0.0          122,143          1     122,143.0     122,143.0     122,143      122,143           0.0  k_init_parent(unsigned int *, unsigned int *, int)                                                  
      0.0          111,135        121         918.5         928.0         895          992          19.2  void cub::CUB_200700_860_1200_NS::DeviceRadixSortExclusiveSumKernel<cub::CUB_200700_860_1200_NS::De...
      0.0           92,544          3      30,848.0      29,792.0      24,448       38,304       6,988.1  k_clear_slab_z_faces(unsigned char *, long, long, long)                                             
      0.0           80,158          5      16,031.6      16,096.0      15,807       16,128         136.0  void cub::CUB_200700_860_1200_NS::DeviceRadixSortSingleTileKernel<cub::CUB_200700_860_1200_NS::Devi...
      0.0           55,103          1      55,103.0      55,103.0      55,103       55,103           0.0  k_init_root_list(unsigned int *, int)                                                               
      0.0           23,136         18       1,285.3       1,376.0         832        1,568         219.0  void cub::CUB_200700_860_1200_NS::DeviceScanInitKernel<cub::CUB_200700_860_1200_NS::ScanTileState<u...
      0.0            5,280          5       1,056.0       1,120.0         704        1,216         201.1  void cub::CUB_200700_860_1200_NS::DeviceCompactInitKernel<cub::CUB_200700_860_1200_NS::ReduceByKeyS...
      0.0            5,216          5       1,043.2       1,088.0         896        1,088          83.4  void cub::CUB_200700_860_1200_NS::DeviceCompactInitKernel<cub::CUB_200700_860_1200_NS::ScanTileStat...
      0.0            4,992          5         998.4         928.0         928        1,120          97.1  void cub::CUB_200700_860_1200_NS::DeviceRadixSortExclusiveSumKernel<cub::CUB_200700_860_120
```
