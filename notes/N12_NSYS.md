# N12 nsys

Not a 2 Gvox/s claim. Not 3090 Ti. Parks-off UF=3.

**Owner:** 2.16 WS is `w5_stitch` (NVTX 52.8%, kernels `k_w5_compress_list` 21.1% + `k_uf_compress_c` 18.0% of GPU). Agg NVTX is `hash_rewrite` 8.1%; `k_hash_insert` is **3.4%** of kernels, `compact_radix` **1.4%** NVTX. Hash/radix do **not** own 2229 ms. cudaMemcpy API 76.6% (D2H sync tax).

- ident identity True, nfrag=2175400, bg=506568
- 2.16 e2e=4943.3 ms (N11 4913.5, +0.6%) WS=2591 RAG=84 agg=2248
- val e2e=451.3 ms WS=224 RAG=14 agg=211

## 2.16 NVTX (share of NVTX time)

- w5_stitch 52.8%
- e9c 32.1%
- hash_rewrite 8.1%
- vcount 4.0%
- compact_radix 1.4%
- hash_insert ~0%

## 2.16 top kernel lines

- `21.1      968,594,195         30   32,286,473.2    9,947,110.0    3,471,180  101,297,680  36,101,924.7  k_w5_compress_list(unsigned int *, int *, const unsigned int *, int)`
- `18.0      823,668,583          6  137,278,097.2  137,291,037.5  112,739,942  161,789,103  26,673,249.4  k_uf_compress_c(unsigned int *, int *, long)`
- `6.0      275,552,729        438      629,115.8      304,270.0      129,247    1,774,070     518,781.7  k_rebuild_active(const unsigned int *, const unsigned int *, const double *, const long *, const un…`
- `5.3      241,265,729        438      550,835.0      297,951.0      129,855    4,313,800     478,120.5  k_rewrite_dirty(unsigned int *, unsigned int *, long *, const unsigned int *, const unsigned char *…`
- `5.1      233,220,103        438      532,466.0      214,927.0       63,968    1,365,689     509,529.5  k_list_holes(const long *, long, unsigned int *, int *)`
- `4.9      226,243,150        438      516,536.9      296,206.5      158,111    1,192,442     393,412.4  k_count_live(const long *, long, int *)`
- `3.4      155,270,092        289      537,266.8      259,647.0       33,536   12,186,748   1,111,251.2  k_hash_insert(const unsigned int *, const unsigned int *, const double *, const long *, const unsig…`
- `3.2      145,392,011          3   48,464,003.7   48,530,306.0   47,816,744   49,044,961     616,787.0  k_count_v2(const unsigned char *, const unsigned int *, unsigned int *, long, long, long)`
- `3.1      140,288,695        680      206,306.9        8,304.0        2,720    7,080,761     586,961.6  k_propose_listed(const unsigned int *, int, const unsigned int *, const unsigned int *, const doubl…`
- `2.5      113,203,533        246      460,177.0      475,629.0      176,575      532,893      57,025.6  k_copy_sz_list(const unsigned int *, unsigned int *, const unsigned int *, int)`
- `2.4      107,674,667        438      245,832.6       66,912.0        1,216    2,928,239     375,987.9  k_fill_holes(const unsigned int *, const unsigned int *, const double *, const long *, const unsign…`
- `2.0       93,374,523         15    6,224,968.2    5,510,495.0    5,468,639    9,079,883   1,453,475.7  void k_w5_hook_list<(bool)1>(const unsigned char *, unsigned int *, int *, const unsigned int *, in…`
- `2.0       91,676,380        443      206,944.4       71,103.0       25,696    1,336,249     245,662.0  k_pack_amask(const unsigned char *, const long *, long, unsigned int *, int *)`
- `1.9       87,358,270         15    5,823,884.7    3,605,291.0    3,589,131   14,055,118   3,988,438.3  void k_w5_hook_list<(bool)0>(const unsigned char *, unsigned int *, int *, const unsigned int *, in…`
- `1.7       77,327,948        438      176,547.8      159,647.0      121,663      400,894      52,278.6  k_compact_roots(const unsigned int *, unsigned int *, const unsigned int *, int, int *)`
- `1.7       76,101,222        438      173,747.1      142,479.0       24,448    1,199,834     165,440.0  k_count_u8(const unsigned char *, long, int *)`
- `1.2       57,134,694        438      130,444.5       81,327.5       21,696      919,707     135,844.2  k_vacate_kept(long *, const unsigned char *, long)`
- `1.2       57,020,036        289      197,301.2       49,952.0        3,712    5,027,107     426,448.9  k_hash_emit(HSlot *, int, unsigned int *, unsigned int *, double *, long *, int *)`
- `1.2       53,934,499         19    2,838,657.8    3,770,410.0    1,142,617    3,783,753   1,260,440.8  void cub::CUB_200700_1200_NS::DeviceScanKernel<cub::CUB_200700_1200_NS::DeviceScanPolicy<unsigned i…`
- `1.0       47,894,376          1   47,894,376.0   47,894,376.0   47,894,376   47,894,376           0.0  k_hash_faces(const unsigned int *, const unsigned char *, long, long, long, Slot *, unsigned long)`

## val NVTX/kernel dump

```
## cuda_gpu_kern_sum


WARNING: Existing SQLite export found: /tmp/n12_val.sqlite
         File is older than input file: /tmp/n12_val.nsys-rep
         Use --force-export=true to update export file.

usage: nsys stats [<args>] <input-file>
Try 'nsys stats --help' for more information.

## nvtx_sum


WARNING: Existing SQLite export found: /tmp/n12_val.sqlite
         File is older than input file: /tmp/n12_val.nsys-rep
         Use --force-export=true to update export file.

usage: nsys stats [<args>] <input-file>
Try 'nsys stats --help' for more information.

## cuda_api_sum


WARNING: Existing SQLite export found: /tmp/n12_val.sqlite
         File is older than input file: /tmp/n12_val.nsys-rep
         Use --force-export=true to update export file.

usage: nsys stats [<args>] <input-file>
Try 'nsys stats --help' for more information.

```

## 2.16 NVTX/kernel dump

```
## cuda_gpu_kern_sum

NOTICE: Existing SQLite export found: /tmp/n12_216.sqlite
        It is assumed file was previously exported from: /tmp/n12_216.nsys-rep
        Consider using --force-export=true if needed.

Processing [/tmp/n12_216.sqlite] with [/opt/nvidia/nsight-systems/2024.6.2/host-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)                                                   Name                                                
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  ------------  ----------------------------------------------------------------------------------------------------
     21.1      968,594,195         30   32,286,473.2    9,947,110.0    3,471,180  101,297,680  36,101,924.7  k_w5_compress_list(unsigned int *, int *, const unsigned int *, int)                                
     18.0      823,668,583          6  137,278,097.2  137,291,037.5  112,739,942  161,789,103  26,673,249.4  k_uf_compress_c(unsigned int *, int *, long)                                                        
      6.0      275,552,729        438      629,115.8      304,270.0      129,247    1,774,070     518,781.7  k_rebuild_active(const unsigned int *, const unsigned int *, const double *, const long *, const un…
      5.3      241,265,729        438      550,835.0      297,951.0      129,855    4,313,800     478,120.5  k_rewrite_dirty(unsigned int *, unsigned int *, long *, const unsigned int *, const unsigned char *…
      5.1      233,220,103        438      532,466.0      214,927.0       63,968    1,365,689     509,529.5  k_list_holes(const long *, long, unsigned int *, int *)                                             
      4.9      226,243,150        438      516,536.9      296,206.5      158,111    1,192,442     393,412.4  k_count_live(const long *, long, int *)                                                             
      3.4      155,270,092        289      537,266.8      259,647.0       33,536   12,186,748   1,111,251.2  k_hash_insert(const unsigned int *, const unsigned int *, const double *, const long *, const unsig…
      3.2      145,392,011          3   48,464,003.7   48,530,306.0   47,816,744   49,044,961     616,787.0  k_count_v2(const unsigned char *, const unsigned int *, unsigned int *, long, long, long)           
      3.1      140,288,695        680      206,306.9        8,304.0        2,720    7,080,761     586,961.6  k_propose_listed(const unsigned int *, int, const unsigned int *, const unsigned int *, const doubl…
      2.5      113,203,533        246      460,177.0      475,629.0      176,575      532,893      57,025.6  k_copy_sz_list(const unsigned int *, unsigned int *, const unsigned int *, int)                     
      2.4      107,674,667        438      245,832.6       66,912.0        1,216    2,928,239     375,987.9  k_fill_holes(const unsigned int *, const unsigned int *, const double *, const long *, const unsign…
      2.0       93,374,523         15    6,224,968.2    5,510,495.0    5,468,639    9,079,883   1,453,475.7  void k_w5_hook_list<(bool)1>(const unsigned char *, unsigned int *, int *, const unsigned int *, in…
      2.0       91,676,380        443      206,944.4       71,103.0       25,696    1,336,249     245,662.0  k_pack_amask(const unsigned char *, const long *, long, unsigned int *, int *)                      
      1.9       87,358,270         15    5,823,884.7    3,605,291.0    3,589,131   14,055,118   3,988,438.3  void k_w5_hook_list<(bool)0>(const unsigned char *, unsigned int *, int *, const unsigned int *, in…
      1.7       77,327,948        438      176,547.8      159,647.0      121,663      400,894      52,278.6  k_compact_roots(const unsigned int *, unsigned int *, const unsigned int *, int, int *)             
      1.7       76,101,222        438      173,747.1      142,479.0       24,448    1,199,834     165,440.0  k_count_u8(const unsigned char *, long, int *)                                                      
      1.2       57,134,694        438      130,444.5       81,327.5       21,696      919,707     135,844.2  k_vacate_kept(long *, const unsigned char *, long)                                                  
      1.2       57,020,036        289      197,301.2       49,952.0        3,712    5,027,107     426,448.9  k_hash_emit(HSlot *, int, unsigned int *, unsigned int *, double *, long *, int *)                  
      1.2       53,934,499         19    2,838,657.8    3,770,410.0    1,142,617    3,783,753   1,260,440.8  void cub::CUB_200700_1200_NS::DeviceScanKernel<cub::CUB_200700_1200_NS::DeviceScanPolicy<unsigned i…
      1.0       47,894,376          1   47,894,376.0   47,894,376.0   47,894,376   47,894,376           0.0  k_hash_faces(const unsigned int *, const unsigned char *, long, long, long, Slot *, unsigned long)  
      1.0       46,939,371          3   15,646,457.0   15,665,636.0   15,607,267   15,666,468      33,942.1  void k_uf_tile_local<(bool)1>(const unsigned char *, unsigned int *, long, long, long)              
      0.9       41,282,606          3   13,760,868.7   13,774,191.0   13,727,856   13,780,559      28,766.6  void k_uf_tile_local<(bool)0>(const unsigned char *, unsigned int *, long, long, long)              
      0.8       37,651,128        438       85,961.5        4,960.0        2,016    1,345,848     213,294.9  k_accept_reds(const unsigned int *, const unsigned int *, const unsigned int *, int, unsigned int *…
      0.7       31,563,687         12    2,630,307.3    2,634,192.5    2,604,913    2,643,281      12,203.5  void cub::CUB_200700_1200_NS::DeviceRadixSortOnesweepKernel<cub::CUB_200700_1200_NS::DeviceRadixSor…
      0.5       25,127,084        246      102,142.6       86,943.0       68,224      186,303      28,993.2  k_clear_frozen_list(unsigned char *, const unsigned int *, int)                                     
      0.5       24,821,034        864       28,728.0        5,376.0        4,479    1,468,344     147,403.2  void cub::CUB_200700_1200_NS::DeviceRadixSortOnesweepKernel<cub::CUB_200700_1200_NS::DeviceRadixSor…
      0.5       23,875,508          9    2,652,834.2    2,581,777.0    2,470,802    2,906,863     193,840.9  k_scatter_idx_u32(const unsigned int *, unsigned int, unsigned int *, long)                         
      0.4       18,419,700          6    3,069,950.0    3,068,062.0    3,063,534    3,081,070       7,110.3  k_w5_list_flag(const unsigned int *, unsigned int *, long, long, long)                              
      0.4       18,184,758        154      118,082.8        2,400.0        1,472   11,692,284     972,034.7  k_gather_smct(const double *, const long *, const unsigned int *, double *, long *, long)           
      0.4       16,252,224          3    5,417,408.0    5,403,008.0    5,390,496    5,458,720      36,320.1  k_corner_flag(const unsigned char *, unsigned int *, long, long, long)                              
      0.3       15,210,278          3    5,070,092.7    5,072,002.0    5,059,842    5,078,434       9,441.9  k_root_flag(const unsigned char *, const unsigned int *, unsigned int *, long)                      
      0.3       14,252,365          3    4,750,788.3    4,750,533.0    4,749,892    4,751,940       1,047.6  k_write_labels(const unsigned char *, const unsigned int *, const unsigned int *, unsigned int *, l…
      0.3       14,239,437          3    4,746,479.0    4,726,917.0    4,722,468    4,790,052      37,800.8  k_indep_bfs(unsigned char *, const unsigned int *, const int *, unsigned int *, const unsigned int …
      0.3       11,641,761          1   11,641,761.0   11,641,761.0   11,641,761   11,641,761           0.0  k_extract(const unsigned int *, const unsigned int *, unsigned int *, long)                         
      0.2       11,091,038          6    1,848,506.3    1,850,181.0    1,839,765    1,853,749       4,818.0  k_gather_u32(const unsigned int *, const unsigned int *, unsigned int *, int)                       
      0.2       10,896,928          6    1,816,154.7      968,346.5      555,677    5,075,618   1,781,414.9  k_wmax_live(const unsigned int *, const unsigned int *, const double *, const long *, unsigned int …
      0.2       10,125,611        968       10,460.3        6,464.0        4,224      168,479      17,221.8  void cub::CUB_200700_1200_NS::DeviceRadixSortOnesweepKernel<cub::CUB_200700_1200_NS::DeviceRadixSor…
      0.2        9,868,617        684       14,427.8        1,504.0          800    1,080,250      80,139.0  k_pack_listed_fused(const unsigned int *, const int *, unsigned long long *, const unsigned int *, …
      0.2        9,651,701        154       62,673.4       35,504.0       18,592      279,710      53,615.8  void cub::CUB_200700_1200_NS::DeviceSelectSweepKernel<cub::CUB_200700_1200_NS::detail::device_selec…
      0.2        9,369,256          3    3,123,085.3    3,128,493.0    3,107,182    3,133,581      14,005.7  k_keys_from_parent_u32(const unsigned int *, const unsigned int *, unsigned int *, int)             
      0.2        9,336,137          1    9,336,137.0    9,336,137.0    9,336,137    9,336,137           0.0  k_flow(const unsigned char *, long, long, long, float, float, unsigned char *)                      
      0.2        9,257,834          5    1,851,566.8      782,300.0      449,501    5,653,919   2,184,154.2  k_rewrite(unsigned int *, unsigned int *, const double *, const long *, unsigned int *, long, unsig…
      0.2        8,894,956          2    4,447,478.0    4,447,478.0    4,436,582    4,458,374      15,409.3  k_offset_labels(unsigned int *, long, unsigned int)                                                 
      0.2        7,313,397          3    2,437,799.0    2,437,329.0    2,436,402    2,439,666       1,682.0  k_gather_vcount_u32(const unsigned int *, const unsigned int *, unsigned int *, int)                
      0.1        6,739,735          3    2,246,578.3    2,245,587.0    2,242,354    2,251,794       4,797.4  k_or40_u32(unsigned char *, const unsigned int *, int)                                              
      0.1        5,356,544          3    1,785,514.7    1,784,949.0    1,784,662    1,786,933       1,236.7  k_scatter_plat(const unsigned int *, const unsigned int *, int *, int)                              
      0.1        5,155,394          3    1,718,464.7    1,718,390.0    1,717,750    1,719,254         754.8  k_plat_meta(const int *, const unsigned int *, unsigned int *, int, int)                            
      0.1        4,757,061        312       15,247.0       15,072.0       14,752       17,920         540.7  void k_sort_prop_block<(int)256, (int)8>(const unsigned long long *, const unsigned long long *, un…
      0.1        4,690,757          1    4,690,757.0    4,690,757.0    4,690,757    4,690,757           0.0  k_scatter_edges(const Slot *, unsigned long, const unsigned int *, unsigned int *, unsigned int *, …
      0.1        4,123,689        154       26,777.2        6,448.0        4,832    1,870,133     159,526.5  void cub::CUB_200700_1200_NS::DeviceReduceByKeyKernel<cub::CUB_200700_1200_NS::detail::device_reduc…
      0.1        3,872,524        154       25,146.3        3,760.0        2,272    1,901,525     163,657.3  void cub::CUB_200700_1200_NS::DeviceReduceByKeyKernel<cub::CUB_200700_1200_NS::detail::device_reduc…
      0.1        3,784,779          3    1,261,593.0    1,263,801.0    1,254,809    1,266,169       5,993.2  k_run_start(const unsigned int *, unsigned int *, int)                                              
      0.1        3,653,834          5      730,766.8      541,373.0      284,670    1,946,004     689,490.0  k_init_amask(const double *, const long *, long, double, unsigned char *)                           
      0.1        3,343,565          1    3,343,565.0    3,343,565.0    3,343,565    3,343,565           0.0  k_count_occ(const Slot *, unsigned long, unsigned int *)                                            
      0.1        3,315,338        438        7,569.3        1,184.0          992      312,126      30,303.1  k_compress_list(unsigned int *, const unsigned int *, int)                                          
      0.1        2,803,982        438        6,401.8        1,824.0        1,472      184,735      17,065.4  k_mark_acc_blue(const unsigned int *, const unsigned int *, int, const unsigned int *, unsigned cha…
      0.1        2,531,313          1    2,531,313.0    2,531,313.0    2,531,313    2,531,313           0.0  k_hash_clear(HSlot *, int)                                                                          
      0.1        2,465,393        154       16,009.0        2,176.0        1,152    1,166,777     101,983.4  k_key_from_sel(const unsigned int *, const unsigned int *, const unsigned int *, unsigned long long…
      0.0        1,832,693          3      610,897.7      611,004.0      609,404      612,285       1,443.4  void cub::CUB_200700_1200_NS::DeviceRadixSortHistogramKernel<cub::CUB_200700_1200_NS::DeviceRadixSo…
      0.0        1,804,183        154       11,715.5          800.0          768      949,403      81,862.9  k_unpack_uvkey(const unsigned long long *, unsigned int *, unsigned int *, long)                    
      0.0        1,696,823          3      565,607.7      566,429.0      562,749      567,645       2,549.2  k_iota_u32(unsigned int *, int)                                                                     
      0.0        1,636,855          4      409,213.8      394,686.0      286,398      561,085     126,575.1  k_propose_bluelist(const unsigned int *, const unsigned int *, const double *, const long *, unsign…
      0.0        1,614,619         11      146,783.5      116,928.0       84,896      466,238     106,680.2  k_compress(unsigned int *, int)                                                                     
      0.0        1,403,958        438        3,205.4        1,472.0        1,088      140,256      10,391.0  k_freeze_reds(const unsigned int *, const int *, unsigned int *, const unsigned int *, unsigned cha…
      0.0        1,010,009        108        9,351.9        2,400.0        2,336      454,429      45,778.8  void cub::CUB_200700_1200_NS::DeviceRadixSortHistogramKernel<cub::CUB_200700_1200_NS::DeviceRadixSo…
      0.0          946,427          1      946,427.0      946,427.0      946,427      946,427           0.0  k_scale_sm_bytes(double *, long)                                                                    
      0.0          895,511        438        2,044.5        1,280.0          959      136,959       7,530.8  k_unpack_prop(const unsigned long long *, const unsigned long long *, unsigned int *, unsigned int …
      0.0          646,620          1      646,620.0      646,620.0      646,620      646,620           0.0  void cub::CUB_200700_1200_NS::DeviceReduceKernel<cub::CUB_200700_1200_NS::DeviceReducePolicy<unsign…
      0.0          627,355         46       13,638.2       13,504.0       13,312       14,880         358.0  void cub::CUB_200700_1200_NS::DeviceRadixSortSingleTileKernel<cub::CUB_200700_1200_NS::DeviceRadixS…
      0.0          581,854        121        4,808.7        2,464.0        2,336       43,200       6,286.0  void cub::CUB_200700_1200_NS::DeviceRadixSortHistogramKernel<cub::CUB_200700_1200_NS::DeviceRadixSo…
      0.0          118,272          1      118,272.0      118,272.0      118,272      118,272           0.0  k_init_parent(unsigned int *, unsigned int *, int)                                                  
      0.0          113,407        121          937.2          928.0          927          960          14.6  void cub::CUB_200700_1200_NS::DeviceRadixSortExclusiveSumKernel<cub::CUB_200700_1200_NS::DeviceRadi…
      0.0          110,527        154          717.7          704.0          672        1,152          70.1  void cub::CUB_200700_1200_NS::DeviceCompactInitKernel<cub::CUB_200700_1200_NS::ReduceByKeyScanTileS…
      0.0          110,495        154          717.5          704.0          672        1,184          69.6  void cub::CUB_200700_1200_NS::DeviceCompactInitKernel<cub::CUB_200700_1200_NS::ScanTileState<int, (…
      0.0          108,703        154          705.9          704.0          672          800          20.8  void cub::CUB_200700_1200_NS::DeviceCompactInitKernel<cub::CUB_200700_1200_NS::ReduceByKeyScanTileS…
      0.0          100,222        108          928.0          928.0          896        1,152          33.6  void cub::CUB_200700_1200_NS::DeviceRadixSortExclusiveSumKernel<cub::CUB_200700_1200_NS::DeviceRadi…
      0.0           95,168          3       31,722.7       33,824.0       24,704       36,640       6,239.3  k_clear_slab_z_faces(unsigned char *, long, long, long)                                             
      0.0           79,967          5       15,993.4       16,032.0       15,776       16,128         130.8  void cub::CUB_200700_1200_NS::DeviceRadixSortSingleTileKernel<cub::CUB_200700_1200_NS::DeviceRadixS…
      0.0           57,919          1       57,919.0       57,919.0       57,919       57,919           0.0  k_init_root_list(unsigned int *, int)                                                               
      0.0           24,704         19        1,300.2        1,408.0          832        1,472         217.4  void cub::CUB_200700_1200_NS::DeviceScanInitKernel<cub::CUB_200700_1200_NS::ScanTileState<unsigned …
      0.0            2,944          3          981.3          960.0          928        1,056          66.6  void cub::CUB_200700_1200_NS::DeviceRadixSortExclusiveSumKernel<cub::CUB_200700_1200_NS::DeviceRadi…
      0.0            2,112          1        2,112.0        2,112.0        2,112        2,112           0.0  void cub::CUB_200700_1200_NS::DeviceReduceSingleTileKernel<cub::CUB_200700_1200_NS::DeviceReducePol…



## nvtx_sum

NOTICE: Existing SQLite export found: /tmp/n12_216.sqlite
        It is assumed file was previously exported from: /tmp/n12_216.nsys-rep
        Consider using --force-export=true if needed.

Processing [/tmp/n12_216.sqlite] with [/opt/nvidia/nsight-systems/2024.6.2/host-linux-x64/reports/nvtx_sum.py]... 

 ** NVTX Range Summary (nvtx_sum):

 Time (%)  Total Time (ns)  Instances    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)    Style                  Range               
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  ------------  -------  -----------------------------------
     52.8    2,136,566,811          6  356,094,468.5  357,089,692.5  293,195,984  418,554,107  65,979,368.9  PushPop  :w5_stitch                         
     32.1    1,297,079,089          3  432,359,696.3  431,541,176.0  431,005,357  434,532,556   1,900,727.5  PushPop  :e9c                               
      8.1      329,284,448        438      751,791.0      439,954.5      194,180    5,466,200     639,045.8  PushPop  :hash_rewrite                      
      4.0      163,263,894          3   54,421,298.0   54,589,174.0   53,688,589   54,986,131     664,861.3  PushPop  :vcount                            
      1.4       55,287,036          5   11,057,407.2    6,311,262.0    3,481,885   33,149,912  12,476,111.5  PushPop  :compact_radix                     
      0.9       36,496,399          3   12,165,466.3   11,862,914.0   11,843,121   12,790,364     541,267.7  PushPop  :sort                              
      0.2        8,841,395        283       31,241.7       35,902.0        2,318       87,486      14,152.6  PushPop  CCCL:cub::DeviceRadixSort          
      0.1        6,004,132        438       13,708.1        3,643.0        3,406       90,329      16,347.2  PushPop  :sort_prop                         
      0.1        4,344,557          1    4,344,557.0    4,344,557.0    
```
