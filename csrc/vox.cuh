// Voxel indexing shared by ws.cu and rag.cu.
//
// A full-volume kernel has to know its (z,y,x), and recovering it from a linear
// thread index costs more than it looks like:
//
//     int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
//
// NVIDIA GPUs have no integer divide instruction, and because Y and X arrive as
// runtime arguments the compiler cannot fold the divisions into multiply-shift
// either, so it emits the full expansion. Measured on the sm_120 build with
// scripts/b1_sass.py, k_flow came out at 244 arithmetic instructions against 23
// memory ops, carrying three MUFU.RCP -- the float-reciprocal step of that
// expansion. Carrying y and z in blockIdx.y/z removes all of it.
//
// Consecutive threadIdx.x still map to consecutive x, so no access pattern
// changes. It does fix one thing beyond arithmetic: with a 1D launch a warp can
// straddle a row boundary and cover two y values, which a warp-level
// aggregation keyed on spatial locality would rather it did not.
#pragma once

#include <cstdio>
#include <cstdlib>

struct Vox {
    int64_t i, z, y, x;
    bool ok;
};

static inline __device__ Vox vox_of(int64_t Y, int64_t X) {
    Vox v;
    v.x = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    v.y = blockIdx.y;
    v.z = blockIdx.z;
    // vox_grid sizes the y and z extents exactly, so only x can overrun.
    v.ok = v.x < X;
    v.i = (v.z * Y + v.y) * X + v.x;
    return v;
}

// gridDim.y and gridDim.z cap at 65535. The graded volume is Y=2400, Z=375, so
// one block row per (y,z) fits comfortably. A volume past the cap would need
// tiling, and silently launching a grid that covers only part of it would be a
// wrong answer, so refuse instead.
static inline dim3 vox_grid(int64_t Z, int64_t Y, int64_t X, int threads) {
    if (Y > 65535 || Z > 65535) {
        fprintf(stderr, "vox_grid FATAL Y=%lld Z=%lld exceeds the 3D grid limit\n",
                (long long)Y, (long long)Z);
        std::abort();
    }
    return dim3((unsigned)((X + threads - 1) / threads),
                (unsigned)Y, (unsigned)Z);
}
