#pragma once

namespace kernels {

void matmul_simple(float const* const a, float const* const b, float* dest, int m, int n, int k);

void matmul_tiled(float const* const a, float const* const b, float* dest, int m, int n, int k);

}
