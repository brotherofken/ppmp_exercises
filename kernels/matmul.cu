#include "matmul.cuh"

namespace kernels {

__global__
void matmul_simple_cuda(const float* const a, const float* const b, float* dest, int m, int n, int k) {

}

void matmul_simple(const float* const a, const float* const b, float* dest, int m, int n, int k) {

}

void matmul_tiled(const float* const a, const float* const b, float* dest, int m, int n, int k) {
}

}