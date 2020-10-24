#include "matmul.cuh"

#include <fmt/format.h>

namespace kernels {

__global__
void matmul_simple_cuda(const float* const a, const float* const b, float* dest, size_t m, size_t n, size_t k) {
    /*
     * A[m, n] @ B[n, k] = C[m, k]
     */
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    const size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= k || y >= m) {
        return;
    }

    float sum = 0.f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i + n * y] * b[x + k * i];
    }
    dest[x + k * y] = sum;
}

template<class T>
void matmul_simple(const NDBuffer<T, 2>& a, const NDBuffer<T, 2>& b, NDBuffer<T, 2>& dest) {
    const size_t tile_size = 16;
    const size_t m = a.sizes()[0];
    const size_t n = a.sizes()[1];
    const size_t k = b.sizes()[1];
    const dim3 dim_grid(std::ceil(1.0 * m / tile_size), std::ceil(1.0 * m / tile_size));
    const dim3 dim_block(tile_size, tile_size);
    matmul_simple_cuda<<<dim_grid, dim_block>>>(a.ptr(), b.ptr(), dest.ptr(), m, n, k);
}


template<class T>
void matmul_tiled(const NDBuffer<T, 2>& a, const NDBuffer<T, 2>& b, NDBuffer<T, 2>& dest) {

}

template void matmul_simple<float>(const NDBuffer<float, 2>& a, const NDBuffer<float, 2>& b, NDBuffer<float, 2>& dest);
template void matmul_tiled<float>(const NDBuffer<float, 2>& a, const NDBuffer<float, 2>& b, NDBuffer<float, 2>& dest);


}