#include "matmul.cuh"

#include <fmt/format.h>

namespace kernels {

template<class T>
__global__ void matmul_simple_cuda(const T* const a, const T* const b, T* dest, const size_t m, const size_t n, size_t k) {
    /*
     * A[m, n] @ B[n, k] = C[m, k]
     */
    const size_t tx = threadIdx.x + blockDim.x * blockIdx.x;
    const size_t ty = threadIdx.y + blockDim.y * blockIdx.y;

    if (tx >= k || ty >= m) {
        return;
    }

    float sum = 0.f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i + n * ty] * b[tx + k * i];
    }
    dest[tx + k * ty] = sum;
}


template<class T>
void matmul_simple(const Buffer2D<T>& a, const Buffer2D<T>& b, Buffer2D<T>& dest) {
    const size_t tile_size = 16;
    const size_t m = a.sizes()[0];
    const size_t n = a.sizes()[1];
    const size_t k = b.sizes()[1];
    const dim3 dim_grid(std::ceil(1.0 * m / tile_size), std::ceil(1.0 * m / tile_size));
    const dim3 dim_block(tile_size, tile_size);
    matmul_simple_cuda<<<dim_grid, dim_block>>>(a.ptr(), b.ptr(), dest.ptr(), m, n, k);
}


template<class T>
__global__ void matmul_tiled_cuda(const T* const a, const T* const b, T* dest, size_t m, size_t n, size_t k) {
    const size_t tile_size = 16;

    __shared__ float a_cache[tile_size][tile_size];
    __shared__ float b_cache[tile_size][tile_size];

    const size_t thread_x = threadIdx.x;
    const size_t thread_y = threadIdx.y;
    const size_t x = threadIdx.x + tile_size * blockIdx.x;
    const size_t y = threadIdx.y + tile_size * blockIdx.y;

    float sum = 0.f;

    const auto tiles_count = static_cast<size_t>(std::ceil(1.0 * m / tile_size));
    for (size_t t = 0; t < tiles_count; ++t) {
        const auto tile_offset = t * tile_size;
        a_cache[thread_y][thread_x] = a[thread_x + tile_offset + y * n];
        b_cache[thread_y][thread_x] = b[x + (y + tile_offset) * k];
        __syncthreads();

        for (size_t i = 0; i < tile_size;++i) {
            sum += a_cache[thread_y][i] * b_cache[i][thread_x];
        }
        __syncthreads();
    }

    if (x < k || y < m) {
        dest[x + k * y] = sum;
    }
}


template<class T>
void matmul_tiled(const Buffer2D<T>& a, const Buffer2D<T>& b, Buffer2D<T>& dest) {
    const size_t tile_size = 16;
    const size_t m = a.sizes()[0];
    const size_t n = a.sizes()[1];
    const size_t k = b.sizes()[1];
    const dim3 dim_grid(std::ceil(1.0 * m / tile_size), std::ceil(1.0 * m / tile_size));
    const dim3 dim_block(tile_size, tile_size);
    matmul_tiled_cuda<<<dim_grid, dim_block>>>(a.ptr(), b.ptr(), dest.ptr(), m, n, k);
}

template void matmul_simple<float>(const Buffer2D<float>& a, const Buffer2D<float>& b, Buffer2D<float>& dest);
template void matmul_simple<double>(const Buffer2D<double>& a, const Buffer2D<double>& b, Buffer2D<double>& dest);

template void matmul_tiled<float>(const Buffer2D<float>& a, const Buffer2D<float>& b, Buffer2D<float>& dest);
template void matmul_tiled<double>(const Buffer2D<double>& a, const Buffer2D<double>& b, Buffer2D<double>& dest);

}