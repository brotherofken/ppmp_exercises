#include "elementwise.cuh"

namespace {
template<class T>
__device__ T multiplies(const T &lhs, const T &rhs) { return lhs * rhs; }

template<class T>
__device__  T plus(const T &lhs, const T &rhs) { return lhs + rhs; }

}

namespace kernels {

template<class T>
__global__ void vector_add_cuda(const T* const a, const T* const b, T* dest, int size) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dest[i] = a[i] + b[i];
    }
}

template<class T>
__global__ void vector_mul_cuda(const T* const a, const T* const b, T* dest, int size) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dest[i] = a[i] * b[i];
    }
}

template<class T>
void vector_add(const NDBuffer<T>& a, const NDBuffer<T>& b, NDBuffer<T>& dest) {
    const size_t block_size = 256;
    const int size = a.size();
    vector_add_cuda<<<std::ceil(1.0 * size / block_size), block_size>>>(a.ptr(), b.ptr(), dest.ptr(), size);
}

template<class T>
void vector_multiply(const NDBuffer<T>& a, const NDBuffer<T>& b, NDBuffer<T>& dest) {
    const size_t block_size = 256;
    const int size = a.size();
    vector_mul_cuda<<<std::ceil(1.0 * size / block_size), block_size>>>(a.ptr(), b.ptr(), dest.ptr(), size);
}

template void vector_add<float>(const NDBuffer<float>& a, const NDBuffer<float>& b, NDBuffer<float>& dest);
template void vector_multiply<float>(const NDBuffer<float>& a, const NDBuffer<float>& b, NDBuffer<float>& dest);

}