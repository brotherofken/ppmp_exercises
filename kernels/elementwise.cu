#include "elementwise.cuh"

#include <functional>

#include <fmt/format.h>

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
void vector_add(const NDSpan<T>& a, const NDSpan<T>& b, NDSpan<T>& dest) {
    const int size = a.size();
    vector_add_cuda<<<std::ceil(size / 256.0), 256>>>(a.ptr(), b.ptr(), dest.ptr(), size);
}

template<class T>
void vector_multiply(const NDSpan<T>& a, const NDSpan<T>& b, NDSpan<T>& dest) {
    const int size = a.size();
    vector_mul_cuda<<<std::ceil(size / 256.0), 256>>>(a.ptr(), b.ptr(), dest.ptr(), size);
}

template void vector_add<float>(const NDSpan<float>& a, const NDSpan<float>& b, NDSpan<float>& dest);
template void vector_multiply<float>(const NDSpan<float>& a, const NDSpan<float>& b, NDSpan<float>& dest);

}