#include "core.cuh"
#include "internal.cuh"

#include <string>

namespace kernels {

template<class T, int dim>
NDBuffer<T, dim>::NDBuffer(const std::array<size_t, dim>& sizes, T* const copy_from)
        : m_sizes(sizes)
        , m_size(std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<size_t>()))
        , dev_ptr(NDBuffer::allocate(copy_from))
{}

template<class T, int dim>
T* NDBuffer<T, dim>::allocate(T* const copy_from) {
    const auto buffer_size = m_size * sizeof(T);
    CHECK_CUDA_CODE(cudaMalloc(&dev_ptr, buffer_size));
    if (copy_from) {
        CHECK_CUDA_CODE(cudaMemcpy(dev_ptr, copy_from, buffer_size, cudaMemcpyHostToDevice));
    }
    return dev_ptr;
}

template<class T, int dim>
void NDBuffer<T, dim>::copy_to_host(T* host_ptr, int size) {
    CHECK_CUDA_CODE(cudaMemcpy((void*)host_ptr, this->vptr(), size * sizeof(T), cudaMemcpyDeviceToHost));
}

template<class T, int dim>
NDBuffer<T, dim>::~NDBuffer() {
    CHECK_CUDA_CODE(cudaFree(dev_ptr));
}

template class NDBuffer<float, 1>;
template class NDBuffer<float, 2>;

}

