#include "elementwise.cuh"

#include <functional>

#include <fmt/format.h>

#define CHECK_CUDA_CODE(code) { check_cuda_error_code((code), std::string(__FILE__), __LINE__); }

namespace {

void check_cuda_error_code(const cudaError_t code, const std::string& file, int line)
{
    if (code != cudaSuccess) {
        const auto msg = fmt::format("Something weird happened at {}:{}. Error {}, message: {}\n",
            file,
            line,
            cudaGetErrorName(code),
            cudaGetErrorString(code)
        );
        fmt::print(msg);
        throw std::runtime_error(msg);
    }
}


template<class T>
class DeviceMem {
public:
    explicit DeviceMem(int size, const T* const copy_from = nullptr)
        : dev_ptr(DeviceMem::allocate(size, copy_from))
    {}
    ~DeviceMem() {
        CHECK_CUDA_CODE(cudaFree(dev_ptr));
    }

    void* vptr() {return reinterpret_cast<void*>(dev_ptr);}
    T* ptr() {return dev_ptr;}

    void copy_to_host(T* host_ptr, int size) {
        CHECK_CUDA_CODE(cudaMemcpy((void*)host_ptr, this->vptr(), size, cudaMemcpyDeviceToHost));
    }
private:
    T* allocate(int size, const T* const copy_from) {
        CHECK_CUDA_CODE(cudaMalloc(&dev_ptr, size));
        if (copy_from) {
            CHECK_CUDA_CODE(cudaMemcpy(dev_ptr, copy_from, size, cudaMemcpyHostToDevice));
        }
        return dev_ptr;
    }

    T* dev_ptr;
};

template<class T>
__device__ T multiplies(const T &lhs, const T &rhs) { return lhs * rhs; }

template<class T>
__device__  T plus(const T &lhs, const T &rhs) { return lhs + rhs; }
}


namespace kernels {
__global__ void vector_add_cuda(const float* const a, const float* const b, float* dest, int size) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dest[i] = a[i] + b[i];
    }
}

__global__ void vector_mul_cuda(const float* const a, const float* const b, float* dest, int size) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dest[i] = a[i] * b[i];
    }
}

__host__
void vector_add(const float* const a, const float* const b, float* dest, const int size) {
    DeviceMem<float> da(size, a);
    DeviceMem<float> db(size, b);
    DeviceMem<float> ddest(size);

    vector_add_cuda<<<std::ceil(size / 256.0), 256>>>(da.ptr(), db.ptr(), ddest.ptr(), size);

    ddest.copy_to_host(dest, size);
}


__host__
void vector_multiply(const float* const a, const float* const b, float* dest, const int size) {
    DeviceMem<float> da(size, a);
    DeviceMem<float> db(size, b);
    DeviceMem<float> ddest(size);

    vector_mul_cuda<<<std::ceil(size / 256.0), 256>>>(da.ptr(), db.ptr(), ddest.ptr(), size);

    ddest.copy_to_host(dest, size);
}

}