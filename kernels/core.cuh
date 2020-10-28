#pragma once

#include <algorithm>
#include <array>
#include <memory>

namespace kernels {

template<class T, int dim = 1>
class NDBuffer {
public:
    explicit NDBuffer(const std::array<size_t, dim>& sizes, T* const copy_from = nullptr);

    ~NDBuffer();

    void* vptr() { return reinterpret_cast<void*>(dev_ptr); }
    T* ptr() { return dev_ptr; }
    const T* ptr() const { return dev_ptr; }

    const std::array<size_t, dim>& sizes() const { return m_sizes; }
    size_t size() const { return m_size; }

    void copy_to_host(T* host_ptr, int size);
    void copy_to_device(T* host_ptr, int size);
private:
    T* allocate(T* const copy_from);

    const std::array<size_t, dim> m_sizes;
    const size_t m_size;
    T* dev_ptr;
};

template <class T>
using Buffer1D = NDBuffer<T, 1>;

template <class T>
using Buffer2D = NDBuffer<T, 2>;

struct CudaStopwatchData;
class CudaStopwatch {
public:
    using SPtr = std::shared_ptr<CudaStopwatch>;

    CudaStopwatch();

    void start();

    void stop();

    float elapsedMs();
    float elapsedS();
private:
    std::shared_ptr<CudaStopwatchData> data;
};

struct StopwatchContext {
    StopwatchContext(CudaStopwatch::SPtr watch)
        : watch(watch)
    {
        if (watch) { watch->start(); }
    }

    ~StopwatchContext() {
        if (watch) { watch->stop(); }
    }

    CudaStopwatch::SPtr watch;
};

}