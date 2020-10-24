#pragma once

#include <algorithm>
#include <array>

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
private:
    T* allocate(T* const copy_from);

    const std::array<size_t, dim> m_sizes;
    const size_t m_size;
    T* dev_ptr;
};

}