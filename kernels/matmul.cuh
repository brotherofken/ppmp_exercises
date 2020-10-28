#pragma once

#include "core.cuh"

namespace kernels {

template<class T>
void matmul_simple(const Buffer2D<T>& a, const Buffer2D<T>& b, Buffer2D<T>& dest, CudaStopwatch::SPtr watch = nullptr);


template<class T>
void matmul_tiled(const Buffer2D<T>& a, const Buffer2D<T>& b, Buffer2D<T>& dest, CudaStopwatch::SPtr watch = nullptr);


}
