#pragma once

#include "core.cuh"

namespace kernels {

template<class T>
void matmul_simple(const NDBuffer<T, 2>& a, const NDBuffer<T, 2>& b, NDBuffer<T, 2>& dest);


template<class T>
void matmul_tiled(const NDBuffer<T, 2>& a, const NDBuffer<T, 2>& b, NDBuffer<T, 2>& dest);


}
