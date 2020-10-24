#pragma once

#include "core.cuh"

namespace kernels {

template<class T>
void vector_add(const NDBuffer<T>& a, const NDBuffer<T>& b, NDBuffer<T>& dest);


template<class T>
void vector_multiply(const NDBuffer<T>& a, const NDBuffer<T>& b, NDBuffer<T>& dest);

}
