#pragma once

#include "core.cuh"

namespace kernels {

template<class T>
void vector_add(const NDSpan<T>& a, const NDSpan<T>& b, NDSpan<T>& dest);


template<class T>
void vector_multiply(const NDSpan<T>& a, const NDSpan<T>& b, NDSpan<T>& dest);

}
