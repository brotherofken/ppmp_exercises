#pragma once

namespace kernels {

void vector_add(float const* const a, float const* const b, float* dest, int size);

void vector_multiply(float const* const a, float const* const b, float* dest, int size);

}
