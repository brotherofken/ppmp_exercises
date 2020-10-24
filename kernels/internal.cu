#include "internal.cuh"

#include <stdexcept>
#include <string>

#include <fmt/format.h>

namespace kernels {

    void check_cuda_error_code(cudaError_t code, const char* file, int line) {
        if (code != cudaSuccess) {
            const auto msg = fmt::format("Something weird happened at {}:{}. Error {}, message: {}\n",
                                         std::string(file),
                                         line,
                                         cudaGetErrorName(code),
                                         cudaGetErrorString(code)
            );
            fmt::print(msg);
            throw std::runtime_error(msg);
        }
    }

}