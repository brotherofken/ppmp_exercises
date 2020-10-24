#define CHECK_CUDA_CODE(code) { check_cuda_error_code((code), __FILE__, __LINE__); }

namespace kernels {
    void check_cuda_error_code(cudaError_t code, const char* file, int line);
}