add_executable(devicequery devicequery.cpp)

# CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto)

set_target_properties(devicequery PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_dependencies(devicequery fmt)
target_link_libraries(devicequery kernels fmt)
