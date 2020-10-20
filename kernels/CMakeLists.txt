add_library(kernels
    device_query.cu
)

target_compile_features(kernels PUBLIC cxx_std_11)
set_target_properties(kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )

add_dependencies(kernels fmt)
target_link_libraries(kernels ${CUDA_LIBRARIES} fmt)
include_directories(${CUDA_INCLUDE_DIRS})