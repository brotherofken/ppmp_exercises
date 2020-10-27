include (ExternalProject)

set(GOOGLE_BENCHMARK_PATH "${CMAKE_CURRENT_BINARY_DIR}/google-benchmark")
externalproject_add(ex_benchmark
        GIT_REPOSITORY "https://github.com/google/benchmark.git"
        GIT_TAG "v1.5.2"
        INSTALL_DIR "${GOOGLE_BENCHMARK_PATH}"
        CMAKE_ARGS
            -DBENCHMARK_ENABLE_TESTING=OFF
            -DCMAKE_INSTALL_PREFIX:PATH=${GOOGLE_BENCHMARK_PATH}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
            -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
            -DCMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE}
            -DBENCHMARK_ENABLE_LTO=true
        )
link_directories("${GOOGLE_BENCHMARK_PATH}/lib")
include_directories("${GOOGLE_BENCHMARK_PATH}/include")
