set(SRC_FILES test_cuda_functions.cpp devicequery.cpp)

foreach(SRC_FILE IN LISTS SRC_FILES)
    get_filename_component(EXECUTABLE_NAME "${SRC_FILE}" NAME_WE)
    message(STATUS "Add target ${EXECUTABLE_NAME} for ${SRC_FILE}")
    add_executable(${EXECUTABLE_NAME} ${SRC_FILE})

    set_target_properties(${EXECUTABLE_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    add_dependencies(${EXECUTABLE_NAME} fmt)
    target_link_libraries(${EXECUTABLE_NAME} kernels fmt)
endforeach()