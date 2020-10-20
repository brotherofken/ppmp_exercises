cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(pmpp_labs LANGUAGES CXX)
enable_language(CUDA)

include(CTest)
include(ExternalProject)
include(FeatureSummary)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_CXX_STANDARD 17)
set(DEFAULT_BUILD_TYPE "Release")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(fmt "7.0.0")
message(STATUS "fmt_INCLUDE_DIR: ${fmt_INCLUDE_DIR}")
include_directories(${PROJECT_SOURCE_DIR} ${fmt_INCLUDE_DIR})

add_subdirectory(kernels)
add_subdirectory(bin)
