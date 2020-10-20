#include "device_query.cuh"

#include <iostream>
#include <fmt/format.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <typename Arg, typename... Args>
void print(std::ostream& out, Arg&& arg, Args&&... args) {
    out << std::forward<Arg>(arg);
    using expander = int[];
    (void)expander{0, (void(out << ',' << std::forward<Args>(args) << std::endl), 0)...};
}

}

namespace kernels {

__host__ void device_query() {
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;

        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
                fmt::print("No CUDA GPU has been detected\n");
                return;
            } else if (deviceCount == 1) {
                fmt::print("There is 1 device supporting CUDA\n");
            } else {
                fmt::print("There are {} devices supporting CUDA\n", deviceCount);
            }
        }

        fmt::print("Device {} name {}", dev, deviceProp.name);
        fmt::print("\tComputational Capabilities: {}.{}\n", deviceProp.major, deviceProp.minor);
        fmt::print("\tMaximum global memory size: {}\n", deviceProp.totalGlobalMem);
        fmt::print("\tMaximum constant memory size: {}\n", deviceProp.totalConstMem);
        fmt::print("\tMaximum shared memory size per block: {}\n", deviceProp.sharedMemPerBlock);
        fmt::print("\tMaximum block dimensions: {}x{}x{}\n", deviceProp.maxThreadsDim[0],
              deviceProp.maxThreadsDim[1],
              deviceProp.maxThreadsDim[2]);
        fmt::print("\tMaximum grid dimensions: {}x{}x{}\n", deviceProp.maxGridSize[0],
              deviceProp.maxGridSize[1],
              deviceProp.maxGridSize[2]);
        fmt::print("\tWarp size: {}\n", deviceProp.warpSize);
    }
}

}