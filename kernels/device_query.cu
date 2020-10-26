#include "device_query.cuh"

#include <fmt/format.h>

namespace kernels {

__host__ void device_query() {
    const int deviceCount = []{
        int result = 0;
        cudaGetDeviceCount(&result);
        return result;
    }();

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp{};
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
        fmt::print("\tMaximum global memory msize: {}\n", deviceProp.totalGlobalMem);
        fmt::print("\tMaximum constant memory msize: {}\n", deviceProp.totalConstMem);
        fmt::print("\tMaximum shared memory msize per block: {}\n", deviceProp.sharedMemPerBlock);
        fmt::print("\tMaximum block dimensions: [{}]\n", fmt::join(deviceProp.maxThreadsDim, " x "));
        fmt::print("\tMaximum grid dimensions: [{}]\n", fmt::join(deviceProp.maxGridSize, " x "));
        fmt::print("\tWarp msize: {}\n", deviceProp.warpSize);
    }
}

}
