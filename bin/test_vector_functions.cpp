#include <vector>

#include <fmt/format.h>

#include <kernels/elementwise.cuh>

int main(int argc, char ** argv) {
    std::vector<float> a(1000, 2);
    std::vector<float> b(1000, 3);

    std::vector<float> result(1000, -1);

    kernels::NDSpan<float> ca({a.size()}, a.data());
    kernels::NDSpan<float> cb({b.size()}, b.data());
    kernels::NDSpan<float> cresult({result.size()});

    kernels::vector_add<float>(ca, cb, cresult);
    cresult.copy_to_host(result.data(), result.size());
    fmt::print("Vector addition result: [{}]\n", fmt::join(result, ", "));

    kernels::vector_multiply<float>(ca, cb, cresult);
    cresult.copy_to_host(result.data(), result.size());
    fmt::print("Vector multiplication result: [{}]\n", fmt::join(result, ", "));

    return 0;
}

