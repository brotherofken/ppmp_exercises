#include <vector>

#include <fmt/format.h>

#include <kernels/elementwise.cuh>
#include <kernels/matmul.cuh>

int main(int argc, char ** argv) {
    {
        std::vector<float> a(50, 2);
        std::vector<float> b(50, 3);
        std::vector<float> result(50, -1);

        kernels::NDBuffer<float> ca({a.size()}, a.data());
        kernels::NDBuffer<float> cb({b.size()}, b.data());
        kernels::NDBuffer<float> cresult({result.size()});

        kernels::vector_add<float>(ca, cb, cresult);
        cresult.copy_to_host(result.data(), result.size());
        fmt::print("Vector addition result: [{}]\n", fmt::join(result, ", "));

        kernels::vector_multiply<float>(ca, cb, cresult);
        cresult.copy_to_host(result.data(), result.size());
        fmt::print("Vector multiplication result: [{}]\n", fmt::join(result, ", "));
    }
    {
        std::vector<float> a{1, 2, 3, 4, 5, 6};
        std::vector<float> b{4, 3, 2, 1};
        std::vector<float> result(6, -1);

        kernels::NDBuffer<float, 2> ca({3, 2}, a.data());
        kernels::NDBuffer<float, 2> cb({2, 2}, b.data());
        kernels::NDBuffer<float, 2> cresult({3, 2});

        kernels::matmul_simple<float>(ca, cb, cresult);
        cresult.copy_to_host(result.data(), result.size());
        fmt::print("matmul_simple result: [{}]\n", fmt::join(result, ", "));

        kernels::matmul_tiled<float>(ca, cb, cresult);
        cresult.copy_to_host(result.data(), result.size());
        fmt::print("matmul_tiled result: [{}]\n", fmt::join(result, ", "));
    }
    {
        std::vector<float> a{1, 2, 3, 4, 5, 6};
        std::vector<float> b{4, 3, 2};
        std::vector<float> result(2, -1);

        kernels::NDBuffer<float, 2> ca({2, 3}, a.data());
        kernels::NDBuffer<float, 2> cb({3, 1}, b.data());
        kernels::NDBuffer<float, 2> cresult({2, 1});

        kernels::matmul_simple<float>(ca, cb, cresult);
        cresult.copy_to_host(result.data(), result.size());
        fmt::print("matmul_simple result: [{}]\n", fmt::join(result, ", "));

        kernels::matmul_tiled<float>(ca, cb, cresult);
        cresult.copy_to_host(result.data(), result.size());
        fmt::print("matmul_tiled result: [{}]\n", fmt::join(result, ", "));
    }
    return 0;
}

