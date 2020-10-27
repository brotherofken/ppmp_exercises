#include <benchmark/benchmark.h>

#include <kernels/matmul.cuh>

#include <functional>
#include <random>


class MatmulData : public ::benchmark::Fixture {
public:
    MatmulData()
        : k(2048)
        , m(2048)
        , n(2048)
        , a(m * n, -1.f)
        , b(n * k, -1.f)
        , result(m * k, -1.f)
        , ca({m, n})
        , cb({n, k})
        , cresult({m, k})
    {
        std::random_device rd;
        std::mt19937 mersenne_engine{rd()};
        std::uniform_real_distribution<float> dist(-1.0, 1.0);

        const auto gen = [&dist, &mersenne_engine](){
            return dist(mersenne_engine);
        };

        generate(begin(a), end(a), gen);
        generate(begin(b), end(b), gen);
        ca.copy_to_device(a.data(), a.size());
        cb.copy_to_device(b.data(), b.size());
    }

    void SetUp(const ::benchmark::State& state) { }

    void TearDown(const ::benchmark::State& state) { }

    ~MatmulData() { }

    const size_t m;
    const size_t n;
    const size_t k;
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> result;
    kernels::Buffer2D<float> ca;
    kernels::Buffer2D<float> cb;
    kernels::Buffer2D<float> cresult;
};


BENCHMARK_F(MatmulData, MatmulSimple)(benchmark::State& state) {
    for (auto _ : state) {
        kernels::matmul_simple<float>(ca, cb, cresult);
        cresult.copy_to_host(result.data(), result.size());
    }
}

BENCHMARK_F(MatmulData, MatmulTiled)(benchmark::State& state) {
    for (auto _ : state) {
        kernels::matmul_tiled<float>(ca, cb, cresult);
        cresult.copy_to_host(result.data(), result.size());
    }
}

BENCHMARK_MAIN();