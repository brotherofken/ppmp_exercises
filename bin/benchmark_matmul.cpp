#include <benchmark/benchmark.h>

#include <kernels/core.cuh>
#include <kernels/matmul.cuh>

#include <functional>
#include <random>


struct MatmulData {
    explicit MatmulData(size_t size)
        : k(size)
        , m(size)
        , n(size)
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


static void MatmulSimple(benchmark::State& state) {
    MatmulData data(state.range(0));
    for (auto _ : state) {
        auto watch = std::make_shared<kernels::CudaStopwatch>();
        kernels::matmul_simple<float>(data.ca, data.cb, data.cresult, watch);
        state.SetIterationTime(watch->elapsedS());

        data.cresult.copy_to_host(data.result.data(), data.result.size());
    }
}
BENCHMARK(MatmulSimple)->UseManualTime()->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 4096);

static void MatmulTiled(benchmark::State& state) {
    MatmulData data(state.range(0));
    for (auto _ : state) {
        auto watch = std::make_shared<kernels::CudaStopwatch>();
        kernels::matmul_tiled<float>(data.ca, data.cb, data.cresult, watch);
        state.SetIterationTime(watch->elapsedS());

        data.cresult.copy_to_host(data.result.data(), data.result.size());
    }
}
BENCHMARK(MatmulTiled)->UseManualTime()->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 4096);

BENCHMARK_MAIN();