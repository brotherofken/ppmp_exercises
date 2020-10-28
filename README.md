# Build & Run

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
make -j all
./bin/devicequery
./bin/benchmark_matmul
```

# Materials
- [Kirk, David B., and Wen-mei W. Hwu. Programming Massively Parallel Processors: A Hands-on Approach. 2nd ed. San Francisco, CA, USA: Morgan Kaufmann Publishers Inc., 2012.](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0124159923)
- [CoffeeBeforeArch, CUDA Crash Course, 2019](https://www.youtube.com/playlist?list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU)
- [PUMPS+AI Summer School | PUMPS+AI 2019, accessed Oct. 29, 2020](https://pumps.bsc.es/2019/)
- [PUMPS+AI Summer School | PUMPS+AI 2019, labs](https://github.com/illinois-impact/gpu-algorithms-labs)


# Plan

I'm updating it on the way.
- ☑ build system
- ☑ device query
- ☑ benchmarking
- ☐ unit tests
- CUDA thingies
  - ☐ unified memory
- algorithms
  - ☑ vector add
  - ☑ naive matrix multiplication
  - ☑ tiled matrix multiplication
  - ☐ 2D convolution
  - ☐ histogram

# Benchmarks

## Matrix multiplication
| Benchmark                       | Time           | CPU             | Iterations |
|---------------------------------|----------------|-----------------|-----------|
| MatmulSimple/256/manual_time    |    0.371 ms    |     0.495 ms    |      1887 |
| MatmulSimple/512/manual_time    |     2.68 ms    |      3.08 ms    |       261 |
| MatmulSimple/1024/manual_time   |    21.5 ms     |      22.7 ms    |        33 |
| MatmulSimple/2048/manual_time   |     172 ms     |       176 ms    |         4 |
| MatmulSimple/4096/manual_time   |    1373 ms     |      1393 ms    |         1 |
| MatmulTiled/256/manual_time     |     0.160 ms   |     0.284 ms    |      4363 |
| MatmulTiled/512/manual_time     |      1.11 ms   |      1.53 ms    |       631 |
| MatmulTiled/1024/manual_time    |     8.97 ms    |      10.3 ms    |        78 |
| MatmulTiled/2048/manual_time    |     71.5 ms    |      76.3 ms    |        10 |
| MatmulTiled/4096/manual_time    |      571 ms    |       590 ms    |         1 |
