/*
 * cuda_kmeans.cu — GPU K-means with cuBLAS distance computation
 *
 * Hot path per iteration:
 *   1. cublasSgemm: XCt[N×K] = X[N×D] · C[K×D]ᵀ       (~5ms on A5000)
 *   2. assignKernel: dist = x_norm + c_norm - 2·XCt, argmin  (~1ms)
 *   3. updateKernel: accumulate + divide centroids              (~1ms)
 *
 * Total: ~7ms/iter → 50 iters in 0.35 sec (vs 500 min on CPU)
 */

#include "cuda_kmeans.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>

#define CUDA_CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return {}; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__, __LINE__, s); \
        return {}; \
    } \
} while(0)

// ═══════════════════════════════════════════════════════════════════════════
// Wall-clock helper
// ═══════════════════════════════════════════════════════════════════════════

static const char* ts() {
    static thread_local char buf[16];
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    struct tm tm;
    localtime_r(&tt, &tm);
    std::snprintf(buf, sizeof(buf), "[%02d:%02d:%02d]", tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

using SteadyClock = std::chrono::steady_clock;

static double elapsedSec(SteadyClock::time_point start) {
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

static std::string fmtT(double sec) {
    if (sec < 60.0) { char b[32]; std::snprintf(b, 32, "%.1fs", sec); return b; }
    int m = int(sec) / 60; double s = sec - m * 60;
    char b[32]; std::snprintf(b, 32, "%dm%.0fs", m, s); return b;
}

#define tlog(fmt, ...) std::printf("%s " fmt, ts(), ##__VA_ARGS__)

// ═══════════════════════════════════════════════════════════════════════════
// CUDA Kernels
// ═══════════════════════════════════════════════════════════════════════════

// Compute squared L2 norm of each row: norms[i] = sum_j(data[i*D+j]²)
__global__ void rowNormsKernel(const float* __restrict__ data,
                                float* __restrict__ norms,
                                int N, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float sum = 0.0f;
    const float* row = data + (size_t)i * D;
    for (int j = 0; j < D; ++j) {
        float v = row[j];
        sum += v * v;
    }
    norms[i] = sum;
}

// K-means++ distance update: minDist[i] = min(minDist[i], ||x_i - centroid||²)
__global__ void kppDistKernel(const float* __restrict__ data,
                               const float* __restrict__ centroid,
                               float* __restrict__ minDist,
                               int N, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float sum = 0.0f;
    const float* row = data + (size_t)i * D;
    for (int j = 0; j < D; ++j) {
        float d = row[j] - centroid[j];
        sum += d * d;
    }
    minDist[i] = fminf(minDist[i], sum);
}

// Assignment: dist[i,k] = x_norm[i] + c_norm[k] - 2*XCt[i,k], find argmin
// Also accumulates MSE for quality tracking
__global__ void assignKernel(const float* __restrict__ XCt,    // N×K row-major
                              const float* __restrict__ x_norm, // N
                              const float* __restrict__ c_norm, // K
                              int* __restrict__ assignments,     // N
                              float* __restrict__ minDistOut,    // N (for MSE)
                              int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float xn = x_norm[i];
    float bestDist = 1e30f;
    int bestK = 0;

    for (int k = 0; k < K; ++k) {
        // XCt already contains -2·dot(x,c) from GEMM alpha=-2
        float d = xn + c_norm[k] + XCt[(size_t)i * K + k];
        if (d < bestDist) { bestDist = d; bestK = k; }
    }

    assignments[i] = bestK;
    if (minDistOut) minDistOut[i] = bestDist;
}

// Zero out centroid accumulators and counts
__global__ void clearCentroidsKernel(float* __restrict__ centroids,
                                      int* __restrict__ counts,
                                      int K, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) counts[idx] = 0;
    if (idx < K * D) centroids[idx] = 0.0f;
}

// Accumulate: centroids[assign[i]] += data[i], counts[assign[i]]++
// Uses atomicAdd — fine for K≤4096 (low contention)
__global__ void accumulateKernel(const float* __restrict__ data,
                                  const int* __restrict__ assignments,
                                  float* __restrict__ centroids,
                                  int* __restrict__ counts,
                                  int N, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int k = assignments[i];
    atomicAdd(&counts[k], 1);
    const float* row = data + (size_t)i * D;
    float* cent = centroids + (size_t)k * D;
    for (int j = 0; j < D; ++j)
        atomicAdd(&cent[j], row[j]);
}

// Divide centroids by count; reinit empty clusters from random data point
__global__ void divideCentroidsKernel(float* __restrict__ centroids,
                                       const int* __restrict__ counts,
                                       const float* __restrict__ data,
                                       int K, int D, int N,
                                       unsigned int seed)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    int cnt = counts[k];
    float* cent = centroids + (size_t)k * D;

    if (cnt > 0) {
        float inv = 1.0f / cnt;
        for (int j = 0; j < D; ++j)
            cent[j] *= inv;
    } else {
        // Reinit from pseudo-random data point
        unsigned int h = seed ^ (k * 2654435761u);
        int ri = h % N;
        const float* row = data + (size_t)ri * D;
        for (int j = 0; j < D; ++j)
            cent[j] = row[j];
    }
}

// Sum reduction for MSE
__global__ void sumReduceKernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    float sum = 0.0f;
    if (i < N) sum += input[i];
    if (i + blockDim.x < N) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Host implementation
// ═══════════════════════════════════════════════════════════════════════════

CudaKMeansResult cudaKMeans(
    const float* h_data,
    int N, int D, int K,
    int iterations,
    int subsampleN)
{
    auto tTotal = SteadyClock::now();

    // ── Subsample for training ──
    int trainN = (subsampleN > 0 && subsampleN < N) ? subsampleN : N;
    std::vector<int> trainIdx(N);
    std::iota(trainIdx.begin(), trainIdx.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(trainIdx.begin(), trainIdx.end(), rng);

    // Build contiguous training array
    auto tStep = SteadyClock::now();
    std::vector<float> h_train(trainN * D);
    for (int i = 0; i < trainN; ++i)
        std::memcpy(&h_train[i * D], &h_data[trainIdx[i] * D], D * sizeof(float));
    tlog("  │  Training data prepared: %d/%d leaves (%.1fs)\n", trainN, N, elapsedSec(tStep));

    // ── Upload to GPU ──
    tStep = SteadyClock::now();
    float* d_train = nullptr;    // trainN × D
    float* d_centroids = nullptr;// K × D
    float* d_XCt = nullptr;      // trainN × K (reused for batched final)
    float* d_x_norm = nullptr;   // trainN
    float* d_c_norm = nullptr;   // K
    float* d_minDist = nullptr;  // trainN
    int*   d_assign = nullptr;   // trainN
    int*   d_counts = nullptr;   // K
    float* d_mseSum = nullptr;   // 1

    CUDA_CHECK(cudaMalloc(&d_train,     (size_t)trainN * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, (size_t)K * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_XCt,       (size_t)trainN * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_norm,    trainN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c_norm,    K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_minDist,   trainN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_assign,    trainN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts,    K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mseSum,    sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_train, h_train.data(), (size_t)trainN * D * sizeof(float),
                           cudaMemcpyHostToDevice));

    double gpuMB = ((size_t)trainN * D + (size_t)K * D + (size_t)trainN * K + trainN * 2 + K * 2)
                   * sizeof(float) / (1024.0 * 1024.0);
    tlog("  │  GPU upload: %.1f MB (%.1fs)\n", gpuMB, elapsedSec(tStep));

    // ── cuBLAS handle ──
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    const int BLK = 256;

    // ── K-means++ initialization ──
    tStep = SteadyClock::now();
    tlog("  │  K-means++ init (%d centroids)...\n", K);

    // First centroid = random training point
    std::vector<float> h_centroids(K * D);
    std::memcpy(h_centroids.data(), &h_train[0], D * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(), D * sizeof(float),
                           cudaMemcpyHostToDevice));

    // Init minDist to infinity
    std::vector<float> h_minDist(trainN, 1e30f);
    CUDA_CHECK(cudaMemcpy(d_minDist, h_minDist.data(), trainN * sizeof(float),
                           cudaMemcpyHostToDevice));

    for (int k = 1; k < K; ++k) {
        // Update minDist with distance to centroid k-1
        int gridN = (trainN + BLK - 1) / BLK;
        kppDistKernel<<<gridN, BLK>>>(d_train, d_centroids + (size_t)(k - 1) * D,
                                       d_minDist, trainN, D);

        // Download minDist for weighted selection
        CUDA_CHECK(cudaMemcpy(h_minDist.data(), d_minDist, trainN * sizeof(float),
                               cudaMemcpyDeviceToHost));

        double totalDist = 0.0;
        for (int i = 0; i < trainN; ++i) totalDist += h_minDist[i];

        double r = std::uniform_real_distribution<double>(0.0, totalDist)(rng);
        double cumul = 0.0;
        int selected = 0;
        for (int i = 0; i < trainN; ++i) {
            cumul += h_minDist[i];
            if (cumul >= r) { selected = i; break; }
        }

        // Copy selected point as new centroid
        std::memcpy(&h_centroids[k * D], &h_train[selected * D], D * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_centroids + (size_t)k * D,
                               &h_centroids[k * D], D * sizeof(float),
                               cudaMemcpyHostToDevice));

        if (k % 100 == 0 || k == K - 1) {
            tlog("  │    init: %d/%d centroids (%.1fs)\n", k, K, elapsedSec(tStep));
        }
    }
    tlog("  │  K-means++ done: %s\n", fmtT(elapsedSec(tStep)).c_str());

    // Upload full centroids
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(), (size_t)K * D * sizeof(float),
                           cudaMemcpyHostToDevice));

    // ── Compute training data norms (once) ──
    {
        int gridN = (trainN + BLK - 1) / BLK;
        rowNormsKernel<<<gridN, BLK>>>(d_train, d_x_norm, trainN, D);
    }

    // ── Lloyd iterations ──
    tStep = SteadyClock::now();
    tlog("  │  Lloyd iterations (%d)...\n", iterations);

    float alpha = -2.0f, beta = 0.0f;

    for (int iter = 0; iter < iterations; ++iter) {
        auto tIter = SteadyClock::now();

        // 1. Centroid norms
        {
            int gridK = (K + BLK - 1) / BLK;
            rowNormsKernel<<<gridK, BLK>>>(d_centroids, d_c_norm, K, D);
        }

        // 2. GEMM: XCt = -2 * X · Cᵀ  (cublas column-major trick)
        //    cublas sees X as D×trainN col-major, C as D×K col-major
        //    We want Cᵀ·X = (K×D)·(D×trainN) = K×trainN col-major = trainN×K row-major
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  K, trainN, D,
                                  &alpha,
                                  d_centroids, D,  // Cᵀ
                                  d_train, D,       // X
                                  &beta,
                                  d_XCt, K));        // result: trainN×K row-major

        // 3. Assignment: dist = x_norm + c_norm + XCt (already has -2 factor)
        {
            int gridN = (trainN + BLK - 1) / BLK;
            assignKernel<<<gridN, BLK>>>(d_XCt, d_x_norm, d_c_norm,
                                          d_assign, nullptr, trainN, K);
        }

        // 4. Update centroids
        {
            int clearN = (K * D + BLK - 1) / BLK;
            clearCentroidsKernel<<<clearN, BLK>>>(d_centroids, d_counts, K, D);
        }
        {
            int gridN = (trainN + BLK - 1) / BLK;
            accumulateKernel<<<gridN, BLK>>>(d_train, d_assign, d_centroids, d_counts, trainN, D);
        }
        {
            int gridK = (K + BLK - 1) / BLK;
            divideCentroidsKernel<<<gridK, BLK>>>(d_centroids, d_counts, d_train,
                                                    K, D, trainN, 42 + iter);
        }

        cudaDeviceSynchronize();

        if ((iter + 1) % 5 == 0 || iter == 0 || iter == iterations - 1) {
            double totalSec = elapsedSec(tStep);
            double perIter = totalSec / (iter + 1);
            double eta = perIter * (iterations - iter - 1);
            tlog("  │    Iter %2d/%d: %.2fs/iter, elapsed %s, ETA %s\n",
                 iter + 1, iterations, elapsedSec(tIter),
                 fmtT(totalSec).c_str(), fmtT(eta).c_str());
        }
    }
    tlog("  │  Lloyd done: %s\n", fmtT(elapsedSec(tStep)).c_str());

    // ── Final assignment of ALL N leaves (batched for VRAM safety) ──
    tStep = SteadyClock::now();
    tlog("  │  Final assignment (%d leaves, batched)...\n", N);

    // Free training buffers to reclaim VRAM
    cudaFree(d_train); d_train = nullptr;

    // Centroid norms (final)
    {
        int gridK = (K + BLK - 1) / BLK;
        rowNormsKernel<<<gridK, BLK>>>(d_centroids, d_c_norm, K, D);
    }

    CudaKMeansResult result;
    result.K = K;
    result.codebook.resize(K * D);
    result.indices.resize(N);
    CUDA_CHECK(cudaMemcpy(result.codebook.data(), d_centroids,
                           (size_t)K * D * sizeof(float), cudaMemcpyDeviceToHost));

    // Batch size: keep VRAM under control
    // Each leaf needs: D floats (data) + K floats (GEMM) + 3 values (norm/dist/assign)
    // = (D + K + 3) * 4 bytes per leaf
    // Target: stay under 16 GB for batch buffers (leave headroom on 24 GB card)
    size_t bytesPerLeaf = ((size_t)D + K + 3) * sizeof(float);
    size_t targetBytes = (size_t)16 * 1024 * 1024 * 1024ULL; // 16 GB
    int batchSize = std::min(N, std::max(1, (int)(targetBytes / bytesPerLeaf)));
    batchSize = std::min(batchSize, 1000000); // cap at 1M for sanity
    tlog("  │    Batch size: %d leaves (%.1f GB per batch)\n",
         batchSize, batchSize * bytesPerLeaf / (1024.0*1024.0*1024.0));

    // Free training buffers and reallocate for batch
    cudaFree(d_XCt);     cudaFree(d_x_norm);
    cudaFree(d_minDist);  cudaFree(d_assign);
    CUDA_CHECK(cudaMalloc(&d_XCt,     (size_t)batchSize * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_norm,  batchSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_minDist, batchSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_assign,  batchSize * sizeof(int)));

    float* d_batch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_batch, (size_t)batchSize * D * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_mseSum, 0, sizeof(float)));
    std::vector<int> h_assign(batchSize);

    for (int offset = 0; offset < N; offset += batchSize) {
        int count = std::min(batchSize, N - offset);

        // Upload batch
        CUDA_CHECK(cudaMemcpy(d_batch, h_data + (size_t)offset * D,
                               (size_t)count * D * sizeof(float),
                               cudaMemcpyHostToDevice));

        // Batch norms
        int gridN = (count + BLK - 1) / BLK;
        rowNormsKernel<<<gridN, BLK>>>(d_batch, d_x_norm, count, D);

        // GEMM
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  K, count, D,
                                  &alpha,
                                  d_centroids, D,
                                  d_batch, D,
                                  &beta,
                                  d_XCt, K));

        // Assign + dist
        assignKernel<<<gridN, BLK>>>(d_XCt, d_x_norm, d_c_norm,
                                      d_assign, d_minDist, count, K);

        // MSE accumulation
        {
            int gridR = (count + BLK * 2 - 1) / (BLK * 2);
            sumReduceKernel<<<gridR, BLK, BLK * sizeof(float)>>>(d_minDist, d_mseSum, count);
        }

        // Download assignments
        CUDA_CHECK(cudaMemcpy(h_assign.data(), d_assign, count * sizeof(int),
                               cudaMemcpyDeviceToHost));
        for (int i = 0; i < count; ++i)
            result.indices[offset + i] = static_cast<uint16_t>(h_assign[i]);

        if (offset == 0 || (offset + count) == N || offset % (batchSize * 4) == 0) {
            tlog("  │    %d/%d (%.0f%%)\n", offset + count, N,
                 100.0 * (offset + count) / N);
        }
    }

    cudaDeviceSynchronize();

    float h_mseTotal = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_mseTotal, d_mseSum, sizeof(float), cudaMemcpyDeviceToHost));
    result.mse = h_mseTotal / ((double)N * D);
    result.psnr = (result.mse > 0) ? 10.0 * std::log10(1.0 / result.mse) : 99.0;

    tlog("  │  Final assignment: %s\n", fmtT(elapsedSec(tStep)).c_str());
    tlog("  └─ K-means total: %s | MSE=%.6f, PSNR=%.1f dB\n",
         fmtT(elapsedSec(tTotal)).c_str(), result.mse, result.psnr);

    // ── Cleanup ──
    cublasDestroy(cublas);
    if (d_train) cudaFree(d_train);
    cudaFree(d_batch);
    cudaFree(d_centroids);
    cudaFree(d_XCt);
    cudaFree(d_x_norm);
    cudaFree(d_c_norm);
    cudaFree(d_minDist);
    cudaFree(d_assign);
    cudaFree(d_counts);
    cudaFree(d_mseSum);

    return result;
}
