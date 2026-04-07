/*
 * cuda_kmeans.h — GPU-accelerated K-means via cuBLAS GEMM
 *
 * Distance computation uses the identity:
 *   ||x - c||² = ||x||² + ||c||² - 2·x·c
 *
 * The x·c term is a matrix multiply handled by cublasSgemm,
 * giving ~1000x speedup over CPU for large N×K×D.
 */

#pragma once
#include <cstdint>
#include <vector>

struct CudaKMeansResult {
    std::vector<float>    codebook;  // K × 512 floats
    std::vector<uint16_t> indices;   // N assignments
    int    K;
    double mse;
    double psnr;
};

/*
 * Run K-means clustering on GPU.
 *
 * @param h_data      Contiguous N×D float array (all leaf voxels)
 * @param N           Total number of leaves
 * @param D           Dimensionality (512 for 8³ blocks)
 * @param K           Codebook size
 * @param iterations  Lloyd iterations
 * @param subsampleN  Training subset size (0 = use all)
 * @return            Codebook + per-leaf assignments + quality metrics
 */
CudaKMeansResult cudaKMeans(
    const float* h_data,
    int N, int D, int K,
    int iterations,
    int subsampleN);
