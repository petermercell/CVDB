# CVDB — Codebook VDB

**Experimental vector-quantized volume compression with direct GPU ray marching.**

CVDB compresses OpenVDB fog volumes by clustering leaf sub-blocks into a compact codebook using GPU-accelerated K-means. Unlike traditional compression, CVDB requires **no decompression step** — the ray marcher samples density directly from a GPU hash table and codebook during rendering.

- **4–5× file compression**, **5–6× VRAM reduction**
- **28.3 dB PSNR** on WDAS Cloud at highest quality tier
- Direct codebook sampling in the inner ray march loop — identical lighting to uncompressed NanoVDB
- Full K-means clustering of 15M sub-blocks in ~30 seconds on an RTX A5000

## How It Works

OpenVDB 8³ leaf nodes are split into 4³ sub-blocks (64-dimensional vectors). These are clustered via K-means using cuBLAS SGEMM for distance computation (`||x−c||² = ||x||² + ||c||² − 2x·c`). Each sub-block is stored as a codebook index (uint16) plus a 2³ gain map for local intensity correction. At render time, a spatial hash table provides O(1) lookup from voxel position to codebook entry.

### Quality Tiers

| Format | Block Size | Gain Maps | Compression | PSNR | Use Case |
|--------|-----------|-----------|-------------|------|----------|
| CVD1 | 8³ | No | ~85× | 23.7 dB | Preview / lookdev |
| CVD6 | 4³ | Yes + smoothing (FP16) | ~4.2× | 28.3 dB | Final rendering |

## Repository Contents

| File | Description |
|------|-------------|
| `vdb2cvdb.cpp` | OpenVDB → CVDB converter (sub-block extraction, gain maps, file I/O) |
| `cuda_kmeans.cu` | GPU K-means via cuBLAS SGEMM with K-means++ init |
| `cuda_kmeans.h` | K-means API header |
| `cvdb_temporal_encoder.py` | Temporal coherence encoder for animated sequences |
| `CMakeLists.txt` | Build system (requires CUDA, OpenVDB, TBB, Blosc, Boost) |

## Building

### Requirements

- CUDA Toolkit (tested with 12.8)
- OpenVDB (static library)
- TBB, Blosc, Boost.Iostreams, zlib
- CMake ≥ 3.14
- GPU with compute capability ≥ 7.5 (Turing+)

### Build

```bash
mkdir build && cd build
cmake ..
make -j8
```

## Usage

### Single file conversion

```bash
./vdb2cvdb input.vdb -k 8192 -i 50 -s 400000
```

### Batch sequence conversion

```bash
./vdb2cvdb /path/to/sequence/ -o /path/to/output/ -k 8192 -i 50 -s 400000
```

### Temporal encoding (animated sequences)

```bash
pip install numpy scikit-learn
conda install -c conda-forge faiss-cpu

python cvdb_temporal_encoder.py \
    "/path/to/input/Explosion_####.cvdb" \
    "/path/to/output/Explosion_####.cvdb" \
    --frames 1-100
```

### Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `-k` | Codebook size (number of centroids) | 4096 |
| `-i` | Lloyd iterations | 50 |
| `-s` | Training subsample size (0 = all) | 0 |
| `-o` | Output directory (batch mode) | — |

## File Format

The `.cvdb` binary format (CVD6) stores multiple grids (density, flames, temperature) in a single file, each independently compressed with per-grid normalization:

1. **Magic** — 4 bytes (`CVD6`)
2. **Grid count** — uint32
3. **Per-grid sections**, each containing:
   - **Grid metadata** — name, voxel size, bounding box
   - **Normalization scale** — float32 (maps values back to original range)
   - **Codebook** — K × 64 half-precision floats (FP16, 4³ sub-block centroids)
   - **Block indices** — uint16 per sub-block
   - **Block origins** — int32 × 3 per sub-block
   - **Gain maps** — 8 floats per sub-block (2³ quadrant corrections)

All arrays are tightly packed for direct memory mapping.

## Technical Paper

A detailed description of the method, including compression results and performance benchmarks, is available in:

> Peter Mercell. *Codebook VDB: Vector-Quantized Volume Compression with Direct GPU Ray Marching.* 2026. [CVDB_Paper_Mercell_2026.docx](CVDB_Paper_Mercell_2026.docx)

## Acknowledgments

This project was inspired by [VQVDB](https://github.com/ZephirFXEC/VQVDB) by Enzo Crema, which applies VQ-VAE neural network compression to OpenVDB volumes. CVDB takes a different approach — classical vector quantization with GPU-accelerated K-means instead of learned encoders — but the idea of applying codebook-based compression to VDB leaf data originates from VQVDB. Paper edited with the assistance of Claude (Anthropic).

## Support

CVDB is free and open source. If you find it useful in your work and want to help keep these tools alive, consider supporting development on **[Patreon](https://www.patreon.com/cw/PeterMercell)**. Your support helps me dedicate time to building new open-source VFX tools and keeping everything available to the community.

## License

[MIT](LICENSE)
