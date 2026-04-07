/*
 * vdb2cvdb — Convert OpenVDB to Codebook VDB for direct GPU ray march.
 *
 * Compresses ALL float grids (density, flames, temperature, etc.) into
 * a single .cvdb file. Each grid gets independent K-means compression.
 *
 * Usage:
 *   vdb2cvdb input.vdb [output.cvdb] [-k 1024] [-s 200000] [-i 50]
 *   vdb2cvdb /path/to/directory/   (batch convert all .vdb files)
 */

#include <openvdb/openvdb.h>
#include "cuda_kmeans.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <climits>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <filesystem>
#include <unordered_map>

namespace fs = std::filesystem;

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

static double elapsed(TimePoint start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
static std::string fmtTime(double sec) {
    if (sec < 60.0) { char b[32]; std::snprintf(b,sizeof(b),"%.1f sec",sec); return b; }
    int m=int(sec)/60; double s=sec-m*60;
    char b[32]; std::snprintf(b,sizeof(b),"%dm %.1fs",m,s); return b;
}

// ═══════════════════════════════════════════════════════════════════════════
// Leaf / SubBlock extraction
// ═══════════════════════════════════════════════════════════════════════════

struct Leaf { int32_t origin[3]; float voxels[512]; };
struct SubBlock { int32_t origin[3]; float voxels[64]; };

std::vector<Leaf> extractLeaves(openvdb::FloatGrid::Ptr grid)
{
    std::vector<Leaf> leaves;
    leaves.reserve(grid->tree().leafCount());
    for (auto it = grid->tree().cbeginLeaf(); it; ++it) {
        Leaf leaf;
        const auto& o = it->origin();
        leaf.origin[0]=o.x(); leaf.origin[1]=o.y(); leaf.origin[2]=o.z();
        for (int i=0; i<512; ++i) leaf.voxels[i] = it->getValue(i);
        leaves.push_back(leaf);
    }
    return leaves;
}

std::vector<SubBlock> splitIntoSubBlocks(const std::vector<Leaf>& leaves)
{
    std::vector<SubBlock> subs;
    subs.reserve(leaves.size() * 8);
    for (const auto& leaf : leaves) {
        for (int sz=0;sz<2;++sz) for(int sy=0;sy<2;++sy) for(int sx=0;sx<2;++sx) {
            SubBlock sb;
            sb.origin[0]=leaf.origin[0]+sx*4;
            sb.origin[1]=leaf.origin[1]+sy*4;
            sb.origin[2]=leaf.origin[2]+sz*4;
            for (int x=0;x<4;++x) for(int y=0;y<4;++y) for(int z=0;z<4;++z)
                sb.voxels[x*16+y*4+z] = leaf.voxels[(sx*4+x)*64+(sy*4+y)*8+(sz*4+z)];
            subs.push_back(sb);
        }
    }
    return subs;
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-grid compressed result
// ═══════════════════════════════════════════════════════════════════════════

struct CompressedGrid {
    std::string           name;
    std::vector<float>    codebook;     // K × vpb (normalized to 0-1)
    std::vector<uint16_t> indices;      // numBlocks
    std::vector<int32_t>  origins;      // numBlocks × 3
    std::vector<float>    gainMaps;     // numBlocks × 8
    uint32_t              numBlocks;
    int                   K;
    int                   blockSize;    // always 4 for CVD4/5
    float                 voxelSize;
    float                 normScale;    // max value before normalization
    double                bboxMin[3];
    double                bboxMax[3];
    double                psnr;
};

// ═══════════════════════════════════════════════════════════════════════════
// Compress one grid → CompressedGrid
// ═══════════════════════════════════════════════════════════════════════════

CompressedGrid compressOneGrid(openvdb::FloatGrid::Ptr floatGrid,
                                int K, int subsampleN, int iterations)
{
    CompressedGrid cg;
    cg.name = floatGrid->getName();
    cg.blockSize = 4;
    const int vpb = 64;

    std::printf("\n  ┌─── Grid: '%s' (%lld active voxels) ───\n",
        cg.name.c_str(), static_cast<long long>(floatGrid->activeVoxelCount()));

    // Extract & split
    auto tStep = Clock::now();
    auto leaves = extractLeaves(floatGrid);
    std::printf("  │  Extract: %zu leaves in %s\n", leaves.size(), fmtTime(elapsed(tStep)).c_str());

    tStep = Clock::now();
    auto subs = splitIntoSubBlocks(leaves);
    std::printf("  │  Split: %zu sub-blocks (4³) in %s\n", subs.size(), fmtTime(elapsed(tStep)).c_str());
    leaves.clear();

    uint32_t N = static_cast<uint32_t>(subs.size());
    cg.numBlocks = N;

    // Flatten for GPU K-means
    tStep = Clock::now();
    std::vector<float> flatData(subs.size() * vpb);
    for (size_t i = 0; i < subs.size(); ++i)
        std::memcpy(&flatData[i * vpb], subs[i].voxels, vpb * sizeof(float));

    // ── Normalize to [0, 1] for better K-means clustering ──
    float maxVal = 0.0f;
    for (size_t i = 0; i < flatData.size(); ++i)
        if (flatData[i] > maxVal) maxVal = flatData[i];
    cg.normScale = (maxVal > 1e-10f) ? maxVal : 1.0f;

    if (cg.normScale != 1.0f) {
        float invScale = 1.0f / cg.normScale;
        for (size_t i = 0; i < flatData.size(); ++i)
            flatData[i] *= invScale;
        // Also normalize the sub-block voxels (needed for gain map computation later)
        for (auto& sb : subs)
            for (int j = 0; j < vpb; ++j)
                sb.voxels[j] *= invScale;
        std::printf("  │  Normalized: max=%.4f → [0, 1] (scale stored in file)\n", cg.normScale);
    } else {
        std::printf("  │  Already [0, 1] range (normScale=1.0)\n");
    }

    std::printf("  │  CUDA K-means (K=%d, %u blocks, 64-dim)...\n", K, N);
    auto ckm = cudaKMeans(flatData.data(), static_cast<int>(subs.size()), vpb, K, iterations, subsampleN);
    flatData.clear();

    cg.codebook = std::move(ckm.codebook);
    cg.indices  = std::move(ckm.indices);
    cg.K        = ckm.K;
    cg.psnr     = ckm.psnr;
    std::printf("  │  K-means done: PSNR %.1f dB\n", cg.psnr);

    // Bounding box — computed from sub-block origins + grid world origin.
    // This ensures gridOffset = constant across all frames in a sequence.
    // (Using evalActiveVoxelBoundingBox would give a different offset per frame
    // because the active region grows/moves, causing temporal jitter.)
    const auto& xform = floatGrid->transform();
    openvdb::Vec3d gridOrigin = xform.indexToWorld(openvdb::Vec3d(0, 0, 0));
    cg.voxelSize = static_cast<float>(floatGrid->voxelSize()[0]);

    int32_t oMin[3] = {INT_MAX, INT_MAX, INT_MAX};
    int32_t oMax[3] = {INT_MIN, INT_MIN, INT_MIN};
    for (uint32_t i = 0; i < N; ++i) {
        for (int j = 0; j < 3; ++j) {
            oMin[j] = std::min(oMin[j], subs[i].origin[j]);
            oMax[j] = std::max(oMax[j], subs[i].origin[j] + cg.blockSize);
        }
    }
    for (int j = 0; j < 3; ++j) {
        cg.bboxMin[j] = oMin[j] * cg.voxelSize + gridOrigin[j];
        cg.bboxMax[j] = oMax[j] * cg.voxelSize + gridOrigin[j];
    }
    std::printf("  │  Grid origin: (%.2f, %.2f, %.2f)\n",
        gridOrigin[0], gridOrigin[1], gridOrigin[2]);

    // Gain maps (2³ per sub-block)
    tStep = Clock::now();
    cg.gainMaps.resize(N * 8);
    for (uint32_t i = 0; i < N; ++i) {
        int k = cg.indices[i];
        const float* cb = cg.codebook.data() + k * vpb;
        const float* lv = subs[i].voxels;
        for (int gz=0;gz<2;++gz) for(int gy=0;gy<2;++gy) for(int gx=0;gx<2;++gx) {
            float sumA=0, sumC=0;
            for(int x=gx*2;x<gx*2+2;++x) for(int y=gy*2;y<gy*2+2;++y) for(int z=gz*2;z<gz*2+2;++z) {
                int idx=x*16+y*4+z; sumA+=lv[idx]; sumC+=cb[idx];
            }
            float gain = (sumC>1e-6f) ? (sumA/sumC) : 1.0f;
            cg.gainMaps[i*8 + gx+gy*2+gz*4] = std::max(0.0f, std::min(gain, 4.0f));
        }
    }
    std::printf("  │  Gain maps: %s\n", fmtTime(elapsed(tStep)).c_str());

    // Smooth gain maps
    tStep = Clock::now();
    {
        std::unordered_map<int64_t, uint32_t> originMap;
        originMap.reserve(N * 2);
        for (uint32_t i = 0; i < N; ++i) {
            int bx=subs[i].origin[0]/4, by=subs[i].origin[1]/4, bz=subs[i].origin[2]/4;
            int64_t key=(int64_t(bx)&0xFFFF)|((int64_t(by)&0xFFFF)<<16)|((int64_t(bz)&0xFFFF)<<32);
            originMap[key] = i;
        }
        for (int pass=0; pass<3; ++pass) {
            std::vector<float> smoothed = cg.gainMaps;
            for (uint32_t i = 0; i < N; ++i) {
                int bx=subs[i].origin[0]/4, by=subs[i].origin[1]/4, bz=subs[i].origin[2]/4;
                // +X
                { int64_t nk=(int64_t(bx+1)&0xFFFF)|((int64_t(by)&0xFFFF)<<16)|((int64_t(bz)&0xFFFF)<<32);
                  auto it=originMap.find(nk); if(it!=originMap.end()){ uint32_t j=it->second;
                    for(int gy=0;gy<2;++gy) for(int gz=0;gz<2;++gz){
                        int a=1+gy*2+gz*4, b=0+gy*2+gz*4;
                        float avg=(cg.gainMaps[i*8+a]+cg.gainMaps[j*8+b])*0.5f;
                        smoothed[i*8+a]=avg; smoothed[j*8+b]=avg; }}}
                // +Y
                { int64_t nk=(int64_t(bx)&0xFFFF)|((int64_t(by+1)&0xFFFF)<<16)|((int64_t(bz)&0xFFFF)<<32);
                  auto it=originMap.find(nk); if(it!=originMap.end()){ uint32_t j=it->second;
                    for(int gx=0;gx<2;++gx) for(int gz=0;gz<2;++gz){
                        int a=gx+2+gz*4, b=gx+0+gz*4;
                        float avg=(cg.gainMaps[i*8+a]+cg.gainMaps[j*8+b])*0.5f;
                        smoothed[i*8+a]=avg; smoothed[j*8+b]=avg; }}}
                // +Z
                { int64_t nk=(int64_t(bx)&0xFFFF)|((int64_t(by)&0xFFFF)<<16)|((int64_t(bz+1)&0xFFFF)<<32);
                  auto it=originMap.find(nk); if(it!=originMap.end()){ uint32_t j=it->second;
                    for(int gx=0;gx<2;++gx) for(int gy=0;gy<2;++gy){
                        int a=gx+gy*2+4, b=gx+gy*2+0;
                        float avg=(cg.gainMaps[i*8+a]+cg.gainMaps[j*8+b])*0.5f;
                        smoothed[i*8+a]=avg; smoothed[j*8+b]=avg; }}}
            }
            cg.gainMaps = smoothed;
        }
    }
    std::printf("  │  Gain smooth: %s\n", fmtTime(elapsed(tStep)).c_str());

    // Origins
    cg.origins.resize(N * 3);
    for (uint32_t i = 0; i < N; ++i) {
        cg.origins[i*3+0]=subs[i].origin[0];
        cg.origins[i*3+1]=subs[i].origin[1];
        cg.origins[i*3+2]=subs[i].origin[2];
    }

    std::printf("  └─── '%s' done: K=%d, %u blocks, PSNR %.1f dB\n",
        cg.name.c_str(), cg.K, N, cg.psnr);
    return cg;
}

// ═══════════════════════════════════════════════════════════════════════════
// .cvdb writer — multi-grid
// ═══════════════════════════════════════════════════════════════════════════

bool writeCVDB(const std::string& path, const std::vector<CompressedGrid>& grids)
{
    if (grids.empty()) return false;
    auto tWrite = Clock::now();
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;

    int blockSize = grids[0].blockSize;
    bool hasGain = !grids[0].gainMaps.empty();
    // CVD6 = FP16 codebook (default for 4³ blocks)
    // CVD5 = FP32 codebook (legacy)
    const char* magic = "CVD6";
    if (blockSize != 4) magic = hasGain ? "CVD2" : "CVD1";  // legacy fallback
    out.write(magic, 4);

    uint32_t numGrids = static_cast<uint32_t>(grids.size());
    out.write(reinterpret_cast<const char*>(&numGrids), 4);

    for (const auto& cg : grids) {
        int vpb = cg.blockSize * cg.blockSize * cg.blockSize;

        char name[64] = {};
        std::strncpy(name, cg.name.c_str(), 63);
        out.write(name, 64);

        uint32_t numBlocks = cg.numBlocks;
        uint32_t codebookK = static_cast<uint32_t>(cg.K);
        uint32_t indexBytes = (cg.K <= 256) ? 1 : 2;
        float vs[3] = { cg.voxelSize, cg.voxelSize, cg.voxelSize };

        out.write(reinterpret_cast<const char*>(&numBlocks), 4);
        out.write(reinterpret_cast<const char*>(&codebookK), 4);
        out.write(reinterpret_cast<const char*>(&indexBytes), 4);
        out.write(reinterpret_cast<const char*>(vs), 12);
        out.write(reinterpret_cast<const char*>(cg.bboxMin), 24);
        out.write(reinterpret_cast<const char*>(cg.bboxMax), 24);

        // CVD5: write per-grid normalization scale
        float ns = cg.normScale;
        out.write(reinterpret_cast<const char*>(&ns), 4);

        // Codebook — FP16 for CVD6 (half the bytes)
        {
            size_t n = cg.K * vpb;
            std::vector<uint16_t> cbHalf(n);
            for (size_t i = 0; i < n; ++i) {
                float v = cg.codebook[i];
                if (v > 65504.f) v = 65504.f;
                if (v < -65504.f) v = -65504.f;
                uint32_t f = *reinterpret_cast<const uint32_t*>(&v);
                uint32_t s = (f >> 16) & 0x8000;
                int32_t e = ((f >> 23) & 0xFF) - 127 + 15;
                uint32_t m = (f >> 13) & 0x3FF;
                if (e <= 0) cbHalf[i] = uint16_t(s);
                else if (e >= 31) cbHalf[i] = uint16_t(s | 0x7C00);
                else cbHalf[i] = uint16_t(s | (e << 10) | m);
            }
            out.write(reinterpret_cast<const char*>(cbHalf.data()), n * sizeof(uint16_t));
        }

        // Indices
        if (indexBytes == 1) {
            std::vector<uint8_t> idx8(numBlocks);
            for (uint32_t i = 0; i < numBlocks; ++i) idx8[i] = static_cast<uint8_t>(cg.indices[i]);
            out.write(reinterpret_cast<const char*>(idx8.data()), numBlocks);
        } else {
            out.write(reinterpret_cast<const char*>(cg.indices.data()), numBlocks * sizeof(uint16_t));
        }

        // Origins
        out.write(reinterpret_cast<const char*>(cg.origins.data()), numBlocks * 3 * sizeof(int32_t));

        // Gain maps
        if (hasGain && !cg.gainMaps.empty()) {
            out.write(reinterpret_cast<const char*>(cg.gainMaps.data()), numBlocks * 8 * sizeof(float));
        }

        std::printf("  Grid '%s': K=%d, %u blocks, %.1f MB codebook (FP16), PSNR %.1f dB, normScale=%.4f\n",
            cg.name.c_str(), cg.K, numBlocks,
            cg.K * vpb * sizeof(uint16_t) / (1024.0*1024.0), cg.psnr, cg.normScale);
    }

    out.close();
    std::printf("  Write: %s\n", fmtTime(elapsed(tWrite)).c_str());
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Convert one VDB file — all float grids
// ═══════════════════════════════════════════════════════════════════════════

bool convertFile(const std::string& vdbPath, const std::string& cvdbPath,
                 int K, int subsampleN, int iterations)
{
    auto tFile = Clock::now();

    openvdb::io::File file(vdbPath);
    try { file.open(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Error opening %s: %s\n", vdbPath.c_str(), e.what());
        return false;
    }
    auto grids = file.getGrids();
    file.close();

    if (!grids || grids->empty()) {
        std::fprintf(stderr, "No grids in %s\n", vdbPath.c_str());
        return false;
    }

    // Collect all float grids
    std::vector<openvdb::FloatGrid::Ptr> floatGrids;
    for (auto& gridBase : *grids) {
        auto fg = openvdb::gridPtrCast<openvdb::FloatGrid>(gridBase);
        if (fg && !fg->empty()) {
            std::printf("  Found grid: '%s' (%lld active voxels, voxel %.4f)\n",
                fg->getName().c_str(),
                static_cast<long long>(fg->activeVoxelCount()),
                fg->voxelSize()[0]);
            floatGrids.push_back(fg);
        }
    }

    if (floatGrids.empty()) {
        std::fprintf(stderr, "No float grids found in %s\n", vdbPath.c_str());
        return false;
    }

    std::printf("\n  Converting %zu grid%s: ", floatGrids.size(), floatGrids.size()>1?"s":"");
    for (size_t i = 0; i < floatGrids.size(); ++i) {
        if (i > 0) std::printf(", ");
        std::printf("'%s'", floatGrids[i]->getName().c_str());
    }
    std::printf("\n");

    // Compress each grid independently → all into one .cvdb
    std::vector<CompressedGrid> compressed;
    for (auto& fg : floatGrids)
        compressed.push_back(compressOneGrid(fg, K, subsampleN, iterations));

    std::printf("\n  Writing %zu grids to %s...\n", compressed.size(), cvdbPath.c_str());
    writeCVDB(cvdbPath, compressed);

    // Summary
    auto origVDB = fs::file_size(vdbPath);
    auto newCVDB = fs::file_size(cvdbPath);
    double ratio = static_cast<double>(origVDB) / newCVDB;
    std::printf("\n  ════════════════════════════════════════════\n");
    std::printf("  SUMMARY\n");
    std::printf("  ════════════════════════════════════════════\n");
    std::printf("  Input:        %s (%.1f MB)\n", vdbPath.c_str(), origVDB/(1024.0*1024.0));
    std::printf("  Output:       %s (%.1f MB)\n", cvdbPath.c_str(), newCVDB/(1024.0*1024.0));
    std::printf("  Compression:  %.1fx\n", ratio);
    std::printf("  Grids:        %zu\n", compressed.size());
    for (const auto& cg : compressed)
        std::printf("    %-16s K=%4d  %6u blocks  PSNR %.1f dB  normScale=%.4f\n",
            cg.name.c_str(), cg.K, cg.numBlocks, cg.psnr, cg.normScale);
    std::printf("  Total time:   %s\n", fmtTime(elapsed(tFile)).c_str());
    std::printf("  ════════════════════════════════════════════\n");
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[])
{
    int K = 1024;
    int subsampleN = 200000;
    int iterations = 50;
    std::string input, outputDir;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-k" && i+1 < argc) K = std::atoi(argv[++i]);
        else if (arg == "-s" && i+1 < argc) subsampleN = std::atoi(argv[++i]);
        else if (arg == "-i" && i+1 < argc) iterations = std::atoi(argv[++i]);
        else if (arg == "-o" && i+1 < argc) outputDir = argv[++i];
        else if (input.empty()) input = arg;
    }

    if (input.empty()) {
        std::printf("Usage: vdb2cvdb <input.vdb|directory> [-o output_dir] [-k 1024] [-s 200000] [-i 50]\n");
        std::printf("\nAll float grids (density, flames, temperature) go into ONE .cvdb file.\n");
        std::printf("The Nuke plugin auto-discovers grids by name.\n");
        std::printf("\nExamples:\n");
        std::printf("  vdb2cvdb explosion_0050.vdb                    → explosion_0050.cvdb (same dir)\n");
        std::printf("  vdb2cvdb explosion_0050.vdb -o /tmp/cvdb/      → /tmp/cvdb/explosion_0050.cvdb\n");
        std::printf("  vdb2cvdb /path/to/vdb_sequence/ -o /tmp/cvdb/  → batch convert all .vdb files\n");
        std::printf("\nOptions:\n");
        std::printf("  -k N   Codebook size per grid (default: 1024)\n");
        std::printf("  -s N   Training subsample (default: 200000)\n");
        std::printf("  -i N   K-means iterations (default: 50)\n");
        std::printf("  -o DIR Output directory (default: same as input)\n");
        return 1;
    }

    openvdb::initialize();

    // Helper: compute output path from input path
    auto makeOutputPath = [&](const std::string& vdbPath) -> std::string {
        std::string stem = fs::path(vdbPath).stem().string();
        // Strip double extension like .vdb.vdb
        if (stem.size() > 4 && stem.substr(stem.size()-4) == ".vdb")
            stem = stem.substr(0, stem.size()-4);
        std::string dir = outputDir.empty()
            ? fs::path(vdbPath).parent_path().string()
            : outputDir;
        if (!dir.empty() && dir.back() != '/') dir += '/';
        if (!outputDir.empty()) fs::create_directories(dir);
        return dir + stem + ".cvdb";
    };

    if (fs::is_directory(input)) {
        auto tBatch = Clock::now();
        std::vector<std::string> files;
        for (auto& e : fs::directory_iterator(input))
            if (e.path().extension() == ".vdb") files.push_back(e.path().string());
        std::sort(files.begin(), files.end());

        std::printf("Found %zu .vdb files, K=%d\n", files.size(), K);
        if (!outputDir.empty())
            std::printf("Output dir: %s\n", outputDir.c_str());
        int ok=0, fail=0;
        for (auto& f : files) {
            std::string out = makeOutputPath(f);
            std::printf("\n[%d/%zu] %s → %s\n", ok+fail+1, files.size(), f.c_str(), out.c_str());
            if (convertFile(f, out, K, subsampleN, iterations)) ++ok; else ++fail;
        }
        std::printf("\nBatch done: %d converted, %d failed, total %s\n",
            ok, fail, fmtTime(elapsed(tBatch)).c_str());
        return fail > 0 ? 1 : 0;
    }

    std::string output = makeOutputPath(input);
    std::printf("Converting %s → %s (K=%d, all grids in one file)\n", input.c_str(), output.c_str(), K);
    return convertFile(input, output, K, subsampleN, iterations) ? 0 : 1;
}
