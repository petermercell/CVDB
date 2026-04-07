#!/usr/bin/env python3
"""
cvdb_temporal_encoder.py — Re-encode .cvdb sequences with shared temporal codebook.

Reads your existing per-frame .cvdb files, pools block data across frames,
trains ONE shared codebook via k-means, then re-encodes every frame against
that shared codebook. The Nuke plugin detects identical codebooks between
frames and skips the GPU upload.

Requirements:
    pip install numpy
    (optional) pip install scikit-learn   — 5-10x faster k-means

Usage:
    # Basic — auto-discovers all grids in each .cvdb file
    python cvdb_temporal_encoder.py \
        /path/to/explosion_####.cvdb \
        /path/to/output_####.cvdb \
        --frames 1-100

    # Explicit frame list and settings
    python cvdb_temporal_encoder.py \
        /sims/smoke_%04d.cvdb \
        /cvdb/smoke_%04d.cvdb \
        --frames 1-200 \
        --K 8192 \
        --sample-frames 16 \
        --sample-blocks 80000
"""

import argparse
import math
import os
import re
import struct
import sys
import time

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# CVD6 reader (pure Python — matches CVDBLoader.h exactly)
# ═══════════════════════════════════════════════════════════════════════════

def _read(f, fmt):
    sz = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(sz))

def fp16_to_fp32(arr_u16):
    """uint16 viewed as IEEE 754 half → float32, via numpy."""
    return arr_u16.view(np.float16).astype(np.float32)


def read_cvdb(path):
    """Read a .cvdb file. Returns (magic_str, [grid_dict, ...])."""
    with open(path, 'rb') as f:
        magic = f.read(4).decode('ascii')
        ver = magic  # e.g. "CVD6"
        if magic[:3] != 'CVD' or magic[3] not in '123456':
            raise ValueError(f"Not a valid .cvdb file: {path} (magic={magic!r})")

        is_v4plus = magic[3] in '456'
        is_v5plus = magic[3] in '56'
        is_v6     = magic[3] == '6'
        is_v3     = magic[3] == '3'
        is_v2plus = magic[3] in '23456'

        num_grids = _read(f, '<I')[0]
        grids = []

        for _ in range(num_grids):
            g = {}
            name_raw = f.read(64)
            g['name'] = name_raw.split(b'\x00', 1)[0].decode('utf-8', errors='replace')

            g['num_leaves'], g['K'], g['index_bytes'] = _read(f, '<III')
            g['voxel_size'] = _read(f, '<fff')
            g['bbox_min'] = _read(f, '<ddd')
            g['bbox_max'] = _read(f, '<ddd')

            if is_v5plus:
                g['norm_scale'] = _read(f, '<f')[0]
            else:
                g['norm_scale'] = 1.0

            bs = 4 if is_v4plus else 8
            g['block_size'] = bs
            vpb = bs ** 3
            g['vpb'] = vpb
            N = g['num_leaves']
            K = g['K']

            # Codebook
            if is_v6:
                raw = np.frombuffer(f.read(K * vpb * 2), dtype=np.uint16).copy()
                g['codebook_fp16'] = raw
                g['codebook'] = fp16_to_fp32(raw) * g['norm_scale']
                g['is_fp16'] = True
            else:
                g['codebook'] = np.frombuffer(f.read(K * vpb * 4), dtype=np.float32).copy()
                if is_v5plus:
                    g['codebook'] *= g['norm_scale']
                g['codebook_fp16'] = None
                g['is_fp16'] = False

            # Indices
            if g['index_bytes'] == 1:
                idx8 = np.frombuffer(f.read(N), dtype=np.uint8)
                g['indices'] = idx8.astype(np.uint16)
            else:
                g['indices'] = np.frombuffer(f.read(N * 2), dtype=np.uint16).copy()

            # Origins
            g['origins'] = np.frombuffer(f.read(N * 3 * 4), dtype=np.int32).copy().reshape(N, 3)

            # Gain maps
            if is_v2plus:
                g['gain_maps'] = np.frombuffer(f.read(N * 8 * 4), dtype=np.float32).copy().reshape(N, 8)
                g['has_gain'] = True
            else:
                g['gain_maps'] = np.ones((N, 8), dtype=np.float32)
                g['has_gain'] = False

            # Residuals (CVD3 only)
            if is_v3:
                g['residuals'] = np.frombuffer(f.read(N * 216 * 4), dtype=np.float32).copy()
            else:
                g['residuals'] = None

            grids.append(g)

    return ver, grids


# ═══════════════════════════════════════════════════════════════════════════
# Block reconstruction from codebook + indices + gain maps
# ═══════════════════════════════════════════════════════════════════════════

def reconstruct_blocks(grid):
    """Reconstruct per-leaf block data from codebook, indices, and gain maps.

    Returns: np.ndarray of shape (N, vpb), float32
    """
    N = grid['num_leaves']
    K = grid['K']
    bs = grid['block_size']
    vpb = grid['vpb']

    codebook = grid['codebook'].reshape(K, vpb)  # already in original scale
    indices = grid['indices']
    gain_maps = grid['gain_maps']  # (N, 8)

    # Look up codebook entry per leaf
    blocks = codebook[indices]  # (N, vpb)

    # Apply gain maps (2³ per-octant scale)
    half = bs // 2
    for gi in range(8):
        gx = (gi >> 0) & 1
        gy = (gi >> 1) & 1
        gz = (gi >> 2) & 1

        # Build mask of voxel indices belonging to this octant
        voxel_mask = []
        for lx in range(half * gx, half * (gx + 1)):
            for ly in range(half * gy, half * (gy + 1)):
                for lz in range(half * gz, half * (gz + 1)):
                    voxel_mask.append(lx * bs * bs + ly * bs + lz)
        voxel_mask = np.array(voxel_mask)

        # Scale octant voxels by gain
        blocks[:, voxel_mask] *= gain_maps[:, gi:gi+1]

    return blocks


# ═══════════════════════════════════════════════════════════════════════════
# K-means codebook training
# ═══════════════════════════════════════════════════════════════════════════

def kmeans_codebook(blocks, K, max_iter=30, verbose=True):
    """Train a codebook via mini-batch k-means.

    Args:
        blocks: (N, vpb) float32 — pooled training blocks
        K: codebook size
    Returns:
        codebook: (K, vpb) float32
    """
    N, vpb = blocks.shape

    if N <= K:
        cb = np.zeros((K, vpb), dtype=np.float32)
        cb[:N] = blocks
        if verbose:
            print(f"    Only {N} blocks ≤ K={K} — using blocks directly")
        return cb

    # ── Try sklearn (5-10× faster) ──
    try:
        from sklearn.cluster import MiniBatchKMeans
        if verbose:
            print(f"    sklearn MiniBatchKMeans: K={K}, N={N}, dims={vpb}")
        km = MiniBatchKMeans(
            n_clusters=K, max_iter=max_iter,
            batch_size=min(10000, N), n_init=1,
            random_state=42, verbose=0
        )
        km.fit(blocks)
        if verbose:
            print(f"    inertia={km.inertia_:.2f}")
        return km.cluster_centers_.astype(np.float32)
    except ImportError:
        pass

    # ── Numpy fallback ──
    if verbose:
        print(f"    numpy k-means: K={K}, N={N}, dims={vpb}")
        print(f"    (install scikit-learn for 5-10x speedup)")

    rng = np.random.RandomState(42)
    idx = rng.choice(N, K, replace=False)
    codebook = blocks[idx].copy()

    BATCH = 40000
    for it in range(max_iter):
        t0 = time.time()

        # ── Assign ──
        assignments = np.empty(N, dtype=np.int32)
        for s in range(0, N, BATCH):
            e = min(s + BATCH, N)
            chunk = blocks[s:e]
            # ||a-b||² = ||a||² + ||b||² - 2 a·b
            a2 = np.sum(chunk ** 2, axis=1, keepdims=True)
            b2 = np.sum(codebook ** 2, axis=1, keepdims=True).T
            dists = a2 + b2 - 2.0 * (chunk @ codebook.T)
            assignments[s:e] = np.argmin(dists, axis=1)

        # ── Update ──
        new_cb = np.zeros_like(codebook)
        counts = np.zeros(K, dtype=np.int64)
        # Use np.add.at for efficiency
        np.add.at(new_cb, assignments, blocks)
        np.add.at(counts, assignments, 1)

        alive = counts > 0
        new_cb[alive] /= counts[alive, None]

        # Reinit dead centroids
        dead = ~alive
        n_dead = dead.sum()
        if n_dead > 0:
            new_cb[dead] = blocks[rng.choice(N, n_dead, replace=False)]

        shift = np.sqrt(np.mean((new_cb - codebook) ** 2))
        codebook = new_cb
        dt = time.time() - t0

        if verbose and (it < 3 or (it + 1) % 5 == 0 or it == max_iter - 1):
            print(f"      iter {it+1:3d}/{max_iter}: shift={shift:.6f}, "
                  f"alive={alive.sum()}/{K}, dead={n_dead}, {dt:.1f}s")

        if shift < 1e-6:
            if verbose:
                print(f"      Converged at iteration {it+1}")
            break

    return codebook


# ═══════════════════════════════════════════════════════════════════════════
# Per-frame encoding against shared codebook
# ═══════════════════════════════════════════════════════════════════════════

def encode_blocks(blocks, codebook, block_size=4):
    """Encode blocks against a codebook.

    Returns:
        indices:   (N,) uint16
        gain_maps: (N, 8) float32
    """
    N, vpb = blocks.shape
    K = codebook.shape[0]
    BATCH = 20000

    # ── Find nearest codebook entry per block ──
    indices = np.empty(N, dtype=np.uint16)
    for s in range(0, N, BATCH):
        e = min(s + BATCH, N)
        chunk = blocks[s:e]
        a2 = np.sum(chunk ** 2, axis=1, keepdims=True)
        b2 = np.sum(codebook ** 2, axis=1, keepdims=True).T
        dists = a2 + b2 - 2.0 * (chunk @ codebook.T)
        indices[s:e] = np.argmin(dists, axis=1).astype(np.uint16)

    # ── Compute 2³ gain maps ──
    half = block_size // 2
    gain_maps = np.ones((N, 8), dtype=np.float32)

    # Precompute octant voxel masks once
    octant_masks = []
    for gi in range(8):
        gx = (gi >> 0) & 1
        gy = (gi >> 1) & 1
        gz = (gi >> 2) & 1
        mask = []
        for lx in range(half * gx, half * (gx + 1)):
            for ly in range(half * gy, half * (gy + 1)):
                for lz in range(half * gz, half * (gz + 1)):
                    mask.append(lx * block_size * block_size + ly * block_size + lz)
        octant_masks.append(np.array(mask))

    cb_entries = codebook[indices]  # (N, vpb)

    for gi in range(8):
        oi = octant_masks[gi]
        actual_sum = np.sum(blocks[:, oi], axis=1)
        cb_sum = np.sum(cb_entries[:, oi], axis=1)

        safe = np.abs(cb_sum) > 1e-8
        gain_maps[safe, gi] = actual_sum[safe] / cb_sum[safe]
        gain_maps[~safe, gi] = 1.0

    # Clamp extreme gains (indicates bad codebook match — rare with shared codebook)
    gain_maps = np.clip(gain_maps, 0.0, 10.0)

    return indices, gain_maps


# ═══════════════════════════════════════════════════════════════════════════
# CVD6 writer
# ═══════════════════════════════════════════════════════════════════════════

def write_cvdb6(filepath, grids_data):
    """Write a CVD6 .cvdb file.

    grids_data: list of dicts with keys:
        name, K, block_size, voxel_size, bbox_min, bbox_max,
        norm_scale, codebook_fp16, indices, origins, gain_maps
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, 'wb') as f:
        f.write(b'CVD6')
        f.write(struct.pack('<I', len(grids_data)))

        for g in grids_data:
            N = len(g['indices'])
            K = g['K']
            bs = g['block_size']
            vpb = bs ** 3
            index_bytes = 2 if K > 256 else 1

            # Name (64 bytes null-padded)
            name_b = g['name'].encode('utf-8')[:63]
            f.write(name_b + b'\x00' * (64 - len(name_b)))

            f.write(struct.pack('<III', N, K, index_bytes))

            vs = g['voxel_size']
            vs = vs[0] if hasattr(vs, '__len__') else vs
            f.write(struct.pack('<fff', vs, vs, vs))

            for v in g['bbox_min']:
                f.write(struct.pack('<d', float(v)))
            for v in g['bbox_max']:
                f.write(struct.pack('<d', float(v)))

            f.write(struct.pack('<f', g['norm_scale']))

            # Codebook FP16
            f.write(g['codebook_fp16'].tobytes())

            # Indices
            if index_bytes == 1:
                f.write(g['indices'].astype(np.uint8).tobytes())
            else:
                f.write(g['indices'].astype(np.uint16).tobytes())

            # Origins
            f.write(g['origins'].astype(np.int32).tobytes())

            # Gain maps
            f.write(g['gain_maps'].astype(np.float32).tobytes())

    return os.path.getsize(filepath)


# ═══════════════════════════════════════════════════════════════════════════
# Frame path resolution
# ═══════════════════════════════════════════════════════════════════════════

def resolve_path(pattern, frame):
    """Resolve ####, %04d, or auto-detect numeric portion."""
    # #### pattern
    m = re.search(r'(#+)', pattern)
    if m:
        pad = len(m.group(1))
        return pattern[:m.start()] + str(frame).zfill(pad) + pattern[m.end():]

    # %0Nd pattern
    m = re.search(r'(%0?\d*d)', pattern)
    if m:
        return pattern[:m.start()] + (m.group(1) % frame) + pattern[m.end():]

    # Auto-detect last digit group before extension
    m = re.search(r'(\d+)(\.[^.]+)$', pattern)
    if m:
        pad = len(m.group(1))
        return pattern[:m.start(1)] + str(frame).zfill(pad) + m.group(2)

    return pattern


def parse_frames(s):
    """Parse "1-100" or "1,5,10-20" → sorted list."""
    frames = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            frames.extend(range(int(a), int(b) + 1))
        else:
            frames.append(int(part))
    return sorted(set(frames))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Re-encode .cvdb sequences with shared temporal codebook (CVD6)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cvdb_temporal_encoder.py \\
      "/sims/explosion_####.cvdb" \\
      "/out/explosion_####.cvdb" \\
      --frames 1-100

  python cvdb_temporal_encoder.py \\
      "/sims/smoke_%04d.cvdb" \\
      "/out/smoke_%04d.cvdb" \\
      --frames 1-200 --K 8192 --sample-frames 16
        """)

    p.add_argument('input', help='Input .cvdb path pattern (#### or %%04d)')
    p.add_argument('output', help='Output .cvdb path pattern')
    p.add_argument('--frames', required=True, help='Frame range: "1-100"')
    p.add_argument('--K', type=int, default=0,
                   help='Codebook size (default: match input files)')
    p.add_argument('--sample-frames', type=int, default=16,
                   help='Frames to sample for training (default: 16)')
    p.add_argument('--sample-blocks', type=int, default=80000,
                   help='Max blocks per grid for training (default: 80000)')
    p.add_argument('--max-iter', type=int, default=30,
                   help='K-means iterations (default: 30)')
    p.add_argument('--grids', default='',
                   help='Only encode these grids (comma-separated). Empty = all.')

    args = p.parse_args()
    frames = parse_frames(args.frames)

    # ── Verify inputs exist ──
    existing = []
    for fr in frames:
        path = resolve_path(args.input, fr)
        if os.path.exists(path):
            existing.append(fr)
    if not existing:
        print(f"ERROR: No input files found for pattern: {args.input}")
        print(f"  Tried frames {frames[0]}-{frames[-1]}")
        sys.exit(1)

    frames = existing

    print("═══════════════════════════════════════════════")
    print("  CVDB Temporal Encoder — Shared Codebook")
    print("═══════════════════════════════════════════════")
    print(f"  Input:   {args.input}")
    print(f"  Output:  {args.output}")
    print(f"  Frames:  {frames[0]}-{frames[-1]} ({len(frames)} found)")
    print()

    # ── Read first frame to discover grids and settings ──
    first_path = resolve_path(args.input, frames[0])
    ver, first_grids = read_cvdb(first_path)
    print(f"  Source format: {ver}")

    grid_names = [g['name'] for g in first_grids]
    if args.grids:
        grid_names = [n.strip() for n in args.grids.split(',')]

    K_default = first_grids[0]['K']
    K = args.K if args.K > 0 else K_default
    block_size = first_grids[0]['block_size']
    vpb = block_size ** 3

    print(f"  Grids:   {grid_names}")
    print(f"  K={K}, block_size={block_size}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1 — Sample blocks across frames, train shared codebook per grid
    # ══════════════════════════════════════════════════════════════════════

    print("── Phase 1: Training shared codebooks ──")

    n_sample = min(args.sample_frames, len(frames))
    sample_idx = np.linspace(0, len(frames) - 1, n_sample, dtype=int)
    sample_frames = [frames[i] for i in sample_idx]
    print(f"  Sampling {n_sample} frames: {sample_frames[:6]}"
          f"{'...' if n_sample > 6 else ''}")
    print()

    # Collect reconstructed blocks per grid
    grid_training_blocks = {name: [] for name in grid_names}
    max_per_frame = args.sample_blocks // max(n_sample, 1)

    for si, fr in enumerate(sample_frames):
        path = resolve_path(args.input, fr)
        if not os.path.exists(path):
            continue

        _, grids = read_cvdb(path)
        gmap = {g['name']: g for g in grids}

        for name in grid_names:
            if name not in gmap:
                continue
            g = gmap[name]

            # Reconstruct actual block data from codebook + indices + gains
            blocks = reconstruct_blocks(g)

            # Subsample if too many
            if len(blocks) > max_per_frame:
                idx = np.random.RandomState(fr + hash(name) % 10000).choice(
                    len(blocks), max_per_frame, replace=False)
                blocks = blocks[idx]

            grid_training_blocks[name].append(blocks)

        parts = []
        for name in grid_names:
            if name in gmap:
                nl = len(gmap[name].get('indices', []))
                parts.append(f"{name}={nl} leaves")
        print(f"  Frame {fr:4d} ({si+1}/{n_sample}): {', '.join(parts)}")

    # Train shared codebooks
    shared_codebooks = {}
    shared_norm_scales = {}
    shared_codebooks_fp16 = {}

    print()
    for name in grid_names:
        all_blocks = grid_training_blocks[name]
        if not all_blocks:
            print(f"  WARNING: No blocks for '{name}' — skipping")
            continue

        pooled = np.concatenate(all_blocks, axis=0)
        print(f"  Training '{name}': {len(pooled)} pooled blocks → K={K}")

        t0 = time.time()
        cb = kmeans_codebook(pooled, K, max_iter=args.max_iter, verbose=True)
        dt = time.time() - t0
        print(f"    Done in {dt:.1f}s")

        # Compute norm_scale and FP16 codebook
        ns = float(np.max(np.abs(cb)))
        if ns < 1e-10:
            ns = 1.0
        shared_norm_scales[name] = ns

        # Normalize → FP16
        cb_norm = (cb / ns).astype(np.float32)
        cb_fp16 = cb_norm.astype(np.float16).view(np.uint16).ravel()

        shared_codebooks[name] = cb
        shared_codebooks_fp16[name] = cb_fp16

        cb_mb = len(cb_fp16) * 2 / (1024 * 1024)
        print(f"    norm_scale={ns:.4f}, FP16 codebook={cb_mb:.2f} MB")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2 — Re-encode every frame against shared codebooks
    # ══════════════════════════════════════════════════════════════════════

    print("── Phase 2: Encoding frames ──")
    total_saved = 0
    t_start = time.time()

    for fi, fr in enumerate(frames):
        in_path = resolve_path(args.input, fr)
        out_path = resolve_path(args.output, fr)

        if not os.path.exists(in_path):
            continue

        _, grids = read_cvdb(in_path)
        gmap = {g['name']: g for g in grids}

        out_grids = []
        for name in grid_names:
            if name not in gmap or name not in shared_codebooks:
                continue

            g = gmap[name]

            # Reconstruct the original block data
            blocks = reconstruct_blocks(g)

            # Normalize by shared norm_scale for encoding
            ns = shared_norm_scales[name]
            blocks_norm = blocks / ns
            cb_norm = shared_codebooks[name] / ns

            # Encode against shared codebook
            indices, gain_maps = encode_blocks(blocks_norm, cb_norm, block_size)

            out_grids.append({
                'name': name,
                'K': K,
                'block_size': block_size,
                'voxel_size': g['voxel_size'],
                'bbox_min': g['bbox_min'],
                'bbox_max': g['bbox_max'],
                'norm_scale': ns,
                'codebook_fp16': shared_codebooks_fp16[name],
                'indices': indices,
                'origins': g['origins'],
                'gain_maps': gain_maps,
            })

        if out_grids:
            out_size = write_cvdb6(out_path, out_grids)
            in_size = os.path.getsize(in_path)
            total_saved += (in_size - out_size)

        # Progress
        elapsed = time.time() - t_start
        if fi > 0 and (fi % 10 == 0 or fi == len(frames) - 1):
            fps = (fi + 1) / elapsed
            eta = (len(frames) - fi - 1) / fps if fps > 0 else 0
            print(f"  [{fi+1:4d}/{len(frames)}] frame {fr}, "
                  f"{fps:.1f} frames/s, ETA {eta:.0f}s")
        elif fi < 5:
            print(f"  [{fi+1:4d}/{len(frames)}] frame {fr} → {out_path}")

    total_time = time.time() - t_start

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════

    print()
    print("═══════════════════════════════════════════════")
    print(f"  Done! {len(frames)} frames encoded in {total_time:.1f}s")
    print(f"  Size delta: {total_saved/1024/1024:+.1f} MB total")
    print()
    print("  In Nuke, scrubbing this sequence you'll see:")
    print('    "Codebook unchanged (shared temporal) — skip upload"')
    print("  for every frame after the first.")
    print()
    print("  Codebook bytes are IDENTICAL across all frames,")
    print("  so the Phase 3 fingerprint match triggers every time.")
    print("═══════════════════════════════════════════════")


if __name__ == '__main__':
    main()
