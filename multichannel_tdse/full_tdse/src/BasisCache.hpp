// BasisCache.hpp -- on-disk caching of a built PooledBasis.
//
// The basis-build phase (Numerov scan + bisection + reconstruction +
// dipole assembly) takes hours at production size and depends ONLY on
// time-INDEPENDENT inputs:
//
//   * B_gauss, V_T, V_S, r_0, μ           (model)
//   * L, dr, N_grid, p_init, N_ch_keep    (numerical grid / truncation)
//   * E_cut per block, use_analytic_halo  (block options)
//   * block_MFs                           (which blocks)
//
// Pulse parameters (Ω_R, ω, τ, χ shape, t_start, t_end, dt) are NOT
// part of the cache key -- the same cached basis is reused across an
// arbitrary number of pulse-parameter sweeps.
//
// File layout under cache_dir:
//
//     {cache_dir}/{key}.bin            binary blob: eigenstates + d_plus
//     {cache_dir}/{key}.meta.txt       human-readable parameters
//
// `key` is a 16-hex-char FNV-1a hash of the canonical parameter string.
// A cache HIT requires the binary to load AND the meta.txt parameters
// to match the request exactly (defense against hash collisions).
#pragma once

#include "BlockEigenstates.hpp"
#include "PooledBasis.hpp"

#include <string>
#include <vector>

namespace mc_tdse {

// Canonical parameter string.  Stable across runs; identical inputs ⇒
// identical string.  Hash this with FNV-1a to get the cache key.
std::string basis_cache_canonical_string(const std::vector<int>& block_MFs,
                                         const std::vector<BlockBuildOptions>& opts,
                                         double B_gauss);

// 16-hex-char FNV-1a hash of `basis_cache_canonical_string(...)`.
std::string basis_cache_key(const std::vector<int>& block_MFs,
                             const std::vector<BlockBuildOptions>& opts,
                             double B_gauss);

// Try to load a cached PooledBasis.  Returns true if found AND meta
// matches; populates *out.  Returns false on cache miss or any
// integrity failure.  Never throws.
bool try_load_pooled_basis(PooledBasis* out,
                           const std::string& cache_dir,
                           const std::string& key,
                           const std::string& canonical_string);

// Save a built PooledBasis to disk under `cache_dir/{key}.bin` plus
// `.meta.txt`.  Creates `cache_dir` if missing.  Throws on I/O error.
void save_pooled_basis(const PooledBasis& pb,
                       const std::string& cache_dir,
                       const std::string& key,
                       const std::string& canonical_string);

}  // namespace mc_tdse
