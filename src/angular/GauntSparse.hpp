// GauntSparse.hpp -- sparse real-Gaunt matrix for fast potential-matrix assembly.
//
// Port of version_0/src/potential_sparse_matrix.hpp. Keeping the math and
// storage layout EXACTLY as in version_0 so results are bit-identical. The
// only change is removing the `sparse_types.hpp` indirection by inlining
// the 64-bit index typedefs here (they're needed because at Lmax=100 the
// upper-triangle pair count n_pairs ~ 5.2e7 overflows 32-bit signed indices).
//
// Physical picture:
//   V_{mu',mu}(r) = sum_sigma V_sigma(r) * G^R(mu', sigma, mu)
// We precompute G^R as a sparse matrix
//   G_sparse(row, col) = G^R(mu', sigma, mu)
// where row = upper-triangle pair index (mu', mu) with mu >= mu', and
//      col = sigma  in [0, n_exp).
// For each radial point the V(r) matrix is then one sparse matvec:
//   V_upper_triangle(r) = G_sparse * V_sigma(r).
// We unpack to full symmetric matrix at the end.
//
// At Lmax=100 (channels = 10201) and l_exp_max = 200 (n_exp ~ 40401), the
// selection rules give sparsity ~5% of n_pairs * n_exp -- without this
// trick the dense contraction would take ~(Lmax+1)^4 memory references
// per radial point, which is unworkable.

#pragma once

#include "angular/Gaunt.hpp"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
#include <omp.h>
#endif

namespace scatt::angular {

// 64-bit index type for the sparse matrix. Needed because at L=100 the
// (channels * (channels+1))/2 upper-triangle pair count is ~ 5.2e7 which
// fits in int32 but can overflow in intermediate products. Using int64
// throughout kills this class of bug.
using SparseIndex  = std::int64_t;
using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, SparseIndex>;
using SparseTriplet = Eigen::Triplet<double, SparseIndex>;

class GauntSparseMatrix {
public:
    int          channels_   = 0;    // number of scattering channels = (l_max+1)^2
    int          l_max_      = 0;    // continuum angular cutoff
    int          l_exp_max_  = 0;    // expansion angular cutoff (usually 2*l_max_)
    int          n_exp_      = 0;    // (l_exp_max_+1)^2
    SparseIndex  n_pairs_    = 0;    // channels_ * (channels_+1) / 2

    SparseMatrix G_sparse_;
    std::vector<std::pair<int, int>> pair_to_mu_;  // inverse of pair_index

    GauntSparseMatrix() = default;

    // Upper-triangle row-major pair index: requires mu >= mu_p.
    inline SparseIndex pair_index(int mu_p, int mu) const {
        return static_cast<SparseIndex>(mu_p) * channels_
             - (static_cast<SparseIndex>(mu_p) * (mu_p - 1)) / 2
             + (mu - mu_p);
    }

    // Build the sparse matrix.
    //   channels     = (l_max + 1)^2
    //   l_exp_max    = angular cutoff of V_sigma (tipically 2 * l_max)
    //   V_lm_max_per_sigma : maximum |V_sigma(r)| across r for every sigma,
    //       so we can skip sigmas that are identically zero (e.g. the
    //       spherically-symmetric expansion of V_ee for highly symmetric
    //       molecules has very few non-zero sigmas). Pass an empty vector
    //       to include ALL sigmas up to n_exp.
    //   threshold    = sigmas with |V|_max < threshold are skipped.
    void build(int channels, int l_exp_max,
               const std::vector<double>& V_sigma_max_over_r,
               double threshold = 1e-20,
               bool verbose = true)
    {
        channels_  = channels;
        l_max_     = static_cast<int>(std::sqrt(static_cast<double>(channels))) - 1;
        l_exp_max_ = l_exp_max;
        n_exp_     = (l_exp_max + 1) * (l_exp_max + 1);
        n_pairs_   = static_cast<SparseIndex>(channels) * (channels + 1) / 2;

        if (verbose) {
            std::cout << "[GauntSparse] channels=" << channels_
                      << " l_max=" << l_max_
                      << " l_exp_max=" << l_exp_max_
                      << " n_exp=" << n_exp_
                      << " n_pairs(upper triangle)=" << n_pairs_ << "\n";
        }

        // pair_to_mu inverse table.
        pair_to_mu_.resize(static_cast<std::size_t>(n_pairs_));
        for (int mu_p = 0; mu_p < channels_; ++mu_p) {
            for (int mu = mu_p; mu < channels_; ++mu) {
                pair_to_mu_[static_cast<std::size_t>(pair_index(mu_p, mu))] = {mu_p, mu};
            }
        }

        // Significant-sigma filter. If caller passed empty or all-zeros,
        // include every sigma (pessimistic but always correct).
        std::vector<int> significant_sigma;
        significant_sigma.reserve(n_exp_);
        if (V_sigma_max_over_r.empty()) {
            for (int s = 0; s < n_exp_; ++s) significant_sigma.push_back(s);
        } else {
            const int nmax = std::min<int>(n_exp_, static_cast<int>(V_sigma_max_over_r.size()));
            for (int s = 0; s < nmax; ++s) {
                if (V_sigma_max_over_r[s] > threshold) significant_sigma.push_back(s);
            }
        }
        if (verbose) {
            std::cout << "[GauntSparse] significant sigmas: "
                      << significant_sigma.size() << " / " << n_exp_
                      << " (" << (100.0 * significant_sigma.size() / n_exp_) << "%)\n";
        }

        // Compute Gaunt triplets, parallel over sigma.
#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        const int nthreads = omp_get_max_threads();
#else
        const int nthreads = 1;
#endif
        std::vector<std::vector<SparseTriplet>> thread_triplets(static_cast<std::size_t>(nthreads));

#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        #pragma omp parallel
#endif
        {
#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            auto& local = thread_triplets[static_cast<std::size_t>(tid)];

#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
            #pragma omp for schedule(dynamic, 1)
#endif
            for (std::size_t i = 0; i < significant_sigma.size(); ++i) {
                const int sigma = significant_sigma[i];
                int l_s, m_s; idx_to_lm(sigma, l_s, m_s);

                // For each (l2, m2) = mu_p (row), enumerate allowed (l1, m1) = mu
                // via the triangle + parity selection rules. Keep only mu >= mu_p
                // (upper triangle).
                for (int l2 = 0; l2 <= l_max_; ++l2) {
                    const int l1_min = std::abs(l2 - l_s);
                    const int l1_max = std::min(l_max_, l2 + l_s);
                    for (int m2 = -l2; m2 <= l2; ++m2) {
                        const int mu_p = lm_to_idx(l2, m2);
                        for (int l1 = l1_min; l1 <= l1_max; ++l1) {
                            if (((l1 + l_s + l2) & 1) != 0) continue;
                            for (int m1 = -l1; m1 <= l1; ++m1) {
                                const int mu = lm_to_idx(l1, m1);
                                if (mu < mu_p) continue;           // upper triangle
                                const double G = gaunt_real(l2, m2, l_s, m_s, l1, m1);
                                if (std::abs(G) > 1e-15) {
                                    local.emplace_back(pair_index(mu_p, mu), sigma, G);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Merge thread-local triplets.
        std::size_t total = 0;
        for (const auto& t : thread_triplets) total += t.size();
        std::vector<SparseTriplet> triplets;
        triplets.reserve(total);
        for (auto& t : thread_triplets) triplets.insert(triplets.end(), t.begin(), t.end());

        if (verbose) {
            std::cout << "[GauntSparse] non-zero Gaunt triplets: " << triplets.size()
                      << "  (sparsity " << (100.0 * triplets.size()
                             / (static_cast<double>(n_pairs_) * n_exp_)) << "%)\n";
        }

        G_sparse_.resize(n_pairs_, n_exp_);
        G_sparse_.setFromTriplets(triplets.begin(), triplets.end());
        G_sparse_.makeCompressed();

        if (verbose) {
            const std::size_t nnz = static_cast<std::size_t>(G_sparse_.nonZeros());
            const std::size_t bytes =
                nnz * (sizeof(double) + sizeof(SparseIndex))
              + static_cast<std::size_t>(n_pairs_ + 1) * sizeof(SparseIndex);
            std::cout << "[GauntSparse] memory: " << (bytes / (1024 * 1024)) << " MB\n";
        }
    }

    // Getters for introspection (used by the scaling benchmark).
    std::size_t nonzeros() const {
        return static_cast<std::size_t>(G_sparse_.nonZeros());
    }
    std::size_t memory_bytes() const {
        return nonzeros() * (sizeof(double) + sizeof(SparseIndex))
             + static_cast<std::size_t>(n_pairs_ + 1) * sizeof(SparseIndex);
    }

    // Assemble V(r) = G_sparse * V_sigma. Returns dense symmetric (channels, channels).
    Eigen::MatrixXd compute_V(const Eigen::VectorXd& V_sigma) const {
        Eigen::VectorXd V_flat = G_sparse_ * V_sigma;
        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(channels_, channels_);
        for (SparseIndex idx = 0; idx < n_pairs_; ++idx) {
            const auto& pr = pair_to_mu_[static_cast<std::size_t>(idx)];
            const double val = V_flat(idx);
            V(pr.first, pr.second) = val;
            if (pr.first != pr.second) V(pr.second, pr.first) = val;
        }
        return V;
    }

    // Same but add to an existing matrix (no zeroing). Useful when the
    // caller adds centrifugal + polarization + V_en_ee separately.
    void add_V_into(const Eigen::VectorXd& V_sigma, Eigen::MatrixXd& out) const {
        Eigen::VectorXd V_flat = G_sparse_ * V_sigma;
        for (SparseIndex idx = 0; idx < n_pairs_; ++idx) {
            const auto& pr = pair_to_mu_[static_cast<std::size_t>(idx)];
            const double val = V_flat(idx);
            out(pr.first, pr.second) += val;
            if (pr.first != pr.second) out(pr.second, pr.first) += val;
        }
    }
};

}  // namespace scatt::angular
