// Vnuclear.hpp — nuclear-electron potential V_en in real-SH representation
// on a uniform radial grid.
//
// Exact identity (Laplace expansion of 1/|r-R_i| in real spherical harmonics,
// valid for r != |R_i|):
//
//     1 / |r - R_i|
//       = sum_{l=0..inf} sum_{m=-l..l}  (4 pi / (2l+1))
//                                   * ( r_<^l / r_>^{l+1} )
//                                   * Y^R_{l,m}(hat R_i) * Y^R_{l,m}(hat r)
//
// with r_< = min(r, |R_i|), r_> = max(r, |R_i|).
//
// Therefore
//
//     V_en(r_vec) = -sum_i Z_i / |r_vec - R_i|
//
//     V_en^R_{l,m}(r) = - sum_i Z_i * (4 pi / (2l+1))
//                              * r_<^l / r_>^{l+1}
//                              * Y^R_{l,m}(hat R_i)
//
// which is what we assemble here. No quadrature needed.
//
// IMPORTANT on efficiency (aimed at Lmax ~ 300 for C8F8):
//
//  - We factor the evaluation into: (1) per-atom cache of Y^R_{l,m}(hat R_i)
//    as a Nlm-vector; (2) per-atom, per-r radial factor w_l(r, R_i) =
//    (4 pi / (2l+1)) * r_<^l / r_>^{l+1}, computed by multiplicative recurrence
//    in l (O(Lmax) per (r, atom)); (3) inner (l, m) loop that multiplies.
//
//  - Main hot loop is parallelized over radial points with OpenMP. Threads
//    write to distinct columns of V (column-major storage in Eigen), so no
//    synchronization is needed.
//
//  - Per-atom Y^R_{l,m}(hat R_i) table is Nlm doubles = (Lmax+1)^2. Very
//    small even at Lmax=300 (~700 kB per atom).
//
// The returned matrix V has shape (Nlm, Nr), column-major, where the column
// index is the radial index and the row index is the (l, m) channel.
//
// Origin convention: R_i is the *position of atom i relative to the chosen
// SCE origin* (typically the molecular center). `atoms_xyz` is the absolute
// position; pass `origin` to subtract.

#pragma once

#include "angular/Ylm.hpp"
#include "molden/Molden.hpp"
#include "sce/RadialGrid.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#if defined(PREPROC_HAS_OPENMP) && PREPROC_HAS_OPENMP
#include <omp.h>
#endif

namespace preproc::potential {

// Precompute Y^R_{l,m}(hat R) for a single direction on the unit sphere,
// indexed by angular::lm_index(l,m). Uses the stable no-CS associated
// Legendre recurrence from angular/Legendre.hpp and the m-multiplier recurrence
// cos((m+1)phi) = 2 cos(phi) cos(m phi) - cos((m-1) phi), sin likewise, to
// avoid calling cos/sin (Lmax)^2 times.
//
// Output: vector of length (Lmax+1)^2.
inline std::vector<double> real_Ylm_all(int Lmax, double theta, double phi) {
    const int Nlm = angular::n_channels(Lmax);
    std::vector<double> Y(Nlm, 0.0);

    // Precompute cos(m*phi), sin(m*phi) for m = 0 .. Lmax.
    std::vector<double> cphi(Lmax + 1), sphi(Lmax + 1);
    cphi[0] = 1.0; sphi[0] = 0.0;
    if (Lmax >= 1) { cphi[1] = std::cos(phi); sphi[1] = std::sin(phi); }
    for (int m = 2; m <= Lmax; ++m) {
        cphi[m] = 2.0 * cphi[1] * cphi[m - 1] - cphi[m - 2];
        sphi[m] = 2.0 * cphi[1] * sphi[m - 1] - sphi[m - 2];
    }

    // For each (l, m>=0), S_{l,m} = angular::sph_legendre_noCS(l, m, theta).
    // We compute S column-by-m (fixed m, ascending l) using the same
    // recurrence embedded in angular/Legendre.hpp for a single call. The
    // cheapest here is to call sph_legendre_noCS per (l, m): Lmax^2/2 calls,
    // each O(Lmax) internally -> O(Lmax^3). For Lmax <= 300 that is 2.7e7
    // ops per direction; negligible compared to the main assembly.
    //
    // If profiling shows this is a bottleneck, specialize a dense S_{l,m}
    // recurrence table here.
    for (int m = 0; m <= Lmax; ++m) {
        for (int l = m; l <= Lmax; ++l) {
            const double S = angular::sph_legendre_noCS(l, m, theta);
            if (m == 0) {
                Y[angular::lm_index(l, 0)] = S;
            } else {
                const double c = std::sqrt(2.0);
                Y[angular::lm_index(l,  m)] = c * S * cphi[m];
                Y[angular::lm_index(l, -m)] = c * S * sphi[m];
            }
        }
    }
    return Y;
}

// Build V_en on an (Lmax, r_grid) SCE grid, using the multipole expansion.
//
// Arguments:
//   atoms   : list of atoms (position in Bohr, nuclear charge Z)
//   origin  : SCE origin (atoms are taken relative to this point)
//   r_grid  : uniform radial grid
//   Lmax    : angular cutoff; output has (Lmax+1)^2 channels
//
// Returns Eigen::MatrixXd of shape (Nlm, Nr).
inline Eigen::MatrixXd build_V_en(
    const std::vector<molden::Atom>& atoms,
    const Eigen::Vector3d& origin,
    const sce::RadialGrid& r_grid,
    int Lmax,
    bool verbose = false)
{
    if (Lmax < 0) throw std::runtime_error("build_V_en: Lmax < 0");
    const int Nr  = r_grid.N;
    const int Nlm = angular::n_channels(Lmax);

    // Shifted atom positions (relative to origin) + per-atom cached Y^R.
    const int Na = static_cast<int>(atoms.size());
    std::vector<double> R_mag(Na);
    std::vector<std::vector<double>> Ytab(Na);    // Y^R_{l,m}(hat R_i) per atom
    std::vector<int> Z_col(Na);

    for (int i = 0; i < Na; ++i) {
        Eigen::Vector3d P = atoms[i].xyz - origin;
        R_mag[i] = P.norm();
        Z_col[i] = atoms[i].Z;
        double theta_i = 0.0, phi_i = 0.0;
        if (R_mag[i] > 1e-14) {
            theta_i = std::acos(std::clamp(P.z() / R_mag[i], -1.0, 1.0));
            phi_i   = std::atan2(P.y(), P.x());
        }
        Ytab[i] = real_Ylm_all(Lmax, theta_i, phi_i);
        if (verbose) {
            std::cerr << "[Ven] atom " << i << "  Z=" << Z_col[i]
                      << "  |R|=" << R_mag[i] << "  theta=" << theta_i
                      << "  phi=" << phi_i << "\n";
        }
    }

    // Precompute  pref_l = 4*pi / (2l+1)  for all l.
    std::vector<double> pref_l(Lmax + 1);
    for (int l = 0; l <= Lmax; ++l) pref_l[l] = 4.0 * M_PI / (2.0 * l + 1.0);

    Eigen::MatrixXd V(Nlm, Nr);
    V.setZero();

    // Parallel over radial points: each thread writes to a distinct column.
    // OpenMP safety: `Ytab`, `pref_l`, `R_mag`, `Z_col`, `V` are shared and
    // read/written in disjoint regions. No reductions, no critical sections.
    #if defined(PREPROC_HAS_OPENMP) && PREPROC_HAS_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int k = 0; k < Nr; ++k) {
        const double r = r_grid.r[k];
        // Thread-local radial weight buffer w_l(r; R_i), reused across atoms.
        std::vector<double> wl(Lmax + 1);
        for (int i = 0; i < Na; ++i) {
            const double Ri = R_mag[i];
            double r_less, r_greater;
            if (r <= Ri) { r_less = r;  r_greater = Ri; }
            else         { r_less = Ri; r_greater = r;  }
            // Handle r == r_greater == 0: V_en diverges at nucleus at the
            // origin; set zero at that grid point (will not be integrated
            // against a smooth rho at the same point because rho(0) is finite).
            if (r_greater < 1e-300) {
                continue;
            }
            // Stable recurrence in l for (r_<^l / r_>^{l+1}):
            //   w_0 = 1 / r_>
            //   w_{l+1} = w_l * (r_< / r_>)
            const double ratio = r_less / r_greater;
            double w = 1.0 / r_greater;
            for (int l = 0; l <= Lmax; ++l) {
                wl[l] = w * pref_l[l];
                w *= ratio;
            }
            const double mZ = -static_cast<double>(Z_col[i]);
            const std::vector<double>& Yi = Ytab[i];
            // Accumulate  V(lm, k) += mZ * wl[l] * Yi[lm]
            // Contiguous in (l, m): we walk channels in lm_index order.
            double* __restrict__ col = V.data() + static_cast<size_t>(k) * Nlm;
            for (int l = 0; l <= Lmax; ++l) {
                const double alpha = mZ * wl[l];
                const int base = l * l + l;   // lm_index(l, 0)
                for (int m = -l; m <= l; ++m) {
                    col[base + m] += alpha * Yi[base + m];
                }
            }
        }
    }

    return V;
}

// Energy <rho|V> = int (sum_{lm} rho_{lm}(r) V_{lm}(r)) r^2 dr,
// using orthonormality of Y^R.
//
// rho_lm and V_lm must be expressed on the same radial grid and with the
// same (and matching) (l,m) channel layout.
inline double inner_product_radial(const Eigen::MatrixXd& Fa,
                                   const Eigen::MatrixXd& Fb,
                                   const sce::RadialGrid& r_grid) {
    if (Fa.rows() != Fb.rows() || Fa.cols() != Fb.cols())
        throw std::runtime_error("inner_product_radial: shape mismatch");
    const int Nlm = static_cast<int>(Fa.rows());
    const int Nr  = r_grid.N;
    std::vector<double> integrand(Nr, 0.0);
    #if defined(PREPROC_HAS_OPENMP) && PREPROC_HAS_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int k = 0; k < Nr; ++k) {
        double s = 0.0;
        const double* __restrict__ A = Fa.data() + static_cast<size_t>(k) * Nlm;
        const double* __restrict__ B = Fb.data() + static_cast<size_t>(k) * Nlm;
        for (int c = 0; c < Nlm; ++c) s += A[c] * B[c];
        integrand[k] = r_grid.r[k] * r_grid.r[k] * s;
    }
    return r_grid.integrate(integrand);
}

}  // namespace preproc::potential
