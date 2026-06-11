// SCE.hpp — Single-Center Expansion engine.
//
// Projects a scalar 3D function F(r_vec) onto real spherical harmonics:
//
//     F^R_{l,m}(r) = int_S F(origin + r * n_hat) Y^R_{l,m}(n_hat) dOmega
//
// using Gauss-Legendre in cos(theta) and uniform trapezoidal in phi with
//     nTheta = Lmax + 1           (exact P(cos theta) up to degree 2 Lmax + 1)
//     nPhi   = 2 (Lmax + 1)       (exact Fourier up to |m| <= Lmax)
//
// The Y^R factorizes into S_{l,m}(theta) * {cos, sin}(m phi):
//     m = 0:  Y^R_{l,0}   = S_{l,0}(theta)
//     m > 0:  Y^R_{l,+m}  = sqrt(2) * S_{l,m}(theta) * cos(m phi)
//     m < 0:  Y^R_{l,-|m|}= sqrt(2) * S_{l,|m|}(theta) * sin(|m| phi)
// where S = angular::sph_legendre_noCS (no Condon-Shortley, normalized).
//
// Factored projection:
//     F^R_{l,m=0}(r) =    (2 pi / nPhi) * sum_i w_i * S_{l,0}(theta_i) * A_0(theta_i)
//     F^R_{l,m>0}(r) = sqrt(2) * (2 pi / nPhi) * sum_i w_i * S_{l,m}(theta_i) * A_m(theta_i)
//     F^R_{l,m<0}(r) = sqrt(2) * (2 pi / nPhi) * sum_i w_i * S_{l,|m|}(theta_i) * B_m(theta_i)
// where for real F:
//     A_m(theta_i) =  sum_j F(theta_i, phi_j) cos(m phi_j) =  Re{ DFT_nPhi F(theta_i, .) }_m
//     B_m(theta_i) =  sum_j F(theta_i, phi_j) sin(m phi_j) = -Im{ DFT_nPhi F(theta_i, .) }_m
//
// Two backends, selected at compile time by PREPROC_HAS_FFTW:
//
//   1. FFTW backend (preferred):   each theta row goes through a length-nPhi
//      real-to-complex FFT (fftw_plan_dft_r2c_1d). For Lmax=300 this is ~300x
//      faster than the direct DFT.
//
//   2. Direct-DFT backend (fallback): O(nPhi * (Lmax+1)) per theta. Works
//      when FFTW is not available. Used by default at low Lmax where its
//      simplicity is a feature.
//
// Parallelism: OpenMP over radial points. Each thread has thread-local
// scratch buffers; FFTW plans are created per thread under a critical
// section since fftw_plan_* is NOT thread-safe (plan *execution* is).

#pragma once

#include "angular/Grid.hpp"
#include "angular/Legendre.hpp"
#include "angular/Ylm.hpp"
#include "sce/RadialGrid.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <vector>

#if defined(PREPROC_HAS_OPENMP) && PREPROC_HAS_OPENMP
#include <omp.h>
#endif

#if defined(PREPROC_HAS_FFTW) && PREPROC_HAS_FFTW
#include <fftw3.h>
#endif

namespace preproc::sce {

using F3D = std::function<double(const Eigen::Vector3d&)>;

// Precomputed S_{l, |m|}(theta_i) table. Indexed as
//   S[ (m * (Lmax+1) + l) * nTheta + i ]
// for 0 <= m <= Lmax and m <= l <= Lmax.
struct STable {
    int Lmax;
    int nTheta;
    std::vector<double> S;

    static STable build(int Lmax, const angular::AngGrid& a_grid) {
        STable t;
        t.Lmax = Lmax;
        t.nTheta = a_grid.nTheta;
        const int Lp1 = Lmax + 1;
        t.S.assign(static_cast<size_t>(Lp1) * Lp1 * t.nTheta, 0.0);
        for (int m = 0; m <= Lmax; ++m) {
            for (int l = m; l <= Lmax; ++l) {
                for (int i = 0; i < t.nTheta; ++i) {
                    t.S[(static_cast<size_t>(m) * Lp1 + l) * t.nTheta + i]
                        = angular::sph_legendre_noCS(l, m, a_grid.gl.theta[i]);
                }
            }
        }
        return t;
    }
    inline double operator()(int l, int m_abs, int i_theta) const {
        return S[(static_cast<size_t>(m_abs) * (Lmax + 1) + l) * nTheta + i_theta];
    }
};


#if defined(PREPROC_HAS_FFTW) && PREPROC_HAS_FFTW
// RAII wrapper for an FFTW r2c plan of length nPhi.
struct FftwPlanR2C {
    int nP;
    std::vector<double>              in;
    std::vector<std::complex<double>> out;
    fftw_plan plan;

    FftwPlanR2C(int nPhi) : nP(nPhi), in(nPhi), out(nPhi / 2 + 1) {
        // fftw_plan_* is NOT thread-safe; callers must serialize creation.
        // ESTIMATE (not MEASURE) since we create one plan per OpenMP thread
        // and MEASURE's plan-search overhead dominates small problem sizes.
        plan = fftw_plan_dft_r2c_1d(
            nP, in.data(),
            reinterpret_cast<fftw_complex*>(out.data()),
            FFTW_ESTIMATE
        );
    }
    ~FftwPlanR2C() { if (plan) fftw_destroy_plan(plan); }
    FftwPlanR2C(const FftwPlanR2C&)            = delete;
    FftwPlanR2C& operator=(const FftwPlanR2C&) = delete;
};
#endif


// Project F onto real spherical harmonics on (r_grid, a_grid).
// Returns Nlm x Nr matrix F^R_{l,m}(r_kr).
inline Eigen::MatrixXd project(const RadialGrid& r_grid,
                               const angular::AngGrid& a_grid,
                               const Eigen::Vector3d& origin,
                               const F3D& F,
                               bool verbose = false) {
    const int Nr  = r_grid.N;
    const int L   = a_grid.Lmax;
    const int Lp1 = L + 1;
    const int Nlm = angular::n_channels(L);
    const int nT  = a_grid.nTheta;
    const int nP  = a_grid.nPhi;

    if (verbose) {
        std::cerr << "[sce] Nr=" << Nr << " Nlm=" << Nlm
                  << " nTheta=" << nT << " nPhi=" << nP
#if defined(PREPROC_HAS_FFTW) && PREPROC_HAS_FFTW
                  << " (FFTW backend)\n";
#else
                  << " (direct DFT backend)\n";
#endif
    }

    auto S_tab = STable::build(L, a_grid);

    const double dphi = 2.0 * M_PI / nP;
    const double sqrt2 = std::sqrt(2.0);

    Eigen::MatrixXd Flm(Nlm, Nr);
    Flm.setZero();

#if !(defined(PREPROC_HAS_FFTW) && PREPROC_HAS_FFTW)
    // --- Direct-DFT backend: precompute cos/sin tables, O(nPhi * Lp1) per theta. ---
    std::vector<double> cosMP(static_cast<size_t>(Lp1) * nP);
    std::vector<double> sinMP(static_cast<size_t>(Lp1) * nP);
    for (int j = 0; j < nP; ++j) {
        const double p = a_grid.phi[j];
        cosMP[0 * nP + j] = 1.0; sinMP[0 * nP + j] = 0.0;
        if (Lp1 > 1) { cosMP[1 * nP + j] = std::cos(p); sinMP[1 * nP + j] = std::sin(p); }
        for (int m = 2; m <= L; ++m) {
            cosMP[static_cast<size_t>(m) * nP + j] =
                2.0 * cosMP[1 * nP + j] * cosMP[static_cast<size_t>(m - 1) * nP + j]
              - cosMP[static_cast<size_t>(m - 2) * nP + j];
            sinMP[static_cast<size_t>(m) * nP + j] =
                2.0 * cosMP[1 * nP + j] * sinMP[static_cast<size_t>(m - 1) * nP + j]
              - sinMP[static_cast<size_t>(m - 2) * nP + j];
        }
    }
#endif

#if defined(PREPROC_HAS_OPENMP) && PREPROC_HAS_OPENMP
    #pragma omp parallel
#endif
    {
        std::vector<double> Fval(static_cast<size_t>(nT) * nP);
        std::vector<double> Am(static_cast<size_t>(Lp1) * nT);
        std::vector<double> Bm(static_cast<size_t>(Lp1) * nT);

#if defined(PREPROC_HAS_FFTW) && PREPROC_HAS_FFTW
        // One FFTW plan per OpenMP thread. fftw_plan_* is NOT thread-safe.
        FftwPlanR2C* plan_rc = nullptr;
        #pragma omp critical(preproc_fftw_plan)
        { plan_rc = new FftwPlanR2C(nP); }
#endif

#if defined(PREPROC_HAS_OPENMP) && PREPROC_HAS_OPENMP
        #pragma omp for schedule(static)
#endif
        for (int kr = 0; kr < Nr; ++kr) {
            const double r = r_grid.r[kr];
            // 1) Sample F on the sphere of radius r.
            for (int i = 0; i < nT; ++i) {
                for (int j = 0; j < nP; ++j) {
                    const Eigen::Vector3d n = a_grid.dir(i, j);
                    Fval[static_cast<size_t>(i) * nP + j] = F(origin + r * n);
                }
            }

            // 2) Phi projection: extract A_m, B_m for m=0..Lmax.
#if defined(PREPROC_HAS_FFTW) && PREPROC_HAS_FFTW
            // FFTW backend: one real-to-complex FFT per theta row.
            // Convention: FFTW r2c computes X_m = sum_j x_j exp(-2 pi i m j / N).
            // With x_j = F(phi_j), phi_j = 2 pi j / N:
            //   X_m = sum_j F(phi_j) [cos(m phi_j) - i sin(m phi_j)]
            // so A_m = Re(X_m), B_m = -Im(X_m).
            for (int i = 0; i < nT; ++i) {
                std::copy(Fval.begin() + static_cast<std::ptrdiff_t>(i) * nP,
                          Fval.begin() + static_cast<std::ptrdiff_t>(i + 1) * nP,
                          plan_rc->in.begin());
                fftw_execute(plan_rc->plan);
                for (int m = 0; m <= L; ++m) {
                    Am[static_cast<size_t>(m) * nT + i] =  plan_rc->out[m].real();
                    Bm[static_cast<size_t>(m) * nT + i] = -plan_rc->out[m].imag();
                }
            }
#else
            // Direct-DFT backend.
            for (int i = 0; i < nT; ++i) {
                const double* row = Fval.data() + static_cast<size_t>(i) * nP;
                for (int m = 0; m <= L; ++m) {
                    const double* cM = cosMP.data() + static_cast<size_t>(m) * nP;
                    const double* sM = sinMP.data() + static_cast<size_t>(m) * nP;
                    double a = 0.0, b = 0.0;
                    for (int j = 0; j < nP; ++j) {
                        const double f = row[j];
                        a += f * cM[j];
                        b += f * sM[j];
                    }
                    Am[static_cast<size_t>(m) * nT + i] = a;
                    Bm[static_cast<size_t>(m) * nT + i] = b;
                }
            }
#endif

            // 3) Theta integration for each (l, m).
            double* outcol = Flm.data() + static_cast<size_t>(kr) * Nlm;
            for (int l = 0; l <= L; ++l) {
                double acc = 0.0;
                for (int i = 0; i < nT; ++i) {
                    acc += a_grid.gl.w[i] * S_tab(l, 0, i) * Am[0 * nT + i];
                }
                outcol[l * l + l] = dphi * acc;

                for (int m = 1; m <= l; ++m) {
                    double acc_a = 0.0, acc_b = 0.0;
                    for (int i = 0; i < nT; ++i) {
                        const double wsi = a_grid.gl.w[i] * S_tab(l, m, i);
                        acc_a += wsi * Am[static_cast<size_t>(m) * nT + i];
                        acc_b += wsi * Bm[static_cast<size_t>(m) * nT + i];
                    }
                    outcol[l * l + l + m] = sqrt2 * dphi * acc_a;
                    outcol[l * l + l - m] = sqrt2 * dphi * acc_b;
                }
            }
        }

#if defined(PREPROC_HAS_FFTW) && PREPROC_HAS_FFTW
        #pragma omp critical(preproc_fftw_plan)
        { delete plan_rc; }
#endif
    }

    return Flm;
}

// ---- convenience checks -----------------------------------------------------

inline double norm_squared(const Eigen::MatrixXd& Flm, const RadialGrid& rg) {
    std::vector<double> integrand(rg.N, 0.0);
    for (int k = 0; k < rg.N; ++k) {
        double s = 0.0;
        for (int ch = 0; ch < Flm.rows(); ++ch) s += Flm(ch, k) * Flm(ch, k);
        integrand[k] = rg.r[k] * rg.r[k] * s;
    }
    return rg.integrate(integrand);
}

inline double integrate_total(const Eigen::MatrixXd& Flm, const RadialGrid& rg) {
    const int ch_00 = angular::lm_index(0, 0);
    std::vector<double> integrand(rg.N, 0.0);
    for (int k = 0; k < rg.N; ++k) {
        integrand[k] = rg.r[k] * rg.r[k] * Flm(ch_00, k);
    }
    return std::sqrt(4.0 * M_PI) * rg.integrate(integrand);
}

}  // namespace preproc::sce
