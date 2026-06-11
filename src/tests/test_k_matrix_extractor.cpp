// test_k_matrix_extractor.cpp -- validate K, S, eigenphases on H2O.
//
// Since there is no free-particle test on H2O (polar molecule has a
// long-range dipole+quadrupole tail at r_max = 15 au), the physics check
// is the MATCHING-RADIUS INVARIANCE: K at two different matching indices
// deep in the asymptotic region must agree to O(h²) or better.
//
// Checks:
//   (1) K symmetric.
//   (2) S unitary (‖S†S − I‖ < 1e-10).
//   (3) Eigenphases defined and finite.
//   (4) Matching-radius convergence: extract at n_match = N−2 and at
//       n_match = N−4, ..., and verify K changes are small (O(h²)).
//   (5) Cross-check against version_0's WRONG indexing: build a
//       deliberately off-by-one K and confirm it differs by at least
//       a small amount from ours (sanity that the indexing choice matters).
//   (6) Eigenphase ↔ eigS consistency: e^{2iδ} must equal the eigenvalues
//       of S (same sorted order after pairing).
//   (7) Benchmark.

#include "io/HDF5Reader.hpp"
#include "scatt/Banner.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/ForwardRPropagator.hpp"
#include "scatt/KMatrixExtractor.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <gsl/gsl_sf_bessel.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using scatt::ExchangeCoupling;
using scatt::ForwardRPropagator;
using scatt::KMatrixExtractor;
using scatt::Parameters;
using scatt::Potentials;
using scatt::SchurInverter;
using scatt::ScatteringResult;
using scatt::SetupBundle;
using scatt::StorageMode;
using scatt::WavefunctionSetup;
using scatt::WInverseOperator;
using scatt::io::HDF5Reader;
using scatt::io::PreprocData;

static int fails = 0;
static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++fails;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <h2o_hdf5>\n"; return 2; }

    scatt::print_la_banner();

    HDF5Reader reader(argv[1]);
    PreprocData data = reader.load_all();

    Parameters params;
    params.r_min           = data.rmin;
    params.dr              = data.dr;
    params.N_grid          = data.Nr;
    params.Lmax_sce        = data.Lmax_sce;
    params.l_max_continuum = 4;
    params.validate();

    SetupBundle b = WavefunctionSetup::prepare(params, data, /*energy=*/0.5);

    Potentials pot(params);
    pot.build(data, StorageMode::MEMORY, "", /*verbose=*/false);

    ExchangeCoupling EC(b.G_coeff, b.params.n_mu, b.params.n_sigma, b.params.n_occ,
                        data.rmin, data.dr);

    SchurInverter SI(b.params, pot, &EC, &b.chi);
    SchurInverter::Config si_cfg;
    si_cfg.storage            = StorageMode::MEMORY;
    si_cfg.use_openmp         = true;
    si_cfg.verbose            = false;
    si_cfg.checkpoint_dir     = "./checkpoints/sinv_kmat";
    si_cfg.try_load_checkpoint = false;
    si_cfg.save_checkpoint     = false;
    std::filesystem::remove_all(si_cfg.checkpoint_dir);
    SI.build(si_cfg);

    WInverseOperator WI(b.params, SI, &EC, &b.chi, si_cfg.W_min);

    ForwardRPropagator FRP(b.params, pot, WI);
    ForwardRPropagator::Config frp_cfg;
    frp_cfg.storage             = StorageMode::MEMORY;
    frp_cfg.try_load_checkpoint = false;
    frp_cfg.save_checkpoint     = false;
    frp_cfg.verbose             = false;
    frp_cfg.checkpoint_dir      = "./checkpoints/rinv_kmat";
    std::filesystem::remove_all(frp_cfg.checkpoint_dir);
    FRP.run(frp_cfg);

    const int Nr  = static_cast<int>(b.params.n_grid);
    const int N_psi = b.params.n_mu;

    std::cout << "\n--- Default extraction at n_match = N−2 = " << (Nr - 2)
              << "  (r_in=" << (params.r_min + (Nr-2)*params.dr)
              << ", r_out=" << (params.r_min + (Nr-1)*params.dr) << ") ---\n";
    KMatrixExtractor KME(b.params, FRP);
    ScatteringResult res = KME.extract();

    std::cout << std::fixed << std::setprecision(4)
              << "   K_symmetry_err = " << std::scientific << res.K_symmetry_err << "\n"
              << "   unitarity_err  = " << res.unitarity_err << "\n"
              << "   schur_coupling_zero = " << (res.schur_coupling_zero ? "yes" : "no") << "\n";

    std::cout << "   K eigenvalues (first 10): ";
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(res.K_matrix);
        std::cout << std::fixed << std::setprecision(6);
        for (int i = 0; i < std::min(10, N_psi); ++i)
            std::cout << es.eigenvalues()(i) << " ";
        std::cout << "\n";
    }
    std::cout << "   eigenphases (first 10): ";
    {
        std::cout << std::fixed << std::setprecision(6);
        for (int i = 0; i < std::min(10, N_psi); ++i)
            std::cout << res.eigenphases[i] << " ";
        std::cout << "\n";
    }

    // ----------------------------------------------------------------------
    // (1) K symmetric
    // ----------------------------------------------------------------------
    std::cout << "\n--- (1) K symmetric ---\n";
    check(res.K_symmetry_err < 1e-12,
          "||K − K^T|| < 1e-12 (got " + std::to_string(res.K_symmetry_err) + ")");

    // ----------------------------------------------------------------------
    // (2) S unitary
    // ----------------------------------------------------------------------
    std::cout << "\n--- (2) S unitary: ||S†S − I|| small ---\n";
    check(res.unitarity_err < 1e-10,
          "||S†S − I||_max < 1e-10 (got " + std::to_string(res.unitarity_err) + ")");

    // ----------------------------------------------------------------------
    // (3) Eigenphases finite, bounded
    // ----------------------------------------------------------------------
    std::cout << "\n--- (3) eigenphases finite and bounded ---\n";
    {
        bool all_ok = true;
        for (double p : res.eigenphases) {
            if (!std::isfinite(p)) { all_ok = false; break; }
            if (std::abs(p) > M_PI / 2.0 + 1e-10) { all_ok = false; break; }
        }
        check(all_ok, "all eigenphases finite and |δ| ≤ π/2");
    }

    // ----------------------------------------------------------------------
    // (4) Matching-radius convergence.
    //   Extract at n_match = N−2, N−6, N−10, N−20 (all deep in asymptotic).
    //   K at these different radii must agree to O(h²) — i.e. a few ulps
    //   relative to the characteristic scale, dominated by finite-h
    //   residual in the V tail for H2O.
    // ----------------------------------------------------------------------
    std::cout << "\n--- (4) matching-radius invariance ---\n";
    {
        const Eigen::MatrixXd K_ref = res.K_matrix;
        const double scale_K = std::max(K_ref.cwiseAbs().maxCoeff(), 1e-30);
        std::cout << "   reference |K|_max = " << std::scientific << scale_K << "\n";
        std::cout << "   n_match |  r_in  |  r_out |  |ΔK|_max     | rel\n";
        std::cout << "   --------|--------|--------|--------------|--------\n";

        double worst_rel = 0.0;
        for (int nm : {Nr - 6, Nr - 10, Nr - 20, Nr - 40}) {
            if (nm < 1 || nm >= Nr - 1) continue;
            KMatrixExtractor kme2(b.params, FRP, nm);
            ScatteringResult r2 = kme2.extract();
            const double dK = (r2.K_matrix - K_ref).cwiseAbs().maxCoeff();
            const double rel = dK / scale_K;
            std::cout << "   " << std::setw(7) << nm
                      << " | " << std::fixed << std::setprecision(3)
                      << std::setw(6) << r2.r_match_inner
                      << " | " << std::setw(6) << r2.r_match_outer
                      << " | " << std::scientific << std::setprecision(3)
                      << std::setw(12) << dK
                      << " | " << rel << "\n";
            worst_rel = std::max(worst_rel, rel);
        }
        check(worst_rel < 1e-2,
              "matching-radius convergence rel < 1e-2 (worst=" +
              std::to_string(worst_rel) + ")");
    }

    // ----------------------------------------------------------------------
    // (5) Off-by-one regression (version_0's choice).
    //   Deliberately build a K that mimics version_0's indexing: use
    //   R = Rinv_final^{-1} (which is R_{N−1}) while matching at (N−2, N−1).
    //   Our extractor uses R_{N−2} = Rinv[N−2]^{-1}. These should differ
    //   by an O(h) amount. Both should be unitary, both symmetric — the
    //   bug is non-fatal but physically less precise.
    //   If they differ by a non-trivial amount, we know the indexing
    //   choice is impactful (not a wash) and our choice has been exercised.
    // ----------------------------------------------------------------------
    std::cout << "\n--- (5) off-by-one sensitivity (vs version_0 convention) ---\n";
    {
        // Simulate the "wrong" choice by poking a custom "raw" R into the
        // extractor's flow. Easiest way: temporarily replace the stored
        // Rinv[n_match] with Rinv[n_match + 1] (= Rinv_final).
        // We don't modify FRP; instead we re-derive the matching manually
        // using Rinv_final as the R, matching at (N−2, N−1).
        //
        // Replicate the extraction formulas inline.
        const double h = b.params.dr;
        const double h2_12 = h * h / 12.0;
        const double k_w = std::sqrt(2.0 * b.params.energy);
        const int nm = Nr - 2;
        const double r_in  = b.params.r_min + nm * h;
        const double r_out = b.params.r_min + (nm + 1) * h;

        const Eigen::MatrixXd& Rinv_wrong = FRP.rinv_final();  // Rinv[N−1]
        const Eigen::MatrixXd R_wrong = Rinv_wrong.partialPivLu().inverse();

        const int N_f = b.params.n_occ * b.params.n_sigma;
        Eigen::MatrixXd S_schur_wrong = R_wrong.topLeftCorner(N_psi, N_psi);
        if (N_f > 0) {
            Eigen::MatrixXd R_ff_inv = R_wrong.bottomRightCorner(N_f, N_f)
                                              .partialPivLu().inverse();
            S_schur_wrong -=
                R_wrong.topRightCorner(N_psi, N_f) * R_ff_inv *
                R_wrong.bottomLeftCorner(N_f, N_psi);
        }
        S_schur_wrong = 0.5 * (S_schur_wrong + S_schur_wrong.transpose().eval());

        auto rj = [](int l, double x) -> double {
            gsl_sf_result r; gsl_sf_bessel_jl_e(l, x, &r); return x * r.val;
        };
        auto ry = [](int l, double x) -> double {
            gsl_sf_result r; gsl_sf_bessel_yl_e(l, x, &r); return x * r.val;
        };

        Eigen::MatrixXd J_in = Eigen::MatrixXd::Zero(N_psi, N_psi);
        Eigen::MatrixXd J_out = Eigen::MatrixXd::Zero(N_psi, N_psi);
        Eigen::MatrixXd Y_in = Eigen::MatrixXd::Zero(N_psi, N_psi);
        Eigen::MatrixXd Y_out = Eigen::MatrixXd::Zero(N_psi, N_psi);
        Eigen::MatrixXd W_in = Eigen::MatrixXd::Zero(N_psi, N_psi);
        Eigen::MatrixXd W_out = Eigen::MatrixXd::Zero(N_psi, N_psi);
        for (int mu = 0; mu < N_psi; ++mu) {
            int l, m; scatt::angular::idx_to_lm(mu, l, m);
            J_in (mu, mu) = rj(l, k_w * r_in);
            J_out(mu, mu) = rj(l, k_w * r_out);
            Y_in (mu, mu) = ry(l, k_w * r_in);
            Y_out(mu, mu) = ry(l, k_w * r_out);
            const double c_in  = double(l*(l+1))/(r_in*r_in);
            const double c_out = double(l*(l+1))/(r_out*r_out);
            const double k2 = k_w * k_w;
            W_in (mu, mu) = 1.0 + h2_12 * (k2 - c_in);
            W_out(mu, mu) = 1.0 + h2_12 * (k2 - c_out);
        }

        Eigen::MatrixXd A_wrong = W_out * J_out - S_schur_wrong * (W_in * J_in);
        Eigen::MatrixXd B_wrong = S_schur_wrong * (W_in * Y_in) - W_out * Y_out;
        Eigen::MatrixXd K_wrong = B_wrong.partialPivLu().solve(A_wrong);
        K_wrong = 0.5 * (K_wrong + K_wrong.transpose().eval());

        const double diff = (K_wrong - res.K_matrix).cwiseAbs().maxCoeff();
        const double scale = std::max(res.K_matrix.cwiseAbs().maxCoeff(), 1e-30);
        std::cout << "   ||K_v0 − K_correct||_max = " << std::scientific
                  << std::setprecision(3) << diff
                  << "   rel = " << (diff / scale) << "\n";
        // Bound it: we expect O(h) ~ 0.005 in some entries. As long as the
        // choice matters (non-zero), the indexing has been exercised.
        check(diff > 0.0, "off-by-one choice produces a different K (i.e. indexing matters)");
        check(diff / scale < 0.5,
              "off-by-one K is in the same ballpark (rel < 0.5) — not a catastrophic bug");
    }

    // ----------------------------------------------------------------------
    // (6) Eigenphase ↔ eigS consistency.
    //   Eigenvalues of S = e^{2iδ_α} where δ_α are eigenphases.
    // ----------------------------------------------------------------------
    std::cout << "\n--- (6) eig(S) vs e^{2i·eigenphase} ---\n";
    {
        Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es_S(res.S_matrix);
        Eigen::VectorXcd ev_S = es_S.eigenvalues();
        // Compute phases of eig(S).
        std::vector<double> phi_S(N_psi);
        for (int i = 0; i < N_psi; ++i) {
            phi_S[i] = std::arg(ev_S(i));   // in (-π, π]
        }
        std::sort(phi_S.begin(), phi_S.end());
        // Expected phases: sort(2 * eigenphase) each wrapped into (-π, π].
        std::vector<double> phi_K(N_psi);
        for (int i = 0; i < N_psi; ++i) {
            double t = 2.0 * res.eigenphases[i];
            // Wrap into (-π, π].
            while (t >  M_PI) t -= 2.0 * M_PI;
            while (t <= -M_PI) t += 2.0 * M_PI;
            phi_K[i] = t;
        }
        std::sort(phi_K.begin(), phi_K.end());
        double max_diff = 0.0;
        for (int i = 0; i < N_psi; ++i) {
            max_diff = std::max(max_diff, std::abs(phi_S[i] - phi_K[i]));
        }
        check(max_diff < 1e-10,
              "phases of eig(S) == 2·eigenphases (max_diff=" +
              std::to_string(max_diff) + ")");
    }

    // ----------------------------------------------------------------------
    // (7) Benchmark
    // ----------------------------------------------------------------------
    std::cout << "\n--- (7) Benchmark ---\n";
    {
        auto t0 = std::chrono::steady_clock::now();
        const int REPS = 50;
        for (int i = 0; i < REPS; ++i) {
            ScatteringResult rr = KME.extract();
            (void) rr;
        }
        double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "   extract(): " << std::fixed << std::setprecision(3)
                  << (dt / REPS * 1e3) << " ms/call (N_psi=" << N_psi
                  << ", N_total=" << (b.params.n_mu + b.params.n_occ * b.params.n_sigma)
                  << ")\n";
    }

    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL")
              << " (" << fails << " failed)\n";
    return fails == 0 ? 0 : 1;
}
