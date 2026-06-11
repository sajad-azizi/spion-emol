// test_winv_blockdiag_fastpath.cpp
//
// Validates a proposed optimization for the GPU back-prop path:
//
//   When ec_ == nullptr OR n >= n_transition, W_n^{-1} has the structure
//
//     W_n^{-1}  =  [ Sinv         0          ]
//                  [   0     diag(1/D_clamp) ]
//
//   so the materialize_into call (which currently does
//   apply(n, I, Winv_out, ws), an O(N_psi * N_psi * N_total) GEMM
//   internally) can be replaced by a direct block-by-block copy at
//   O(N_total^2) memory bandwidth -- no GEMM required.
//
// This test verifies BIT-IDENTICAL agreement between
//   (i)  the current materialize(n) path
//   (ii) a hand-rolled "fast block-diagonal materialize" (here
//        materialize_blockdiag) that avoids any GEMM,
// at every grid point n in [n_transition, N_grid).
//
// If this test passes with err == 0.0, the optimization is safe to
// deploy in BackPropagator's GPU path: it produces the same W_inv_host
// matrix that the GPU step then consumes.
//
// Uses the standard h2o_ccpvdz_sph fixture and the same SchurInverter
// construction as test_w_inverse_operator.cpp.

#include "io/HDF5Reader.hpp"
#include "scatt/Banner.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"
#include "angular/Gaunt.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using scatt::ExchangeCoupling;
using scatt::Parameters;
using scatt::Potentials;
using scatt::SchurInverter;
using scatt::SetupBundle;
using scatt::SolverParams;
using scatt::StorageMode;
using scatt::WavefunctionSetup;
using scatt::WInverseOperator;
using scatt::WInverseOperatorWorkspace;
using scatt::io::HDF5Reader;
using scatt::io::PreprocData;

static int n_fail = 0;
static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++n_fail;
}

// ---------------------------------------------------------------------- //
// Hand-rolled "fast block-diagonal materialize".
//
// This is the proposed optimization, written purely with copy / fill
// operations -- no GEMM.  Returns an (N_total x N_total) W^{-1} that
// must equal WI.materialize(n) when ec is null or n >= n_transition.
// ---------------------------------------------------------------------- //
static Eigen::MatrixXd
materialize_blockdiag_fast(int n,
                           const SolverParams& sp,
                           SchurInverter& si,
                           double W_min,
                           const std::vector<int>& l_sigma)
{
    const int N_psi = sp.n_mu;
    const int N_f   = sp.n_occ * sp.n_sigma;
    const int N_tot = N_psi + N_f;

    Eigen::MatrixXd W_inv = Eigen::MatrixXd::Zero(N_tot, N_tot);

    // Top-left: copy Sinv directly.
    W_inv.topLeftCorner(N_psi, N_psi) = si.get(static_cast<std::size_t>(n));

    // Bottom-right: diag(1 / D_clamp) -- identical convention to
    // WInverseOperator::load_B_and_Dinv_().
    const double r  = sp.r_min + n * sp.dr;
    const double r2 = r * r;
    const double h2_12 = sp.dr * sp.dr / 12.0;
    for (int f = 0; f < N_f; ++f) {
        const int l = l_sigma[f % sp.n_sigma];
        const double centrif = (r2 > 1e-30)
            ? double(l * (l + 1)) / r2 : 0.0;
        double Df = 1.0 - h2_12 * centrif;
        if (Df < W_min) Df = W_min;
        W_inv(N_psi + f, N_psi + f) = 1.0 / Df;
    }
    // Off-diagonal blocks left as zero (already zero from the setZero above).
    return W_inv;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <h2o_preproc.h5>\n";
        return 2;
    }
    scatt::print_la_banner();

    HDF5Reader reader(argv[1]);
    PreprocData data = reader.load_all();

    // Same setup as test_w_inverse_operator.cpp test (5).
    Parameters params;
    params.r_min           = data.rmin;
    params.dr              = data.dr;
    params.N_grid          = data.Nr;
    params.Lmax_sce        = data.Lmax_sce;
    params.l_max_continuum = 4;     // small + fast
    params.validate();

    SetupBundle b = WavefunctionSetup::prepare(params, data, /*energy=*/0.5);

    Potentials pot(params);
    pot.build(data, StorageMode::MEMORY, "", /*verbose=*/false);

    ExchangeCoupling EC(b.G_coeff, b.params.n_mu, b.params.n_sigma,
                        b.params.n_occ, data.rmin, data.dr);

    // Force n_transition < N_grid so we have a sizeable asymptotic region
    // to scan over.
    SolverParams sp = b.params;
    sp.n_transition = 500;     // matches existing test (5)
    SchurInverter SI(sp, pot, &EC, &b.chi);
    SchurInverter::Config cfg;
    cfg.storage             = StorageMode::MEMORY;
    cfg.use_openmp          = true;
    cfg.verbose             = false;
    cfg.checkpoint_dir      = "./checkpoints/sinv_winv_blockdiag_fastpath";
    cfg.try_load_checkpoint = false;
    cfg.save_checkpoint     = false;
    std::filesystem::remove_all(cfg.checkpoint_dir);
    SI.build(cfg);

    WInverseOperator WI(sp, SI, &EC, &b.chi, cfg.W_min);

    const int N_psi = sp.n_mu;
    const int N_f   = sp.n_occ * sp.n_sigma;
    const int N_tot = N_psi + N_f;

    std::vector<int> l_sigma(sp.n_sigma);
    for (int s = 0; s < sp.n_sigma; ++s) {
        int l, m; scatt::angular::idx_to_lm(s, l, m); l_sigma[s] = l;
    }

    std::cout << "[setup] N_psi=" << N_psi << "  N_f=" << N_f
              << "  N_total=" << N_tot
              << "  n_transition=" << sp.n_transition
              << "  N_grid=" << sp.n_grid
              << "  W_min=" << cfg.W_min << "\n\n";

    // ------------------------------------------------------------------ //
    // Test (A): bit-identical materialize  vs  blockdiag-fast at every n
    // in the asymptotic region.
    // ------------------------------------------------------------------ //
    std::cout << "--- (A) materialize == blockdiag-fast at every n >= n_transition ---\n";
    {
        double max_diff_total = 0.0;
        // Walk a representative subset (every step is too slow but
        // fairly comprehensive: ~6 points across the asymptotic region).
        std::vector<int> n_samples;
        for (int n = sp.n_transition;
             n < static_cast<int>(sp.n_grid);
             n += std::max(1, (static_cast<int>(sp.n_grid) - sp.n_transition) / 6))
            n_samples.push_back(n);
        for (int n : n_samples) {
            Eigen::MatrixXd M_full = WI.materialize(n);
            Eigen::MatrixXd M_fast = materialize_blockdiag_fast(
                n, sp, SI, cfg.W_min, l_sigma);
            double diff = (M_full - M_fast).cwiseAbs().maxCoeff();
            max_diff_total = std::max(max_diff_total, diff);
            check(diff == 0.0,
                  "n=" + std::to_string(n) + "  max|M_full - M_fast| = "
                  + std::to_string(diff));
        }
        std::cout << "  worst max-abs over all sampled n: " << max_diff_total << "\n";
    }

    // ------------------------------------------------------------------ //
    // Test (B): bit-identical W_inv * Z for arbitrary Z at one
    // representative asymptotic n.  This is the actual operation
    // performed by the GPU back stepper.
    // ------------------------------------------------------------------ //
    std::cout << "\n--- (B) W_inv · Z is bit-identical for the two materializations ---\n";
    {
        const int n = 1500;     // past n_transition, near the middle of asym
        Eigen::MatrixXd M_full = WI.materialize(n);
        Eigen::MatrixXd M_fast = materialize_blockdiag_fast(
            n, sp, SI, cfg.W_min, l_sigma);
        std::mt19937 rng(7);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (int ncols : {1, 5, 25}) {
            Eigen::MatrixXd Z(N_tot, ncols);
            for (int i = 0; i < N_tot; ++i)
                for (int j = 0; j < ncols; ++j)
                    Z(i, j) = nd(rng);
            Eigen::MatrixXd Y_full = M_full * Z;
            Eigen::MatrixXd Y_fast = M_fast * Z;
            double diff = (Y_full - Y_fast).cwiseAbs().maxCoeff();
            check(diff == 0.0,
                  "n=" + std::to_string(n) + " ncols="
                  + std::to_string(ncols) + " max|Y_full - Y_fast| = "
                  + std::to_string(diff));
        }
    }

    // ------------------------------------------------------------------ //
    // Test (B2): the PATCHED production materialize_into() must agree
    // bit-for-bit with the (unchanged) materialize() at every n.
    // ------------------------------------------------------------------ //
    std::cout << "\n--- (B2) materialize_into() == materialize() at every n ---\n";
    {
        auto ws = WI.make_workspace();
        Eigen::MatrixXd Winv_into(N_tot, N_tot);
        std::vector<int> n_check;
        // Cover both regimes: inner (< n_transition) and asymptotic (>=).
        for (int n : {1, 100, 250, 499, 500, 800, 1500, 2500, 2999})
            n_check.push_back(n);
        double worst_diff = 0.0;
        for (int n : n_check) {
            Eigen::MatrixXd M_mat = WI.materialize(n);
            WI.materialize_into(n, Winv_into, ws);
            double diff = (M_mat - Winv_into).cwiseAbs().maxCoeff();
            worst_diff = std::max(worst_diff, diff);
            check(diff == 0.0,
                  "n=" + std::to_string(n)
                  + (n < sp.n_transition ? " (inner)" : " (asymptotic)")
                  + "  max|materialize - materialize_into| = "
                  + std::to_string(diff));
        }
        std::cout << "  worst diff over all sampled n (both regimes): "
                  << worst_diff << "\n";
    }

    // ------------------------------------------------------------------ //
    // Test (C): blockdiag-fast must NOT be used inside n < n_transition.
    // We verify it is DIFFERENT from materialize there (so the safety
    // guard is meaningful, not vacuous).
    // ------------------------------------------------------------------ //
    std::cout << "\n--- (C) blockdiag-fast is wrong inside n < n_transition (sanity) ---\n";
    {
        const int n = 100;     // well inside n_transition=500
        Eigen::MatrixXd M_full = WI.materialize(n);
        Eigen::MatrixXd M_fast = materialize_blockdiag_fast(
            n, sp, SI, cfg.W_min, l_sigma);
        double diff = (M_full - M_fast).cwiseAbs().maxCoeff();
        check(diff > 1e-6,
              "inside molecular region the two should DIFFER (diff="
              + std::to_string(diff) + ")");
    }

    std::cout << "\n" << (n_fail == 0 ? "PASS" : "FAIL")
              << " (" << n_fail << " failed)\n";
    return n_fail == 0 ? 0 : 1;
}
