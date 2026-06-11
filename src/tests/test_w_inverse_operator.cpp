// test_w_inverse_operator.cpp -- validate WInverseOperator on H2O.
//
// We compare the block-fused W^{-1} application against a *direct* inverse
// of the regularized full W matrix:
//
//     W_reg(n) = [ A(n) + shift_A·I    B(n)        ]    (shift_A applied
//                 [ B(n)^T             D_clamp(n)  ]     iff n < stab_n_max
//                                                        AND the A-shift
//                                                        fired during
//                                                        SchurInverter.build;
//                                                        same for S shift.)
//
// To avoid tracking exactly which shifts fired at which n, we use two
// regimes:
//   A. Gold-standard comparison at n ≥ stab_n_max, where NO shift is
//      applied. W_reg(n) == W(n) exactly. materialize(n) must equal
//      W(n)^{-1} to LU precision.
//   B. Apply / apply_U spot-checks at arbitrary n by comparing
//      apply(n, Z) against materialize(n) · Z. Internal consistency, not
//      a physical comparison.
//
// Additional checks:
//   (3) Identity: W(n) · (W_inv · Z) = Z for random Z at n ≥ stab_n_max.
//   (4) Symmetry: materialize(n) is symmetric within 1e-12.
//   (5) Exchange-off (n ≥ n_transition): W^{-1} is block-diagonal =
//       [Sinv  0; 0  diag(Dinv)].
//   (6) apply_U identity: apply_U(n, X) == 12·apply(n, X) − 10·X.
//   (7) Benchmark: apply wall time as a function of ncols.

#include "io/HDF5Reader.hpp"
#include "scatt/Banner.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

using scatt::ChiRadial;
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

static int fails = 0;
static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++fails;
}

// Build W_reg(n) = [A(n)  B(n); B(n)^T D_clamp(n)] with the SAME D-clamp the
// operator uses but no A/S shift. Only valid for n >= stab_n_max.
static Eigen::MatrixXd
build_W_unshifted(int n, const SolverParams& sp, Potentials& pot,
                  const ExchangeCoupling* ec, const ChiRadial* chi,
                  const std::vector<int>& l_sigma, double W_min)
{
    const int N_psi = sp.n_mu;
    const int N_f   = sp.n_occ * sp.n_sigma;
    const int N_tot = N_psi + N_f;
    const double h  = sp.dr;
    const double h2_12 = h * h / 12.0;
    const double h2_6  = h * h / 6.0;
    const double r  = sp.r_min + n * sp.dr;
    const double r2 = r * r;

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(N_tot, N_tot);

    // A = I + (h²/6)(E·I − V)
    for (int i = 0; i < N_psi; ++i) W(i, i) = 1.0 + h2_6 * sp.energy;
    W.block(0, 0, N_psi, N_psi).noalias() -= h2_6 * pot.get((std::size_t)n);

    // D = max(1 − (h²/12) ℓ(ℓ+1)/r², W_min)
    for (int f = 0; f < N_f; ++f) {
        const int l = l_sigma[f % sp.n_sigma];
        const double centrif = (r2 > 1e-30) ? double(l * (l + 1)) / r2 : 0.0;
        double Df = 1.0 - h2_12 * centrif;
        if (Df < W_min) Df = W_min;
        W(N_psi + f, N_psi + f) = Df;
    }
    // B
    if (ec && n < sp.n_transition) {
        auto ws = ec->make_workspace();
        Eigen::MatrixXd Q = ec->make_output();
        ec->compute_into(n, (*chi)[(std::size_t)n], ws, Q);
        const Eigen::MatrixXd B = h2_12 * Q;
        W.block(0, N_psi, N_psi, N_f) = B;
        W.block(N_psi, 0, N_f, N_psi) = B.transpose();
    }
    return W;
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
    params.l_max_continuum = 4;                 // N_psi=25, N_f=125, N_tot=150
    params.validate();

    SetupBundle b = WavefunctionSetup::prepare(params, data, /*energy=*/0.5);

    Potentials pot(params);
    pot.build(data, StorageMode::MEMORY, "", /*verbose=*/false);

    ExchangeCoupling EC(b.G_coeff, b.params.n_mu, b.params.n_sigma, b.params.n_occ,
                        data.rmin, data.dr);

    SchurInverter SI(b.params, pot, &EC, &b.chi);
    SchurInverter::Config si_cfg;
    si_cfg.storage = StorageMode::MEMORY;
    si_cfg.use_openmp = true;
    si_cfg.verbose = true;
    si_cfg.checkpoint_dir = "./checkpoints/sinv_winv_test";
    std::filesystem::remove_all(si_cfg.checkpoint_dir);
    SI.build(si_cfg);

    WInverseOperator WI(b.params, SI, &EC, &b.chi, si_cfg.W_min);

    std::vector<int> l_sigma(b.params.n_sigma);
    for (int s = 0; s < b.params.n_sigma; ++s) {
        int l, m; scatt::angular::idx_to_lm(s, l, m);
        l_sigma[s] = l;
    }

    const int N_psi = b.params.n_mu;
    const int N_f   = b.params.n_occ * b.params.n_sigma;
    const int N_tot = N_psi + N_f;
    const std::size_t Nr = b.params.n_grid;

    std::cout << "\n--- Dimensions: N_psi=" << N_psi << "  N_f=" << N_f
              << "  N_total=" << N_tot << "  Nr=" << Nr << " ---\n";

    // ----------------------------------------------------------------------
    // (1) Gold standard: materialize(n) == LU-inverse of W_reg(n)  for n ≥ stab
    // ----------------------------------------------------------------------
    std::cout << "\n--- (1) materialize(n) vs dense LU(W) for n >= stab_n_max ---\n";
    double worst_mat = 0.0;
    for (int n : {200, 500, 1000, 1500, 2500, (int)Nr - 2}) {
        if (n < si_cfg.stab_n_max) continue;
        Eigen::MatrixXd W = build_W_unshifted(n, b.params, pot, &EC, &b.chi,
                                              l_sigma, si_cfg.W_min);
        Eigen::MatrixXd W_inv_ref = W.partialPivLu().inverse();
        Eigen::MatrixXd W_inv_mat = WI.materialize(n);
        double err = (W_inv_ref - W_inv_mat).cwiseAbs().maxCoeff();
        double scale = std::max(W_inv_ref.cwiseAbs().maxCoeff(), 1e-30);
        worst_mat = std::max(worst_mat, err / scale);
        std::cout << "     n=" << std::setw(5) << n
                  << "  |W^-1|_max=" << std::scientific << std::setprecision(3) << scale
                  << "  err=" << err << "  rel=" << (err/scale) << "\n";
        check(err < 1e-9, "materialize vs LU inverse at n=" + std::to_string(n));
    }
    std::cout << "     worst relative error: " << worst_mat << "\n";

    // ----------------------------------------------------------------------
    // (2) Apply-operator parity: apply(n, Z) == materialize(n) · Z
    // ----------------------------------------------------------------------
    std::cout << "\n--- (2) apply(n, Z) == materialize(n) · Z ---\n";
    {
        auto ws = WI.make_workspace();
        std::mt19937 rng(42);
        std::normal_distribution<double> nd(0.0, 1.0);

        for (int n : {1, 50, 200, 1500, 2500}) {
            Eigen::MatrixXd M = WI.materialize(n);
            for (int ncols : {1, 5, N_psi}) {
                Eigen::MatrixXd Z(N_tot, ncols);
                for (int i = 0; i < N_tot; ++i)
                    for (int j = 0; j < ncols; ++j)
                        Z(i, j) = nd(rng);
                Eigen::MatrixXd Y_ref = M * Z;
                Eigen::MatrixXd Y_op(N_tot, ncols);
                WI.apply(n, Z, Y_op, ws);
                double err = (Y_ref - Y_op).cwiseAbs().maxCoeff();
                double scale = std::max(Y_ref.cwiseAbs().maxCoeff(), 1e-30);
                check(err / scale < 1e-12,
                      "apply == materialize·Z at n=" + std::to_string(n) +
                      " ncols=" + std::to_string(ncols) +
                      " (rel=" + std::to_string(err/scale) + ")");
            }
        }
    }

    // ----------------------------------------------------------------------
    // (3) Identity: W · (W^{-1} · Z) = Z for random Z at n >= stab
    // ----------------------------------------------------------------------
    std::cout << "\n--- (3) W · (W^{-1} · Z) == Z ---\n";
    {
        auto ws = WI.make_workspace();
        std::mt19937 rng(17);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (int n : {200, 500, 1500, 2500}) {
            Eigen::MatrixXd W = build_W_unshifted(n, b.params, pot, &EC, &b.chi,
                                                  l_sigma, si_cfg.W_min);
            const int ncols = 3;
            Eigen::MatrixXd Z(N_tot, ncols);
            for (int i = 0; i < N_tot; ++i)
                for (int j = 0; j < ncols; ++j)
                    Z(i, j) = nd(rng);
            Eigen::MatrixXd Y(N_tot, ncols);
            WI.apply(n, Z, Y, ws);
            Eigen::MatrixXd Zhat = W * Y;
            double err = (Zhat - Z).cwiseAbs().maxCoeff();
            double scale = std::max(Z.cwiseAbs().maxCoeff(), 1e-30);
            check(err / scale < 1e-10,
                  "W·(W^-1·Z)==Z at n=" + std::to_string(n) +
                  " (rel=" + std::to_string(err/scale) + ")");
        }
    }

    // ----------------------------------------------------------------------
    // (4) Symmetry of materialize(n)
    // ----------------------------------------------------------------------
    std::cout << "\n--- (4) materialize(n) symmetric ---\n";
    {
        double worst = 0.0;
        for (int n : {1, 200, 1500, (int)Nr - 1}) {
            Eigen::MatrixXd M = WI.materialize(n);
            double asym = (M - M.transpose()).cwiseAbs().maxCoeff();
            double scale = std::max(M.cwiseAbs().maxCoeff(), 1e-30);
            worst = std::max(worst, asym / scale);
        }
        check(worst < 1e-12, "||W^-1 - (W^-1)^T|| rel < 1e-12 (worst=" +
                             std::to_string(worst) + ")");
    }

    // ----------------------------------------------------------------------
    // (5) Exchange-off: W^{-1} block-diagonal  [Sinv 0; 0 diag(Dinv)]
    // ----------------------------------------------------------------------
    std::cout << "\n--- (5) exchange-off: W^{-1} is block-diagonal ---\n";
    {
        // Use a second SolverParams with n_transition < N_grid to force this.
        SolverParams sp2 = b.params;
        sp2.n_transition = 500;
        SchurInverter SI2(sp2, pot, &EC, &b.chi);
        SchurInverter::Config cfg2 = si_cfg;
        cfg2.checkpoint_dir = "./checkpoints/sinv_winv_test_exchoff";
        cfg2.verbose = false;
        std::filesystem::remove_all(cfg2.checkpoint_dir);
        SI2.build(cfg2);
        WInverseOperator WI2(sp2, SI2, &EC, &b.chi, cfg2.W_min);

        const int n = 1000;  // safely past n_transition
        Eigen::MatrixXd M = WI2.materialize(n);
        // Top-left must equal Sinv.
        const auto& Sinv = SI2.get(n);
        double err_tl = (M.topLeftCorner(N_psi, N_psi) - Sinv).cwiseAbs().maxCoeff();
        // Bottom-right must be diagonal with entries 1/D_clamp.
        double err_br = 0.0, err_off = 0.0;
        const double r2 = std::pow(sp2.r_min + n * sp2.dr, 2);
        const double h2_12 = sp2.dr * sp2.dr / 12.0;
        for (int f = 0; f < N_f; ++f) {
            const int l = l_sigma[f % sp2.n_sigma];
            const double centrif = double(l * (l + 1)) / r2;
            double Df = 1.0 - h2_12 * centrif;
            if (Df < si_cfg.W_min) Df = si_cfg.W_min;
            err_br = std::max(err_br,
                std::abs(M(N_psi + f, N_psi + f) - 1.0 / Df));
            for (int g = 0; g < N_f; ++g)
                if (g != f)
                    err_br = std::max(err_br,
                        std::abs(M(N_psi + f, N_psi + g)));
        }
        // Off-diagonal blocks must be zero.
        err_off = std::max(
            M.topRightCorner(N_psi, N_f).cwiseAbs().maxCoeff(),
            M.bottomLeftCorner(N_f, N_psi).cwiseAbs().maxCoeff());

        check(err_tl < 1e-12, "top-left = Sinv (err=" + std::to_string(err_tl) + ")");
        check(err_br < 1e-12, "bottom-right = diag(1/D_clamp) (err=" + std::to_string(err_br) + ")");
        check(err_off < 1e-12, "off-diagonal blocks = 0 (err=" + std::to_string(err_off) + ")");
    }

    // ----------------------------------------------------------------------
    // (6) apply_U == 12·apply − 10·X
    // ----------------------------------------------------------------------
    std::cout << "\n--- (6) apply_U == 12·apply − 10·I ---\n";
    {
        auto ws = WI.make_workspace();
        std::mt19937 rng(3);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (int n : {1, 200, 1500}) {
            const int ncols = 4;
            Eigen::MatrixXd X(N_tot, ncols);
            for (int i = 0; i < N_tot; ++i)
                for (int j = 0; j < ncols; ++j)
                    X(i, j) = nd(rng);
            Eigen::MatrixXd Y_apply(N_tot, ncols);
            WI.apply(n, X, Y_apply, ws);
            Eigen::MatrixXd Y_ref = 12.0 * Y_apply - 10.0 * X;
            Eigen::MatrixXd Y_U(N_tot, ncols);
            WI.apply_U(n, X, Y_U, ws);
            double err = (Y_ref - Y_U).cwiseAbs().maxCoeff();
            double scale = std::max(Y_ref.cwiseAbs().maxCoeff(), 1e-30);
            check(err / scale < 1e-12,
                  "apply_U identity at n=" + std::to_string(n) +
                  " (rel=" + std::to_string(err/scale) + ")");
        }
    }

    // ----------------------------------------------------------------------
    // (7) Benchmark
    // ----------------------------------------------------------------------
    std::cout << "\n--- (7) Benchmark: apply per n ---\n";
    {
        auto ws = WI.make_workspace();
        std::cout << "   ncols |  total s |  μs/apply | pts/s\n";
        std::cout << "   ------|----------|-----------|-------\n";
        for (int ncols : {1, N_psi, N_tot}) {
            Eigen::MatrixXd Z = Eigen::MatrixXd::Random(N_tot, ncols);
            Eigen::MatrixXd Y(N_tot, ncols);

            // Warm-up
            for (int n = 0; n < (int)Nr; ++n) WI.apply(n, Z, Y, ws);

            const int REPS = 3;
            auto t0 = std::chrono::steady_clock::now();
            for (int rep = 0; rep < REPS; ++rep)
                for (int n = 0; n < (int)Nr; ++n) WI.apply(n, Z, Y, ws);
            double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            double calls = (double)Nr * REPS;
            std::cout << "   " << std::setw(5) << ncols
                      << " | " << std::fixed << std::setprecision(3) << std::setw(8) << dt
                      << " | " << std::setprecision(2) << std::setw(9) << (dt / calls * 1e6)
                      << " | " << std::setprecision(0) << std::setw(6) << (calls / dt) << "\n";
        }

        // Compare against "materialize + matmul" baseline at ncols = N_psi.
        const int ncols = N_psi;
        Eigen::MatrixXd Z = Eigen::MatrixXd::Random(N_tot, ncols);
        auto t0 = std::chrono::steady_clock::now();
        for (int n = 0; n < (int)Nr; n += 10) {
            Eigen::MatrixXd M = WI.materialize(n);
            Eigen::MatrixXd Y = M * Z;
            (void) Y;
        }
        double dt_ref = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "\n   ref: materialize + matmul (ncols=" << ncols
                  << ", stride=10): " << std::fixed << std::setprecision(3) << dt_ref << " s  ("
                  << std::setprecision(2) << (dt_ref / ((double)Nr / 10.0) * 1e6)
                  << " μs/call)\n";
    }

    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL")
              << " (" << fails << " failed)\n";
    return fails == 0 ? 0 : 1;
}
