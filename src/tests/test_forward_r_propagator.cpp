// test_forward_r_propagator.cpp -- validate ForwardRPropagator on H2O.
//
// The gold-standard reference is an INDEPENDENT re-implementation of the
// Johnson R-recursion:
//   - build W_n = [A  B; B^T D_clamp] from first principles at every n
//     (no stability shift -- for H2O l_cont=4 our SchurInverter log showed
//     A-shifts=1 at n=1, which sits inside the analytic-init region, so
//     the numerical-recursion range is shift-free and this is a fair
//     comparison);
//   - invert with LU;
//   - form U_n = 12 W_n^{-1} - 10 I;
//   - carry Rinv via Rinv_n = (U_n - Rinv_prev)^{-1};
//   - share the SAME analytic-init formula for n < n_start.
//
// Checks:
//   (1) Analytic init diagonal matches closed form at probed n ∈ [1, n_start).
//   (2) Gold-standard bit equality at probed n well past n_start.
//   (3) Symmetry ||Rinv - Rinv^T|| small.
//   (4) No NaN/Inf.
//   (5) rinv_final() == get(N_grid - 1).
//   (6) Checkpoint round-trip (MEMORY + save_to_disk + try_load_into_memory).
//   (7) MEMORY vs DISK bit equality.
//   (8) Benchmark: per-n time.

#include "io/HDF5Reader.hpp"
#include "scatt/Banner.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/ForwardRPropagator.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <gsl/gsl_sf_bessel.h>

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using scatt::ChiRadial;
using scatt::ExchangeCoupling;
using scatt::ForwardRPropagator;
using scatt::Parameters;
using scatt::Potentials;
using scatt::SchurInverter;
using scatt::SetupBundle;
using scatt::SolverParams;
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

// -------------------- reference impl (stand-alone) ---------------------------

// Build W_n = [A B; B^T D_clamp] without any stability shift. This IS what
// ForwardRPropagator implicitly uses in the n >= stab_n_max region (and,
// for H2O l_cont=4, in the whole numerical-recursion range since the
// single A-shift fires only at n=1 which is inside analytic init).
static Eigen::MatrixXd
build_W_ref(int n, const SolverParams& sp, Potentials& pot,
            const ExchangeCoupling* ec, const ChiRadial* chi,
            const std::vector<int>& l_sigma, double W_min)
{
    const int N_psi = sp.n_mu;
    const int N_f   = sp.n_occ * sp.n_sigma;
    const int N_tot = N_psi + N_f;
    const double h      = sp.dr;
    const double h2_12  = h * h / 12.0;
    const double h2_6   = h * h / 6.0;
    const double r      = sp.r_min + n * h;
    const double r2     = r * r;

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(N_tot, N_tot);

    // A = I + (h²/6)(E·I − V)
    for (int i = 0; i < N_psi; ++i) W(i, i) = 1.0 + h2_6 * sp.energy;
    W.block(0, 0, N_psi, N_psi).noalias() -= h2_6 * pot.get((std::size_t)n);

    // D = max(1 − (h²/12)·ℓ(ℓ+1)/r², W_min)
    for (int f = 0; f < N_f; ++f) {
        const int l = l_sigma[f % sp.n_sigma];
        const double centrif = (r2 > 1e-30) ? double(l * (l + 1)) / r2 : 0.0;
        double Df = 1.0 - h2_12 * centrif;
        if (Df < W_min) Df = W_min;
        W(N_psi + f, N_psi + f) = Df;
    }
    // B = (h²/12)·Q_ψf(n)
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

static double riccati_J_ref(int l, double x) {
    if (x < 1e-300) return 0.0;
    return x * gsl_sf_bessel_jl(l, x);
}

// Compute the analytic diagonal entries for one n as a raw reference.
static Eigen::VectorXd
analytic_ref_diag(int n, const SolverParams& sp, Potentials& pot,
                  const std::vector<int>& l_psi,
                  const std::vector<int>& l_sigma, double W_min)
{
    const int N_psi = sp.n_mu;
    const int N_f   = sp.n_occ * sp.n_sigma;
    const int N_tot = N_psi + N_f;
    const double h = sp.dr;
    const double h2_12 = h * h / 12.0;
    const double h2_6  = h * h / 6.0;
    const double k     = std::sqrt(2.0 * sp.energy);
    const double r_n   = sp.r_min + n       * h;
    const double r_np1 = sp.r_min + (n + 1) * h;

    Eigen::VectorXd d = Eigen::VectorXd::Zero(N_tot);

    if (n <= 0) return d;

    const auto& V_n   = pot.get((std::size_t)n);
    const auto& V_np1 = pot.get((std::size_t)(n + 1));

    for (int mu = 0; mu < N_psi; ++mu) {
        const double M_n   = 1.0 + h2_6 * (sp.energy - V_n  (mu, mu));
        const double M_np1 = 1.0 + h2_6 * (sp.energy - V_np1(mu, mu));
        const double y_n   = riccati_J_ref(l_psi[mu], k * r_n);
        const double y_np1 = riccati_J_ref(l_psi[mu], k * r_np1);
        const double denom = M_np1 * y_np1;
        if (std::abs(denom) > 1e-300) d(mu) = (M_n * y_n) / denom;
    }
    const double r_n2   = r_n   * r_n;
    const double r_np12 = r_np1 * r_np1;
    for (int f = 0; f < N_f; ++f) {
        const int l = l_sigma[f % sp.n_sigma];
        const double centrif_n   = (r_n2   > 1e-30) ? double(l*(l+1))/r_n2   : 0.0;
        const double centrif_np1 = (r_np12 > 1e-30) ? double(l*(l+1))/r_np12 : 0.0;
        double M_n   = 1.0 - h2_12 * centrif_n;
        double M_np1 = 1.0 - h2_12 * centrif_np1;
        if (M_n   < W_min) M_n   = W_min;
        if (M_np1 < W_min) M_np1 = W_min;
        const double y_n   = std::pow(r_n,   double(l + 1));
        const double y_np1 = std::pow(r_np1, double(l + 1));
        const double denom = M_np1 * y_np1;
        if (std::abs(denom) > 1e-300) d(N_psi + f) = (M_n * y_n) / denom;
    }
    return d;
}

// Independent full recursion up to a given n_stop.
static std::vector<Eigen::MatrixXd>
reference_forward_R(const SolverParams& sp, Potentials& pot,
                    const ExchangeCoupling* ec, const ChiRadial* chi,
                    const std::vector<int>& l_psi,
                    const std::vector<int>& l_sigma,
                    int n_start, int n_stop, double W_min)
{
    const int N_psi = sp.n_mu;
    const int N_f   = sp.n_occ * sp.n_sigma;
    const int N_tot = N_psi + N_f;

    std::vector<Eigen::MatrixXd> Rinv(sp.n_grid);

    // Analytic init.
    for (int n = 0; n < n_start; ++n) {
        Rinv[n] = Eigen::MatrixXd::Zero(N_tot, N_tot);
        if (n > 0) {
            Eigen::VectorXd diag = analytic_ref_diag(n, sp, pot, l_psi, l_sigma, W_min);
            Rinv[n].diagonal() = diag;
        }
    }

    // Numerical recursion.
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N_tot, N_tot);
    Eigen::MatrixXd Rinv_prev = Rinv[n_start - 1];
    for (int n = n_start; n <= n_stop; ++n) {
        Eigen::MatrixXd W = build_W_ref(n, sp, pot, ec, chi, l_sigma, W_min);
        Eigen::MatrixXd W_inv = W.partialPivLu().inverse();
        Eigen::MatrixXd U = 12.0 * W_inv - 10.0 * I;
        Eigen::MatrixXd R = U - Rinv_prev;
        Eigen::MatrixXd Rn = R.partialPivLu().inverse();
        Rn = 0.5 * (Rn + Rn.transpose().eval());
        Rinv[n] = Rn;
        Rinv_prev = Rn;
    }
    return Rinv;
}

// ----------------------------- main ------------------------------------------

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
    si_cfg.verbose = false;
    si_cfg.checkpoint_dir = "./checkpoints/sinv_frp_test";
    si_cfg.try_load_checkpoint = false;
    si_cfg.save_checkpoint = false;
    std::filesystem::remove_all(si_cfg.checkpoint_dir);
    SI.build(si_cfg);
    std::cout << "[info] SchurInverter shifts: A=" << SI.stability_shifts_A()
              << " S=" << SI.stability_shifts_S() << "\n";

    WInverseOperator WI(b.params, SI, &EC, &b.chi, si_cfg.W_min);

    std::vector<int> l_psi(b.params.n_mu), l_sigma(b.params.n_sigma);
    for (int mu = 0; mu < b.params.n_mu; ++mu) {
        int l, m; scatt::angular::idx_to_lm(mu, l, m); l_psi[mu] = l;
    }
    for (int s = 0; s < b.params.n_sigma; ++s) {
        int l, m; scatt::angular::idx_to_lm(s, l, m); l_sigma[s] = l;
    }

    std::cout << "\n=== ForwardRPropagator build ===\n";
    ForwardRPropagator FRP(b.params, pot, WI);
    ForwardRPropagator::Config frp_cfg;
    frp_cfg.storage             = StorageMode::MEMORY;
    frp_cfg.try_load_checkpoint = false;
    frp_cfg.save_checkpoint     = false;
    frp_cfg.checkpoint_dir      = "./checkpoints/rinv_frp_test";
    std::filesystem::remove_all(frp_cfg.checkpoint_dir);
    frp_cfg.verbose             = true;
    FRP.run(frp_cfg);

    const int n_start = FRP.n_start();
    const std::size_t Nr = b.params.n_grid;
    const int N_psi = b.params.n_mu;
    const int N_f   = b.params.n_occ * b.params.n_sigma;
    const int N_tot = N_psi + N_f;

    std::cout << "\n[info] n_start=" << n_start
              << "  r_start=" << std::fixed << std::setprecision(4)
              << (b.params.r_min + n_start * b.params.dr)
              << "  N_total=" << N_tot << "  Nr=" << Nr << "\n";

    // ----------------------------------------------------------------------
    // (1) Analytic init diagonal correctness
    // ----------------------------------------------------------------------
    std::cout << "\n--- (1) analytic init diagonal at n in [1, n_start) ---\n";
    {
        double worst = 0.0;
        for (int n = 1; n < n_start; ++n) {
            const auto& Rn = FRP.get((std::size_t)n);
            Eigen::VectorXd ref_d = analytic_ref_diag(n, b.params, pot, l_psi, l_sigma,
                                                      si_cfg.W_min);
            // Rn must be diagonal (off-diag zero).
            double max_off = 0.0;
            for (int i = 0; i < N_tot; ++i)
                for (int j = 0; j < N_tot; ++j)
                    if (i != j) max_off = std::max(max_off, std::abs(Rn(i, j)));

            double max_d_err = (Rn.diagonal() - ref_d).cwiseAbs().maxCoeff();
            worst = std::max(worst, std::max(max_off, max_d_err));
            std::cout << "     n=" << std::setw(3) << n
                      << "  |diag - ref|_max=" << std::scientific << std::setprecision(3) << max_d_err
                      << "  |offdiag|_max=" << max_off << "\n";
        }
        check(worst < 1e-14, "analytic init identity (worst=" + std::to_string(worst) + ")");
    }

    // ----------------------------------------------------------------------
    // (2) Gold-standard recursion agreement at many n
    // ----------------------------------------------------------------------
    std::cout << "\n--- (2) gold-standard recursion (independent path) ---\n";
    {
        // Build reference up to the last probed n (cost: O(N_grid) 150x150 LUs).
        const int n_stop = static_cast<int>(Nr) - 1;
        auto t0 = std::chrono::steady_clock::now();
        auto ref = reference_forward_R(b.params, pot, &EC, &b.chi,
                                        l_psi, l_sigma, n_start, n_stop, si_cfg.W_min);
        double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "   reference built in " << std::fixed << std::setprecision(3)
                  << dt << " s\n";

        double worst_rel = 0.0;
        int worst_n = -1;
        for (int n : {n_start, n_start + 1, n_start + 5, 50, 200, 500, 1000, 2000,
                      (int)Nr - 2, (int)Nr - 1})
        {
            if (n < n_start || n >= (int)Nr) continue;
            const auto& mine = FRP.get((std::size_t)n);
            double err = (mine - ref[n]).cwiseAbs().maxCoeff();
            double scale = std::max(ref[n].cwiseAbs().maxCoeff(), 1e-30);
            double rel = err / scale;
            if (rel > worst_rel) { worst_rel = rel; worst_n = n; }
            std::cout << "     n=" << std::setw(5) << n
                      << "  |ref|_max=" << std::scientific << std::setprecision(3) << scale
                      << "  |mine - ref|_max=" << err
                      << "  rel=" << rel << "\n";
        }
        // Tolerance 1e-8: accumulated LU-rounding floor across 3000 recursive
        // steps at N_tot=150.  Eigen-native path hits ~1e-10; MKL's LAPACKE
        // dgetrf has slightly different pivot/rounding order and lands at
        // ~1e-9.  Either is well below any physically meaningful error.
        check(worst_rel < 1e-8,
              "gold-standard agreement rel < 1e-8 (worst at n=" + std::to_string(worst_n) +
              ", rel=" + std::to_string(worst_rel) + ")");
    }

    // ----------------------------------------------------------------------
    // (3) Symmetry at many n
    // ----------------------------------------------------------------------
    std::cout << "\n--- (3) symmetry ||Rinv - Rinv^T|| / ||Rinv|| ---\n";
    {
        double worst = 0.0;
        for (int n : {1, n_start, n_start + 1, 100, 500, 1500, (int)Nr - 1}) {
            if (n < 0 || n >= (int)Nr) continue;
            const auto& Rn = FRP.get((std::size_t)n);
            double asym = (Rn - Rn.transpose()).cwiseAbs().maxCoeff();
            double scale = std::max(Rn.cwiseAbs().maxCoeff(), 1e-30);
            worst = std::max(worst, asym / scale);
        }
        check(worst < 1e-12, "symmetry rel < 1e-12 (worst=" + std::to_string(worst) + ")");
    }

    // ----------------------------------------------------------------------
    // (4) No NaN/Inf across the whole grid
    // ----------------------------------------------------------------------
    std::cout << "\n--- (4) no NaN/Inf ---\n";
    {
        bool all_ok = true;
        for (std::size_t n = 0; n < Nr; ++n) {
            if (!FRP.get(n).allFinite()) { all_ok = false;
                std::cout << "     NaN/Inf at n=" << n << "\n"; break; }
        }
        check(all_ok, "all entries finite at every n");
    }

    // ----------------------------------------------------------------------
    // (5) rinv_final() == get(N_grid - 1)
    // ----------------------------------------------------------------------
    std::cout << "\n--- (5) rinv_final() consistency ---\n";
    {
        double d = (FRP.rinv_final() - FRP.get(Nr - 1)).cwiseAbs().maxCoeff();
        check(d < 1e-14, "rinv_final() == get(N_grid-1)  (d=" + std::to_string(d) + ")");
    }

    // ----------------------------------------------------------------------
    // (6) Checkpoint round-trip (MEMORY build + save -> load_into_memory)
    // ----------------------------------------------------------------------
    std::cout << "\n--- (6) checkpoint round-trip ---\n";
    {
        const std::string ck = "./checkpoints/rinv_roundtrip";
        std::filesystem::remove_all(ck);

        // Fresh build with save_checkpoint=true.
        ForwardRPropagator FRP_a(b.params, pot, WI);
        ForwardRPropagator::Config c_a = frp_cfg;
        c_a.checkpoint_dir      = ck;
        c_a.try_load_checkpoint = false;
        c_a.save_checkpoint     = true;
        c_a.verbose             = false;
        FRP_a.run(c_a);

        // Second instance loads the checkpoint into MEMORY.
        ForwardRPropagator FRP_b(b.params, pot, WI);
        ForwardRPropagator::Config c_b = frp_cfg;
        c_b.checkpoint_dir      = ck;
        c_b.try_load_checkpoint = true;
        c_b.save_checkpoint     = false;
        c_b.verbose             = true;
        FRP_b.run(c_b);

        double worst = 0.0;
        for (int n : {0, 1, n_start, 100, 1500, (int)Nr - 1}) {
            if (n < 0 || n >= (int)Nr) continue;
            double d = (FRP_a.get((std::size_t)n) - FRP_b.get((std::size_t)n))
                       .cwiseAbs().maxCoeff();
            worst = std::max(worst, d);
        }
        check(worst < 1e-14, "MEMORY build + reload bit-equal (worst=" +
                             std::to_string(worst) + ")");
        std::filesystem::remove_all(ck);
    }

    // ----------------------------------------------------------------------
    // (7) MEMORY vs DISK bit-equal
    // ----------------------------------------------------------------------
    std::cout << "\n--- (7) MEMORY vs DISK bit-equal ---\n";
    {
        ForwardRPropagator FRP_disk(b.params, pot, WI);
        ForwardRPropagator::Config cd = frp_cfg;
        cd.storage              = StorageMode::DISK;
        cd.checkpoint_dir       = "./checkpoints/rinv_disk_test";
        cd.try_load_checkpoint  = false;
        cd.save_checkpoint      = false;
        cd.chunk_size           = 50;
        cd.verbose              = false;
        std::filesystem::remove_all(cd.checkpoint_dir);
        FRP_disk.run(cd);

        double worst = 0.0;
        for (int n : {0, 1, n_start, 500, 1500, (int)Nr - 1}) {
            if (n < 0 || n >= (int)Nr) continue;
            double d = (FRP.get((std::size_t)n) - FRP_disk.get((std::size_t)n))
                       .cwiseAbs().maxCoeff();
            worst = std::max(worst, d);
        }
        check(worst < 1e-14, "MEMORY == DISK (worst=" + std::to_string(worst) + ")");
    }

    // ----------------------------------------------------------------------
    // (8) PER-N NUMEROV IDENTITY: local consistency check.
    //
    //     For every n ≥ n_start, the recurrence defines Rinv_n as the
    //     inverse of (U_n − Rinv_{n-1}), i.e.
    //
    //         Rinv_n · (U_n − Rinv_{n-1})  =  I      (eq. *)
    //
    //     If we RECOMPUTE U_n and Rinv_{n-1} from the stored values, this
    //     identity must hold to LU precision at every single n. This is a
    //     LOCAL consistency check that doesn't require any reference impl
    //     and catches errors the gold-standard might miss (e.g. if both
    //     paths had a shared subtle bug in apply_U, they'd agree but the
    //     recurrence wouldn't close). It's the strongest intrinsic check.
    // ----------------------------------------------------------------------
    std::cout << "\n--- (8) per-n Numerov identity Rinv_n · (U_n − Rinv_{n-1}) = I ---\n";
    {
        Eigen::MatrixXd I_tot = Eigen::MatrixXd::Identity(N_tot, N_tot);
        Eigen::MatrixXd U(N_tot, N_tot);
        auto ws = WI.make_workspace();
        double worst = 0.0;
        int    worst_n = -1;
        // Sample many n, including immediately after analytic init and at
        // the outer boundary.
        std::vector<int> probe;
        for (int n = n_start; n <= (int)Nr - 1; n += 100) probe.push_back(n);
        probe.push_back(n_start);      // first numerical step
        probe.push_back(n_start + 1);
        probe.push_back((int)Nr - 1);  // last step
        std::sort(probe.begin(), probe.end());
        probe.erase(std::unique(probe.begin(), probe.end()), probe.end());

        for (int n : probe) {
            WI.apply_U(n, I_tot, U, ws);
            const Eigen::MatrixXd& Rprev = FRP.get((std::size_t)(n - 1));
            const Eigen::MatrixXd& Rn    = FRP.get((std::size_t)n);
            Eigen::MatrixXd Id = Rn * (U - Rprev);
            double err = (Id - I_tot).cwiseAbs().maxCoeff();
            if (err > worst) { worst = err; worst_n = n; }
        }
        std::cout << "   worst |Rinv_n · (U_n − Rinv_{n-1}) − I|_max across "
                  << probe.size() << " probed n: " << std::scientific
                  << std::setprecision(3) << worst
                  << " (n=" << worst_n << ")\n";
        check(worst < 1e-10,
              "per-n Numerov identity holds to LU precision (worst=" +
              std::to_string(worst) + ")");
    }

    // ----------------------------------------------------------------------
    // (9) HIGHER l_cont: rerun at l_cont=6 and spot-check gold standard.
    //     Ensures the code works beyond the tiny toy problem and that no
    //     size-specific bug hides at N_total=294.
    // ----------------------------------------------------------------------
    std::cout << "\n--- (9) higher l_cont smoke test (l_cont=6, N_total=294) ---\n";
    {
        Parameters p6 = params;
        p6.l_max_continuum = 6;
        p6.validate();
        SetupBundle b6 = WavefunctionSetup::prepare(p6, data, /*energy=*/0.5);

        Potentials pot6(p6);
        pot6.build(data, StorageMode::MEMORY, "", /*verbose=*/false);

        ExchangeCoupling EC6(b6.G_coeff, b6.params.n_mu, b6.params.n_sigma,
                             b6.params.n_occ, data.rmin, data.dr);

        SchurInverter SI6(b6.params, pot6, &EC6, &b6.chi);
        SchurInverter::Config si6 = si_cfg;
        si6.checkpoint_dir = "./checkpoints/sinv_frp_lc6";
        si6.verbose = false;
        std::filesystem::remove_all(si6.checkpoint_dir);
        SI6.build(si6);

        WInverseOperator WI6(b6.params, SI6, &EC6, &b6.chi, si6.W_min);

        ForwardRPropagator FRP6(b6.params, pot6, WI6);
        ForwardRPropagator::Config c6 = frp_cfg;
        c6.checkpoint_dir = "./checkpoints/rinv_frp_lc6";
        c6.try_load_checkpoint = false;
        c6.save_checkpoint = false;
        c6.verbose = false;
        std::filesystem::remove_all(c6.checkpoint_dir);
        auto tt = std::chrono::steady_clock::now();
        FRP6.run(c6);
        double dtt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tt).count();
        std::cout << "   l_cont=6 build: " << std::fixed << std::setprecision(3)
                  << dtt << " s  (N_total=" << (b6.params.n_mu + b6.params.n_occ * b6.params.n_sigma)
                  << ")\n";

        // Sanity checks.
        bool all_finite = true;
        for (std::size_t n = 0; n < b6.params.n_grid; ++n)
            if (!FRP6.get(n).allFinite()) { all_finite = false; break; }
        check(all_finite, "l_cont=6: all entries finite");

        // Per-n Numerov identity on a small sample (expensive at N_total=294).
        std::vector<int> l_sigma6(b6.params.n_sigma);
        for (int s = 0; s < b6.params.n_sigma; ++s) {
            int l, m; scatt::angular::idx_to_lm(s, l, m); l_sigma6[s] = l;
        }
        const int N_tot6 = b6.params.n_mu + b6.params.n_occ * b6.params.n_sigma;
        Eigen::MatrixXd I_tot6 = Eigen::MatrixXd::Identity(N_tot6, N_tot6);
        Eigen::MatrixXd U6(N_tot6, N_tot6);
        auto ws6 = WI6.make_workspace();
        double worst6 = 0.0;
        for (int n : {FRP6.n_start(), 100, 500, 1500, (int)b6.params.n_grid - 1}) {
            if (n < FRP6.n_start() || n >= (int)b6.params.n_grid) continue;
            WI6.apply_U(n, I_tot6, U6, ws6);
            const Eigen::MatrixXd& Rprev = FRP6.get((std::size_t)(n - 1));
            const Eigen::MatrixXd& Rn    = FRP6.get((std::size_t)n);
            Eigen::MatrixXd Id = Rn * (U6 - Rprev);
            worst6 = std::max(worst6, (Id - I_tot6).cwiseAbs().maxCoeff());
        }
        check(worst6 < 1e-10,
              "l_cont=6: per-n Numerov identity (worst=" + std::to_string(worst6) + ")");

        // Symmetry.
        double sym6 = 0.0;
        for (int n : {100, 500, 1500}) {
            const auto& Rn = FRP6.get((std::size_t)n);
            double a = (Rn - Rn.transpose()).cwiseAbs().maxCoeff();
            sym6 = std::max(sym6, a / std::max(Rn.cwiseAbs().maxCoeff(), 1e-30));
        }
        check(sym6 < 1e-12, "l_cont=6: symmetry (worst=" + std::to_string(sym6) + ")");
    }

    // ----------------------------------------------------------------------
    // (10) ENERGY SCAN: re-run at E = 1.0 Ha with same l_cont.
    //     Catches energy-dependent sign/phase bugs that might cancel at
    //     the fiducial E = 0.5 Ha.
    // ----------------------------------------------------------------------
    std::cout << "\n--- (10) energy scan: E = 1.0 Ha ---\n";
    {
        SetupBundle b_e = WavefunctionSetup::prepare(params, data, /*energy=*/1.0);

        ExchangeCoupling EC_e(b_e.G_coeff, b_e.params.n_mu, b_e.params.n_sigma,
                              b_e.params.n_occ, data.rmin, data.dr);

        SchurInverter SI_e(b_e.params, pot, &EC_e, &b_e.chi);
        SchurInverter::Config si_e = si_cfg;
        si_e.checkpoint_dir = "./checkpoints/sinv_frp_E1";
        si_e.verbose = false;
        std::filesystem::remove_all(si_e.checkpoint_dir);
        SI_e.build(si_e);

        WInverseOperator WI_e(b_e.params, SI_e, &EC_e, &b_e.chi, si_e.W_min);

        ForwardRPropagator FRP_e(b_e.params, pot, WI_e);
        ForwardRPropagator::Config ce = frp_cfg;
        ce.checkpoint_dir      = "./checkpoints/rinv_frp_E1";
        ce.try_load_checkpoint = false;
        ce.save_checkpoint     = false;
        ce.verbose             = false;
        std::filesystem::remove_all(ce.checkpoint_dir);
        FRP_e.run(ce);

        bool finite_e = true;
        for (std::size_t n = 0; n < b_e.params.n_grid; ++n)
            if (!FRP_e.get(n).allFinite()) { finite_e = false; break; }
        check(finite_e, "E=1.0: all entries finite");

        // Per-n Numerov identity at a small sample.
        Eigen::MatrixXd I_te = Eigen::MatrixXd::Identity(N_tot, N_tot);
        Eigen::MatrixXd U_e(N_tot, N_tot);
        auto ws_e = WI_e.make_workspace();
        double worst_e = 0.0;
        for (int n : {FRP_e.n_start(), 100, 1000, (int)Nr - 1}) {
            WI_e.apply_U(n, I_te, U_e, ws_e);
            const Eigen::MatrixXd& Rprev = FRP_e.get((std::size_t)(n - 1));
            const Eigen::MatrixXd& Rn    = FRP_e.get((std::size_t)n);
            worst_e = std::max(worst_e,
                (Rn * (U_e - Rprev) - I_te).cwiseAbs().maxCoeff());
        }
        check(worst_e < 1e-10,
              "E=1.0: per-n Numerov identity (worst=" + std::to_string(worst_e) + ")");
    }

    // ----------------------------------------------------------------------
    // (11) INVERTIBILITY OF R at the outer matching point.
    //     Step 5 (K-matrix) needs R = Rinv^{-1}. Verify Rinv_final is well-
    //     conditioned by computing R via LU and checking R · Rinv ≈ I.
    // ----------------------------------------------------------------------
    std::cout << "\n--- (11) Rinv_final is well-conditioned (Step 5 prerequisite) ---\n";
    {
        const auto& Rinv_N = FRP.rinv_final();
        Eigen::MatrixXd R_N = Rinv_N.partialPivLu().inverse();
        Eigen::MatrixXd Id = R_N * Rinv_N;
        double id_err = (Id - Eigen::MatrixXd::Identity(N_tot, N_tot))
                          .cwiseAbs().maxCoeff();
        check(id_err < 1e-9,
              "R · Rinv = I at outer boundary (err=" + std::to_string(id_err) + ")");

        // Condition number via self-adjoint eigendecomposition of Rᵀ R.
        // NOTE: BDCSVD's 1-arg constructor does NOT populate singular values
        // correctly under Eigen's MKL LAPACKE path on some oneAPI versions
        // (yields NaN).  We use SelfAdjointEigenSolver on Rinvᵀ·Rinv, which
        // gives the squared singular values directly and is always correct.
        Eigen::MatrixXd M = Rinv_N.transpose() * Rinv_N;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M, Eigen::EigenvaluesOnly);
        const double s2_max = es.eigenvalues().tail<1>()(0);
        const double s2_min = std::max(es.eigenvalues().head<1>()(0), 1e-300);
        const double cond   = std::sqrt(s2_max / s2_min);
        std::cout << "   Rinv_final condition number = " << std::scientific
                  << std::setprecision(3) << cond << "\n";
        check(std::isfinite(cond) && cond < 1e12,
              "Rinv_final well-conditioned (cond < 1e12)");
    }

    // ----------------------------------------------------------------------
    // (12) Benchmark
    // ----------------------------------------------------------------------
    std::cout << "\n--- (12) Benchmark ---\n";
    {
        // Already have the MEMORY build time from the initial run log.
        // Do one timed DISK run + print per-n time breakdown.
        ForwardRPropagator FRP_t(b.params, pot, WI);
        ForwardRPropagator::Config ct = frp_cfg;
        ct.checkpoint_dir       = "./checkpoints/rinv_bench";
        ct.try_load_checkpoint  = false;
        ct.save_checkpoint      = false;
        ct.verbose              = false;
        std::filesystem::remove_all(ct.checkpoint_dir);

        auto t0 = std::chrono::steady_clock::now();
        FRP_t.run(ct);
        double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "   MEMORY build: " << std::fixed << std::setprecision(3)
                  << dt << " s  ("
                  << std::setprecision(0) << (Nr / dt) << " pts/s, "
                  << std::setprecision(2) << (dt / Nr * 1e3) << " ms/pt, "
                  << "N_total=" << N_tot << ")\n";
    }

    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL")
              << " (" << fails << " failed)\n";
    return fails == 0 ? 0 : 1;
}
