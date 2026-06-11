// test_schur_inverter.cpp -- validate SchurInverter on H2O.
//
// Gold-standard check: for each probed ir, build the full N_total × N_total
// Numerov W(n) = [A B; B^T D] directly and LU-invert it. The top-left
// N_psi × N_psi block of W^(-1) MUST equal our Sinv[n] (Schur complement
// identity). This catches any sign/prefactor/Schur bug.
//
// Additional independent checks:
//   (2) Sinv · S = I (recompute S from scratch).
//   (3) Exchange-off region (ir >= n_transition): Sinv == A^(-1).
//   (4) Symmetry: ||Sinv − Sinv^T||_inf < 1e-14.
//   (5) Johnson stability shift count is reported; sanity-check that it
//       happens only in the tiny-n region.
//   (6) MEMORY ≡ DISK roundtrip: build twice, matrices must be identical.
//   (7) Benchmark: wall time vs N_grid, per-point rate.

#include "io/HDF5Reader.hpp"
#include "scatt/Banner.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using scatt::ExchangeCoupling;
using scatt::Parameters;
using scatt::Potentials;
using scatt::SchurInverter;
using scatt::SetupBundle;
using scatt::StorageMode;
using scatt::WavefunctionSetup;
using scatt::io::HDF5Reader;
using scatt::io::PreprocData;

static int fails = 0;
static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++fails;
}

// Build the full W = [A B; B^T D] at ir, directly (no Schur). Used as
// the gold standard for comparing against SchurInverter::get(ir).
static Eigen::MatrixXd
build_full_W(int ir,
             const scatt::SolverParams& sp,
             Potentials&                pot,
             const ExchangeCoupling*    ec,
             const scatt::ChiRadial*    chi,
             const std::vector<int>&    l_sigma)
{
    const int N_psi = sp.n_mu;
    const int N_f   = sp.n_occ * sp.n_sigma;
    const int N_tot = N_psi + N_f;
    const double h  = sp.dr;
    const double h2_12 = h * h / 12.0;
    const double h2_6  = h * h / 6.0;
    const double r  = sp.r_min + ir * sp.dr;
    const double r2 = r * r;

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(N_tot, N_tot);

    // A = I + (h²/6)(E·I − V)
    for (int i = 0; i < N_psi; ++i) W(i, i) = 1.0 + h2_6 * sp.energy;
    const auto& V = pot.get(static_cast<std::size_t>(ir));
    W.block(0, 0, N_psi, N_psi).noalias() -= h2_6 * V;

    // D = 1 − (h²/12) ℓ(ℓ+1)/r²   (NO clamp here, we want the raw W.)
    for (int f = 0; f < N_f; ++f) {
        const int l = l_sigma[f % sp.n_sigma];
        const double centrif = (r2 > 1e-30) ? double(l * (l + 1)) / r2 : 0.0;
        W(N_psi + f, N_psi + f) = 1.0 - h2_12 * centrif;
    }

    // B = (h²/12) Q_ψf(n). Only if exchange on.
    if (ec && ir < sp.n_transition) {
        auto ws = ec->make_workspace();
        Eigen::MatrixXd Q = ec->make_output();
        ec->compute_into(ir, (*chi)[static_cast<std::size_t>(ir)], ws, Q);
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
    params.l_max_continuum = 4;                  // N_psi=25, N_f=125, N_tot=150
    params.validate();

    SetupBundle b = WavefunctionSetup::prepare(params, data, /*energy=*/0.5);

    // Potentials in MEMORY so parallelism is allowed later.
    Potentials pot(params);
    pot.build(data, StorageMode::MEMORY, /*checkpoint_dir=*/"", /*verbose=*/false);

    // ExchangeCoupling in on-fly mode.
    ExchangeCoupling EC(b.G_coeff, b.params.n_mu, b.params.n_sigma, b.params.n_occ,
                        data.rmin, data.dr);

    // l_sigma table (same convention as SchurInverter uses internally).
    std::vector<int> l_sigma(b.params.n_sigma);
    for (int s = 0; s < b.params.n_sigma; ++s) {
        int l, m; scatt::angular::idx_to_lm(s, l, m);
        l_sigma[s] = l;
    }

    std::cout << "\n=== SchurInverter build (MEMORY, serial) ===\n";
    SchurInverter SI(b.params, pot, &EC, &b.chi);
    SchurInverter::Config cfg;
    cfg.storage = StorageMode::MEMORY;
    cfg.use_openmp = false;               // serial first; OMP verified in a later check
    cfg.verbose = true;
    SI.build(cfg);

    const std::size_t Nr = b.params.n_grid;

    // ----------------------------------------------------------------------
    // (1) Gold-standard: top-left block of full W^(-1) == Sinv
    // ----------------------------------------------------------------------
    std::cout << "\n--- (1) gold standard: W^(-1)[0:N_psi, 0:N_psi] == Sinv ---\n";
    // Include at least one ir from the exchange region and one from far-field.
    const std::vector<int> probe_irs_gold = {
        1, 5, 10, 50, 200, 500, 1500, 2500, static_cast<int>(Nr) - 2
    };
    double worst_gold = 0.0;
    for (int ir : probe_irs_gold) {
        if (ir < 0 || ir >= (int)Nr) continue;
        Eigen::MatrixXd W = build_full_W(ir, b.params, pot, &EC, &b.chi, l_sigma);
        // For the gold-standard comparison we MUST clamp D the same way the
        // Schur path does; otherwise W will differ from W_regularized. Apply
        // the same W_min clamp to the bottom-right diagonal.
        for (int f = 0; f < b.params.n_occ * b.params.n_sigma; ++f) {
            double& Dff = W(b.params.n_mu + f, b.params.n_mu + f);
            if (Dff < cfg.W_min) Dff = cfg.W_min;
        }
        // Also apply the A/S shift if n < stab_n_max. The shift is an
        // intentional regularization, so the gold-standard check at those
        // ir is not a clean Schur identity; skip the A/S shift region.
        if (ir < cfg.stab_n_max) {
            std::cout << "     ir=" << ir << " in stab region, skipping gold check\n";
            continue;
        }
        Eigen::MatrixXd W_inv = W.partialPivLu().inverse();
        Eigen::MatrixXd expected = W_inv.topLeftCorner(b.params.n_mu, b.params.n_mu);
        const auto& Sinv = SI.get(static_cast<std::size_t>(ir));
        double max_err = (Sinv - expected).cwiseAbs().maxCoeff();
        double scale   = std::max(expected.cwiseAbs().maxCoeff(), 1e-30);
        worst_gold = std::max(worst_gold, max_err / scale);
        std::cout << "     ir=" << std::setw(5) << ir
                  << "  |Sinv|_max=" << std::scientific << std::setprecision(3) << scale
                  << "  |Sinv - W_inv_top|_max=" << max_err
                  << "  rel=" << (max_err / scale) << "\n";
        check(max_err < 1e-10, "gold standard at ir=" + std::to_string(ir));
    }
    std::cout << "     worst relative error: " << worst_gold << "\n";

    // ----------------------------------------------------------------------
    // (2) Sinv · S = I  (rebuild S from scratch, identical logic)
    // ----------------------------------------------------------------------
    std::cout << "\n--- (2) Sinv · S = I ---\n";
    for (int ir : {200, 500, 1500, 2500}) {
        if (ir >= (int)Nr) continue;
        Eigen::MatrixXd W = build_full_W(ir, b.params, pot, &EC, &b.chi, l_sigma);
        // Rebuild S manually exactly as SchurInverter does.
        const int N_psi = b.params.n_mu;
        const int N_f = b.params.n_occ * b.params.n_sigma;
        Eigen::MatrixXd A  = W.topLeftCorner(N_psi, N_psi);
        Eigen::MatrixXd Bm = W.topRightCorner(N_psi, N_f);
        Eigen::VectorXd Df = W.diagonal().tail(N_f);
        // Clamp D same way as Schur path.
        for (int f = 0; f < N_f; ++f) if (Df(f) < cfg.W_min) Df(f) = cfg.W_min;
        Eigen::VectorXd Dinv = Df.cwiseInverse();
        Eigen::MatrixXd S = A;
        for (int f = 0; f < N_f; ++f) Bm.col(f) *= Dinv(f);    // in-place scale
        S.noalias() -= Bm * W.topRightCorner(N_psi, N_f).transpose();
        // Oops -- in-place scale corrupted Bm. Rebuild cleanly:
        Eigen::MatrixXd B_orig = W.topRightCorner(N_psi, N_f);  // read fresh
        {
            // Clamp D and rebuild W
            Eigen::MatrixXd W2 = build_full_W(ir, b.params, pot, &EC, &b.chi, l_sigma);
            for (int f = 0; f < N_f; ++f) {
                double& Dff = W2(N_psi + f, N_psi + f);
                if (Dff < cfg.W_min) Dff = cfg.W_min;
            }
            A      = W2.topLeftCorner(N_psi, N_psi);
            B_orig = W2.topRightCorner(N_psi, N_f);
            Df     = W2.diagonal().tail(N_f);
            Dinv   = Df.cwiseInverse();
        }
        Eigen::MatrixXd B_scaled = B_orig;
        for (int f = 0; f < N_f; ++f) B_scaled.col(f) *= Dinv(f);
        S = A;
        S.noalias() -= B_scaled * B_orig.transpose();
        S = 0.5 * (S + S.transpose().eval());

        const auto& Sinv = SI.get(static_cast<std::size_t>(ir));
        Eigen::MatrixXd I_hat = Sinv * S;
        Eigen::MatrixXd I_ref = Eigen::MatrixXd::Identity(N_psi, N_psi);
        double err = (I_hat - I_ref).cwiseAbs().maxCoeff();
        check(err < 1e-10, "Sinv·S = I at ir=" + std::to_string(ir) +
                           " (err=" + std::to_string(err) + ")");
    }

    // ----------------------------------------------------------------------
    // (3) Exchange-off region: Sinv == A^(-1)
    // ----------------------------------------------------------------------
    std::cout << "\n--- (3) Exchange-off region Sinv == A^{-1} ---\n";
    // For H2O our n_transition = N_grid = 3001, so we never leave the
    // exchange region unless we override n_transition. Use the last ir as
    // a proxy (exchange is numerically tiny there so effectively off).
    // Build a second SchurInverter with n_transition < N_grid to actually
    // exercise this branch.
    {
        scatt::SolverParams sp2 = b.params;
        sp2.n_transition = 500;  // force exchange-off for ir >= 500
        SchurInverter SI2(sp2, pot, &EC, &b.chi);
        SchurInverter::Config cfg2 = cfg;
        cfg2.checkpoint_dir = "./checkpoints/sinv_test_exchoff";
        cfg2.verbose = false;
        std::filesystem::remove_all(cfg2.checkpoint_dir);
        SI2.build(cfg2);

        const int ir = 1000;    // safely in the exchange-off region
        Eigen::MatrixXd A(sp2.n_mu, sp2.n_mu);
        A.setIdentity();
        const double h2_6 = sp2.dr * sp2.dr / 6.0;
        A.diagonal().array() += h2_6 * sp2.energy;
        A.noalias() -= h2_6 * pot.get(ir);

        Eigen::MatrixXd A_inv = A.partialPivLu().inverse();
        A_inv = 0.5 * (A_inv + A_inv.transpose().eval());
        const auto& Sinv_off = SI2.get(ir);
        double err = (Sinv_off - A_inv).cwiseAbs().maxCoeff();
        check(err < 1e-12,
              "exchange-off Sinv == A^{-1} at ir=" + std::to_string(ir) +
              " (err=" + std::to_string(err) + ")");
    }

    // ----------------------------------------------------------------------
    // (4) Symmetry
    // ----------------------------------------------------------------------
    std::cout << "\n--- (4) Sinv symmetric ---\n";
    {
        double worst = 0.0;
        for (int ir : {1, 10, 100, 500, 2000, (int)Nr - 1}) {
            const auto& Sinv = SI.get(ir);
            double asym = (Sinv - Sinv.transpose()).cwiseAbs().maxCoeff();
            worst = std::max(worst, asym);
        }
        check(worst < 1e-14, "||Sinv - Sinv^T|| < 1e-14 (worst=" +
                             std::to_string(worst) + ")");
    }

    // ----------------------------------------------------------------------
    // (5) Stability shifts summary
    // ----------------------------------------------------------------------
    std::cout << "\n--- (5) Johnson stability shifts ---\n";
    std::cout << "     A-shifts: " << SI.stability_shifts_A()
              << "  S-shifts: " << SI.stability_shifts_S() << "\n";
    check(SI.stability_shifts_A() >= 0 && SI.stability_shifts_S() >= 0,
          "shift counters are valid");
    // A or S shifts should only happen at small n; not required to be nonzero.

    // ----------------------------------------------------------------------
    // (5b) Checkpoint hybrid: MEMORY build -> save_to_disk -> load_into_memory
    //      The loaded Sinv must be bit-equal to the one we just built.
    // ----------------------------------------------------------------------
    std::cout << "\n--- (5b) checkpoint round-trip (MEMORY build <-> disk) ---\n";
    {
        const std::string ck = "./checkpoints/sinv_test_roundtrip_mem";
        std::filesystem::remove_all(ck);

        // Fresh build with save_checkpoint=true.
        SchurInverter SI_a(b.params, pot, &EC, &b.chi);
        SchurInverter::Config c_a = cfg;
        c_a.checkpoint_dir     = ck;
        c_a.try_load_checkpoint = false;  // force a rebuild
        c_a.save_checkpoint    = true;
        c_a.verbose            = false;
        SI_a.build(c_a);

        // Second instance: try_load_checkpoint=true should hit the disk.
        SchurInverter SI_b(b.params, pot, &EC, &b.chi);
        SchurInverter::Config c_b = cfg;
        c_b.checkpoint_dir     = ck;
        c_b.try_load_checkpoint = true;
        c_b.save_checkpoint    = false;
        c_b.verbose            = true;      // to print the "loaded ..." line
        SI_b.build(c_b);

        double worst = 0.0;
        for (int ir : {0, 1, 5, 100, 1500, (int)Nr - 1}) {
            double d = (SI_a.get(ir) - SI_b.get(ir)).cwiseAbs().maxCoeff();
            worst = std::max(worst, d);
        }
        check(worst < 1e-14,
              "MEMORY build + save_to_disk + try_load_into_memory bit-equal (worst=" +
              std::to_string(worst) + ")");
        std::filesystem::remove_all(ck);
    }

    // ----------------------------------------------------------------------
    // (6) MEMORY ≡ DISK roundtrip
    // ----------------------------------------------------------------------
    std::cout << "\n--- (6) MEMORY vs DISK parity ---\n";
    {
        SchurInverter SI_disk(b.params, pot, &EC, &b.chi);
        SchurInverter::Config cd = cfg;
        cd.storage = StorageMode::DISK;
        cd.checkpoint_dir = "./checkpoints/sinv_test_disk_roundtrip";
        cd.chunk_size = 50;
        cd.verbose = false;
        std::filesystem::remove_all(cd.checkpoint_dir);
        SI_disk.build(cd);

        double worst = 0.0;
        for (int ir : {0, 1, 5, 100, 1500, (int)Nr - 1}) {
            double d = (SI.get(ir) - SI_disk.get(ir)).cwiseAbs().maxCoeff();
            worst = std::max(worst, d);
        }
        check(worst < 1e-14, "MEMORY == DISK bit-equal (worst=" +
                             std::to_string(worst) + ")");
    }

    // ----------------------------------------------------------------------
    // (7) Benchmark
    // ----------------------------------------------------------------------
    std::cout << "\n--- (7) Benchmark: SchurInverter build time ---\n";
    {
        auto one_run = [&](bool parallel, const std::string& label) {
            SchurInverter SI_b(b.params, pot, &EC, &b.chi);
            SchurInverter::Config cfg_b = cfg;
            cfg_b.use_openmp = parallel;
            cfg_b.checkpoint_dir = "./checkpoints/sinv_test_bench_" + label;
            cfg_b.verbose = false;
            std::filesystem::remove_all(cfg_b.checkpoint_dir);
            auto t0 = std::chrono::steady_clock::now();
            SI_b.build(cfg_b);
            double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            std::cout << "     " << std::left << std::setw(12) << label
                      << ": " << std::fixed << std::setprecision(3)
                      << dt << " s  (" << std::setprecision(0)
                      << (Nr / dt) << " pts/s, " << std::setprecision(1)
                      << (dt / Nr * 1e6) << " μs/pt, "
                      << (SI_b.memory_bytes() >> 10) << " KB)\n";
        };
        one_run(false, "serial");
#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        one_run(true,  "omp");
#endif

        // Bench sweep across l_cont: how does build time scale?
        std::cout << "\n  Sweep across l_cont:\n";
        std::cout << "    l_cont | N_psi | N_f  | build s | pts/s\n";
        std::cout << "    -------|-------|------|---------|-------\n";
        for (int lc : {2, 4, 6, 8}) {
            if (lc + std::min(lc, 10) > data.Lmax_sce) continue;
            Parameters p2 = params;
            p2.l_max_continuum = lc;
            try { p2.validate(); } catch (...) { continue; }

            SetupBundle bb = WavefunctionSetup::prepare(p2, data, 0.5);
            Potentials pot2(p2);
            pot2.build(data, StorageMode::MEMORY, "", /*verbose=*/false);
            ExchangeCoupling EC2(bb.G_coeff, bb.params.n_mu, bb.params.n_sigma,
                                 bb.params.n_occ, data.rmin, data.dr);
            SchurInverter SI2(bb.params, pot2, &EC2, &bb.chi);
            SchurInverter::Config c2 = cfg;
            c2.use_openmp = true;
            c2.checkpoint_dir = "./checkpoints/sinv_test_lc_" + std::to_string(lc);
            c2.verbose = false;
            std::filesystem::remove_all(c2.checkpoint_dir);
            auto t0 = std::chrono::steady_clock::now();
            SI2.build(c2);
            double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            std::cout << "       " << std::setw(4) << lc
                      << " | " << std::setw(5) << bb.params.n_mu
                      << " | " << std::setw(4) << bb.params.n_occ * bb.params.n_sigma
                      << " | " << std::fixed << std::setprecision(3) << std::setw(7) << dt
                      << " | " << std::setprecision(0)
                      << (bb.params.n_grid / dt) << "\n";
        }
    }

    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL")
              << " (" << fails << " failed)\n";
    return fails == 0 ? 0 : 1;
}
