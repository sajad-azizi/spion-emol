// test_asymptotic_amplitudes.cpp -- accuracy checks for A, B, K, S.

#include "io/HDF5Reader.hpp"
#include "scatt/AsymptoticAmplitudes.hpp"
#include "scatt/BackPropagator.hpp"
#include "scatt/Banner.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/ForwardRPropagator.hpp"
#include "scatt/KMatrixExtractor.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

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

using namespace scatt;

static int fails = 0;
static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++fails;
}

// Full pipeline up to ψ, returning (bundle, pot, FRP, KME result, BP).
struct Pipeline {
    SetupBundle          bundle;
    Potentials*          pot = nullptr;
    ExchangeCoupling*    ec  = nullptr;
    SchurInverter*       si  = nullptr;
    WInverseOperator*    wi  = nullptr;
    ForwardRPropagator*  frp = nullptr;
    ScatteringResult     kme_result;
    BackPropagator*      bp  = nullptr;
};

static void run_pipeline(Pipeline& p, const Parameters& params,
                         const io::PreprocData& data, double energy,
                         const std::string& tag)
{
    p.bundle = WavefunctionSetup::prepare(params, data, energy);
    p.pot = new Potentials(params);
    p.pot->build(data, StorageMode::MEMORY, "", false,
                 /*try_load=*/false, /*save=*/false);

    p.ec  = new ExchangeCoupling(p.bundle.G_coeff, p.bundle.params.n_mu,
                                 p.bundle.params.n_sigma, p.bundle.params.n_occ,
                                 data.rmin, data.dr);
    p.si  = new SchurInverter(p.bundle.params, *p.pot, p.ec, &p.bundle.chi);
    SchurInverter::Config sic; sic.verbose = false;
    sic.try_load_checkpoint = false; sic.save_checkpoint = false;
    sic.checkpoint_dir = "./checkpoints/aa_sinv_" + tag;
    std::filesystem::remove_all(sic.checkpoint_dir);
    p.si->build(sic);

    p.wi  = new WInverseOperator(p.bundle.params, *p.si, p.ec, &p.bundle.chi,
                                 sic.W_min);
    p.frp = new ForwardRPropagator(p.bundle.params, *p.pot, *p.wi);
    ForwardRPropagator::Config frc; frc.verbose = false;
    frc.try_load_checkpoint = false; frc.save_checkpoint = false;
    frc.checkpoint_dir = "./checkpoints/aa_rinv_" + tag;
    std::filesystem::remove_all(frc.checkpoint_dir);
    p.frp->run(frc);

    KMatrixExtractor kme(p.bundle.params, *p.frp);
    p.kme_result = kme.extract();
    auto bc = KMatrixExtractor::make_psi_boundary(p.bundle.params,
                                                   p.kme_result.K_matrix);

    p.bp  = new BackPropagator(p.bundle.params, *p.pot, *p.frp, *p.wi);
    BackPropagator::Config bpc;
    bpc.n_keep_lo = 0;
    bpc.n_keep_hi = static_cast<int>(p.bundle.params.n_grid) - 1;
    bpc.compute_f = false;
    bpc.psi_storage = StorageMode::MEMORY;
    bpc.try_load_checkpoint = false; bpc.save_checkpoint = false;
    bpc.verbose = false;
    bpc.checkpoint_dir = "./checkpoints/aa_psi_" + tag;
    std::filesystem::remove_all(bpc.checkpoint_dir);
    p.bp->run(bc, bpc);
}
static void teardown(Pipeline& p) {
    delete p.bp; delete p.frp; delete p.wi; delete p.si; delete p.ec; delete p.pot;
    p.bp = nullptr; p.frp = nullptr; p.wi = nullptr; p.si = nullptr; p.ec = nullptr; p.pot = nullptr;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <h2o_hdf5>\n"; return 2; }
    print_la_banner();

    io::HDF5Reader reader(argv[1]);
    auto data = reader.load_all();

    Parameters params;
    params.r_min           = data.rmin;
    params.dr              = data.dr;
    params.N_grid          = data.Nr;
    params.Lmax_sce        = data.Lmax_sce;
    params.l_max_continuum = 4;
    params.validate();

    Pipeline P;
    run_pipeline(P, params, data, 0.5, "E05");
    const int N_psi = P.bundle.params.n_mu;

    std::cout << "\n=== AsymptoticAmplitudes extract (default window) ===\n";
    AsymptoticAmplitudes AA(P.bundle.params, *P.bp);
    AsymptoticAmplitudes::Config cfg;
    cfg.verbose = true;
    auto res = AA.extract(cfg);

    // Tolerance note for H2O at dr=0.005, r_max=15 au, l_cont=4:
    //   The fit is to ψ_μj = A·ĵ + B·ŷ (pure free asymptotic form), but
    //   the actual ψ obeys ψ'' + 2(E − V)·ψ = 0. H2O is polar, so its V
    //   has 1/r² tail (quadrupole) at r ≈ 14 au with V ~ 1e-3 Ha → the
    //   "unfit" contribution is O(V·ψ/k²) ~ 1e-3. This sets the floor
    //   for fit residual, K asymmetry, S non-unitarity.
    //   To get tighter agreement we'd need (a) larger r_max (push V
    //   tail further out), (b) fit with V-corrected basis, or (c) higher
    //   l_cont (more channels to absorb the anisotropy).

    // ------------------------------------------------------------------
    // (1) Fit residual small (relative to max |ψ|)
    // ------------------------------------------------------------------
    std::cout << "\n--- (1) fit residual small ---\n";
    check(res.fit_residual_rel < 1e-3,
          "max |ψ − (A·ĵ + B·ŷ)| / |ψ|_max < 1e-3 (got " +
          std::to_string(res.fit_residual_rel) + ")");

    // ------------------------------------------------------------------
    // (2) A ≈ I (regular normalisation)
    // ------------------------------------------------------------------
    std::cout << "\n--- (2) A matrix ≈ I (regular normalisation from BC) ---\n";
    check(res.A_minus_I_max < 1e-2,
          "‖A − I‖_∞ < 1e-2 (got " + std::to_string(res.A_minus_I_max) + ")");

    // ------------------------------------------------------------------
    // (3) K symmetric (physical indicator)
    // ------------------------------------------------------------------
    std::cout << "\n--- (3) K symmetric ---\n";
    check(res.K_symmetry_err < 1e-2,
          "‖K − Kᵀ‖_∞ < 1e-2 (got " + std::to_string(res.K_symmetry_err) + ")");

    // ------------------------------------------------------------------
    // (4) S unitary
    // ------------------------------------------------------------------
    std::cout << "\n--- (4) S unitary ---\n";
    check(res.S_unitarity_err < 1e-2,
          "‖S†S − I‖_∞ < 1e-2 (got " + std::to_string(res.S_unitarity_err) + ")");

    // ------------------------------------------------------------------
    // (5) S consistency via K: (I + iK)(I − iK)^{-1} must equal res.S
    // ------------------------------------------------------------------
    std::cout << "\n--- (5) S consistency (I+iK)(I−iK)^{-1} ---\n";
    {
        const std::complex<double> im(0.0, 1.0);
        Eigen::MatrixXcd I_c = Eigen::MatrixXcd::Identity(N_psi, N_psi);
        Eigen::MatrixXcd Kc  = res.K.cast<std::complex<double>>();
        Eigen::MatrixXcd S2  = (I_c + im * Kc) * (I_c - im * Kc).partialPivLu().inverse();
        double err = (S2 - res.S).cwiseAbs().maxCoeff();
        double scl = res.S.cwiseAbs().maxCoeff();
        // Exact algebraic identity now that K is raw B·A⁻¹.
        check(err / scl < 1e-12,
              "(I+iK)(I−iK)^{-1} == S from (A,B) (rel " +
              std::to_string(err / scl) + ")");
    }

    // ------------------------------------------------------------------
    // (6) K from fit agrees with Step 5 KMatrixExtractor
    // ------------------------------------------------------------------
    std::cout << "\n--- (6) K_fit vs K_KME (Step 5) ---\n";
    {
        double diff = (res.K - P.kme_result.K_matrix).cwiseAbs().maxCoeff();
        double scl  = res.K.cwiseAbs().maxCoeff();
        std::cout << "     |K_fit|_max=" << scl
                  << "   |K_fit − K_KME|_max=" << diff
                  << "   rel=" << (diff / scl) << "\n";
        // The two methods (LSQ fit vs two-point match at outer edge) have
        // different V-tail biases, so we allow 1% relative disagreement.
        check(diff / scl < 1e-2,
              "K_fit agrees with K_KME within 1%");
    }

    // ------------------------------------------------------------------
    // (7) Eigenphases consistent with arg(eig S) / 2 (mod π)
    // ------------------------------------------------------------------
    std::cout << "\n--- (7) eigenphases = ½·arg(eig S) ---\n";
    {
        Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eS(res.S);
        Eigen::VectorXcd ev = eS.eigenvalues();
        std::vector<double> phi_S(N_psi);
        for (int i = 0; i < N_psi; ++i) phi_S[i] = std::arg(ev(i));
        std::sort(phi_S.begin(), phi_S.end());
        std::vector<double> phi_K(N_psi);
        for (int i = 0; i < N_psi; ++i) {
            double t = 2.0 * res.eigenphases[i];
            while (t >  M_PI) t -= 2.0 * M_PI;
            while (t <= -M_PI) t += 2.0 * M_PI;
            phi_K[i] = t;
        }
        std::sort(phi_K.begin(), phi_K.end());
        double max_diff = 0.0;
        for (int i = 0; i < N_psi; ++i)
            max_diff = std::max(max_diff, std::abs(phi_S[i] - phi_K[i]));
        // Exact equality only if K is perfectly symmetric. Our raw K has
        // O(K_symmetry_err) asymmetry from the fit, so this identity holds
        // only up to that noise floor.
        check(max_diff < std::max(1e-9, 10.0 * res.K_symmetry_err),
              "arg(eig S) ≡ 2·δ_α within K-asymmetry noise "
              "(max diff " + std::to_string(max_diff) +
              ", floor " + std::to_string(res.K_symmetry_err) + ")");
    }

    // ------------------------------------------------------------------
    // (8) Stability vs fit window
    // ------------------------------------------------------------------
    std::cout << "\n--- (8) K stable vs window ---\n";
    {
        AsymptoticAmplitudes::Config c_narrow = cfg;
        c_narrow.n_fit_start = (int)P.bundle.params.n_grid - 150;
        c_narrow.verbose = false;
        auto r_narrow = AA.extract(c_narrow);

        AsymptoticAmplitudes::Config c_wide = cfg;
        c_wide.n_fit_start = (int)P.bundle.params.n_grid - 400;
        c_wide.verbose = false;
        auto r_wide = AA.extract(c_wide);

        const double scl = res.K.cwiseAbs().maxCoeff();
        double diff_n = (r_narrow.K - res.K).cwiseAbs().maxCoeff();
        double diff_w = (r_wide.K   - res.K).cwiseAbs().maxCoeff();
        std::cout << "     default window:    K scale=" << scl << "\n";
        std::cout << "     narrow (last 150): |ΔK|=" << diff_n
                  << "   rel=" << (diff_n / scl) << "\n";
        std::cout << "     wide   (last 400): |ΔK|=" << diff_w
                  << "   rel=" << (diff_w / scl) << "\n";
        check(diff_n / scl < 1e-2 && diff_w / scl < 1e-2,
              "K stable within 1% across fit windows");
    }

    // ------------------------------------------------------------------
    // (9) Higher-E stability (E = 4.0 au)
    // ------------------------------------------------------------------
    std::cout << "\n--- (9) E=4.0 au stability ---\n";
    {
        Pipeline P4;
        run_pipeline(P4, params, data, 4.0, "E4");
        AsymptoticAmplitudes AA4(P4.bundle.params, *P4.bp);
        AsymptoticAmplitudes::Config c4 = cfg;
        c4.verbose = true;
        auto r4 = AA4.extract(c4);
        check(r4.fit_residual_rel < 1e-3, "E=4.0: fit residual rel < 1e-3");
        check(r4.A_minus_I_max    < 1e-2, "E=4.0: ‖A−I‖ < 1e-2");
        check(r4.K_symmetry_err   < 1e-2, "E=4.0: ‖K−Kᵀ‖ < 1e-2");
        check(r4.S_unitarity_err  < 1e-2, "E=4.0: ‖S†S−I‖ < 1e-2");
        teardown(P4);
    }

    // ------------------------------------------------------------------
    // (10) Benchmark
    // ------------------------------------------------------------------
    std::cout << "\n--- (10) Benchmark ---\n";
    {
        const int REPS = 20;
        // warm-up
        AA.extract(cfg);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < REPS; ++i) { auto r = AA.extract(cfg); (void) r; }
        double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "   extract(): " << std::fixed << std::setprecision(3)
                  << (dt / REPS * 1e3) << " ms/call"
                  << "   (N_psi=" << N_psi << ")\n";
    }

    teardown(P);
    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL")
              << " (" << fails << " failed)\n";
    return fails == 0 ? 0 : 1;
}
