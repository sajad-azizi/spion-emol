// test_backprop_v_outer_cache.cpp
//
// Gate test for the BP "pre-cache V_N + release pot read buffer"
// optimisation in main.cpp (the production --check-residual=OFF path).
//
// Two BP runs on identical inputs:
//
//   A. Reference: legacy path.
//        - cfg.cached_V_outer = nullptr
//        - pot's chunk read cache stays alive
//        - compute_Z_at_outer_boundary_ calls pot_.get(N) directly
//
//   B. Optimised: production path.
//        - V_N is deep-copied from pot.get(N) BEFORE BP starts
//        - pot.release_read_buffer() is called (drops the chunk cache)
//        - cfg.cached_V_outer = &V_N_cached
//        - BP uses the cached copy; pot_ is never touched during run()
//
// Then compare every ψ(n) for n in [0, n_keep_hi] AND every (A, B)
// asymptotic-fit matrix.  Required: bit-identical (literal max|diff| == 0).
//
// Rationale: V_N_cached is a deep copy of pot.get(N).  The bytes are
// byte-identical to read_buffer_[offset_N].  Both paths feed those bytes to
// the same Eigen GEMM (`-= h2_6 * V_N * psi_boundary`) producing the same
// dgemm output.  Every downstream step is deterministic.  Therefore the
// expected diff is 0.0 -- we are testing that there is no path that breaks
// this guarantee (e.g. accidentally using a stale read_buffer_ reference
// after release).

#include "io/HDF5Reader.hpp"
#include "scatt/AsymptoticAmplitudes.hpp"
#include "scatt/BackPropagator.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/ForwardRPropagator.hpp"
#include "scatt/KMatrixExtractor.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Dense>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

using namespace scatt;

static int g_fail = 0;
static void check(bool cond, const std::string& what) {
    if (!cond) { std::cerr << "FAIL  " << what << "\n"; ++g_fail; }
    else       { std::cout << "ok    " << what << "\n"; }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <preproc.h5>\n";
        return 2;
    }

    io::HDF5Reader reader(argv[1]);
    io::PreprocData data = reader.load_all_except_V_H();

    Parameters params;
    params.l_max_continuum = 6;
    params.r_min           = data.rmin;
    params.dr              = data.dr;
    params.N_grid          = data.Nr;
    params.Lmax_sce        = data.Lmax_sce;
    data.V_H               = reader.load_V_H(params.n_exp());
    params.validate();

    const double E_kin = 0.5;
    auto bundle = WavefunctionSetup::prepare(params, data, E_kin);
    ExchangeCoupling EC(bundle.G_coeff, bundle.params.n_mu, bundle.params.n_sigma,
                        bundle.params.n_occ, data.rmin, data.dr);

    // Run pot in DISK mode -- this is the production case where the
    // optimisation actually saves memory.  release_read_buffer() is a no-op
    // for MEMORY mode; we want to exercise the DISK path so we can also
    // verify that pot.get() AFTER release works (it should lazily re-read).
    const std::string pot_dir = "./checkpoints/test_v_outer_cache_pot";
    std::filesystem::remove_all(pot_dir);
    Potentials pot(params);
    pot.build(data, StorageMode::DISK, pot_dir, /*verbose=*/false);

    SchurInverter SI(bundle.params, pot, &EC, &bundle.chi);
    SchurInverter::Config sic;
    sic.verbose = false; sic.try_load_checkpoint = false; sic.save_checkpoint = false;
    sic.storage = StorageMode::MEMORY;     // small; keep test fast
    SI.build(sic);

    WInverseOperator WI(bundle.params, SI, &EC, &bundle.chi, sic.W_min);
    ForwardRPropagator FRP(bundle.params, pot, WI);
    ForwardRPropagator::Config frc;
    frc.verbose = false; frc.try_load_checkpoint = false; frc.save_checkpoint = false;
    frc.storage = StorageMode::MEMORY;
    FRP.run(frc);

    KMatrixExtractor KME(bundle.params, FRP);
    auto res_K = KME.extract();
    auto bc = KMatrixExtractor::make_psi_boundary(bundle.params, res_K.K_matrix);

    const int N      = static_cast<int>(bundle.params.n_grid) - 1;
    const int N_psi  = bundle.params.n_mu;

    // ---- Reference: legacy path (cached_V_outer = nullptr) ----
    BackPropagator BP_ref(bundle.params, pot, FRP, WI);
    BackPropagator::Config c_ref;
    c_ref.n_keep_lo = 0;
    c_ref.n_keep_hi = N;
    c_ref.n_asym    = 200;
    c_ref.compute_f = false;
    c_ref.psi_storage = StorageMode::MEMORY;
    c_ref.try_load_checkpoint = false;
    c_ref.save_checkpoint     = false;
    c_ref.verbose             = false;
    c_ref.cached_V_outer      = nullptr;       // <-- legacy
    BP_ref.run(bc, c_ref);

    AsymptoticAmplitudes AA_ref(bundle.params, BP_ref);
    AsymptoticAmplitudes::Config aac; aac.verbose = false;
    auto AB_ref = AA_ref.extract(aac);

    // ---- Optimised: deep-copy V_N, release pot's read cache, pass pointer ----
    Eigen::MatrixXd V_N_cached = pot.get(static_cast<std::size_t>(N));   // deep copy
    pot.release_read_buffer();      // exercises the new code path

    // After release, pot's chunk cache is empty.  A subsequent get() should
    // lazily re-read from disk (we don't expect BP to call it, but it must
    // still work).
    {
        const Eigen::MatrixXd& V_again = pot.get(static_cast<std::size_t>(N));
        const double diff = (V_again - V_N_cached).cwiseAbs().maxCoeff();
        check(diff == 0.0,
              "pot.get(N) after release_read_buffer returns identical bytes "
              "(re-read from disk)");
        // Release again so BP is run with the cache empty.
        pot.release_read_buffer();
    }

    BackPropagator BP_opt(bundle.params, pot, FRP, WI);
    BackPropagator::Config c_opt = c_ref;
    c_opt.cached_V_outer = &V_N_cached;        // <-- new path
    BP_opt.run(bc, c_opt);

    AsymptoticAmplitudes AA_opt(bundle.params, BP_opt);
    auto AB_opt = AA_opt.extract(aac);

    // ---- Bit-identical psi at every grid index in the keep range ----
    int n_psi_diff = 0;
    double max_psi_diff = 0.0;
    for (int n = 0; n <= N; ++n) {
        const Eigen::MatrixXd& P_ref = BP_ref.get_psi(static_cast<std::size_t>(n));
        const Eigen::MatrixXd& P_opt = BP_opt.get_psi(static_cast<std::size_t>(n));
        if (P_ref.rows() != N_psi || P_ref.cols() != N_psi
            || P_opt.rows() != N_psi || P_opt.cols() != N_psi) {
            std::cerr << "shape mismatch at n=" << n << "\n"; return 1;
        }
        const double d = (P_ref - P_opt).cwiseAbs().maxCoeff();
        if (d != 0.0) ++n_psi_diff;
        max_psi_diff = std::max(max_psi_diff, d);
    }
    check(n_psi_diff == 0,
          "psi(n) bit-identical at every n in [0," + std::to_string(N) +
          "]  (worst diff = " + std::to_string(max_psi_diff) + ")");

    // ---- Bit-identical asymptotic A, B ----
    {
        const double dA = (AB_ref.A - AB_opt.A).cwiseAbs().maxCoeff();
        const double dB = (AB_ref.B - AB_opt.B).cwiseAbs().maxCoeff();
        check(dA == 0.0,
              "A matrix bit-identical (got " + std::to_string(dA) + ")");
        check(dB == 0.0,
              "B matrix bit-identical (got " + std::to_string(dB) + ")");
    }

    // Cleanup
    std::filesystem::remove_all(pot_dir);

    std::cout << "\n" << (g_fail == 0 ? "PASS" : "FAIL")
              << " backprop_v_outer_cache  (" << g_fail << " failures)\n";
    return g_fail == 0 ? 0 : 1;
}
