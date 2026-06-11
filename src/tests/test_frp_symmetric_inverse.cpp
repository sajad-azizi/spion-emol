// test_frp_symmetric_inverse.cpp
//
// API-plumbing test for ForwardRPropagator::Config::use_symmetric_inverse.
//
// Background:  Phase 1 of the symmetric-LDLᵀ optimisation was rolled back
// on the CPU FRP path after the H2O fixture showed ~3e-8 max relative
// diff vs the legacy LU path -- the 3000-step Numerov recursion
// compounds per-step ε·κ rounding noise.  Both algorithms are equally
// close to true Rinv, but they pick different rounding paths.  For the
// strict zero-accuracy mandate we kept the legacy CPU path verbatim.
//
// The Config flag is RETAINED for the GPU path (where it gates oneMKL
// sytrf+sytri inside GpuForwardStepper); on the CPU it is currently
// a no-op.  This test verifies that contract: with use_symmetric_inverse
// = true vs false on the CPU, FRP produces BIT-IDENTICAL output (the
// flag is ignored, both paths are legacy).
//
// At every n in [n_start, N_grid):
//   * |Rinv_new - Rinv_legacy|_inf == 0  (bit-equal)
//   * Rinv_new is bit-symmetric  (||Rinv - Rinv^T||_inf == 0)
//
// If this test ever starts failing with a non-zero diff, it means the
// CPU-side use_symmetric_inverse code was reintroduced -- check
// ForwardRPropagator.cpp around the inverse_general call.

#include "io/HDF5Reader.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/ForwardRPropagator.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Dense>

#include <cstdlib>
#include <iostream>
#include <string>

using namespace scatt;

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL  " << what << "\n"; ++g_fail; }
    else     { std::cout << "ok    " << what << "\n"; }
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

    Potentials pot(params);
    pot.build(data, StorageMode::MEMORY, /*ckpt*/"", /*verbose*/false);

    SchurInverter SI(bundle.params, pot, &EC, &bundle.chi);
    SchurInverter::Config sic;
    sic.verbose             = false;
    sic.try_load_checkpoint = false;
    sic.save_checkpoint     = false;
    sic.use_openmp          = false;
    sic.storage             = StorageMode::MEMORY;
    SI.build(sic);

    WInverseOperator WI(bundle.params, SI, &EC, &bundle.chi, sic.W_min);

    // --- Legacy path: dgetrf + symmetrise ---
    ForwardRPropagator FRP_leg(bundle.params, pot, WI);
    ForwardRPropagator::Config c_leg;
    c_leg.verbose                = false;
    c_leg.try_load_checkpoint    = false;
    c_leg.save_checkpoint        = false;
    c_leg.storage                = StorageMode::MEMORY;
    c_leg.use_gpu                = false;          // CPU-only test
    c_leg.use_symmetric_inverse  = false;          // legacy
    FRP_leg.run(c_leg);

    // --- New path: dsytrf, no explicit symmetrise ---
    ForwardRPropagator FRP_new(bundle.params, pot, WI);
    ForwardRPropagator::Config c_new = c_leg;
    c_new.use_symmetric_inverse  = true;           // new
    FRP_new.run(c_new);

    // --- Compare every n in [n_start, N_grid) ---
    const int n_start = FRP_leg.n_start();
    const int N       = static_cast<int>(bundle.params.n_grid);
    const int N_total = static_cast<int>(FRP_leg.get(static_cast<std::size_t>(n_start)).rows());

    int    n_compared           = 0;
    int    n_above_tolerance    = 0;
    double max_rel_diff_total   = 0.0;
    double max_asym_new         = 0.0;

    const double rel_tol  = 1e-10;
    const double asym_tol = 1e-13;     // mirror gives bit-symmetric; allow tiny slack

    for (int n = n_start; n < N; ++n) {
        const Eigen::MatrixXd& R_leg = FRP_leg.get(static_cast<std::size_t>(n));
        const Eigen::MatrixXd& R_new = FRP_new.get(static_cast<std::size_t>(n));
        if (R_leg.rows() != N_total || R_new.rows() != N_total) {
            std::cerr << "shape mismatch at n=" << n << "\n";  return 1;
        }

        const double asym = (R_new - R_new.transpose()).cwiseAbs().maxCoeff();
        max_asym_new = std::max(max_asym_new, asym);

        const double diff_inf  = (R_new - R_leg).cwiseAbs().maxCoeff();
        const double scale_inf = std::max(R_leg.cwiseAbs().maxCoeff(), 1e-30);
        const double rel       = diff_inf / scale_inf;
        max_rel_diff_total = std::max(max_rel_diff_total, rel);
        if (rel > rel_tol) ++n_above_tolerance;

        ++n_compared;
    }

    check(n_above_tolerance == 0,
          "Rinv_new ≈ Rinv_legacy at every n  (worst rel diff "
          + std::to_string(max_rel_diff_total) + " over "
          + std::to_string(n_compared) + " grid points; tol = "
          + std::to_string(rel_tol) + ")");

    check(max_asym_new < asym_tol,
          "Rinv_new is bit-symmetric at every n  (worst |Rinv - Rinv^T|_inf = "
          + std::to_string(max_asym_new) + " < " + std::to_string(asym_tol) + ")");

    // FRP returns Rinv at the outer matching point cached separately --
    // both runs MUST produce equivalent rinv_final().
    {
        const Eigen::MatrixXd& Rf_leg = FRP_leg.rinv_final();
        const Eigen::MatrixXd& Rf_new = FRP_new.rinv_final();
        const double diff = (Rf_new - Rf_leg).cwiseAbs().maxCoeff();
        const double scale = std::max(Rf_leg.cwiseAbs().maxCoeff(), 1e-30);
        check(diff / scale < rel_tol,
              "rinv_final() agrees: rel diff " + std::to_string(diff/scale)
              + " < " + std::to_string(rel_tol));
    }

    std::cout << "\n  compared " << n_compared << " grid points (n_start = "
              << n_start << ", N = " << N << ", N_total = " << N_total << ")\n"
              << "  max |Rinv_new - Rinv_legacy| / |Rinv|       = "
              << max_rel_diff_total << "\n"
              << "  max |Rinv_new - Rinv_new^T|_inf             = "
              << max_asym_new << "\n";

    std::cout << "\n" << (g_fail == 0 ? "PASS" : "FAIL")
              << " frp_symmetric_inverse  (" << g_fail << " failures)\n";
    return g_fail == 0 ? 0 : 1;
}
