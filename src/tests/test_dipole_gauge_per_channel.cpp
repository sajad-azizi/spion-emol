// test_dipole_gauge_per_channel.cpp -- per-channel length-velocity
// dipole gauge identity check on the H2O fixture.
//
// The existing test_dipole_matrix_element.cpp has a CROSS-SECTION-LEVEL
// gauge identity check (σ_V ≈ ω²·σ_L) with a [0.1, 10] tolerance window
// — that's wide enough to miss a sign-mapping bug in the radial gradient
// operator (a real bug that bit a sister `model_potentials/spherical_3d`
// code: with the bug, sub-percent agreement at the per-channel level
// turned into a ratio of 0.228, but |σ_L|² and |σ_V|² magnitudes still
// landed in the [0.1, 10] window).
//
// This test asserts the IDENTITY at the level of the raw matrix element:
//
//     d_raw^V[β] = -ω · d_raw^L[β]    (per channel, dominant channels)
//
// using `cfg.orthogonalize = false` so we compare the bare matrix
// elements before the orthogonalization correction.
//
// Tolerances on dominant channels (those carrying ≥5% of max |d_raw^L|):
//
//   (a) SIGN check: -ω·d_raw^L[β] and d_raw^V[β] must agree in sign for
//       every dominant channel.  This is the high-confidence sign-bug
//       trip-wire — independent of any Hamiltonian-mismatch baseline.
//
//   (b) MAGNITUDE check: max rel-err < 50% on dominant channels.
//       For SE-HF on H2O the per-channel error is empirically ~25–40 %
//       because the bound state used full non-local V_x while the
//       continuum uses local V_x — that residual is real physics,
//       not a code bug.  A genuine sign / branch / Wigner-factor bug
//       produces O(1)+ errors, way outside this band.
//
//   (c) DIAGNOSTIC: also print σ_V/(ω²·σ_L) cross-section ratio for
//       comparison against the existing test_dipole_matrix_element check.

#include "io/HDF5Reader.hpp"
#include "scatt/AsymptoticAmplitudes.hpp"
#include "scatt/BackPropagator.hpp"
#include "scatt/DipoleMatrixElement.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/ForwardRPropagator.hpp"
#include "scatt/KMatrixExtractor.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace scatt;

int main(int argc, char** argv) {
    std::string h5_path;
    if (argc > 1) {
        h5_path = argv[1];
    } else {
        const char* env = std::getenv("PREPROC_H5_DIR");
        if (env) h5_path = std::string(env) + "/h2o_ccpvdz_sph.orbitals.h5";
    }
    if (h5_path.empty() || !std::filesystem::exists(h5_path)) {
        std::cerr << "test_dipole_gauge_per_channel: H2O fixture not found.\n"
                     "  Expected at $PREPROC_H5_DIR/h2o_ccpvdz_sph.orbitals.h5\n"
                     "  or pass path as argv[1].\n";
        return 77;   // skip
    }

    io::HDF5Reader reader(h5_path);
    auto data = reader.load_all();

    Parameters params;
    params.r_min          = data.rmin;
    params.dr             = data.dr;
    params.N_grid         = data.Nr;
    params.Lmax_sce       = data.Lmax_sce;
    params.l_max_continuum = 4;
    params.validate();

    const double E_kin = 0.5;
    auto bundle = WavefunctionSetup::prepare(params, data, E_kin);
    Potentials pot(params);
    pot.build(data, StorageMode::MEMORY, "", false, false, false);

    ExchangeCoupling EC(bundle.G_coeff, bundle.params.n_mu,
                        bundle.params.n_sigma, bundle.params.n_occ,
                        data.rmin, data.dr);

    SchurInverter SI(bundle.params, pot, &EC, &bundle.chi);
    SchurInverter::Config sic;
    sic.verbose = false; sic.try_load_checkpoint = false;
    sic.save_checkpoint = false;
    sic.checkpoint_dir = "./checkpoints/gauge_test_sinv";
    std::filesystem::remove_all(sic.checkpoint_dir);
    SI.build(sic);
    WInverseOperator WI(bundle.params, SI, &EC, &bundle.chi, sic.W_min);

    ForwardRPropagator FRP(bundle.params, pot, WI);
    ForwardRPropagator::Config frc;
    frc.verbose = false; frc.try_load_checkpoint = false;
    frc.save_checkpoint = false;
    frc.checkpoint_dir = "./checkpoints/gauge_test_rinv";
    std::filesystem::remove_all(frc.checkpoint_dir);
    FRP.run(frc);

    KMatrixExtractor KME(bundle.params, FRP);
    auto res_K = KME.extract();
    auto bc = KMatrixExtractor::make_psi_boundary(bundle.params, res_K.K_matrix);

    BackPropagator BP(bundle.params, pot, FRP, WI);
    BackPropagator::Config bpc;
    bpc.n_keep_lo = 0;
    bpc.n_keep_hi = static_cast<int>(bundle.params.n_grid) - 1;
    bpc.compute_f = false;
    bpc.psi_storage = StorageMode::MEMORY;
    bpc.try_load_checkpoint = false; bpc.save_checkpoint = false;
    bpc.verbose = false;
    bpc.checkpoint_dir = "./checkpoints/gauge_test_psi";
    std::filesystem::remove_all(bpc.checkpoint_dir);
    BP.run(bc, bpc);

    AsymptoticAmplitudes AA(bundle.params, BP);
    AsymptoticAmplitudes::Config aac;  aac.verbose = false;
    auto res_AB = AA.extract(aac);

    const int Nr        = static_cast<int>(bundle.params.n_grid);
    const int n_occ     = bundle.params.n_occ;
    const int Nlm_init  = (data.Lmax_sce + 1) * (data.Lmax_sce + 1);
    const int n_lambda  = static_cast<int>(bundle.chi[0].cols());
    const int i_homo    = n_occ - 1;

    Eigen::MatrixXd chi_init = Eigen::MatrixXd::Zero(Nr, Nlm_init);
    for (int ir = 0; ir < Nr; ++ir)
        for (int lam = 0; lam < n_lambda && lam < Nlm_init; ++lam)
            chi_init(ir, lam) = bundle.chi[ir](i_homo, lam);

    DipoleMatrixElement DME(bundle.params, BP, chi_init, {});

    const double E_homo = (data.orb_energies.size() >= static_cast<std::size_t>(n_occ))
                          ? data.orb_energies[i_homo] : -0.5;
    const double omega  = bundle.params.energy - E_homo;

    std::printf("[gauge per-channel]  E_kin=%.6f  E_HOMO=%.6f  omega=%.6f\n",
                bundle.params.energy, E_homo, omega);

    DipoleMatrixElement::Config cfg;
    cfg.orthogonalize = false;
    cfg.verbose       = false;

    int n_fail_pol = 0;
    for (auto pol : {Polarization::X, Polarization::Y, Polarization::Z}) {
        auto L = DME.compute(res_AB.A, res_AB.B, DipoleGauge::Length,   pol, cfg);
        auto V = DME.compute(res_AB.A, res_AB.B, DipoleGauge::Velocity, pol, cfg);

        double max_aL = 0.0;
        for (int b = 0; b < L.d_raw.size(); ++b)
            max_aL = std::max(max_aL, std::fabs(L.d_raw(b)));
        const double aL_floor = 0.05 * max_aL;

        std::printf("\n[%s]  max |d_raw^L| = %.3e  (dominant cutoff = %.3e)\n",
                    name_of(pol), max_aL, aL_floor);

        double max_dev_dom    = 0.0;
        int    n_dom          = 0, n_print = 0;
        int    n_sign_mismatch = 0;
        for (int b = 0; b < L.d_raw.size(); ++b) {
            const double aL = std::fabs(L.d_raw(b));
            if (aL < 0.01 * max_aL) continue;
            const double target = -omega * L.d_raw(b);
            const double got    = V.d_raw(b);
            const double dev    = std::fabs(target - got)
                                / std::max(std::fabs(target), 1e-30);
            const bool   sign_ok = (target * got > 0.0);
            if (aL >= aL_floor) {
                ++n_dom;
                if (dev > max_dev_dom) max_dev_dom = dev;
                if (!sign_ok) ++n_sign_mismatch;
            }
            if (n_print < 8) {
                std::printf("    beta=%-3d  |d^L|=%.3e  -omega*d^L=%+.4e  "
                            "d^V=%+.4e   rel-err=%.3e  sign:%s %s\n",
                            b, aL, target, got, dev,
                            sign_ok ? "ok " : "BAD",
                            (aL >= aL_floor) ? "[DOMINANT]" : "");
                ++n_print;
            }
        }
        // σ_V / (ω² · σ_L) — the existing check inside test_dipole_matrix_element.
        const double sigma_L = L.partial_sigma;
        const double sigma_V = V.partial_sigma;
        const double sigma_ratio = (sigma_L > 0.0)
            ? sigma_V / (omega * omega * sigma_L) : 0.0;

        std::printf("  dominant channels (>=5%% of max) : %d\n", n_dom);
        std::printf("  sign mismatches on dominants    : %d\n", n_sign_mismatch);
        std::printf("  max rel-err (dominant only)     : %.3e\n", max_dev_dom);
        std::printf("  sigma_V / (omega^2 * sigma_L)   : %.4f\n", sigma_ratio);

        // Two-tier check: signs MUST agree on dominants (catches sign /
        // branch / Wigner-factor bugs).  Magnitudes must be within 50 %
        // (SE-HF baseline ~25-40 %; sign bug ~100 %+).
        const bool ok_sign = (n_sign_mismatch == 0);
        const bool ok_mag  = (max_dev_dom < 0.50);
        const bool ok      = ok_sign && ok_mag;
        std::printf("  sign check  : %s\n", ok_sign ? "PASS" : "FAIL");
        std::printf("  magnitude   : %s\n", ok_mag  ? "PASS" : "FAIL");
        std::printf("  result      : %s\n", ok      ? "PASS" : "FAIL");
        if (!ok) ++n_fail_pol;
    }

    std::filesystem::remove_all("./checkpoints/gauge_test_sinv");
    std::filesystem::remove_all("./checkpoints/gauge_test_rinv");
    std::filesystem::remove_all("./checkpoints/gauge_test_psi");
    return (n_fail_pol == 0) ? 0 : 1;
}
