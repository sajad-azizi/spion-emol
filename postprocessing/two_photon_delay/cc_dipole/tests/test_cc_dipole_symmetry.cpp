// test_cc_dipole_symmetry.cpp
//
// PURPOSE
//   Verify the transpose symmetry of compute_cc_dipole at SAME energy
//   (κ = ν).  The dipole operator r·ε̂ is Hermitian, both ψ are real,
//   and the angular Gaunt A^q[μ, ν] is symmetric under (μ ↔ ν), so:
//
//      cc_raw[β, α] = cc_raw[α, β]   when ψ_κ ≡ ψ_ν.
//
// We use the H2O fixture (same as test_dipole_matrix_element) to
// generate a real ψ at one energy, then call compute_cc_dipole(κ, κ).
// The resulting matrix MUST be symmetric to numerical precision.
//
// This is the foundational structural test before doing two-energy
// runs.  If this fails, every downstream c-c result is wrong.

#include "CCDipole.hpp"

#include "io/HDF5Reader.hpp"
#include "scatt/AsymptoticAmplitudes.hpp"
#include "scatt/BackPropagator.hpp"
#include "scatt/Banner.hpp"
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

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>

using namespace scatt;

static int n_fail = 0;
static void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++n_fail;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
            "usage: %s <h2o_orbitals.h5>\n", argv[0]);
        return 2;
    }
    print_la_banner();

    // ---- Same setup pattern as test_dipole_matrix_element.cpp ----
    io::HDF5Reader reader(argv[1]);
    auto data = reader.load_all();

    Parameters params;
    params.r_min = data.rmin;  params.dr = data.dr;  params.N_grid = data.Nr;
    params.Lmax_sce = data.Lmax_sce;
    params.l_max_continuum = 4;     // small for test speed
    params.validate();

    auto bundle = WavefunctionSetup::prepare(params, data, 0.5);
    Potentials pot(params);
    pot.build(data, StorageMode::MEMORY, "", false, false, false);
    ExchangeCoupling EC(bundle.G_coeff, bundle.params.n_mu, bundle.params.n_sigma,
                        bundle.params.n_occ, data.rmin, data.dr);
    SchurInverter SI(bundle.params, pot, &EC, &bundle.chi);
    SchurInverter::Config sic; sic.verbose = false; sic.try_load_checkpoint = false;
    sic.save_checkpoint = false; sic.checkpoint_dir = "./checkpoints/cc_test_sinv";
    std::filesystem::remove_all(sic.checkpoint_dir); SI.build(sic);
    WInverseOperator WI(bundle.params, SI, &EC, &bundle.chi, sic.W_min);
    ForwardRPropagator FRP(bundle.params, pot, WI);
    ForwardRPropagator::Config frc;
    frc.verbose = false;  frc.try_load_checkpoint = false;
    frc.save_checkpoint = false;
    frc.checkpoint_dir = "./checkpoints/cc_test_rinv";
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
    bpc.try_load_checkpoint = false;  bpc.save_checkpoint = false;
    bpc.verbose = false;
    bpc.checkpoint_dir = "./checkpoints/cc_test_psi";
    std::filesystem::remove_all(bpc.checkpoint_dir);
    BP.run(bc, bpc);

    // ---- Build a SolverParams from bundle.params (only fields used by
    // compute_cc_dipole). ----
    SolverParams sp;
    sp.n_grid          = bundle.params.n_grid;
    sp.dr              = bundle.params.dr;
    sp.r_min           = bundle.params.r_min;
    sp.energy          = bundle.params.energy;
    sp.n_mu            = bundle.params.n_mu;
    sp.l_max_continuum = bundle.params.l_max_continuum;

    // ---- Run cc_dipole(κ, κ) for each polarization, check symmetry ----
    std::cout << "\n--- cc_dipole(κ, κ) symmetry test ---\n";
    std::cout << "  energy = " << sp.energy << " Ha,  N_psi = " << sp.n_mu
              << ",  Nr = " << sp.n_grid << "\n";
    for (auto pol : {Polarization::X, Polarization::Y, Polarization::Z}) {
        auto result = cc_dipole::compute_cc_dipole(
            sp, BP, BP, DipoleGauge::Length, pol);
        const auto& M = result.cc_raw;
        // Maximum |M[β,α] - M[α,β]| across all entries.
        double max_asym = 0.0;
        double max_abs  = 0.0;
        for (int i = 0; i < M.rows(); ++i) {
            for (int j = 0; j < M.cols(); ++j) {
                max_asym = std::max(max_asym, std::abs(M(i, j) - M(j, i)));
                max_abs  = std::max(max_abs, std::abs(M(i, j)));
            }
        }
        const double rel = (max_abs > 0) ? max_asym / max_abs : 0.0;
        std::printf("  pol=%s   max|M-M^T| = %.3e   max|M| = %.3e   rel=%.3e\n",
                    name_of(pol), max_asym, max_abs, rel);
        // Tolerance: ~ machine eps × N_psi × (typical magnitude) for
        // FP roundoff in the GEMMs.  Allow 1e-12 absolute or 1e-10 relative.
        check(max_asym < 1e-12 || rel < 1e-10,
              std::string("symmetry  pol=") + name_of(pol));
    }

    std::printf("\n  Total failures: %d\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
