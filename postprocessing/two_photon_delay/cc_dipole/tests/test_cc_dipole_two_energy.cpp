// test_cc_dipole_two_energy.cpp
//
// Two-energy structural test:
//
//   cc(κ, ν)[β, α] = cc(ν, κ)[α, β]
//
// holds because (i) the dipole r·ε̂ is Hermitian, (ii) both ψ's are
// real in the back-prop basis, (iii) the angular Gaunt A^q[μ, ν] is
// symmetric under (μ ↔ ν) for our real-Y convention.
//
// We run H2O scattering at two distinct energies (0.5 Ha and 0.7 Ha),
// build BackPropagators for each, then call compute_cc_dipole twice
// (with arguments swapped) and compare.
//
// This is the FIRST test that uses two genuinely different ψ's; passing
// it means the implementation correctly handles asymmetric inputs.
//
// Tolerance: same as the one-energy symmetry test (~ 1e-14 absolute,
// 1e-12 relative) -- the asymmetry comes only from FP roundoff in the
// two GEMMs per ir, accumulated over Nr ≈ 3000 grid points.

#include "CCDipole.hpp"

#include "io/HDF5Reader.hpp"
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
#include <filesystem>
#include <iostream>
#include <memory>

using namespace scatt;

namespace {

// Bundle of the per-energy scattering objects.  We keep them in a
// struct so the lifetime spans the cc_dipole call.
struct OneEnergy {
    Parameters                       params;
    SetupBundle                      bundle;
    std::unique_ptr<Potentials>      pot;
    std::unique_ptr<ExchangeCoupling> EC;
    std::unique_ptr<SchurInverter>    SI;
    std::unique_ptr<WInverseOperator> WI;
    std::unique_ptr<ForwardRPropagator> FRP;
    std::unique_ptr<BackPropagator>     BP;
};

OneEnergy run_scattering(const io::PreprocData& data, double E_ha,
                          const std::string& tag) {
    OneEnergy oe;
    oe.params.r_min = data.rmin;
    oe.params.dr    = data.dr;
    oe.params.N_grid = data.Nr;
    oe.params.Lmax_sce = data.Lmax_sce;
    oe.params.l_max_continuum = 4;     // small, fast for the test
    oe.params.validate();

    oe.bundle = WavefunctionSetup::prepare(oe.params, data, E_ha);

    oe.pot = std::make_unique<Potentials>(oe.params);
    oe.pot->build(data, StorageMode::MEMORY, "", false, false, false);

    oe.EC = std::make_unique<ExchangeCoupling>(
        oe.bundle.G_coeff, oe.bundle.params.n_mu, oe.bundle.params.n_sigma,
        oe.bundle.params.n_occ, data.rmin, data.dr);

    oe.SI = std::make_unique<SchurInverter>(
        oe.bundle.params, *oe.pot, oe.EC.get(), &oe.bundle.chi);
    SchurInverter::Config sic;
    sic.verbose = false;
    sic.try_load_checkpoint = false;  sic.save_checkpoint = false;
    sic.checkpoint_dir = "./checkpoints/cc_test_sinv_" + tag;
    std::filesystem::remove_all(sic.checkpoint_dir);
    oe.SI->build(sic);

    oe.WI = std::make_unique<WInverseOperator>(
        oe.bundle.params, *oe.SI, oe.EC.get(), &oe.bundle.chi, sic.W_min);

    oe.FRP = std::make_unique<ForwardRPropagator>(
        oe.bundle.params, *oe.pot, *oe.WI);
    ForwardRPropagator::Config frc;
    frc.verbose = false;
    frc.try_load_checkpoint = false;  frc.save_checkpoint = false;
    frc.checkpoint_dir = "./checkpoints/cc_test_rinv_" + tag;
    std::filesystem::remove_all(frc.checkpoint_dir);
    oe.FRP->run(frc);

    KMatrixExtractor KME(oe.bundle.params, *oe.FRP);
    auto res_K = KME.extract();
    auto bc = KMatrixExtractor::make_psi_boundary(oe.bundle.params, res_K.K_matrix);

    oe.BP = std::make_unique<BackPropagator>(
        oe.bundle.params, *oe.pot, *oe.FRP, *oe.WI);
    BackPropagator::Config bpc;
    bpc.n_keep_lo = 0;
    bpc.n_keep_hi = static_cast<int>(oe.bundle.params.n_grid) - 1;
    bpc.compute_f = false;
    bpc.psi_storage = StorageMode::MEMORY;
    bpc.try_load_checkpoint = false;  bpc.save_checkpoint = false;
    bpc.verbose = false;
    bpc.checkpoint_dir = "./checkpoints/cc_test_psi_" + tag;
    std::filesystem::remove_all(bpc.checkpoint_dir);
    oe.BP->run(bc, bpc);
    return oe;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
            "usage: %s <h2o_orbitals.h5>\n", argv[0]);
        return 2;
    }
    print_la_banner();

    io::HDF5Reader reader(argv[1]);
    auto data = reader.load_all();

    std::cout << "[two_energy] running scattering at E1 = 0.5 Ha ...\n";
    auto e1 = run_scattering(data, 0.5, "E1");
    std::cout << "[two_energy] running scattering at E2 = 0.7 Ha ...\n";
    auto e2 = run_scattering(data, 0.7, "E2");

    SolverParams sp;
    sp.n_grid          = e1.bundle.params.n_grid;
    sp.dr              = e1.bundle.params.dr;
    sp.r_min           = e1.bundle.params.r_min;
    sp.energy          = e1.bundle.params.energy;   // unused inside compute_cc_dipole
    sp.n_mu            = e1.bundle.params.n_mu;
    sp.l_max_continuum = e1.bundle.params.l_max_continuum;

    int n_fail = 0;
    auto check = [&](bool ok, const std::string& what) {
        std::printf("  [%s] %s\n", ok ? "ok  " : "FAIL", what.c_str());
        if (!ok) ++n_fail;
    };

    std::cout << "\n--- cc_dipole(κ, ν)[β, α] = cc_dipole(ν, κ)[α, β] ---\n";
    for (auto pol : {Polarization::X, Polarization::Y, Polarization::Z}) {
        auto cc_kn = cc_dipole::compute_cc_dipole(
            sp, *e1.BP, *e2.BP, DipoleGauge::Length, pol);
        auto cc_nk = cc_dipole::compute_cc_dipole(
            sp, *e2.BP, *e1.BP, DipoleGauge::Length, pol);

        const auto& A = cc_kn.cc_raw;
        const auto& B = cc_nk.cc_raw;
        if (A.rows() != B.cols() || A.cols() != B.rows()) {
            std::fprintf(stderr,
                "  shape mismatch: A=%lldx%lld  B=%lldx%lld\n",
                (long long)A.rows(), (long long)A.cols(),
                (long long)B.rows(), (long long)B.cols());
            ++n_fail;
            continue;
        }
        // |A - B^T|_∞ ÷ |A|_∞
        double max_diff = 0.0, max_abs = 0.0;
        for (int i = 0; i < A.rows(); ++i)
            for (int j = 0; j < A.cols(); ++j) {
                max_diff = std::max(max_diff, std::abs(A(i, j) - B(j, i)));
                max_abs  = std::max(max_abs,  std::abs(A(i, j)));
            }
        const double rel = (max_abs > 0) ? max_diff / max_abs : 0.0;
        std::printf("  pol=%s   max|A-B^T| = %.3e   max|A| = %.3e   rel = %.3e\n",
                    name_of(pol), max_diff, max_abs, rel);
        check(max_diff < 1e-12 || rel < 1e-10,
              std::string("two-energy transpose  pol=") + name_of(pol));
    }

    std::printf("\n  Total failures: %d\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
