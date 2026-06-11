// test_storage_symmetric_end_to_end.cpp
//
// End-to-end gate test: run the actual Potentials/SchurInverter/FRP
// pipeline on the H2O fixture twice -- once with symmetric_storage=true
// (Phase 1 disk optimisation) and once with =false (legacy) -- and
// assert that every stored matrix at every grid point is BYTE-EQUAL
// between the two runs.
//
// This is the proof that the optimisation does not lose any accuracy
// in production use.  It exercises:
//   - Potentials::build (DISK mode, symmetric flag)
//   - SchurInverter::build (DISK mode, symmetric flag)
//   - ForwardRPropagator::run (CPU path, DISK mode, symmetric flag)
//
// Acceptance: every byte-equality check returns 0 max-diff.  A nonzero
// diff is a real bug.

#include "io/HDF5Reader.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/ForwardRPropagator.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Dense>

#include <filesystem>
#include <iostream>
#include <string>

using namespace scatt;
namespace fs = std::filesystem;

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

    const std::string root_full = "./checkpoints/test_e2e_FULL";
    const std::string root_sym  = "./checkpoints/test_e2e_SYM";
    fs::remove_all(root_full);
    fs::remove_all(root_sym);

    // ============================================================
    // (1) POT: build twice (full + symmetric), compare every V(n)
    // ============================================================
    Potentials pot_full(params);
    Potentials pot_sym (params);

    pot_full.build(data, StorageMode::DISK, root_full + "/pot",
                   /*verbose=*/false,
                   /*try_load=*/false,
                   /*save_ckpt=*/true,
                   /*symmetric_storage=*/false);
    pot_sym .build(data, StorageMode::DISK, root_sym  + "/pot",
                   /*verbose=*/false,
                   /*try_load=*/false,
                   /*save_ckpt=*/true,
                   /*symmetric_storage=*/true);

    {
        const int Nr_pot = static_cast<int>(params.N_grid);
        int n_diff = 0;
        double max_diff = 0.0;
        for (int n = 0; n < Nr_pot; ++n) {
            const Eigen::MatrixXd& V_full = pot_full.get(static_cast<std::size_t>(n));
            const Eigen::MatrixXd& V_sym  = pot_sym .get(static_cast<std::size_t>(n));
            const double d = (V_full - V_sym).cwiseAbs().maxCoeff();
            if (d != 0.0) ++n_diff;
            if (d > max_diff) max_diff = d;
        }
        check(n_diff == 0,
              "(POT) byte-equal V(n) between full and symmetric on-disk paths "
              "at every n  (n_diff=" + std::to_string(n_diff) +
              ", max=" + std::to_string(max_diff) + ")");
    }

    // ============================================================
    // (2) SINV: build twice, compare every Sinv(n)
    // ============================================================
    SchurInverter SI_full(bundle.params, pot_full, &EC, &bundle.chi);
    SchurInverter SI_sym (bundle.params, pot_sym , &EC, &bundle.chi);
    SchurInverter::Config sic_full;
    sic_full.verbose                = false;
    sic_full.try_load_checkpoint    = false;
    sic_full.save_checkpoint        = false;
    sic_full.use_openmp             = false;
    sic_full.storage                = StorageMode::DISK;
    sic_full.checkpoint_dir         = root_full + "/sinv";
    sic_full.use_symmetric_inverse  = false;       // legacy LU path -- byte-equal between runs
    sic_full.symmetric_storage      = false;
    SchurInverter::Config sic_sym = sic_full;
    sic_sym.checkpoint_dir          = root_sym  + "/sinv";
    sic_sym.symmetric_storage       = true;
    fs::remove_all(sic_full.checkpoint_dir);
    fs::remove_all(sic_sym .checkpoint_dir);

    SI_full.build(sic_full);
    SI_sym .build(sic_sym);

    {
        const int Nr = static_cast<int>(bundle.params.n_grid);
        int n_diff = 0;
        double max_diff = 0.0;
        for (int n = 0; n < Nr; ++n) {
            const Eigen::MatrixXd& S_full = SI_full.get(static_cast<std::size_t>(n));
            const Eigen::MatrixXd& S_sym  = SI_sym .get(static_cast<std::size_t>(n));
            const double d = (S_full - S_sym).cwiseAbs().maxCoeff();
            if (d != 0.0) ++n_diff;
            if (d > max_diff) max_diff = d;
        }
        check(n_diff == 0,
              "(SINV) byte-equal Sinv(n) between full and symmetric on-disk paths "
              "at every n  (n_diff=" + std::to_string(n_diff) +
              ", max=" + std::to_string(max_diff) + ")");
    }

    // ============================================================
    // (3) FRP: run twice on CPU, compare every Rinv(n)
    // ============================================================
    WInverseOperator WI_full(bundle.params, SI_full, &EC, &bundle.chi, sic_full.W_min);
    WInverseOperator WI_sym (bundle.params, SI_sym , &EC, &bundle.chi, sic_sym .W_min);
    ForwardRPropagator FRP_full(bundle.params, pot_full, WI_full);
    ForwardRPropagator FRP_sym (bundle.params, pot_sym , WI_sym );
    ForwardRPropagator::Config frc_full;
    frc_full.verbose             = false;
    frc_full.try_load_checkpoint = false;
    frc_full.save_checkpoint     = false;
    frc_full.storage             = StorageMode::DISK;
    frc_full.use_gpu             = false;
    frc_full.checkpoint_dir      = root_full + "/rinv";
    frc_full.symmetric_storage   = false;
    ForwardRPropagator::Config frc_sym = frc_full;
    frc_sym.checkpoint_dir       = root_sym  + "/rinv";
    frc_sym.symmetric_storage    = true;
    fs::remove_all(frc_full.checkpoint_dir);
    fs::remove_all(frc_sym .checkpoint_dir);

    FRP_full.run(frc_full);
    FRP_sym .run(frc_sym );

    {
        const int n_start = FRP_full.n_start();
        const int N       = static_cast<int>(bundle.params.n_grid);
        int n_diff = 0;
        double max_diff = 0.0;
        for (int n = n_start; n < N; ++n) {
            const Eigen::MatrixXd& R_full = FRP_full.get(static_cast<std::size_t>(n));
            const Eigen::MatrixXd& R_sym  = FRP_sym .get(static_cast<std::size_t>(n));
            const double d = (R_full - R_sym).cwiseAbs().maxCoeff();
            if (d != 0.0) ++n_diff;
            if (d > max_diff) max_diff = d;
        }
        check(n_diff == 0,
              "(FRP)  byte-equal Rinv(n) between full and symmetric on-disk paths "
              "at every n  (n_diff=" + std::to_string(n_diff) +
              ", max=" + std::to_string(max_diff) + ")");
    }

    // ============================================================
    // (4) Cleanup
    // ============================================================
    fs::remove_all(root_full);
    fs::remove_all(root_sym);

    std::cout << "\n" << (g_fail == 0 ? "PASS" : "FAIL")
              << " storage_symmetric_end_to_end  (" << g_fail << " failures)\n";
    return g_fail == 0 ? 0 : 1;
}
