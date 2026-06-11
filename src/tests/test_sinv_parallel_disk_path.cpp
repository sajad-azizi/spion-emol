// test_sinv_parallel_disk_path.cpp
//
// Gate-4 verification of the new opt-in parallel-DISK SchurInverter
// path:
//
//   (A) Default DISK build (parallel_disk_chunks = false) -> serial
//   (B) New      DISK build (parallel_disk_chunks = true)  -> chunk-blocked OpenMP parallel
//
// For every n in [0, N_grid):
//   assert (Sinv_A(n) - Sinv_B(n)).cwiseAbs().maxCoeff() == 0.0
//
// If this passes bit-identical at multiple thread counts, the new path
// is mathematically equivalent to the old serial-DISK path and can be
// turned on for production runs.
//
// We also check the inner-region stability shifts match across the two
// paths (per-ir compute is deterministic, so the same ir's get the
// same shifts).

#include "io/HDF5Reader.hpp"
#include "scatt/Banner.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Dense>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

using scatt::ExchangeCoupling;
using scatt::Parameters;
using scatt::Potentials;
using scatt::SchurInverter;
using scatt::SetupBundle;
using scatt::SolverParams;
using scatt::StorageMode;
using scatt::WavefunctionSetup;
using scatt::io::HDF5Reader;
using scatt::io::PreprocData;

static int n_fail = 0;
static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++n_fail;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <h2o_preproc.h5>\n";
        return 2;
    }
    scatt::print_la_banner();

    HDF5Reader reader(argv[1]);
    PreprocData data = reader.load_all();

    Parameters params;
    params.r_min            = data.rmin;
    params.dr               = data.dr;
    params.N_grid           = data.Nr;
    params.Lmax_sce         = data.Lmax_sce;
    params.l_max_continuum  = 4;       // small + fast
    params.validate();

    SetupBundle b = WavefunctionSetup::prepare(params, data, /*energy=*/0.5);

    Potentials pot(params);
    // Force pot to DISK so we exercise the DISK-mode pot.get() path
    // exactly as in the production from-scratch L=80 run.
    pot.build(data, StorageMode::DISK,
              "./checkpoints/pot_sinv_parallel_test", /*verbose=*/false);

    ExchangeCoupling EC(b.G_coeff, b.params.n_mu, b.params.n_sigma,
                        b.params.n_occ, data.rmin, data.dr);

    const int N_psi = b.params.n_mu;

    SchurInverter SI_A(b.params, pot, &EC, &b.chi);
    SchurInverter SI_B(b.params, pot, &EC, &b.chi);

    SchurInverter::Config cfg_A;
    cfg_A.storage             = StorageMode::DISK;
    cfg_A.use_openmp          = true;
    cfg_A.verbose             = false;
    cfg_A.checkpoint_dir      = "./checkpoints/sinv_disk_serial";
    cfg_A.try_load_checkpoint = false;
    cfg_A.save_checkpoint     = false;
    cfg_A.parallel_disk_chunks = false;        // <- DEFAULT path
    std::filesystem::remove_all(cfg_A.checkpoint_dir);

    SchurInverter::Config cfg_B = cfg_A;
    cfg_B.checkpoint_dir       = "./checkpoints/sinv_disk_chunked_parallel";
    cfg_B.parallel_disk_chunks = true;          // <- NEW opt-in path
    std::filesystem::remove_all(cfg_B.checkpoint_dir);

    std::cout << "[setup] building SI_A (DISK / serial, current default) ...\n";
    SI_A.build(cfg_A);
    std::cout << "[setup] SI_A shifts A/S = "
              << SI_A.stability_shifts_A() << "/"
              << SI_A.stability_shifts_S() << "\n";

    std::cout << "[setup] building SI_B (DISK / chunk-blocked parallel, NEW) ...\n";
    SI_B.build(cfg_B);
    std::cout << "[setup] SI_B shifts A/S = "
              << SI_B.stability_shifts_A() << "/"
              << SI_B.stability_shifts_S() << "\n";

    // ------------------------------------------------------------------ //
    // Test (X) shifts must match
    // ------------------------------------------------------------------ //
    std::cout << "\n--- (X) per-ir stability shifts match exactly ---\n";
    check(SI_A.stability_shifts_A() == SI_B.stability_shifts_A(),
          "shifts_A: A=" + std::to_string(SI_A.stability_shifts_A())
          + "  B=" + std::to_string(SI_B.stability_shifts_A()));
    check(SI_A.stability_shifts_S() == SI_B.stability_shifts_S(),
          "shifts_S: A=" + std::to_string(SI_A.stability_shifts_S())
          + "  B=" + std::to_string(SI_B.stability_shifts_S()));

    // ------------------------------------------------------------------ //
    // Test (Y) per-n bit-identical comparison.
    // ------------------------------------------------------------------ //
    std::cout << "\n--- (Y) per-n bit-identical Sinv comparison ---\n";
    int n_grid = static_cast<int>(b.params.n_grid);
    int n_compared = 0;
    int n_diff = 0;
    double max_abs_diff_total = 0.0;

    for (int n = 0; n < n_grid; ++n) {
        const Eigen::MatrixXd& M_A = SI_A.get(static_cast<std::size_t>(n));
        const Eigen::MatrixXd& M_B = SI_B.get(static_cast<std::size_t>(n));
        if (M_A.rows() != N_psi || M_A.cols() != N_psi
            || M_B.rows() != N_psi || M_B.cols() != N_psi) {
            std::cerr << "shape mismatch at n=" << n << "\n"; return 1;
        }
        const double diff = (M_A - M_B).cwiseAbs().maxCoeff();
        max_abs_diff_total = std::max(max_abs_diff_total, diff);
        if (diff != 0.0) ++n_diff;
        ++n_compared;
        if (n < 3 || n == 100 || n == 500 || n == n_grid - 1) {
            check(diff == 0.0,
                  "n=" + std::to_string(n) + "  max|A-B| = "
                  + std::to_string(diff));
        }
    }
    std::cout << "  compared " << n_compared << " grid points  ("
              << n_diff << " differed; worst diff = " << max_abs_diff_total
              << ")\n";
    check(n_diff == 0,
          "all " + std::to_string(n_compared) + " n's bit-identical "
          "(max diff = " + std::to_string(max_abs_diff_total) + ")");

    std::cout << "\n" << (n_fail == 0 ? "PASS" : "FAIL")
              << " (" << n_fail << " failed)\n";
    return n_fail == 0 ? 0 : 1;
}
