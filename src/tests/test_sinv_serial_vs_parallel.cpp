// test_sinv_serial_vs_parallel.cpp
//
// Gate-2 safety proof for the upcoming parallel-DISK SchurInverter
// refactor:
//
//   Build Sinv twice on the H2O fixture using the EXISTING SchurInverter
//   code:
//     (A) DISK mode, ordered single-threaded loop (current production
//         from-scratch path at l_cont >> 10).
//     (B) MEMORY mode, OpenMP parallel-for loop (current production
//         cached path).
//
//   For every n in [0, N_grid):
//     assert (Sinv_A(n) - Sinv_B(n)).cwiseAbs().maxCoeff() == 0.0
//
//   If bit-identical, the per-ir compute is deterministic across thread
//   orderings — meaning any future "chunk-blocked parallel-DISK"
//   refactor that uses the same per-ir WS computation will also be
//   bit-identical to the current serial-DISK output.
//
//   This is a NECESSARY pre-condition before deploying the parallel-DISK
//   refactor.  If THIS test fails, the algorithm has a hidden order
//   dependency and parallel-DISK MUST NOT be deployed.

#include "io/HDF5Reader.hpp"
#include "scatt/Banner.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Dense>

#include <cmath>
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
    // MUST be MEMORY for both build paths so that pot.get(n) is read-safe
    // both serially and from parallel threads.
    pot.build(data, StorageMode::MEMORY, "", /*verbose=*/false);

    ExchangeCoupling EC(b.G_coeff, b.params.n_mu, b.params.n_sigma,
                        b.params.n_occ, data.rmin, data.dr);

    const int N_psi = b.params.n_mu;

    // Two independent SchurInverter instances.
    SchurInverter SI_A(b.params, pot, &EC, &b.chi);   // DISK / serial
    SchurInverter SI_B(b.params, pot, &EC, &b.chi);   // MEMORY / parallel

    SchurInverter::Config cfg_A;
    cfg_A.storage             = StorageMode::DISK;
    cfg_A.use_openmp          = true;          // ignored: DISK forces serial
    cfg_A.verbose             = false;
    cfg_A.checkpoint_dir      = "./checkpoints/sinv_serial_disk";
    cfg_A.try_load_checkpoint = false;
    cfg_A.save_checkpoint     = false;
    std::filesystem::remove_all(cfg_A.checkpoint_dir);

    SchurInverter::Config cfg_B;
    cfg_B.storage             = StorageMode::MEMORY;
    cfg_B.use_openmp          = true;          // honoured: MEMORY allows parallel
    cfg_B.verbose             = false;
    cfg_B.checkpoint_dir      = "./checkpoints/sinv_parallel_mem";
    cfg_B.try_load_checkpoint = false;
    cfg_B.save_checkpoint     = false;
    std::filesystem::remove_all(cfg_B.checkpoint_dir);

    std::cout << "[setup] building SI_A (DISK / serial) ...\n";
    SI_A.build(cfg_A);
    std::cout << "[setup] building SI_B (MEMORY / parallel) ...\n";
    SI_B.build(cfg_B);

    std::cout << "\n--- (A) per-n bit-identical comparison ---\n";
    int n_grid = static_cast<int>(b.params.n_grid);
    int n_compared = 0;
    int n_diff = 0;
    double max_abs_diff_total = 0.0;

    // Walk every n.
    for (int n = 0; n < n_grid; ++n) {
        const Eigen::MatrixXd& M_A = SI_A.get(static_cast<std::size_t>(n));
        const Eigen::MatrixXd& M_B = SI_B.get(static_cast<std::size_t>(n));
        if (M_A.rows() != N_psi || M_A.cols() != N_psi) {
            std::cerr << "shape mismatch at n=" << n << "\n"; return 1;
        }
        if (M_B.rows() != N_psi || M_B.cols() != N_psi) {
            std::cerr << "shape mismatch at n=" << n << "\n"; return 1;
        }
        const double diff = (M_A - M_B).cwiseAbs().maxCoeff();
        max_abs_diff_total = std::max(max_abs_diff_total, diff);
        if (diff != 0.0) ++n_diff;
        ++n_compared;
        // First few n's: print explicitly so the reader sees concrete numbers.
        if (n < 3 || n == 100 || n == 500 || n == n_grid - 1) {
            check(diff == 0.0,
                  "n=" + std::to_string(n) + "  max|A-B| = "
                  + std::to_string(diff));
        }
    }
    std::cout << "  compared " << n_compared << " grid points  ("
              << n_diff << " differed; worst diff = " << max_abs_diff_total
              << ")\n";
    // Aggregate verdict: every n must be bit-equal.
    check(n_diff == 0,
          "all " + std::to_string(n_compared) + " n's bit-identical "
          "(max diff = " + std::to_string(max_abs_diff_total) + ")");

    std::cout << "\n" << (n_fail == 0 ? "PASS" : "FAIL")
              << " (" << n_fail << " failed)\n";
    return n_fail == 0 ? 0 : 1;
}
