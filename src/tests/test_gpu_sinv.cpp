// test_gpu_sinv.cpp -- GPU vs CPU Sinv byte-equivalence gate.
//
// Builds Sinv on the H2O fixture twice:
//   A. CPU: cfg.use_gpu = false  (LEGACY: full CPU Schur + LAPACKE inverse)
//   B. GPU: cfg.use_gpu = true   (NEW:   GpuSinvStepper Schur+inverse)
//
// Then compares Sinv(n) byte-by-byte at every n in [0, N_grid).
//
// Acceptance:
//   * On a node with a SYCL GPU visible: max per-element relative diff
//     |Sinv_gpu − Sinv_cpu|_inf / |Sinv_cpu|_inf must be < 1e-10 at
//     EVERY grid point n. A failure fails the test loud.
//   * On a node WITHOUT a SYCL GPU (e.g. login node, macOS dev box):
//     the test prints [skip] and exits 0 -- we cannot exercise the
//     GPU path here.  NOTE: the SAME test must be RE-RUN on a GPU
//     node before deploying.  The script's exit code is the only
//     signal.
//
// FAILURE MODE:
//   If ANY n shows rel_diff > 1e-10, the test FAILS.  Do NOT consider
//   this safe to deploy on production until this test passes on the
//   PVC node (not just the macOS skip path).

#include "io/HDF5Reader.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/GpuPropagate.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Dense>

#include <filesystem>
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

    std::cout << "=== GPU vs CPU Sinv byte-equivalence gate ===\n";

    if (!GpuContext::gpu_available()) {
        std::cout << "  [skip] no SYCL GPU visible (or SCATT_HAS_SYCL not "
                     "defined).  This test MUST be re-run on a GPU node "
                     "(LRZ PVC) before deploying.  Rebuild with "
                     "-DSCATT_WITH_SYCL=ON using icpx.\n";
        return 0;
    }
    {
        GpuContext probe(/*prefer_gpu=*/true);
        std::cout << "  device: " << probe.info().device_name
                  << "  (HBM " << (probe.info().global_mem_bytes >> 30)
                  << " GB)\n";
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

    // -------- CPU build (gold-standard reference) --------
    SchurInverter SI_cpu(bundle.params, pot, &EC, &bundle.chi);
    SchurInverter::Config c_cpu;
    c_cpu.verbose                 = false;
    c_cpu.try_load_checkpoint     = false;
    c_cpu.save_checkpoint         = false;
    c_cpu.use_openmp              = false;
    c_cpu.storage                 = StorageMode::MEMORY;
    c_cpu.use_symmetric_inverse   = false;     // pure legacy LAPACKE dgetrf+dgetri
    c_cpu.use_gpu                 = false;     // CPU
    SI_cpu.build(c_cpu);

    // -------- GPU build (the path under test) --------
    SchurInverter SI_gpu(bundle.params, pot, &EC, &bundle.chi);
    SchurInverter::Config c_gpu = c_cpu;
    c_gpu.use_gpu                 = true;
    SI_gpu.build(c_gpu);

    // -------- Per-n byte/relative comparison --------
    const int N_psi  = bundle.params.n_mu;
    const int n_grid = static_cast<int>(bundle.params.n_grid);

    int    n_compared        = 0;
    int    n_above_tolerance = 0;
    double max_rel_diff      = 0.0;
    double max_asym_gpu      = 0.0;
    int    n_at_worst        = -1;

    const double rel_tol  = 1e-10;
    const double asym_tol = 1e-13;

    for (int n = 0; n < n_grid; ++n) {
        const Eigen::MatrixXd& Scpu = SI_cpu.get(static_cast<std::size_t>(n));
        const Eigen::MatrixXd& Sgpu = SI_gpu.get(static_cast<std::size_t>(n));
        if (Scpu.rows() != N_psi || Sgpu.rows() != N_psi) {
            std::cerr << "shape mismatch at n=" << n << "\n";
            return 1;
        }

        // Bit-symmetry of GPU Sinv (kernel_symmetrize should make it exact).
        const double asym = (Sgpu - Sgpu.transpose()).cwiseAbs().maxCoeff();
        max_asym_gpu = std::max(max_asym_gpu, asym);

        // Relative diff vs CPU.
        const double diff = (Sgpu - Scpu).cwiseAbs().maxCoeff();
        const double scale = std::max(Scpu.cwiseAbs().maxCoeff(), 1e-30);
        const double rel = diff / scale;
        if (rel > max_rel_diff) { max_rel_diff = rel; n_at_worst = n; }
        if (rel > rel_tol) ++n_above_tolerance;

        ++n_compared;
    }

    std::cout << "\n  compared " << n_compared << " grid points\n"
              << "  max relative |Sinv_gpu - Sinv_cpu| / |Sinv_cpu| = "
              << max_rel_diff << "  at n = " << n_at_worst << "\n"
              << "  max |Sinv_gpu - Sinv_gpu^T|_inf                 = "
              << max_asym_gpu << "\n"
              << "  tolerance: rel < " << rel_tol
              << "  asym < " << asym_tol << "\n\n";

    check(max_asym_gpu < asym_tol,
          "Sinv from GPU is bit-symmetric at every n  (worst "
          + std::to_string(max_asym_gpu) + " < " + std::to_string(asym_tol) + ")");
    check(n_above_tolerance == 0,
          "Sinv_gpu matches Sinv_cpu to < " + std::to_string(rel_tol)
          + " relative at every n  (n_above_tol = "
          + std::to_string(n_above_tolerance)
          + ", worst rel = " + std::to_string(max_rel_diff)
          + " at n=" + std::to_string(n_at_worst) + ")");

    // Stability shift counts must also match exactly: shifts are CPU-side
    // for both runs and depend only on A and S after the (bit-equivalent)
    // Schur build.  A mismatch here would indicate a deeper bug.
    check(SI_cpu.stability_shifts_A() == SI_gpu.stability_shifts_A(),
          "shifts_A match: CPU=" + std::to_string(SI_cpu.stability_shifts_A())
          + "  GPU=" + std::to_string(SI_gpu.stability_shifts_A()));
    check(SI_cpu.stability_shifts_S() == SI_gpu.stability_shifts_S(),
          "shifts_S match: CPU=" + std::to_string(SI_cpu.stability_shifts_S())
          + "  GPU=" + std::to_string(SI_gpu.stability_shifts_S()));

    std::cout << "\n" << (g_fail == 0 ? "PASS" : "FAIL")
              << " gpu_sinv  (" << g_fail << " failures)\n";
    return g_fail == 0 ? 0 : 1;
}
