// test_gpu_propagate.cpp -- CPU-vs-GPU bit-equality check for the
// forward R-recursion and the backward ψ/f reconstruction.
//
// Design:
//   1. Build the H2O fixture (same path as test_back_propagator).
//   2. Run FRP + BP on CPU, keep final Rinv + full ψ in memory.
//   3. If a SYCL GPU is visible at runtime, rerun both with use_gpu=true
//      on fresh checkpoint dirs and compare observables.
//   4. If no GPU is visible (local dev, non-SYCL build), print "skip" and
//      exit 0.  The build always compiles; only the active path is gated.
//
// Tolerance: the GPU uses oneMKL getrf/getri on PVC while the CPU uses
// Eigen (with MKL LAPACKE dispatch when available).  Both are double-
// precision partial-pivot LU but the pivot order can differ at tied
// columns.  For a serial recursion across ~3000 steps that accumulates
// ~1e-12 · n_steps · κ(R) error.  We check rel < 1e-8 which catches real
// bugs (wrong kernel, wrong layout, off-by-one) without flagging benign
// rounding-order differences.

#include "io/HDF5Reader.hpp"
#include "scatt/AsymptoticAmplitudes.hpp"
#include "scatt/BackPropagator.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/ForwardRPropagator.hpp"
#include "scatt/GpuPropagate.hpp"
#include "scatt/KMatrixExtractor.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Dense>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

using scatt::BackPropagator;
using scatt::ExchangeCoupling;
using scatt::ForwardRPropagator;
using scatt::GpuContext;
using scatt::KMatrixExtractor;
using scatt::Parameters;
using scatt::Potentials;
using scatt::SchurInverter;
using scatt::ScatteringResult;
using scatt::SetupBundle;
using scatt::StorageMode;
using scatt::WavefunctionSetup;
using scatt::WInverseOperator;
using scatt::io::HDF5Reader;
using scatt::io::PreprocData;

static int fails = 0;
static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++fails;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <h2o_hdf5>\n"; return 2; }

    std::cout << "=== GPU propagate regression (CPU vs GPU) ===\n";

    if (!GpuContext::gpu_available()) {
        std::cout << "  [skip] no SYCL GPU visible (or SCATT_HAS_SYCL not "
                     "defined). Rebuild with -DSCATT_WITH_SYCL=ON using icpx "
                     "and run on a GPU node to exercise this test.\n";
        return 0;
    }
    {
        GpuContext probe(/*prefer_gpu=*/true);
        std::cout << "  device: " << probe.info().device_name
                  << "  (HBM " << (probe.info().global_mem_bytes >> 30) << " GB)\n";
    }

    HDF5Reader reader(argv[1]);
    PreprocData data = reader.load_all();

    Parameters params;
    params.r_min           = data.rmin;
    params.dr              = data.dr;
    params.N_grid          = data.Nr;
    params.Lmax_sce        = data.Lmax_sce;
    params.l_max_continuum = 4;
    params.validate();

    SetupBundle b = WavefunctionSetup::prepare(params, data, /*energy=*/0.5);

    Potentials pot(params);
    pot.build(data, StorageMode::MEMORY, "", /*verbose=*/false);

    ExchangeCoupling EC(b.G_coeff, b.params.n_mu, b.params.n_sigma, b.params.n_occ,
                        data.rmin, data.dr);

    SchurInverter SI(b.params, pot, &EC, &b.chi);
    SchurInverter::Config si_cfg;
    si_cfg.storage             = StorageMode::MEMORY;
    si_cfg.use_openmp          = true;
    si_cfg.verbose             = false;
    si_cfg.try_load_checkpoint = false;
    si_cfg.save_checkpoint     = false;
    si_cfg.checkpoint_dir      = "./checkpoints/sinv_gpu_test";
    std::filesystem::remove_all(si_cfg.checkpoint_dir);
    SI.build(si_cfg);

    WInverseOperator WI(b.params, SI, &EC, &b.chi, si_cfg.W_min);

    // ------------ CPU reference ------------
    ForwardRPropagator FRP_cpu(b.params, pot, WI);
    ForwardRPropagator::Config frp_cpu_cfg;
    frp_cpu_cfg.storage             = StorageMode::MEMORY;
    frp_cpu_cfg.try_load_checkpoint = false;
    frp_cpu_cfg.save_checkpoint     = false;
    frp_cpu_cfg.verbose             = false;
    frp_cpu_cfg.use_gpu             = false;
    frp_cpu_cfg.checkpoint_dir      = "./checkpoints/rinv_gpu_test_cpu";
    std::filesystem::remove_all(frp_cpu_cfg.checkpoint_dir);
    FRP_cpu.run(frp_cpu_cfg);

    KMatrixExtractor KME(b.params, FRP_cpu);
    ScatteringResult res = KME.extract();
    Eigen::MatrixXd psi_bc = KMatrixExtractor::make_psi_boundary(b.params, res.K_matrix);
    const Eigen::MatrixXd Rinv_final_cpu = FRP_cpu.rinv_final();

    BackPropagator BP_cpu(b.params, pot, FRP_cpu, WI);
    BackPropagator::Config bp_cpu_cfg;
    bp_cpu_cfg.n_keep_lo       = 0;
    bp_cpu_cfg.n_keep_hi       = static_cast<int>(b.params.n_grid) - 1;
    bp_cpu_cfg.compute_f       = true;                     // <-- f now tested too
    bp_cpu_cfg.psi_storage     = StorageMode::MEMORY;
    bp_cpu_cfg.verbose         = false;
    bp_cpu_cfg.use_gpu         = false;
    bp_cpu_cfg.try_load_checkpoint = false;
    bp_cpu_cfg.save_checkpoint     = false;
    BP_cpu.run(psi_bc, bp_cpu_cfg);

    using scatt::AsymptoticAmplitudes;
    AsymptoticAmplitudes AA_cpu(b.params, BP_cpu);
    AsymptoticAmplitudes::Config aa_cfg;
    aa_cfg.verbose = false;
    auto AB_cpu = AA_cpu.extract(aa_cfg);

    // ------------ GPU run ------------
    ForwardRPropagator FRP_gpu(b.params, pot, WI);
    ForwardRPropagator::Config frp_gpu_cfg = frp_cpu_cfg;
    frp_gpu_cfg.use_gpu        = true;
    frp_gpu_cfg.checkpoint_dir = "./checkpoints/rinv_gpu_test_gpu";
    std::filesystem::remove_all(frp_gpu_cfg.checkpoint_dir);
    FRP_gpu.run(frp_gpu_cfg);

    KMatrixExtractor KME_gpu(b.params, FRP_gpu);
    ScatteringResult res_gpu = KME_gpu.extract();

    BackPropagator BP_gpu(b.params, pot, FRP_gpu, WI);
    BackPropagator::Config bp_gpu_cfg = bp_cpu_cfg;
    bp_gpu_cfg.use_gpu = true;
    BP_gpu.run(psi_bc, bp_gpu_cfg);

    AsymptoticAmplitudes AA_gpu(b.params, BP_gpu);
    auto AB_gpu = AA_gpu.extract(aa_cfg);

    // ------------ (1) Forward: Rinv_final ------------
    std::cout << "\n--- (1) Forward: Rinv_final CPU vs GPU ---\n";
    {
        double scale = std::max(Rinv_final_cpu.cwiseAbs().maxCoeff(), 1e-30);
        double err   = (Rinv_final_cpu - FRP_gpu.rinv_final()).cwiseAbs().maxCoeff();
        double rel   = err / scale;
        std::cout << "  ||Rinv_cpu||_max=" << std::scientific << std::setprecision(3) << scale
                  << "  ||cpu - gpu||_max=" << err
                  << "  rel=" << rel << "\n";
        check(rel < 1e-8, "Rinv_final CPU vs GPU rel < 1e-8");
    }

    // ------------ (2) Rinv symmetry drift ------------
    // GPU symmetrize-on-device should produce the same |R - R^T| envelope
    // as CPU's explicit 0.5*(R + R^T).
    std::cout << "\n--- (2) Rinv symmetry: max |R - R^T| at samples ---\n";
    {
        const int Nr = static_cast<int>(b.params.n_grid);
        double worst_cpu = 0.0, worst_gpu = 0.0;
        for (int n : {100, 500, 1000, 1500, 2000, 2500, Nr - 2}) {
            if (n < 0 || n >= Nr) continue;
            const auto& R_cpu = FRP_cpu.get(static_cast<std::size_t>(n));
            const auto& R_gpu = FRP_gpu.get(static_cast<std::size_t>(n));
            double s_cpu = (R_cpu - R_cpu.transpose()).cwiseAbs().maxCoeff();
            double s_gpu = (R_gpu - R_gpu.transpose()).cwiseAbs().maxCoeff();
            worst_cpu = std::max(worst_cpu, s_cpu);
            worst_gpu = std::max(worst_gpu, s_gpu);
        }
        std::cout << "  worst |R-R^T|  cpu=" << std::scientific << std::setprecision(3) << worst_cpu
                  << "   gpu=" << worst_gpu << "\n";
        // The symmetrize kernel enforces bit-symmetry up to FP rounding of
        // the average; in practice < 1e-14 for N ~ 25.
        check(worst_gpu < 1e-12, "GPU Rinv symmetry within 1e-12");
        // And that the GPU isn't drifting asymmetrically relative to CPU.
        check(worst_gpu < 10 * std::max(worst_cpu, 1e-15),
              "GPU Rinv symmetry no worse than 10x CPU");
    }

    // ------------ (3) K-matrix ------------
    // A single LU-solve downstream of Rinv_final; amplifies any forward error.
    std::cout << "\n--- (3) K-matrix CPU vs GPU ---\n";
    {
        double scale = std::max(res.K_matrix.cwiseAbs().maxCoeff(), 1e-30);
        double err   = (res.K_matrix - res_gpu.K_matrix).cwiseAbs().maxCoeff();
        double rel   = err / scale;
        std::cout << "  ||K_cpu||_max=" << std::scientific << std::setprecision(3) << scale
                  << "  ||cpu - gpu||_max=" << err
                  << "  rel=" << rel << "\n";
        check(rel < 1e-7, "K-matrix CPU vs GPU rel < 1e-7");
    }

    // ------------ (4) Back: ψ(n) at sample grid points ------------
    std::cout << "\n--- (4) Back: ψ(n) CPU vs GPU at sample grid points ---\n";
    {
        const int Nr = static_cast<int>(b.params.n_grid);
        double worst = 0.0;
        int    worst_n = -1;
        for (int n : {10, 100, 500, 1000, 1500, 2000, 2500, Nr - 2}) {
            if (n < 0 || n >= Nr) continue;
            const auto& psi_cpu = BP_cpu.get_psi(static_cast<std::size_t>(n));
            const auto& psi_gpu = BP_gpu.get_psi(static_cast<std::size_t>(n));
            double scale = std::max(psi_cpu.cwiseAbs().maxCoeff(), 1e-30);
            double err   = (psi_cpu - psi_gpu).cwiseAbs().maxCoeff();
            double rel   = err / scale;
            if (rel > worst) { worst = rel; worst_n = n; }
            std::cout << "  n=" << std::setw(5) << n
                      << "  ||ψ_cpu||_max=" << std::scientific << std::setprecision(3) << scale
                      << "  ||cpu - gpu||_max=" << err
                      << "  rel=" << rel << "\n";
        }
        check(worst < 1e-8,
              "ψ CPU vs GPU worst rel < 1e-8 (worst at n=" +
              std::to_string(worst_n) + ", rel=" + std::to_string(worst) + ")");
    }

    // ------------ (5) Back: f(n) at sample grid points ------------
    // This exercises GpuBackStepper::step with compute_f=true, which is
    // a code path not touched by the production scattering driver.
    std::cout << "\n--- (5) Back: f(n) CPU vs GPU at sample grid points ---\n";
    {
        const int Nr = static_cast<int>(b.params.n_grid);
        double worst = 0.0;
        int    worst_n = -1;
        for (int n : {10, 100, 500, 1000, 1500, 2000, 2500, Nr - 2}) {
            if (n < 0 || n >= Nr) continue;
            const auto& f_cpu = BP_cpu.get_f(static_cast<std::size_t>(n));
            const auto& f_gpu = BP_gpu.get_f(static_cast<std::size_t>(n));
            double scale = std::max(f_cpu.cwiseAbs().maxCoeff(), 1e-30);
            double err   = (f_cpu - f_gpu).cwiseAbs().maxCoeff();
            double rel   = err / scale;
            if (rel > worst) { worst = rel; worst_n = n; }
            std::cout << "  n=" << std::setw(5) << n
                      << "  ||f_cpu||_max=" << std::scientific << std::setprecision(3) << scale
                      << "  ||cpu - gpu||_max=" << err
                      << "  rel=" << rel << "\n";
        }
        check(worst < 1e-8,
              "f CPU vs GPU worst rel < 1e-8 (worst at n=" +
              std::to_string(worst_n) + ", rel=" + std::to_string(worst) + ")");
    }

    // ------------ (6) A and B (asymptotic amplitudes) ------------
    // These are computed by a separate least-squares fit downstream of ψ,
    // so agreement here verifies that GPU-produced ψ is good enough for
    // production scattering observables.
    std::cout << "\n--- (6) AsymptoticAmplitudes A, B CPU vs GPU ---\n";
    {
        double sA = std::max(AB_cpu.A.cwiseAbs().maxCoeff(), 1e-30);
        double sB = std::max(AB_cpu.B.cwiseAbs().maxCoeff(), 1e-30);
        double eA = (AB_cpu.A - AB_gpu.A).cwiseAbs().maxCoeff() / sA;
        double eB = (AB_cpu.B - AB_gpu.B).cwiseAbs().maxCoeff() / sB;
        std::cout << "  ||A_cpu||=" << std::scientific << std::setprecision(3) << sA
                  << "  A rel=" << eA << "\n";
        std::cout << "  ||B_cpu||=" << std::scientific << std::setprecision(3) << sB
                  << "  B rel=" << eB << "\n";
        check(eA < 1e-7, "A CPU vs GPU rel < 1e-7");
        check(eB < 1e-7, "B CPU vs GPU rel < 1e-7");
    }

    // ------------ (7) Second energy -- independent E ------------
    // Catches E-dependent bugs: analytic init depends on E via k;
    // W_min clamping engages at different n with different E.
    std::cout << "\n--- (7) Second energy E=1.5: ψ CPU vs GPU at middle sample ---\n";
    {
        SetupBundle b2 = WavefunctionSetup::prepare(params, data, /*energy=*/1.5);
        ExchangeCoupling EC2(b2.G_coeff, b2.params.n_mu, b2.params.n_sigma, b2.params.n_occ,
                             data.rmin, data.dr);
        SchurInverter SI2(b2.params, pot, &EC2, &b2.chi);
        SchurInverter::Config si2 = si_cfg;
        si2.checkpoint_dir = "./checkpoints/sinv_gpu_test_E2";
        std::filesystem::remove_all(si2.checkpoint_dir);
        SI2.build(si2);
        WInverseOperator WI2(b2.params, SI2, &EC2, &b2.chi, si2.W_min);

        ForwardRPropagator FRP2_cpu(b2.params, pot, WI2);
        ForwardRPropagator::Config frp2_cpu = frp_cpu_cfg;
        frp2_cpu.checkpoint_dir = "./checkpoints/rinv_gpu_test_cpu_E2";
        std::filesystem::remove_all(frp2_cpu.checkpoint_dir);
        FRP2_cpu.run(frp2_cpu);

        KMatrixExtractor KME2(b2.params, FRP2_cpu);
        auto res2    = KME2.extract();
        auto psi_bc2 = KMatrixExtractor::make_psi_boundary(b2.params, res2.K_matrix);

        BackPropagator BP2_cpu(b2.params, pot, FRP2_cpu, WI2);
        BackPropagator::Config bp2 = bp_cpu_cfg;
        BP2_cpu.run(psi_bc2, bp2);

        ForwardRPropagator FRP2_gpu(b2.params, pot, WI2);
        ForwardRPropagator::Config frp2_gpu = frp2_cpu;
        frp2_gpu.use_gpu        = true;
        frp2_gpu.checkpoint_dir = "./checkpoints/rinv_gpu_test_gpu_E2";
        std::filesystem::remove_all(frp2_gpu.checkpoint_dir);
        FRP2_gpu.run(frp2_gpu);

        BackPropagator BP2_gpu(b2.params, pot, FRP2_gpu, WI2);
        BackPropagator::Config bp2_gpu_cfg = bp2;
        bp2_gpu_cfg.use_gpu = true;
        BP2_gpu.run(psi_bc2, bp2_gpu_cfg);

        const int Nr = static_cast<int>(b2.params.n_grid);
        double worst = 0.0;
        int    worst_n = -1;
        for (int n : {100, 500, 1000, 1500, 2000, 2500, Nr - 2}) {
            if (n < 0 || n >= Nr) continue;
            const auto& psi_c = BP2_cpu.get_psi(static_cast<std::size_t>(n));
            const auto& psi_g = BP2_gpu.get_psi(static_cast<std::size_t>(n));
            double rel = (psi_c - psi_g).cwiseAbs().maxCoeff() /
                         std::max(psi_c.cwiseAbs().maxCoeff(), 1e-30);
            if (rel > worst) { worst = rel; worst_n = n; }
        }
        std::cout << "  worst rel at E=1.5: " << std::scientific
                  << std::setprecision(3) << worst
                  << "  (at n=" << worst_n << ")\n";
        check(worst < 1e-8, "ψ at E=1.5 CPU vs GPU worst rel < 1e-8");
    }

    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL")
              << " (" << fails << " failures)\n";
    return fails == 0 ? 0 : 1;
}
