// test_gpu_dme.cpp -- CPU vs GPU bit-equivalence for
// DipoleMatrixElement::compute_six on the H2O fixture.
//
// Design (mirrors test_gpu_propagate / test_gpu_sinv):
//   1. Build the H2O fixture (pot, sinv, rinv, BP, A/B fit, occ orbitals).
//   2. Run DME::compute_six on CPU.
//   3. If a SYCL GPU is visible at runtime, rerun with cfg.use_gpu=true
//      on the SAME BackPropagator (so ψ bytes are identical) and compare
//      every per-slot observable.
//   4. If no GPU is visible (local dev, non-SYCL build), print "skip" and
//      exit 0.  The build always compiles; only the active path is gated.
//
// Tolerance: GPU does ONE column-major DGEMM (V · psi_ir) per ir, replacing
// (6 + n_occ) MKL DGEMVs.  Mathematically every output element is the same
// dot product; FP-wise the GEMM and GEMV summation orders differ inside
// oneMKL / MKL, so per-element relative discrepancy is bounded by
// ε_mach × N_psi (~1e-13 for H2O at l_cont=4).  Through the Simpson sum
// (n_pts ~ 1500) and the complex M·d_raw orthogonalization, the final
// partial_sigma absolute error is ≤ ~ε_mach × N_psi × n_pts ~ 1e-9 of
// the peak.  We assert
//     max rel diff on D_reduced      < 1e-9
//     max rel diff on d_raw          < 1e-10
//     max rel diff on b_overlap      < 1e-10
//     max rel diff on partial_sigma  < 1e-9
// across all 6 (gauge × pol) slots.  These bounds catch any kernel /
// layout / lda bug while not flagging benign GEMM-vs-GEMV rounding.
//
// This is the gate test for the Phase 4 GPU DME offload (the cuBLAS /
// cuSOLVER CUDA backend uses the same C++ entry point so once this
// passes on NVIDIA hardware the same tolerance applies there).

#include "io/HDF5Reader.hpp"
#include "scatt/AsymptoticAmplitudes.hpp"
#include "scatt/BackPropagator.hpp"
#include "scatt/DipoleIO.hpp"           // slice_index, name_of
#include "scatt/DipoleMatrixElement.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/ForwardRPropagator.hpp"
#include "scatt/GpuPropagate.hpp"       // GpuContext::gpu_available()
#include "scatt/KMatrixExtractor.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <complex>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace scatt;

static int fails = 0;
static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++fails;
}

// Relative max-abs diff helper for real Eigen matrices/vectors.
template <class A, class B>
static double rel_max(const A& cpu, const B& gpu) {
    const double scale = std::max(cpu.cwiseAbs().maxCoeff(), 1e-300);
    return (cpu - gpu).cwiseAbs().maxCoeff() / scale;
}

// Same helper for complex VectorXcd.
static double rel_max_cx(const Eigen::VectorXcd& cpu,
                          const Eigen::VectorXcd& gpu) {
    if (cpu.size() == 0) return 0.0;
    double scale = 0.0;
    for (int i = 0; i < cpu.size(); ++i) {
        scale = std::max(scale, std::abs(cpu[i]));
    }
    scale = std::max(scale, 1e-300);
    double err = 0.0;
    for (int i = 0; i < cpu.size(); ++i) {
        err = std::max(err, std::abs(cpu[i] - gpu[i]));
    }
    return err / scale;
}


int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <h2o_hdf5>\n"; return 2; }

    std::cout << "=== GPU DipoleMatrixElement::compute_six  CPU vs GPU ===\n";

    if (!GpuContext::gpu_available()) {
        std::cout << "  [skip] no SYCL GPU visible (or SCATT_HAS_SYCL not "
                     "defined). Rebuild with -DSCATT_WITH_SYCL=ON using "
                     "icpx and run on a GPU node to exercise this test.\n";
        return 0;
    }
    {
        GpuContext probe(/*prefer_gpu=*/true);
        std::cout << "  device: " << probe.info().device_name
                  << "  (HBM " << (probe.info().global_mem_bytes >> 30)
                  << " GB)\n";
    }

    // ---- Build H2O fixture state (identical to test_dipole_compute_six) ----
    io::HDF5Reader reader(argv[1]);
    auto data = reader.load_all();

    Parameters params;
    params.r_min          = data.rmin;
    params.dr             = data.dr;
    params.N_grid         = data.Nr;
    params.Lmax_sce       = data.Lmax_sce;
    params.l_max_continuum = 4;
    params.validate();

    auto bundle = WavefunctionSetup::prepare(params, data, 0.5);
    Potentials pot(params);
    pot.build(data, StorageMode::MEMORY, "", false, false, false);
    ExchangeCoupling EC(bundle.G_coeff, bundle.params.n_mu, bundle.params.n_sigma,
                        bundle.params.n_occ, data.rmin, data.dr);

    SchurInverter SI(bundle.params, pot, &EC, &bundle.chi);
    SchurInverter::Config sic;
    sic.verbose = false; sic.try_load_checkpoint = false;
    sic.save_checkpoint = false;
    sic.checkpoint_dir  = "./checkpoints/gpu_dme_sinv";
    std::filesystem::remove_all(sic.checkpoint_dir);
    SI.build(sic);

    WInverseOperator WI(bundle.params, SI, &EC, &bundle.chi, sic.W_min);
    ForwardRPropagator FRP(bundle.params, pot, WI);
    ForwardRPropagator::Config frc;
    frc.verbose = false; frc.try_load_checkpoint = false;
    frc.save_checkpoint = false;
    frc.checkpoint_dir  = "./checkpoints/gpu_dme_rinv";
    std::filesystem::remove_all(frc.checkpoint_dir);
    FRP.run(frc);

    KMatrixExtractor KME(bundle.params, FRP);
    auto res_K = KME.extract();
    auto bc    = KMatrixExtractor::make_psi_boundary(bundle.params, res_K.K_matrix);

    BackPropagator BP(bundle.params, pot, FRP, WI);
    BackPropagator::Config bpc;
    bpc.n_keep_lo = 0;
    bpc.n_keep_hi = static_cast<int>(bundle.params.n_grid) - 1;
    bpc.compute_f = false;
    bpc.psi_storage = StorageMode::MEMORY;
    bpc.try_load_checkpoint = false;
    bpc.save_checkpoint     = false;
    bpc.verbose             = false;
    bpc.checkpoint_dir      = "./checkpoints/gpu_dme_psi";
    std::filesystem::remove_all(bpc.checkpoint_dir);
    BP.run(bc, bpc);

    AsymptoticAmplitudes AA(bundle.params, BP);
    AsymptoticAmplitudes::Config aac; aac.verbose = false;
    auto res_AB = AA.extract(aac);

    // Build chi_init_homo + occupied bystanders (production shape).
    const int N_psi    = bundle.params.n_mu;
    const int Nr       = static_cast<int>(bundle.params.n_grid);
    const int n_occ    = bundle.params.n_occ;
    const int Nlm_init = (data.Lmax_sce + 1) * (data.Lmax_sce + 1);
    const int i_homo   = n_occ - 1;
    const int n_lambda = static_cast<int>(bundle.chi[0].cols());
    Eigen::MatrixXd chi_init_homo = Eigen::MatrixXd::Zero(Nr, Nlm_init);
    for (int ir = 0; ir < Nr; ++ir) {
        for (int lam = 0; lam < n_lambda && lam < Nlm_init; ++lam) {
            chi_init_homo(ir, lam) = bundle.chi[ir](i_homo, lam);
        }
    }
    std::vector<OccupiedOrbital> occ;
    for (int j = 0; j < n_occ - 1; ++j) {
        OccupiedOrbital o;
        o.phi = Eigen::MatrixXd::Zero(Nr, n_lambda);
        for (int ir = 0; ir < Nr; ++ir) {
            for (int lam = 0; lam < n_lambda; ++lam) {
                o.phi(ir, lam) = bundle.chi[ir](j, lam);
            }
        }
        o.spin_factor = 2.0;
        occ.push_back(std::move(o));
    }

    std::cout << "  H2O fixture:  N_psi=" << N_psi
              << "  n_occ=" << n_occ
              << "  Nr=" << Nr
              << "  Nlm_init=" << Nlm_init << "\n";

    DipoleMatrixElement DME(bundle.params, BP, chi_init_homo, occ);

    DipoleMatrixElement::Config cfg_cpu;
    cfg_cpu.orthogonalize = true;
    cfg_cpu.verbose       = false;
    cfg_cpu.n_overlap_hi  = -1;
    cfg_cpu.use_gpu       = false;

    DipoleMatrixElement::Config cfg_gpu = cfg_cpu;
    cfg_gpu.use_gpu       = true;

    // ----- CPU reference -----
    std::cout << "\n--- CPU compute_six (reference) ---\n";
    const std::array<DipoleResult, 6> cpu_results =
        DME.compute_six(res_AB.A, res_AB.B, cfg_cpu);

    // ----- GPU run on the SAME BP / chi_init / occ (so ψ bytes identical) -----
    std::cout << "\n--- GPU compute_six ---\n";
    const std::array<DipoleResult, 6> gpu_results =
        DME.compute_six(res_AB.A, res_AB.B, cfg_gpu);

    // ----- Per-slot comparison at eps_mach * N tolerance -----
    constexpr double TOL_D       = 1e-9;
    constexpr double TOL_dRAW    = 1e-10;
    constexpr double TOL_BOV     = 1e-10;
    constexpr double TOL_DCOR    = 1e-10;
    constexpr double TOL_SIGMA   = 1e-9;

    std::cout << std::scientific << std::setprecision(3);
    std::cout << "\n--- per-slot CPU vs GPU (rel max diff) ---\n";
    for (int slot = 0; slot < 6; ++slot) {
        const DipoleResult& a = cpu_results[slot];
        const DipoleResult& b = gpu_results[slot];
        const std::string tag =
            "slot=" + std::to_string(slot) + " (" +
            (a.gauge == DipoleGauge::Length ? "L" : "V") + "," +
            std::string(name_of(a.pol)) + ")";

        check(a.gauge == b.gauge && a.pol == b.pol,
              tag + " gauge/pol tag");

        const double r_D   = rel_max_cx(a.D_reduced, b.D_reduced);
        const double r_dr  = rel_max(a.d_raw, b.d_raw);
        const double r_bov = (a.b_overlap.size() > 0)
                            ? rel_max(a.b_overlap, b.b_overlap) : 0.0;
        const double r_dc  = (a.d_correction.size() > 0)
                            ? rel_max(a.d_correction, b.d_correction) : 0.0;
        const double sigma_cpu = a.partial_sigma;
        const double sigma_gpu = b.partial_sigma;
        const double r_sig = std::abs(sigma_cpu - sigma_gpu)
                            / std::max(std::abs(sigma_cpu), 1e-300);

        std::cout << "  " << tag
                  << "  D_red rel=" << r_D
                  << "  d_raw rel=" << r_dr
                  << "  b_overlap rel=" << r_bov
                  << "  d_corr rel=" << r_dc
                  << "  sigma rel=" << r_sig
                  << "  sigma=(" << sigma_cpu << " vs " << sigma_gpu << ")\n";

        check(r_D   < TOL_D,     tag + " D_reduced rel < 1e-9");
        check(r_dr  < TOL_dRAW,  tag + " d_raw     rel < 1e-10");
        check(r_bov < TOL_BOV,   tag + " b_overlap rel < 1e-10");
        check(r_dc  < TOL_DCOR,  tag + " d_correction rel < 1e-10");
        check(r_sig < TOL_SIGMA, tag + " partial_sigma rel < 1e-9");
    }

    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL")
              << "  test_gpu_dme  (" << fails << " failures)\n";
    return fails == 0 ? 0 : 1;
}
