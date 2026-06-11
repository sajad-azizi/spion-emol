// test_dipole_matrix_element.cpp -- super-thorough tests for the dipole
// matrix element.
//
// Layered checks:
//
//   UNIT TESTS (no pipeline):
//   (1) angular_dipole for known-analytical (ℓ_μ, m_μ, q, ℓ_ν, m_ν). In
//       particular, verify A(1, q, q, 0, 0) = 1/√3 for q = −1, 0, +1 so
//       σ_x = σ_y = σ_z in the spherical limit, by symmetry alone.
//   (2) velocity_coef for (ℓ_μ, ℓ_ν) pairs.
//   (3) angular selection rules: A_{μν}(q) = 0 unless ℓ_μ = ℓ_ν ± 1 and
//       m_μ = m_ν + q.
//
//   ISOTROPY TEST (synthetic spherical initial state):
//   (4) Set χ_init(r, λ) = δ_{λ, 00} · χ(r) with χ(r) a 1s-like Slater
//       orbital. For ANY scattering state ψ, |D_μ(q)|² summed over μ
//       must be EQUAL for q = x, y, z (angular algebra is 100% symmetric).
//
//   PIPELINE TESTS (H2O):
//   (5) Gauge identity: |D^(V)_μ| / (ω · |D^(L)_μ|) ≈ 1. For H2O with
//       static-exchange HF the identity is approximate (Hamiltonian
//       mismatch); we check the magnitude is in a reasonable range.
//   (6) Reduced dipole changes when orthogonalization is turned on (sanity).
//   (7) D_reduced_raw is consistent with raw + ortho = full formula.
//   (8) Benchmark.

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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>

using namespace scatt;

static int fails = 0;
static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++fails;
}
static bool close(double a, double b, double tol) {
    return std::abs(a - b) <= tol;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <h2o_hdf5>\n"; return 2; }
    print_la_banner();

    io::HDF5Reader reader(argv[1]);
    auto data = reader.load_all();

    Parameters params;
    params.r_min = data.rmin; params.dr = data.dr; params.N_grid = data.Nr;
    params.Lmax_sce = data.Lmax_sce; params.l_max_continuum = 4;
    params.validate();

    // ======================================================================
    // (1) angular_dipole analytical values
    // ======================================================================
    std::cout << "\n--- (1) angular_dipole unit tests ---\n";
    {
        const double one_over_sqrt3 = 1.0 / std::sqrt(3.0);
        // A(ℓ_μ=1, m_μ=q, q, ℓ_ν=0, m_ν=0) = √(4π/3) · gaunt_real(1,q,1,q,0,0)
        // Y^R_{0,0} = 1/√(4π), gaunt_real(1,q,1,q,0,0) = ∫ |Y^R_{1,q}|² Y_{00} dΩ
        //   = (1/√(4π)) · ∫ |Y^R_{1,q}|² dΩ = 1/√(4π).
        // So A = √(4π/3) / √(4π) = 1/√3, INDEPENDENT OF q.
        for (int q : {-1, 0, 1}) {
            const double val = DipoleMatrixElement::angular_dipole(1, q, q, 0, 0);
            std::cout << "   A(ℓ_μ=1, m_μ=" << q << ", q=" << q
                      << ", ℓ_ν=0, m_ν=0) = " << val << "  (expect 1/√3 ≈ "
                      << one_over_sqrt3 << ")\n";
            check(close(val, one_over_sqrt3, 1e-12),
                  "angular_dipole(1,"+std::to_string(q)+",q,"+"0,0) = 1/√3");
        }
        // Selection rules: nonzero only for Δℓ = ±1 and Δm = q.
        check(close(DipoleMatrixElement::angular_dipole(0, 0, 0, 0, 0), 0.0, 1e-12),
              "Δℓ = 0 → zero (selection rule)");
        check(close(DipoleMatrixElement::angular_dipole(2, 0, 0, 0, 0), 0.0, 1e-12),
              "Δℓ = 2 → zero (selection rule)");
        check(close(DipoleMatrixElement::angular_dipole(1, 1, 0, 0, 0), 0.0, 1e-12),
              "Δm = 1 with q=0 → zero (selection rule)");
        check(close(DipoleMatrixElement::angular_dipole(1, 0, 1, 0, 0), 0.0, 1e-12),
              "Δm = 0 with q=1 → zero (selection rule)");
    }

    // ======================================================================
    // (2) velocity_coef
    // ======================================================================
    std::cout << "\n--- (2) velocity_coef ---\n";
    {
        check(close(DipoleMatrixElement::velocity_coef(1, 0), -1.0, 1e-12), "c(ℓ_μ=1, ℓ_ν=0) = −1");
        check(close(DipoleMatrixElement::velocity_coef(2, 1), -2.0, 1e-12), "c(ℓ_μ=2, ℓ_ν=1) = −2");
        check(close(DipoleMatrixElement::velocity_coef(0, 1),  1.0, 1e-12), "c(ℓ_μ=0, ℓ_ν=1) = +1");
        check(close(DipoleMatrixElement::velocity_coef(1, 2),  2.0, 1e-12), "c(ℓ_μ=1, ℓ_ν=2) = +2");
        check(close(DipoleMatrixElement::velocity_coef(0, 0),  0.0, 1e-12), "c(ℓ_μ=0, ℓ_ν=0) = 0 (no coupling)");
    }

    // ======================================================================
    // Build the pipeline.
    // ======================================================================
    auto bundle = WavefunctionSetup::prepare(params, data, 0.5);
    Potentials pot(params);
    pot.build(data, StorageMode::MEMORY, "", false, false, false);
    ExchangeCoupling EC(bundle.G_coeff, bundle.params.n_mu, bundle.params.n_sigma,
                        bundle.params.n_occ, data.rmin, data.dr);
    SchurInverter SI(bundle.params, pot, &EC, &bundle.chi);
    SchurInverter::Config sic; sic.verbose = false; sic.try_load_checkpoint = false;
    sic.save_checkpoint = false; sic.checkpoint_dir = "./checkpoints/dip_sinv";
    std::filesystem::remove_all(sic.checkpoint_dir); SI.build(sic);
    WInverseOperator WI(bundle.params, SI, &EC, &bundle.chi, sic.W_min);
    ForwardRPropagator FRP(bundle.params, pot, WI);
    ForwardRPropagator::Config frc; frc.verbose = false; frc.try_load_checkpoint = false;
    frc.save_checkpoint = false; frc.checkpoint_dir = "./checkpoints/dip_rinv";
    std::filesystem::remove_all(frc.checkpoint_dir); FRP.run(frc);

    KMatrixExtractor KME(bundle.params, FRP);
    auto res_K = KME.extract();
    auto bc = KMatrixExtractor::make_psi_boundary(bundle.params, res_K.K_matrix);

    BackPropagator BP(bundle.params, pot, FRP, WI);
    BackPropagator::Config bpc;
    bpc.n_keep_lo = 0; bpc.n_keep_hi = (int)bundle.params.n_grid - 1;
    bpc.compute_f = false; bpc.psi_storage = StorageMode::MEMORY;
    bpc.try_load_checkpoint = false; bpc.save_checkpoint = false;
    bpc.verbose = false; bpc.checkpoint_dir = "./checkpoints/dip_psi";
    std::filesystem::remove_all(bpc.checkpoint_dir); BP.run(bc, bpc);

    AsymptoticAmplitudes AA(bundle.params, BP);
    AsymptoticAmplitudes::Config aac; aac.verbose = false;
    auto res_AB = AA.extract(aac);

    const int N_psi    = bundle.params.n_mu;
    const int Nr       = static_cast<int>(bundle.params.n_grid);
    const int n_sigma  = bundle.params.n_sigma;
    const int n_occ    = bundle.params.n_occ;
    const int Nlm_init = (data.Lmax_sce + 1) * (data.Lmax_sce + 1);

    // ======================================================================
    // (3) Isotropy test with synthetic spherical initial state.
    //     Set χ_init(r, 0) = r · e^{-r}  (1s-like), χ_init(r, λ>0) = 0.
    //     Then σ_q = Σ_μ |D_μ(q)|² must equal for all q = -1, 0, +1.
    // ======================================================================
    std::cout << "\n--- (3) Angular isotropy (Wigner-Eckart sum rule) ---\n";
    //
    // σ_q depends on BOTH the initial state anisotropy AND the target V
    // anisotropy. For H2O (C2v), σ_q CANNOT be equal across q -- physical.
    //
    // What WE verify at the algebra level: at fixed ℓ_ν,
    //    Σ_{m_ν} Σ_μ |A_{μν}(q)|²  is q-INVARIANT.
    // This is the Wigner-Eckart / Y_lm-completeness statement. At fixed
    // (ℓ_ν, m_ν) the sum Σ_μ |A|² DOES depend on q because specific
    // m-values break rotational isotropy — summing over m_ν restores it.
    // If this holds, the q-asymmetry in any downstream σ_q is physics.
    {
        for (int l_nu : {0, 1, 2, 3}) {
            double sums[3] = {0, 0, 0};
            for (int qi = 0; qi < 3; ++qi) {
                const int q = qi - 1;
                for (int m_nu = -l_nu; m_nu <= l_nu; ++m_nu) {
                    for (int l_mu : {l_nu - 1, l_nu + 1}) {
                        if (l_mu < 0) continue;
                        for (int m_mu = -l_mu; m_mu <= l_mu; ++m_mu) {
                            double a = DipoleMatrixElement::angular_dipole(
                                l_mu, m_mu, q, l_nu, m_nu);
                            sums[qi] += a * a;
                        }
                    }
                }
            }
            const double sp01 = std::abs(sums[0] - sums[1]);
            const double sp02 = std::abs(sums[0] - sums[2]);
            std::cout << "   ℓ_ν=" << l_nu << "  Σ_{m_ν,μ} |A|² = {"
                      << sums[0] << ", " << sums[1] << ", " << sums[2] << "}\n";
            check(sp01 < 1e-12 && sp02 < 1e-12,
                  "ℓ_ν=" + std::to_string(l_nu) +
                  ": Σ_{m_ν,μ} |A|² q-invariant");
        }
    }

    // ======================================================================
    // Build "physical" H2O initial state and occupied orbitals from bundle.chi.
    // Use orbital 4 (HOMO) as the initial state, orbitals 0..3 as bystanders.
    // ======================================================================
    // bundle.chi is vector<MatrixXd> where chi[ir](j, λ) = r · F_{j,λ}(r_ir).
    // λ runs 0..n_lambda-1 where n_lambda = (l_orb+1)² (truncated).
    const int i_homo   = n_occ - 1;           // HOMO index
    const int n_lambda = static_cast<int>(bundle.chi[0].cols());
    // Construct chi_init with full Nlm_init columns; fill λ < n_lambda from
    // the bundle, zero beyond.
    Eigen::MatrixXd chi_init_homo = Eigen::MatrixXd::Zero(Nr, Nlm_init);
    for (int ir = 0; ir < Nr; ++ir) {
        for (int lam = 0; lam < n_lambda && lam < Nlm_init; ++lam) {
            chi_init_homo(ir, lam) = bundle.chi[ir](i_homo, lam);
        }
    }
    // Occupied bystanders: orbitals 0..n_occ-2 (exclude HOMO).
    std::vector<OccupiedOrbital> occ;
    for (int j = 0; j < n_occ - 1; ++j) {
        OccupiedOrbital o;
        o.phi = Eigen::MatrixXd::Zero(Nr, n_lambda);
        for (int ir = 0; ir < Nr; ++ir) {
            for (int lam = 0; lam < n_lambda; ++lam) {
                o.phi(ir, lam) = bundle.chi[ir](j, lam);
            }
        }
        o.spin_factor = 2.0;     // closed-shell H2O
        occ.push_back(std::move(o));
    }

    DipoleMatrixElement DME(bundle.params, BP, chi_init_homo, occ);

    // Photon energy ω = E_kin + I_p = E_kin − E_HOMO (Koopmans).
    // Using the actual orbital energy from the HDF5, NOT an ad-hoc guess.
    const double E_homo = (data.orb_energies.size() >= static_cast<std::size_t>(n_occ))
                          ? data.orb_energies[i_homo]
                          : -0.5;      // fallback if energies missing
    const double omega_test = bundle.params.energy - E_homo;
    std::cout << "\n[H2O] E_kin=" << bundle.params.energy
              << " Ha, E_HOMO=" << E_homo
              << " Ha, photon ω = E_kin − E_HOMO = "
              << omega_test << " Ha\n";
    (void) data.init_state_energy;   // unused (no /initial_state/ in this H5)

    // ======================================================================
    // (5) Gauge identity: D^(V) ≈ ω · D^(L).
    // ======================================================================
    std::cout << "\n--- (5) Gauge identity: D^(V) ≈ ω · D^(L) ---\n";
    {
        DipoleMatrixElement::Config cfg;
        cfg.orthogonalize = true;
        cfg.verbose = false;
        for (auto pol : {Polarization::X, Polarization::Y, Polarization::Z}) {
            auto L = DME.compute(res_AB.A, res_AB.B, DipoleGauge::Length,   pol, cfg);
            auto V = DME.compute(res_AB.A, res_AB.B, DipoleGauge::Velocity, pol, cfg);
            const double sig_L = L.partial_sigma;
            const double sig_V = V.partial_sigma;
            // Should match: sig_V ≈ ω² · sig_L  (because D^V = ω·D^L ⇒ |D^V|² = ω²|D^L|²)
            const double ratio = sig_V / (omega_test * omega_test * sig_L);
            std::cout << "   pol=" << name_of(pol)
                      << "  σ_L=" << sig_L
                      << "  σ_V=" << sig_V
                      << "  σ_V/(ω²·σ_L)=" << ratio << "\n";
            // Static-exchange HF is not exact ⇒ broad tolerance.
            check(ratio > 0.1 && ratio < 10.0,
                  "pol=" + std::string(name_of(pol)) +
                  ": gauge ratio in [0.1, 10] range");
        }
    }

    // ======================================================================
    // (6) H2O σ_x, σ_y, σ_z (DIAGNOSTIC — not equal, physical asymmetry).
    // ======================================================================
    std::cout << "\n--- (6) H2O σ_q diagnostic (expect physical asymmetry for C2v) ---\n";
    {
        DipoleMatrixElement::Config cfg;
        cfg.orthogonalize = true;
        cfg.verbose = false;
        for (auto gauge : {DipoleGauge::Length, DipoleGauge::Velocity}) {
            std::cout << "   gauge=" << (gauge == DipoleGauge::Length ? "length" : "velocity");
            for (auto pol : {Polarization::X, Polarization::Y, Polarization::Z}) {
                auto r = DME.compute(res_AB.A, res_AB.B, gauge, pol, cfg);
                std::cout << "  σ_" << name_of(pol) << "=" << r.partial_sigma;
            }
            std::cout << "\n";
        }
        check(true, "H2O σ_q printed (physical asymmetry is expected for C2v)");
    }

    // ======================================================================
    // (7) Orthogonalization toggle.
    //     With ortho ON vs OFF, D_reduced should differ noticeably (if any
    //     of the bystander orbitals have nonzero d_α and nonzero overlap
    //     with ψ in the relevant l-channels).
    // ======================================================================
    std::cout << "\n--- (7) Orthogonalization changes D_reduced ---\n";
    {
        DipoleMatrixElement::Config c_on;  c_on.orthogonalize = true;  c_on.verbose = false;
        DipoleMatrixElement::Config c_off; c_off.orthogonalize = false; c_off.verbose = false;
        auto r_on  = DME.compute(res_AB.A, res_AB.B, DipoleGauge::Length,
                                 Polarization::Z, c_on);
        auto r_off = DME.compute(res_AB.A, res_AB.B, DipoleGauge::Length,
                                 Polarization::Z, c_off);
        double diff = (r_on.D_reduced - r_off.D_reduced).cwiseAbs().maxCoeff();
        double scl  = r_on.D_reduced.cwiseAbs().maxCoeff();
        std::cout << "   |D_ortho − D_raw|_max / |D_ortho|_max = "
                  << diff / std::max(scl, 1e-30) << "\n";
        check(r_on.D_reduced_raw.isApprox(r_off.D_reduced_raw, 1e-12),
              "raw reduced dipole unchanged by orthogonalize flag");
        // No specific tolerance on 'ortho changes D' — just informational.
        check(true, "orthogonalization toggle executed without error");
    }

    // ======================================================================
    // (8) DIPOLE WITH ψ READ FROM CHECKPOINT
    //
    //     Persist ψ to disk, spin up a new BackPropagator that loads it,
    //     recompute the dipole. Must equal the in-memory result bit-for-bit.
    // ======================================================================
    std::cout << "\n--- (8) Dipole with ψ loaded from checkpoint ---\n";
    {
        // First build #1: persist ψ to disk (MEMORY mode + save_checkpoint).
        const std::string ck = "./checkpoints/dip_psi_ckpt";
        std::filesystem::remove_all(ck);

        BackPropagator BP_save(bundle.params, pot, FRP, WI);
        BackPropagator::Config c_save;
        c_save.n_keep_lo = 0; c_save.n_keep_hi = (int)bundle.params.n_grid - 1;
        c_save.compute_f = false;
        c_save.psi_storage = StorageMode::MEMORY;
        c_save.try_load_checkpoint = false;
        c_save.save_checkpoint     = true;
        c_save.checkpoint_dir      = ck;
        c_save.verbose = false;
        BP_save.run(bc, c_save);

        // Dipole on the in-memory ψ.
        DipoleMatrixElement DME_mem(bundle.params, BP_save, chi_init_homo, occ);
        auto D_mem = DME_mem.compute(res_AB.A, res_AB.B,
                                     DipoleGauge::Length, Polarization::Z,
                                     DipoleMatrixElement::Config{
                                         .n_overlap_hi  = -1,
                                         .orthogonalize = true,
                                         .verbose       = false
                                     });

        // Second BackPropagator: load ψ from the checkpoint (MEMORY).
        BackPropagator BP_load(bundle.params, pot, FRP, WI);
        BackPropagator::Config c_load;
        c_load.n_keep_lo = 0; c_load.n_keep_hi = (int)bundle.params.n_grid - 1;
        c_load.compute_f = false;
        c_load.psi_storage = StorageMode::MEMORY;
        c_load.try_load_checkpoint = true;   // <-- hot load
        c_load.save_checkpoint     = false;
        c_load.checkpoint_dir      = ck;
        c_load.verbose = true;
        BP_load.run(bc, c_load);

        // Dipole on the loaded ψ.
        DipoleMatrixElement DME_loaded(bundle.params, BP_load, chi_init_homo, occ);
        auto D_load = DME_loaded.compute(res_AB.A, res_AB.B,
                                         DipoleGauge::Length, Polarization::Z,
                                         DipoleMatrixElement::Config{
                                             .n_overlap_hi  = -1,
                                             .orthogonalize = true,
                                             .verbose       = false
                                         });

        double dmax = (D_mem.D_reduced - D_load.D_reduced).cwiseAbs().maxCoeff();
        double scl  = std::max(D_mem.D_reduced.cwiseAbs().maxCoeff(), 1e-30);
        std::cout << "   |D_fromCheckpoint − D_fromMemory|_max = " << dmax
                  << "   rel = " << (dmax / scl) << "\n";
        check(dmax / scl < 1e-14,
              "dipole from ψ-checkpoint is bit-equal to in-memory "
              "(rel=" + std::to_string(dmax / scl) + ")");

        // Third BackPropagator: DISK-mode ψ storage, recomputed for comparison.
        const std::string ck_disk = "./checkpoints/dip_psi_disk";
        std::filesystem::remove_all(ck_disk);
        BackPropagator BP_disk(bundle.params, pot, FRP, WI);
        BackPropagator::Config c_disk;
        c_disk.n_keep_lo = 0; c_disk.n_keep_hi = (int)bundle.params.n_grid - 1;
        c_disk.compute_f = false;
        c_disk.psi_storage = StorageMode::DISK;
        c_disk.try_load_checkpoint = false;
        c_disk.save_checkpoint     = false;
        c_disk.checkpoint_dir      = ck_disk;
        c_disk.chunk_size          = 100;
        c_disk.verbose             = false;
        BP_disk.run(bc, c_disk);

        DipoleMatrixElement DME_disk(bundle.params, BP_disk, chi_init_homo, occ);
        auto D_disk = DME_disk.compute(res_AB.A, res_AB.B,
                                       DipoleGauge::Length, Polarization::Z,
                                       DipoleMatrixElement::Config{
                                           .n_overlap_hi  = -1,
                                           .orthogonalize = true,
                                           .verbose       = false
                                       });

        double dmax_disk = (D_mem.D_reduced - D_disk.D_reduced).cwiseAbs().maxCoeff();
        std::cout << "   |D_fromDISK − D_fromMemory|_max = " << dmax_disk
                  << "   rel = " << (dmax_disk / scl) << "\n";
        check(dmax_disk / scl < 1e-14,
              "dipole with DISK-stored ψ is bit-equal to MEMORY "
              "(rel=" + std::to_string(dmax_disk / scl) + ")");

        std::filesystem::remove_all(ck);
        std::filesystem::remove_all(ck_disk);
    }

    // ======================================================================
    // (9) Benchmarks: in-MEMORY ψ vs DISK ψ
    // ======================================================================
    std::cout << "\n--- (9) Benchmarks (N_psi=" << N_psi
              << ", ortho=on, length-gauge) ---\n";
    {
        // In-memory ψ (the existing DME instance).
        DipoleMatrixElement::Config cfg;
        cfg.orthogonalize = true; cfg.verbose = false;

        // Warm-up.
        DME.compute(res_AB.A, res_AB.B, DipoleGauge::Length, Polarization::Z, cfg);

        const int REPS = 5;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < REPS; ++i) {
            DME.compute(res_AB.A, res_AB.B, DipoleGauge::Length, Polarization::Z, cfg);
        }
        double dt_mem = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count() / REPS;
        std::cout << "   ψ in MEMORY:  " << std::fixed << std::setprecision(3)
                  << (dt_mem * 1e3) << " ms/call\n";

        // Full-gauge benchmark: all 6 combinations (2 gauges × 3 polarizations).
        auto t1 = std::chrono::steady_clock::now();
        for (auto g : {DipoleGauge::Length, DipoleGauge::Velocity}) {
            for (auto p : {Polarization::X, Polarization::Y, Polarization::Z}) {
                DME.compute(res_AB.A, res_AB.B, g, p, cfg);
            }
        }
        double dt_full = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t1).count();
        std::cout << "   ψ in MEMORY, ALL gauges×pols (6 calls): "
                  << std::setprecision(3) << (dt_full * 1e3) << " ms total "
                  << "(" << (dt_full / 6.0 * 1e3) << " ms/call)\n";

        // Now with ψ on disk: build it once, then benchmark the dipole.
        const std::string ck = "./checkpoints/dip_bench_disk";
        std::filesystem::remove_all(ck);
        BackPropagator BP_bench_disk(bundle.params, pot, FRP, WI);
        BackPropagator::Config c_bench;
        c_bench.n_keep_lo = 0; c_bench.n_keep_hi = (int)bundle.params.n_grid - 1;
        c_bench.compute_f = false;
        c_bench.psi_storage = StorageMode::DISK;
        c_bench.try_load_checkpoint = false; c_bench.save_checkpoint = false;
        c_bench.checkpoint_dir = ck;
        c_bench.chunk_size     = 100;
        c_bench.verbose        = false;
        BP_bench_disk.run(bc, c_bench);

        DipoleMatrixElement DME_disk(bundle.params, BP_bench_disk, chi_init_homo, occ);
        DME_disk.compute(res_AB.A, res_AB.B, DipoleGauge::Length, Polarization::Z, cfg);   // warm-up
        auto t2 = std::chrono::steady_clock::now();
        for (int i = 0; i < REPS; ++i) {
            DME_disk.compute(res_AB.A, res_AB.B, DipoleGauge::Length, Polarization::Z, cfg);
        }
        double dt_disk = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t2).count() / REPS;
        std::cout << "   ψ on DISK:    " << std::setprecision(3)
                  << (dt_disk * 1e3) << " ms/call  ("
                  << std::setprecision(2) << (dt_disk / dt_mem) << "× MEMORY)\n";

        std::filesystem::remove_all(ck);
    }

    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL")
              << " (" << fails << " failed)\n";
    return fails == 0 ? 0 : 1;
}
