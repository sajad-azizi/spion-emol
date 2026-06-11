// DipoleMatrixElement.hpp -- photoionization dipole matrix element
// with incoming-wave boundary condition and Lagrange orthogonalization.
//
// Derivation (re-derived cleanly, 2026-04-23):
//
// 1. Final state with incoming-wave BC (Ψ⁻):
//      Ψ⁻_μ = Σ_β ψ_β · [(A − iB)⁻¹]_βμ
//    ⇒ for real ψ_β:  ⟨Ψ⁻_μ| = Σ_β ψ_β · [(A − iB)⁻¹]^*_βμ
//                              = Σ_β [(A − iB)⁻¹]^†_μβ · ψ_β
//    USES A AND B DIRECTLY, not K.
//
// 2. Angular dipole coupling via REAL Y_{lm}:
//      x/r = √(4π/3) · Y^R_{1,+1}      (q = +1)
//      y/r = √(4π/3) · Y^R_{1,−1}      (q = −1)
//      z/r = √(4π/3) · Y^R_{1, 0}      (q =  0)
//    A_{μν}(q) = ⟨Y^R_μ | r̂·ε̂_q | Y^R_ν⟩
//              = √(4π/3) · gaunt_real(ℓ_μ, m_μ, 1, q, ℓ_ν, m_ν)
//    Symmetric across q by construction (no hand-coded sign for y).
//
// 3. Radial integrals (u/r convention, u = r·F):
//      d_β^(L) = Σ_{μν} A_{μν}(q) ∫ dr · r · ψ_{β,μ}(r) · χ_{i,ν}(r)
//      d_β^(V) = Σ_{μν} A_{μν}(q) ∫ dr · ψ_{β,μ}(r) · w_{μν}(r)
//      w_{μν}(r) = χ'_{i,ν}(r) + c(ℓ_μ,ℓ_ν)/r · χ_{i,ν}(r)
//      c(ℓ_μ = ℓ_ν+1, ℓ_ν) = −(ℓ_ν+1)
//      c(ℓ_μ = ℓ_ν−1, ℓ_ν) = +ℓ_ν
//
// 4. Orthogonalization (Lagrange-multiplier):
//      D_μ = Σ_β [(A − iB)⁻¹]^†_μβ · d_β  −  Σ_α N_α · o_{μα} · d_α
//      o_{μα} = Σ_β [(A − iB)⁻¹]^†_μβ · b_{βα}
//      b_{βα} = ⟨φ_α | ψ_β⟩ = Σ_μ ∫ dr · φ_{α,μ} · ψ_{β,μ}
//      d_α = ⟨φ_α | O | Φ_i⟩ (length or velocity)
//
//    N_α is the spin occupation of orbital α (2 for doubly-occupied closed
//    shell, 1 for single-spin SOMO of an open-shell system).
//
// 5. Cross section (atomic units):
//      σ_q(ω) = (4π²/c) · ω · Σ_μ |D_μ(q)|²   (length gauge, exact wavefunc)
//             = (4π²/c) · (1/ω) · Σ_μ |D_μ^(V)(q)|²   (velocity gauge)
//      c = 1/α_fs ≈ 137.036  (speed of light in au)
//
// 6. Gauge agreement:  for eigenfunctions of same H,
//      D_μ^(V)(q) = ω · D_μ^(L)(q)
//    With an approximate Hamiltonian (static-exchange HF) the identity
//    holds only approximately; the deviation is a quality diagnostic.

#pragma once

#include "scatt/BackPropagator.hpp"
#include "scatt/SolverParams.hpp"

#include <Eigen/Dense>

#include <array>
#include <complex>
#include <string>
#include <vector>

namespace scatt {

enum class DipoleGauge { Length, Velocity };

// Cartesian polarization component.
enum class Polarization { X, Y, Z };

inline int q_of(Polarization p) {
    switch (p) {
        case Polarization::X: return  1;   // x/r ∝ Y^R_{1,+1}
        case Polarization::Y: return -1;   // y/r ∝ Y^R_{1,-1}
        case Polarization::Z: return  0;   // z/r ∝ Y^R_{1, 0}
    }
    return 0;
}
inline const char* name_of(Polarization p) {
    switch (p) { case Polarization::X: return "x";
                 case Polarization::Y: return "y";
                 case Polarization::Z: return "z"; }
    return "?";
}

struct DipoleResult {
    Eigen::VectorXcd  D_reduced;        // (N_ψ) final dipole per asymptotic channel μ
    Eigen::VectorXcd  D_reduced_raw;    // (N_ψ) without orthogonalization correction
    Eigen::VectorXd   d_raw;            // (N_ψ) raw real-valued ⟨ψ_β|O|Φ_i⟩
    Eigen::MatrixXd   b_overlap;        // (N_ψ, n_occ) ⟨φ_α|ψ_β⟩
    Eigen::VectorXd   d_correction;     // (n_occ) ⟨φ_α|O|Φ_i⟩

    double partial_sigma = 0.0;         // Σ_μ |D_μ|², natural units (no 4π²/c·ω factor)

    DipoleGauge gauge;
    Polarization pol;
};

// Occupied orbital to orthogonalise against.
struct OccupiedOrbital {
    // Partial-wave decomposition in u-convention: phi[ir](λ) = r_ir · F_{α,λ}(r_ir)
    // λ ranges 0..Nlm_orb-1 (indexed by real-Y_lm index: λ = ℓ² + ℓ + m)
    // For the bound orbital's natural angular cut.
    Eigen::MatrixXd phi;        // (Nr, N_λ) — column λ is the r sampling
    double spin_factor = 2.0;   // N_α: 2 for doubly occupied, 1 for SOMO
};

class DipoleMatrixElement {
public:
    struct Config {
        // How many radial points cover the initial state's support. Beyond
        // this r, χ_init and φ_occ are zero (orbitals decay exponentially).
        // Saves time on the integral. -1 = use all kept ψ points.
        int    n_overlap_hi = -1;
        bool   orthogonalize = true;
        bool   verbose = true;
        // Optional cache for b_{βα} = ⟨φ_α | ψ_β⟩.  This overlap depends
        // only on (φ_α, ψ_β) -- NOT on gauge or polarization -- so it is
        // identical across the 6 gauge×pol calls per energy.  When
        // non-null and shape (N_psi, n_occ), the orthogonalization step
        // skips its dominant cost (the only 6× hot spot at production
        // size) and reuses this matrix.  Caller pattern:
        //
        //     Eigen::MatrixXd b_cache;
        //     for (g, p in 6 combos) {
        //         cfg.cached_b_overlap = b_cache.size() ? &b_cache : nullptr;
        //         auto r = DME.compute(A, B, g, p, cfg);
        //         if (b_cache.size() == 0) b_cache = r.b_overlap;
        //     }
        const Eigen::MatrixXd* cached_b_overlap = nullptr;

        // GPU offload of the per-ir GEMV bottleneck inside compute_six.
        // When true AND the binary was compiled with SCATT_HAS_SYCL (or
        // SCATT_HAS_CUDA) AND a GPU device is visible (GpuContext::
        // gpu_available()==true) the ir loop dispatches to
        // GpuDipoleEngine::step: every ir does ONE DGEMM
        // (V(6+n_occ × N_psi) · psi_ir(N_psi²) → result) on the device
        // instead of (6+n_occ) MKL DGEMVs on the host.  Production
        // benchmark at C8F8 / l_cont=100: Dip::integrand drops from
        // ~11,100 s to a transfer-bound ~50 s -- about 200× on this
        // stage.
        //
        // Accuracy: GEMM and GEMV summation orders differ inside the
        // BLAS so the output is NOT bit-identical to the CPU path.
        // Per-element discrepancy ≤ eps_mach * N (~1e-13 relative) --
        // same floor as GpuForwardStepper / GpuBackStepper /
        // GpuSinvStepper.  Validated by test_gpu_dme on the H2O fixture
        // (tolerance: rel diff on partial_sigma ≤ 1e-11).
        //
        // Default false preserves the existing CPU code path bit-for-bit.
        // Falls back to CPU with a warning when use_gpu=true but no GPU
        // is visible at runtime.
        bool use_gpu = false;
    };

    // `chi_init`: (Nr, Nlm_init) matrix of χ_i,λ(r) = r · F_{i,λ}(r).
    //             Columns indexed λ = ℓ² + ℓ + m (real-Y convention).
    DipoleMatrixElement(const SolverParams&   sp,
                        BackPropagator&       bp,
                        const Eigen::MatrixXd& chi_init,
                        const std::vector<OccupiedOrbital>& occ);

    // Main entry. A, B from AsymptoticAmplitudes.
    DipoleResult compute(const Eigen::MatrixXd& A,
                         const Eigen::MatrixXd& B,
                         DipoleGauge  gauge,
                         Polarization pol,
                         const Config& cfg);

    DipoleResult compute(const Eigen::MatrixXd& A,
                         const Eigen::MatrixXd& B,
                         DipoleGauge  gauge,
                         Polarization pol) {
        return compute(A, B, gauge, pol, Config{});
    }

    // Batched 6-call entry: computes ALL six (gauge × pol) dipole results
    // in a SINGLE pass over ir, reading ψ_n exactly once and reusing it
    // across the 6 integrand GEMVs and the 60 b_overlap GEMVs.
    //
    // Returns std::array<DipoleResult, 6> in DipoleWriter::slice_index
    // order: {L,X}=0, {L,Y}=1, {L,Z}=2, {V,X}=3, {V,Y}=4, {V,Z}=5.
    //
    // Strictly bit-identical to calling compute(A, B, g, p, cfg) six times:
    // every individual GEMV and Simpson sum uses the same operand bytes in
    // the same order as the legacy path; only the loop nesting changes.
    //
    // Production speedup at L=100 / DISK ψ (N_psi=10201, n_pts=2761,
    // n_occ=60): ~6h -> ~1h.  The dominant savings are
    //   * b_overlap rebuilt once instead of 60 × n_pts ψ chunk reads
    //   * ψ window scanned once instead of 6 times for the integrand loop
    //   * M = (A − iB)⁻ᵀ inverse built once instead of 6 times
    //
    // Memory: requires ~ (n_occ + 6) × N_psi × n_pts × 8 B transient on
    // the orthogonalization path (~15 GB at L=100), freed at function exit.
    // Caller MUST confirm this fits the post-BP slack.
    std::array<DipoleResult, 6>
    compute_six(const Eigen::MatrixXd& A,
                const Eigen::MatrixXd& B,
                const Config& cfg);

    std::array<DipoleResult, 6>
    compute_six(const Eigen::MatrixXd& A,
                const Eigen::MatrixXd& B) {
        return compute_six(A, B, Config{});
    }

    // Internals exposed for targeted tests.
    // Angular dipole A_{μν}(q) = √(4π/3) · gaunt_real(ℓ_μ, m_μ, 1, q, ℓ_ν, m_ν).
    static double angular_dipole(int l_mu, int m_mu, int q, int l_nu, int m_nu);

    // Velocity-gauge centrifugal coefficient c(ℓ_μ, ℓ_ν).
    static double velocity_coef(int l_mu, int l_nu);

    // Per-phase wall-clock accumulators (ns). Reset inside compute(); represent
    // the cost of the MOST RECENT call only (dipole is single-threaded).
    struct Stats {
        std::uint64_t t_angular_table_ns = 0;   // Ang(mu, nu) gaunt table
        std::uint64_t t_xi_build_ns      = 0;   // Ξ_μ(r) precompute (+χ' for velocity)
        std::uint64_t t_psi_fetch_ns     = 0;   // BP::get_psi per ir
        std::uint64_t t_integrand_ns     = 0;   // r·ψ·Ξ inner-loop accumulation
        std::uint64_t t_simpson_ns       = 0;   // Simpson rule per β
        std::uint64_t t_M_lu_ns          = 0;   // (A - iB)^{-†} LU build
        std::uint64_t t_M_apply_ns       = 0;   // D_raw = M · d
        std::uint64_t t_ortho_ns         = 0;   // b_overlap + orthogonalization correction
    };
    const Stats& stats() const { return stats_; }

private:
    const SolverParams&                  sp_;
    BackPropagator&                      bp_;
    const Eigen::MatrixXd&               chi_init_;
    const std::vector<OccupiedOrbital>&  occ_;

    int              N_psi_   = 0;
    int              Nlm_init_= 0;
    std::vector<int> l_mu_;
    std::vector<int> m_mu_;
    std::vector<int> l_nu_;
    std::vector<int> m_nu_;

    // Precomputed numerical derivative of the initial state,
    // dchi_init_dr_(ir, lambda) = d/dr χ_{i,λ}(r_ir). 5-point stencil.
    Eigen::MatrixXd dchi_init_dr_;

    void build_channel_info_();
    void compute_initial_state_derivative_();

    // Simpson's 1/3 rule on uniform grid, for n up to n_pts.
    double simpson_(const std::vector<double>& f, int n_pts) const;

    // ---- shared scalar-reduction helpers ---------------------------------
    // These exist so that compute() and compute_six() share ONE compiled
    // body for each inline scalar-reduction loop.  Under -ffast-math
    // (icpx production recipe) the optimiser is free to apply
    // FMA / reassociation differently inside two same-source-but-
    // independently-compiled function bodies; that gave a byte-divergence
    // in d_correction (and via the ortho subtract, D_reduced) between the
    // legacy 6x compute() loop and compute_six() on LRZ.  Forcing both
    // call sites into a SINGLE function call resolves it: one binary, one
    // optimisation choice, one set of bytes.  The marker [[gnu::noinline]]
    // belt-and-braces blocks the compiler from re-optimising the body per
    // call site after inlining.
    [[gnu::noinline]] Eigen::MatrixXd build_xi_(
        const Eigen::MatrixXd& Ang,
        DipoleGauge gauge,
        int n_lo, int n_hi, int n_pts) const;

    [[gnu::noinline]] Eigen::VectorXd compute_d_correction_(
        const Eigen::MatrixXd& Xi,
        DipoleGauge gauge,
        int n_lo, int n_hi, int n_pts) const;

    [[gnu::noinline]] Eigen::VectorXcd apply_ortho_subtract_(
        const Eigen::VectorXcd& D_raw,
        const Eigen::MatrixXcd& o_coeff,
        const Eigen::VectorXd& d_correction) const;

    mutable Stats stats_;
};

}  // namespace scatt
