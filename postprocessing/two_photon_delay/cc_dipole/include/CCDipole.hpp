// CCDipole.hpp -- continuum-continuum dipole matrix element between two
// scattering states stored on disk by the main scattering pipeline.
//
// We compute, in the BACK-PROP basis (no asymptotic-amplitude rotation
// applied yet),
//
//   cc_raw[β, α] = ∫ dr  [r or 1] · Σ_{μν} ψ_β^{(κ)}(r, μ) · A^q_{μν} · ψ_α^{(ν)}(r, ν)
//
// where:
//   * ψ_β^{(κ)}(r, μ)  =  channel-μ component of the back-prop solution β at
//                         radial index r, energy ε_κ (final state)
//   * ψ_α^{(ν)}(r, ν)  =  same for the intermediate state α at energy ε_ν
//   * A^q_{μν}        = √(4π/3) · gaunt_real(ℓ_μ, m_μ, 1, q, ℓ_ν, m_ν)
//                       (the dipole angular factor; identical to the table
//                       used by `scatt::DipoleMatrixElement`)
//   * q = q_of(pol)   = +1 / 0 / -1  for x / z / y polarisation
//   * Length gauge has the extra `r` weight.
//   * Velocity gauge: the b-c formula `χ'_{i,ν}(r) + c(ℓμ,ℓν)/r · χ_{i,ν}(r)`
//     uses the operator acting on the INITIAL bound state.  For c-c we
//     could in principle apply the same form to either argument; the
//     two choices differ off-shell when the Hamiltonian is approximate,
//     and the difference is a useful numerical diagnostic.
//
//     For this first implementation we DO ONLY THE LENGTH GAUGE and
//     verify it against analytic limits.  Velocity gauge is added in a
//     follow-up, once length is proved correct, with both forms of the
//     operator (left- and right-acting derivative) computed so we can
//     compare.
//
// The output `cc_raw` is REAL (both ψ are real in the back-prop basis;
// the angular factor is real for our real-Y_lm convention; r and the
// integration are real).  Conversion to the LM-asymptotic basis (where
// the matrix element is complex due to incoming-wave BC) is done in
// Python with the saved A, B matrices.
//
// Implementation note: this tool is parallel-safe per radial index; we
// use the same OpenMP pattern as `scatt::DipoleMatrixElement::compute`.
#pragma once

#include "scatt/SolverParams.hpp"
#include "scatt/BackPropagator.hpp"
#include "scatt/DipoleMatrixElement.hpp"   // for DipoleGauge, Polarization
#include "scatt/PotentialStorage.hpp"      // for the disk-loader overload

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace cc_dipole {

struct CCResult {
    // cc_raw is shape (N_psi_κ, N_psi_ν).  Both ψ are stored on the same
    // radial grid (we assert this in the implementation).
    Eigen::MatrixXd  cc_raw;          // length OR velocity gauge per call
    scatt::DipoleGauge   gauge;
    scatt::Polarization  pol;
    double           E_kappa = 0.0;   // final  state kinetic energy (au)
    double           E_nu    = 0.0;   // intermediate kinetic energy (au)
    int              ik_kappa = -1;
    int              ik_nu    = -1;
};

// Compute one (κ, ν) c-c matrix element for one (gauge, pol).
// `bp_kappa` and `bp_nu` must already have psi loaded (read-ready).
// Both must share the same radial grid (sp.n_grid, sp.dr, sp.r_min) and
// the same channel layout (N_psi).
//
// `n_overlap_hi` truncates the radial integration window.  -1 = use all
// kept radial points of bp_kappa AND bp_nu (intersection of the two).
CCResult compute_cc_dipole(const scatt::SolverParams&  sp,
                            scatt::BackPropagator&      bp_kappa,
                            scatt::BackPropagator&      bp_nu,
                            scatt::DipoleGauge          gauge,
                            scatt::Polarization         pol,
                            int                         n_overlap_hi = -1);

// Same algorithm but reading ψ directly from on-disk `PotentialStorage`
// checkpoints written by the main scattering driver.  The caller passes
// the [n_keep_lo, n_keep_hi] window from the BackPropagator manifest
// (these indices fix the radial slice of psi that's actually populated
// on disk; see BackPropagator.cpp for how they're derived).
CCResult compute_cc_dipole(const scatt::SolverParams&  sp,
                            scatt::PotentialStorage&    psi_kappa,
                            scatt::PotentialStorage&    psi_nu,
                            int                         n_keep_lo,
                            int                         n_keep_hi,
                            scatt::DipoleGauge          gauge,
                            scatt::Polarization         pol,
                            int                         n_overlap_hi = -1);

// ---- Low-level range-accumulating API (used by chunk-blocked driver) ----
//
// The single-pair compute_cc_dipole is fine for small jobs but re-reads
// every chunk of ψ_κ and ψ_ν for every (κ, ν) pair.  For C8F8 at
// l_cont=100 we have 39 κ × 91 ν = 3549 pairs and bytes_per_psi ~ 2 TB,
// so re-reading ψ_κ for each ν is the bottleneck.
//
// The state-based API below lets the caller hoist the κ chunk-read out
// of the ν loop: read ψ_κ chunk c once, then iterate all ν against that
// chunk, then advance to chunk c+1.  Per κ this drops κ-chunk reads
// from |ν|·num_chunks to num_chunks (a |ν|× reduction on the κ side
// of I/O; ~2× total wall-time reduction at C8F8 sizes).
//
// Bit-identical to the single-pair compute up to the (already non-
// deterministic) MKL/Eigen GEMM summation order, since the angular
// table, Simpson weights, and per-ir GEMM math are identical -- only
// the OUTER loop order over ir changes (we now visit ir in chunk-
// blocked order, but the partial accumulations into cc_raw are added
// in the same overall order as the original sequential pass within
// each chunk).
// Sparse storage for one row of the angular Gaunt matrix A^q_{μν}.
// In real-Y_lm basis with the dipole selection rules (Δl = ±1, plus
// m-coupling determined by q), each row has at most ~4 nonzeros (and
// typically fewer at the l-boundaries).  Storing the table densely as
// (N_psi × N_psi) is overwhelmingly sparse for production-size grids
// (e.g. at l_cont=100 the density is ~0.04%).
//
// The row-major-grouped representation below lets the first hot-loop
// product
//
//       tmp[μ, α] = Σ_ν  A^q[μ, ν] · ψ_ν[ν, α]
//
// be evaluated as Σ_{k ∈ nnz(row(μ))} a_k · ψ_ν[ν_k, α] in O(nnz · N_psi)
// instead of the dense O(N_psi²) GEMM.  Net savings on that step are
// (N_psi / nnz-per-row) ≈ 2500× at l_cont=100; bit-equivalent to the
// dense product up to summation-order FP rounding (validated by
// test_cc_dipole_bruteforce against a hand-summed reference).
struct AngRow {
    int                 mu = 0;     // row index in A^q
    std::vector<int>    nu;         // column indices of the nonzero couplings
    std::vector<double> a;          // matching coupling values A^q[μ, ν_k]
};

struct CCAccumState {
    std::vector<AngRow>  ang;       // sparse angular Gaunt table (row-grouped)
    std::vector<double>  w;         // Simpson weights for ir in [n_lo, n_hi)
    int                  n_lo  = 0;
    int                  n_hi  = 0;
    int                  N_psi = 0;
    double               dr    = 0.0;
    double               r_min = 0.0;
    scatt::DipoleGauge   gauge = scatt::DipoleGauge::Length;
    scatt::Polarization  pol   = scatt::Polarization::Z;
};

// Build the state for one (κ, ν, gauge, pol) integral.  The integration
// window [n_lo, n_hi) is determined from the two psi keep-ranges (the
// intersection of their [n_keep_lo, n_keep_hi] windows), optionally
// capped by n_overlap_hi (-1 = no cap).
//
// Currently length-gauge only.  Throws on velocity-gauge inputs.
CCAccumState make_accum_state(const scatt::SolverParams& sp,
                               int n_keep_lo_kappa, int n_keep_hi_kappa,
                               int n_keep_lo_nu,    int n_keep_hi_nu,
                               scatt::DipoleGauge   gauge,
                               scatt::Polarization  pol,
                               int n_overlap_hi = -1);

// Accumulate
//     cc_raw += Σ_{ir in [ir_lo, ir_hi)}  w[ir-st.n_lo] · r · ψ_κ(ir)ᵀ · Ang · ψ_ν(ir)
//
// into the caller-supplied cc_raw matrix (which must already be sized
// (N_psi, N_psi); a freshly Zero'd matrix at the start of the per-κ-pair
// loop is the expected pattern).
//
// THREAD SAFETY:
//   The caller must guarantee that for every ir in [ir_lo, ir_hi),
//   psi_kappa.get(ir) and psi_nu.get(ir) will not trigger a chunk
//   re-read (DISK mode).  The recommended invocation pattern is:
//
//     psi_kappa.get(c * chunk_size);   // pre-load chunk c
//     psi_nu   .get(c * chunk_size);   // pre-load chunk c
//     int lo = std::max(state.n_lo, c * chunk_size);
//     int hi = std::min(state.n_hi, (c+1) * chunk_size);
//     if (hi > lo) cc_dipole::accumulate_cc_range(state, psi_kappa, psi_nu,
//                                                 lo, hi, cc_raw);
//
//   The function does NOT verify chunk residency (cheap calls in the
//   hot path); pre-loading is the caller's responsibility.
//
//   If both psi are in MEMORY mode, ir-loop is parallel.  Otherwise
//   ir-loop is serial (DISK mode would otherwise race on read_buffer_).
void accumulate_cc_range(const CCAccumState&       st,
                         scatt::PotentialStorage&  psi_kappa,
                         scatt::PotentialStorage&  psi_nu,
                         int                       ir_lo,
                         int                       ir_hi,
                         Eigen::MatrixXd&          cc_raw);

}  // namespace cc_dipole
