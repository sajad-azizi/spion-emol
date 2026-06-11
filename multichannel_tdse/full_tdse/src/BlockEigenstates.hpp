// BlockEigenstates.hpp -- per-M_F eigenstate set for the recipe TDSE.
//
// At each total M_F ∈ {-5, -4, -3, -2}, the radial Schrödinger equation
//
//   -(1/2μ) u''_α(r) + Σ_β V_{αβ}(r) u_β(r) = E u_α(r),    α = 1..N_ch
//
// where the channel basis is the field-dressed two-atom basis sorted by
// threshold (`Rb85Spin::channels(MF)`).
//
// We need eigenstates for the TDSE driver's interaction picture:
//   * energies E_n
//   * radial wavefunctions u_n^α(r) on the propagation grid
// For each block, the spectrum has BOTH bound states (E < lowest_threshold)
// and continuum (E > lowest_threshold).  Box quantization with
// Dirichlet u(L) = 0 turns the continuum into a discrete grid; the
// eigenvalue spacing is set by L (recipe uses L large enough that dE
// is well below the photon ω).
//
// Solver routes:
//   * Halo state of M_F=-4 (single, near-threshold bound state):
//     ANALYTIC square-well solver (exact for the model; Numerov can't
//     resolve 1-part-in-10^6 binding fraction).  See Rb85Halo.
//   * Other deep bound states + continuum: Renormalized Numerov on the
//     uniform radial grid (validated by test_numerov_stack).
//
// All eigenstates are returned normalized to ∫dr Σ_α |u_α(r)|² = 1.
#pragma once

#include <Eigen/Dense>
#include <vector>

namespace mc_tdse {

struct BlockEigenstates {
    int    M_F          = 0;
    int    N_ch         = 0;        // = channels(M_F).size()
    int    N_grid       = 0;
    double dr           = 0.0;      // radial grid spacing (a_0)

    // Energies (atomic units, Hartree), referenced to the M_F=-4
    // entrance threshold (recipe zero).  Length = n_states.
    std::vector<double> E_au;

    // u[n] is an (N_grid × N_ch) matrix: u[n](ir, c) = u_α^(n)(r=ir·dr).
    // Channel ordering matches Rb85Spin::channels(M_F).
    // Length = n_states.  Each is normalized: Σ_c ∫|u_n^c|² dr = 1.
    std::vector<Eigen::MatrixXd> u;

    int n_states() const { return static_cast<int>(E_au.size()); }
};

// Options for building eigenstates of a block.  All energies in kHz at
// the recipe origin (M_F=-4 entrance threshold = 0).
struct BlockBuildOptions {
    double B_gauss      = 155.04;
    double V_T_GHz      = 9.6930959056;
    double V_S_over_T   = 1.02;
    double r0_a0        = 82.1;
    // Number of channels to keep in each M_F block, sorted by ascending
    // threshold.  Recipe uses N_ch_keep=2 (the entrance + first closed
    // channel) for every block — the higher-threshold channels are
    // closed at the photon-driven energies and decoupled from the σ⁺
    // ladder we care about.  Use 0 to disable truncation.
    int    N_ch_keep    = 2;

    // Common radial grid for ALL blocks.  Must be the same one used by
    // the TDSE driver; the dipole assembler integrates u_α(r) * u_β(r)
    // on this grid.
    int    N_grid       = 200000;
    double dr_a0        = 0.5;
    int    p_init       = 3;            // Numerov small-r init steps

    // Box quantization energy window.  Two ways to specify:
    //   (A) E_max_kHz_above_threshold > 0 (and E_window_kHz_lo == E_window_kHz_hi):
    //       window is [block_threshold − 0.1 kHz, block_threshold + E_max_above].
    //   (B) E_window_kHz_lo, E_window_kHz_hi:
    //       absolute window in the RECIPE ORIGIN frame (M_F=-4 entrance = 0).
    //       Use this when the relevant TDSE final-state energy is far above
    //       the block's threshold (e.g. M_F=-3 at the photon resonance).
    double E_max_kHz_above_threshold = 120.0;
    double E_window_kHz_lo = 0.0;
    double E_window_kHz_hi = 0.0;

    // Optional override: if true, halo (M_F=-4) uses AnalyticSquareWell.
    bool   use_analytic_halo = true;
};

// Build the full eigenstate set for one M_F block.
BlockEigenstates build_block_eigenstates(int M_F,
                                         const BlockBuildOptions& opt = {});

// Diagnostic: outward Numerov node count for a given M_F block at
// a given trial energy (atomic units, recipe origin).
int block_node_count(int M_F, double E_au, const BlockBuildOptions& opt = {});

}  // namespace mc_tdse
