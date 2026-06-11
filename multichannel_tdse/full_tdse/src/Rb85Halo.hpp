// Rb85Halo.hpp -- bridge for the M_F=-4 entrance halo bound state.
//
// Given a B field and the recipe square-well parameters (V_T, V_S, r_0,
// μ), compute the halo binding energy E_h (in atomic units, negative)
// via two independent routes:
//
//   * ANALYTIC square-well solution (no grid; exact for the model)
//   * Renormalized Numerov on a finite radial grid
//
// Cross-checking the two routes against each other AND against the
// recipe-quoted target (E_h/h = -10.108 kHz at B = 155.04 G) is the
// validation backbone of Stage 4b.
//
// Halo channel weights (P_open in M_F=-4 entrance channel,
// P_closed in the second M_F=-4 channel) are also exposed; recipe
// targets are P_open = 0.99981, P_closed = 1.901e-4 (item 2 of the
// full-TDSE convergence checklist).
//
// The bridge isolates the reference code: this header pulls only
// standard / Eigen types; reference Common.hpp lives only inside
// Rb85Halo.cpp.
#pragma once

#include <Eigen/Dense>
#include <vector>

namespace mc_tdse {

struct Rb85HaloOptions {
    double B_gauss   = 155.04;            // recipe operating point
    double V_T_GHz   = 9.6930959056;      // recipe Sec 1.1 (positive number)
    double V_S_over_T= 1.02;              // V_S = 1.02 * V_T
    double r0_a0     = 82.1;              // square-well range
    int    N_ch_keep = 2;                 // 2-channel per M_F block (recipe)

    // Numerov-only parameters (analytic path is grid-free).
    int    N_grid_numerov = 200000;       // radial grid points for Numerov
    double dr_numerov     = 0.5;          // step in a_0  (L = 100 000 a_0)
    int    p_init_numerov = 3;            // small-r initialization steps

    // Numerov bisection energy bracket (atomic units, must contain E_h).
    // Recipe halo: E_h ≈ -1.5e-12 Ha = -10 kHz.  Default bracket
    // [-1 MHz, -0.001 kHz] is safe.
    double E_bracket_lo_kHz = -1000.0;    // "-1 MHz"
    double E_bracket_hi_kHz = -0.001;
};

// Halo channel weights from the analytic state.
struct HaloWeights {
    double P_open;        // ⟨u_open|u_open⟩  -- entrance channel
    double P_closed;      // ⟨u_closed|u_closed⟩
};

// Compute halo binding energy by analytic square-well matching.
// Returns the negative-energy root of det M(E) closest to threshold.
// Result in ATOMIC UNITS (Hartree).  This IS the halo binding used
// by downstream code -- no Numerov approximation enters; the analytic
// solution is exact for the square-well model.
double halo_binding_analytic(const Rb85HaloOptions& opt = {});

// Halo channel weights using the analytic state, normalized so that
// P_open + P_closed = 1 over the radial integration window.  The
// integration window is [0, N_grid·dr] from the options' Numerov grid.
HaloWeights halo_weights_analytic(const Rb85HaloOptions& opt = {});

// ---- Numerov-stack validator -------------------------------------
// The halo (1-part-in-10^6 binding fraction) is below the resolution
// of any uniform-grid Numerov bisection -- and even for deeper states,
// node-count bisection finds shifted Numerov-discretized eigenvalues
// rather than the analytic ones (the discretization can move levels by
// MHz at dr=0.5, since the wavefunction has a kink at r_0).
//
// Instead we validate the stack with a CORRELATION test: at every
// analytic eigenvalue E_n, the Numerov node-counter must register a
// transition (nc(E_n + δ) − nc(E_n − δ) ≥ 1) for a small δ.  This
// does not depend on the discretized eigenvalue location -- only on
// the existence of a discretized state in a small window around each
// analytic state.  It catches everything wired wrong (potential matrix,
// mu, propagator coefficients, init-near-origin) without being fooled
// by O(h^4) shifts.

// All M_F=-4 bound states in [E_lo_kHz, E_hi_kHz] from analytic.
std::vector<double> mf4_bound_states_analytic(const Rb85HaloOptions& opt,
                                              double E_lo_kHz,
                                              double E_hi_kHz);

// Numerov outward node count at a single energy (atomic units).
int mf4_node_count_numerov(double E_au, const Rb85HaloOptions& opt = {});

}  // namespace mc_tdse
