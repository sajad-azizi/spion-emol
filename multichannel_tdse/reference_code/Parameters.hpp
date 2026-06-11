// Parameters.hpp
//
// Runtime parameters for the multichannel Feshbach calculation.
//
// Simpler than the 2D code's Parameters class:
//   - no N_phi, dphi (no angular integration)
//   - no l_max (no partial-wave expansion)
//   - adds: reduced mass mu, well range r0, well depths V_T, V_S,
//     magnetic field B, target M_F block
//
#pragma once

#include "Common.hpp"

class Parameters {
public:
    Parameters() = default;

    // ---- Grid ----
    int    N_grid;        // number of radial grid points
    double dr;            // grid spacing in a_0

    // ---- Physics: 85Rb ----
    double mu;            // reduced mass in m_e (default AU::mu_Rb85)
    double r0;            // well range in a_0 (default AU::R_vdW_Rb85)
    double V_T;           // triplet well depth in atomic units (Hartree)
    double V_S;           // singlet well depth in atomic units
    double B_gauss;       // magnetic field in G
    int    MF_target;     // total magnetic quantum number of target block

    // ---- Channel truncation ----
    // Truncate the block to this many channels (lowest-threshold first).
    // Use 0 (or negative) to keep ALL channels in the block.
    int    N_ch_keep;

    // ---- Energy search range (for bound-state finder, in Hartree) ----
    double Emin;
    double Emax;

    // ---- Integration ----
    int    Nroots;        // Gauss-Legendre nodes (unused for now, kept for symmetry)
    int    NTHREADS;
    int    divide;        // output thinning (as in 2D code)
    int    p;             // Numerov initialization steps (>= 1)

    int    external_parameter;   // CLI parameter for parameter sweeps

    // ---- Potential smoothing at the well edge ----
    //
    // When using finite-difference methods (Numerov) on a step-potential
    // square well, the discontinuity at r = r0 causes the error to 
    // degrade from O(h^4) to O(h): Numerov's higher-order terms require 
    // smoothness of V through 4 derivatives.
    //
    // If smooth_width > 0, the well edge is replaced by a tanh profile:
    //   V_eff(r) = V_inside  * (1 - s(r))/2  +  V_outside * (1 + s(r))/2
    //   s(r) = tanh((r - r0) / smooth_width)
    // The effective width is ~6 * smooth_width (3 each side). A typical 
    // choice is smooth_width = 3 * dr, which spreads the transition over 
    // about 10 grid points — enough for Numerov to resolve. 
    //
    // For AnalyticSquareWell (which uses exact matching at r0), the 
    // smoothing is inactive — set smooth_width = 0.
    //
    double smooth_width = 0.0;    // 0 = sharp step (analytic compat)

    // Store V(r) at every grid point for legacy Numerov classes. Analytic
    // square-well and closed-channel runs can set this false and evaluate
    // the potential on the fly, which avoids huge grid-sized allocations.
    bool store_potential_grid = true;
};
