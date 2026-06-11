// Common.hpp -- shared types and atomic-unit constants for the
// multichannel TDSE solver (Route A: full TDSE).
//
// Convention: atomic units throughout the code (ℏ = m_e = e = a_0 = 1).
//   energy : Hartree
//   length : a_0 (Bohr radius)
//   time   : ℏ/E_h ≈ 2.4188843265857e-17 s
//   mass   : m_e
//
// External I/O (CLI flags, plots, recipe constants in MHz/kHz/GHz/G/μs)
// is converted at the boundary using AU::*.
//
// ⁸⁵Rb halo (Feshbach) physics constants are in `namespace AU::Rb85`.
#pragma once

#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

using dcompx = std::complex<double>;
inline constexpr dcompx I_unit{0.0, 1.0};

namespace AU {

// Hartree-to-frequency: E = h·ν  ⇒  1 Ha / h = 6.579 683 920 502e15 Hz.
// (Source: CODATA 2018; E_h = 4.359 744 722 2071e-18 J, h = 6.626 070 15e-34 J·s.)
inline constexpr double Hartree_in_Hz  = 6.579683920502e15;
inline constexpr double Hartree_in_kHz = Hartree_in_Hz / 1.0e3;
inline constexpr double Hartree_in_MHz = Hartree_in_Hz / 1.0e6;
inline constexpr double Hartree_in_GHz = Hartree_in_Hz / 1.0e9;

// 1 atomic mass unit in m_e (CODATA 2018).
inline constexpr double amu_in_me = 1822.888486209;

// Atomic time unit in SI seconds: t_au = ℏ/E_h.
inline constexpr double atomic_time_in_s   = 2.4188843265857e-17;
inline constexpr double atomic_time_in_us  = atomic_time_in_s * 1.0e6;   // ~2.42e-11 μs/a.u.
inline constexpr double us_in_atomic_time  = 1.0 / atomic_time_in_us;     // ~4.13e10 a.u./μs

// Frequency conversions.
inline constexpr double MHz_to_au(double x_MHz) { return x_MHz / Hartree_in_MHz; }
inline constexpr double kHz_to_au(double x_kHz) { return x_kHz / Hartree_in_kHz; }
inline constexpr double GHz_to_au(double x_GHz) { return x_GHz / Hartree_in_GHz; }
inline constexpr double au_to_MHz(double x_au)  { return x_au * Hartree_in_MHz; }
inline constexpr double au_to_kHz(double x_au)  { return x_au * Hartree_in_kHz; }
inline constexpr double au_to_GHz(double x_au)  { return x_au * Hartree_in_GHz; }

// Time conversions.
inline constexpr double us_to_au(double x_us)   { return x_us * us_in_atomic_time; }
inline constexpr double au_to_us(double x_au)   { return x_au * atomic_time_in_us; }

// ⁸⁵Rb constants used by the recipe.
namespace Rb85 {
    inline constexpr double m_amu = 84.911789738;          // atomic mass (u)
    inline constexpr double m_au  = m_amu * amu_in_me;     // in m_e
    inline constexpr double mu_au = m_au / 2.0;            // reduced mass (≈77 392 m_e)

    // Hyperfine splitting Δ_hf / h = 3035.7324390 MHz (Steck/Arimondo).
    inline constexpr double delta_hf_MHz = 3035.7324390;

    // Lande g-factors for 5S_{1/2} ground state (e- and nuclear).
    inline constexpr double g_J = 2.00233113;
    inline constexpr double g_I = -0.00029364;

    inline constexpr double I_nuc  = 2.5;     // nuclear spin
    inline constexpr double S_elec = 0.5;     // electron spin

    // Square-well range r0 (van der Waals length).
    inline constexpr double r0_a0 = 82.107;

    // Singlet/triplet depths from recipe Sec. 1.1 (positive numbers,
    // applied with negative sign in V_short = -V̄·𝟙 + ΔV·s₁·s₂).
    inline constexpr double V_T_GHz = 9.6930959;       // triplet
    inline constexpr double V_S_GHz = 1.02 * V_T_GHz;  // singlet
}  // namespace Rb85

}  // namespace AU
