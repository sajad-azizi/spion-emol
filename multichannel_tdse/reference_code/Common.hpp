// Common.hpp
//
// Shared definitions for the multichannel Feshbach code.
// Direct descendant of Common.hpp from the 2D polar code, with:
//   - CODATA 2018 physical constants as `constexpr` in namespace AU
//   - 85Rb-specific reduced mass precomputed at compile time
//   - conversion factors for external data (MHz, kHz, GHz, Gauss)
//
// All physics code uses atomic units: ℏ = m_e = e = a_0 = 1.
// Energies are in Hartree, lengths in a_0, masses in m_e, time in ℏ/E_h.
//
#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <complex>
#include <cmath>
#include <utility>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <omp.h>

#include <Eigen/Dense>

using std::cout;
using std::cerr;
using std::endl;

typedef std::complex<double> dcompx;

constexpr dcompx I_unit{0.0, 1.0};
constexpr dcompx zero_c{0.0, 0.0};

// ------------------------------------------------------------
// Atomic units and conversion factors (CODATA 2018)
// ------------------------------------------------------------
namespace AU {

    // Fundamental conversion: Hartree to Hz via E = h*nu.
    // 1 Hartree = 4.3597447222071e-18 J (CODATA 2018)
    // Planck constant h = 6.62607015e-34 J*s (exact in SI 2019)
    // 1 Hartree / h = 6.579683920502e15 Hz
    constexpr double Hartree_in_Hz  = 6.579683920502e15;
    constexpr double Hartree_in_kHz = Hartree_in_Hz / 1.0e3;
    constexpr double Hartree_in_MHz = Hartree_in_Hz / 1.0e6;
    constexpr double Hartree_in_GHz = Hartree_in_Hz / 1.0e9;

    // 1 atomic mass unit expressed in m_e (CODATA 2018)
    //   m_u / m_e = 1822.888486209
    constexpr double amu_in_me = 1822.888486209;

    // Bohr magneton in MHz/Gauss (for Zeeman energies)
    //   mu_B = 1.39962449361(42) MHz/G
    constexpr double mu_B_MHz_per_G = 1.39962449361;

    // ------------------------------------------------------------
    // 85Rb parameters (mass, hyperfine, g-factors)
    // ------------------------------------------------------------
    // Atomic mass of 85Rb: 84.911789738 u (IUPAC / CODATA)
    constexpr double m_Rb85_amu = 84.911789738;
    constexpr double m_Rb85     = m_Rb85_amu * amu_in_me;  // in m_e
    constexpr double mu_Rb85    = m_Rb85 / 2.0;            // reduced mass

    // 85Rb hyperfine splitting (Arimondo 1977, Steck 2021)
    //   Delta_hf / h = 3035.7324390 MHz
    constexpr double delta_hf_MHz = 3035.7324390;

    // Lande g-factors for 5S_{1/2} ground state
    //   g_J = 2.00233113  (from g_s = 2.002319...)
    //   g_I = -0.00029364 (85Rb, Arimondo 1977)
    constexpr double g_J_Rb85 = 2.00233113;
    constexpr double g_I_Rb85 = -0.00029364;

    // Nuclear and electronic spin of 85Rb ground state
    //   I = 5/2, S = 1/2 (J = 1/2 for 5S_{1/2})
    constexpr double I_nuc  = 2.5;
    constexpr double S_elec = 0.5;

    // van der Waals length for Rb_2 (CGJT RMP 2010, C_6 = 4698 a.u. triplet)
    //   R_vdW = (1/2) * (2 mu C_6)^{1/4}  = 82.11 a_0
    constexpr double R_vdW_Rb85 = 82.107;   // a_0

    // ------------------------------------------------------------
    // Helper: convert MHz -> atomic units
    // ------------------------------------------------------------
    constexpr double MHz_to_au(double x_MHz) { return x_MHz / Hartree_in_MHz; }
    constexpr double kHz_to_au(double x_kHz) { return x_kHz / Hartree_in_kHz; }
    constexpr double GHz_to_au(double x_GHz) { return x_GHz / Hartree_in_GHz; }

    constexpr double au_to_MHz(double x_au) { return x_au * Hartree_in_MHz; }
    constexpr double au_to_kHz(double x_au) { return x_au * Hartree_in_kHz; }
    constexpr double au_to_GHz(double x_au) { return x_au * Hartree_in_GHz; }

} // namespace AU
