// Potentials.hpp -- 3D model potentials and the coupled-channel matrix
// V_eff(r) = diag(l(l+1)/(2 r²)) + V^R(r), where
//   V^R_{(lm),(l'm')}(r) = ∫ Y^R_{l,m}(Ω) V(r,Ω) Y^R_{l',m'}(Ω) dΩ.
//
// V is real → V^R is symmetric (real-symmetric).  V_eff is also real-
// symmetric and the radial integration uses real arithmetic, but the
// solver stores it as complex to mirror the polar_2d code (and to allow
// complex-valued continuum solutions later if needed).
#pragma once

#include "Common.hpp"
#include "Parameters.hpp"

#include <vector>

class Potentials {
public:
    Potentials(const Parameters& params);

    // Library of 3D model potentials.  Selected by `kind`:
    //   "cubic"     -- box well -V0 inside |x|,|y|,|z| <= L/2  (default)
    //   "spherical" -- box well -V0 inside r <= R0
    //   "gaussian"  -- isotropic -V0 exp(-r²/a²)
    //   "anis_gauss"-- -V0 exp(-(x²/a + y²/b + z²/c))
    //   "harmonic"  -- 0.5 * r²
    //   "soft_coul" -- -1 / sqrt(r² + a²)
    //   "h2plus"    -- two-center SOFT Coulomb on the z-axis at z=±R_h2/2;
    //                  V = -1/sqrt(x²+y²+(z-R_h2/2)²+a²) -1/sqrt(x²+y²+(z+R_h2/2)²+a²)
    //   "h2plus_johnson" -- HARD Coulomb H₂⁺ via analytic single-center
    //                  Legendre expansion (Johnson, J. Chem. Phys. 69
    //                  (1978) 4678, Sec. IV).  Bypasses the angular
    //                  grid: the integrable 1/r_A singularity is
    //                  resolved analytically, no quadrature error.
    //                  Protons at z = ±R_h2/2 (internuclear distance R_h2).
    //   "free"      -- V == 0
    void set_potential(const std::string& kind);

    // H₂⁺ knobs (only meaningful for kind == "h2plus").  Set BEFORE build().
    void set_h2plus(double R, double a) { R_h2_ = R; a_h2_ = a; }

    // Universal depth and box-size knobs (set BEFORE build()):
    //   V₀  = well depth in atomic units (positive number; potential is -V₀)
    //   L   = box half-side or radius in a.u. (overrides external_parameter)
    void set_V0(double V0)    { V0_ = V0; }
    void set_L(double L)      { L_box_ = L; }
    double V(double r, double theta, double phi) const;

    // Build V^R_{(lm),(l'm')}(r) on the radial grid by direct angular
    // integration (Gauss-Legendre × trapezoid).  Adds the centrifugal
    // l(l+1)/(2r²) on the diagonal at the end so the result is V_eff
    // ready to feed Numerov.
    void build();

    // Read access:
    const Eigen::MatrixXcd& Veff(int ir) const { return Veff_[ir]; }

    // V^R only (no centrifugal) -- exposed for tests.
    const Eigen::MatrixXcd& VR(int ir) const { return VR_[ir]; }

    // Underlying scalar V at (r, θ, φ); used for the small-r
    // proper-initialization seed.
    double V_origin() const { return V(0.0, 0.0, 0.0); }

    int n_grid() const     { return params_.N_grid; }
    int n_channels() const { return params_.n_channels; }

private:
    const Parameters&            params_;
    std::string                  kind_   = "cubic";
    double                       V0_     = 0.75;     // depth (a.u.)
    double                       L_box_  = 1.5;     // box half-side / radius (a.u.)
    double                       a_gauss_= 1.0;     // width
    double                       b_gauss_= 1.5;     // anisotropy y-width
    double                       c_gauss_= 2.0;     // anisotropy z-width
    double                       a_soft_ = 0.5;     // soft-Coulomb regulator
    double                       R_h2_   = 2.0;     // H2+ inter-proton distance (a.u.)
    double                       a_h2_   = 0.7990;  // H2+ regulator (sqrt(0.6384))

    std::vector<Eigen::MatrixXcd> VR_;
    std::vector<Eigen::MatrixXcd> Veff_;
};
