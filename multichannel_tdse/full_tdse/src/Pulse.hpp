// Pulse.hpp -- flexible pulse-envelope abstraction χ(t).
//
// χ(t) is the dimensionless real envelope multiplying the carrier
// e^{∓iωt} in the recipe's RF interaction:
//
//     V_RF(t) = (Ω_R / 2) χ(t) [e^{-iωt} η^(+) + e^{+iωt} η^(-)]
//
// We expose χ as a `PulseShape = std::function<double(double)>` so the
// production code can pick Gaussian, sin², flat-top, or any callable
// without an extra inheritance hierarchy.  Default for the main driver
// is `make_gaussian`; tests cover sin² and flat-top to verify the
// abstraction is honestly polymorphic.
//
// Conventions: t in atomic units (use `AU::us_to_au(...)` to enter μs).
// Each shape exposes its FWHM in the same unit so callers can match
// experimentally meaningful pulse widths.
#pragma once

#include "Common.hpp"

#include <cmath>
#include <functional>

namespace mc_tdse {

using PulseShape = std::function<double(double)>;

// ---- Gaussian ------------------------------------------------------
//   χ(t) = exp(-(t - t_c)² / (2 τ²))
//   FWHM = 2 √(2 ln 2) · τ ≈ 2.3548 τ
//   ∫χ²    = √π · τ
inline PulseShape make_gaussian(double tau, double t_center) {
    return [tau, t_center](double t) {
        const double x = (t - t_center) / tau;
        return std::exp(-0.5 * x * x);
    };
}
inline double gaussian_FWHM(double tau)    { return 2.0 * std::sqrt(2.0 * std::log(2.0)) * tau; }
inline double gaussian_int_chi2(double tau){ return std::sqrt(M_PI) * tau; }

// ---- sin²  ---------------------------------------------------------
//   χ(t) = sin²(π (t - t_start) / T_pulse)  on [t_start, t_start+T_pulse]
//        = 0                              elsewhere
//   FWHM = T_pulse / 2
//   ∫χ²    = 3 T_pulse / 8         (closed form)
inline PulseShape make_sin_squared(double T_pulse, double t_start) {
    return [T_pulse, t_start](double t) {
        if (t < t_start || t > t_start + T_pulse) return 0.0;
        const double x = M_PI * (t - t_start) / T_pulse;
        const double s = std::sin(x);
        return s * s;
    };
}
inline double sin_squared_FWHM(double T_pulse)     { return 0.5 * T_pulse; }
inline double sin_squared_int_chi2(double T_pulse) { return 0.375 * T_pulse; }

// ---- Flat-top (sin² ramp on, flat, sin² ramp off) ------------------
//   ramp duration t_ramp on each edge, flat duration T_pulse-2·t_ramp
//   total duration T_pulse, starting at t_start.
inline PulseShape make_flat_top(double T_pulse, double t_start,
                                double t_ramp)
{
    return [T_pulse, t_start, t_ramp](double t) {
        if (t < t_start || t > t_start + T_pulse) return 0.0;
        const double dt_l = t - t_start;
        const double dt_r = t_start + T_pulse - t;
        if (dt_l < t_ramp) {
            const double s = std::sin(0.5 * M_PI * dt_l / t_ramp);
            return s * s;
        }
        if (dt_r < t_ramp) {
            const double s = std::sin(0.5 * M_PI * dt_r / t_ramp);
            return s * s;
        }
        return 1.0;
    };
}

// ---- CW (constant amplitude over [t_start, t_start + T_pulse]) -----
// Useful for analytic Rabi tests.  χ(t) = 1 inside the window.
inline PulseShape make_cw(double T_pulse, double t_start) {
    return [T_pulse, t_start](double t) {
        return (t >= t_start && t <= t_start + T_pulse) ? 1.0 : 0.0;
    };
}

}  // namespace mc_tdse
