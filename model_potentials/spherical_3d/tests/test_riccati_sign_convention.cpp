// test_riccati_sign_convention.cpp
//
// Verify the sign convention of riccati_S = +x·j_l(x) and
// riccati_C = -x·y_l(x) (defined in Angular.hpp) at the GSL boundary.
//
// The test uses gsl_sf_bessel_jl and gsl_sf_bessel_yl DIRECTLY (not via
// the Angular.hpp wrappers) so that any change to the wrapper or any
// version drift in GSL gets caught here.
//
// Standard DLMF / GSL / NIST convention:
//
//   spherical_j_l(x) → sin(x - lπ/2) / x      as x → ∞
//   spherical_y_l(x) → -cos(x - lπ/2) / x     as x → ∞    (note the MINUS)
//
// Therefore the spherical-3d code's Riccati definitions
//
//   S_l(x) = +x · gsl_sf_bessel_jl(l, x) → +sin(x - lπ/2)
//   C_l(x) = -x · gsl_sf_bessel_yl(l, x) → +cos(x - lπ/2)
//
// are correct.  Dropping the minus in riccati_C would flip the K-matrix
// sign (because K = B / A with ψ ~ A·S + B·C and tan δ_l = K), which in
// turn flips the sign of every scattering phase shift δ_l and the
// Wigner time delay τ_W computed via the dipole matrix element.
//
// Failure here is a STRUCTURAL bug: every (A, B) fit downstream that
// extracts a phase shift from the numerical wavefunction will be wrong.

#include "Angular.hpp"            // sph3d::ang3d::riccati_S, riccati_C
#include <gsl/gsl_sf_bessel.h>     // gsl_sf_bessel_jl, gsl_sf_bessel_yl

#include <cmath>
#include <cstdio>
#include <iostream>

using namespace ang3d;

static int fails = 0;
static void check_close(double got, double want, double tol,
                        const char* what)
{
    const double err = std::abs(got - want);
    if (err > tol) {
        std::cerr << "FAIL " << what
                  << " : got " << got << ", want " << want
                  << ", err=" << err << "\n";
        ++fails;
    } else {
        std::cout << "ok   " << what
                  << " : got=" << got << "  want=" << want
                  << "  err=" << err << "\n";
    }
}

int main()
{
    // Far-asymptotic regime: at x = 5e4 the leading 1/x correction to
    // y_l, j_l is ~2e-5 per partial wave; the centrifugal sub-leading
    // term l(l+1)/(2 x²) reaches ~1.4e-8 even at l=8.  A tolerance of
    // 1e-3 catches any sign flip (which would change the value by O(1))
    // with several orders of margin.
    constexpr double x       = 50000.0;
    constexpr double tol_abs = 1.0e-3;

    std::cout << "=== GSL asymptotic sign check at x = " << x << " ===\n\n";

    // (1) Direct gsl_sf_bessel_yl: expect -cos(x - lπ/2)/x.
    std::cout << "(1) GSL gsl_sf_bessel_yl(l, x) -> -cos(x - lπ/2)/x?\n";
    for (int l : {0, 1, 2, 3, 5, 8}) {
        const double yl_gsl   = gsl_sf_bessel_yl(l, x);
        const double expect   = -std::cos(x - 0.5 * M_PI * l) / x;
        char buf[80];
        std::snprintf(buf, sizeof(buf),
                      "gsl_sf_bessel_yl(%d, %.0f) -> -cos(x - lπ/2)/x", l, x);
        check_close(yl_gsl, expect, tol_abs / x, buf);
    }

    // (2) Direct gsl_sf_bessel_jl: expect +sin(x - lπ/2)/x.
    std::cout << "\n(2) GSL gsl_sf_bessel_jl(l, x) -> +sin(x - lπ/2)/x?\n";
    for (int l : {0, 1, 2, 3, 5, 8}) {
        const double jl_gsl   = gsl_sf_bessel_jl(l, x);
        const double expect   = +std::sin(x - 0.5 * M_PI * l) / x;
        char buf[80];
        std::snprintf(buf, sizeof(buf),
                      "gsl_sf_bessel_jl(%d, %.0f) -> +sin(x - lπ/2)/x", l, x);
        check_close(jl_gsl, expect, tol_abs / x, buf);
    }

    // (3) Wrapper riccati_S / riccati_C agree with the standard
    // S = +sin, C = +cos asymptotes.
    std::cout << "\n(3) Angular.hpp wrappers: S_l -> +sin, C_l -> +cos?\n";
    for (int l : {0, 1, 2, 3, 5, 8}) {
        const double S       = riccati_S(l, x);
        const double C       = riccati_C(l, x);
        const double phase   = x - 0.5 * M_PI * l;
        const double exp_S   = +std::sin(phase);   // +sin
        const double exp_C   = +std::cos(phase);   // +cos  (NOT -cos!)
        char buf[80];
        std::snprintf(buf, sizeof(buf), "S_%d(x=%.0f) -> +sin(x-lπ/2)", l, x);
        check_close(S, exp_S, tol_abs, buf);
        std::snprintf(buf, sizeof(buf), "C_%d(x=%.0f) -> +cos(x-lπ/2)", l, x);
        check_close(C, exp_C, tol_abs, buf);
    }

    // (4) Cross-check: riccati_C == -x * gsl_sf_bessel_yl (the literal
    //     definition).  Catches a wrapper drift even if the asymptote
    //     test above happens to pass by accident.
    std::cout << "\n(4) Wrapper formula consistency riccati_C(l,x) == -x*yl?\n";
    for (int l : {0, 1, 2, 3, 5, 8}) {
        const double C_wrap  = riccati_C(l, x);
        const double C_lit   = -x * gsl_sf_bessel_yl(l, x);
        char buf[80];
        std::snprintf(buf, sizeof(buf), "riccati_C(%d, x) == -x * gsl_sf_bessel_yl(%d, x)", l, l);
        check_close(C_wrap, C_lit, 1.0e-12, buf);
    }

    // (5) Sign assertion: riccati_C must match +cos, NOT -cos.  If
    //     someone deletes the minus in riccati_C this branch trips.
    std::cout << "\n(5) Sign assertion: C_l closer to +cos than to -cos?\n";
    for (int l : {0, 1, 2, 3}) {
        const double C  = riccati_C(l, x);
        const double pc = std::cos(x - 0.5 * M_PI * l);
        const double nc = -pc;
        if (std::abs(C - pc) > std::abs(C - nc)) {
            std::cerr << "FAIL C_" << l << " matches -cos rather than +cos -- "
                         "minus sign in Angular.hpp::riccati_C was dropped.\n";
            ++fails;
        } else {
            std::cout << "ok   C_" << l << " matches +cos (minus sign in "
                         "Angular.hpp::riccati_C is intact)\n";
        }
    }

    std::cout << "\n";
    if (fails == 0) std::cout << "PASS  test_riccati_sign_convention\n";
    else            std::cerr << "FAIL  test_riccati_sign_convention ("
                              << fails << " failures)\n";
    return fails == 0 ? 0 : 1;
}
