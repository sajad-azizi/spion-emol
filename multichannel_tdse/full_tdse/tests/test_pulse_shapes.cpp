// test_pulse_shapes.cpp -- envelope sanity:
//   * Gaussian:    χ(t_c) = 1, χ(t_c ± τ) = exp(-1/2);  ∫χ² = √π τ
//   * sin²:        χ(t_start + T/2) = 1, χ outside support = 0;
//                  ∫χ² = 3T/8 (closed form).
//   * flat-top:    flat segment value = 1; outside support = 0.
//
// Numerical integrals use Simpson; tolerance 1e-8 (Simpson on smooth
// shapes for the Gaussian / sin² cases).
#include "Common.hpp"
#include "Pulse.hpp"

#include <cstdio>

namespace {

double simpson(const std::function<double(double)>& f, double a, double b, int n) {
    if (n % 2) ++n;
    const double h = (b - a) / n;
    double s = f(a) + f(b);
    for (int i = 1; i < n; ++i) {
        s += (i & 1 ? 4.0 : 2.0) * f(a + i * h);
    }
    return s * h / 3.0;
}

bool check_close(double got, double want, double tol, const char* what) {
    const double err = std::fabs(got - want);
    const double ok  = err < tol;
    std::printf("    %-40s got=%+.6e  want=%+.6e  err=%.2e  %s\n",
                what, got, want, err, ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    using namespace mc_tdse;
    int n_fail = 0;

    // ---- Gaussian -----------------------------------------------
    std::printf("[Gaussian]  τ = 1.0, t_c = 0.0\n");
    {
        auto chi = make_gaussian(1.0, 0.0);
        if (!check_close(chi(0.0),     1.0,                                 1e-12, "chi(t_c) == 1"))
            ++n_fail;
        if (!check_close(chi(1.0),     std::exp(-0.5),                      1e-12, "chi(t_c + τ) == e^{-1/2}"))
            ++n_fail;
        if (!check_close(chi(-1.0),    std::exp(-0.5),                      1e-12, "chi(t_c - τ) == e^{-1/2}  (symmetry)"))
            ++n_fail;
        const double I_chi2_num = simpson(
            [&](double t) { const double v = chi(t); return v * v; },
            -10.0, 10.0, 4000);
        if (!check_close(I_chi2_num, gaussian_int_chi2(1.0),                1e-9,
                         "∫chi²dt == √π τ"))
            ++n_fail;
        if (!check_close(gaussian_FWHM(1.0),
                         2.0 * std::sqrt(2.0 * std::log(2.0)),               1e-12,
                         "FWHM == 2√(2 ln 2) τ"))
            ++n_fail;
    }

    // ---- sin² ---------------------------------------------------
    std::printf("\n[sin²]   T = 4.0, t_start = 1.0\n");
    {
        const double T = 4.0, t0 = 1.0;
        auto chi = make_sin_squared(T, t0);
        if (!check_close(chi(t0 + 0.5 * T), 1.0,                               1e-12, "chi(t_mid) == 1"))
            ++n_fail;
        if (!check_close(chi(t0 - 0.1),     0.0,                               1e-12, "chi(before) == 0"))
            ++n_fail;
        if (!check_close(chi(t0 + T + 0.1), 0.0,                               1e-12, "chi(after) == 0"))
            ++n_fail;
        const double I_chi2_num = simpson(
            [&](double t) { const double v = chi(t); return v * v; },
            t0, t0 + T, 4000);
        if (!check_close(I_chi2_num, sin_squared_int_chi2(T),                  1e-9,
                         "∫chi²dt == 3T/8"))
            ++n_fail;
        if (!check_close(sin_squared_FWHM(T), 0.5 * T,                        1e-12, "FWHM == T/2"))
            ++n_fail;
    }

    // ---- flat-top -----------------------------------------------
    std::printf("\n[flat-top]  T = 10.0, t_start = 0.0, ramp = 2.0\n");
    {
        auto chi = make_flat_top(10.0, 0.0, 2.0);
        if (!check_close(chi(5.0),  1.0, 1e-12, "chi(middle) == 1"))     ++n_fail;
        if (!check_close(chi(2.0),  1.0, 1e-12, "chi(end of ramp) == 1"))    ++n_fail;
        if (!check_close(chi(8.0),  1.0, 1e-12, "chi(start of ramp-down) == 1"))  ++n_fail;
        if (!check_close(chi(0.0),  0.0, 1e-12, "chi(t_start) == 0 (ramp start)"))    ++n_fail;
        if (!check_close(chi(10.0), 0.0, 1e-12, "chi(t_start+T) == 0"))   ++n_fail;
        if (!check_close(chi(1.0),  std::sin(0.25 * M_PI) * std::sin(0.25 * M_PI),
                         1e-12, "chi(half-ramp) == sin²(π/4) = 1/2"))    ++n_fail;
    }

    std::printf("\nTotal failures: %d\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
