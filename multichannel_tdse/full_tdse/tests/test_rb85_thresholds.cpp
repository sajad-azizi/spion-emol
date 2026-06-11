// test_rb85_thresholds.cpp -- recipe Sec. 1.1 reference values.
//
// At B = 155.04 G, the lowest two-channel pair thresholds (in MHz,
// referenced to the M_F = -4 entrance threshold) per the recipe are:
//
//   M_F = -4:  {0.00,    2605.76}
//   M_F = -3:  {-77.42,  2532.24}
//   M_F = -5:  {2683.18, 5448.24}
//
// (M_F = -2 not given numerically by the recipe but we sanity-check
//  it has a sensible structure.)
//
// We tolerate a few-MHz absolute error in the comparison: the recipe
// rounded its values, and the SpinAlgebra reference implementation
// uses CODATA 2018 with mu_B = 1.3996...MHz/G, g_J = 2.00233...
//
// This test ALSO verifies that channel ordering returned by
// `channels(MF)` is sorted by threshold ascending.
#include "Rb85Spin.hpp"

#include <cstdio>
#include <vector>

namespace {

bool check_threshold(const std::vector<mc_tdse::ChannelInfo>& chs,
                     int idx, double want_MHz, double tol_MHz,
                     const char* label)
{
    if (idx >= static_cast<int>(chs.size())) {
        std::printf("    FAIL  %s: only %zu channels, idx %d out of range\n",
                    label, chs.size(), idx);
        return false;
    }
    const double got = chs[idx].E_th_MHz;
    const double err = std::fabs(got - want_MHz);
    const bool   ok  = err < tol_MHz;
    std::printf("    %-26s got=%+10.4f MHz  want=%+10.4f MHz  err=%6.3f MHz  %s\n",
                label, got, want_MHz, err, ok ? "ok" : "FAIL");
    return ok;
}

bool check_sorted(const std::vector<mc_tdse::ChannelInfo>& chs,
                  const char* label)
{
    for (std::size_t i = 1; i < chs.size(); ++i) {
        if (chs[i].E_th_MHz < chs[i-1].E_th_MHz - 1e-9) {
            std::printf("    FAIL  %s: channel %zu (%.3f MHz) < %zu (%.3f MHz)\n",
                        label, i, chs[i].E_th_MHz, i-1, chs[i-1].E_th_MHz);
            return false;
        }
    }
    std::printf("    %-26s sorted ascending  (%zu channels)  ok\n",
                label, chs.size());
    return true;
}

}  // namespace

int main() {
    using namespace mc_tdse;
    int n_fail = 0;

    Rb85Spin spin(155.04);

    auto chs_m4 = spin.channels(-4);
    auto chs_m3 = spin.channels(-3);
    auto chs_m5 = spin.channels(-5);
    auto chs_m2 = spin.channels(-2);

    std::printf("[Rb85 thresholds]  B=%.2f G\n", spin.B_gauss());

    std::printf("\n  channel counts:\n");
    std::printf("    M_F = -2  : %zu channels\n", chs_m2.size());
    std::printf("    M_F = -3  : %zu channels\n", chs_m3.size());
    std::printf("    M_F = -4  : %zu channels\n", chs_m4.size());
    std::printf("    M_F = -5  : %zu channels\n", chs_m5.size());

    std::printf("\n  recipe-quoted thresholds (tol = 5 MHz to absorb recipe rounding):\n");
    if (!check_threshold(chs_m4, 0, 0.00,    5.0, "M_F=-4 lowest"))  ++n_fail;
    if (!check_threshold(chs_m4, 1, 2605.76, 5.0, "M_F=-4 second"))  ++n_fail;
    if (!check_threshold(chs_m3, 0, -77.42,  5.0, "M_F=-3 lowest"))  ++n_fail;
    if (!check_threshold(chs_m3, 1, 2532.24, 5.0, "M_F=-3 second"))  ++n_fail;
    if (!check_threshold(chs_m5, 0, 2683.18, 5.0, "M_F=-5 lowest"))  ++n_fail;
    if (!check_threshold(chs_m5, 1, 5448.24, 5.0, "M_F=-5 second"))  ++n_fail;

    std::printf("\n  channel-ordering sanity:\n");
    if (!check_sorted(chs_m2, "M_F=-2 channels"))  ++n_fail;
    if (!check_sorted(chs_m3, "M_F=-3 channels"))  ++n_fail;
    if (!check_sorted(chs_m4, "M_F=-4 channels"))  ++n_fail;
    if (!check_sorted(chs_m5, "M_F=-5 channels"))  ++n_fail;

    std::printf("\n  full M_F=-4 spectrum (first 6, in MHz from entrance):\n");
    for (std::size_t i = 0; i < std::min<std::size_t>(6, chs_m4.size()); ++i) {
        std::printf("    [%zu]  %s   E_th = %+12.4f MHz\n",
                    i, chs_m4[i].label.c_str(), chs_m4[i].E_th_MHz);
    }

    std::printf("\nTotal failures: %d\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
