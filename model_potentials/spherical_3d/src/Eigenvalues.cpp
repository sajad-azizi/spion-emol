#include "Eigenvalues.hpp"

Eigenvalues::Eigenvalues(Equations& eqs, const Parameters& params)
    : eqs_(eqs), params_(params)
{
}

void Eigenvalues::groundstate_finder(int desire_node, double tol)
{
    double e_L = params_.Emin;
    double e_H = params_.Emax;
    double e_tr = e_L;

    while (true) {
        auto [n_count, n_pos] = eqs_.OutwardNodeCounting(e_tr);
        if (n_count < desire_node) e_L = e_tr;
        else                       e_H = e_tr;

        if (std::fabs(e_H - e_L) < tol) {
            gsEnergy = e_tr;
            i_match  = matching_point_finder(e_tr);
            cout << "[gs] E = " << e_tr
                 << "  i_match = " << i_match
                 << "  nodes = " << n_count
                 << "  node_pos = " << n_pos << endl;
            print_turning_point_diagnostic(e_tr, i_match);
            return;
        }
        e_tr = 0.5 * (e_L + e_H);
    }
}

// Diagnostic: after matching_point_finder returns, verify that i_match
// actually corresponds to a physical classical turning point and that
// the surrounding region behaves correctly (V_s monotonically rises
// across the boundary in the simple-well case).  Prints a one-block
// summary and flags anything suspicious.
//
// What we check:
//   * r(i_match) is physically plausible (positive, well below r_max).
//   * V_s(i_match) ≈ E   (true turning point: V_s should cross E here).
//   * V_s is rising across i_match (so r=i_match is the OUTER turning
//     point, not a spurious local crossing from a multi-basin potential).
//   * i_match is comfortably inside [i_low, 2N/3] (the contamination-
//     free zone for forward Numerov).
//   * The bound state should be primarily concentrated for r < r_match;
//     we can't check the wavefunction yet (it's computed AFTER this
//     diagnostic) but we report kappa and the expected decay scale.
//
// The output is one block of lines prefixed with [gs::diag].
void Eigenvalues::print_turning_point_diagnostic(double E, int i_m) const
{
    const int    N    = params_.N_grid;
    const double dr   = params_.dr;
    const double r_m  = i_m * dr;
    const double r_max = (N - 1) * dr;

    const double Vs_m   = eqs_.Veff_swave(i_m);
    const double Vs_mm1 = (i_m - 1 >= 0)  ? eqs_.Veff_swave(i_m - 1) : Vs_m;
    const double Vs_mp1 = (i_m + 1 <  N)  ? eqs_.Veff_swave(i_m + 1) : Vs_m;

    // Asymptotic decay constant.  For a bound state, kappa^2 = 2*(V_inf - E).
    // Use V_s at i = N - 5 as a proxy for V_inf.
    const int i_far     = std::max(0, N - 5);
    const double V_inf  = eqs_.Veff_swave(i_far);
    const double kappa2 = 2.0 * (V_inf - E);
    const double kappa  = (kappa2 > 0.0) ? std::sqrt(kappa2) : 0.0;
    // 1/e decay length past the turning point.
    const double decay_len = (kappa > 0.0) ? 1.0 / kappa : 0.0;

    // Where, asymptotically, will |chi(r)| reach round-off (1e-16 vs 1)
    // starting from r_m?  This tells us roughly how far the wavefunction
    // is meaningful before the irregular companion takes over.
    const double r_underflow = r_m + (kappa > 0.0 ? (16.0 * std::log(10.0) / kappa) : 0.0);

    // Safe-zone bounds (must match matching_point_finder).
    const int i_low_safe  = std::max(6, params_.p + 1);
    const int i_high_safe = (2 * N) / 3;

    cout << "[gs::diag] turning-point sanity check:\n";
    cout << "[gs::diag]   r(i_match)   = " << r_m
         << "  (out of r_max = " << r_max << ")\n";
    cout << "[gs::diag]   V_s(i_match-1) = " << Vs_mm1
         << "   <classically allowed if < E>\n";
    cout << "[gs::diag]   V_s(i_match)   = " << Vs_m
         << "   E = " << E
         << "   delta = V_s - E = " << (Vs_m - E) << "\n";
    cout << "[gs::diag]   V_s(i_match+1) = " << Vs_mp1
         << "   <classically forbidden if > E>\n";
    cout << "[gs::diag]   V_inf (~ r_max) = " << V_inf
         << "   kappa = sqrt(2(V_inf-E)) = " << kappa
         << "   1/e decay length = " << decay_len << " bohr\n";
    cout << "[gs::diag]   safe zone for forward Numerov: i in ["
         << i_low_safe << ", " << i_high_safe << "]\n";
    cout << "[gs::diag]   r where |chi| would hit round-off (1e-16): r ~ "
         << r_underflow << " bohr\n";

    // Physical-plausibility flags.
    int   n_warn = 0;
    auto  warn = [&](const std::string& msg) {
        ++n_warn;
        cout << "[gs::diag]   WARNING: " << msg << "\n";
    };

    if (i_m < i_low_safe) {
        warn("i_match < i_low_safe -- forward seed region not yet stable.");
    }
    if (i_m > i_high_safe) {
        warn("i_match > 2N/3 -- forward propagation likely contaminated "
             "by exponentially growing irregular solution at this distance. "
             "Bound state will be unphysical (peak at r_max instead of r ~ a_0).");
    }
    // Classical turning point: V_s(i_match-1) should be < E (classically
    // allowed), V_s(i_match+1) should be > E (classically forbidden).
    if (Vs_mm1 > E) {
        warn("V_s(i_match-1) > E -- one step before i_match is already "
             "classically forbidden.  Turning point detection may have "
             "skipped past the true crossing.");
    }
    if (Vs_mp1 < E) {
        warn("V_s(i_match+1) < E -- one step after i_match is still "
             "classically allowed.  Turning point detection returned a "
             "premature crossing (multi-basin potential?).");
    }
    // Smoothness-aware grid-resolution check:
    // For a SMOOTH potential, V_s should vary slowly across a single
    // grid step, so V_s(i_match) ≈ E to within (V_s(i_m+1) - V_s(i_m-1))/2.
    // For a HARD-EDGE potential (cubic/spherical well, etc.), V_s
    // jumps discontinuously; the grid CAN'T resolve sub-step crossings
    // and the bracketing V_s(i_m) ≤ E < V_s(i_m+1) is the best we can
    // do.  Only flag a problem if V_s(i_match) is far from E AND far
    // from V_s(i_match+1) too -- i.e., neither side of the bracket is
    // anywhere near E (e.g., a multi-step jump or a bug in V_s).
    const double bracket_span = std::fabs(Vs_mp1 - Vs_m);
    const double delta_below  = std::fabs(Vs_m   - E);
    const double delta_above  = std::fabs(Vs_mp1 - E);
    const double smooth_thresh = std::max(1e-3, 0.05 * std::fabs(E));
    if (delta_below > smooth_thresh && delta_above > smooth_thresh
        && std::min(delta_below, delta_above) > bracket_span) {
        warn("Neither V_s(i_match) nor V_s(i_match+1) is close to E "
             "(min |V_s - E| > bracket span).  Suggests V_s is non-"
             "monotonic across i_match or the grid badly under-resolves "
             "the turning region.");
    }
    if (kappa <= 0.0) {
        warn("kappa^2 = 2(V_inf - E) <= 0  -- E is at or above V_inf; "
             "this is not a bound state.");
    }
    if (r_underflow > r_max) {
        warn("r_max is smaller than the decay distance to round-off. "
             "The exponential tail may not have decayed enough at r_max; "
             "consider increasing N_grid * dr.");
    }

    if (n_warn == 0) {
        cout << "[gs::diag]   OK -- turning point and surrounding region "
                "look physical.\n";
    } else {
        cout << "[gs::diag]   " << n_warn << " warning(s) above.\n";
    }
}

void Eigenvalues::Boundstates_finder(double tol)
{
    double e_L_outer = params_.Emin;

    auto [node_max, node_pos_max] = eqs_.OutwardNodeCounting(params_.Emax);
    cout << "[bound] node_max at Emax = " << node_max
         << "  pos = " << node_pos_max << endl;

    std::ofstream eout("eigenvalues.dat");
    eout << std::fixed << std::setprecision(15);
    int idx = 0;
    for (int desire_node = 1; desire_node < node_max; ++desire_node) {
        double e_L = e_L_outer;
        double e_H = params_.Emax;
        double e_tr = e_L;
        while (true) {
            auto [n_count, n_pos] = eqs_.OutwardNodeCounting(e_tr);
            if (n_count < desire_node) e_L = e_tr;
            else                       e_H = e_tr;
            if (std::fabs(e_H - e_L) < tol) {
                int i_m = (e_tr < 0.0) ? matching_point_finder(e_tr)
                                       : (params_.N_grid - 1);
                eout << idx++ << "\t" << e_tr << "\t" << i_m
                     << "\t" << n_pos << "\t" << n_count << "\n";
                cout << "[bound] E = " << e_tr
                     << "  i_match = " << i_m
                     << "  nodes = " << n_count << endl;
                e_L_outer = e_tr;
                break;
            }
            e_tr = 0.5 * (e_L + e_H);
        }
    }
}

// Find the OUTER classical turning point of the s-wave bound state at
// energy E.  Definition: the largest grid index ir for which the
// spherically-averaged effective potential V_s(r) ≤ E.  Beyond this
// point the wavefunction is classically forbidden and decays
// exponentially; before it the wavefunction is oscillatory/regular.
//
// This is THE textbook matching seam for two-sided Numerov propagation
// of a bound state.  Forward integration is stable inside the box
// (oscillatory regular solution from r=0); backward integration is
// stable outside (exponentially decaying solution from r=r_max).
// Matching at the turning point joins both clean integrations.
//
// HISTORICAL NOTE: the previous heuristic ("first uptick of
// |det(Rinv_fwd - Rinv_back)|") was fragile under round-off: for a
// near-eigenvalue E, both propagators agree to ~ε everywhere and the
// "first uptick" got driven by FP noise, frequently returning i_match
// near r_max.  At i_match ~ r_max, the forward propagation has
// accumulated the irregular companion solution (~exp(+κr) amplified
// by ~10^10 over the full grid) and the resulting bound state was
// dominated by that contamination -- producing the cross-section
// oscillations we observed in the spherical_3d outputs.
//
// The new method is V-based, deterministic, immune to FP noise, and
// matches the standard prescription in Johnson 1973 and every
// textbook bound-state Numerov code.
int Eigenvalues::matching_point_finder(double E)
{
    const int N      = params_.N_grid;
    const int i_low  = std::max(6, params_.p + 1);
    // Hard cap at 2/3 of the grid -- forward integration past this point
    // is increasingly polluted by the round-off-amplified irregular
    // solution in the classically-forbidden region.  Even on a wildly
    // shallow potential (where the turning point really is far out) we
    // prefer matching earlier and letting the backward sweep do the
    // remaining work.
    const int i_high = (2 * N) / 3;

    // Walk OUTWARD; find the largest ir in [i_low, i_high] where V_s(ir) <= E.
    int turning = -1;
    for (int i = i_low; i <= i_high; ++i) {
        if (eqs_.Veff_swave(i) <= E) {
            turning = i;
        }
        // Note: we don't break on the first crossing -- some potentials
        // (e.g. anisotropic wells whose s-wave projection has multiple
        // basins) can have V_s dip below E again briefly.  We want the
        // LAST classically allowed point before the final asymptotic
        // forbidden region.
    }

    if (turning < 0) {
        // V_s(r) > E everywhere in [i_low, i_high].  This is unusual for
        // a bound state (would mean E is below the s-wave well minimum
        // or the asymptotic V) -- fall back to a conservative point
        // well inside the contamination-free zone.
        return std::max(i_low, N / 3);
    }
    // Clamp to safe range (turning is already in [i_low, i_high]).
    return turning;
}
