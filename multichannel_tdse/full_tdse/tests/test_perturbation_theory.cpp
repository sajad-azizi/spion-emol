// test_perturbation_theory.cpp
//
// Validates the closed-form 1st- + 2nd-order PT module against the
// full TDSE.  Three concentric checks:
//
//   1. Dawson function matches reference values to 1e-12.
//   2. PT scaling: at half Ω_R, |b^(1)|² scales by 1/4, |b^(2)|² by 1/16,
//      independent of any TDSE comparison (pure algebraic test of the
//      closed-form expression).
//   3. TDSE-vs-PT agreement: in deep-PT (Ω_R = 2π·0.01 kHz) the
//      eigenstate-resolved populations from `propagate_pooled` agree
//      with `compute_pt` to ~Ω_R²-level relative error.
//
// The basis is the same toy 4-block pool (E_cut = 1 kHz) used by
// test_pooled_tdse_toy.cpp, so this is fast (~5 s).
#include "PerturbationTheory.hpp"
#include "PooledBasis.hpp"
#include "PooledTDSE.hpp"
#include "Pulse.hpp"
#include "Rb85Spin.hpp"
#include "Common.hpp"

#include <cmath>
#include <cstdio>

using namespace mc_tdse;

namespace {

// Reference Dawson values from scipy.special.dawsn.
struct DawsonRef { double x, F; };
constexpr DawsonRef DAWSON_REFS[] = {
    {   0.0,   0.0000000000000000e+00 },
    {   0.5,   4.2443638350202229e-01 },
    {   1.0,   5.3807950691276840e-01 },
    {   1.5,   4.2824907108539867e-01 },
    {   2.0,   3.0134038892379200e-01 },
    {   3.0,   1.7827103061055827e-01 },
    {   5.0,   1.0213407442427686e-01 },
    {  10.0,   5.0253847187598538e-02 },
    {  -1.0,  -5.3807950691276840e-01 },
    {  -3.0,  -1.7827103061055827e-01 },
};

}  // namespace

int main() {
    int n_fail = 0;

    // ============================================================== //
    // 1. Dawson function spot-check                                   //
    // ============================================================== //
    std::printf("[1] Dawson function reference values\n");
    for (auto r : DAWSON_REFS) {
        const double F   = pt::dawson(r.x);
        const double err = std::fabs(F - r.F);
        const bool ok    = err < 1.0e-12 || err < 1e-12 * std::fabs(r.F);
        std::printf("    F(%6.2f) = %15.12f  (ref %15.12f, err %.2e)  %s\n",
                    r.x, F, r.F, err, ok ? "PASS" : "FAIL");
        if (!ok) ++n_fail;
    }

    // ============================================================== //
    // 2. Build toy basis once for the remaining checks                //
    // ============================================================== //
    std::printf("\n[2] Building pooled basis (E_cut spans the 1γ resonance ω≈80 kHz)\n");
    Rb85Spin spin(155.04);
    BlockBuildOptions base;
    base.N_grid = 20000;                            // small box (L = 1e4 a_0)
    base.dr_a0  = 0.5;
    base.E_max_kHz_above_threshold = 200.0;         // captures 1γ at ω-E_b ≈ 70 kHz
    std::vector<int> MFs = {-5, -4, -3, -2};
    std::vector<BlockBuildOptions> per_block(4, base);
    per_block[0].E_max_kHz_above_threshold = 1.0;   // M_F=-5 stays 'virtual'
    per_block[1].use_analytic_halo = true;
    PooledBasis pb = build_pooled_basis(MFs, per_block, spin);
    std::printf("    block sizes: ");
    for (size_t k = 0; k < pb.blocks.size(); ++k)
        std::printf("M=%+d:%d  ", pb.block_MFs[k], pb.blocks[k].n_states());
    std::printf("(N_total=%d)\n", pb.N_total);

    auto make_cfg = [&](double Omega_R_kHz) {
        PooledTDSEConfig cfg;
        cfg.chi        = make_gaussian(AU::us_to_au(30.0), 0.0);
        cfg.omega_au   = AU::kHz_to_au(80.896);
        cfg.Omega_R_au = 2.0 * M_PI * AU::kHz_to_au(Omega_R_kHz);
        // Use ±5τ so the PT closed form (which integrates from -∞ to +∞)
        // is matched to ~1 − erf(5/√2) ≈ 6e-7 of the pulse area.
        cfg.t_start    = AU::us_to_au(-150.0);
        cfg.t_end      = AU::us_to_au(+150.0);
        cfg.dt         = AU::us_to_au(0.5);
        return cfg;
    };
    pt::PTConfig ptc;
    ptc.initial_block      = 1;     // M_F=-4
    ptc.initial_state      = 0;     // halo
    ptc.tau_au             = AU::us_to_au(30.0);
    ptc.t_center_au        = 0.0;
    ptc.compute_2nd_order  = true;

    // ============================================================== //
    // 3. Closed-form scaling: P_pt(Ω_R) ∝ Ω_R² (b1) or Ω_R⁴ (b2)      //
    // ============================================================== //
    std::printf("\n[3] PT closed-form scaling test (no TDSE involved)\n");
    auto cfg_1 = make_cfg(1.0);
    auto cfg_2 = make_cfg(2.0);
    auto pt_1  = pt::compute_pt(pb, cfg_1, ptc);
    auto pt_2  = pt::compute_pt(pb, cfg_2, ptc);
    // Pick a most-populated state in each block (excluding halo) and
    // verify the predicted populations scale as expected.
    auto top_state_in_block = [&](int kk) -> int {
        const int off = pb.block_offset[kk];
        const int n   = pb.blocks[kk].n_states();
        const int halo_idx = pb.block_offset[ptc.initial_block] + ptc.initial_state;
        int    best = -1;
        double bp   = 0.0;
        for (int i = 0; i < n; ++i) {
            if (off + i == halo_idx) continue;
            const double p = pt_1.prob_pt(off + i);
            if (p > bp) { bp = p; best = off + i; }
        }
        return best;
    };
    for (size_t k = 0; k < pb.blocks.size(); ++k) {
        const int dM  = pb.block_MFs[k] - pb.block_MFs[ptc.initial_block];
        const int idx = top_state_in_block(static_cast<int>(k));
        if (idx < 0) continue;
        const double P1 = pt_1.prob_pt(idx);
        const double P2 = pt_2.prob_pt(idx);
        const int    n_expected = (std::abs(dM) == 1) ? 2 : 4;
        const double ratio_pred = std::pow(2.0, n_expected);
        const double ratio_obs  = (P1 > 0) ? P2 / P1 : 0.0;
        const double err  = std::fabs(ratio_obs - ratio_pred) / ratio_pred;
        const bool   ok   = err < 1.0e-10;
        std::printf("    M_F=%+d  top state idx=%d  P(Ω=1)=%.3e  P(Ω=2)=%.3e  "
                    "ratio %.3f (pred %d) %s\n",
                    pb.block_MFs[k], idx, P1, P2, ratio_obs, (int)ratio_pred,
                    ok ? "PASS" : "FAIL");
        if (!ok) ++n_fail;
    }

    // ============================================================== //
    // 4. TDSE vs PT in deep-PT regime (Ω_R = 2π × 0.1 kHz)            //
    //    Pulse area ≈ 0.05 rad — leading-order PT regime             //
    // ============================================================== //
    std::printf("\n[4] TDSE-vs-PT agreement at deep PT (Ω_R = 2π·0.1 kHz)\n");
    auto cfg_pt = make_cfg(0.1);
    Eigen::VectorXcd c0 = Eigen::VectorXcd::Zero(pb.N_total);
    const int halo_idx = pb.block_offset[ptc.initial_block] + ptc.initial_state;
    c0(halo_idx) = 1.0;

    PooledTDSEStats stats;
    Eigen::VectorXcd c_T = propagate_pooled(pb, c0, cfg_pt, &stats);
    auto pt_res = pt::compute_pt(pb, cfg_pt, ptc);

    auto P_block_tdse = block_populations(pb, c_T);

    // Compare PER-BLOCK totals first (smooth aggregate; insensitive to
    // per-state phase artefacts).  Skip the halo's own block — its
    // value is dominated by 1, and the 2nd-order Stark shift correction
    // |1 + b2|² requires Re(b2) which is sensitive to the high-energy
    // tail of the basis.
    std::printf("    block totals (TDSE vs PT-prob_pt sum):\n");
    for (size_t k = 0; k < pb.blocks.size(); ++k) {
        if (static_cast<int>(k) == ptc.initial_block) continue;
        const int off = pb.block_offset[k];
        const int n   = pb.blocks[k].n_states();
        double S_pt = 0.0;
        for (int i = 0; i < n; ++i) S_pt += pt_res.prob_pt(off + i);
        const double S_tdse = P_block_tdse[k];
        // Skip blocks where PT predicts negligible population (the basis
        // doesn't span the resonance, so there is no signal to compare).
        // The M_F=-5 block has E_th ≈ +2.7 GHz so the σ⁻ resonance is
        // huge-detuned — PT correctly predicts ~0.
        if (S_pt < 1.0e-20) {
            std::printf("        M_F=%+d  PT predicts negligible (%.2e); skipping\n",
                        pb.block_MFs[k], S_pt);
            continue;
        }
        const double rel = std::fabs(S_tdse - S_pt) / S_pt;
        // Tolerance:
        //   1γ (M_F = halo±1): 1st-order PT, basis-truncation only → 0.1% achievable
        //   2γ / ZEPE (|ΔM_F|=2 or 0): sum over 6 intermediates → 2-state-basis
        //     truncation tail leaks ~1% off the closed form
        const int dM = std::abs(pb.block_MFs[k] - pb.block_MFs[ptc.initial_block]);
        const double tol = (dM == 1) ? 1.0e-3 : 2.0e-2;
        const bool   ok  = rel < tol;
        std::printf("        M_F=%+d  TDSE %.5e   PT %.5e   rel %.2e   %s\n",
                    pb.block_MFs[k], S_tdse, S_pt, rel, ok ? "PASS" : "FAIL");
        if (!ok) ++n_fail;
    }

    // Per-state agreement on the TOP-population states of each block
    // where PT predicts > 1e-20 total population.
    std::printf("\n    top-state per-state agreement (skip halo block, skip noise floor):\n");
    for (size_t k = 0; k < pb.blocks.size(); ++k) {
        if (static_cast<int>(k) == ptc.initial_block) continue;
        const int off = pb.block_offset[k];
        const int n   = pb.blocks[k].n_states();
        double S_pt_block = 0.0;
        for (int i = 0; i < n; ++i) S_pt_block += pt_res.prob_pt(off + i);
        if (S_pt_block < 1.0e-20) continue;
        std::vector<int> idxs(n);
        for (int i = 0; i < n; ++i) idxs[i] = off + i;
        std::sort(idxs.begin(), idxs.end(), [&](int a, int b) {
            return pt_res.prob_pt(a) > pt_res.prob_pt(b);
        });
        const int n_show = std::min(3, n);
        for (int j = 0; j < n_show; ++j) {
            const int idx = idxs[j];
            const double P_t  = std::norm(c_T(idx));
            const double P_pt = pt_res.prob_pt(idx);
            if (P_pt < 1.0e-15) continue;       // noise floor for individual state
            const double rel = std::fabs(P_t - P_pt) / P_pt;
            const int dM = std::abs(pb.block_MFs[k] - pb.block_MFs[ptc.initial_block]);
            const double tol = (dM == 1) ? 1.0e-3 : 2.0e-2;
            const bool ok = rel < tol;
            std::printf("        M_F=%+d  state %d  TDSE %.4e  PT %.4e  rel %.2e  %s\n",
                        pb.block_MFs[k], idx - off, P_t, P_pt, rel,
                        ok ? "PASS" : "FAIL");
            if (!ok) ++n_fail;
        }
    }

    std::printf("\nTotal failures: %d\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
