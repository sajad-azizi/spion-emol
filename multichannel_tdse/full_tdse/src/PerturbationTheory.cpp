#include "PerturbationTheory.hpp"

#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace mc_tdse::pt {

// ---------------------------------------------------------------- //
// Dawson function
// ---------------------------------------------------------------- //
double dawson(double x) {
    if (x == 0.0) return 0.0;
    const double ax = std::abs(x);

    // Power series  F(x) = x · Σ_{n=0}^∞ c_n x^{2n}  with  c_0 = 1,
    // c_{n+1}/c_n = -2/(2n+3).  Catastrophic cancellation kicks in
    // for |x| ≳ 2.5; switch to quadrature there.
    if (ax < 2.5) {
        const double x2 = x * x;
        double term = 1.0;
        double sum  = 1.0;
        for (int n = 0; n < 60; ++n) {
            term *= -2.0 * x2 / (2 * n + 3);
            sum  += term;
            if (std::abs(term) < 1.0e-17 * std::abs(sum)) break;
        }
        return x * sum;
    }
    // Asymptotic  F(x) ~ 1/(2x) [1 + 1/(2x²) + 3/(2x²)² + ...]
    if (ax >= 8.0) {
        const double inv_2x2 = 1.0 / (2.0 * x * x);
        double term = 1.0;
        double sum  = 1.0;
        double prev = 1.0;
        for (int n = 1; n < 30; ++n) {
            term *= (2 * n - 1) * inv_2x2;
            const double a = std::abs(term);
            if (a > prev) break;            // asymptotic divergence
            prev = a;
            sum += term;
        }
        return sum / (2.0 * x);
    }
    // Mid-range 2.5 ≤ |x| < 8: composite Simpson on the SHIFTED form
    //   ∫_0^x e^{t²-x²} dt = ∫_0^x e^{-2xu+u²} du   (sub u = x − t)
    // The integrand here is monotone and convex in [0, x], starting at
    // 1 and decaying to e^{-x²}.  Simpson error ~ x·h⁴·(12 + 96x² + 16x⁴)/180.
    // N = 8192 gives < 1e-13 relative at x = 8.
    constexpr int N = 8192;
    const double h    = ax / N;
    const double m2x  = -2.0 * ax;
    double s = 1.0 + std::exp(-ax * ax);          // u=0 and u=x endpoints
    for (int i = 1; i < N; ++i) {
        const double u = i * h;
        const double v = std::exp(m2x * u + u * u);
        s += (i % 2 ? 4.0 : 2.0) * v;
    }
    return std::copysign(s * h / 3.0, x);
}

// ---------------------------------------------------------------- //
// Compute PT amplitudes
// ---------------------------------------------------------------- //
namespace {

// Locate pooled-block index k with block_MFs[k] == M.  Returns -1 if absent.
int find_block(const PooledBasis& pb, int M) {
    for (size_t k = 0; k < pb.block_MFs.size(); ++k)
        if (pb.block_MFs[k] == M) return static_cast<int>(k);
    return -1;
}

// 2-photon double-Gaussian-phase integral, closed form via Dawson F.
//
//   I(Δ_in, Δ_out, τ, t_c) = ∫dt χ(t) e^{iΔ_out t} ∫_{-∞}^t dt' χ(t') e^{iΔ_in t'}
//                          = πτ² · [exp(-(Δ_in²+Δ_out²)τ²/2)
//                                   + (2i/√π) exp(-Δ_total²τ²/4) · F((Δ_out-Δ_in)τ/2)]
//                            · e^{i Δ_total t_c}
//
// where χ(t) = exp(-(t-t_c)²/2τ²).  The phase factor at the end is the
// only place where t_c (≠ 0) enters: shifting the pulse by t_c
// multiplies a(E) by e^{iΔ_total t_c}; the |a|² is unchanged.
inline dcompx I_PQ_closed(double Delta_in, double Delta_out,
                          double tau, double t_center) {
    const double tau2  = tau * tau;
    const double D_tot = Delta_in + Delta_out;
    const double exp1  = std::exp(-0.5 * (Delta_in*Delta_in + Delta_out*Delta_out) * tau2);
    const double exp2  = std::exp(-0.25 * D_tot * D_tot * tau2);
    const double Farg  = (Delta_out - Delta_in) * 0.5 * tau;
    const double Fval  = dawson(Farg);
    constexpr double TWO_OVER_SQRTPI = 1.1283791670955125738961589031215;
    const dcompx core  = exp1 + I_unit * (TWO_OVER_SQRTPI * exp2 * Fval);
    const dcompx phase = std::exp(I_unit * D_tot * t_center);
    return M_PI * tau2 * core * phase;
}

}  // namespace

PTAmplitudes compute_pt(const PooledBasis&      pb,
                         const PooledTDSEConfig& cfg,
                         const PTConfig&         pt) {
    if (pt.initial_block < 0
        || pt.initial_block >= static_cast<int>(pb.blocks.size()))
        throw std::runtime_error("compute_pt: bad initial_block");
    if (pt.initial_state < 0
        || pt.initial_state >= pb.blocks[pt.initial_block].n_states())
        throw std::runtime_error("compute_pt: bad initial_state");
    if (pt.tau_au <= 0.0)
        throw std::runtime_error("compute_pt: tau_au must be > 0 (Gaussian only)");

    const int    N        = pb.N_total;
    const int    M_halo   = pb.block_MFs[pt.initial_block];
    const int    halo_off = pb.block_offset[pt.initial_block];
    const int    halo_loc = pt.initial_state;
    const int    halo_idx = halo_off + halo_loc;
    const double E_h_rel  = pb.E_au_block_rel[halo_idx];
    const double omega    = cfg.omega_au;
    const double half_OmR = 0.5 * cfg.Omega_R_au;
    const double tau      = pt.tau_au;
    const double tau2_2   = 0.5 * tau * tau;
    const double sqrt_2pi_tau = std::sqrt(2.0 * M_PI) * tau;
    const double t_c      = pt.t_center_au;

    PTAmplitudes R;
    R.b1      = Eigen::VectorXcd::Zero(N);
    R.b2      = Eigen::VectorXcd::Zero(N);
    R.prob_pt = Eigen::VectorXd::Zero(N);

    // ---------------------------------------------------------- //
    // 1st order: halo --σ^P--> α                                  //
    // ---------------------------------------------------------- //
    auto first_order_branch = [&](int P) {
        // P = +1 (σ⁺ absorption to M_halo+1) or -1 (σ⁻ emission to M_halo-1).
        // pair_index_of_low_MF takes the LOWER M_F of the (low, high) pair:
        //   P = +1: low = M_halo, high = M_halo+1 → pair stores (M_halo)
        //   P = -1: low = M_halo-1, high = M_halo → pair stores (M_halo-1)
        const int M_low_pair = (P == +1) ? M_halo : (M_halo - 1);
        const int kp        = pb.pair_index_of_low_MF(M_low_pair);
        if (kp < 0) return;
        const auto& dp_mat   = pb.d_plus_pair[kp];           // (N_high, N_low)
        const int   k_target = (P == +1) ? (kp + 1) : kp;    // block of α
        const int   off_t    = pb.block_offset[k_target];
        const int   N_t      = pb.blocks[k_target].n_states();
        // Off-resonance cutoff: |Δ·τ| > CUT means exp(-Δ²τ²/2) < e^{-CUT²/2}.
        // CUT = 12 → 2e-32, far below double precision noise.
        constexpr double CUT_PER_SIGMA = 12.0;
        const double Delta_max = CUT_PER_SIGMA / tau;
        #pragma omp parallel for schedule(static)
        for (int alpha = 0; alpha < N_t; ++alpha) {
            const int    idx     = off_t + alpha;
            const double E_a_rel = pb.E_au_block_rel[idx];
            const double Delta   = E_a_rel - E_h_rel - P * omega;
            if (std::abs(Delta) > Delta_max) continue;       // exp suppression > 12σ
            // Matrix element halo --σ^P--> α:
            //   P=+1: (high, low) = (α∈block_high, halo∈block_low)
            //         d⁺_{α,halo} = dp_mat(alpha, halo_loc)
            //   P=-1: (high, low) = (halo∈block_high, α∈block_low)
            //         σ⁻ element = conj(dp_mat(halo_loc, alpha))
            const dcompx d_el = (P == +1)
                              ? dp_mat(alpha, halo_loc)
                              : std::conj(dp_mat(halo_loc, alpha));
            const dcompx amp  = -I_unit * half_OmR * d_el * sqrt_2pi_tau
                              * std::exp(-Delta * Delta * tau2_2)
                              * std::exp(I_unit * Delta * t_c);
            // Different α indices, no race; +=  is also fine.
            R.b1(idx) = amp;     // 1st-order branch is a unique write per α.
        }
    };
    first_order_branch(+1);
    first_order_branch(-1);

    if (!pt.compute_2nd_order) {
        // Build prob_pt with b2 = 0.
        for (size_t k = 0; k < pb.block_MFs.size(); ++k) {
            const int dM  = pb.block_MFs[k] - M_halo;
            const int off = pb.block_offset[k];
            const int n   = pb.blocks[k].n_states();
            for (int i = 0; i < n; ++i) {
                const int idx = off + i;
                if (idx == halo_idx)        R.prob_pt(idx) = 1.0;
                else if (std::abs(dM) == 1) R.prob_pt(idx) = std::norm(R.b1(idx));
                else                        R.prob_pt(idx) = 0.0;
            }
        }
        return R;
    }

    // ---------------------------------------------------------- //
    // 2nd order: halo --σ^P--> k --σ^Q--> β                       //
    // ---------------------------------------------------------- //
    const double prefactor = -half_OmR * half_OmR;     // (-i)²·(Ω_R/2)² = -(Ω_R/2)²

    struct PathwayInfo {
        int P, Q;
        int k_inter, k_final;
    };
    std::vector<PathwayInfo> pathways;
    for (int P : {+1, -1}) {
        const int M_inter = M_halo + P;
        const int k_inter = find_block(pb, M_inter);
        if (k_inter < 0) continue;
        for (int Q : {+1, -1}) {
            const int M_final = M_inter + Q;
            const int k_final = find_block(pb, M_final);
            if (k_final < 0) continue;
            // Need both halo→inter (pair_low = min(M_halo, M_inter)) AND
            // inter→final (pair_low = min(M_inter, M_final)) pair indices.
            const int pair_in  = pb.pair_index_of_low_MF(std::min(M_halo, M_inter));
            const int pair_out = pb.pair_index_of_low_MF(std::min(M_inter, M_final));
            if (pair_in < 0 || pair_out < 0) continue;
            pathways.push_back({P, Q, k_inter, k_final});
        }
    }

    // Per-pathway accumulation into R.b2.  Each pathway (P,Q) targets
    // a specific final-block; two pathways may target the SAME final
    // block (P=+1,Q=-1 and P=-1,Q=+1 both target M_halo).  No race:
    // the OpenMP parallel-for is over β within a single pathway, and
    // the outer loop over pathways is sequential.
    for (const auto& pw : pathways) {
        const int P = pw.P, Q = pw.Q;
        const int M_inter = M_halo + P;
        const int M_final = M_inter + Q;
        (void)M_final;

        const int N_inter = pb.blocks[pw.k_inter].n_states();
        const int N_final = pb.blocks[pw.k_final].n_states();
        const int off_inter = pb.block_offset[pw.k_inter];
        const int off_final = pb.block_offset[pw.k_final];

        const int pair_in  = pb.pair_index_of_low_MF(std::min(M_halo, M_inter));
        const int pair_out = pb.pair_index_of_low_MF(std::min(M_inter, M_final));
        const auto& dp_in  = pb.d_plus_pair[pair_in];
        const auto& dp_out = pb.d_plus_pair[pair_out];

        // Hoist halo→k matrix elements and Δ_in[k] outside the β loop.
        std::vector<dcompx> dP_k(N_inter);
        std::vector<double> Delta_in(N_inter);
        #pragma omp parallel for schedule(static)
        for (int k_loc = 0; k_loc < N_inter; ++k_loc) {
            dP_k[k_loc] = (P == +1)
                        ? dp_in(k_loc, halo_loc)
                        : std::conj(dp_in(halo_loc, k_loc));
            const int idx_k = off_inter + k_loc;
            Delta_in[k_loc] = pb.E_au_block_rel[idx_k] - E_h_rel - P * omega;
        }

        // β cutoff: I_PQ has BOTH terms exponentially suppressed by exp(-Δ_total² τ²/4)
        // when |Δ_total · τ| > 12.  (First term: Δ_in² + Δ_out² ≥ Δ_total²/2 by AM-QM,
        // so exp(-(D_in² + D_out²)τ²/2) ≤ exp(-Δ_total²τ²/4).  Second term: same factor.)
        // So β with huge Δ_total contribute < e^{-36} ≈ 2e-16, well below FP noise.
        constexpr double DELTA_TOT_TAU_CUT = 12.0;
        const double Delta_total_max = DELTA_TOT_TAU_CUT / tau;
        const double PQomega = (P + Q) * omega;

        // schedule(dynamic, 16): some β are inside the resonance window
        // (do full inner sum), most are outside (skipped); dynamic gives
        // the active β to free workers as they finish quick skips.
        #pragma omp parallel for schedule(dynamic, 16)
        for (int beta = 0; beta < N_final; ++beta) {
            const int    idx_b = off_final + beta;
            const double E_b   = pb.E_au_block_rel[idx_b];
            const double Delta_total = E_b - E_h_rel - PQomega;
            if (std::abs(Delta_total) > Delta_total_max) continue;

            dcompx accum(0.0, 0.0);
            for (int k_loc = 0; k_loc < N_inter; ++k_loc) {
                const double D_in  = Delta_in[k_loc];
                const double D_out = Delta_total - D_in;
                const dcompx dQ = (Q == +1)
                                ? dp_out(beta, k_loc)
                                : std::conj(dp_out(k_loc, beta));
                accum += dQ * dP_k[k_loc] * I_PQ_closed(D_in, D_out, tau, t_c);
            }
            // No race: across different (P,Q) pathways with the SAME final
            // block, the outer for-loop is serial, so this += is safe.
            R.b2(idx_b) += prefactor * accum;
        }
    }

    // ---------------------------------------------------------- //
    // Build prob_pt                                               //
    // ---------------------------------------------------------- //
    for (size_t k = 0; k < pb.block_MFs.size(); ++k) {
        const int dM  = pb.block_MFs[k] - M_halo;
        const int off = pb.block_offset[k];
        const int n   = pb.blocks[k].n_states();
        for (int i = 0; i < n; ++i) {
            const int idx = off + i;
            if (idx == halo_idx)
                R.prob_pt(idx) = std::norm(dcompx(1.0, 0.0) + R.b2(idx));
            else if (std::abs(dM) == 1)
                R.prob_pt(idx) = std::norm(R.b1(idx));
            else if (dM == 0 || std::abs(dM) == 2)
                R.prob_pt(idx) = std::norm(R.b2(idx));
            else
                R.prob_pt(idx) = 0.0;
        }
    }

    return R;
}

}  // namespace mc_tdse::pt
