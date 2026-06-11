#include "PooledTDSE.hpp"

#include <stdexcept>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace mc_tdse {

Eigen::MatrixXcd build_HI(const PooledBasis& pb, double t,
                          const PooledTDSEConfig& cfg) {
    const int N = pb.N_total;
    Eigen::MatrixXcd HI = Eigen::MatrixXcd::Zero(N, N);

    const double chi_t = cfg.chi(t);
    if (chi_t == 0.0) return HI;
    const double half_OmR_chi = 0.5 * cfg.Omega_R_au * chi_t;

    // For each adjacent pair (k, k+1) with M_F[k+1] == M_F[k] + 1, fill
    //
    //   HI(α, β) += half_OmR_chi · e^{i(E_α - E_β)t} · e^{-iωt} · d^(+)(αβ)
    //   HI(β, α) += half_OmR_chi · e^{i(E_β - E_α)t} · e^{+iωt} · (d^(+))*(αβ)
    //
    // where α ∈ block k+1, β ∈ block k.  The second line is the η^(-)
    // (emission) coupling and is the conjugate-transpose of the first
    // (so HI is Hermitian, as required for unitary evolution).
    //
    // Energy phases include the carrier frequency: ω carries M_F+1 → M_F+1
    // through e^{-iωt} on the absorption term.
    // ROTATING-FRAME ENERGIES.  Recipe convention: ω is the SMALL
    // detuning from the Zeeman-resonant lab-frame photon (which
    // bridges the ~MHz threshold gap between adjacent M_F blocks).
    // Going to the rotating frame absorbs the Zeeman ladder energy
    // into the U_carrier transform, leaving each block's eigenenergies
    // referenced to its OWN threshold.  In this frame, the resonance
    // halo + ω → continuum sits at  E_f^block - E_i^block = ω, so
    // 1γ final-state kinetic energy ≈ ω above the block's threshold.
    // That is what the recipe's E_cut^open ≈ 20-30 E_b above threshold
    // is sized to cover.
    //
    // Concretely: the phase factor uses block-RELATIVE energies
    // (E_α - E_th(block of α)).  Lab-frame energies pb.E_au[*] are
    // kept around for I/O and final-state readout; only the phase
    // formula uses the block-relative copies.
    for (size_t k = 0; k + 1 < pb.blocks.size(); ++k) {
        if (pb.block_MFs[k + 1] != pb.block_MFs[k] + 1) continue;
        const auto& d_plus = pb.d_plus_pair[k];
        const int off_low  = pb.block_offset[k];
        const int off_high = pb.block_offset[k + 1];
        const int Nb       = pb.blocks[k].n_states();
        const int Na       = pb.blocks[k + 1].n_states();
        for (int alpha = 0; alpha < Na; ++alpha) {
            const int A = off_high + alpha;
            for (int beta = 0; beta < Nb; ++beta) {
                const int B = off_low + beta;
                const double dE_ab = pb.E_au_block_rel[A]
                                   - pb.E_au_block_rel[B];
                const dcompx phase_abs = std::exp(I_unit * (dE_ab - cfg.omega_au) * t);
                const dcompx phase_emi = std::exp(I_unit * (-dE_ab + cfg.omega_au) * t);
                const dcompx coup_AB = half_OmR_chi * phase_abs * d_plus(alpha, beta);
                HI(A, B) += coup_AB;
                HI(B, A) += half_OmR_chi * phase_emi * std::conj(d_plus(alpha, beta));
            }
        }
    }
    return HI;
}

// ---- HIApplier: avoid materializing dense H_I per step ---------
//
// Phase factorization:
//   H_I,αβ(t) = (Ω_R/2)·χ(t) · e^{i(E_α^block - E_β^block)t}
//              · [ e^{-iωt} d⁺_{αβ}        (absorption: α∈ block_high, β∈ block_low)
//                + e^{+iωt} (d⁺)*_{βα}     (emission: α∈ block_low, β∈ block_high) ]
//
// Define φ(α, t) = e^{i E_α^block t}  for each pooled index α.  Then
//   phase_abs(α, β, t) = e^{-iωt} · φ(α) · conj(φ(β))
// is a product of three pre-computable scalars (no per-pair exp).
//
// For matvec out = H_I · b:
//   out_high += (Ω_R/2)·χ·e^{-iωt} · φ_high ⊙ [ d⁺_block · (conj(φ_low) ⊙ b_low) ]
//   out_low  += (Ω_R/2)·χ·e^{+iωt} · φ_low  ⊙ [ d⁺_block^H · (conj(φ_high) ⊙ b_high) ]
//
// Per matvec: two (Na × Nb) GEMVs per adjacent pair plus O(N) elementwise
// products.  Per Taylor step (K matvecs): N_total exps amortized.
// Saves >100× over the dense build at production size.

HIApplier::HIApplier(const PooledBasis& pb,
                     const PooledTDSEConfig& cfg,
                     double t)
    : pb_(pb), cfg_(cfg), t_(t)
{
    const double chi_t = cfg.chi(t);
    half_OmR_chi_ = 0.5 * cfg.Omega_R_au * chi_t;
    if (half_OmR_chi_ == 0.0) {
        // Zero pulse: apply() returns zero; phase tables not needed.
        return;
    }
    psi_abs_ = half_OmR_chi_ * std::exp(-I_unit * cfg.omega_au * t);
    psi_emi_ = half_OmR_chi_ * std::exp(+I_unit * cfg.omega_au * t);
    const int N = pb.N_total;
    phi_.resize(N);
    phi_conj_.resize(N);
    for (int a = 0; a < N; ++a) {
        const dcompx p = std::exp(I_unit * pb.E_au_block_rel[a] * t);
        phi_(a) = p;
        phi_conj_(a) = std::conj(p);
    }
}

Eigen::VectorXcd HIApplier::apply(const Eigen::VectorXcd& b) const {
    Eigen::VectorXcd out = Eigen::VectorXcd::Zero(b.size());
    if (half_OmR_chi_ == 0.0) return out;

    // Parallel matvec via raw pointer arithmetic.  Eigen's VectorXcd *
    // MatrixXcd is SERIAL on a header-only Eigen build (no MKL/BLAS);
    // since each step does ~10⁶ complex multiplies in the (-5, -4)
    // pair alone, the propagation hits a ~3 hr wall on 1 thread.  Below
    // we explicitly split the output dimension across OMP threads.
    for (size_t k = 0; k + 1 < pb_.blocks.size(); ++k) {
        if (pb_.block_MFs[k + 1] != pb_.block_MFs[k] + 1) continue;
        const auto& d_plus = pb_.d_plus_pair[k];      // (Na, Nb), col-major
        const int off_low  = pb_.block_offset[k];
        const int off_high = pb_.block_offset[k + 1];
        const int Nb       = pb_.blocks[k].n_states();
        const int Na       = pb_.blocks[k + 1].n_states();

        // Build phase-de-rotated slices (cheap O(N) elementwise).
        Eigen::VectorXcd b_low_phased(Nb), b_high_phased(Na);
        const dcompx* phi_conj = phi_conj_.data();
        const dcompx* b_ptr    = b.data();
        for (int i = 0; i < Nb; ++i)
            b_low_phased(i)  = phi_conj[off_low + i] * b_ptr[off_low + i];
        for (int i = 0; i < Na; ++i)
            b_high_phased(i) = phi_conj[off_high + i] * b_ptr[off_high + i];

        const dcompx* dp = d_plus.data();   // col-major: dp[i + j*Na] = d_plus(i, j)
        const dcompx* xL = b_low_phased.data();
        const dcompx* xH = b_high_phased.data();

        // ---- Absorption: c_abs[i] = Σ_j d_plus(i, j) · b_low_phased[j] ----
        // Output dim Na.  When Na is small (e.g. 170) and Nb large (9761),
        // per-row dot products of length Nb give plenty of work even with
        // 100+ threads.  Each thread takes a chunk of i.
        Eigen::VectorXcd c_abs(Na);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Na; ++i) {
            dcompx s(0.0, 0.0);
            for (int j = 0; j < Nb; ++j)
                s += dp[i + j * Na] * xL[j];
            c_abs(i) = s;
        }
        const dcompx* phi_data = phi_.data();
        for (int i = 0; i < Na; ++i)
            out(off_high + i) += psi_abs_ * phi_data[off_high + i] * c_abs(i);

        // ---- Emission: c_emi[j] = Σ_i conj(d_plus(i, j)) · b_high_phased[i] ----
        // Output dim Nb.  Easy parallelism: split j across threads.
        Eigen::VectorXcd c_emi(Nb);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < Nb; ++j) {
            dcompx s(0.0, 0.0);
            for (int i = 0; i < Na; ++i)
                s += std::conj(dp[i + j * Na]) * xH[i];
            c_emi(j) = s;
        }
        for (int j = 0; j < Nb; ++j)
            out(off_low + j) += psi_emi_ * phi_data[off_low + j] * c_emi(j);
    }
    return out;
}

// ---- Taylor step using a callable matvec, no dense H ------------
// Equivalent to taylor_step_const_H but takes a HIApplier (or any
// callable returning H · y) instead of a dense matrix.  Saves the
// O(N²) matrix allocation per step.
namespace {
template <typename Apply>
Eigen::VectorXcd
taylor_step_apply(Apply apply,
                  const Eigen::VectorXcd& b,
                  double dt,
                  const TaylorOptions& opt,
                  TaylorStats* stats)
{
    Eigen::VectorXcd out = b;
    Eigen::VectorXcd y   = b;
    const double y0_norm = b.norm();
    const double tol     = opt.eps_rel * std::max(y0_norm, 1.0e-300);
    int  k_used = 0;
    double last_res = 0.0;
    for (int k = 1; k <= opt.K_max; ++k) {
        const dcompx coef = -I_unit * dt / static_cast<double>(k);
        y = coef * apply(y);
        out += y;
        const double yn = y.norm();
        last_res = (y0_norm > 0) ? yn / y0_norm : yn;
        k_used = k;
        if (yn < tol) break;
    }
    if (stats) { stats->last_K = k_used; stats->last_res = last_res; }
    return out;
}
}  // namespace

Eigen::VectorXcd propagate_pooled(const PooledBasis& pb,
                                  Eigen::VectorXcd c,
                                  const PooledTDSEConfig& cfg,
                                  PooledTDSEStats* stats) {
    if (c.size() != pb.N_total)
        throw std::runtime_error("propagate_pooled: c size != pb.N_total");
    if (cfg.dt <= 0.0)
        throw std::runtime_error("propagate_pooled: dt must be > 0");

    const double T = cfg.t_end - cfg.t_start;
    if (T <= 0.0) return c;
    const int n_steps = static_cast<int>(std::ceil(T / cfg.dt));
    long long sum_K = 0;
    double max_err = 0.0;

    TaylorStats st;
    for (int step = 0; step < n_steps; ++step) {
        const double t_mid = cfg.t_start + (step + 0.5) * cfg.dt;
        // Precompute phase tables for the midpoint t once; reuse for all
        // K Taylor matvecs of this step.
        HIApplier ha(pb, cfg, t_mid);
        c = taylor_step_apply(
                [&](const Eigen::VectorXcd& v) { return ha.apply(v); },
                c, cfg.dt, cfg.taylor, &st);
        sum_K  += st.last_K;
        if (st.last_res > max_err) max_err = st.last_res;

        if (n_steps >= 10 && (step + 1) % (n_steps / 10) == 0) {
            std::printf("[propagate_pooled] step %d / %d   K=%d  max_taylor_res=%.2e\n",
                        step + 1, n_steps, st.last_K, st.last_res);
        }
    }
    if (stats) {
        stats->n_steps = n_steps;
        stats->K_avg   = (n_steps > 0)
                       ? static_cast<int>(sum_K / n_steps)
                       : 0;
        stats->max_err = max_err;
    }
    return c;
}

std::vector<double> block_populations(const PooledBasis& pb,
                                      const Eigen::VectorXcd& c) {
    std::vector<double> P(pb.blocks.size(), 0.0);
    for (size_t k = 0; k < pb.blocks.size(); ++k) {
        const int o = pb.block_offset[k];
        const int n = pb.blocks[k].n_states();
        for (int i = 0; i < n; ++i)
            P[k] += std::norm(c(o + i));
    }
    return P;
}

}  // namespace mc_tdse
