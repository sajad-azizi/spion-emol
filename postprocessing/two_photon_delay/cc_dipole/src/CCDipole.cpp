#include "CCDipole.hpp"

#include "scatt/SolverParams.hpp"
#include "scatt/BackPropagator.hpp"
#include "scatt/DipoleMatrixElement.hpp"
#include "scatt/PotentialStorage.hpp"
#include "angular/Gaunt.hpp"           // for the real-Y_lm Gaunt-coefficient table

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace cc_dipole {

namespace {

// Thin abstract adapter so the algorithm doesn't care whether ψ comes
// from a live BackPropagator (post-scattering, in-memory) or a loaded
// PotentialStorage checkpoint (post-run on disk).  Both cases provide
// the same minimal interface: a getter for ψ at radial index ir, and
// the [n_keep_lo, n_keep_hi] window covered by the storage.
struct PsiAccess {
    std::function<const Eigen::MatrixXd&(std::size_t)> get_psi;
    int  n_keep_lo     = 0;
    int  n_keep_hi     = -1;
    bool psi_in_memory = false;
};

// Local copy of the (lm) ↔ flat-index mapping used elsewhere in the
// scattering code (real-Y_lm convention; index = l² + l + m).
inline void idx_to_lm(int idx, int& l, int& m) {
    l = static_cast<int>(std::sqrt(static_cast<double>(idx)));
    if ((l + 1) * (l + 1) <= idx) ++l;
    m = idx - l * l - l;
}

// Composite Simpson 1/3 + (3/8 last-interval) on n samples, step h.
// Accepts either even or odd # of subintervals.  Bit-identical to
// `scatt::DipoleMatrixElement::simpson_` for the same inputs.
double simpson_uniform(const double* f, int n_pts, double h) {
    if (n_pts < 2) return 0.0;
    auto simpson13 = [&](int n) {
        if (n < 2) return 0.0;
        if (n == 2) return 0.5 * h * (f[0] + f[1]);
        double s = f[0] + f[n - 1];
        for (int i = 1; i < n - 1; ++i)
            s += (i & 1 ? 4.0 : 2.0) * f[i];
        return s * h / 3.0;
    };
    if (n_pts % 2 == 1) return simpson13(n_pts);
    // Even subintervals → mix Simpson 1/3 over first n−4 + Simpson 3/8 over last 4.
    const double s13 = simpson13(n_pts - 3);
    const int    j   = n_pts - 4;
    const double s38 = (3.0 * h / 8.0) *
        (f[j] + 3.0 * f[j + 1] + 3.0 * f[j + 2] + f[j + 3]);
    return s13 + s38;
}

}  // namespace

// ---- Building blocks -----------------------------------------------------
// 1) build_state_impl: compute integration window, Ang table, Simpson weights.
// 2) accumulate_range_impl: do the per-ir GEMM accumulation over a sub-range.
// 3) compute_impl: glue, used by the original single-pair entrypoints.
// All ψ access goes through the PsiAccess adapter; both BackPropagator
// (live) and PotentialStorage (on-disk) bind to it.
namespace {

// Compute the integration window [n_lo, n_hi) from the two keep ranges.
void compute_window_(int n_grid,
                     int n_keep_lo_kappa, int n_keep_hi_kappa,
                     int n_keep_lo_nu,    int n_keep_hi_nu,
                     int n_overlap_hi,
                     int& n_lo, int& n_hi)
{
    n_lo = std::max({0, n_keep_lo_kappa, n_keep_lo_nu});
    n_hi = std::min({n_grid,
                     n_keep_hi_kappa + 1,
                     n_keep_hi_nu    + 1});
    if (n_overlap_hi > 0) n_hi = std::min(n_hi, n_overlap_hi);
    if (n_hi - n_lo < 5) {
        throw std::runtime_error(
            "cc_dipole: too few overlapping radial points (n_lo="
            + std::to_string(n_lo) + ", n_hi=" + std::to_string(n_hi) + ")");
    }
}

// Composite-Simpson weights for ir indices in [n_lo, n_hi); stored
// indexed by (ir - n_lo).  Mirror of simpson_uniform's per-sample
// weight pattern (Simpson 1/3 + last-4 Simpson 3/8 if even subintervals).
std::vector<double> simpson_weights_(int n_pts, double h)
{
    std::vector<double> w(static_cast<std::size_t>(n_pts), 0.0);
    if (n_pts % 2 == 1) {
        w[0]         = h / 3.0;
        w[n_pts - 1] = h / 3.0;
        for (int i = 1; i < n_pts - 1; ++i)
            w[i] = (i & 1 ? 4.0 : 2.0) * h / 3.0;
    } else {
        const int n13 = n_pts - 3;
        if (n13 >= 2) {
            w[0]         += h / 3.0;
            w[n13 - 1]   += h / 3.0;
            for (int i = 1; i < n13 - 1; ++i)
                w[i] += (i & 1 ? 4.0 : 2.0) * h / 3.0;
        }
        const double s38 = 3.0 * h / 8.0;
        const int j = n_pts - 4;
        w[j]     += s38;
        w[j + 1] += 3.0 * s38;
        w[j + 2] += 3.0 * s38;
        w[j + 3] += s38;
    }
    return w;
}

// Build the SPARSE angular table A^q_{μν} (real-Y_lm gauge), row-grouped.
// Same selection rules and coupling values as the original dense
// build_ang_table_; only the storage layout differs.  For every μ we
// record the (ν, A^q[μ, ν]) pairs in ascending-ν order so the sparse-
// dense product below has a deterministic, reproducible summation order.
//
// Density check: at l_cont=100 each row has at most 4 nonzero couplings
// (l_ν = l_μ±1 × m_ν ∈ {m_μ-1, m_μ, m_μ+1} subject to validity), so
// total nnz / N_psi ≈ 4 and total memory is ~1 MB instead of 833 MB for
// the dense N_psi² table.
std::vector<AngRow> build_ang_couplings_(int N_psi, int q)
{
    std::vector<int> l_idx(N_psi), m_idx(N_psi);
    for (int mu = 0; mu < N_psi; ++mu) idx_to_lm(mu, l_idx[mu], m_idx[mu]);
    std::vector<AngRow> rows;
    rows.reserve(static_cast<std::size_t>(N_psi));
    for (int mu = 0; mu < N_psi; ++mu) {
        const int lm = l_idx[mu], mm = m_idx[mu];
        AngRow row;
        row.mu = mu;
        // Scan ν in ASCENDING order so the inner sparse-dense sum
        // visits the same operand sequence on every call -- bit-stable.
        for (int nu = 0; nu < N_psi; ++nu) {
            const int ln = l_idx[nu], mn = m_idx[nu];
            if (std::abs(lm - ln) != 1) continue;
            const double a =
                scatt::DipoleMatrixElement::angular_dipole(lm, mm, q, ln, mn);
            if (a != 0.0) {
                row.nu.push_back(nu);
                row.a .push_back(a);
            }
        }
        if (!row.nu.empty()) rows.push_back(std::move(row));
    }
    return rows;
}

// Sparse first product:  tmp[μ, α] = Σ_{(ν, a) ∈ row(μ)}  a · ψ_ν[ν, α]
//
// Column-major access pattern (α outer, μ inner over the sparse rows)
// keeps the dense reads contiguous: psi_nu_ir.col(α) and tmp.col(α)
// are both column-contiguous in Eigen's default column-major layout.
//
// Summation order is fixed by the (μ ascending, k ascending within row)
// nesting; this is the bit-stable reference order used by the brute-
// force test.  Replacing the dense MKL/Eigen GEMM with this sparse path
// changes the summation order from MKL's tile sums to this ascending
// one -- physical difference per element is bounded by
// ε_mach × nnz_per_row × |tmp| ≈ 1e-15 × 4 × |tmp|, well below the
// existing 1e-12 abs / 1e-10 rel tolerance.
inline void sparse_ang_times_dense_(const std::vector<AngRow>& ang,
                                     const Eigen::MatrixXd&    psi_nu_ir,
                                     Eigen::MatrixXd&          tmp)
{
    const int N_psi = static_cast<int>(psi_nu_ir.cols());
    tmp.setZero();
    for (const AngRow& row : ang) {
        const int mu  = row.mu;
        const int nnz = static_cast<int>(row.nu.size());
        // Inner loop: (k ascending) over nonzero couplings of row μ.
        // For each output column α we accumulate `nnz` terms; nnz ≤ 4
        // for the dipole Gaunt selection rules, so the inner loop is
        // small enough for the compiler to fully unroll under -O3.
        for (int alpha = 0; alpha < N_psi; ++alpha) {
            double s = 0.0;
            for (int k = 0; k < nnz; ++k) {
                s += row.a[k] * psi_nu_ir(row.nu[k], alpha);
            }
            tmp(mu, alpha) = s;
        }
    }
}

// Accumulate w·r·ψ_κᵀ·Ang·ψ_ν into cc_raw for ir in [ir_lo, ir_hi).
// `w_global` is the Simpson-weight vector indexed by (ir - n_lo_global).
// Per-thread scratch (`tmp_scratch`, `M_scratch`) is preallocated once
// and reused across all ir's -- the original implementation freshly
// allocated a 833 MB MatrixXd per ir at l_cont=100, churning the
// allocator with ~5 PB of throwaway pages per κ.
//
// The first product (Ang · ψ_ν) goes through `sparse_ang_times_dense_`
// (sparse Ang × dense ψ_ν), saving ~50% of the per-ir compute relative
// to a dense MKL/Eigen first GEMM.  The second product (ψ_κᵀ · tmp)
// stays dense -- `tmp` is row-dense once Ang has been applied.
void accumulate_range_impl_(const PsiAccess&            src_kappa,
                            const PsiAccess&            src_nu,
                            const std::vector<AngRow>&  ang,
                            const std::vector<double>&  w_global,
                            int                         n_lo_global,
                            double                      r_min,
                            double                      dr,
                            int                         N_psi,
                            int                         ir_lo,
                            int                         ir_hi,
                            Eigen::MatrixXd&            cc_raw)
{
    const bool both_in_memory =
        src_kappa.psi_in_memory && src_nu.psi_in_memory;

    // Per-call accumulator for a single (ir).  Splits out the math so
    // it can be reused by both the parallel-ir and serial-ir branches.
    auto accumulate_one_ir = [&](int                     ir,
                                  Eigen::MatrixXd&        tmp_scratch,
                                  Eigen::MatrixXd&        M_scratch,
                                  Eigen::MatrixXd&        out_local) {
        const Eigen::MatrixXd& psi_kappa_ir =
            src_kappa.get_psi(static_cast<std::size_t>(ir));
        const Eigen::MatrixXd& psi_nu_ir =
            src_nu   .get_psi(static_cast<std::size_t>(ir));
        const double r   = r_min + ir * dr;
        const double w_r = w_global[ir - n_lo_global] * r;
        // First product: sparse Ang × dense ψ_ν.
        sparse_ang_times_dense_(ang, psi_nu_ir, tmp_scratch);
        // Second product: dense ψ_κᵀ × tmp (no allocation due to .noalias()).
        M_scratch.noalias() = psi_kappa_ir.transpose() * tmp_scratch;
        out_local.noalias() += w_r * M_scratch;
    };

    if (both_in_memory) {
#ifdef _OPENMP
        const int nth = omp_get_max_threads();
#else
        const int nth = 1;
#endif
        // Per-thread accumulator (final reduction) + per-thread scratch
        // (`tmp`, `M_ir`) so no thread allocates inside the parallel
        // for-loop.  Memory: nth · 3 · N_psi² · 8 B.  At l_cont=100 and
        // 50 threads this is ~50·3·833 MB = ~125 GB -- still fits the
        // MEMORY-mode budget that gated us into this branch.
        std::vector<Eigen::MatrixXd> partial(
            static_cast<size_t>(nth), Eigen::MatrixXd::Zero(N_psi, N_psi));
        std::vector<Eigen::MatrixXd> tmp_scratch(
            static_cast<size_t>(nth));
        std::vector<Eigen::MatrixXd> M_scratch(
            static_cast<size_t>(nth));
        for (int t = 0; t < nth; ++t) {
            tmp_scratch[t].resize(N_psi, N_psi);
            M_scratch  [t].resize(N_psi, N_psi);
        }
        #pragma omp parallel
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            #pragma omp for schedule(static)
            for (int ir = ir_lo; ir < ir_hi; ++ir) {
                accumulate_one_ir(ir,
                                   tmp_scratch[tid],
                                   M_scratch  [tid],
                                   partial    [tid]);
            }
        }
        for (auto& p : partial) cc_raw += p;
    } else {
        // Serial outer loop (DISK mode: PotentialStorage::get is NOT
        // thread-safe even when the requested chunk is resident, because
        // it returns a reference into a shared read_buffer_).  One pair
        // of scratch matrices is enough; allocate once outside the loop.
        Eigen::MatrixXd tmp_scratch(N_psi, N_psi);
        Eigen::MatrixXd M_scratch  (N_psi, N_psi);
        for (int ir = ir_lo; ir < ir_hi; ++ir) {
            accumulate_one_ir(ir, tmp_scratch, M_scratch, cc_raw);
        }
    }
}

CCResult compute_impl(const scatt::SolverParams&  sp,
                      const PsiAccess&            src_kappa,
                      const PsiAccess&            src_nu,
                      scatt::DipoleGauge          gauge,
                      scatt::Polarization         pol,
                      int                         n_overlap_hi)
{
    if (gauge != scatt::DipoleGauge::Length) {
        throw std::runtime_error(
            "compute_cc_dipole: only length gauge implemented in this pass");
    }

    const int N_psi = sp.n_mu;
    const int Nr    = static_cast<int>(sp.n_grid);
    const int q     = scatt::q_of(pol);

    int n_lo = 0, n_hi = 0;
    compute_window_(Nr,
                    src_kappa.n_keep_lo, src_kappa.n_keep_hi,
                    src_nu   .n_keep_lo, src_nu   .n_keep_hi,
                    n_overlap_hi, n_lo, n_hi);
    const int n_pts = n_hi - n_lo;
    std::vector<AngRow> ang = build_ang_couplings_(N_psi, q);
    std::vector<double> w   = simpson_weights_(n_pts, sp.dr);
    Eigen::MatrixXd cc_raw = Eigen::MatrixXd::Zero(N_psi, N_psi);
    accumulate_range_impl_(src_kappa, src_nu, ang, w, n_lo,
                           sp.r_min, sp.dr, N_psi,
                           n_lo, n_hi, cc_raw);

    CCResult out;
    out.cc_raw  = std::move(cc_raw);
    out.gauge   = gauge;
    out.pol     = pol;
    out.E_kappa = sp.energy;
    out.E_nu    = sp.energy;
    out.ik_kappa = -1;
    out.ik_nu    = -1;
    return out;
}
}  // anonymous namespace

// ---- Public overload: BackPropagator (live, in-memory) ----
CCResult compute_cc_dipole(const scatt::SolverParams&  sp,
                            scatt::BackPropagator&      bp_kappa,
                            scatt::BackPropagator&      bp_nu,
                            scatt::DipoleGauge          gauge,
                            scatt::Polarization         pol,
                            int                         n_overlap_hi)
{
    PsiAccess sk;
    sk.get_psi = [&bp_kappa](std::size_t ir) -> const Eigen::MatrixXd& {
        return bp_kappa.get_psi(ir);
    };
    sk.n_keep_lo     = bp_kappa.n_keep_lo();
    sk.n_keep_hi     = bp_kappa.n_keep_hi();
    sk.psi_in_memory = bp_kappa.psi_in_memory();

    PsiAccess sn;
    sn.get_psi = [&bp_nu](std::size_t ir) -> const Eigen::MatrixXd& {
        return bp_nu.get_psi(ir);
    };
    sn.n_keep_lo     = bp_nu.n_keep_lo();
    sn.n_keep_hi     = bp_nu.n_keep_hi();
    sn.psi_in_memory = bp_nu.psi_in_memory();

    return compute_impl(sp, sk, sn, gauge, pol, n_overlap_hi);
}

// ---- Public overload: PotentialStorage (loaded from on-disk checkpoint) ----
CCResult compute_cc_dipole(const scatt::SolverParams&  sp,
                            scatt::PotentialStorage&    psi_kappa,
                            scatt::PotentialStorage&    psi_nu,
                            int                         n_keep_lo,
                            int                         n_keep_hi,
                            scatt::DipoleGauge          gauge,
                            scatt::Polarization         pol,
                            int                         n_overlap_hi)
{
    using scatt::PotentialStorage;
    PsiAccess sk;
    // PotentialStorage stores ψ at logical ir's 0..N-1 INSIDE the kept
    // window; the caller's ir corresponds to global radial index, but
    // BackPropagator + scattering pipelines write the storage with the
    // SAME global indexing, so .get(ir) is correct here.
    sk.get_psi = [&psi_kappa](std::size_t ir) -> const Eigen::MatrixXd& {
        return psi_kappa.get(ir);
    };
    sk.n_keep_lo     = n_keep_lo;
    sk.n_keep_hi     = n_keep_hi;
    sk.psi_in_memory = (psi_kappa.mode() == PotentialStorage::Mode::MEMORY);

    PsiAccess sn;
    sn.get_psi = [&psi_nu](std::size_t ir) -> const Eigen::MatrixXd& {
        return psi_nu.get(ir);
    };
    sn.n_keep_lo     = n_keep_lo;
    sn.n_keep_hi     = n_keep_hi;
    sn.psi_in_memory = (psi_nu.mode() == PotentialStorage::Mode::MEMORY);

    return compute_impl(sp, sk, sn, gauge, pol, n_overlap_hi);
}

// ---- Low-level state + range API (used by chunk-blocked driver) ----------
CCAccumState
make_accum_state(const scatt::SolverParams& sp,
                  int n_keep_lo_kappa, int n_keep_hi_kappa,
                  int n_keep_lo_nu,    int n_keep_hi_nu,
                  scatt::DipoleGauge   gauge,
                  scatt::Polarization  pol,
                  int n_overlap_hi)
{
    if (gauge != scatt::DipoleGauge::Length) {
        throw std::runtime_error(
            "cc_dipole::make_accum_state: only length gauge implemented");
    }
    CCAccumState st;
    st.N_psi = sp.n_mu;
    st.dr    = sp.dr;
    st.r_min = sp.r_min;
    st.gauge = gauge;
    st.pol   = pol;
    compute_window_(static_cast<int>(sp.n_grid),
                    n_keep_lo_kappa, n_keep_hi_kappa,
                    n_keep_lo_nu,    n_keep_hi_nu,
                    n_overlap_hi, st.n_lo, st.n_hi);
    st.ang = build_ang_couplings_(st.N_psi, scatt::q_of(pol));
    st.w   = simpson_weights_(st.n_hi - st.n_lo, st.dr);
    return st;
}

void accumulate_cc_range(const CCAccumState&       st,
                         scatt::PotentialStorage&  psi_kappa,
                         scatt::PotentialStorage&  psi_nu,
                         int                       ir_lo,
                         int                       ir_hi,
                         Eigen::MatrixXd&          cc_raw)
{
    // Clamp to the global integration window: anything outside is a no-op.
    ir_lo = std::max(ir_lo, st.n_lo);
    ir_hi = std::min(ir_hi, st.n_hi);
    if (ir_hi <= ir_lo) return;

    if (cc_raw.rows() != st.N_psi || cc_raw.cols() != st.N_psi) {
        throw std::runtime_error(
            "cc_dipole::accumulate_cc_range: cc_raw shape "
            + std::to_string(cc_raw.rows()) + "×"
            + std::to_string(cc_raw.cols()) + " ≠ N_psi×N_psi");
    }

    PsiAccess sk;
    sk.get_psi = [&psi_kappa](std::size_t ir) -> const Eigen::MatrixXd& {
        return psi_kappa.get(ir);
    };
    sk.n_keep_lo     = 0;        // unused inside accumulate_range_impl_
    sk.n_keep_hi     = -1;
    sk.psi_in_memory =
        (psi_kappa.mode() == scatt::PotentialStorage::Mode::MEMORY);

    PsiAccess sn;
    sn.get_psi = [&psi_nu](std::size_t ir) -> const Eigen::MatrixXd& {
        return psi_nu.get(ir);
    };
    sn.n_keep_lo     = 0;
    sn.n_keep_hi     = -1;
    sn.psi_in_memory =
        (psi_nu.mode() == scatt::PotentialStorage::Mode::MEMORY);

    accumulate_range_impl_(sk, sn, st.ang, st.w, st.n_lo,
                           st.r_min, st.dr, st.N_psi,
                           ir_lo, ir_hi, cc_raw);
}

}  // namespace cc_dipole
