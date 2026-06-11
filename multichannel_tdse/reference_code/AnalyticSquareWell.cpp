// AnalyticSquareWell.cpp
//
// Exact analytic bound-state and scattering-state solver for the 
// multichannel square-well potential. Handles the halo bound state 
// (all channels closed) and single-open-channel scattering states.
//
#include "AnalyticSquareWell.hpp"

namespace {
template <class F>
double integrate_uniform_grid_local(int N_grid, double dr, F&& f)
{
    const int intervals = N_grid - 1;
    if (N_grid < 2) return 0.0;
    if (intervals == 1) {
        return 0.5 * dr * (f(0) + f(1));
    }

    if (intervals % 2 == 0) {
        double s = f(0) + f(N_grid - 1);
        for (int ir = 1; ir < N_grid - 1; ++ir) {
            s += (ir % 2 == 1 ? 4.0 : 2.0) * f(ir);
        }
        return s * dr / 3.0;
    }

    if (intervals == 3) {
        return 3.0 * dr / 8.0
             * (f(0) + 3.0 * f(1) + 3.0 * f(2) + f(3));
    }

    const int last13 = N_grid - 4;
    double s = f(0) + f(last13);
    for (int ir = 1; ir < last13; ++ir) {
        s += (ir % 2 == 1 ? 4.0 : 2.0) * f(ir);
    }
    double total13 = s * dr / 3.0;
    double total38 = 3.0 * dr / 8.0
        * (f(N_grid - 4) + 3.0 * f(N_grid - 3)
         + 3.0 * f(N_grid - 2) + f(N_grid - 1));
    return total13 + total38;
}

double sin_sin_integral(double a, double b, double R)
{
    if (std::abs(a - b) < 1e-12 * std::max({1.0, std::abs(a), std::abs(b)})) {
        if (std::abs(a) < 1e-14) return 0.0;
        return 0.5 * R - std::sin(2.0 * a * R) / (4.0 * a);
    }
    return std::sin((a - b) * R) / (2.0 * (a - b))
         - std::sin((a + b) * R) / (2.0 * (a + b));
}

double sinh_sinh_integral(double a, double b, double R)
{
    if (std::abs(a - b) < 1e-12 * std::max({1.0, std::abs(a), std::abs(b)})) {
        if (std::abs(a) < 1e-14) return 0.0;
        return std::sinh(2.0 * a * R) / (4.0 * a) - 0.5 * R;
    }
    return std::sinh((a + b) * R) / (2.0 * (a + b))
         - std::sinh((a - b) * R) / (2.0 * (a - b));
}

double sin_sinh_integral(double a, double b, double R)
{
    const double den = a * a + b * b;
    if (den == 0.0) return 0.0;
    return (b * std::sin(a * R) * std::cosh(b * R)
          - a * std::cos(a * R) * std::sinh(b * R)) / den;
}

double inside_basis_integral(double qa, bool a_sin,
                             double qb, bool b_sin,
                             double R)
{
    if (a_sin && b_sin) return sin_sin_integral(qa, qb, R);
    if (!a_sin && !b_sin) return sinh_sinh_integral(qa, qb, R);
    if (a_sin && !b_sin) return sin_sinh_integral(qa, qb, R);
    return sin_sinh_integral(qb, qa, R);
}

double bound_state_norm2_exact(const AnalyticState& state)
{
    double norm2 = 0.0;
    const double R = state.r0;

    for (int a = 0; a < state.N_ch; ++a) {
        for (int k = 0; k < state.N_ch; ++k) {
            const double ck = state.U_in(a, k) * state.A_in(k);
            if (std::abs(ck) < 1e-30) continue;
            for (int l = 0; l < state.N_ch; ++l) {
                const double cl = state.U_in(a, l) * state.A_in(l);
                if (std::abs(cl) < 1e-30) continue;
                norm2 += ck * cl
                       * inside_basis_integral(state.q_in(k), state.is_sin[k],
                                               state.q_in(l), state.is_sin[l],
                                               R);
            }
        }

        const double kappa = state.k_out(a);
        if (kappa <= 0.0) {
            throw std::runtime_error(
                "bound_state_norm2_exact: bound state has a non-decaying channel");
        }
        norm2 += state.C_out(a) * state.C_out(a)
               * std::exp(-2.0 * kappa * R) / (2.0 * kappa);
    }

    return norm2;
}
}

// ============================================================
// AnalyticState::evaluate
// ============================================================
Eigen::VectorXd AnalyticState::evaluate(double r) const
{
    Eigen::VectorXd u = Eigen::VectorXd::Zero(N_ch);

    if (r <= r0) {
        // Inside: u_alpha(r) = Σ_k U_{α,k} A_k f_k(q_k r)
        for (int k = 0; k < N_ch; ++k) {
            double fk;
            if (is_sin[k]) fk = std::sin(q_in(k) * r);
            else           fk = std::sinh(q_in(k) * r);
            for (int a = 0; a < N_ch; ++a) {
                u(a) += U_in(a, k) * A_in(k) * fk;
            }
        }
    } else {
        // Outside: per-channel expression
        for (int a = 0; a < N_ch; ++a) {
            if (is_open[a]) {
                double k_r = k_out(a) * r;
                u(a) = P_out(a) * std::sin(k_r) + Q_out(a) * std::cos(k_r);
            } else {
                u(a) = C_out(a) * std::exp(-k_out(a) * r);
            }
        }
    }
    return u;
}

Eigen::Vector2d AnalyticState::evaluate2(double r) const
{
    if (N_ch != 2) {
        throw std::runtime_error("AnalyticState::evaluate2 requires N_ch == 2");
    }

    Eigen::Vector2d u = Eigen::Vector2d::Zero();
    if (r <= r0) {
        for (int k = 0; k < 2; ++k) {
            double fk;
            if (is_sin[k]) fk = std::sin(q_in(k) * r);
            else           fk = std::sinh(q_in(k) * r);
            u(0) += U_in(0, k) * A_in(k) * fk;
            u(1) += U_in(1, k) * A_in(k) * fk;
        }
    } else {
        for (int a = 0; a < 2; ++a) {
            if (is_open[a]) {
                double k_r = k_out(a) * r;
                u(a) = P_out(a) * std::sin(k_r) + Q_out(a) * std::cos(k_r);
            } else {
                u(a) = C_out(a) * std::exp(-k_out(a) * r);
            }
        }
    }
    return u;
}

// ============================================================
// Constructor
// ============================================================
AnalyticSquareWell::AnalyticSquareWell(Potentials* potentials, Parameters* parameters)
    : pot_(potentials), par_(parameters)
{
    N_ch_  = pot_->num_channels();
    mu_    = par_->mu;
    r0_    = pot_->r0();
    V_bar_ = pot_->V_bar();
    dV_    = pot_->dV();
    s1s2_  = pot_->s1s2();
    thresholds_ = pot_->thresholds();
}

// ============================================================
// Inside-well eigendecomposition
// ============================================================
void AnalyticSquareWell::inside_decomposition(double E,
                                                Eigen::MatrixXd& U_in,
                                                Eigen::VectorXd& q_vals,
                                                std::vector<bool>& is_sin) const
{
    // V_short = -Vbar I + dV S  (from singlet/triplet projector expansion;
    // see Potentials.cpp for derivation)
    Eigen::MatrixXd V_in_mat = -V_bar_ * Eigen::MatrixXd::Identity(N_ch_, N_ch_)
                              + dV_ * s1s2_;
    for (int a = 0; a < N_ch_; ++a) V_in_mat(a, a) += thresholds_(a);

    Eigen::MatrixXd K_in = 2.0 * mu_
        * (E * Eigen::MatrixXd::Identity(N_ch_, N_ch_) - V_in_mat);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K_in);
    Eigen::VectorXd evals = es.eigenvalues();
    U_in = es.eigenvectors();

    q_vals.resize(N_ch_);
    is_sin.assign(N_ch_, false);
    for (int k = 0; k < N_ch_; ++k) {
        if (evals(k) >= 0.0) {
            q_vals(k) = std::sqrt(evals(k));
            is_sin[k] = true;
        } else {
            q_vals(k) = std::sqrt(-evals(k));
            is_sin[k] = false;
        }
    }
}

// ============================================================
// Outside wave-vectors
// ============================================================
void AnalyticSquareWell::outside_wavevectors(double E,
                                               Eigen::VectorXd& k_out,
                                               std::vector<bool>& is_open) const
{
    k_out.resize(N_ch_);
    is_open.assign(N_ch_, false);
    for (int a = 0; a < N_ch_; ++a) {
        double d = E - thresholds_(a);
        if (d >= 0.0) {
            k_out(a) = std::sqrt(2.0 * mu_ * d);
            is_open[a] = true;
        } else {
            k_out(a) = std::sqrt(-2.0 * mu_ * d);
            is_open[a] = false;
        }
    }
}

// ============================================================
// Bound-state matching matrix (all channels closed)
// ============================================================
void AnalyticSquareWell::build_bound_matching_matrix(double E,
                                                      Eigen::MatrixXd& M,
                                                      Eigen::MatrixXd& U_in,
                                                      Eigen::VectorXd& q_vals,
                                                      std::vector<bool>& is_sin,
                                                      Eigen::VectorXd& kappa_out) const
{
    inside_decomposition(E, U_in, q_vals, is_sin);

    std::vector<bool> is_open_ignored;
    outside_wavevectors(E, kappa_out, is_open_ignored);

    Eigen::VectorXd f_r0  = Eigen::VectorXd::Zero(N_ch_);
    Eigen::VectorXd df_r0 = Eigen::VectorXd::Zero(N_ch_);
    for (int k = 0; k < N_ch_; ++k) {
        if (is_sin[k]) {
            f_r0(k)  = std::sin(q_vals(k) * r0_);
            df_r0(k) = q_vals(k) * std::cos(q_vals(k) * r0_);
        } else {
            f_r0(k)  = std::sinh(q_vals(k) * r0_);
            df_r0(k) = q_vals(k) * std::cosh(q_vals(k) * r0_);
        }
    }

    Eigen::VectorXd exp_r0(N_ch_);
    for (int a = 0; a < N_ch_; ++a) {
        exp_r0(a) = std::exp(-kappa_out(a) * r0_);
    }

    M = Eigen::MatrixXd::Zero(2 * N_ch_, 2 * N_ch_);
    for (int a = 0; a < N_ch_; ++a)
        for (int k = 0; k < N_ch_; ++k)
            M(a, k) = U_in(a, k) * f_r0(k);
    for (int a = 0; a < N_ch_; ++a)
        M(a, N_ch_ + a) = -exp_r0(a);
    for (int a = 0; a < N_ch_; ++a)
        for (int k = 0; k < N_ch_; ++k)
            M(N_ch_ + a, k) = U_in(a, k) * df_r0(k);
    for (int a = 0; a < N_ch_; ++a)
        M(N_ch_ + a, N_ch_ + a) = kappa_out(a) * exp_r0(a);
}

double AnalyticSquareWell::matching_determinant(double E)
{
    Eigen::MatrixXd M, U_in;
    Eigen::VectorXd q_vals, kappa_out;
    std::vector<bool> is_sin;
    build_bound_matching_matrix(E, M, U_in, q_vals, is_sin, kappa_out);
    return M.determinant();
}

double AnalyticSquareWell::find_highest_bound_state(double E_lo, double E_hi, int n_scan)
{
    double dE = (E_hi - E_lo) / (n_scan - 1);
    double prev_det = matching_determinant(E_lo);
    double highest_root = 0.0;
    bool found = false;

    for (int i = 1; i < n_scan; ++i) {
        double E_try = E_lo + i * dE;
        double det = matching_determinant(E_try);
        if (prev_det * det < 0.0) {
            double a = E_lo + (i - 1) * dE;
            double b = E_try;
            double fa = prev_det;
            double fb = 0.0;   // assigned in loop body, only read after assignment
            (void)fb;
            for (int iter = 0; iter < 200; ++iter) {
                double m = 0.5 * (a + b);
                double fm = matching_determinant(m);
                if (fa * fm < 0.0) { b = m; fb = fm; }
                else               { a = m; fa = fm; }
                if ((b - a) < 1e-25) break;
            }
            highest_root = 0.5 * (a + b);
            found = true;
        }
        prev_det = det;
    }

    if (!found) {
        throw std::runtime_error("AnalyticSquareWell: no bound state found in range");
    }
    return highest_root;
}

// Find ALL bound states in [E_lo, E_hi] by scanning det M(E) for sign changes.
// For a well where all channels are closed over this range, each sign change
// corresponds to a bound-state eigenvalue; bisect to converge.
std::vector<double>
AnalyticSquareWell::find_all_bound_states(double E_lo, double E_hi, int n_scan)
{
    std::vector<double> roots;
    if (n_scan < 10) n_scan = 10;
    double dE = (E_hi - E_lo) / (n_scan - 1);
    double prev_det = matching_determinant(E_lo);

    for (int i = 1; i < n_scan; ++i) {
        double E_try = E_lo + i * dE;
        double det = matching_determinant(E_try);
        if (prev_det * det < 0.0) {
            // bisect within [E_prev, E_try]
            double a = E_lo + (i - 1) * dE;
            double b = E_try;
            double fa = prev_det;
            for (int iter = 0; iter < 200; ++iter) {
                double m = 0.5 * (a + b);
                double fm = matching_determinant(m);
                if (fa * fm < 0.0) { b = m; }
                else               { a = m; fa = fm; }
                if ((b - a) < 1e-25) break;
            }
            roots.push_back(0.5 * (a + b));
        }
        prev_det = det;
    }
    return roots;
}

// ============================================================
// Bound-state wave function reconstruction
// ============================================================
AnalyticState AnalyticSquareWell::build_bound_state(double E)
{
    Eigen::MatrixXd M, U_in;
    Eigen::VectorXd q_vals, kappa_out;
    std::vector<bool> is_sin;
    build_bound_matching_matrix(E, M, U_in, q_vals, is_sin, kappa_out);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullV);
    Eigen::VectorXd null_vec = svd.matrixV().col(2 * N_ch_ - 1);

    Eigen::VectorXd A = null_vec.head(N_ch_);
    Eigen::VectorXd C = null_vec.tail(N_ch_);

    AnalyticState state;
    state.E       = E;
    state.N_ch    = N_ch_;
    state.r0      = r0_;
    state.U_in    = U_in;
    state.q_in    = q_vals;
    state.is_sin  = is_sin;
    state.A_in    = A;
    state.is_open.assign(N_ch_, false);
    state.k_out   = kappa_out;
    state.P_out   = Eigen::VectorXd::Zero(N_ch_);
    state.Q_out   = Eigen::VectorXd::Zero(N_ch_);
    state.C_out   = C;

    double norm2 = bound_state_norm2_exact(state);
    double norm = std::sqrt(norm2);
    if (norm > 0.0) {
        state.A_in /= norm;
        state.C_out /= norm;
    }
    return state;
}

void AnalyticSquareWell::build_bound_wavefunction(double E,
                                                   std::vector<Eigen::VectorXd>& u_out)
{
    AnalyticState state = build_bound_state(E);

    const int N_grid = par_->N_grid;
    const double dr  = par_->dr;
    u_out.assign(N_grid, Eigen::VectorXd::Zero(N_ch_));
    for (int ir = 0; ir < N_grid; ++ir) {
        double r = ir * dr;
        u_out[ir] = state.evaluate(r);
    }
}

// ============================================================
// Scattering state construction (single open channel)
// ============================================================
AnalyticState AnalyticSquareWell::solve_scattering(double E, int incoming_channel) const
{
    AnalyticState state;
    state.E    = E;
    state.N_ch = N_ch_;
    state.r0   = r0_;

    inside_decomposition(E, state.U_in, state.q_in, state.is_sin);
    outside_wavevectors(E, state.k_out, state.is_open);

    int N_open = 0;
    std::vector<int> open_idx;
    std::vector<int> closed_idx;
    for (int a = 0; a < N_ch_; ++a) {
        if (state.is_open[a]) { open_idx.push_back(a); ++N_open; }
        else                   closed_idx.push_back(a);
    }
    const int N_closed = N_ch_ - N_open;

    if (N_open == 0) {
        throw std::runtime_error(
            "solve_scattering: no open channels at this energy");
    }
    if (N_open > 1) {
        throw std::runtime_error(
            "solve_scattering: multi-open-channel not yet implemented");
    }

    Eigen::VectorXd f_r0(N_ch_), df_r0(N_ch_);
    for (int k = 0; k < N_ch_; ++k) {
        if (state.is_sin[k]) {
            f_r0(k)  = std::sin(state.q_in(k) * r0_);
            df_r0(k) = state.q_in(k) * std::cos(state.q_in(k) * r0_);
        } else {
            f_r0(k)  = std::sinh(state.q_in(k) * r0_);
            df_r0(k) = state.q_in(k) * std::cosh(state.q_in(k) * r0_);
        }
    }

    // Unknown layout: [ A_0 ... A_{N-1}, P_0, Q_0, C_closed_0, C_closed_1, ... ]
    const int N_unknown = N_ch_ + 2 + N_closed;
    const int N_eqn     = 2 * N_ch_;
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(N_eqn, N_unknown);

    const int a_open = open_idx[0];
    const double k0  = state.k_out(a_open);

    // Top half: continuity
    for (int a = 0; a < N_ch_; ++a) {
        for (int k = 0; k < N_ch_; ++k) {
            M(a, k) = state.U_in(a, k) * f_r0(k);
        }
        if (a == a_open) {
            M(a, N_ch_)     = -std::sin(k0 * r0_);
            M(a, N_ch_ + 1) = -std::cos(k0 * r0_);
        } else {
            int slot = -1;
            for (size_t i = 0; i < closed_idx.size(); ++i) {
                if (closed_idx[i] == a) { slot = (int)i; break; }
            }
            double kap = state.k_out(a);
            M(a, N_ch_ + 2 + slot) = -std::exp(-kap * r0_);
        }
    }

    // Bottom half: derivative
    for (int a = 0; a < N_ch_; ++a) {
        for (int k = 0; k < N_ch_; ++k) {
            M(N_ch_ + a, k) = state.U_in(a, k) * df_r0(k);
        }
        if (a == a_open) {
            M(N_ch_ + a, N_ch_)     = -k0 * std::cos(k0 * r0_);
            M(N_ch_ + a, N_ch_ + 1) = +k0 * std::sin(k0 * r0_);
        } else {
            int slot = -1;
            for (size_t i = 0; i < closed_idx.size(); ++i) {
                if (closed_idx[i] == a) { slot = (int)i; break; }
            }
            double kap = state.k_out(a);
            M(N_ch_ + a, N_ch_ + 2 + slot) = kap * std::exp(-kap * r0_);
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullV);
    Eigen::VectorXd null_vec = svd.matrixV().col(N_unknown - 1);

    state.A_in  = null_vec.head(N_ch_);
    state.P_out = Eigen::VectorXd::Zero(N_ch_);
    state.Q_out = Eigen::VectorXd::Zero(N_ch_);
    state.C_out = Eigen::VectorXd::Zero(N_ch_);

    double P0 = null_vec(N_ch_);
    double Q0 = null_vec(N_ch_ + 1);
    state.P_out(a_open) = P0;
    state.Q_out(a_open) = Q0;
    for (size_t i = 0; i < closed_idx.size(); ++i) {
        state.C_out(closed_idx[i]) = null_vec(N_ch_ + 2 + (int)i);
    }

    // Energy normalize: sqrt(P^2+Q^2) -> sqrt(2mu/(pi k0))
    double amp_now = std::sqrt(P0*P0 + Q0*Q0);
    double target  = std::sqrt(2.0 * mu_ / (M_PI * k0));
    double scale   = target / amp_now;

    state.A_in  *= scale;
    state.P_out *= scale;
    state.Q_out *= scale;
    state.C_out *= scale;

    // Sign convention
    if (state.P_out(a_open) < 0.0) {
        state.A_in  *= -1.0;
        state.P_out *= -1.0;
        state.Q_out *= -1.0;
        state.C_out *= -1.0;
    }

    (void)incoming_channel;
    return state;
}

AnalyticState AnalyticSquareWell::build_scattering_state(double E, int incoming_channel)
{
    return solve_scattering(E, incoming_channel);
}

void AnalyticSquareWell::scattering_wavefunction(double E, int incoming_channel,
                                                   std::vector<Eigen::VectorXd>& u_out)
{
    AnalyticState state = solve_scattering(E, incoming_channel);

    const int N_grid = par_->N_grid;
    const double dr  = par_->dr;
    u_out.assign(N_grid, Eigen::VectorXd::Zero(N_ch_));
    for (int ir = 0; ir < N_grid; ++ir) {
        double r = ir * dr;
        u_out[ir] = state.evaluate(r);
    }
}

// ============================================================
// Multi-open energy-normalized scattering states (K-matrix)
// ============================================================
//
// For a given E above the lowest threshold, build N_open (= number of
// channels open at this E) energy-normalized scattering states. State
// beta is labelled by the open-channel index it carries the incoming
// wave on.
//
// Asymptotic form (per state beta):
//   alpha open:  u^beta_alpha(r) = P^beta_alpha sin(k_alpha r) + Q^beta_alpha cos(k_alpha r)
//                P^beta_alpha = sqrt(2 mu/(pi k_alpha)) * delta_{alpha, open(beta)}
//   alpha closed: u^beta_alpha(r) = C^beta_alpha exp(-kappa_alpha r)
//
// Matching at r=r0 yields a 2 N_ch x 2 N_ch linear system with unknowns
// [A_0..A_{Nch-1}, Q_at_open_channels_0..N_open-1, C_at_closed_channels_0..N_closed-1].
// The matrix is the same for all beta; only the RHS depends on beta.
// One LU factorization handles all beta.
//
std::vector<AnalyticState>
AnalyticSquareWell::build_scattering_states_kmatrix(double E) const
{
    AnalyticState scratch;
    scratch.E    = E;
    scratch.N_ch = N_ch_;
    scratch.r0   = r0_;

    inside_decomposition(E, scratch.U_in, scratch.q_in, scratch.is_sin);
    outside_wavevectors(E, scratch.k_out, scratch.is_open);

    std::vector<int> open_idx;
    std::vector<int> closed_idx;
    for (int a = 0; a < N_ch_; ++a) {
        if (scratch.is_open[a]) open_idx.push_back(a);
        else                    closed_idx.push_back(a);
    }
    const int N_open   = (int)open_idx.size();
    const int N_closed = (int)closed_idx.size();

    if (N_open == 0) {
        throw std::runtime_error(
            "build_scattering_states_kmatrix: no open channels at E");
    }

    // Slot lookup: open_slot[a] = j if a is the j-th open channel, else -1
    // Same for closed.
    std::vector<int> open_slot(N_ch_, -1), closed_slot(N_ch_, -1);
    for (int j = 0; j < N_open;   ++j) open_slot[open_idx[j]]     = j;
    for (int j = 0; j < N_closed; ++j) closed_slot[closed_idx[j]] = j;

    // Inside-well basis evaluated at r0
    Eigen::VectorXd f_r0(N_ch_), df_r0(N_ch_);
    for (int k = 0; k < N_ch_; ++k) {
        if (scratch.is_sin[k]) {
            f_r0(k)  = std::sin(scratch.q_in(k) * r0_);
            df_r0(k) = scratch.q_in(k) * std::cos(scratch.q_in(k) * r0_);
        } else {
            f_r0(k)  = std::sinh(scratch.q_in(k) * r0_);
            df_r0(k) = scratch.q_in(k) * std::cosh(scratch.q_in(k) * r0_);
        }
    }

    // Build the matching matrix M (size 2 N_ch x 2 N_ch).
    // Layout of unknowns:
    //   [0 .. N_ch-1]                   : A_k inside amplitudes
    //   [N_ch .. N_ch+N_open-1]         : Q at open channels (same order as open_idx)
    //   [N_ch+N_open .. 2 N_ch-1]       : C at closed channels (same order as closed_idx)
    // Layout of equations:
    //   [0 .. N_ch-1]                   : continuity at r0, channel alpha
    //   [N_ch .. 2 N_ch-1]              : derivative at r0, channel alpha
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(2 * N_ch_, 2 * N_ch_);
    for (int a = 0; a < N_ch_; ++a) {
        for (int k = 0; k < N_ch_; ++k) {
            M(a, k)         = scratch.U_in(a, k) * f_r0(k);
            M(N_ch_ + a, k) = scratch.U_in(a, k) * df_r0(k);
        }
        if (scratch.is_open[a]) {
            const int j = open_slot[a];
            const double k_a = scratch.k_out(a);
            M(a,         N_ch_ + j) = -std::cos(k_a * r0_);
            M(N_ch_ + a, N_ch_ + j) = +k_a * std::sin(k_a * r0_);
        } else {
            const int m = closed_slot[a];
            const double kap = scratch.k_out(a);
            M(a,         N_ch_ + N_open + m) = -std::exp(-kap * r0_);
            M(N_ch_ + a, N_ch_ + N_open + m) = +kap * std::exp(-kap * r0_);
        }
    }

    Eigen::PartialPivLU<Eigen::MatrixXd> lu(M);

    // Solve N_open RHS to get the un-orthonormalized K-matrix states.
    //   u^beta_alpha(r->inf) = sqrt(2mu/(pi k_alpha)) [delta_{alpha,beta} sin(k r)
    //                                                  - K_{alpha,beta} cos(k r)]
    // These are linearly independent but NOT individually energy-normalized
    // (their pairwise overlap is (1+KK^T) delta(E-E'), not the identity).
    // Below we orthonormalize them by multiplying with (1+KK^T)^{-1/2}.
    std::vector<Eigen::VectorXd> A_raw(N_open);     // inside amplitudes
    std::vector<Eigen::VectorXd> Q_raw(N_open);     // Q at open channels (size N_open)
    std::vector<Eigen::VectorXd> C_raw(N_open);     // C at closed channels (size N_closed)
    Eigen::MatrixXd Kmat = Eigen::MatrixXd::Zero(N_open, N_open);

    for (int beta = 0; beta < N_open; ++beta) {
        const int    a_in = open_idx[beta];
        const double k_in = scratch.k_out(a_in);
        const double P_norm = std::sqrt(2.0 * mu_ / (M_PI * k_in));

        Eigen::VectorXd b = Eigen::VectorXd::Zero(2 * N_ch_);
        b(a_in)         = P_norm * std::sin(k_in * r0_);
        b(N_ch_ + a_in) = P_norm * k_in * std::cos(k_in * r0_);

        Eigen::VectorXd x = lu.solve(b);

        A_raw[beta] = x.head(N_ch_);
        Q_raw[beta] = Eigen::VectorXd::Zero(N_open);
        for (int j = 0; j < N_open; ++j) {
            Q_raw[beta](j) = x(N_ch_ + j);
        }
        C_raw[beta] = Eigen::VectorXd::Zero(N_closed);
        for (int m = 0; m < N_closed; ++m) {
            C_raw[beta](m) = x(N_ch_ + N_open + m);
        }

        // K_{alpha,beta} = -Q^beta_{open_idx[alpha]} / sqrt(2mu/(pi k_{open_idx[alpha]}))
        for (int j = 0; j < N_open; ++j) {
            const double k_a = scratch.k_out(open_idx[j]);
            const double norm_a = std::sqrt(2.0 * mu_ / (M_PI * k_a));
            Kmat(j, beta) = -Q_raw[beta](j) / norm_a;
        }
    }

    // Orthonormalizer O = (I + K K^T)^{-1/2}, real symmetric positive-definite.
    // Energy-normalized states are u^β_orth = sum_gamma u^gamma_raw * O_{gamma,beta}.
    Eigen::MatrixXd KKt = Kmat * Kmat.transpose();
    Eigen::MatrixXd Mort = Eigen::MatrixXd::Identity(N_open, N_open) + KKt;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esort(Mort);
    Eigen::VectorXd evals = esort.eigenvalues();
    Eigen::MatrixXd evecs = esort.eigenvectors();
    Eigen::VectorXd inv_sqrt_evals(N_open);
    for (int j = 0; j < N_open; ++j) {
        if (!(evals(j) > 0.0)) {
            throw std::runtime_error(
                "build_scattering_states_kmatrix: 1+KK^T not positive-definite");
        }
        inv_sqrt_evals(j) = 1.0 / std::sqrt(evals(j));
    }
    const Eigen::MatrixXd Oort =
        evecs * inv_sqrt_evals.asDiagonal() * evecs.transpose();

    // Apply Oort to assemble the orthonormalized output states.
    std::vector<AnalyticState> out;
    out.reserve(N_open);
    for (int beta_new = 0; beta_new < N_open; ++beta_new) {
        AnalyticState state = scratch;
        state.A_in  = Eigen::VectorXd::Zero(N_ch_);
        state.P_out = Eigen::VectorXd::Zero(N_ch_);
        state.Q_out = Eigen::VectorXd::Zero(N_ch_);
        state.C_out = Eigen::VectorXd::Zero(N_ch_);

        for (int beta_old = 0; beta_old < N_open; ++beta_old) {
            const double w = Oort(beta_old, beta_new);
            state.A_in += w * A_raw[beta_old];
            for (int j = 0; j < N_open; ++j) {
                state.Q_out(open_idx[j]) += w * Q_raw[beta_old](j);
            }
            for (int m = 0; m < N_closed; ++m) {
                state.C_out(closed_idx[m]) += w * C_raw[beta_old](m);
            }
            // P^beta_alpha_old = sqrt(2mu/(pi k_alpha)) delta_{alpha, open_idx[beta_old]}
            const int a_in_old = open_idx[beta_old];
            const double k_in_old = scratch.k_out(a_in_old);
            const double P_norm_old = std::sqrt(2.0 * mu_ / (M_PI * k_in_old));
            state.P_out(a_in_old) += w * P_norm_old;
        }
        out.push_back(std::move(state));
    }
    return out;
}
