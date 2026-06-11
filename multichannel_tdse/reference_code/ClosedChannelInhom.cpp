// ClosedChannelInhom.cpp
//
// Finite-difference block-Thomas solver for
//
//   (E_int - H) X = S
//
// where H is the multichannel Hamiltonian of a closed block on a
// uniform radial grid, discretized with three-point central difference:
//
//   -1/(2mu dr^2) (X_{i-1} - 2 X_i + X_{i+1}) + V_i X_i
//      = E_int X_i - S_i
//
// which we rewrite as
//
//   A X_{i-1} + B_i X_i + C X_{i+1} = -S_i,
//
//   A = C = -1/(2 mu dr^2) I_{N_ch}
//   B_i = (1/(mu dr^2)) I + V_i - E_int I
//
// Boundary conditions:
//   - At r=0: X_0 = 0  (regular)
//   - At r=L: Robin  X'(L) = -kappa X(L), per channel
//
// The Robin BC at the last grid point is enforced by modifying the
// last row of the block-tridiagonal system.
//
// Block-Thomas pass:
//   Forward:  B'_0 = B_0; d'_0 = -S_0
//             for i in 1..N-1:
//                 M = A * B'_{i-1}^{-1}
//                 B'_i = B_i - M * C
//                 d'_i = -S_i - M * d'_{i-1}
//   Backward: X_{N-1} = B'_{N-1}^{-1} d'_{N-1}
//             for i in N-2..0:
//                 X_i = B'_i^{-1} (d'_i - C * X_{i+1})
//
// Cost: O(N_grid * N_ch^3). For N_grid=10000, N_ch=2 this is fast.

#include "ClosedChannelInhom.hpp"
#include <stdexcept>
#include <cmath>

ClosedChannelInhom::ClosedChannelInhom(const Potentials* pot,
                                        const Parameters* params,
                                        double E_int)
    : pot_(pot),
      params_(params),
      E_int_(E_int),
      N_grid_(params->N_grid),
      N_ch_(pot->num_channels()),
      dr_(params->dr),
      mu_(params->mu)
{
    // Verify all channels are closed at this energy
    const Eigen::VectorXd& thr = pot_->thresholds();
    kappa_out_ = Eigen::VectorXd::Zero(N_ch_);
    for (int a = 0; a < N_ch_; ++a) {
        double gap = thr(a) - E_int;
        if (gap <= 0.0) {
            throw std::runtime_error(
                "ClosedChannelInhom: channel is open at E_int "
                "(not a strictly closed problem)");
        }
        kappa_out_(a) = std::sqrt(2.0 * mu_ * gap);
    }
}

std::vector<Eigen::VectorXd>
ClosedChannelInhom::solve(const std::vector<Eigen::VectorXd>& S) const
{
    const int N = N_grid_;
    const int Nc = N_ch_;
    if ((int)S.size() != N) {
        throw std::runtime_error("ClosedChannelInhom::solve: source has wrong grid size");
    }
    for (int i = 0; i < N; ++i) {
        if (S[i].size() != Nc) {
            throw std::runtime_error("ClosedChannelInhom::solve: source has wrong channel size");
        }
    }

    const double h2 = dr_ * dr_;
    const double a_coef = -1.0 / (2.0 * mu_ * h2);   // off-diagonal coeff

    // A and C are identical: a_coef * I
    const Eigen::MatrixXd A = a_coef * Eigen::MatrixXd::Identity(Nc, Nc);
    const Eigen::MatrixXd C = A;

    // Regular BC at r=0: X_0 = 0. We therefore exclude index 0 and solve
    // for X_1..X_{N-1}. For the interior equation at i=1, the A*X_0 term
    // vanishes, so we can just set up the system from index 1 to N-1.
    //
    // For clarity, redefine the local system to be indexed j = 0..M-1
    // where j = i - 1 (so j=0 corresponds to grid index i=1). Size M = N-1.
    const int M = N - 1;

    // Build storage for B[j] and S[j] (j-indexed, j from 0 to M-1)
    //
    // Sign convention: the physical equation is (E_int - H) X = S.
    // Discretized with H = -(1/2mu) d^2/dr^2 + V and using
    // A = C = -1/(2 mu dr^2), we get
    //   A X_{i-1} + B_i X_i + C X_{i+1} = -S_i
    // where B_i = 1/(mu dr^2) + V_i - E_int, i.e. the tridiagonal system
    // has -S on the RHS, not +S.
    std::vector<Eigen::MatrixXd> B(M);
    std::vector<Eigen::VectorXd> d(M);
    for (int j = 0; j < M; ++j) {
        const int i = j + 1;
        Eigen::MatrixXd V = pot_->real_matrix_at_index(i);
        B[j] = (1.0 / (mu_ * h2)) * Eigen::MatrixXd::Identity(Nc, Nc)
             + V
             - E_int_ * Eigen::MatrixXd::Identity(Nc, Nc);
        d[j] = -S[i];
    }

    // Apply the decaying tail boundary at r=L.
    // The exterior asymptote X(r) ∝ e^{-kappa r} implies
    //   X_N = exp(-kappa dr) X_{N-1}
    // Substituting X_N in the last row of the tridiagonal system:
    //   A X_{N-2} + [B_{M-1} + C * diag(exp(-kappa dr))] X_{N-1} = -S_{N-1}
    // This is the "exponential tail" closure — exact for a purely
    // exponential asymptote, accurate when V has died off well before
    // r=L and X has reached the asymptotic form by then.
    {
        Eigen::MatrixXd tail = Eigen::MatrixXd::Zero(Nc, Nc);
        for (int a = 0; a < Nc; ++a) {
            tail(a, a) = std::exp(-kappa_out_(a) * dr_);
        }
        B[M - 1] += C * tail;
    }

    // Forward block-Thomas sweep.
    // Keep the LU decomposition of each modified B[j] so we don't
    // recompute inverses.
    std::vector<Eigen::PartialPivLU<Eigen::MatrixXd>> Blu(M);
    Blu[0].compute(B[0]);

    for (int j = 1; j < M; ++j) {
        // M_mat = A * B[j-1]^{-1}
        Eigen::MatrixXd M_mat = Blu[j - 1].solve(A.transpose()).transpose();
        // Update B[j] -> B[j] - M_mat * C
        B[j] -= M_mat * C;
        // Update d[j] -> d[j] - M_mat * d[j-1]
        d[j] -= M_mat * d[j - 1];
        Blu[j].compute(B[j]);
    }

    // Backward substitution
    std::vector<Eigen::VectorXd> X_local(M);
    X_local[M - 1] = Blu[M - 1].solve(d[M - 1]);
    for (int j = M - 2; j >= 0; --j) {
        X_local[j] = Blu[j].solve(d[j] - C * X_local[j + 1]);
    }

    // Pack back to grid indexing with X_0 = 0
    std::vector<Eigen::VectorXd> X(N, Eigen::VectorXd::Zero(Nc));
    for (int j = 0; j < M; ++j) {
        X[j + 1] = X_local[j];
    }
    return X;
}

std::vector<Eigen::Vector2d>
ClosedChannelInhom::solve_2ch(const std::function<Eigen::Vector2d(int)>& source) const
{
    if (N_ch_ != 2) {
        throw std::runtime_error("ClosedChannelInhom::solve_2ch requires N_ch == 2");
    }

    const int N = N_grid_;
    const int M = N - 1;
    const double h2 = dr_ * dr_;
    const double a_coef = -1.0 / (2.0 * mu_ * h2);
    const Eigen::Matrix2d I2 = Eigen::Matrix2d::Identity();
    const Eigen::Matrix2d A = a_coef * I2;
    const Eigen::Matrix2d C = A;

    auto build_B = [&](int i) {
        return (1.0 / (mu_ * h2)) * I2
             + pot_->real_matrix2_at_index(i)
             - E_int_ * I2;
    };

    Eigen::Matrix2d tail = Eigen::Matrix2d::Zero();
    for (int a = 0; a < 2; ++a) {
        tail(a, a) = std::exp(-kappa_out_(a) * dr_);
    }

    // Modified block diagonal and RHS after the forward Thomas sweep.
    // Fixed-size Eigen blocks avoid one heap allocation per grid point.
    std::vector<Eigen::Matrix2d> Bmod(M);
    std::vector<Eigen::Vector2d> dmod(M);

    for (int j = 0; j < M; ++j) {
        const int i = j + 1;
        Eigen::Matrix2d Bj = build_B(i);
        if (j == M - 1) {
            Bj += C * tail;
        }
        Eigen::Vector2d dj = -source(i);

        if (j == 0) {
            Bmod[j] = Bj;
            dmod[j] = dj;
        } else {
            Eigen::PartialPivLU<Eigen::Matrix2d> lu_prev(Bmod[j - 1]);
            Eigen::Matrix2d M_mat = lu_prev.solve(A.transpose()).transpose();
            Bmod[j] = Bj - M_mat * C;
            dmod[j] = dj - M_mat * dmod[j - 1];
        }
    }

    std::vector<Eigen::Vector2d> X(N, Eigen::Vector2d::Zero());
    {
        Eigen::PartialPivLU<Eigen::Matrix2d> lu_last(Bmod[M - 1]);
        X[N - 1] = lu_last.solve(dmod[M - 1]);
    }
    for (int j = M - 2; j >= 0; --j) {
        const int i = j + 1;
        Eigen::PartialPivLU<Eigen::Matrix2d> lu(Bmod[j]);
        X[i] = lu.solve(dmod[j] - C * X[i + 1]);
    }
    return X;
}

double ClosedChannelInhom::relative_residual(const std::vector<Eigen::VectorXd>& X,
                                             const std::vector<Eigen::VectorXd>& S) const
{
    const int N = N_grid_;
    const int Nc = N_ch_;
    if ((int)X.size() != N || (int)S.size() != N) {
        throw std::runtime_error("ClosedChannelInhom::relative_residual: wrong grid size");
    }
    for (int i = 0; i < N; ++i) {
        if (X[i].size() != Nc || S[i].size() != Nc) {
            throw std::runtime_error("ClosedChannelInhom::relative_residual: wrong channel size");
        }
    }

    const double h2 = dr_ * dr_;
    const double a_coef = -1.0 / (2.0 * mu_ * h2);
    const Eigen::MatrixXd A = a_coef * Eigen::MatrixXd::Identity(Nc, Nc);
    const Eigen::MatrixXd C = A;

    double res2 = 0.0;
    double src2 = 0.0;
    for (int i = 1; i < N; ++i) {
        const Eigen::MatrixXd V = pot_->real_matrix_at_index(i);
        Eigen::MatrixXd B = (1.0 / (mu_ * h2)) * Eigen::MatrixXd::Identity(Nc, Nc)
                          + V
                          - E_int_ * Eigen::MatrixXd::Identity(Nc, Nc);

        Eigen::VectorXd x_next;
        if (i + 1 < N) {
            x_next = X[i + 1];
        } else {
            x_next = Eigen::VectorXd::Zero(Nc);
            for (int a = 0; a < Nc; ++a) {
                x_next(a) = std::exp(-kappa_out_(a) * dr_) * X[i](a);
            }
        }

        // solve() enforces (H - E_int)X = -S. This is equivalent to
        // (E_int - H)X = S, so the physical residual is:
        //   -[(H - E_int)X] - S
        const Eigen::VectorXd hx_minus_ex = A * X[i - 1] + B * X[i] + C * x_next;
        const Eigen::VectorXd r = -hx_minus_ex - S[i];
        res2 += r.squaredNorm();
        src2 += S[i].squaredNorm();
    }

    res2 *= dr_;
    src2 *= dr_;
    return std::sqrt(res2) / std::max(std::sqrt(src2), 1e-300);
}

double ClosedChannelInhom::relative_residual_2ch(
    const std::vector<Eigen::Vector2d>& X,
    const std::function<Eigen::Vector2d(int)>& source) const
{
    if (N_ch_ != 2) {
        throw std::runtime_error("ClosedChannelInhom::relative_residual_2ch requires N_ch == 2");
    }
    const int N = N_grid_;
    if ((int)X.size() != N) {
        throw std::runtime_error("ClosedChannelInhom::relative_residual_2ch: wrong grid size");
    }

    const double h2 = dr_ * dr_;
    const double a_coef = -1.0 / (2.0 * mu_ * h2);
    const Eigen::Matrix2d I2 = Eigen::Matrix2d::Identity();
    const Eigen::Matrix2d A = a_coef * I2;
    const Eigen::Matrix2d C = A;

    double res2 = 0.0;
    double src2 = 0.0;
    for (int i = 1; i < N; ++i) {
        Eigen::Matrix2d B = (1.0 / (mu_ * h2)) * I2
                          + pot_->real_matrix2_at_index(i)
                          - E_int_ * I2;

        Eigen::Vector2d x_next;
        if (i + 1 < N) {
            x_next = X[i + 1];
        } else {
            x_next = Eigen::Vector2d::Zero();
            for (int a = 0; a < 2; ++a) {
                x_next(a) = std::exp(-kappa_out_(a) * dr_) * X[i](a);
            }
        }

        const Eigen::Vector2d S = source(i);
        const Eigen::Vector2d hx_minus_ex = A * X[i - 1] + B * X[i] + C * x_next;
        const Eigen::Vector2d r = -hx_minus_ex - S;
        res2 += r.squaredNorm();
        src2 += S.squaredNorm();
    }

    res2 *= dr_;
    src2 *= dr_;
    return std::sqrt(res2) / std::max(std::sqrt(src2), 1e-300);
}
