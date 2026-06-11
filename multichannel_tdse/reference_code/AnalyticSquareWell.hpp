// AnalyticSquareWell.hpp
//
// Exact analytic solver for the multichannel square-well potential
// used in the Feshbach problem. Handles:
//
//   - BOUND STATES: all channels closed, match to decaying exponentials
//     outside. The 2N x 2N matching matrix must be singular; its null
//     vector gives the amplitudes.
//
//   - SCATTERING STATES: one or more open channels (k_alpha real) and
//     the rest closed. The matching matrix is 2N x (2N + N_open) and
//     the null space is N_open dimensional. For single-open-channel 
//     problems this reduces to a 1D null space (picked by energy 
//     normalization).
//
// Physics:
//   Inside (r <= r_0):  u'' = -K_in(E) u,  K_in = 2mu (E*I - V_in)
//   Diagonalize K_in = U diag(q^2) U^T; each mode is sin(q_k r) if 
//   q_k^2 > 0 or sinh(|q_k| r) if q_k^2 < 0. Inside solution is
//     u_alpha(r) = Σ_k U_{α,k} A_k f_k(q_k r)
//
//   Outside (r > r_0):  decoupled
//     open channels:   u_alpha = P_alpha sin(k_alpha r) + Q_alpha cos(k_alpha r)
//                                k_alpha = sqrt(2 mu (E - E_th_alpha))
//     closed channels: u_alpha = C_alpha exp(-kappa_alpha r)
//                                kappa_alpha = sqrt(2 mu (E_th_alpha - E))
//
// For a scattering state, the outside open-channel amplitudes are
// energy-normalized:
//   sqrt(P_alpha^2 + Q_alpha^2) = sqrt(2 mu / (pi k_alpha))
// which is the standard energy-delta normalization
//   <E'| E> = delta(E - E').
//
#pragma once

#include "Common.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"

// Internal representation of a fully evaluated state (bound or scattering)
// at a given energy. Stores everything needed to evaluate the wave function
// at any radial point.
struct AnalyticState {
    double E;                         // energy (Hartree)
    int N_ch;
    double r0;

    // Inside-well eigendecomposition
    Eigen::MatrixXd U_in;             // N_ch x N_ch (orthogonal)
    Eigen::VectorXd q_in;             // N_ch (positive)
    std::vector<bool> is_sin;         // N_ch (true if sin, false if sinh)
    Eigen::VectorXd A_in;             // N_ch inside amplitudes

    // Outside: per-channel arrays
    std::vector<bool> is_open;        // N_ch (true if channel is open)
    Eigen::VectorXd k_out;            // k (open) or kappa (closed)
    // If is_open[alpha]: P_out[alpha] and Q_out[alpha] are the sin/cos coefs
    // Else:              C_out[alpha] is the exponential coef, Q_out unused
    Eigen::VectorXd P_out;            // size N_ch
    Eigen::VectorXd Q_out;            // size N_ch
    Eigen::VectorXd C_out;            // size N_ch

    // Evaluate the full N_ch-vector u_alpha(r) at a given radial position
    Eigen::VectorXd evaluate(double r) const;
    Eigen::Vector2d evaluate2(double r) const;
};

class AnalyticSquareWell {
public:
    AnalyticSquareWell(Potentials* potentials, Parameters* parameters);

    // ---- Bound states ----

    // Find the highest bound state (shallowest binding) by scanning
    // det M(E) for sign changes in [E_lo, E_hi] and bisecting the highest.
    // Returns the binding energy (negative Hartree).
    double find_highest_bound_state(double E_lo, double E_hi, int n_scan = 10000);

    // Find ALL bound states in [E_lo, E_hi] by scanning det M(E) for sign
    // changes and bisecting each. For a well whose all channels are closed,
    // this includes every eigenvalue in the range. For a block with some
    // open channels at E_hi, call with E_hi below the lowest open threshold.
    // Returns eigenvalues sorted ascending.
    std::vector<double> find_all_bound_states(double E_lo, double E_hi,
                                              int n_scan = 20000);

    // Construct the wave function for a bound state at energy E (E < 0,
    // and below all thresholds). Populates u_out on the radial grid and
    // normalizes to unit integral: sum_alpha int |u|^2 dr = 1.
    void build_bound_wavefunction(double E, std::vector<Eigen::VectorXd>& u_out);

    // Construct the normalized analytic bound-state object without allocating
    // a full grid wavefunction.
    AnalyticState build_bound_state(double E);

    // Matching-matrix determinant for bound-state search.
    double matching_determinant(double E);

    // ---- Scattering states ----

    // Construct a scattering state at energy E. At least one channel must
    // be open (k_alpha real); the remaining channels are closed. The state
    // is energy-normalized.
    //
    // For single-open-channel problems (our M_F=-4 continuum and M_F=-3
    // intermediates near threshold), the null space is 1D and the energy
    // normalization picks a unique state.
    //
    // For multi-open-channel problems the user must specify which open
    // channel is the "incoming" channel (open_channel_index) — the 
    // resulting state is the one where only that open channel has an
    // incoming wave; all other open channels have outgoing waves only.
    // This matches the standard K-matrix / S-matrix boundary conditions.
    //
    AnalyticState build_scattering_state(double E, int incoming_channel = 0);

    // Multi-open energy-normalized scattering states in K-matrix convention.
    // Returns N_open AnalyticState's, indexed by the incoming-channel label
    // beta = 0..N_open-1. For each beta:
    //   u^beta_alpha(r > r0) = P^beta_alpha sin(k_alpha r) + Q^beta_alpha cos(k_alpha r),
    //   P^beta_alpha = sqrt(2 mu/(pi k_alpha)) delta_{alpha,beta},
    //   Q^beta_alpha = -sqrt(2 mu/(pi k_alpha)) K_{alpha,beta},
    // closed channels exponentially decaying. <E,alpha|E',beta> = delta_{alpha,beta} delta(E-E').
    //
    // Works for any N_open >= 1. For N_open == 1 the result is the same
    // physical state as build_scattering_state up to an overall sign choice
    // (K-matrix convention puts the entire phase into Q rather than rotating
    // the (P, Q) pair to put a positive P).
    std::vector<AnalyticState> build_scattering_states_kmatrix(double E) const;

    // Evaluate a scattering state on the grid and write to u_out.
    void scattering_wavefunction(double E, int incoming_channel,
                                  std::vector<Eigen::VectorXd>& u_out);

    // Accessors for inspection
    int num_channels() const { return N_ch_; }
    double mu() const { return mu_; }
    double r0_val() const { return r0_; }
    const Eigen::VectorXd& thresholds() const { return thresholds_; }

private:
    Potentials* pot_;
    Parameters* par_;

    int    N_ch_;
    double mu_;
    double r0_;
    double V_bar_;
    double dV_;
    Eigen::MatrixXd s1s2_;
    Eigen::VectorXd thresholds_;    // in a.u.

    // Internal: build the inside-well eigendecomposition at energy E
    void inside_decomposition(double E,
                               Eigen::MatrixXd& U_in,
                               Eigen::VectorXd& q_vals,
                               std::vector<bool>& is_sin) const;

    // Internal: build the outside wave-vectors (k or kappa) and open/closed flags
    void outside_wavevectors(double E,
                              Eigen::VectorXd& k_out,
                              std::vector<bool>& is_open) const;

    // Internal: build the matching matrix for a bound-state problem
    // (all channels closed). Returns a 2N x 2N matrix whose singular
    // nature indicates a bound-state energy.
    void build_bound_matching_matrix(double E,
                                      Eigen::MatrixXd& M,
                                      Eigen::MatrixXd& U_in,
                                      Eigen::VectorXd& q_vals,
                                      std::vector<bool>& is_sin,
                                      Eigen::VectorXd& kappa_out) const;

    // Internal: solve the scattering state by matching.
    // The matching is a 2*N_ch x (2*N_ch + N_open) linear system with an
    // N_open-dimensional null space.
    AnalyticState solve_scattering(double E, int incoming_channel) const;
};
