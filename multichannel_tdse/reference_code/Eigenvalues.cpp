// Eigenvalues.cpp
//
// Bound-state finder by bisection on the number of nodes. The logic is
// identical to the 2D code's Eigenvalues class, with two refinements:
//   1) highest_bound_state() finds the SHALLOWEST (highest-lying) bound
//      state, which is the halo we want. It does this by first counting
//      nodes at Emax to determine n_total, then bisecting to find the
//      energy at which the node count drops from n_total to n_total - 1.
//   2) The inner bisection uses a tighter tolerance (1e-14) because the
//      halo binding is extremely small (~10 kHz out of ~8 GHz well depth,
//      a relative precision of ~1e-6).
//
#include "Eigenvalues.hpp"

Eigenvalues::Eigenvalues(Equations* equations, Parameters* parameters)
    : gsEnergy(0.0), i_match(0), equations_(equations), parameters_(parameters)
{
    Emin_   = parameters_->Emin;
    Emax_   = parameters_->Emax;
    N_grid_ = parameters_->N_grid;
}

// ============================================================
// Highest bound state (shallowest binding) finder
// ============================================================
// Algorithm:
//   1) Count nodes at E = Emax (should be just above threshold).
//      This is the TOTAL number of bound states in the well.
//   2) Bisect on E to find the crossover where the node count goes
//      from n_total to n_total - 1. That crossover is the highest
//      bound state.
//
// For the tuned halo problem, n_total is the number of states the well
// supports below threshold. The highest state is the halo.
//
void Eigenvalues::highest_bound_state()
{
    // Count nodes at the upper end
    auto [n_total, pos_total] = equations_->OutwardNodeCounting(Emax_);
    cout << "Total bound states at E=Emax: " << n_total << endl;
    if (n_total == 0) {
        throw std::runtime_error("No bound states found below Emax");
    }
    const int target_n = n_total;    // we want the state whose bisection
                                      // boundary sits just at n_total
    // Bisect
    double e_L = Emin_;
    double e_H = Emax_;
    double e_tr = 0.0;

    const double tol = 1e-20;    // HALO precision: ~1 part in 1e9 of well depth
    const int max_iter = 200;

    for (int iter = 0; iter < max_iter; ++iter) {
        e_tr = 0.5 * (e_L + e_H);
        auto [nc, np] = equations_->OutwardNodeCounting(e_tr);

        // If node count at e_tr < target_n, we are BELOW the highest state
        // (not enough nodes yet), so move up. If >= target_n, move down.
        if (nc < target_n) {
            e_L = e_tr;
        } else {
            e_H = e_tr;
        }

        if (std::abs(e_H - e_L) < tol || iter >= max_iter - 1) {
            // Converged
            gsEnergy = 0.5 * (e_L + e_H);
            i_match  = matching_point_finder(gsEnergy);
            cout << "Halo: E = " << std::scientific << std::setprecision(12)
                 << gsEnergy << " Hartree = "
                 << std::fixed << std::setprecision(6)
                 << gsEnergy * AU::Hartree_in_kHz << " kHz\n";
            cout << "i_match = " << i_match
                 << " (r = " << i_match * parameters_->dr << " a_0)\n";
            return;
        }
    }
}

// ============================================================
// Matching-point finder
// ============================================================
// Picks the grid index where the forward and backward ratio matrices
// differ the MOST (by determinant) — that is where we have the cleanest
// matching condition for the bound-state energy.
//
int Eigenvalues::matching_point_finder(double Energy)
{
    Eigen::MatrixXcd Rm, Rmp1;
    equations_->propagateForward(Energy, N_grid_ - 1, Rm, true);
    equations_->propagateBackward(Energy, 0, Rmp1, false);

    double best_det = -1.0;
    int    best_i   = 0;

    for (int i = 5; i < N_grid_ - 5; ++i) {
        dcompx det = (equations_->Rinv_vector[i].inverse()
                    - equations_->Rinv_vector_back[i]).determinant();
        double mag = std::abs(det);
        if (mag > best_det) {
            best_det = mag;
            best_i   = i;
        }
    }
    return best_i;
}
