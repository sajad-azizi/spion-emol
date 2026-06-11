// Eigenvalues.hpp -- bound-state finder via outward node-counting and
// matching-point selection.  Same recipe as polar_2d.
#pragma once

#include "Common.hpp"
#include "Equations.hpp"
#include "Parameters.hpp"

class Eigenvalues {
public:
    Eigenvalues(Equations& eqs, const Parameters& params);

    // Locate the lowest bound state with `desire_node` outward-node
    // counts in the coupled system.  Default desire_node = 1 finds the
    // lowest bound state of the multi-channel problem.
    void groundstate_finder(int desire_node = 1, double tol = 1e-9);

    // Sweep multiple bound states up to (Emax) by incrementing the node
    // target.
    void Boundstates_finder(double tol = 1e-12);

    // Find a good matching point at fixed E:  outer classical turning
    // point of the s-wave bound state, i.e. the largest grid index ir
    // for which the spherically-averaged V_s(r) ≤ E.  Capped at 2N/3
    // for forward-Numerov contamination safety.
    int matching_point_finder(double E);

    // After matching_point_finder, print a one-block diagnostic
    // verifying r(i_match), V_s vs E across i_match, kappa, the
    // contamination-safe range, and flag any unphysical configuration.
    void print_turning_point_diagnostic(double E, int i_m) const;

    double gsEnergy = 0.0;
    int    i_match  = 0;

private:
    Equations&        eqs_;
    const Parameters& params_;
};
