// Eigenvalues.hpp
//
// Bound-state finder by bisection on node count.
// Identical interface to the 2D code's Eigenvalues class.
//
#pragma once

#include "Common.hpp"
#include "Parameters.hpp"
#include "Equations.hpp"

class Eigenvalues {
public:
    Eigenvalues(Equations* equations, Parameters* parameters);

    // Find the HIGHEST bound state (shallowest binding).
    // Sets gsEnergy (in Hartree) and i_match (grid index of the matching point).
    void highest_bound_state();

    // Find all bound states up to the energy cutoff Emax. Writes to a file.
    void bound_states_finder();

    // Locate the matching point: the grid index where forward and backward
    // propagations of the ratio matrix differ by the most (max |det|).
    int matching_point_finder(double Energy);

    // Results
    double gsEnergy;   // in Hartree
    int    i_match;    // grid index

private:
    Equations*  equations_;
    Parameters* parameters_;

    double Emin_;
    double Emax_;
    int N_grid_;
    int N_ch_;
};
