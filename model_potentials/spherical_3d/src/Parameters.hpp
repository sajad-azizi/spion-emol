// Parameters.hpp -- run parameters for the 3D coupled-channel solver.
#pragma once

#include "Common.hpp"

class Parameters {
public:
    // Radial grid
    int    N_grid;        // number of grid points
    double dr;            // step size (atomic units, bohr)
    // Angular cutoff
    int    l_max;         // max l in real-Y^R basis
    int    n_channels;    // (l_max + 1)^2

    // Energy bracket for bound-state search
    double Emin;
    double Emax;

    // Angular integration grids for V_{(lm)(l'm')}(r)
    int    N_theta;       // Gauss-Legendre nodes in cos(theta)
    int    N_phi;         // trapezoid nodes in phi (uniform on [0, 2pi))

    // Numerov R-matrix initialization length (small-r seed)
    int    p;             // number of seed steps using S_0(kr) trick (>=1)

    // External tunable (e.g. cubic-well half-side L; centi-bohr units to
    // match the polar_2d convention `iAA * 0.01`).
    int    external_parameter;

    // Misc
    int    n_threads;
    int    out_decimation; // output stride for wavefunction dumps

    Parameters() = default;
};
