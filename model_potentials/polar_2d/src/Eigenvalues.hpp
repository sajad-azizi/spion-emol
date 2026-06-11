#pragma once

#include "Common.hpp"
#include "Equations.hpp"

class Eigenvalues {
    public:
        explicit Eigenvalues(Equations *equations, Parameters *parameters);

        void Boundstates_finder();
        void groundstate_finder();
        int matching_point_finder(double Energy);

        double gsEnergy;
        int i_match;

    private:
        Equations *equations;

        double Emin;
        double Emax;
        int N_grid;
};
