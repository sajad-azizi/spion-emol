#pragma once

#include "Common.hpp"
#include "Potentials.hpp"

class Equations {
    public:
        explicit Equations(Potentials *potentials, Parameters *parameters);

        void propagateBackward(double Energy, int i_match, Eigen::MatrixXcd &resRmp1, bool save);
        void propagateForward(double Energy, int i_match, Eigen::MatrixXcd &resRm, bool save);
        void proper_initialization_R(double Energy, Eigen::MatrixXcd &resRinv);
        std::pair<int, double> OutwardNodeCounting(double Energy);
        dcompx imaginary_riccati_Jn(int n, dcompx x);

        // Saved propagation data (needed by Wavefunctions)
        std::vector<Eigen::MatrixXcd> Rinv_vector;
        std::vector<Eigen::MatrixXcd> Rinv_vector_back;
        std::vector<Eigen::MatrixXcd> Winv_vector;

    private:
        Potentials *potentials;
        Parameters *parameters;

        int N_grid;
        int channels;
        double dr;
        int l_max;
        int p;  // number of correction points for ratio matrix R

        // Renormalized Numerov working matrices
        Eigen::MatrixXcd Rinv;
        Eigen::MatrixXcd In;
        Eigen::MatrixXcd R;
        Eigen::MatrixXcd Wmat;
        Eigen::MatrixXcd U;

        // Eigensolvers for node counting
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es1;
};
