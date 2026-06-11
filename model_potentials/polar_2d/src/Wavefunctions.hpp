#pragma once

#include "Common.hpp"
#include <gsl/gsl_sf_bessel.h>
#include "Equations.hpp"

class Wavefunctions{
    public:
        explicit Wavefunctions(Equations *equations, Parameters *parameters);

        void calculate_eigenfunction(double Energy, int i_match);
        void calculate_eigenfunction_continuum(double E);
        void Normalization(std::vector<Eigen::VectorXcd> &eigfunc);
        void calculate_channel_wavefunction(double Energy);
        void calculate_final_continumm_states(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double energy);
        void calculate_A_B_matrices(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double Energy);

        std::pair<double, double> getCoefficents(const std::vector<double> &data, double k, int loc_m);

        static double riccati_Jn(int n, double x);
        static double riccati_Yn(int n, double x);
        static double spherical_jn(int n, double x);
        static double spherical_yn(int n, double x);

        std::vector<Eigen::VectorXcd> eigfunc;             // bound state
        std::vector<Eigen::MatrixXcd> scattering_eigenfunc; // scattering

    private:
        Equations *equations;
        Parameters *parameters;

        int N_grid;
        int channels;
        int l_max;
        double dr;
        int NTHREADS;
        int divide;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es;
};
