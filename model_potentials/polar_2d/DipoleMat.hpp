#pragma once

#include "Common.hpp"
#include "Wavefunctions.hpp"

#define FILE_PATH(x) (std::string(GENERAL_LIB_PATH) + "/" + (x))
#define FILE_PATH_OUT(x) (std::string(OUTPUT_LIB_PATH)  + "/" + (x))

class DipoleMat {
    public:
        explicit DipoleMat(Wavefunctions *wavefunctions, Parameters *parameters);

        void calculate_complex_dipole_matrix_element_ingoingBC(
            Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double energy);

        static double dipole_function(double phi) { return cos(phi); }

        void real_dipole_matrix_element(std::vector<dcompx> &d_real);

    private:
        Wavefunctions *wavefunctions;
        Parameters *parameters;

        int N_grid;
        int channels;
        int l_max;
        double dr;
        int NTHREADS;
        int N_phi;
        double dphi;
        double vphi_max;
        int Nroots;
};
