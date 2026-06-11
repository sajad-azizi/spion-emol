#pragma once

#include "Common.hpp"
#include "Parameters.hpp"

#define FILE_PATH(x) (std::string(GENERAL_LIB_PATH) + "/" + (x))
#define FILE_PATH_OUT(x) (std::string(OUTPUT_LIB_PATH)  + "/" + (x))

class Potentials {
public:
    explicit Potentials(Parameters *parameters);

    double Potential(double r, double phi) const;
    void potMatrixElement();

    static int delta(int k, int l) { return k == l; }

    std::vector<Eigen::MatrixXcd> pot_component;

private:
    Parameters *parameters;

    int external_parameter;
    int channels;
    int N_grid;
    double dr;
    int NTHREADS;
    int Nroots;
    double vphi_max;
    double dphi;
    int N_phi;

    std::vector<std::vector<dcompx>> harmonic_basis;
};
