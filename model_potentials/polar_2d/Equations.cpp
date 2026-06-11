#include "Equations.hpp"

Equations::Equations(Potentials *potentials, Parameters *parameters)
    : potentials(potentials), parameters(parameters)
{
    N_grid   = parameters->N_grid;
    channels = parameters->channels;
    dr       = parameters->dr;
    l_max    = (channels - 1) / 2;
    p        = parameters->p;

    Rinv_vector.resize(N_grid);
    Rinv_vector_back.resize(N_grid);
    Winv_vector.resize(N_grid);
    for (int k = 0; k < N_grid; ++k) {
        Rinv_vector[k]      = Eigen::MatrixXcd::Zero(channels, channels);
        Rinv_vector_back[k] = Eigen::MatrixXcd::Zero(channels, channels);
        Winv_vector[k]      = Eigen::MatrixXcd::Zero(channels, channels);
    }

    Rinv = Eigen::MatrixXcd::Zero(channels, channels);
    In   = Eigen::MatrixXcd::Identity(channels, channels);
    R    = Eigen::MatrixXcd::Zero(channels, channels);
    Wmat = Eigen::MatrixXcd::Zero(channels, channels);
    U    = Eigen::MatrixXcd::Zero(channels, channels);
}


void Equations::proper_initialization_R(double Energy, Eigen::MatrixXcd &resRinv) {

    const double hh12 = dr * dr / 12.0;

    // Correction of m=0 channel for first p steps using J0(kr)
    dcompx k_wave = 0.0;

    for (int i = 1; i < p; i++) {
        Wmat = In + hh12 * (Energy * In - potentials->pot_component[i]) * 2.0;
        U = 12.0 * Wmat.inverse() - 10.0 * In;
        R = U - Rinv;

        double r   = i * dr;
        double rp1 = (i + 1) * dr;

        double loc_energy = Energy - potentials->Potential(r, 0);
        k_wave = (loc_energy < 0.0) ? I * sqrt(2.0 * std::abs(loc_energy))
                                     : sqrt(2.0 * std::abs(loc_energy));

        dcompx F1 = Wmat(l_max, l_max) * imaginary_riccati_Jn(0, k_wave * r) / sqrt(2.0 * M_PI);

        Wmat(l_max, l_max) = In(l_max, l_max)
            + hh12 * (Energy * In(l_max, l_max) - potentials->pot_component[i + 1](l_max, l_max)) * 2.0;

        loc_energy = Energy - potentials->Potential(rp1, 0);
        k_wave = (loc_energy < 0.0) ? I * sqrt(2.0 * std::abs(loc_energy))
                                     : sqrt(2.0 * std::abs(loc_energy));

        dcompx F2 = Wmat(l_max, l_max) * imaginary_riccati_Jn(0, k_wave * rp1) / sqrt(2.0 * M_PI);
        R(l_max, l_max) = F2 / F1;

        Rinv = R.inverse();
        Rinv_vector[i] = Rinv;
        Winv_vector[i] = Wmat.inverse();
    }

    resRinv = Rinv;
}


std::pair<int, double> Equations::OutwardNodeCounting(double Energy) {

    const double hh12 = dr * dr / 12.0;

    proper_initialization_R(Energy, Rinv);

    int node_c = 0;
    double node_pos = 0.0;

    for (int i = p; i < N_grid; i++) {
        Wmat = In + hh12 * (Energy * In - potentials->pot_component[i]) * 2.0;
        U = 12.0 * Wmat.inverse() - 10.0 * In;
        R = U - Rinv;
        Rinv = R.inverse();

        es.compute(R, Eigen::EigenvaluesOnly);
        es1.compute(Wmat, Eigen::EigenvaluesOnly);

        for (int m = 0; m < channels; m++) {
            if (es1.eigenvalues().real()(m) > 0.0 && es.eigenvalues().real()(m) < 0.0) {
                node_c++;
                node_pos = i * dr;
            }
        }
    }

    return {node_c, node_pos};
}


void Equations::propagateBackward(double Energy, int i_match, Eigen::MatrixXcd &resRmp1, bool save) {

    const double hh12 = dr * dr / 12.0;

    for (int i = N_grid - 2; i > i_match; i--) {
        Wmat = In + hh12 * (Energy * In - potentials->pot_component[i]) * 2.0;
        U = 12.0 * Wmat.inverse() - 10.0 * In;
        R = U - Rinv;
        Rinv = R.inverse();

        if (save) {
            Rinv_vector[i] = Rinv;
            Winv_vector[i] = Wmat.inverse();
        }
        Rinv_vector_back[i] = Rinv;
    }

    resRmp1 = R;
}


void Equations::propagateForward(double Energy, int i_match, Eigen::MatrixXcd &resRm, bool save) {

    const double hh12 = dr * dr / 12.0;

    proper_initialization_R(Energy, Rinv);

    for (int i = p; i <= i_match; i++) {
        Wmat = In + hh12 * (Energy * In - potentials->pot_component[i]) * 2.0;
        U = 12.0 * Wmat.inverse() - 10.0 * In;
        R = U - Rinv;
        Rinv = R.inverse();

        if (save) {
            Rinv_vector[i] = Rinv;
            Winv_vector[i] = Wmat.inverse();
        }
    }

    resRm = R;
}


dcompx Equations::imaginary_riccati_Jn(int n, dcompx x) {
    const int abs_n = std::abs(n);
    const double abs_x = std::abs(x);
    const double prefix = sqrt(M_PI * abs_x / 2.0);
    const double sign = (n < 0) ? std::pow(-1.0, abs_n) : 1.0;

    if (real(x) == 0.0) {
        // Pure imaginary argument
        return sign * I * prefix * std::exp(I * M_PI * double(n) / 2.0)
               * std::cyl_bessel_i(abs_n, abs_x);
    } else if (imag(x) == 0.0) {
        // Pure real argument
        return sign * prefix * std::cyl_bessel_j(abs_n, abs_x);
    } else {
        std::cerr << "Error: Bessel argument is complex (not pure real or imaginary)\n";
        exit(1);
        return 0.0;
    }
}
