#include "Common.hpp"
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_bessel.h>

#include "Potentials.hpp"
#include "Equations.hpp"
#include "Eigenvalues.hpp"
#include "Wavefunctions.hpp"
#include "DipoleMat.hpp"

#define FILE_PATH(x) (std::string(GENERAL_LIB_PATH) + "/" + (x))
#define FILE_PATH_OUT(x) (std::string(OUTPUT_LIB_PATH)  + "/" + (x))

int main() {

    // --- Grid parameters ---
    const double r_max = 300.0;
    const double dr    = 0.01;
    const int N_grid   = int(r_max / dr) + 1;

    // --- Angular grid ---
    const double vphi_max = 2.0 * M_PI;
    const double dphi     = 0.001;
    const int N_phi       = int(vphi_max / dphi) + 1;

    // --- Channel structure ---
    const int l_max    = 15;
    const int channels = 2 * l_max + 1;

    // --- Energy bounds ---
    const double Emin = -30.7;
    const double Emax =   1.0;

    // --- Gauss-Legendre integration ---
    const int Nroots = 1000;

    // --- External parameter (box size) ---
    int external_parameter = 150;
    {
        std::ifstream fin("L_ini.dat");
        if (fin.is_open()) fin >> external_parameter;
    }
    cout << "external_parameter L: " << external_parameter << endl;

    // --- OpenMP ---
    const int NTHREADS = omp_get_max_threads();
    cout << "Nthreads: " << NTHREADS << endl;

    // --- Numerov parameters ---
    const int divide = 8;
    const int p = 9;  // correction points for ratio matrix R (min = 1)

    std::cout << std::fixed << std::setprecision(14);
    std::cerr << std::fixed << std::setprecision(14);

    // --- Continuum energy ---
    int iA = 100;
    {
        std::ifstream fin("A_ini.dat");
        if (fin.is_open()) fin >> iA;
    }
    double Energy = iA * 0.01;
    cout << "Continuum energy: " << Energy << endl;

    // ============================================================
    // Build objects
    // ============================================================
    Parameters parameters(N_grid, channels, dr, Nroots, NTHREADS,
                          N_phi, dphi, Emin, Emax, divide, p, external_parameter);

    Potentials potentials(&parameters);
    potentials.potMatrixElement();
    cout << "Potential done.\n";

    Equations equations(&potentials, &parameters);

    // --- Find ground state ---
    Eigenvalues eigenvalues(&equations, &parameters);
    eigenvalues.groundstate_finder();
    cout << "i_match: " << eigenvalues.i_match << "  gsEnergy: " << eigenvalues.gsEnergy << endl;

    // --- Compute bound-state wavefunction ---
    Wavefunctions wavefunctions(&equations, &parameters);
    wavefunctions.calculate_eigenfunction(eigenvalues.gsEnergy, eigenvalues.i_match);
    wavefunctions.Normalization(wavefunctions.eigfunc);

    // Print ground state wavefunction per channel
    {
        std::ofstream bout;
        for (int m = 0; m < channels; m++) {
            bout.open("basissets/psi_" + std::to_string(m) + ".dat");
            for (int k = 0; k < N_grid; ++k)
                bout << k * dr << "\t"
                     << real(wavefunctions.eigfunc[k](m)) << "\t"
                     << imag(wavefunctions.eigfunc[k](m)) << "\n";
            bout.close();
        }
    }

    // --- Scattering calculation ---
    Eigen::MatrixXcd In = Eigen::MatrixXcd::Identity(channels, channels);
    Eigen::MatrixXcd A  = Eigen::MatrixXcd::Zero(channels, channels);
    Eigen::MatrixXcd B  = Eigen::MatrixXcd::Zero(channels, channels);

    wavefunctions.calculate_channel_wavefunction(Energy);
    wavefunctions.calculate_A_B_matrices(A, B, Energy);

    Eigen::MatrixXcd K = B * A.inverse();
    Eigen::MatrixXcd S = (In + I * K) * (In - I * K).inverse();

    Eigen::MatrixXcd unitary = S.conjugate().transpose() * S;
    {
        std::ofstream Sout("sk_matrix.dat");
        for (int j = 0; j < channels; j++) {
            dcompx sum_uni = 0.0;
            for (int m = 0; m < channels; m++) {
                sum_uni += unitary(j, m);
                Sout << real(S(j, m)) << "\t" << imag(S(j, m)) << "\t"
                     << real(K(j, m)) << "\t" << imag(K(j, m)) << "\n";
            }
            cout << -l_max + j
                 << " (~1) " << std::norm(sum_uni)
                 << " (~0) " << 1.0 - std::norm(sum_uni)
                 << " diag |S|^2: " << std::norm(S(j, j))
                 << " diag arg(S)/2: " << std::arg(S(j, j)) / 2.0 << endl;
        }
    }

    // --- Dipole matrix element ---
    DipoleMat dipoleMat(&wavefunctions, &parameters);
    dipoleMat.calculate_complex_dipole_matrix_element_ingoingBC(A, B, Energy);

    cout << "Done.\n";
    return 0;
}
