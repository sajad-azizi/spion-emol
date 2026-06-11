#include "DipoleMat.hpp"

DipoleMat::DipoleMat(Wavefunctions *wavefunctions, Parameters *parameters)
    : wavefunctions(wavefunctions), parameters(parameters)
{
    N_grid   = parameters->N_grid;
    channels = parameters->channels;
    dr       = parameters->dr;
    l_max    = (channels - 1) / 2;
    NTHREADS = parameters->NTHREADS;
    N_phi    = parameters->N_phi;
    dphi     = parameters->dphi;
    vphi_max = 2.0 * M_PI;
    Nroots   = parameters->Nroots;
}


void DipoleMat::calculate_complex_dipole_matrix_element_ingoingBC(
    Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, const double energy)
{
    // Compute real dipole matrix elements
    std::vector<dcompx> d_real(channels, 0.0);
    real_dipole_matrix_element(d_real);

    {
        std::ofstream oout("real_dipole.dat");
        for (int beta = 0; beta < channels; beta++)
            oout << -l_max + beta << "\t" << real(d_real[beta]) << "\t" << imag(d_real[beta]) << "\n";
    }
    cout << "Real dipole done.\n";

    // Transform to complex (ingoing BC) representation
    const double k_wave = sqrt(2.0 * energy);
    Eigen::MatrixXcd AiB_inv = ((A - I * B).inverse()).transpose().conjugate();

    std::vector<dcompx> d_complex(channels, 0.0);
    {
        std::ofstream reduced_dipole_out("reduced_dipole.dat");
        for (int m = 0; m < channels; m++) {
            int loc_m = -l_max + m;
            dcompx sum_beta = 0.0;
            for (int beta = 0; beta < channels; beta++)
                sum_beta += AiB_inv(m, beta) * d_real[beta];

            d_complex[m] = sum_beta;
            reduced_dipole_out << loc_m << "\t"
                << real(d_complex[m]) << "\t" << imag(d_complex[m]) << "\t"
                << std::norm(d_complex[m]) << "\t" << std::arg(d_complex[m]) << "\n";
        }
    }
    cout << "Complex matrix element done.\n";
}


void DipoleMat::real_dipole_matrix_element(std::vector<dcompx> &d_real) {

    // Load Gauss-Legendre roots and weights
    std::vector<double> roots(Nroots), weights(Nroots);
    {
        std::ifstream fin(FILE_PATH("roots_legendre_" + std::to_string(Nroots) + ".dat"));
        if (!fin.is_open()) {
            std::cerr << "Requested roots file does not exist!\n";
            exit(1);
        }
        for (int i = 0; i < Nroots; i++)
            fin >> roots[i] >> weights[i];
    }

    // Angular integration: <e^{i m phi} | cos(phi) | e^{i k phi}>
    Eigen::MatrixXcd angular_integration = Eigen::MatrixXcd::Zero(channels, channels);
    for (int m = 0; m < channels; m++) {
        int loc_m = -l_max + m;
        for (int k = 0; k < channels; k++) {
            int loc_k = -l_max + k;
            double sum_phi_re = 0.0;
            double sum_phi_im = 0.0;

            #pragma omp parallel for default(shared) reduction(+:sum_phi_re,sum_phi_im)
            for (int j = 0; j < Nroots; j++) {
                double tt = 0.5 * vphi_max * roots[j] + vphi_max / 2.0;
                double dip = dipole_function(tt);
                double w   = 0.5 * vphi_max * weights[j] / (2.0 * M_PI);
                sum_phi_re += cos((loc_k - loc_m) * tt) * dip * w;
                sum_phi_im += sin((loc_k - loc_m) * tt) * dip * w;
            }
            angular_integration(m, k) = sum_phi_re + I * sum_phi_im;
        }
    }

    // Radial integration using Simpson's rule
    for (int beta = 0; beta < channels; beta++) {
        dcompx realDipol_sum = 0.0;

        for (int m = 0; m < channels; m++) {
            for (int k = 0; k < channels; k++) {
                double sum_radial_re = 0.0;
                double sum_radial_im = 0.0;

                #pragma omp parallel for default(shared) reduction(+:sum_radial_re,sum_radial_im)
                for (int ir = 1; ir < N_grid - 1; ir++) {
                    double r = ir * dr;
                    dcompx integrand = dr
                        * std::conj(wavefunctions->scattering_eigenfunc[ir](m, beta))
                        * r * wavefunctions->eigfunc[ir](k);

                    double weight = (ir % 2 == 0) ? 2.0 : 4.0;
                    sum_radial_re += weight * real(integrand);
                    sum_radial_im += weight * imag(integrand);
                }

                // Simpson boundary term at last grid point
                dcompx boundary = dr
                    * std::conj(wavefunctions->scattering_eigenfunc[N_grid - 1](m, beta))
                    * double(N_grid - 1) * dr * wavefunctions->eigfunc[N_grid - 1](k);

                realDipol_sum += angular_integration(m, k)
                    * (boundary + sum_radial_re + I * sum_radial_im) / 3.0;
            }
        }
        d_real[beta] = realDipol_sum;
    }
}
