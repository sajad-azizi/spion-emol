#include "Potentials.hpp"

Potentials::Potentials(Parameters *parameters) : parameters(parameters) {
    N_grid = parameters->N_grid;
    channels = parameters->channels;
    dr = parameters->dr;
    Nroots = parameters->Nroots;
    NTHREADS = parameters->NTHREADS;
    external_parameter = parameters->external_parameter;
    vphi_max = 2.0 * M_PI;
    dphi = parameters->dphi;
    N_phi = parameters->N_phi;

    pot_component.resize(N_grid);
    for (int k = 0; k < N_grid; ++k)
        pot_component[k] = Eigen::MatrixXcd::Zero(channels, channels);

    harmonic_basis.resize(Nroots);
    for (int k = 0; k < Nroots; ++k)
        harmonic_basis[k] = std::vector<dcompx>(channels, 0.0);
}

double Potentials::Potential(double r, double phi) const {

    int iAA = external_parameter;

    double x = r * cos(phi);
    double y = r * sin(phi);

    // --- Library of potentials (uncomment/return the one you need) ---

    // Coulomb
    // double v0 = -1.0 / r;

    // Gaussian well
    // double v1 = -exp(-r * r / 3.0);

    // Yukawa
    // double v2 = -exp(-0.2 * r) / r;

    // Harmonic oscillator
    // double v3 = 0.5 * r * r;

    // Shielded Coulomb
    // double alp = 2.1325;
    // double v4 = -(1.0 + exp(-alp * r)) / r;

    // Anisotropic Gaussian
    // double v6 = -exp(-(x * x / 3.0 + y * y / 27.0));

    // Triple Gaussian
    // double v7 = -exp(-((x+3.)*(x+3.)+(y-2.)*(y-2.))/9.0)
    //             -exp(-((x-3.)*(x-3.)+(y+2.)*(y+2.))/9.0)
    //             -exp(-((x+1.)*(x+1.)+(y+3.)*(y+3.))/9.0);

    // Anisotropic harmonic oscillator
    // double w = 1.5;
    // double v8 = 0.5 * (x * x + w * w * y * y);

    // Double-center soft Coulomb
    // double R = 2.0;
    // double v10 = -1.0/sqrt((x-R/2.)*(x-R/2.)+y*y+0.6384)
    //              -1.0/sqrt((x+R/2.)*(x+R/2.)+y*y+0.6384);

    // Quad-center soft Coulomb
    // double v11 = -1.0/sqrt(x*x+(y-3)*(y-3)+0.6384)
    //              -1.0/sqrt(x*x+(y+3.)*(y+3.)+0.6384)
    //              -1.0/sqrt((x-3)*(x-3)+y*y+0.6384)
    //              -1.0/sqrt((x+3)*(x+3)+y*y+0.6384);

    // Dipole-like
    // double theta = M_PI / 2.0;
    // double VV0 = 1.0;
    // double v13 = VV0*(r*r+1-3.0*(r*cos(phi)*sin(theta)+cos(theta))*(r*cos(phi)*sin(theta)+cos(theta)))
    //              / pow(r*r+1.0, 2.5);

    // Soft Coulomb (single center)
    // double v15 = -1.0 / sqrt(x * x + y * y + 0.6384);

    // Isotropic Gaussian
    // double v18 = -exp(-r * r / 3.0);

    // Double Gaussian (parameterized)
    // double R = 2.0;
    // double v29 = -exp(-((x-R/2.)*(x-R/2.)+y*y)/3.)
    //              -exp(-((x+R/2.)*(x+R/2.)+y*y)/3.);

    // Anisotropic single Gaussian
    // double v22 = -0.7 * exp(-0.1 * (2*x*x + 0.8*y*y + 2.4*x*y));

    // Square well (active)
    double LL = iAA * 0.01;
    double bb = 1.5;
    double v33 = (std::abs(x) <= LL / 2.0 && std::abs(y) <= LL / 2.0) ? -bb : 0.0;

    return v33;
}


void Potentials::potMatrixElement() {

    const double r_max = (N_grid - 1) * dr;
    const int l_max = (channels - 1) / 2;

    std::vector<double> roots(Nroots, 0.0);
    std::vector<double> weights(Nroots, 0.0);
    std::ifstream fin(FILE_PATH("roots_legendre_" + std::to_string(Nroots) + ".dat"));
    if (!fin.is_open()) {
        std::cerr << "Requested file roots_legendre_" << Nroots << ".dat does not exist!\n";
        exit(1);
    }
    for (int i = 0; i < Nroots; i++) {
        fin >> roots[i] >> weights[i];
    }

    auto func = [=](double vphi, double r, int m) -> dcompx {
        return (cos(m * vphi) + I * sin(m * vphi)) * Potential(r, vphi);
    };

    std::ifstream pot_in("../pot_component_" + std::to_string(int(r_max)) + "_" + std::to_string(channels) + ".dat");
    if (!pot_in.is_open()) {
        cout << "pot_component not cached, calculating ...\n";

        std::vector<std::vector<dcompx>> dipolePot(channels * channels, std::vector<dcompx>(N_grid, 0.0));

        const int local_NTHREADS = NTHREADS;
        const int CHUNK = std::max(1, N_grid / local_NTHREADS);
        omp_set_num_threads(local_NTHREADS);

        #pragma omp parallel for default(shared) schedule(static, CHUNK)
        for (int ir = 1; ir < N_grid; ir++) {
            double r = ir * dr;
            for (int m = 0; m < channels; m++) {
                int loc_m = -l_max + m;
                for (int k = 0; k < channels; k++) {
                    int loc_k = -l_max + k;

                    dcompx sum_phi = 0.0;
                    for (int j = 0; j < Nroots; j++) {
                        double tt = 0.5 * vphi_max * roots[j] + vphi_max / 2.0;
                        sum_phi += 0.5 * vphi_max * func(tt, r, loc_m - loc_k) * weights[j];
                    }
                    dipolePot[m * channels + k][ir] =
                        ((loc_m * loc_m - 0.25) / (2.0 * r * r)) * delta(loc_m, loc_k)
                        + sum_phi / (2.0 * M_PI);
                }
            }
        }
        cout << "Angular integration done.\n";

        std::ofstream pot_out("pot_component_" + std::to_string(int(r_max)) + "_" + std::to_string(channels) + ".dat");
        for (int ir = 0; ir < N_grid; ir++) {
            for (int m = 0; m < channels; m++) {
                for (int k = 0; k < channels; k++) {
                    pot_component[ir](m, k) = dipolePot[m * channels + k][ir];
                    pot_out << real(pot_component[ir](m, k)) << "\t"
                            << imag(pot_component[ir](m, k)) << "\n";
                }
            }
        }
        pot_out.close();
    } else {
        cout << "pot_component cached, reading ...\n";
        double in_re, in_im;
        for (int ir = 0; ir < N_grid; ir++) {
            for (int m = 0; m < channels; m++) {
                for (int k = 0; k < channels; k++) {
                    pot_in >> in_re >> in_im;
                    pot_component[ir](m, k) = in_re + I * in_im;
                }
            }
        }
    }

    cout << "Potential matrix elements done.\n";
}
