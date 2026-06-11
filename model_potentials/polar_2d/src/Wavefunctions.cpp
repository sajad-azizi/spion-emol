#include "Wavefunctions.hpp"

Wavefunctions::Wavefunctions(Equations *equations, Parameters *parameters)
    : equations(equations), parameters(parameters)
{
    N_grid   = parameters->N_grid;
    channels = parameters->channels;
    dr       = parameters->dr;
    l_max    = (channels - 1) / 2;
    NTHREADS = parameters->NTHREADS;
    divide   = parameters->divide;

    eigfunc.resize(N_grid);
    for (int k = 0; k < N_grid; ++k)
        eigfunc[k] = Eigen::VectorXcd::Zero(channels);

    scattering_eigenfunc.resize(N_grid);
    for (int k = 0; k < N_grid; ++k)
        scattering_eigenfunc[k] = Eigen::MatrixXcd::Zero(channels, channels);
}


void Wavefunctions::calculate_final_continumm_states(
    Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, const double energy)
{
    Eigen::MatrixXcd AiB_inv = (A - I * B).inverse();

    std::vector<Eigen::MatrixXcd> scattering_eigenfunc_energyNorm(N_grid);
    for (int k = 0; k < N_grid; ++k)
        scattering_eigenfunc_energyNorm[k] = scattering_eigenfunc[k] * AiB_inv;

    std::ofstream mout;
    for (int m = 0; m < channels; m++) {
        for (int beta = 0; beta < channels; beta++) {
            mout.open("final_psi_" + std::to_string(m) + "_" + std::to_string(beta) + ".dat");
            for (int k = 0; k < N_grid / divide; k += 10) {
                mout << real(scattering_eigenfunc_energyNorm[k](m, beta)) << "\t"
                     << imag(scattering_eigenfunc_energyNorm[k](m, beta)) << "\n";
            }
            mout.close();
        }
    }
}


void Wavefunctions::calculate_eigenfunction(double E, int i_match) {

    Eigen::MatrixXcd Rm, Rmp1;
    equations->propagateForward(E, i_match, Rm, true);
    equations->propagateBackward(E, i_match, Rmp1, true);

    Eigen::MatrixXcd diff = Rmp1.inverse() - Rm;

    // Find eigenvector corresponding to smallest eigenvalue
    es.compute(diff);
    Eigen::VectorXd Veig(channels);
    for (int m = 0; m < channels; m++)
        Veig(m) = std::abs(es.eigenvalues().real()(m));

    Eigen::VectorXd::Index jm;
    Veig.minCoeff(&jm);

    Eigen::VectorXcd fn_matching_point = es.eigenvectors().col(jm);
    Eigen::VectorXcd fn, fn_next = fn_matching_point, fn_prev = fn_matching_point;
    Eigen::VectorXcd psin;

    // Backward sweep from matching point
    for (int k = i_match - 1; k > 0; --k, fn_next = fn) {
        fn = equations->Rinv_vector[k] * fn_next;
        eigfunc[k] = equations->Winv_vector[k] * fn;
    }

    // Forward sweep from matching point
    for (int k = i_match + 1; k < N_grid - 1; ++k, fn_prev = fn) {
        fn = equations->Rinv_vector[k] * fn_prev;
        eigfunc[k] = equations->Winv_vector[k] * fn;
    }

    eigfunc[i_match] = equations->Winv_vector[i_match] * fn_matching_point;
    cout << "eigfunc[" << i_match << "]:\n" << eigfunc[i_match] << endl;
}


void Wavefunctions::calculate_eigenfunction_continuum(double E) {

    Eigen::MatrixXcd Rm;
    equations->propagateForward(E, N_grid - 1, Rm, true);

    es.compute(Rm);
    Eigen::VectorXd Veig(channels);
    for (int m = 0; m < channels; m++)
        Veig(m) = std::abs(es.eigenvalues().real()(m));

    Eigen::VectorXd::Index jm;
    Veig.minCoeff(&jm);

    Eigen::VectorXcd fn_matching_point = es.eigenvectors().col(jm);
    Eigen::VectorXcd fn, fn_next = fn_matching_point;

    for (int k = N_grid - 2; k > 0; --k, fn_next = fn) {
        fn = equations->Rinv_vector[k] * fn_next;
        eigfunc[k] = equations->Winv_vector[k] * fn;
    }

    eigfunc[N_grid - 1] = equations->Winv_vector[N_grid - 1] * fn_matching_point;
    cout << "eigfunc[" << N_grid - 1 << "]:\n" << eigfunc[N_grid - 1] << endl;
}


void Wavefunctions::Normalization(std::vector<Eigen::VectorXcd> &eigfunc) {
    double sum = 0.0;
    for (int l = 0; l < channels; l++) {
        for (int i = 0; i < N_grid; i++) {
            double a = std::abs(eigfunc[i](l));
            sum += a * a * dr;
        }
    }
    if (sum > 0.0) {
        double norm = sqrt(sum);
        for (int l = 0; l < channels; l++)
            for (int i = 0; i < N_grid; i++)
                eigfunc[i](l) /= norm;
    }
}


void Wavefunctions::calculate_channel_wavefunction(const double Energy) {

    Eigen::MatrixXcd Rm;
    equations->propagateForward(Energy, N_grid - 1, Rm, true);

    std::vector<Eigen::VectorXcd> eigfunc_loc(N_grid);
    for (int k = 0; k < N_grid; ++k)
        eigfunc_loc[k] = Eigen::VectorXcd::Zero(channels);

    Eigen::VectorXcd fn_matching_point(channels);
    Eigen::VectorXcd fn, fn_next, psin;

    for (int j = 0; j < channels; j++) {
        // Far from origin the potential vanishes; each row is an independent open channel
        fn_matching_point = equations->Winv_vector[N_grid - 1].inverse().row(j);
        fn_next = fn_matching_point;

        for (int k = N_grid - 2; k > 0; --k, fn_next = fn) {
            fn = equations->Rinv_vector[k] * fn_next;
            eigfunc_loc[k] = equations->Winv_vector[k] * fn;
        }

        // Store each channel solution as a column
        for (int ir = 0; ir < N_grid; ir++)
            for (int m = 0; m < channels; m++)
                scattering_eigenfunc[ir](m, j) = eigfunc_loc[ir](m);
    }
}


void Wavefunctions::calculate_A_B_matrices(
    Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, const double energy)
{
    const double k = sqrt(2.0 * energy);

    std::vector<double> psi2d_real(N_grid, 0.0);
    std::vector<double> psi2d_imag(N_grid, 0.0);

    for (int j = 0; j < channels; j++) {
        for (int m = 0; m < channels; m++) {
            int loc_m = -l_max + m;

            for (int i = 0; i < N_grid; i++) {
                psi2d_real[i] = real(scattering_eigenfunc[i](m, j));
                psi2d_imag[i] = imag(scattering_eigenfunc[i](m, j));
            }

            auto [a, b] = getCoefficents(psi2d_real, k, loc_m);
            auto [c, d] = getCoefficents(psi2d_imag, k, loc_m);

            A(m, j) = a + I * c;
            B(m, j) = b + I * d;
        }
    }
}


std::pair<double, double> Wavefunctions::getCoefficents(
    const std::vector<double> &data, double k, int loc_m)
{
    // Find fitting window start using Bessel zeros
    int oo = 1;
    while (gsl_sf_bessel_zero_Jnu(std::abs(loc_m), oo) <= k * (N_grid - 1) * dr)
        oo++;

    int i_save = N_grid - 6;
    int i_count = 2;
    while (true) {
        i_save = int(gsl_sf_bessel_zero_Jnu(std::abs(loc_m), oo - i_count) / dr);
        if (i_save < N_grid - 4)
            break;
        i_count++;
    }

    const int wf_size = (N_grid - 5) - i_save;
    if (wf_size <= 0) {
        cout << "Achtung!! fitting window size is zero!\n";
        exit(1);
    }

    // Single OMP loop computing all 5 sums at once
    double sumA = 0.0, sumB = 0.0, sumC = 0.0, sumD = 0.0, sumE = 0.0;

    #pragma omp parallel for default(shared) reduction(+:sumA,sumB,sumC,sumD,sumE)
    for (int i = 0; i < wf_size; i++) {
        double r_loc = (i_save + i) * dr;
        double jval  = riccati_Jn(loc_m, k * r_loc);
        double yval  = riccati_Yn(loc_m, k * r_loc);
        double wf    = data[i_save + i];

        sumA += wf   * jval;
        sumB += jval * jval;
        sumC += yval * jval;
        sumD += yval * yval;
        sumE += wf   * yval;
    }

    double determ = sumB * sumD - sumC * sumC;
    double a_ = (sumA * sumD - sumE * sumC) / determ;
    double b_ = (-sumA * sumC + sumE * sumB) / determ;

    return {a_, b_};
}


double Wavefunctions::riccati_Jn(int n, double x) {
    const double prefix = sqrt(M_PI * x / 2.0);
    if (n < 0) return std::pow(-1.0, std::abs(n)) * prefix * std::cyl_bessel_j(std::abs(n), x);
    return prefix * std::cyl_bessel_j(n, x);
}

double Wavefunctions::riccati_Yn(int n, double x) {
    const double prefix = sqrt(M_PI * x / 2.0);
    if (n < 0) return std::pow(-1.0, std::abs(n)) * prefix * std::cyl_neumann(std::abs(n), x);
    return prefix * std::cyl_neumann(n, x);
}

double Wavefunctions::spherical_jn(int n, double x) {
    if (n < 0) return std::pow(-1.0, std::abs(n)) * std::sph_bessel(std::abs(n), x);
    return std::sph_bessel(n, x);
}

double Wavefunctions::spherical_yn(int n, double x) {
    if (n < 0) return std::pow(-1.0, std::abs(n)) * std::sph_neumann(std::abs(n), x);
    return std::sph_neumann(n, x);
}
