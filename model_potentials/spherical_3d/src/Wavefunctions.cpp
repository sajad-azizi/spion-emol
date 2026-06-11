#include "Wavefunctions.hpp"
#include "Angular.hpp"

Wavefunctions::Wavefunctions(Equations& eqs, const Parameters& params)
    : eqs_(eqs), params_(params)
{
    const int N  = params_.N_grid;
    const int Nc = params_.n_channels;
    eigfunc.assign(N, Eigen::VectorXcd::Zero(Nc));
    scattering_eigenfunc.assign(N, Eigen::MatrixXcd::Zero(Nc, Nc));
}

void Wavefunctions::calculate_eigenfunction(double E, int i_match)
{
    const int N  = params_.N_grid;
    const int Nc = params_.n_channels;

    Eigen::MatrixXcd Rm, Rmp1;
    eqs_.propagateForward (E, i_match, Rm,   /*save*/ true);
    eqs_.propagateBackward(E, i_match, Rmp1, /*save*/ true);

    // Glue: take the eigenvector of (R_back^{-1} - R_fwd) corresponding
    // to the smallest |eigval| (zero eigenvalue at the exact eigen-E).
    Eigen::MatrixXcd diff = Rmp1.inverse() - Rm;
    es_.compute(diff);
    Eigen::VectorXd absev(Nc);
    for (int m = 0; m < Nc; ++m)
        absev(m) = std::abs(es_.eigenvalues()(m));
    Eigen::Index jm;
    absev.minCoeff(&jm);
    Eigen::VectorXcd fn_match = es_.eigenvectors().col(jm);

    auto& Rinv = eqs_.Rinv_vector();
    auto& Winv = eqs_.Winv_vector();

    Eigen::VectorXcd fn = fn_match;
    for (int k = i_match - 1; k > 0; --k) {
        fn = Rinv[k] * fn;
        eigfunc[k] = Winv[k] * fn;
        // The next iteration uses fn (which is the "psi" form, i.e. W·chi),
        // so fn for step k-1 starts from this fn.  Match polar_2d.
    }
    Eigen::VectorXcd fn_fwd = fn_match;
    for (int k = i_match + 1; k < N - 1; ++k) {
        fn_fwd = Rinv[k] * fn_fwd;
        eigfunc[k] = Winv[k] * fn_fwd;
    }
    eigfunc[i_match] = Winv[i_match] * fn_match;
    // Boundary points stay zero.
}

void Wavefunctions::Normalization(std::vector<Eigen::VectorXcd>& f)
{
    const int N  = params_.N_grid;
    const int Nc = params_.n_channels;
    const double dr = params_.dr;
    double sum = 0.0;
    for (int l = 0; l < Nc; ++l) {
        for (int i = 0; i < N; ++i) {
            const double a = std::abs(f[i](l));
            sum += a * a * dr;
        }
    }
    if (sum > 0.0) {
        const double norm = std::sqrt(sum);
        for (int l = 0; l < Nc; ++l)
            for (int i = 0; i < N; ++i)
                f[i](l) /= norm;
    }
}

void Wavefunctions::calculate_eigenfunction_continuum(double E)
{
    const int N  = params_.N_grid;
    const int Nc = params_.n_channels;

    Eigen::MatrixXcd Rm;
    eqs_.propagateForward(E, N - 1, Rm, /*save*/ true);
    es_.compute(Rm);
    Eigen::VectorXd absev(Nc);
    for (int m = 0; m < Nc; ++m)
        absev(m) = std::abs(es_.eigenvalues()(m));
    Eigen::Index jm;
    absev.minCoeff(&jm);
    Eigen::VectorXcd fn_match = es_.eigenvectors().col(jm);

    auto& Rinv = eqs_.Rinv_vector();
    auto& Winv = eqs_.Winv_vector();

    Eigen::VectorXcd fn = fn_match;
    for (int k = N - 2; k > 0; --k) {
        fn = Rinv[k] * fn;
        eigfunc[k] = Winv[k] * fn;
    }
    eigfunc[N - 1] = Winv[N - 1] * fn_match;
}

void Wavefunctions::calculate_channel_wavefunction(double E)
{
    const int N  = params_.N_grid;
    const int Nc = params_.n_channels;

    Eigen::MatrixXcd Rm;
    eqs_.propagateForward(E, N - 1, Rm, /*save*/ true);

    auto& Rinv = eqs_.Rinv_vector();
    auto& Winv = eqs_.Winv_vector();

    std::vector<Eigen::VectorXcd> col(N, Eigen::VectorXcd::Zero(Nc));

    // Match polar_2d trick: "fn_match for column j = j-th row of
    // Winv[N-1]^{-1}".  Equivalent to seeding chi(r_N) = e_j (a unit
    // vector in channel space).
    Eigen::MatrixXcd Winv_last_inv = Winv[N - 1].inverse();

    for (int j = 0; j < Nc; ++j) {
        Eigen::VectorXcd fn_match = Winv_last_inv.row(j).transpose();
        Eigen::VectorXcd fn = fn_match;
        for (int k = N - 2; k > 0; --k) {
            fn = Rinv[k] * fn;
            col[k] = Winv[k] * fn;
        }
        for (int ir = 0; ir < N; ++ir) {
            for (int m = 0; m < Nc; ++m) {
                scattering_eigenfunc[ir](m, j) = col[ir](m);
            }
        }
    }
}

void Wavefunctions::calculate_A_B_matrices(Eigen::MatrixXcd& A,
                                           Eigen::MatrixXcd& B,
                                           double E)
{
    const int N  = params_.N_grid;
    const int Nc = params_.n_channels;
    const double k = std::sqrt(2.0 * E);

    A = Eigen::MatrixXcd::Zero(Nc, Nc);
    B = Eigen::MatrixXcd::Zero(Nc, Nc);

    std::vector<double> re(N), im(N);
    for (int j = 0; j < Nc; ++j) {
        for (int idx = 0; idx < Nc; ++idx) {
            int l, m;
            ang3d::idx_to_lm(idx, l, m);

            for (int i = 0; i < N; ++i) {
                re[i] = scattering_eigenfunc[i](idx, j).real();
                im[i] = scattering_eigenfunc[i](idx, j).imag();
            }
            auto [a_re, b_re] = getCoefficients(re, k, l);
            auto [a_im, b_im] = getCoefficients(im, k, l);

            A(idx, j) = dcompx(a_re, a_im);
            B(idx, j) = dcompx(b_re, b_im);
        }
    }
}

std::pair<double, double>
Wavefunctions::getCoefficients(const std::vector<double>& data,
                               double k, int l) const
{
    const int    N  = params_.N_grid;
    const double dr = params_.dr;

    // Choose a fitting window starting just past the last centrifugal-
    // barrier zero of j_l(kr) inside the box.  Use spherical_bessel_zero
    // for a robust starting index.
    int n_zero = 1;
    while (ang3d::spherical_bessel_zero(l, n_zero) <= k * (N - 1) * dr) {
        ++n_zero;
    }

    int    i_save  = N - 6;
    int    i_count = 2;
    while (true) {
        const double zero_pos =
            ang3d::spherical_bessel_zero(l, std::max(1, n_zero - i_count));
        i_save = static_cast<int>(zero_pos / k / dr);
        if (i_save > 0 && i_save < N - 4) break;
        ++i_count;
        if (i_count > n_zero + 50) {
            // Fallback: just use the outer half of the grid.
            i_save = std::max(2, N / 2);
            break;
        }
    }
    const int wf_size = (N - 5) - i_save;
    if (wf_size <= 0) {
        cerr_no_zero:
        std::cerr << "Wavefunctions::getCoefficients: empty fitting window\n";
        return {0.0, 0.0};
    }

    double sumA = 0.0, sumB = 0.0, sumC = 0.0, sumD = 0.0, sumE = 0.0;
    for (int i = 0; i < wf_size; ++i) {
        const double r  = (i_save + i) * dr;
        const double Sv = ang3d::riccati_S(l, k * r);
        const double Cv = ang3d::riccati_C(l, k * r);
        const double w  = data[i_save + i];
        sumA += w  * Sv;
        sumB += Sv * Sv;
        sumC += Cv * Sv;
        sumD += Cv * Cv;
        sumE += w  * Cv;
    }
    const double det = sumB * sumD - sumC * sumC;
    if (std::fabs(det) < 1e-30) {
        std::cerr << "Wavefunctions::getCoefficients: singular fit "
                     "(window degenerate)\n";
        goto cerr_no_zero;
    }
    const double a_ = ( sumA * sumD - sumE * sumC) / det;
    const double b_ = (-sumA * sumC + sumE * sumB) / det;
    return {a_, b_};
}
