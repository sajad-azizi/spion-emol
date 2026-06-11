#include "DipoleMat.hpp"
#include "Angular.hpp"

DipoleMat::DipoleMat(const Wavefunctions& wfs, const Parameters& params)
    : wfs_(wfs), params_(params)
{
}

void DipoleMat::five_point_derivative(const std::vector<dcompx>& f,
                                      std::vector<dcompx>& df) const
{
    const int    N  = static_cast<int>(f.size());
    const double dr = params_.dr;
    df.assign(N, dcompx(0.0, 0.0));
    if (N < 5) {
        for (int i = 1; i < N - 1; ++i)
            df[i] = (f[i + 1] - f[i - 1]) / (2.0 * dr);
        return;
    }
    const double inv12dr = 1.0 / (12.0 * dr);
    df[0] = (-25.0 * f[0] + 48.0 * f[1] - 36.0 * f[2]
              + 16.0 * f[3] -  3.0 * f[4]) * inv12dr;
    df[1] = (-3.0 * f[0] - 10.0 * f[1] + 18.0 * f[2]
              -  6.0 * f[3] +  1.0 * f[4]) * inv12dr;
    for (int i = 2; i < N - 2; ++i) {
        df[i] = (f[i - 2] - 8.0 * f[i - 1]
                 + 8.0 * f[i + 1] - f[i + 2]) * inv12dr;
    }
    df[N - 2] = (-1.0 * f[N - 5] +  6.0 * f[N - 4]
                  - 18.0 * f[N - 3] + 10.0 * f[N - 2]
                  +  3.0 * f[N - 1]) * inv12dr;
    df[N - 1] = ( 3.0 * f[N - 5] - 16.0 * f[N - 4]
                  + 36.0 * f[N - 3] - 48.0 * f[N - 2]
                  + 25.0 * f[N - 1]) * inv12dr;
}

void DipoleMat::real_dipole_q(int q, std::vector<dcompx>& d_real) const
{
    const int    N    = params_.N_grid;
    const int    Nc   = params_.n_channels;
    const int    Lmax = params_.l_max;
    const double dr   = params_.dr;

    d_real.assign(Nc, dcompx(0.0, 0.0));

    // Precompute non-zero G^R(l,m; 1,q; l',m').  Selection: |Δl| = 1
    // and parity (1 + Δl) odd → Δl = ±1 only; the m-rule is the
    // permissive real-Y rule (allow m_R = ±|m'_R ± q| ... 0 too).
    struct Trip { int idx_a; int idx_b; double g; };
    std::vector<Trip> trips;
    trips.reserve(Nc * 4);
    for (int la = 0; la <= Lmax; ++la) {
        for (int ma = -la; ma <= la; ++ma) {
            const int idx_a = ang3d::lm_to_idx(la, ma);
            for (int dl = -1; dl <= 1; dl += 2) {
                const int lb = la + dl;
                if (lb < 0 || lb > Lmax) continue;
                for (int mb = -lb; mb <= lb; ++mb) {
                    const int idx_b = ang3d::lm_to_idx(lb, mb);
                    const double g  = ang3d::gaunt_real(la, ma, 1, q, lb, mb);
                    if (g != 0.0) trips.push_back({idx_a, idx_b, g});
                }
            }
        }
    }

    // Composite Simpson on the radial integrand
    //   I(beta, a, b) = ∫_0^{r_max} dr  conj(chi_cont(r,a,beta)) · r ·
    //                                       chi_bound(r,b)
    // accumulated as d_real[beta] = sum_{a,b} G^R(a; q; b) · I(beta,a,b).
    for (int beta = 0; beta < Nc; ++beta) {
        dcompx d_sum(0.0, 0.0);
        for (const auto& t : trips) {
            // Composite Simpson 1/3 on [0, r_max].
            // Endpoints contribute 1, even interior ×2, odd interior ×4.
            dcompx I = dcompx(0.0, 0.0);
            for (int ir = 1; ir < N - 1; ++ir) {
                const double r = ir * dr;
                const dcompx integrand =
                    std::conj(wfs_.scattering_eigenfunc[ir](t.idx_a, beta))
                    * r * wfs_.eigfunc[ir](t.idx_b);
                const double w = (ir & 1) ? 4.0 : 2.0;
                I += w * integrand;
            }
            // r=0 contributes 0 (factor r). r=N-1 gets weight 1.
            const double r_end = (N - 1) * dr;
            const dcompx end =
                std::conj(wfs_.scattering_eigenfunc[N - 1](t.idx_a, beta))
                * r_end * wfs_.eigfunc[N - 1](t.idx_b);
            I += end;
            I *= dr / 3.0;
            d_sum += t.g * I;
        }
        d_real[beta] = d_sum;
    }
}

// ---- Velocity-form integrand --------------------------------------------
// d^V_q[β] = Σ_{(l_c m_c),(l_b m_b)} G^R(l_c m_c; 1 q; l_b m_b)
//            · η_{l_c, l_b}
//            · ∫_0^∞ conj(chi_c)(r) · D_{l_b}^(±) chi_b(r) dr
//
// where  D_{l_b}^(+) chi_b = dchi_b/dr − (l_b + 1)/r · chi_b   for l_c = l_b + 1
//        D_{l_b}^(-) chi_b = dchi_b/dr +     l_b/r · chi_b     for l_c = l_b − 1
//
// and η is the Wigner reduced matrix element factor (from comparing the
// length and velocity matrix elements via the Wigner-Eckart theorem):
//        η_+ = √((l_b + 1) / (2 l_b + 3))   for raising
//        η_- = √(   l_b   / (2 l_b − 1))   for lowering
// (multiplied by an overall √(4π/3) so the same G^R-times-radial product
//  matches the length-form Cartesian normalization).
//
// Length-velocity identity (one-electron, local V):
//        ω · d^L_q = − d^V_q
// is checked numerically by test_dipole_gauge.
void DipoleMat::real_velocity_q(int q, std::vector<dcompx>& d_real) const
{
    const int    N    = params_.N_grid;
    const int    Nc   = params_.n_channels;
    const int    Lmax = params_.l_max;
    const double dr   = params_.dr;

    d_real.assign(Nc, dcompx(0.0, 0.0));

    // Precompute dchi_b/dr for the bound state, channel by channel.
    std::vector<std::vector<dcompx>> dchi_b(Nc, std::vector<dcompx>(N));
    for (int idx = 0; idx < Nc; ++idx) {
        std::vector<dcompx> chi_b(N);
        for (int ir = 0; ir < N; ++ir) chi_b[ir] = wfs_.eigfunc[ir](idx);
        five_point_derivative(chi_b, dchi_b[idx]);
    }

    struct VTrip {
        int    idx_c;     // continuum channel (l_c, m_c)
        int    idx_b;     // bound channel     (l_b, m_b)
        int    branch;    // +1 = l_c = l_b + 1, −1 = l_c = l_b − 1
        int    l_b;       // l of bound channel
        double coef;      // η · G^R · √(4π/3)
    };
    std::vector<VTrip> trips;
    trips.reserve(Nc * 4);
    for (int la = 0; la <= Lmax; ++la) {
        for (int ma = -la; ma <= la; ++ma) {
            const int idx_a = ang3d::lm_to_idx(la, ma);
            for (int dl = -1; dl <= 1; dl += 2) {
                const int lb = la + dl;
                if (lb < 0 || lb > Lmax) continue;
                // No extra Wigner factor: the angular Wigner-Eckart
                // structure of ∇_q (rank-1 spherical tensor) in real-Y
                // basis is captured fully by G^R, identical to that of
                // r̂_q.  Only the radial integral differs.  Verified
                // by the length-velocity gauge identity:
                //   d^V = -ω · d^L   (length-velocity equivalence).
                for (int mb = -lb; mb <= lb; ++mb) {
                    const int idx_b = ang3d::lm_to_idx(lb, mb);
                    const double g = ang3d::gaunt_real(la, ma, 1, q, lb, mb);
                    if (g == 0.0) continue;
                    trips.push_back(
                        {idx_a, idx_b, dl, lb, g});
                }
            }
        }
    }

    for (int beta = 0; beta < Nc; ++beta) {
        dcompx acc(0.0, 0.0);
        for (const auto& t : trips) {
            // Radial integrand:
            //   D chi_b(r) = dchi_b/dr ∓ (l_b+1 or -l_b)/r · chi_b
            //   integrand  = conj(chi_c)(r) · D chi_b(r)
            // Composite Simpson 1/3 over [0, r_max].
            dcompx I(0.0, 0.0);
            const auto& chi_c = wfs_.scattering_eigenfunc;  // [ir](idx_c, beta)
            const auto& dchi  = dchi_b[t.idx_b];
            for (int ir = 1; ir < N - 1; ++ir) {
                const double r  = ir * dr;
                const dcompx u_b   = wfs_.eigfunc[ir](t.idx_b);
                const dcompx du_b  = dchi[ir];
                // Match parent project (DipoleMatrixElement::velocity_coef):
                //   w(r) = dχ_ν/dr + c(l_μ, l_ν)/r · χ_ν
                //   c = -(l_ν + 1) when l_μ = l_ν + 1 (raising)
                //   c = +l_ν       when l_μ = l_ν - 1 (lowering)
                // Here la = l_μ (continuum), lb = l_ν (bound).  In my
                // loop, lb = la + dl, so branch = dl = lb − la = l_ν − l_μ,
                // which has OPPOSITE sign from parent's (l_μ − l_ν):
                //   branch = -1  ⟺  l_μ = l_ν + 1  ⟺  RAISING
                //   branch = +1  ⟺  l_μ = l_ν - 1  ⟺  LOWERING
                const dcompx D_u_b = (t.branch == -1)
                    ? (du_b - dcompx((t.l_b + 1.0) / r, 0.0) * u_b)   // raising
                    : (du_b + dcompx( t.l_b        / r, 0.0) * u_b);  // lowering
                const dcompx integrand =
                    std::conj(chi_c[ir](t.idx_c, beta)) * D_u_b;
                const double w = (ir & 1) ? 4.0 : 2.0;
                I += w * integrand;
            }
            // r=0 term: D chi_b at r=0 is finite for chi_b ~ r^{l+1};
            // weight is 1 in Simpson but with chi_c(0)=0 the product
            // is exactly zero -> skip.
            const double r_end = (N - 1) * dr;
            const dcompx u_end  = wfs_.eigfunc[N - 1](t.idx_b);
            const dcompx du_end = dchi[N - 1];
            const dcompx D_end  = (t.branch == -1)
                ? (du_end - dcompx((t.l_b + 1.0) / r_end, 0.0) * u_end)
                : (du_end + dcompx( t.l_b        / r_end, 0.0) * u_end);
            I += std::conj(chi_c[N - 1](t.idx_c, beta)) * D_end;
            I *= dr / 3.0;
            acc += t.coef * I;
        }
        d_real[beta] = acc;
    }
}

std::vector<dcompx>
DipoleMat::compute_velocity(int q, const Eigen::MatrixXcd& A,
                                   const Eigen::MatrixXcd& B,
                                   double E) const
{
    const int Nc = params_.n_channels;
    (void)E;
    std::vector<dcompx> d_real;
    real_velocity_q(q, d_real);

    {
        std::ofstream o("dipole_velocity_real_q" + std::to_string(q) + ".dat");
        o << std::fixed << std::setprecision(12);
        for (int b = 0; b < Nc; ++b) {
            int l, m; ang3d::idx_to_lm(b, l, m);
            o << l << "\t" << m << "\t"
              << d_real[b].real() << "\t" << d_real[b].imag() << "\n";
        }
    }

    Eigen::MatrixXcd AiBmt =
        ((A - I_unit * B).inverse()).transpose().conjugate();
    std::vector<dcompx> d_in(Nc, dcompx(0.0, 0.0));
    for (int m = 0; m < Nc; ++m) {
        dcompx s(0.0, 0.0);
        for (int b = 0; b < Nc; ++b) s += AiBmt(m, b) * d_real[b];
        d_in[m] = s;
    }
    return d_in;
}

std::vector<dcompx>
DipoleMat::compute(int q, const Eigen::MatrixXcd& A,
                          const Eigen::MatrixXcd& B,
                          double E) const
{
    const int Nc = params_.n_channels;
    (void)E;  // E kept for API symmetry; not used in length form.

    std::vector<dcompx> d_real;
    real_dipole_q(q, d_real);

    // Dump real-axis result for diagnostics.
    {
        std::ofstream o("dipole_real_q" + std::to_string(q) + ".dat");
        o << std::fixed << std::setprecision(12);
        for (int b = 0; b < Nc; ++b) {
            int l, m; ang3d::idx_to_lm(b, l, m);
            o << l << "\t" << m << "\t"
              << d_real[b].real() << "\t" << d_real[b].imag() << "\n";
        }
    }

    // Transform to ingoing-BC basis.  Same formula as polar_2d:
    //   d^{in} = ((A - iB)^{-1})^T*  ·  d^{real}.
    Eigen::MatrixXcd AiBmt =
        ((A - I_unit * B).inverse()).transpose().conjugate();

    std::vector<dcompx> d_in(Nc, dcompx(0.0, 0.0));
    for (int m = 0; m < Nc; ++m) {
        dcompx s(0.0, 0.0);
        for (int b = 0; b < Nc; ++b) {
            s += AiBmt(m, b) * d_real[b];
        }
        d_in[m] = s;
    }

    {
        std::ofstream o("dipole_ingoingBC_q" + std::to_string(q) + ".dat");
        o << std::fixed << std::setprecision(12);
        for (int b = 0; b < Nc; ++b) {
            int l, m; ang3d::idx_to_lm(b, l, m);
            o << l << "\t" << m << "\t"
              << d_in[b].real()    << "\t" << d_in[b].imag()
              << "\t" << std::norm(d_in[b])
              << "\t" << std::arg(d_in[b]) << "\n";
        }
    }

    return d_in;
}
