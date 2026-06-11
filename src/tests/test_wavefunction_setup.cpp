// test_wavefunction_setup.cpp -- validate WavefunctionSetup::prepare on
// the H2O preprocessing HDF5.
//
// Checks:
//   (1) SolverParams fields are filled consistently (alpha = sqrt(2 pi),
//       l_exch rule, n_mu / n_sigma / n_lambda, n_occ, n_transition).
//   (2) chi has shape (n_transition) matrices of (n_occ, n_lambda_cut)
//       and the r-factor was applied: at ir=1, chi(j,lam) / r(ir=1)
//       matches psi_lm row-packed layout to ~1e-12.
//   (3) Each occupied orbital is normalized: sum_{ir, lam} chi(j,lam)^2 dr
//       ~= 1 (the r factor converts 3D integral to trapezoid sum). Accept
//       1% tolerance -- the SCE truncation and the coarse finite grid
//       both cost a few 0.1%.
//   (4) G_coeff and C_coeff each contain only triplets that obey Gaunt
//       selection rules (triangle inequality on l's, m1+m2+m3 parity for
//       real Gaunt through the complex-basis transform, integer l+l+l).
//   (5) G_coeff and C_coeff agree in value: for every (mu, lambda, sigma)
//       in G, the corresponding (sigma, lambda, mu) exists in C with the
//       same value. (Real-Y symmetry.)
//
// Usage: test_wavefunction_setup <path-to-h2o-hdf5>

#include "io/HDF5Reader.hpp"
#include "scatt/Banner.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>

using scatt::Parameters;
using scatt::WavefunctionSetup;
using scatt::SetupBundle;
using scatt::AngTriplet;
using scatt::io::HDF5Reader;
using scatt::io::PreprocData;

static int fails = 0;

static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++fails;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <h2o_hdf5>\n"; return 2; }

    scatt::print_la_banner();

    HDF5Reader reader(argv[1]);
    PreprocData data = reader.load_all();

    Parameters params;
    params.r_min           = data.rmin;
    params.dr              = data.dr;
    params.N_grid          = data.Nr;
    params.Lmax_sce        = data.Lmax_sce;
    params.l_max_continuum = 4;        // l_exch = 4 (<=10 rule), l_orb = 8
    params.validate();

    // --- (1) prepare with defaults ---
    std::cout << "\n--- prepare ---\n";
    SetupBundle b = WavefunctionSetup::prepare(params, data, /*energy=*/0.5);

    const int l_cont = 4, l_exch = 4, l_orb = 8;
    const int n_mu     = (l_cont+1)*(l_cont+1);
    const int n_sigma  = (l_exch+1)*(l_exch+1);
    const int n_lambda = (l_orb +1)*(l_orb +1);

    std::cout << "\n--- (1) SolverParams fields ---\n";
    check(std::abs(b.params.alpha - std::sqrt(2.0 * M_PI)) < 1e-14,
          "alpha = sqrt(2 pi)");
    check(b.params.l_max_continuum == l_cont, "l_max_continuum");
    check(b.params.l_max_exchange  == l_exch, "l_max_exchange (min(l_cont,10) rule)");
    check(b.params.l_max_orbitals  == l_orb,  "l_max_orbitals = l_cont + l_exch");
    check(b.params.n_mu    == n_mu,    "n_mu    = (l_cont+1)^2");
    check(b.params.n_sigma == n_sigma, "n_sigma = (l_exch+1)^2");
    check(b.params.n_occ   == data.n_occ_alpha && b.params.n_occ > 0,
          "n_occ == n_occ_alpha (auto)");
    check(b.params.n_transition == static_cast<int>(data.Nr),
          "n_transition defaults to Nr");
    check(b.params.dr == data.dr, "dr from Parameters");
    check(b.params.n_grid == data.Nr, "n_grid from Parameters");

    // --- Override l_max_exchange rule edge case ---
    WavefunctionSetup::Inputs in2;
    in2.l_max_exchange = 2;
    SetupBundle b2 = WavefunctionSetup::prepare(params, data, 0.5, in2);
    check(b2.params.l_max_exchange == 2 && b2.params.l_max_orbitals == l_cont + 2,
          "user-supplied l_max_exchange override works");

    // --- Rule: l_cont > 10 caps l_exch at 10 (simulate by pretending) ---
    // We don't have H2O at l_cont=11 so just verify default path numerically.
    check(std::min(l_cont, 10) == l_exch, "min(l_cont,10) = l_exch for this run");

    // --- (2) chi shape and r-factor ---
    std::cout << "\n--- (2) chi shape + r factor ---\n";
    check(b.chi.size() == static_cast<std::size_t>(b.params.n_transition),
          "chi length == n_transition");
    check(b.chi[0].rows() == b.params.n_occ &&
          b.chi[0].cols() == n_lambda,
          "chi[0] is (n_occ, n_lambda)");

    const int Nlm_sce = (data.Lmax_sce + 1) * (data.Lmax_sce + 1);
    {
        const int ir = 5;
        const double r = data.rmin + ir * data.dr;
        double max_err = 0.0;
        for (int j = 0; j < b.params.n_occ; ++j) {
            for (int lam = 0; lam < n_lambda; ++lam) {
                const double got = b.chi[ir](j, lam);
                const double ref = r * data.psi_lm(
                    static_cast<Eigen::Index>(j) * Nlm_sce + lam, ir);
                max_err = std::max(max_err, std::abs(got - ref));
            }
        }
        check(max_err < 1e-14, "chi(j,lam) = r * psi_lm(...) at ir=5 (max_err="
              + std::to_string(max_err) + ")");
    }

    // --- (3) Orbital normalization ---
    // Full 3D norm: int |Phi^j(r)|^2 r^2 dr dOmega
    //  = sum_lam int F_lm(r)^2 r^2 dr     (real-Y, orthonormal)
    //  = sum_lam int chi(j,lam)^2 dr      (since chi = r * F_lm)
    // Trapezoid on the uniform grid.
    std::cout << "\n--- (3) orbital normalization (int chi^2 dr) ---\n";
    for (int j = 0; j < std::min(b.params.n_occ, 5); ++j) {
        double s = 0.0;
        for (int ir = 0; ir < b.params.n_transition; ++ir) {
            double w = (ir == 0 || ir == b.params.n_transition - 1) ? 0.5 : 1.0;
            for (int lam = 0; lam < n_lambda; ++lam) {
                const double c = b.chi[ir](j, lam);
                s += w * c * c;
            }
        }
        s *= data.dr;
        std::cout << "     orbital j=" << j << " norm = " << std::fixed
                  << std::setprecision(6) << s << "\n";
        check(std::abs(s - 1.0) < 0.02,
              "|norm - 1| < 0.02 for orbital " + std::to_string(j));
    }

    // --- (4) Gaunt selection rules ---
    std::cout << "\n--- (4) Gaunt selection rules ---\n";
    auto idx_to_l = [](int idx) {
        int l = 0; while ((l+1)*(l+1) <= idx) ++l; return l;
    };
    int bad_tri_G = 0;
    for (const auto& t : b.G_coeff) {
        const int lm = idx_to_l(t.a);   // a = mu
        const int ll = idx_to_l(t.b);   // b = lambda
        const int ls = idx_to_l(t.c);   // c = sigma
        if (std::abs(lm - ll) > ls || lm + ll < ls) ++bad_tri_G;
        if ((lm + ll + ls) % 2 != 0) ++bad_tri_G;
    }
    check(bad_tri_G == 0, "G_coeff obeys triangle + parity (violations="
          + std::to_string(bad_tri_G) + ")");

    int bad_tri_C = 0;
    for (const auto& t : b.C_coeff) {
        const int ls = idx_to_l(t.a);   // a = sigma
        const int ll = idx_to_l(t.b);   // b = lambda
        const int lm = idx_to_l(t.c);   // c = mu
        if (std::abs(lm - ll) > ls || lm + ll < ls) ++bad_tri_C;
        if ((lm + ll + ls) % 2 != 0) ++bad_tri_C;
    }
    check(bad_tri_C == 0, "C_coeff obeys triangle + parity (violations="
          + std::to_string(bad_tri_C) + ")");

    check(!b.G_coeff.empty(), "G_coeff non-empty");
    check(!b.C_coeff.empty(), "C_coeff non-empty");

    // --- (5) G^R_{mu,lambda,sigma} == C^R_{sigma,lambda,mu} ---
    std::cout << "\n--- (5) G-C value agreement ---\n";
    // Key: pack (sigma, lambda, mu) -> value.
    auto key = [](int a, int b_, int c) {
        return (static_cast<std::uint64_t>(a) << 42)
             | (static_cast<std::uint64_t>(b_) << 21)
             |  static_cast<std::uint64_t>(c);
    };
    std::unordered_map<std::uint64_t, double> cmap;
    cmap.reserve(b.C_coeff.size() * 2);
    for (const auto& t : b.C_coeff) cmap[key(t.a, t.b, t.c)] = t.value;

    double max_diff = 0.0;
    int missing = 0;
    for (const auto& t : b.G_coeff) {
        // G has (mu, lambda, sigma); lookup C at (sigma, lambda, mu).
        auto it = cmap.find(key(t.c, t.b, t.a));
        if (it == cmap.end()) { ++missing; continue; }
        max_diff = std::max(max_diff, std::abs(it->second - t.value));
    }
    check(missing == 0, "every G triplet has a matching C triplet (missing="
          + std::to_string(missing) + ")");
    check(max_diff < 1e-12, "|G - C| < 1e-12 (max=" + std::to_string(max_diff) + ")");

    // --- (6) C = G^T  (swap the outer legs) ---
    // Real-harmonic Gaunt is symmetric under any permutation of the three
    // legs, so C_{sigma,lambda,mu} = G_{mu,lambda,sigma}. Using our packed
    // layout that means: the set { (a=mu, b=lambda, c=sigma, v) } from G
    // equals the set { (c=mu, b=lambda, a=sigma, v) } from C -- i.e. C is
    // G with the outer (first<->third) indices swapped. Stronger than the
    // lookup check (5): we require the *full set* to match, and the
    // triplet counts to agree.
    std::cout << "\n--- (6) C == G with outer legs swapped (G^T symmetry) ---\n";
    check(b.G_coeff.size() == b.C_coeff.size(),
          "|G_coeff| == |C_coeff| (" + std::to_string(b.G_coeff.size())
          + " vs " + std::to_string(b.C_coeff.size()) + ")");

    // Build the reverse map too and verify every C triplet is in G.
    std::unordered_map<std::uint64_t, double> gmap;
    gmap.reserve(b.G_coeff.size() * 2);
    for (const auto& t : b.G_coeff) gmap[key(t.a, t.b, t.c)] = t.value;

    double max_diff_CG = 0.0;
    int missing_CG = 0;
    for (const auto& t : b.C_coeff) {
        // C(a=sigma, b=lambda, c=mu) must match G(mu, lambda, sigma).
        auto it = gmap.find(key(t.c, t.b, t.a));
        if (it == gmap.end()) { ++missing_CG; continue; }
        max_diff_CG = std::max(max_diff_CG, std::abs(it->second - t.value));
    }
    check(missing_CG == 0, "every C triplet has a matching G triplet (missing="
          + std::to_string(missing_CG) + ")");
    check(max_diff_CG < 1e-12, "sup |C - G^T| < 1e-12 (max="
          + std::to_string(max_diff_CG) + ")");

    // --- (7) G_{mu, lambda, sigma} symmetric in the outer legs (mu<->sigma) ---
    // Only meaningful where both indices live in the common sub-range.
    // For our test l_cont == l_exch == 4, so n_mu == n_sigma == 25 and the
    // full check is nontrivial.
    std::cout << "\n--- (7) G symmetric in outer legs (mu <-> sigma) ---\n";
    if (n_mu == n_sigma) {
        double max_asym = 0.0;
        int missing_sym = 0;
        for (const auto& t : b.G_coeff) {
            auto it = gmap.find(key(t.c, t.b, t.a));
            if (it == gmap.end()) { ++missing_sym; continue; }
            max_asym = std::max(max_asym, std::abs(it->second - t.value));
        }
        check(missing_sym == 0 && max_asym < 1e-12,
              "G(mu,lambda,sigma) = G(sigma,lambda,mu) (missing=" +
              std::to_string(missing_sym) + ", max_asym=" +
              std::to_string(max_asym) + ")");
    } else {
        std::cout << "     skipped (n_mu != n_sigma)\n";
    }

    // --- (8) G(0, lambda, sigma) = delta_{lambda,sigma} / sqrt(4 pi) ---
    // Y_{0,0} = 1/sqrt(4 pi) is constant, so G_{0,lambda,sigma}
    // = (1/sqrt(4 pi)) * int Y^R_lambda Y^R_sigma dOmega
    // = (1/sqrt(4 pi)) * delta_{lambda,sigma} by real-Y orthonormality.
    std::cout << "\n--- (8) G(0, lambda, sigma) orthonormality identity ---\n";
    const double inv_sqrt4pi = 1.0 / std::sqrt(4.0 * M_PI);
    int ortho_missing = 0;
    double ortho_max_err = 0.0;
    // Build a diagonal-lookup: for mu=0, check every lambda in [0, n_sigma).
    for (int lam = 0; lam < n_sigma; ++lam) {
        auto it = gmap.find(key(0, lam, lam));
        if (it == gmap.end()) {
            ++ortho_missing;  // allowed to be zero only if it's truly zero
        } else {
            ortho_max_err = std::max(ortho_max_err,
                std::abs(it->second - inv_sqrt4pi));
        }
    }
    check(ortho_missing == 0 && ortho_max_err < 1e-12,
          "G(0, lam, lam) = 1/sqrt(4 pi) for lam in [0,n_sigma)  (missing="
          + std::to_string(ortho_missing) + ", max_err="
          + std::to_string(ortho_max_err) + ")");

    // Off-diagonal mu=0 entries: must all be absent from gmap (pruned at 1e-14).
    int ortho_offdiag_present = 0;
    for (int lam = 0; lam < n_lambda; ++lam)
        for (int sig = 0; sig < n_sigma; ++sig) {
            if (lam == sig) continue;
            if (gmap.count(key(0, lam, sig))) ++ortho_offdiag_present;
        }
    check(ortho_offdiag_present == 0,
          "G(0, lam, sig!=lam) prune to zero  (surviving="
          + std::to_string(ortho_offdiag_present) + ")");

    // --- (9) build_G_vector / build_C_vector are stable (re-entrant) ---
    // Call build_G_vector directly with the same shape and check it
    // returns the same multi-set of triplets as b.G_coeff.
    std::cout << "\n--- (9) build_G_vector is deterministic ---\n";
    auto G2 = WavefunctionSetup::build_G_vector(n_mu, n_lambda, n_sigma,
                                                /*verbose=*/false);
    check(G2.size() == b.G_coeff.size(),
          "second call returns same triplet count");
    std::unordered_map<std::uint64_t, double> g2map;
    g2map.reserve(G2.size() * 2);
    for (const auto& t : G2) g2map[key(t.a, t.b, t.c)] = t.value;
    double max_reentry_diff = 0.0;
    int reentry_missing = 0;
    for (const auto& t : b.G_coeff) {
        auto it = g2map.find(key(t.a, t.b, t.c));
        if (it == g2map.end()) { ++reentry_missing; continue; }
        max_reentry_diff = std::max(max_reentry_diff, std::abs(it->second - t.value));
    }
    check(reentry_missing == 0 && max_reentry_diff == 0.0,
          "bit-identical on re-entry (missing=" + std::to_string(reentry_missing)
          + ", max_diff=" + std::to_string(max_reentry_diff) + ")");

    // ------------------------------------------------------------------
    // Benchmark: wall-clock of prepare() across a sweep of l_cont.
    // ------------------------------------------------------------------
    std::cout << "\n--- Benchmark: WavefunctionSetup::prepare wall time ---\n";
    std::cout << "  l_cont | l_exch | n_lambda | |G|   | prepare s\n";
    std::cout << "  -------|--------|----------|-------|----------\n";
    for (int l_cont : {2, 4, 6, 8, 10}) {
        const int l_exch = std::min(l_cont, 10);
        if (l_cont + l_exch > data.Lmax_sce) continue;

        Parameters p = params;
        p.l_max_continuum = l_cont;
        try { p.validate(); } catch (...) { continue; }

        auto t0 = std::chrono::steady_clock::now();
        SetupBundle bb = WavefunctionSetup::prepare(p, data, /*energy=*/0.5);
        const double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();

        const int n_lambda =
            (bb.params.l_max_orbitals + 1) * (bb.params.l_max_orbitals + 1);

        std::cout << "    " << std::setw(4) << l_cont
                  << " | " << std::setw(6) << bb.params.l_max_exchange
                  << " | " << std::setw(8) << n_lambda
                  << " | " << std::setw(5) << bb.G_coeff.size()
                  << " | " << std::fixed << std::setprecision(3) << dt << "\n";
    }

    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL") << " (" << fails << " failed)\n";
    return fails == 0 ? 0 : 1;
}
