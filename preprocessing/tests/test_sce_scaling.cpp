// test_sce_scaling.cpp — stress-test the SCE projector + V_en builder at
// increasing Lmax values, to characterize throughput and verify memory
// stays modest. Output is informational only (no pass/fail here).
//
// This test uses a trivial analytic density rho(r) = exp(-alpha r^2) centered
// at the origin, so:
//   - F^R_{l,m} = 0 for (l,m) != (0,0)
//   - F^R_{0,0}(r) = sqrt(4 pi) * rho(r)
// which gives us a no-cost accuracy check (machine-precision zero on all
// non-(0,0) channels, and known F_{00} shape).
//
// The target scaling is Lmax ~ 300 for C8F8. We do NOT run that here
// (memory for output Flm alone is 7 GB), but we go up to Lmax=100 to
// confirm no O((Lmax+1)^4 * stuff) in hot paths.

#include "angular/Grid.hpp"
#include "potential/Vnuclear.hpp"
#include "sce/RadialGrid.hpp"
#include "sce/SCE.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using preproc::angular::AngGrid;
using preproc::angular::lm_index;

int main() {
    // Gaussian density centered at the SCE origin.
    const double alpha = 1.0;
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    preproc::sce::F3D F = [&](const Eigen::Vector3d& r) {
        return std::exp(-alpha * r.squaredNorm());
    };

    // Synthetic "atoms" for V_en timing.
    std::vector<preproc::molden::Atom> atoms;
    atoms.push_back({"H", 1, 1, Eigen::Vector3d(0.5, 0.0, 0.0)});
    atoms.push_back({"H", 1, 2, Eigen::Vector3d(-0.5, 0.0, 0.0)});
    atoms.push_back({"C", 6, 3, Eigen::Vector3d(1.0, 1.0, 0.0)});
    atoms.push_back({"F", 9, 4, Eigen::Vector3d(-1.0, -1.0, 0.0)});

    auto rg = preproc::sce::RadialGrid::build(0.0, 0.05, 201);
    std::cerr << std::setprecision(4);
    std::cerr << "Nr=" << rg.N << "  dr=" << rg.dr << "\n";
    std::cerr << "Lmax  Nlm      nTheta  nPhi   t_sce(s)  t_Ven(s)  |F_{0,0}(0)-sqrt(4pi)|  max|F_{l>0}|\n";

    for (int L : {16, 32, 50, 75, 100}) {
        auto ag = AngGrid::build_basic(L);
        const int Nlm = preproc::angular::n_channels(L);

        auto t0 = std::chrono::steady_clock::now();
        auto Flm = preproc::sce::project(rg, ag, origin, F, false);
        auto t1 = std::chrono::steady_clock::now();
        auto V   = preproc::potential::build_V_en(atoms, origin, rg, L);
        auto t2 = std::chrono::steady_clock::now();

        const double dt_sce = std::chrono::duration<double>(t1 - t0).count();
        const double dt_ven = std::chrono::duration<double>(t2 - t1).count();

        // Accuracy check: non-(0,0) channels should be exactly zero for a
        // centered spherically-symmetric Gaussian.
        double max_non_00 = 0.0;
        for (int ch = 0; ch < Nlm; ++ch) {
            if (ch == lm_index(0, 0)) continue;
            for (int k = 0; k < rg.N; ++k) {
                max_non_00 = std::max(max_non_00, std::abs(Flm(ch, k)));
            }
        }
        const double f00_at_0 = Flm(lm_index(0, 0), 0);
        const double expected = std::sqrt(4.0 * M_PI);

        std::cerr << std::setw(4) << L
                  << "   " << std::setw(6) << Nlm
                  << "  " << std::setw(4) << ag.nTheta
                  << "   " << std::setw(4) << ag.nPhi
                  << "    " << std::setw(7) << dt_sce
                  << "   " << std::setw(7) << dt_ven
                  << "   " << std::setw(10) << std::abs(f00_at_0 - expected)
                  << "     " << std::setw(10) << max_non_00
                  << "\n";
        // If either number blows up, that's a red flag.
        if (max_non_00 > 1e-10) {
            std::cerr << "  [WARN] non-(0,0) channel has magnitude " << max_non_00 << "\n";
        }
    }
    return 0;
}
