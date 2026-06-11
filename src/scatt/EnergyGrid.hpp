// EnergyGrid.hpp -- uniform-in-k energy grid for the scattering scan.
//
// Rationale: uniform-in-k means  dE = k · dk,  so spacing in E becomes finer
// near threshold. This is the right density for Wigner-threshold behavior
// and for resolving low-energy resonances in C8F8.
//
// Convention:
//   k(ik) = ik · dk
//   E(ik) = 0.5 · k(ik)^2        (atomic units; E = k²/2)
//
// File naming: files that live per energy are tagged with the zero-padded
// ik index ("ik0007.h5", "psi_<hash>_ik0007/", ...) so that a run can be
// resumed and post-processing can recover (ik, k, E) from the filename alone.

#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace scatt {

struct EnergyGrid {
    double dk       = 0.01;   // step in momentum (au)
    int    ik_min   = 1;      // ik starts from 1 (ik=0 is k=0, E=0)
    int    ik_max   = 101;    // exclusive upper bound

    inline double k(int ik) const { return ik * dk; }
    inline double E(int ik) const { const double kv = k(ik); return 0.5 * kv * kv; }
    inline int    size() const    { return ik_max - ik_min; }

    // "ik0007"  (padded to 4 digits by default; enough for up to 9999 points).
    std::string tag(int ik, int pad = 4) const {
        if (ik < 0) throw std::invalid_argument("EnergyGrid::tag: ik < 0");
        std::string s = std::to_string(ik);
        while (static_cast<int>(s.size()) < pad) s = "0" + s;
        return std::string("ik") + s;
    }

    void validate() const {
        if (!(dk > 0.0))            throw std::invalid_argument("EnergyGrid: dk must be > 0");
        if (ik_min < 0)             throw std::invalid_argument("EnergyGrid: ik_min < 0");
        if (ik_max <= ik_min)       throw std::invalid_argument("EnergyGrid: ik_max <= ik_min");
    }
};

}  // namespace scatt
