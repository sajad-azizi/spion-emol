// MoleculeHash.hpp -- stable 16-hex-digit hash for a molecule + grid context.
//
// Used to name persistent directories ($WORK/pot_<hash>, $WORK/dipole_<hash>,
// $SCRATCH/{sinv,rinv,psi}_<hash>_ik<nnnn>). Two runs on the same molecule,
// same grid, same angular cutoff share the hash and thus share pot + dipole
// checkpoints. Changing r_max, dr, or l_max_continuum rolls the hash so
// stale pot checkpoints are never silently reused.
//
// Implementation: FNV-1a over a canonical byte stream. Atoms are serialized
// by Z and quantized (x, y, z) at 1e-6 bohr, which tolerates tiny numerical
// differences in the preprocessing HDF5 (~1e-12) without false-sharing.

#pragma once

#include "io/HDF5Reader.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <iomanip>

namespace scatt {

inline std::string molecule_hash(const scatt::io::PreprocData& data,
                                 int                           l_max_continuum,
                                 double                        dr,
                                 std::size_t                   N_grid)
{
    // FNV-1a 64-bit.
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void* p, std::size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        for (std::size_t i = 0; i < n; ++i) {
            h ^= b[i];
            h *= 1099511628211ull;
        }
    };

    // Atoms (Z + quantized coords).
    for (const auto& a : data.atoms) {
        int32_t Z = a.Z;
        int64_t qx = static_cast<int64_t>(std::llround(a.x * 1.0e6));
        int64_t qy = static_cast<int64_t>(std::llround(a.y * 1.0e6));
        int64_t qz = static_cast<int64_t>(std::llround(a.z * 1.0e6));
        mix(&Z, sizeof(Z));
        mix(&qx, sizeof(qx));
        mix(&qy, sizeof(qy));
        mix(&qz, sizeof(qz));
    }

    // Grid + angular context -- different grids are different caches.
    int32_t  lmc = l_max_continuum;
    int64_t  qdr = static_cast<int64_t>(std::llround(dr * 1.0e9));
    int64_t  qN  = static_cast<int64_t>(N_grid);
    int32_t  Lsce= data.Lmax_sce;
    mix(&lmc,  sizeof(lmc));
    mix(&qdr,  sizeof(qdr));
    mix(&qN,   sizeof(qN));
    mix(&Lsce, sizeof(Lsce));

    std::ostringstream s;
    s << std::hex << std::setw(16) << std::setfill('0') << h;
    return s.str();
}

}  // namespace scatt
