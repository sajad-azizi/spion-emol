// RunDirs.hpp -- per-run directory layout (mirrors the parent project):
//
//   $WORK     (persistent across runs)
//       pot_<molhash>.h5            -- V_eff cube (energy-independent)
//       bound_<molhash>.h5          -- ground-state E + chi_lm (energy-indep)
//       dipole_<molhash>_<scan>/    -- one directory per scan
//           manifest.h5             -- scan-level metadata
//           ik<NNNN>.h5             -- per-energy dipole payload
//
//   $SCRATCH  (short-lived, currently unused for spherical_3d -- the
//             bound-state psi is small enough to keep in $WORK)
//
// `molhash` is a 16-hex-char content hash over all parameters that
// change V_eff or the bound-state.  Two runs with identical hash share
// caches; any change (different L, V0, l_max, dr, ...) yields a new hash.
#pragma once

#include <string>

namespace sph3d {

struct RunConfig {
    // Potential identification (everything that affects V_eff(r)):
    std::string kind          = "cubic";
    double      V0            = 0.75;
    double      L_box         = 1.5;
    double      a_gauss       = 1.0;
    double      b_gauss       = 1.5;
    double      c_gauss       = 2.0;
    double      a_soft        = 0.5;
    double      R_h2          = 2.0;
    double      a_h2          = 0.7990;

    // Grid:
    int         N_grid        = 0;
    double      dr            = 0.0;

    // Angular cutoff:
    int         l_max         = 0;

    // Angular-integration grid for V_eff (only matters when V is built
    // by quadrature; the analytic-Legendre path for h2plus_johnson is
    // independent of these but we hash them anyway for safety).
    int         N_theta       = 32;
    int         N_phi         = 64;
};

// 16-hex-char content hash of the RunConfig (deterministic, version-1).
std::string molhash(const RunConfig& c);

// Resolve $WORK / $SCRATCH from environment, with sensible fallbacks.
//   - $WORK    defaults to ./work
//   - $SCRATCH defaults to ./scratch
// CLI overrides win over environment.
struct RunDirs {
    std::string work;
    std::string scratch;
};
RunDirs resolve_dirs(const std::string& cli_work     = "",
                     const std::string& cli_scratch  = "");

// Helpers (mkdir -p):
void ensure_dir(const std::string& path);

}  // namespace sph3d
