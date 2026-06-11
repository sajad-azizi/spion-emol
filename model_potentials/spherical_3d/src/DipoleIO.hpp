// DipoleIO.hpp -- per-scan manifest.h5 and per-energy ik<NNNN>.h5
// writers, byte-compatible with the parent project's gather_dipoles.py.
//
// Schema (matches static_exchangeHF/src/scatt/DipoleIO.{hpp,cpp}):
//
//   manifest.h5 (one per scan):
//       /grid           attrs: r_min, dr, N_grid, l_max_continuum, E_HOMO
//       /kgrid          attrs: dk, ik_min, ik_max
//       /occ/energies            (n_occ,) double      (= [bound-state E])
//       /occ/spin_factors        (n_occ,) double      (= [1.0])
//       /run            attrs: molecule_name, molhash, iso_date_utc
//
//   ik<NNNN>.h5 (one per energy):
//       /meta           attrs: ik, k, E, omega
//       /dipole/length/x   datasets: D_raw_re, D_raw_im, D_ortho_re, D_ortho_im
//       /dipole/length/y   ...
//       /dipole/length/z   ...
//       /dipole/velocity/x ...
//       /dipole/velocity/y ...
//       /dipole/velocity/z ...
//
// q-map for "x", "y", "z" matches parent project: x=Y^R_{1,+1},
// y=Y^R_{1,-1}, z=Y^R_{1,0}.  D_raw / D_ortho are length n_channels
// each (one entry per real-Y channel β).  In spherical_3d's simple
// one-particle model there is no occupied-orbital orthogonalization,
// so D_ortho == D_raw (we still write both fields so post-processing
// works without modification).
//
// Atomic writes: every file is written as <name>.tmp then renamed.
#pragma once

#include "Common.hpp"

#include <array>
#include <complex>
#include <string>
#include <vector>

namespace sph3d {

struct ScanMeta {
    double      r_min        = 0.0;
    double      dr           = 0.0;
    int         N_grid       = 0;
    int         l_max_continuum = 0;
    double      E_HOMO       = 0.0;     // bound-state energy
    double      dk           = 0.01;
    int         ik_min       = 1;
    int         ik_max       = 1;       // INCLUSIVE
    int         n_occ        = 1;       // = 1 for spherical_3d
    std::vector<double> occ_energies;
    std::vector<double> occ_spin_factors;
    std::string molecule_name;
    std::string molhash;
    std::string iso_date_utc;
};

struct DipolePerIK {
    int    ik = 0;
    double k = 0.0;
    double E = 0.0;
    double omega = 0.0;        // photon energy = E - E_HOMO
    // 6 slices indexed [gauge*3 + pol]:
    //   gauge: 0=length, 1=velocity
    //   pol:   0=x (q=+1), 1=y (q=-1), 2=z (q=0)
    std::array<std::vector<dcompx>, 6> D_raw;
    std::array<std::vector<dcompx>, 6> D_ortho;   // == D_raw for spherical_3d
};

// Writers
void write_manifest(const std::string& scan_dir, const ScanMeta& meta);
void write_ik     (const std::string& scan_dir, const DipolePerIK& payload);

// Existence check (used by main to skip already-computed energies).
bool ik_exists(const std::string& scan_dir, int ik);

// Pretty-print ik tag, e.g. ik=42 -> "ik0042".
std::string ik_tag(int ik);

}  // namespace sph3d
