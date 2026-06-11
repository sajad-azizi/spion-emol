// BoundStateIO.hpp -- HDF5 read/write for the energy-independent
// bound-state cache.  Writes once on first run, reads on subsequent
// runs at different continuum energies (skipping the bisection).
//
// File format (little-endian, native double):
//
//   /bound/energy       -- attr (double): gsEnergy
//   /bound/i_match      -- attr (int):    matching grid index
//   /bound/N_grid       -- attr (int)
//   /bound/n_channels   -- attr (int)
//   /bound/dr           -- attr (double)
//   /bound/molhash      -- attr (string): identifies the potential
//   /bound/chi_re       -- (N_grid, n_channels) double
//   /bound/chi_im       -- (N_grid, n_channels) double  (zero for real V)
//
// Atomic write: write to <path>.tmp then rename to <path>.
#pragma once

#include "Common.hpp"

#include <string>
#include <vector>

namespace sph3d {

struct BoundState {
    double                       gsEnergy = 0.0;
    int                          i_match  = 0;
    int                          N_grid   = 0;
    int                          n_channels = 0;
    double                       dr       = 0.0;
    std::string                  molhash;
    // chi[k](idx) -- bound radial wavefunction in real-Y basis,
    // u-convention (chi = r * F_lm).  Normalized to ∫|chi|² dr = 1.
    std::vector<Eigen::VectorXcd> eigfunc;
};

// Save the bound state to <path>.h5 atomically (write to .tmp, rename).
// Throws std::runtime_error on I/O failure.
void save_bound_state(const std::string& path, const BoundState& bs);

// Load the bound state from <path>.h5.  Returns true on success;
// false (without throwing) if the file does not exist OR has a
// mismatched molhash / N_grid / n_channels (treated as cache miss).
// Throws std::runtime_error on I/O failure of an existing file.
bool load_bound_state(const std::string& path,
                      const std::string& expected_molhash,
                      int expected_N_grid,
                      int expected_n_channels,
                      BoundState& out);

}  // namespace sph3d
