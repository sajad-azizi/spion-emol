// Banner.hpp (preprocessing) -- one-line runtime check that the preproc
// build has the important optional backends (FFTW for SCE, HDF5, OpenMP).
// Printed at the top of main_preprocess so the run log records it.

#pragma once

#include <iostream>

#include <hdf5.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace preproc {

inline void print_banner(std::ostream& os = std::cerr) {
    // ---- FFTW (SCE angular projection fast path) ----
#ifdef PREPROC_HAS_FFTW
    os << "[banner] FFTW       : yes  (SCE r2c FFT enabled; ~300x faster at Lmax=300)\n";
#else
    os << "[banner] FFTW       : NO   (falling back to O(N_phi^2) direct DFT -- "
          "very slow for Lmax > 50)\n";
#endif

    // ---- HDF5 library ----
    {
        unsigned mj = 0, mn = 0, rel = 0;
        H5get_libversion(&mj, &mn, &rel);
        os << "[banner] HDF5       : " << mj << "." << mn << "." << rel << "\n";
    }

    // ---- OpenMP ----
#ifdef _OPENMP
    os << "[banner] OpenMP     : yes  (max_threads="
       << omp_get_max_threads() << ")\n";
#else
    os << "[banner] OpenMP     : NO   (SCE + Hartree will run serially)\n";
#endif
}

}  // namespace preproc
