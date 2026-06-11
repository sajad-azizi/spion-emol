// omp.h -- stub for systems without OpenMP (e.g. macOS Apple Clang
// without libomp).  Used ONLY when the reference code's hard-coded
// `#include <omp.h>` is encountered and no real OpenMP is available.
//
// Provides minimum API surface so reference code compiles; runtime
// behaviour is identical to "OMP disabled, 1 thread".
#pragma once
inline int  omp_get_max_threads()       { return 1; }
inline int  omp_get_num_threads()       { return 1; }
inline int  omp_get_thread_num()        { return 0; }
inline void omp_set_num_threads(int)    {}
inline int  omp_in_parallel()           { return 0; }
inline double omp_get_wtime()           { return 0.0; }
