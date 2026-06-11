// Banner.hpp -- one-screen runtime report of the linear-algebra backend,
// GPU device (if any), HDF5 library, and threading fanout.
//
// Called at the start of every scattering main (driver, tools, tests) so
// the log unambiguously records WHICH backend actually executed -- i.e.
// did MKL really load? did SYCL see a GPU? what HDF5 did we link?  If you
// see "GPU: Intel Data Center GPU Max 1550" in the banner you know for
// a fact the offload path is live.

#pragma once

#include <iostream>
#include <string>

#ifdef SCATT_HAS_MKL
#include <mkl_service.h>
#endif

#include <hdf5.h>

// GpuPropagate header is pulled unconditionally because it declares
// GpuContext::gpu_available() for both the SYCL and stub branches.
#include "scatt/GpuPropagate.hpp"

namespace scatt {

inline void print_la_banner(std::ostream& os = std::cout) {
    // ---- Linear algebra backend ----
#ifdef SCATT_HAS_MKL
    char mkl_buf[256] = {0};
    MKL_Get_Version_String(mkl_buf, sizeof(mkl_buf));
    std::string mkl_v(mkl_buf);
    while (!mkl_v.empty() &&
           (mkl_v.back() == ' ' || mkl_v.back() == '\n' || mkl_v.back() == '\r'))
        mkl_v.pop_back();
    const int nmax = mkl_get_max_threads();
    os << "[banner] LA backend : MKL  (" << mkl_v
       << ", max_threads=" << nmax << ")\n";
#else
    os << "[banner] LA backend : Eigen native (no MKL)  -- "
          "dense LU/GEMM will be serial\n";
#endif

    // ---- HDF5 ----
    {
        unsigned mj = 0, mn = 0, rel = 0;
        H5get_libversion(&mj, &mn, &rel);
        os << "[banner] HDF5       : " << mj << "." << mn << "." << rel << "\n";
    }

    // ---- GPU (SYCL / oneMKL) ----
#ifdef SCATT_HAS_SYCL
    if (GpuContext::gpu_available()) {
        try {
            GpuContext probe(/*prefer_gpu=*/true);
            os << "[banner] GPU        : " << probe.info().device_name
               << "  (platform=" << probe.info().platform_name
               << ", HBM=" << (probe.info().global_mem_bytes >> 30) << " GB)\n";
        } catch (const std::exception& e) {
            os << "[banner] GPU        : SYCL compiled in, probe FAILED: "
               << e.what() << "\n";
        }
    } else {
        os << "[banner] GPU        : SYCL compiled in, but no GPU device "
              "visible at runtime\n";
    }
#else
    os << "[banner] GPU        : not compiled in  "
          "(rebuild with -DSCATT_WITH_SYCL=ON to enable)\n";
#endif
}

}  // namespace scatt
