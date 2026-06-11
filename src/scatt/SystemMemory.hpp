// SystemMemory.hpp -- detect physical RAM on the host (macOS + Linux).
//
// Used by StoragePlanner to decide MEMORY vs DISK for each of the four big
// scattering objects (pot, sinv, rinv, psi). Header-only, single function.

#pragma once

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <sys/types.h>
#endif

namespace scatt {

// Returns the total physical RAM in bytes, or 0 if detection failed.
// We intentionally use *total* RAM, not "available" -- MemAvailable on Linux
// fluctuates based on caches, and for capacity planning we want a stable
// number. The reserve fraction (applied by StoragePlanner) covers transient
// use by the OS and Eigen temporaries.
inline std::size_t detect_total_ram_bytes() {
#if defined(__APPLE__)
    int      mib[2] = { CTL_HW, HW_MEMSIZE };
    uint64_t v = 0;
    size_t   len = sizeof(v);
    if (sysctl(mib, 2, &v, &len, nullptr, 0) == 0)
        return static_cast<std::size_t>(v);
    return 0;
#else
    std::ifstream f("/proc/meminfo");
    std::string   line;
    while (std::getline(f, line)) {
        if (line.rfind("MemTotal:", 0) == 0) {
            unsigned long long kb = 0;
            if (std::sscanf(line.c_str(), "MemTotal: %llu kB", &kb) == 1)
                return static_cast<std::size_t>(kb) * 1024ull;
        }
    }
    return 0;
#endif
}

}  // namespace scatt
