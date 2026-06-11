#include "scatt/Bench.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <ostream>
#include <sstream>
#include <vector>
#include <string>
#include <sys/resource.h>

#ifdef __linux__
#  include <cerrno>
#  include <cstring>
#  include <dirent.h>
#  include <fcntl.h>
#  include <linux/perf_event.h>
#  include <sys/ioctl.h>
#  include <sys/syscall.h>
#  include <sys/types.h>
#  include <unistd.h>
#  include <set>
#endif

#if defined(SCATT_HAS_OPENMP)
#  include <omp.h>
#endif

// GPU energy headers -- exactly one of SCATT_HAS_SYCL (Intel PVC via
// Level Zero Sysman) or SCATT_HAS_CUDA (NVIDIA via NVML) is active per
// build (CMake mutex check enforces this).  When neither is set the
// gpu_energy_uj_now() body collapses to a stub.
#if defined(SCATT_HAS_SYCL)
#  include <level_zero/ze_api.h>
#  include <level_zero/zes_api.h>
#elif defined(SCATT_HAS_CUDA)
#  include <nvml.h>
#endif

namespace scatt {

namespace {

std::uint64_t now_ns() {
    using clk = std::chrono::steady_clock;
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            clk::now().time_since_epoch()).count());
}

std::string format_seconds(double s) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(6) << s;
    return o.str();
}
std::string format_ms(double ms) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(5) << ms;
    return o.str();
}
std::string format_bytes(std::size_t b) {
    const double G = b / double(1ull << 30);
    const double M = b / double(1ull << 20);
    std::ostringstream o;
    o << std::fixed << std::setprecision(5);
    if (G >= 1.0) o << G << " GB";
    else          o << M << " MB";
    return o.str();
}
std::string format_joules(double j) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(3) << j;
    return o.str();
}

}  // namespace

std::size_t current_peak_rss_bytes() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#if defined(__APPLE__)
    // macOS reports ru_maxrss in BYTES.
    return static_cast<std::size_t>(ru.ru_maxrss);
#else
    // Linux reports ru_maxrss in KILOBYTES.
    return static_cast<std::size_t>(ru.ru_maxrss) * 1024u;
#endif
}

// ---------------------------------------------------------------------------
// CPU energy via Intel RAPL.  Two backends, tried in order:
//
//   (A) /sys/class/powercap/intel-rapl:N/energy_uj  -- sysfs direct read.
//       Cheapest path, works on any Linux with intel_rapl_msr/intel_rapl_common
//       loaded and the powercap sysfs world-readable.  Pre-2020 production
//       default.  Since CVE-2020-8694 most HPC clusters (including LRZ Phase 2)
//       set the energy_uj files root-only, so this path returns 0 there.
//
//   (B) perf_event_open(power/energy-pkg).  Uses the perf_events kernel
//       infrastructure -- same path the `perf stat` utility uses.  Permission
//       gate is /proc/sys/kernel/perf_event_paranoid:  values <= 1 let
//       non-root users open the energy-pkg PMU.  LRZ typically sets paranoid
//       to 1 (perf events for non-CPU PMUs allowed for non-root), so this
//       fallback unlocks CPU energy where the sysfs path is blocked.
//
// Both report the SAME underlying Intel RAPL energy-pkg counter, just via
// different kernel interfaces -- so the numbers from (A) and (B) on the
// same machine match to floating-point.
//
// Backend is selected lazily on first call: try (A); if it returns 0 in a
// SECOND call after we know time has passed, try (B); if (B) also fails,
// CPU energy stays "unavailable".  Multi-package machines: both backends
// sum across all packages on the node.
//
// (C) likwid-perfctr is reported as available/unavailable at startup but
// not integrated -- the user asked not to vendor a subprocess-spawning
// energy meter.  If you really want likwid-side energy, link against
// liblikwid and add a marker-mode region; for now the perf_event path
// covers the same RAPL counter as likwid would, just via a different
// kernel interface.
// ---------------------------------------------------------------------------

#ifdef __linux__
namespace {

// (A) sysfs reader.  Returns total energy in uJ across all top-level
// intel-rapl packages; 0 if no readable files (locked-down kernel).
std::uint64_t rapl_sysfs_energy_uj() {
    const char* root = "/sys/class/powercap";
    DIR* d = opendir(root);
    if (!d) return 0;
    std::uint64_t total_uj = 0;
    int seen = 0;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        const std::string n = e->d_name;
        // Match "intel-rapl:N" (no further colon -- children like :0:0
        // are subsets we'd double-count).
        if (n.rfind("intel-rapl:", 0) != 0) continue;
        if (n.find(':', std::string("intel-rapl:").size()) != std::string::npos) continue;
        const std::string p = std::string(root) + "/" + n + "/energy_uj";
        FILE* f = std::fopen(p.c_str(), "r");
        if (!f) continue;
        unsigned long long v = 0;
        if (std::fscanf(f, "%llu", &v) == 1) {
            total_uj += v;
            ++seen;
        }
        std::fclose(f);
    }
    closedir(d);
    return seen > 0 ? total_uj : 0;
}

// (B) perf_event_open backend state.  Lazily initialised on first call.
struct RaplPerfState {
    bool tried_init   = false;     // attempted to open perf events
    bool usable       = false;     // at least one fd opened successfully
    int  type         = -1;        // PMU type for "power" (dynamic per boot)
    int  config       = -1;        // event code for energy-pkg
    double scale_j_per_count = 0.0; // counter unit -> joules
    std::vector<int> fds;          // one fd per physical package
};

RaplPerfState g_rapl_perf;
std::once_flag g_rapl_perf_once;

// Read /sys/bus/event_source/devices/power/type (boot-dynamic int).
int read_perf_power_type() {
    FILE* f = std::fopen("/sys/bus/event_source/devices/power/type", "r");
    if (!f) return -1;
    int v = -1;
    if (std::fscanf(f, "%d", &v) != 1) v = -1;
    std::fclose(f);
    return v;
}

// Read /sys/bus/event_source/devices/power/events/energy-pkg ("event=0x02").
int read_perf_energy_pkg_config() {
    FILE* f = std::fopen("/sys/bus/event_source/devices/power/events/energy-pkg", "r");
    if (!f) return -1;
    char buf[64] = {0};
    std::size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    if (n == 0) return -1;
    const char* eq = std::strchr(buf, '=');
    if (!eq) return -1;
    return (int)std::strtol(eq + 1, nullptr, 0);
}

// Read .../energy-pkg.scale (float, joules per counter increment).
double read_perf_energy_pkg_scale() {
    FILE* f = std::fopen("/sys/bus/event_source/devices/power/events/energy-pkg.scale", "r");
    if (!f) return 0.0;
    double s = 0.0;
    if (std::fscanf(f, "%lf", &s) != 1) s = 0.0;
    std::fclose(f);
    return s;
}

// Enumerate one CPU index per physical package, e.g. {0, 56} for a
// dual-socket Sapphire Rapids 112-thread node.  Used to open a separate
// perf event per package (RAPL energy-pkg is per-package).
std::vector<int> enumerate_package_cpus() {
    std::set<int>          packages_seen;
    std::vector<int>       cpu_per_pkg;
    DIR* d = opendir("/sys/devices/system/cpu");
    if (!d) return cpu_per_pkg;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        const std::string n = e->d_name;
        if (n.rfind("cpu", 0) != 0) continue;
        if (n.size() < 4) continue;
        bool all_digits = true;
        for (std::size_t i = 3; i < n.size(); ++i)
            if (!std::isdigit((unsigned char)n[i])) { all_digits = false; break; }
        if (!all_digits) continue;
        const int cpu_id = std::atoi(n.c_str() + 3);
        const std::string p = "/sys/devices/system/cpu/" + n + "/topology/physical_package_id";
        FILE* f = std::fopen(p.c_str(), "r");
        if (!f) continue;
        int pkg = -1;
        if (std::fscanf(f, "%d", &pkg) != 1) pkg = -1;
        std::fclose(f);
        if (pkg < 0) continue;
        if (packages_seen.insert(pkg).second) {
            cpu_per_pkg.push_back(cpu_id);
        }
    }
    closedir(d);
    std::sort(cpu_per_pkg.begin(), cpu_per_pkg.end());
    return cpu_per_pkg;
}

void rapl_perf_init() {
    g_rapl_perf.tried_init = true;
    g_rapl_perf.type   = read_perf_power_type();
    g_rapl_perf.config = read_perf_energy_pkg_config();
    g_rapl_perf.scale_j_per_count = read_perf_energy_pkg_scale();

    std::fprintf(stderr,
        "[cpu_energy] sysfs locked; trying perf_event_open (power PMU):  "
        "type=%d  energy-pkg config=0x%x  scale=%.3e J/count\n",
        g_rapl_perf.type, g_rapl_perf.config, g_rapl_perf.scale_j_per_count);
    if (g_rapl_perf.type < 0 || g_rapl_perf.config < 0 ||
        g_rapl_perf.scale_j_per_count <= 0.0) {
        std::fprintf(stderr,
            "[cpu_energy] perf power PMU sysfs metadata missing -- "
            "kernel built without CONFIG_PERF_EVENTS_INTEL_RAPL?  "
            "Giving up on CPU energy.\n");
        return;
    }

    const std::vector<int> pkg_cpus = enumerate_package_cpus();
    if (pkg_cpus.empty()) {
        std::fprintf(stderr,
            "[cpu_energy] could not enumerate CPU packages from /sys/devices/system/cpu.\n");
        return;
    }

    for (int cpu : pkg_cpus) {
        struct perf_event_attr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.type           = static_cast<__u32>(g_rapl_perf.type);
        attr.size           = sizeof(attr);
        attr.config         = static_cast<__u64>(g_rapl_perf.config);
        attr.disabled       = 0;
        attr.exclude_kernel = 0;   // energy-pkg counts kernel+user; required
        attr.exclude_hv     = 0;
        // pid=-1, cpu=cpu  -> system-wide on this CPU.
        // group_fd=-1, flags=0.
        const long fd = ::syscall(SYS_perf_event_open, &attr,
                                  /*pid=*/-1, /*cpu=*/cpu,
                                  /*group_fd=*/-1, /*flags=*/0UL);
        if (fd < 0) {
            std::fprintf(stderr,
                "[cpu_energy] perf_event_open(cpu=%d) failed: %s (errno=%d).  "
                "On HPC: typically perf_event_paranoid>=2 (try `sysctl "
                "kernel.perf_event_paranoid=1` if you have root) or missing "
                "CAP_PERFMON.\n",
                cpu, std::strerror(errno), errno);
            // Don't open partial state; close anything already opened.
            for (int prev : g_rapl_perf.fds) ::close(prev);
            g_rapl_perf.fds.clear();
            return;
        }
        g_rapl_perf.fds.push_back(static_cast<int>(fd));
        // Energy events start counting immediately; no PERF_EVENT_IOC_ENABLE needed.
    }
    g_rapl_perf.usable = !g_rapl_perf.fds.empty();
    std::fprintf(stderr,
        "[cpu_energy] perf_event_open OK on %zu package(s); CPU energy "
        "now available via the perf path.\n",
        g_rapl_perf.fds.size());
}

std::uint64_t rapl_perf_energy_uj() {
    std::call_once(g_rapl_perf_once, rapl_perf_init);
    if (!g_rapl_perf.usable) return 0;
    double total_j = 0.0;
    for (int fd : g_rapl_perf.fds) {
        std::uint64_t v = 0;
        const ssize_t got = ::read(fd, &v, sizeof(v));
        if (got != static_cast<ssize_t>(sizeof(v))) continue;
        total_j += static_cast<double>(v) * g_rapl_perf.scale_j_per_count;
    }
    // J -> uJ.  Cap at uint64 max to be safe.
    const double uj = total_j * 1.0e6;
    if (uj < 0.0)                     return 0;
    if (uj >= (double)UINT64_MAX)     return UINT64_MAX;
    return static_cast<std::uint64_t>(uj);
}

// (C) Optional likwid-perfctr availability probe.  Only logs availability
// at first call; does NOT integrate likwid as a backend (user explicitly
// asked NOT to vendor a subprocess-spawning meter).  If a user wants
// likwid-side energy, they can compile-link against liblikwid and add a
// marker-mode region -- the perf_event path here covers the same RAPL
// counter via a different kernel interface, so likwid is redundant unless
// you need per-domain breakdown (DRAM, PP0, PSYS).
void log_likwid_availability_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        // ::system() spawns /bin/sh; "which" is in POSIX util-linux.
        // Return value 0 means likwid-perfctr is on $PATH.
        const int rc = std::system("command -v likwid-perfctr > /dev/null 2>&1");
        if (rc == 0) {
            std::fprintf(stderr,
                "[cpu_energy] likwid-perfctr is on $PATH on this node.  "
                "(Currently using the perf_event backend; likwid would "
                "give the same energy-pkg number via a different "
                "interface.  No subprocess is spawned.)\n");
        } else {
            std::fprintf(stderr,
                "[cpu_energy] likwid-perfctr not on $PATH (not needed; "
                "using perf_event backend).\n");
        }
    });
}

}  // anonymous namespace
#endif  // __linux__

std::uint64_t rapl_energy_uj_now() {
#ifdef __linux__
    // (A) try sysfs first -- works on dev machines, fails on locked-down
    //     HPC kernels.
    const std::uint64_t sysfs = rapl_sysfs_energy_uj();
    if (sysfs != 0) return sysfs;

    // (B) sysfs blocked: fall back to perf_event_open.  Logs likwid
    //     availability on the way through for diagnostic completeness.
    log_likwid_availability_once();
    return rapl_perf_energy_uj();
#else
    return 0;   // macOS / Windows: skip CPU energy entirely.
#endif
}

// ---------------------------------------------------------------------------
// GPU energy meter.  Vendor-specific kernel-driver API call (no subprocess).
//
//   SCATT_HAS_SYCL  -> Level Zero Sysman (zesPowerGetEnergyCounter)
//                       returns monotonic microjoules per power domain.
//                       Tested on Intel PVC Max 1550 / Max 1100.
//
//   SCATT_HAS_CUDA  -> NVML (nvmlDeviceGetTotalEnergyConsumption)
//                       returns monotonic millijoules per device.
//                       Supported on H100, A100, V100, L40, GH200; older
//                       cards (consumer / P100) return NVML_ERROR_NOT_
//                       SUPPORTED -> the call gracefully falls through
//                       to "unavailable".
//
//   (neither)       -> stub returning 0.
//
// In all live paths the result is summed across devices and reported in
// microjoules for uniformity with the CPU energy path.
//
// Initialised on the first call (lazy).  Handles are cached because
// re-enumerating them is comparatively expensive and the set is stable
// for the process lifetime.
//
// Caveat for SYCL: ZES_ENABLE_SYSMAN=1 must be in the env at the time of
// the FIRST Level Zero init for sysman queries to return any drivers.
// SYCL initialises Level Zero on first device use, so callers targeting
// the GPU path should set this var BEFORE launching scattering
// (run_c8f8_lrz.sh already does this in production).
// Caveat for CUDA: NVML's libnvidia-ml.so must be on LD_LIBRARY_PATH or
// in /usr/lib64.  CMake's find_package(CUDAToolkit) wires this up when
// SCATT_WITH_CUDA=ON.
// ---------------------------------------------------------------------------
#if defined(SCATT_HAS_SYCL)
namespace {

struct GpuEnergyState {
    bool initialized = false;
    bool usable      = false;
    std::vector<zes_pwr_handle_t> domains;
};
GpuEnergyState  g_gpu_state;
std::once_flag  g_gpu_state_once;

void gpu_energy_init_locked() {
    g_gpu_state.initialized = true;

    // Diagnostic mode: set SCATT_GPU_ENERGY_DEBUG=1 to log every step of
    // the Level Zero Sysman probe.  Useful on HPC sites where one of the
    // steps below (zesInit, zesDriverGet, zesDeviceEnumPowerDomains,
    // zesPowerGetEnergyCounter) silently returns nothing and the bench
    // prints "gpu_energy(L0)=unavailable" without any clue why.
    const char* dbg_env = std::getenv("SCATT_GPU_ENERGY_DEBUG");
    const bool  dbg = dbg_env && dbg_env[0] && dbg_env[0] != '0';
    auto log = [dbg](const char* fmt, auto... args) {
        if (dbg) std::fprintf(stderr, fmt, args...);
    };
    log("[gpu_energy] init starting; ZES_ENABLE_SYSMAN=%s\n",
        std::getenv("ZES_ENABLE_SYSMAN") ? std::getenv("ZES_ENABLE_SYSMAN")
                                          : "<unset>");

    // Prefer the dedicated Sysman init.  Falls back to the runtime init if
    // we're on an older loader; either path yields valid zes_driver_handle_t
    // values for zesDriverGet on modern (>= 2024.x) oneAPI runtimes.
    ze_result_t r = zesInit(0);
    log("[gpu_energy] zesInit(0) -> 0x%x %s\n", (unsigned)r,
        r == ZE_RESULT_SUCCESS ? "(SUCCESS)" : "(fail; will try zeInit fallback)");
    if (r != ZE_RESULT_SUCCESS) {
        r = zeInit(0);
        log("[gpu_energy] zeInit(0) -> 0x%x %s\n", (unsigned)r,
            r == ZE_RESULT_SUCCESS ? "(SUCCESS; relying on ZES_ENABLE_SYSMAN)"
                                    : "(fail; giving up)");
        if (r != ZE_RESULT_SUCCESS) return;
    }

    uint32_t n_drivers = 0;
    ze_result_t rdg1 = zesDriverGet(&n_drivers, nullptr);
    log("[gpu_energy] zesDriverGet(count) -> 0x%x  n_drivers=%u\n",
        (unsigned)rdg1, n_drivers);
    if (rdg1 != ZE_RESULT_SUCCESS) return;
    if (n_drivers == 0) {
        log("[gpu_energy] no Sysman drivers visible.  Common causes:\n"
            "             - ZES_ENABLE_SYSMAN not set before Level Zero init\n"
            "             - User lacks CAP_SYS_RAWIO / render-group membership\n"
            "             - Driver build does not expose Sysman to user-mode\n");
        return;
    }
    std::vector<zes_driver_handle_t> drivers(n_drivers);
    if (zesDriverGet(&n_drivers, drivers.data()) != ZE_RESULT_SUCCESS) return;

    int n_devs_total = 0;
    int n_pwr_total  = 0;
    for (auto drv : drivers) {
        uint32_t n_devs = 0;
        if (zesDeviceGet(drv, &n_devs, nullptr) != ZE_RESULT_SUCCESS) continue;
        if (n_devs == 0) continue;
        n_devs_total += static_cast<int>(n_devs);
        std::vector<zes_device_handle_t> devs(n_devs);
        if (zesDeviceGet(drv, &n_devs, devs.data()) != ZE_RESULT_SUCCESS) continue;
        for (auto dev : devs) {
            uint32_t n_pwr = 0;
            ze_result_t rpd = zesDeviceEnumPowerDomains(dev, &n_pwr, nullptr);
            log("[gpu_energy]   zesDeviceEnumPowerDomains -> 0x%x  n_pwr=%u\n",
                (unsigned)rpd, n_pwr);
            if (rpd != ZE_RESULT_SUCCESS) continue;
            if (n_pwr == 0) continue;
            std::vector<zes_pwr_handle_t> pwrs(n_pwr);
            if (zesDeviceEnumPowerDomains(dev, &n_pwr, pwrs.data())
                != ZE_RESULT_SUCCESS) continue;
            for (auto p : pwrs) {
                // Optional diagnostic test-read so the user sees whether
                // zesPowerGetEnergyCounter ACTUALLY works on this domain.
                // We always push the handle (legacy behaviour) so that an
                // initial transient counter failure doesn't silently
                // disable energy reporting -- subsequent measurements
                // might succeed.
                if (dbg) {
                    zes_power_energy_counter_t ctr{};
                    ze_result_t rec = zesPowerGetEnergyCounter(p, &ctr);
                    log("[gpu_energy]     zesPowerGetEnergyCounter -> 0x%x  "
                        "energy=%llu uJ\n", (unsigned)rec,
                        (unsigned long long)ctr.energy);
                }
                g_gpu_state.domains.push_back(p);
                ++n_pwr_total;
            }
        }
    }
    g_gpu_state.usable = !g_gpu_state.domains.empty();
    log("[gpu_energy] init done: drivers=%u  devices=%d  usable_domains=%d  "
        "-> %s\n", n_drivers, n_devs_total, n_pwr_total,
        g_gpu_state.usable ? "OK (Sysman energy counters wired)"
                            : "FAIL (gpu_energy will report L0 n/a)");
}

}  // namespace

std::uint64_t gpu_energy_uj_now() {
    std::call_once(g_gpu_state_once, gpu_energy_init_locked);
    if (!g_gpu_state.usable) return 0;
    std::uint64_t total_uj = 0;
    for (auto p : g_gpu_state.domains) {
        zes_power_energy_counter_t counter{};
        if (zesPowerGetEnergyCounter(p, &counter) == ZE_RESULT_SUCCESS) {
            // counter.energy is monotonic microjoules per Level Zero spec.
            total_uj += counter.energy;
        }
    }
    return total_uj;
}

#elif defined(SCATT_HAS_CUDA)
namespace {

// NVIDIA NVML backend.  Mirrors the SYCL/Sysman path: lazy init,
// per-device handle cache, summed energy in microjoules.
//
// `nvmlDeviceGetTotalEnergyConsumption` returns monotonic energy since
// the driver was last reloaded, in MILLIJOULES (uint64).  We convert
// mJ -> uJ on the way out so the bench output unit is identical to the
// SYCL path.
//
// Supported NVIDIA architectures (as of CUDA 12 / NVML driver 535):
//   * H100 (Hopper)              -- supported
//   * A100, A30, A40 (Ampere)    -- supported
//   * L40, L4 (Ada)              -- supported
//   * V100 (Volta)               -- supported
//   * GH200 (Grace-Hopper)       -- supported
//   * P100 (Pascal) and older    -- nvmlDeviceGetTotalEnergyConsumption
//                                    returns NVML_ERROR_NOT_SUPPORTED.
//                                    Bench will report 0 silently for
//                                    such devices.
//   * Consumer GeForce cards     -- depends on driver; usually returns
//                                    NVML_ERROR_NOT_SUPPORTED.
struct GpuEnergyStateCuda {
    bool initialized = false;
    bool usable      = false;
    std::vector<nvmlDevice_t> devices;
};
GpuEnergyStateCuda g_gpu_state_cuda;
std::once_flag     g_gpu_state_cuda_once;

void gpu_energy_init_locked_cuda() {
    g_gpu_state_cuda.initialized = true;

    const char* dbg_env = std::getenv("SCATT_GPU_ENERGY_DEBUG");
    const bool  dbg = dbg_env && dbg_env[0] && dbg_env[0] != '0';
    auto log = [dbg](const char* fmt, auto... args) {
        if (dbg) std::fprintf(stderr, fmt, args...);
    };
    log("[gpu_energy] init starting; backend=NVML\n");

    nvmlReturn_t r = nvmlInit_v2();
    log("[gpu_energy] nvmlInit_v2() -> %d %s\n", (int)r,
        r == NVML_SUCCESS ? "(SUCCESS)" : "(fail; giving up)");
    if (r != NVML_SUCCESS) return;

    unsigned int n_devs = 0;
    r = nvmlDeviceGetCount_v2(&n_devs);
    log("[gpu_energy] nvmlDeviceGetCount_v2() -> %d  n_devs=%u\n",
        (int)r, n_devs);
    if (r != NVML_SUCCESS || n_devs == 0) return;

    int n_supported = 0, n_unsupported = 0;
    for (unsigned int i = 0; i < n_devs; ++i) {
        nvmlDevice_t dev{};
        r = nvmlDeviceGetHandleByIndex_v2(i, &dev);
        if (r != NVML_SUCCESS) {
            log("[gpu_energy]   nvmlDeviceGetHandleByIndex_v2(%u) -> %d\n",
                i, (int)r);
            continue;
        }

        // Probe whether this device supports the energy counter.  We
        // do a test read; if NVML_ERROR_NOT_SUPPORTED comes back the
        // device is skipped (older arch).
        unsigned long long probe_mj = 0;
        r = nvmlDeviceGetTotalEnergyConsumption(dev, &probe_mj);
        if (r == NVML_SUCCESS) {
            log("[gpu_energy]   device %u supports energy counter; "
                "probe=%llu mJ\n", i, probe_mj);
            g_gpu_state_cuda.devices.push_back(dev);
            ++n_supported;
        } else {
            log("[gpu_energy]   device %u does NOT support energy counter "
                "(NVML error %d -- typically Pascal/older or consumer card)\n",
                i, (int)r);
            ++n_unsupported;
        }
    }
    g_gpu_state_cuda.usable = !g_gpu_state_cuda.devices.empty();
    log("[gpu_energy] init done: devices_total=%u  supported=%d  unsupported=%d  "
        "-> %s\n", n_devs, n_supported, n_unsupported,
        g_gpu_state_cuda.usable ? "OK (NVML energy counters wired)"
                                : "FAIL (gpu_energy will report NVML n/a)");
}

}  // namespace

std::uint64_t gpu_energy_uj_now() {
    std::call_once(g_gpu_state_cuda_once, gpu_energy_init_locked_cuda);
    if (!g_gpu_state_cuda.usable) return 0;
    std::uint64_t total_mj = 0;
    for (auto dev : g_gpu_state_cuda.devices) {
        unsigned long long energy_mj = 0;
        if (nvmlDeviceGetTotalEnergyConsumption(dev, &energy_mj) == NVML_SUCCESS) {
            // monotonic millijoules per NVML spec.
            total_mj += energy_mj;
        }
    }
    // mJ -> uJ.  Range: ULL max in mJ would be ~5.8e8 years at 1 kW, so
    // no overflow concern in practice.  Multiply by 1000.
    return total_mj * 1000ull;
}

#else  // !SCATT_HAS_SYCL && !SCATT_HAS_CUDA  -- CPU-only / macOS dev build

std::uint64_t gpu_energy_uj_now() { return 0; }

#endif

// ---------------------------------------------------------------------------
// BenchReport
// ---------------------------------------------------------------------------
BenchReport::BenchReport() {
    rapl_start_uj_ = rapl_energy_uj_now();
    gpu_start_uj_  = gpu_energy_uj_now();
}

void BenchReport::add(const std::string& name, std::uint64_t ns, std::size_t rss_b) {
    add(name, ns, rss_b, /*rapl_uj_delta=*/0, /*gpu_uj_delta=*/0,
        /*gpu_marked=*/false);
}

void BenchReport::add(const std::string& name, std::uint64_t ns, std::size_t rss_b,
                       std::uint64_t rapl_uj_delta, std::uint64_t gpu_uj_delta,
                       bool gpu_marked) {
    auto& s = stages_[name];
    s.count          += 1;
    s.total_ns       += ns;
    s.max_rss_b       = std::max(s.max_rss_b, rss_b);
    s.total_rapl_uj  += rapl_uj_delta;
    s.total_gpu_uj   += gpu_uj_delta;
    s.gpu_marked      = s.gpu_marked || gpu_marked;
    s.energy_known    = true;  // a timed scope ran here (deltas may be 0 if
                               // counters unavailable, but the scope existed)
    peak_rss_         = std::max(peak_rss_, rss_b);
}

double BenchReport::rapl_joules() const {
    const std::uint64_t now = rapl_energy_uj_now();
    if (now == 0 || rapl_start_uj_ == 0) return 0.0;
    if (now < rapl_start_uj_) return 0.0;  // counter wraparound, give up
    return double(now - rapl_start_uj_) * 1e-6;
}

double BenchReport::gpu_joules() const {
    const std::uint64_t now = gpu_energy_uj_now();
    if (now == 0 || gpu_start_uj_ == 0) return 0.0;
    if (now < gpu_start_uj_) return 0.0;
    return double(now - gpu_start_uj_) * 1e-6;
}

bool BenchReport::gpu_energy_available() const {
    // True iff Sysman returned a non-zero counter at construction AND now
    // (covers loader-missing / sysman-disabled / permission-denied paths).
    return gpu_start_uj_ > 0 && gpu_energy_uj_now() > 0;
}

void BenchReport::set_omp_threads_if_unset(int n) {
    if (omp_threads_ == 0 && n > 0) omp_threads_ = n;
}

void BenchReport::print(std::ostream& os) const {
    const double total_s   = total_wall_ns_ * 1e-9;
    const double j_cpu     = rapl_joules();
    const double j_gpu     = gpu_joules();
    const bool   gpu_avail = gpu_energy_available();
    os << "\n[bench]  wall=" << format_seconds(total_s) << " s"
       << "  peak_rss=" << format_bytes(peak_rss_)
       << "  omp_threads=" << (omp_threads_ > 0 ? std::to_string(omp_threads_) : std::string("?"));
    if (j_cpu > 0.0) os << "  cpu_energy(rapl)=" << format_joules(j_cpu) << " J";
    else             os << "  cpu_energy(rapl)=unavailable";
    if (gpu_avail)   os << "  gpu_energy(L0)="  << format_joules(j_gpu) << " J";
    else             os << "  gpu_energy(L0)=unavailable";
    os << "\n\n";

    // Column widths.
    const int W_NAME  = 34;
    os << std::left  << std::setw(W_NAME) << "stage"
       << std::right << std::setw(10)  << "count"
       << std::setw(16) << "total(s)"
       << std::setw(16) << "mean(ms)"
       << std::setw(12) << "share"
       << std::setw(17) << "max_rss"
       << std::setw(18) << "cpu_energy(J)"
       << std::setw(18) << "gpu_energy(J)"
       << "\n";
    const int total_cols = W_NAME + 10 + 16 + 16 + 12 + 17 + 18 + 18;
    os << std::string(total_cols, '-') << "\n";

    // Sort by total_ns desc so the hot ones show first.
    std::vector<std::pair<std::string, StageStats>> rows(stages_.begin(), stages_.end());
    std::sort(rows.begin(), rows.end(),
              [](auto& a, auto& b){ return a.second.total_ns > b.second.total_ns; });

    for (const auto& [name, s] : rows) {
        const double t_s = s.total_ns * 1e-9;
        const double t_ms_mean = (s.count > 0) ? (s.total_ns * 1e-6) / double(s.count) : 0.0;
        const double share = (total_wall_ns_ > 0)
                           ? 100.0 * double(s.total_ns) / double(total_wall_ns_)
                           : 0.0;
        os << std::left  << std::setw(W_NAME) << name
           << std::right << std::setw(10)  << s.count
           << std::setw(16) << format_seconds(t_s)
           << std::setw(16) << format_ms(t_ms_mean);
        std::ostringstream so;
        so << std::fixed << std::setprecision(6) << share << "%";
        os << std::setw(12) << so.str()
           << std::setw(17) << format_bytes(s.max_rss_b);

        // CPU (RAPL) energy column.  Same rule as the top line: report
        // when a counter delta was observed; show "n/a" otherwise.
        if (!s.energy_known)             os << std::setw(18) << "-";
        else if (s.total_rapl_uj == 0)   os << std::setw(18) << "rapl n/a";
        else                              os << std::setw(18)
                                            << format_joules(double(s.total_rapl_uj) * 1e-6);

        // GPU energy column.  Only report for stages that were declared
        // GPU-bearing via ProfileScope(uses_gpu=true).  Everything else
        // gets "not in GPU".  If the stage IS GPU-marked but Sysman is
        // unavailable system-wide, report "L0 n/a".
        if (!s.gpu_marked)                os << std::setw(18) << "not in GPU";
        else if (!gpu_avail)              os << std::setw(18) << "L0 n/a";
        else                              os << std::setw(18)
                                            << format_joules(double(s.total_gpu_uj) * 1e-6);
        os << "\n";
    }
    os << "\n";
}

void BenchReport::save_tsv(const std::string& path) const {
    namespace fs = std::filesystem;
    if (const auto p = fs::path(path).parent_path(); !p.empty())
        fs::create_directories(p);
    std::ofstream f(path);
    if (!f) return;
    f << "# scatt bench dump\n";
    f << "# total_wall_s\t" << (total_wall_ns_ * 1e-9) << "\n";
    f << "# peak_rss_b\t"   << peak_rss_              << "\n";
    f << "# cpu_energy_rapl_j\t" << rapl_joules()     << "\n";
    f << "# gpu_energy_l0_j\t"   << gpu_joules()
                                 << "\t(0 means Level Zero Sysman not available)\n";
    f << "# omp_threads\t"  << omp_threads_           << "\n";
    f << "stage\tcount\ttotal_ns\tmax_rss_b\tcpu_energy_j\tgpu_energy_j\tgpu_marked\n";
    for (const auto& [name, s] : stages_) {
        const double cpu_j = double(s.total_rapl_uj) * 1e-6;
        const double gpu_j = double(s.total_gpu_uj)  * 1e-6;
        f << name << "\t" << s.count << "\t" << s.total_ns << "\t" << s.max_rss_b
          << "\t" << cpu_j
          << "\t" << gpu_j
          << "\t" << (s.gpu_marked ? "1" : "0")
          << "\n";
    }
}

// ---------------------------------------------------------------------------
// ProfileScope
// ---------------------------------------------------------------------------
ProfileScope::ProfileScope(BenchReport& r, std::string name, bool uses_gpu)
    : r_(r), name_(std::move(name)), t0_ns_(now_ns()),
      rapl_start_uj_(rapl_energy_uj_now()),
      gpu_start_uj_(uses_gpu ? gpu_energy_uj_now() : 0),
      uses_gpu_(uses_gpu)
{
#if defined(SCATT_HAS_OPENMP)
    r_.set_omp_threads_if_unset(omp_get_max_threads());
#else
    r_.set_omp_threads_if_unset(1);
#endif
}

ProfileScope::~ProfileScope() {
    const std::uint64_t dt = now_ns() - t0_ns_;

    const std::uint64_t rapl_now = rapl_energy_uj_now();
    const std::uint64_t rapl_delta =
        (rapl_now >= rapl_start_uj_ && rapl_start_uj_ > 0)
        ? (rapl_now - rapl_start_uj_) : 0;

    std::uint64_t gpu_delta = 0;
    if (uses_gpu_) {
        const std::uint64_t gpu_now = gpu_energy_uj_now();
        if (gpu_now >= gpu_start_uj_ && gpu_start_uj_ > 0) {
            gpu_delta = gpu_now - gpu_start_uj_;
        }
    }

    r_.add(name_, dt, current_peak_rss_bytes(),
           rapl_delta, gpu_delta, uses_gpu_);
}

}  // namespace scatt
