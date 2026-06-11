// Bench.hpp -- lightweight instrumentation for the scattering pipeline.
//
// What it gives you:
//
//   * Per-stage wall time (accumulated over N calls).
//   * Peak resident set size (RSS) observed so far, portable across
//     macOS and Linux via getrusage(RUSAGE_SELF).
//   * Package-energy tally in Joules, via Intel RAPL
//     (/sys/class/powercap/intel-rapl:*/energy_uj) on Linux only; a silent
//     no-op on macOS or when the sysfs path is not readable.
//   * OpenMP thread count captured at first ProfileScope entry.
//
// Usage:
//
//   BenchReport bench;
//   {   ProfileScope _s(bench, "Potentials::build");
//       pot.build(...);
//   }
//   ...
//   bench.print(std::cout);
//   bench.save_tsv(path);   // tab-separated dump: one row per named stage
//
// A stage called multiple times is aggregated (count, total_time, max_rss).
// Calls to different stages nest freely (we don't try to subtract inner time
// from outer; the table reports each scope's wall clock independently so
// the user can sanity-check the sums).

#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace scatt {

struct StageStats {
    std::size_t   count          = 0;
    std::uint64_t total_ns       = 0;
    std::size_t   max_rss_b      = 0;   // peak RSS observed at any scope end
    // Cumulative energy counter deltas accumulated over this stage's
    // ProfileScope lifetimes.  CPU is via Intel RAPL (sysfs); GPU is via
    // Level Zero Sysman (PVC / Arc).  Both in microjoules; 0 means "not
    // measured or counter unavailable".
    std::uint64_t total_rapl_uj  = 0;
    std::uint64_t total_gpu_uj   = 0;
    bool          gpu_marked     = false;  // caller declared this stage uses GPU
    bool          energy_known   = false;  // a ProfileScope (with timing) ran on it
};

// Portable peak resident set size in bytes.
// Linux ru_maxrss is in kilobytes, macOS in bytes.
std::size_t current_peak_rss_bytes();

// Current package energy counter in microjoules, summed across all RAPL
// packages. Returns 0 if not available.
std::uint64_t rapl_energy_uj_now();

// Current GPU energy counter in microjoules, summed across all Level Zero
// power domains on every visible device (PVC has one per tile = 2 domains).
// Returns 0 when:
//   * the SYCL build is not enabled (no Level Zero loader linked);
//   * ZES_ENABLE_SYSMAN was not set before the first ze*Init call;
//   * the kernel's Level Zero driver does not expose sysman energy.
// On its first call it lazily initialises Sysman and caches power-domain
// handles -- subsequent calls are just N counter reads, microsecond cost.
std::uint64_t gpu_energy_uj_now();

class BenchReport {
public:
    BenchReport();

    // Record a stage completion.  Legacy two-arg form (no energy data) for
    // internal sub-stage timers that come from per-stage `Stats` structs.
    void add(const std::string& name, std::uint64_t ns, std::size_t rss_b);

    // Full add: includes per-scope RAPL + GPU energy deltas (microjoules)
    // and a `gpu_marked` flag declaring whether the caller intended this
    // stage to run on the GPU.  Used by ProfileScope; not normally called
    // directly.
    void add(const std::string& name, std::uint64_t ns, std::size_t rss_b,
             std::uint64_t rapl_uj_delta, std::uint64_t gpu_uj_delta,
             bool gpu_marked);

    // Totals.
    std::uint64_t total_wall_ns() const { return total_wall_ns_; }
    std::size_t   peak_rss()      const { return peak_rss_; }
    double        rapl_joules()   const;   // delta since construction
    double        gpu_joules()    const;   // delta since construction (PVC/Arc)
    bool          gpu_energy_available() const;
    int           omp_threads()   const { return omp_threads_; }

    // Called by the first ProfileScope to latch OMP thread count.
    void set_omp_threads_if_unset(int n);

    // Accumulate wall-total from an outer timer at top level (optional).
    void set_total_wall_ns(std::uint64_t v) { total_wall_ns_ = v; }

    // Formatted table to `os`. Columns:
    //   stage | count | total_s | mean_ms | share_of_total
    void print(std::ostream& os) const;

    // Tab-separated dump (stage TAB count TAB ns TAB max_rss_b).
    void save_tsv(const std::string& path) const;

    const std::map<std::string, StageStats>& stages() const { return stages_; }

private:
    std::map<std::string, StageStats> stages_;
    std::size_t    peak_rss_           = 0;
    std::uint64_t  total_wall_ns_      = 0;
    std::uint64_t  rapl_start_uj_      = 0;
    std::uint64_t  gpu_start_uj_       = 0;
    int            omp_threads_        = 0;
};

// RAII scope guard: starts timer at construction, records on destruction.
//
// `uses_gpu` is a caller-supplied declaration: pass true when the scope
// is expected to drive work on the SYCL GPU (FRP/BP/Sinv with --use-gpu,
// kernels in GpuPropagate, etc.).  When true, the destructor records the
// GPU energy delta (Level Zero Sysman) into the stage's StageStats and
// flags it `gpu_marked`.  When false, the GPU column reports "not in GPU"
// for that stage in the bench printout.
class ProfileScope {
public:
    ProfileScope(BenchReport& r, std::string name, bool uses_gpu = false);
    ~ProfileScope();
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;

private:
    BenchReport&  r_;
    std::string   name_;
    std::uint64_t t0_ns_;
    std::uint64_t rapl_start_uj_;
    std::uint64_t gpu_start_uj_;
    bool          uses_gpu_;
};

}  // namespace scatt
