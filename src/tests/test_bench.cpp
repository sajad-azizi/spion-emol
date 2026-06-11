// test_bench.cpp -- BenchReport + ProfileScope basics.
//
// We test the bookkeeping (accumulation, count, print, tsv round-trip);
// the actual RSS + RAPL values are system-dependent, so we only sanity-
// check that peak_rss is non-zero and rapl_joules() returns a finite
// non-negative number.

#include "scatt/Bench.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using namespace scatt;

static int g_fail = 0;
static void check(bool cond, const std::string& what) {
    if (!cond) { std::cerr << "FAIL  " << what << "\n"; ++g_fail; }
    else       { std::cout << "ok    " << what << "\n"; }
}

int main() {
    BenchReport bench;

    {   ProfileScope _s(bench, "stage_a");
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
    {   ProfileScope _s(bench, "stage_a");                 // same stage twice
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
    {   ProfileScope _s(bench, "stage_b");
        std::this_thread::sleep_for(std::chrono::milliseconds(2)); }

    const auto& s = bench.stages();
    check(s.size() == 2, "two distinct stages recorded");
    check(s.at("stage_a").count == 2, "stage_a called twice");
    check(s.at("stage_b").count == 1, "stage_b called once");
    check(s.at("stage_a").total_ns > 10'000'000ull, "stage_a total ns > 10 ms");
    check(s.at("stage_a").total_ns > s.at("stage_b").total_ns,
          "stage_a longer than stage_b");

    check(current_peak_rss_bytes() > 0, "peak RSS reports something non-zero");
    check(bench.peak_rss() > 0,         "bench peak_rss updated via ProfileScope");

    const double j = bench.rapl_joules();
    check(j >= 0.0, "rapl_joules is non-negative");

    check(bench.omp_threads() >= 1, "omp_threads captured (>= 1)");

    // TSV round-trip: write, read back, find the two stages.
    const std::string tmpdir = fs::temp_directory_path().string() + "/scatt_bench_test";
    fs::create_directories(tmpdir);
    const std::string path = tmpdir + "/bench.dat";
    bench.set_total_wall_ns(20'000'000ull);
    bench.save_tsv(path);
    check(fs::exists(path), "tsv file written");
    std::ifstream in(path);
    std::string line;
    bool found_a = false, found_b = false;
    while (std::getline(in, line)) {
        if (line.rfind("stage_a\t", 0) == 0) found_a = true;
        if (line.rfind("stage_b\t", 0) == 0) found_b = true;
    }
    check(found_a && found_b, "both stages present in tsv");

    // Print shouldn't crash.
    std::ostringstream oss;
    bench.print(oss);
    const std::string out = oss.str();
    check(out.find("stage_a") != std::string::npos, "print includes stage_a");
    check(out.find("stage_b") != std::string::npos, "print includes stage_b");
    check(out.find("peak_rss") != std::string::npos, "print includes peak_rss");

    fs::remove_all(tmpdir);
    std::cout << "\n" << (g_fail == 0 ? "PASS" : "FAIL")
              << " bench  (" << g_fail << " failures)\n";
    return g_fail == 0 ? 0 : 1;
}
