#include "RunDirs.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace sph3d {

// FNV-1a 64-bit hash; deterministic, no external dependency.
static std::uint64_t fnv1a64(const void* data, std::size_t n) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// Hash a string segment.
static std::uint64_t hash_str(std::uint64_t seed, const std::string& s) {
    const std::uint64_t v = fnv1a64(s.data(), s.size());
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

// Hash a double (bit-exact: identical doubles yield identical hash).
static std::uint64_t hash_double(std::uint64_t seed, double x) {
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return seed ^ (bits + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

static std::uint64_t hash_int(std::uint64_t seed, int x) {
    const std::uint64_t v = static_cast<std::uint32_t>(x);
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

std::string molhash(const RunConfig& c) {
    // Mix in a version tag so that future schema changes invalidate
    // old caches automatically.
    std::uint64_t h = fnv1a64("sph3d-molhash-v1", 16);
    h = hash_str   (h, c.kind);
    h = hash_double(h, c.V0);
    h = hash_double(h, c.L_box);
    h = hash_double(h, c.a_gauss);
    h = hash_double(h, c.b_gauss);
    h = hash_double(h, c.c_gauss);
    h = hash_double(h, c.a_soft);
    h = hash_double(h, c.R_h2);
    h = hash_double(h, c.a_h2);
    h = hash_int   (h, c.N_grid);
    h = hash_double(h, c.dr);
    h = hash_int   (h, c.l_max);
    h = hash_int   (h, c.N_theta);
    h = hash_int   (h, c.N_phi);

    std::ostringstream os;
    os << std::hex << std::setw(16) << std::setfill('0') << h;
    return os.str();
}

RunDirs resolve_dirs(const std::string& cli_work,
                     const std::string& cli_scratch)
{
    RunDirs d;
    if (!cli_work.empty()) {
        d.work = cli_work;
    } else if (const char* env = std::getenv("WORK")) {
        d.work = env;
    } else {
        d.work = "./work";
    }
    if (!cli_scratch.empty()) {
        d.scratch = cli_scratch;
    } else if (const char* env = std::getenv("SCRATCH")) {
        d.scratch = env;
    } else {
        d.scratch = "./scratch";
    }
    ensure_dir(d.work);
    ensure_dir(d.scratch);
    return d;
}

void ensure_dir(const std::string& path) {
    namespace fs = std::filesystem;
    if (path.empty()) return;
    std::error_code ec;
    fs::create_directories(path, ec);
    // Don't throw on race; just verify the dir exists.
}

}  // namespace sph3d
