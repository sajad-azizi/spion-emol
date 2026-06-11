#include "BasisCache.hpp"
#include "Common.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mc_tdse {

namespace {

constexpr std::uint32_t kMagic   = 0x4D435450u;   // 'MCTP'
constexpr std::uint32_t kVersion = 1;

// FNV-1a 64-bit.  Stable, simple, good distribution for short strings.
std::uint64_t fnv1a_64(const std::string& s) {
    constexpr std::uint64_t offset = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t prime  = 0x100000001b3ULL;
    std::uint64_t h = offset;
    for (unsigned char c : s) {
        h ^= c;
        h *= prime;
    }
    return h;
}

template <typename T>
void write_pod(std::ostream& os, T v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
T read_pod(std::istream& is) {
    T v{};
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
}

void write_doubles(std::ostream& os, const double* p, std::size_t n) {
    os.write(reinterpret_cast<const char*>(p), n * sizeof(double));
}
void read_doubles(std::istream& is, double* p, std::size_t n) {
    is.read(reinterpret_cast<char*>(p), n * sizeof(double));
}

void write_matrix(std::ostream& os, const Eigen::MatrixXd& m) {
    write_pod<std::int64_t>(os, m.rows());
    write_pod<std::int64_t>(os, m.cols());
    // Eigen MatrixXd is column-major by default — write contiguous data.
    write_doubles(os, m.data(), static_cast<std::size_t>(m.size()));
}
Eigen::MatrixXd read_matrix(std::istream& is) {
    const auto rows = read_pod<std::int64_t>(is);
    const auto cols = read_pod<std::int64_t>(is);
    Eigen::MatrixXd m(rows, cols);
    read_doubles(is, m.data(), static_cast<std::size_t>(m.size()));
    return m;
}

void write_matrix_cd(std::ostream& os, const Eigen::MatrixXcd& m) {
    write_pod<std::int64_t>(os, m.rows());
    write_pod<std::int64_t>(os, m.cols());
    // Two passes: real then imag (avoids std::complex layout assumption).
    Eigen::MatrixXd re = m.real();
    Eigen::MatrixXd im = m.imag();
    write_doubles(os, re.data(), static_cast<std::size_t>(re.size()));
    write_doubles(os, im.data(), static_cast<std::size_t>(im.size()));
}
Eigen::MatrixXcd read_matrix_cd(std::istream& is) {
    const auto rows = read_pod<std::int64_t>(is);
    const auto cols = read_pod<std::int64_t>(is);
    Eigen::MatrixXd re(rows, cols), im(rows, cols);
    read_doubles(is, re.data(), static_cast<std::size_t>(re.size()));
    read_doubles(is, im.data(), static_cast<std::size_t>(im.size()));
    Eigen::MatrixXcd out(rows, cols);
    out.real() = re;
    out.imag() = im;
    return out;
}

}  // namespace

std::string basis_cache_canonical_string(const std::vector<int>& block_MFs,
                                         const std::vector<BlockBuildOptions>& opts,
                                         double B_gauss) {
    std::ostringstream os;
    os.setf(std::ios::scientific);
    os.precision(17);
    os << "# multichannel_tdse basis cache key\n"
       << "version=1\n"
       << "B_gauss=" << B_gauss << "\n"
       << "n_blocks=" << block_MFs.size() << "\n";
    for (std::size_t k = 0; k < block_MFs.size(); ++k) {
        const auto& o = opts[k];
        os << "[block " << k << "]\n"
           << "M_F=" << block_MFs[k] << "\n"
           << "B_gauss=" << o.B_gauss << "\n"
           << "V_T_GHz=" << o.V_T_GHz << "\n"
           << "V_S_over_T=" << o.V_S_over_T << "\n"
           << "r0_a0=" << o.r0_a0 << "\n"
           << "N_ch_keep=" << o.N_ch_keep << "\n"
           << "N_grid=" << o.N_grid << "\n"
           << "dr_a0=" << o.dr_a0 << "\n"
           << "p_init=" << o.p_init << "\n"
           << "E_max_kHz_above_threshold=" << o.E_max_kHz_above_threshold << "\n"
           << "E_window_kHz_lo=" << o.E_window_kHz_lo << "\n"
           << "E_window_kHz_hi=" << o.E_window_kHz_hi << "\n"
           << "use_analytic_halo=" << (o.use_analytic_halo ? 1 : 0) << "\n";
    }
    return os.str();
}

std::string basis_cache_key(const std::vector<int>& block_MFs,
                             const std::vector<BlockBuildOptions>& opts,
                             double B_gauss) {
    const std::string canon = basis_cache_canonical_string(block_MFs, opts, B_gauss);
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(fnv1a_64(canon)));
    return std::string(buf, 16);
}

bool try_load_pooled_basis(PooledBasis* out,
                           const std::string& cache_dir,
                           const std::string& key,
                           const std::string& canonical_string) {
    if (!out) return false;
    namespace fs = std::filesystem;
    const fs::path bin  = fs::path(cache_dir) / (key + ".bin");
    const fs::path meta = fs::path(cache_dir) / (key + ".meta.txt");
    if (!fs::exists(bin)) {
        std::fprintf(stderr, "[basis_cache] miss: %s does not exist\n", bin.c_str());
        return false;
    }
    if (!fs::exists(meta)) {
        std::fprintf(stderr, "[basis_cache] miss: %s does not exist\n", meta.c_str());
        return false;
    }

    // Verify meta matches request exactly (defense against hash collision).
    std::ifstream mf(meta);
    std::stringstream buf;
    buf << mf.rdbuf();
    if (buf.str() != canonical_string) {
        std::fprintf(stderr, "[basis_cache] miss: meta.txt mismatch "
                     "(file %zu bytes, request %zu bytes)\n",
                     buf.str().size(), canonical_string.size());
        // Find first differing byte for diagnostics.
        const std::string& a = buf.str();
        const std::string& b = canonical_string;
        const size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; ++i) {
            if (a[i] != b[i]) {
                std::fprintf(stderr, "[basis_cache] first diff at byte %zu: "
                             "file='%c'(0x%02x) request='%c'(0x%02x)\n",
                             i, a[i], (unsigned char)a[i],
                             b[i], (unsigned char)b[i]);
                break;
            }
        }
        return false;
    }

    std::ifstream is(bin, std::ios::binary);
    if (!is.good()) {
        std::fprintf(stderr, "[basis_cache] miss: cannot open %s\n", bin.c_str());
        return false;
    }
    // Verify file size matches the expected basis size before risking
    // the full read.  An incomplete cache (interrupted write, partial
    // copy, etc.) would bypass the magic check and only show up when
    // mid-stream reads fail.
    const auto bin_size = fs::file_size(bin);

    try {
        const auto magic   = read_pod<std::uint32_t>(is);
        const auto version = read_pod<std::uint32_t>(is);
        if (magic != kMagic || version != kVersion) {
            std::fprintf(stderr,
                "[basis_cache] miss: bad header magic=0x%x (expect 0x%x) "
                "version=%u (expect %u)\n",
                magic, kMagic, version, kVersion);
            return false;
        }

        const auto n_blocks = read_pod<std::int32_t>(is);
        if (n_blocks <= 0 || n_blocks > 16) {
            std::fprintf(stderr,
                "[basis_cache] miss: implausible n_blocks=%d -- truncated file?\n",
                n_blocks);
            return false;
        }
        // Pre-compute expected total size from headers as we walk; we
        // bail with a clear message if any read crosses the file end.
        std::uint64_t expected_size_so_far = 8;   // magic + version
        expected_size_so_far += 4;                // n_blocks
        out->block_MFs.assign(n_blocks, 0);
        out->blocks.resize(n_blocks);
        out->E_th_au.resize(n_blocks);
        for (int k = 0; k < n_blocks; ++k) {
            BlockEigenstates& b = out->blocks[k];
            b.M_F    = read_pod<std::int32_t>(is);
            b.N_ch   = read_pod<std::int32_t>(is);
            b.N_grid = read_pod<std::int32_t>(is);
            b.dr     = read_pod<double>(is);
            const auto n_states = read_pod<std::int32_t>(is);
            const double E_th   = read_pod<double>(is);
            // Sanity bounds: any of these out of range means we read junk.
            if (n_states < 0 || n_states > 1'000'000 ||
                b.N_grid <= 0 || b.N_grid > 100'000'000 ||
                b.N_ch   <= 0 || b.N_ch   > 32) {
                std::fprintf(stderr,
                    "[basis_cache] miss: block %d header garbage "
                    "(N_grid=%d N_ch=%d n_states=%d)\n",
                    k, b.N_grid, b.N_ch, n_states);
                return false;
            }
            const std::uint64_t per_state = 16  // rows/cols header
                + static_cast<std::uint64_t>(b.N_grid) * b.N_ch * sizeof(double);
            const std::uint64_t this_block_payload =
                  4*5 + 8*2          // 5 ints + 2 doubles header
                + 8 * n_states       // E_au[]
                + per_state * n_states;
            expected_size_so_far += this_block_payload;
            if (expected_size_so_far > bin_size) {
                std::fprintf(stderr,
                    "[basis_cache] miss: file too small "
                    "(have %llu B, need %llu B at block %d) -- truncated cache\n",
                    (unsigned long long)bin_size,
                    (unsigned long long)expected_size_so_far, k);
                return false;
            }
            out->block_MFs[k] = b.M_F;
            out->E_th_au[k]   = E_th;
            b.E_au.resize(n_states);
            read_doubles(is, b.E_au.data(), n_states);
            b.u.resize(n_states);
            for (int n = 0; n < n_states; ++n) b.u[n] = read_matrix(is);
            if (!is.good()) {
                std::fprintf(stderr,
                    "[basis_cache] miss: read failed in block %d -- truncated cache\n", k);
                return false;
            }
        }

        const auto n_pairs = read_pod<std::int32_t>(is);
        if (n_pairs < 0 || n_pairs > 16) {
            std::fprintf(stderr,
                "[basis_cache] miss: implausible n_pairs=%d -- truncated file?\n",
                n_pairs);
            return false;
        }
        out->d_plus_pair.resize(n_pairs);
        for (int p = 0; p < n_pairs; ++p) out->d_plus_pair[p] = read_matrix_cd(is);
        if (!is.good()) {
            std::fprintf(stderr, "[basis_cache] miss: read failed in dipole pairs\n");
            return false;
        }

        // Reconstruct pooled-index lookup tables.
        out->E_au.clear();
        out->E_au_block_rel.clear();
        out->of_block.clear();
        out->n_in_block.clear();
        out->block_offset.clear();
        out->block_offset.push_back(0);
        for (int k = 0; k < n_blocks; ++k) {
            const auto& b = out->blocks[k];
            for (int n = 0; n < b.n_states(); ++n) {
                out->E_au.push_back(b.E_au[n]);
                out->E_au_block_rel.push_back(b.E_au[n] - out->E_th_au[k]);
                out->of_block.push_back(k);
                out->n_in_block.push_back(n);
            }
            out->block_offset.push_back(out->block_offset.back() + b.n_states());
        }
        out->N_total = out->block_offset.back();
        return is.good();
    } catch (...) {
        return false;
    }
}

void save_pooled_basis(const PooledBasis& pb,
                       const std::string& cache_dir,
                       const std::string& key,
                       const std::string& canonical_string) {
    namespace fs = std::filesystem;
    fs::create_directories(cache_dir);
    const fs::path bin  = fs::path(cache_dir) / (key + ".bin");
    const fs::path meta = fs::path(cache_dir) / (key + ".meta.txt");
    // Atomic write: serialize to .tmp paths, fsync, close, then rename.
    // This way an interrupted run leaves no half-written .bin -- the
    // reader sees either the old (good) cache or none, never garbage.
    const fs::path bin_tmp  = fs::path(cache_dir) / (key + ".bin.tmp");
    const fs::path meta_tmp = fs::path(cache_dir) / (key + ".meta.txt.tmp");

    {
        std::ofstream mf(meta_tmp);
        mf << canonical_string;
    }
    std::ofstream os(bin_tmp, std::ios::binary);
    if (!os.good())
        throw std::runtime_error("save_pooled_basis: cannot open " + bin_tmp.string());

    write_pod<std::uint32_t>(os, kMagic);
    write_pod<std::uint32_t>(os, kVersion);
    write_pod<std::int32_t>(os, static_cast<std::int32_t>(pb.blocks.size()));
    for (std::size_t k = 0; k < pb.blocks.size(); ++k) {
        const BlockEigenstates& b = pb.blocks[k];
        write_pod<std::int32_t>(os, b.M_F);
        write_pod<std::int32_t>(os, b.N_ch);
        write_pod<std::int32_t>(os, b.N_grid);
        write_pod<double>     (os, b.dr);
        write_pod<std::int32_t>(os, b.n_states());
        write_pod<double>     (os, pb.E_th_au[k]);
        write_doubles(os, b.E_au.data(), b.E_au.size());
        for (const auto& u : b.u) write_matrix(os, u);
    }
    write_pod<std::int32_t>(os, static_cast<std::int32_t>(pb.d_plus_pair.size()));
    for (const auto& d : pb.d_plus_pair) write_matrix_cd(os, d);

    // Force the bytes to disk before atomic rename.  std::ofstream's
    // destructor flushes, but we want to be SURE before the rename
    // (a crash between rename and write would leave a "good-looking"
    // empty file, which the truncation check above would still
    // catch).
    os.flush();
    os.close();
    if (!os.good())
        throw std::runtime_error("save_pooled_basis: write failed for " + bin_tmp.string());

    // Now atomically replace the .bin and .meta.txt at their final paths.
    fs::rename(bin_tmp,  bin);
    fs::rename(meta_tmp, meta);
}

}  // namespace mc_tdse
