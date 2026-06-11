#include "scatt/PotentialStorage.hpp"

#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
#include <omp.h>
#include <string>
#endif

namespace scatt {

namespace {
    constexpr const char* kMetaName      = "metadata.bin";
    constexpr const char* kChunkPrefix   = "pot_chunk_";
    constexpr const char* kChunkSuffix   = ".bin";
    constexpr const char* kManifestName  = "manifest.txt";
    constexpr const char* kSuccessName   = "__SUCCESS__";

    // Magic prefix for v2 (symmetric-aware) metadata.bin.
    // Old (v1) metadata starts with size_t Nr at offset 0.  Our chosen
    // value is large enough that no realistic Nr could collide with it
    // (Nr is at most ~10^5; this magic == 0x53594D4D ≈ 1.4e9).  Reading
    // the first 4 bytes unambiguously distinguishes v1 vs v2.
    constexpr std::int32_t kSymStorageMagic   = 0x53594D4D;  // 'SYMM'
    constexpr std::int32_t kSymStorageVersion = 2;

    // Pack the LOWER triangle of a column-major Eigen::MatrixXd into a
    // contiguous buffer.  out must have at least N*(N+1)/2 doubles.
    // Layout in `out`: column 0 lower (N entries) | col 1 lower (N-1) | ...
    inline void pack_lower_(const Eigen::MatrixXd& M, double* out)
    {
        const int N = static_cast<int>(M.rows());
        std::size_t pos = 0;
        for (int j = 0; j < N; ++j) {
            // Column j's lower triangle starts at row j; in column-major
            // storage that's contiguous at &M(j, j), length N - j.
            const double* col_j_lo = &M(j, j);
            const std::size_t len  = static_cast<std::size_t>(N - j);
            std::memcpy(out + pos, col_j_lo, len * sizeof(double));
            pos += len;
        }
    }

    // Unpack a packed-lower buffer into a full N×N column-major MatrixXd
    // and reconstruct the upper triangle by reflection (M(i,j) := M(j,i)
    // for i < j).  M must already be sized (N, N).
    inline void unpack_lower_and_reflect_(const double* in, Eigen::MatrixXd& M)
    {
        const int N = static_cast<int>(M.rows());
        std::size_t pos = 0;
        for (int j = 0; j < N; ++j) {
            double* col_j_lo = &M(j, j);
            const std::size_t len = static_cast<std::size_t>(N - j);
            std::memcpy(col_j_lo, in + pos, len * sizeof(double));
            pos += len;
        }
        // Mirror lower -> upper: M(i, j) := M(j, i) for i < j.  The
        // diagonal is already correct; we touch only strict-upper bytes.
        for (int j = 1; j < N; ++j) {
            for (int i = 0; i < j; ++i) {
                M(i, j) = M(j, i);
            }
        }
    }

    // packed-lower size (doubles) for an N×N matrix.
    constexpr std::size_t packed_lower_doubles_(int N) {
        return static_cast<std::size_t>(N) * static_cast<std::size_t>(N + 1) / 2;
    }

    // ----------------------------------------------------------------------
    // Helpers used by the OPT-IN parallel chunk writer.
    // The serial write path does NOT use any of these -- its behaviour is
    // unchanged and bit-equivalent to all previous releases.
    // ----------------------------------------------------------------------

    // RAII wrapper for a POSIX fd: ensures close() on scope exit even on
    // exception.  Move-only.
    class unique_fd {
    public:
        explicit unique_fd(int fd = -1) noexcept : fd_(fd) {}
        unique_fd(const unique_fd&)            = delete;
        unique_fd& operator=(const unique_fd&) = delete;
        unique_fd(unique_fd&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
        unique_fd& operator=(unique_fd&& o) noexcept {
            if (this != &o) {
                if (fd_ >= 0) ::close(fd_);
                fd_ = o.fd_; o.fd_ = -1;
            }
            return *this;
        }
        ~unique_fd() { if (fd_ >= 0) ::close(fd_); }
        int get()     const noexcept { return fd_; }
        int release()       noexcept { int t = fd_; fd_ = -1; return t; }
    private:
        int fd_;
    };

    inline off_t checked_off(std::uint64_t x) {
        if (x > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
            throw std::runtime_error("PotentialStorage: file offset overflow");
        }
        return static_cast<off_t>(x);
    }

    // pwrite() at non-overlapping offsets is thread-safe per POSIX.
    // Loops on EINTR and short writes (some kernels short-write at
    // SSIZE_MAX or 2 GB, similar to the read-side cap we already
    // mitigate with pread_full).
    inline void pwrite_all(int fd, const void* buf, std::size_t n,
                           off_t offset, const std::string& fn)
    {
        const char* p = static_cast<const char*>(buf);
        while (n > 0) {
            const std::size_t step =
                std::min<std::size_t>(n, static_cast<std::size_t>(SSIZE_MAX));
            const ssize_t got = ::pwrite(fd, p, step, offset);
            if (got < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(
                    "PotentialStorage: pwrite error on " + fn +
                    ": " + std::strerror(errno));
            }
            if (got == 0) {
                throw std::runtime_error(
                    "PotentialStorage: zero-byte pwrite on " + fn +
                    " (disk full?)");
            }
            p      += got;
            offset += got;
            n      -= static_cast<std::size_t>(got);
        }
    }

    // close() the fd and surface any error.  POSIX requires close()
    // success-checking for cases where data was buffered in the fd's
    // page-cache flush path and the flush failed.
    inline void close_checked(unique_fd& fd, const std::string& fn) {
        const int raw = fd.release();
        if (raw >= 0 && ::close(raw) != 0) {
            throw std::runtime_error(
                "PotentialStorage: close error on " + fn +
                ": " + std::strerror(errno));
        }
    }

    // fsync the parent directory to make a rename() durable.  Required
    // for crash-safe atomic publication on most POSIX filesystems.
    inline void fsync_parent_dir(const std::string& path) {
        namespace fs = std::filesystem;
        fs::path p(path);
        fs::path dir = p.parent_path();
        if (dir.empty()) dir = ".";
        unique_fd dfd(::open(dir.c_str(), O_RDONLY | O_DIRECTORY));
        if (dfd.get() < 0) {
            throw std::runtime_error(
                "PotentialStorage: cannot open parent directory of " + path +
                ": " + std::strerror(errno));
        }
        if (::fsync(dfd.get()) != 0) {
            throw std::runtime_error(
                "PotentialStorage: fsync parent directory failed for " + path +
                ": " + std::strerror(errno));
        }
    }
}

// =============================================================================
// Setup
// =============================================================================
PotentialStorage::~PotentialStorage() {
    // Ensure any in-flight prefetch thread is joined before destruction
    // (otherwise std::thread's destructor would std::terminate the process).
    wait_prefetch_();
}

void PotentialStorage::clear() {
    // Join any in-flight prefetch worker before tearing down its target
    // buffer.  Otherwise the thread could pread into a freed buffer.
    wait_prefetch_();
    prefetched_buffer_.clear();  prefetched_buffer_.shrink_to_fit();
    prefetched_chunk_idx_ = -1;
    prefetched_count_     = 0;
    prefetch_error_       = false;
    prefetch_error_msg_.clear();

    memory_storage_.clear();  memory_storage_.shrink_to_fit();
    write_buffer_.clear();    write_buffer_.shrink_to_fit();
    read_buffer_.clear();     read_buffer_.shrink_to_fit();

    initialized_              = false;
    read_ready_               = false;
    write_mode_               = false;
    n_grid_                   = 0;
    channels_                 = 0;
    bytes_per_matrix_         = 0;
    symmetric_storage_        = false;
    bytes_per_matrix_disk_    = 0;
    parallel_chunk_write_     = false;
    num_chunks_               = 0;
    write_buffer_count_       = 0;
    num_chunks_written_       = 0;
    last_ir_written_.store(-1);
    read_buffer_start_ir_     = -1;
    read_buffer_end_ir_       = -1;
    cached_ir_                = -1;
    checkpoint_dir_.clear();
}

void PotentialStorage::initialize_for_write(std::size_t Nr, int channels, Mode mode,
                                            const std::string& checkpoint_dir,
                                            int chunk_size,
                                            bool symmetric_storage,
                                            bool parallel_chunk_write)
{
    clear();
    // symmetric_storage only affects the on-disk path; MEMORY mode ignores it.
    // For DISK mode it halves the on-disk bytes per matrix.  The in-memory
    // representation (write_buffer_, read_buffer_) is ALWAYS full N×N.
    symmetric_storage_     = (mode == Mode::DISK) ? symmetric_storage : false;
    parallel_chunk_write_  = (mode == Mode::DISK) ? parallel_chunk_write : false;
    bytes_per_matrix_disk_ = symmetric_storage_
        ? packed_lower_doubles_(channels) * sizeof(double)
        : static_cast<std::size_t>(channels) * channels * sizeof(double);
    n_grid_           = Nr;
    channels_         = channels;
    mode_             = mode;
    write_mode_       = true;
    bytes_per_matrix_ = static_cast<std::size_t>(channels) * channels * sizeof(double);

    if (mode_ == Mode::MEMORY) {
        // Pre-allocate every slot so that parallel store() calls touch
        // disjoint already-constructed slots -- no vector resize during
        // the parallel phase, which would be a race.
        memory_storage_.resize(Nr);
        for (std::size_t ir = 0; ir < Nr; ++ir)
            memory_storage_[ir] = Eigen::MatrixXd::Zero(channels, channels);
        read_ready_ = true;  // reads allowed immediately in MEMORY
        std::cout << "[PotentialStorage] MEMORY for WRITE  Nr=" << Nr
                  << "  channels=" << channels
                  << "  footprint=" << (memory_bytes() / (1024 * 1024)) << " MB\n";
    } else {
        if (checkpoint_dir.empty())
            throw std::runtime_error("PotentialStorage DISK mode requires checkpoint_dir");
        checkpoint_dir_ = checkpoint_dir;
        chunk_size_     = chunk_size;
        num_chunks_     = static_cast<int>((Nr + chunk_size - 1) / chunk_size);
        std::filesystem::create_directories(checkpoint_dir_);

        write_buffer_.resize(chunk_size_);
        for (int i = 0; i < chunk_size_; ++i)
            write_buffer_[i] = Eigen::MatrixXd(channels, channels);
        write_buffer_count_ = 0;
        num_chunks_written_ = 0;
        last_ir_written_.store(-1);

        const double chunk_mb = static_cast<double>(chunk_size_) * bytes_per_matrix_ / (1024.0 * 1024.0);
        const double total_mb_disk =
            static_cast<double>(Nr) * bytes_per_matrix_disk_ / (1024.0 * 1024.0);
        std::cout << "[PotentialStorage] DISK for WRITE  Nr=" << Nr
                  << "  channels=" << channels
                  << (symmetric_storage_ ? "  (symmetric: lower triangle on disk)" : "")
                  << "\n  checkpoint_dir=" << checkpoint_dir_
                  << "\n  chunk_size="     << chunk_size_
                  << "  (" << num_chunks_ << " chunks)"
                  << "\n  buffer=" << chunk_mb << " MB"
                  << "  total_disk=" << total_mb_disk << " MB\n";
    }
    initialized_ = true;
}

bool PotentialStorage::initialize_from_checkpoint(std::size_t Nr, int channels,
                                                  const std::string& checkpoint_dir,
                                                  int chunk_size)
{
    clear();
    checkpoint_dir_ = checkpoint_dir;
    chunk_size_     = chunk_size;
    // Integrity gate: require __SUCCESS__ marker AND matching manifest.
    // If either is missing/wrong, this checkpoint is incomplete or from a
    // different run -- reject and let the caller rebuild.
    if (!verify_manifest_and_success(checkpoint_dir)) {
        std::cout << "[PotentialStorage] rejected checkpoint (no SUCCESS or "
                     "manifest mismatch): " << checkpoint_dir << "\n";
        clear();
        return false;
    }
    // Save the runtime-budgeted chunk_size before read_metadata overwrites it.
    const int runtime_budget_cs = chunk_size_;

    // Crash-recovery: if a previous re-chunk crashed during commit, the
    // on-disk layout may be half-old / half-new (some chunks at the old
    // size, some at the new size, indistinguishable by file name alone).
    // The renormalize_chunks_to_() commit phase creates a marker file
    // before invalidating __SUCCESS__ and removes the marker only after
    // a clean swap.  Presence of the marker after the verify_success
    // check above means we got past __SUCCESS__ creation but didn't
    // finish the cleanup -- still safe to load, since __SUCCESS__ is the
    // last step.  But if __SUCCESS__ is ABSENT and the marker is present,
    // verify_manifest_and_success already rejected above.  So in this
    // branch the marker is purely informational; clean it up.
    {
        const std::string marker = checkpoint_dir + "/_rechunk_committing";
        if (std::filesystem::exists(marker)) {
            std::error_code ec;
            std::filesystem::remove(marker, ec);
        }
    }

    if (!read_metadata(Nr, channels)) return false;

    // Bytes-per-record (needed by both the cross-size check below and
    // the renormalize path if it fires).  Must be set BEFORE
    // renormalize_chunks_to_() since that method reads it.
    bytes_per_matrix_ = static_cast<std::size_t>(channels) * channels * sizeof(double);
    bytes_per_matrix_disk_ = symmetric_storage_
        ? packed_lower_doubles_(channels) * sizeof(double)
        : bytes_per_matrix_;
    n_grid_     = Nr;
    channels_   = channels;
    mode_       = Mode::DISK;
    write_mode_ = false;

    // Cross-node-size safety: if the on-disk chunk_size exceeds what the
    // planner budgeted for this stage on this node, the read_buffer would
    // exceed RAM and OOM at first read.  Two policies:
    //   chunk_rechunk_allowed_ == true (default): atomically re-write
    //     the checkpoint at the smaller runtime chunk_size; the bytes
    //     of each ir record are preserved exactly (test_storage_chunk_
    //     rechunk gates this).  Single-shot disk-I/O cost (~minutes)
    //     vs full compute rebuild (~hours).
    //   chunk_rechunk_allowed_ == false: reject the checkpoint (legacy
    //     pre-2026-05-23 behaviour) and let the caller rebuild.
    if (runtime_budget_cs > 0 && chunk_size_ > runtime_budget_cs) {
        if (chunk_rechunk_allowed_) {
            try {
                num_chunks_ = static_cast<int>(
                    (Nr + chunk_size_ - 1) / chunk_size_);   // OLD num_chunks
                renormalize_chunks_to_(runtime_budget_cs);
                // chunk_size_ + num_chunks_ now updated to NEW values by
                // renormalize_chunks_to_().
            } catch (const std::exception& e) {
                std::cout << "[PotentialStorage] rejected checkpoint at "
                          << checkpoint_dir
                          << ": re-chunk failed: " << e.what()
                          << "\n  (caller will rebuild; "
                          << "leftover _rechunk_pending/ may exist for forensics)\n";
                clear();
                return false;
            }
        } else {
            std::cout << "[PotentialStorage] rejected checkpoint at "
                      << checkpoint_dir
                      << ": on-disk chunk_size=" << chunk_size_
                      << " exceeds runtime budget chunk_size=" << runtime_budget_cs
                      << "  (would over-allocate read_buffer; rebuild with "
                         "smaller chunks, or run WITHOUT "
                         "--no-checkpoint-rechunk to re-chunk in place)\n";
            clear();
            return false;
        }
    }

    num_chunks_ = static_cast<int>((Nr + chunk_size_ - 1) / chunk_size_);

    for (int i = 0; i < num_chunks_; ++i) {
        if (!std::filesystem::exists(chunk_filename(i))) {
            std::cout << "[PotentialStorage] missing chunk file " << i
                      << " in " << checkpoint_dir_ << "\n";
            clear();
            return false;
        }
    }

    read_buffer_.resize(chunk_size_);
    for (int i = 0; i < chunk_size_; ++i)
        read_buffer_[i] = Eigen::MatrixXd(channels_, channels_);
    cached_matrix_ = Eigen::MatrixXd(channels_, channels_);
    read_buffer_start_ir_ = -1;
    read_buffer_end_ir_   = -1;
    cached_ir_            = -1;

    initialized_ = true;
    read_ready_  = true;
    std::cout << "[PotentialStorage] DISK from CHECKPOINT  Nr=" << Nr
              << "  channels=" << channels
              << (symmetric_storage_ ? "  (symmetric on-disk)" : "")
              << "  chunks=" << num_chunks_ << "\n";
    return true;
}

// =============================================================================
// Writing
// =============================================================================
void PotentialStorage::store(std::size_t ir, const Eigen::MatrixXd& matrix)
{
    if (!initialized_ || !write_mode_)
        throw std::runtime_error("PotentialStorage::store before initialize_for_write");
    if (ir >= n_grid_)
        throw std::runtime_error("PotentialStorage::store ir >= N_grid");

    if (mode_ == Mode::MEMORY) {
        // THREAD-SAFE: distinct ir = distinct slot. Each thread mutates its
        // own element. The vector is pre-sized in initialize_for_write.
        // We do record last_ir_written_ but don't block on it.
        memory_storage_[ir] = matrix;
        last_ir_written_.store(static_cast<int>(ir), std::memory_order_relaxed);
    } else {
        // DISK: must be serial and ordered. Guard against misuse.
        const int prev = last_ir_written_.load(std::memory_order_relaxed);
        if (prev != static_cast<int>(ir) - 1) {
            throw std::runtime_error(
                "PotentialStorage DISK store(): out-of-order or concurrent "
                "write. Got ir=" + std::to_string(ir) +
                " after last_ir=" + std::to_string(prev) +
                ". DISK mode requires sequential single-threaded writes.");
        }
        write_buffer_[write_buffer_count_++] = matrix;
        if (write_buffer_count_ >= chunk_size_) {
            write_chunk(num_chunks_written_, write_buffer_count_);
            ++num_chunks_written_;
            write_buffer_count_ = 0;
        }
        last_ir_written_.store(static_cast<int>(ir), std::memory_order_relaxed);
    }
}

void PotentialStorage::finalize_write()
{
    if (mode_ != Mode::DISK || !write_mode_) return;
    if (write_buffer_count_ > 0) {
        write_chunk(num_chunks_written_, write_buffer_count_);
        ++num_chunks_written_;
        write_buffer_count_ = 0;
    }
    write_metadata();
    write_manifest_and_success();   // must be LAST so SUCCESS marker certifies completeness
    std::cout << "[PotentialStorage] finalize: wrote " << num_chunks_written_ << " chunks\n";
    // Free the write buffer; readers will allocate their own.
    write_buffer_.clear(); write_buffer_.shrink_to_fit();
    num_chunks_ = num_chunks_written_;
    write_mode_ = false;

    // Prepare for reads.
    read_buffer_.resize(chunk_size_);
    for (int i = 0; i < chunk_size_; ++i)
        read_buffer_[i] = Eigen::MatrixXd(channels_, channels_);
    cached_matrix_ = Eigen::MatrixXd(channels_, channels_);
    read_buffer_start_ir_ = -1;
    read_buffer_end_ir_   = -1;
    cached_ir_            = -1;
    read_ready_           = true;
}

// =============================================================================
// Reading
// =============================================================================
const Eigen::MatrixXd& PotentialStorage::get(std::size_t ir)
{
    if (!initialized_)        throw std::runtime_error("PotentialStorage::get before init");
    if (!read_ready_)         throw std::runtime_error("PotentialStorage::get: not read-ready (finalize_write first)");
    if (ir >= n_grid_)        throw std::runtime_error("PotentialStorage::get: ir out of range");

    if (mode_ == Mode::MEMORY) {
        return memory_storage_[ir];
    }

    // DISK: not thread-safe; the caller must serialize. We return a REF
    // directly into read_buffer_[offset] -- no intermediate copy. The
    // reference is valid until the next call that triggers read_chunk
    // (i.e., when the caller asks for an ir outside the current chunk).
    // Safe for sequential access (the dominant pattern), which is ~20×
    // faster than a per-call copy for the dipole integral.
    if (static_cast<int>(ir) < read_buffer_start_ir_ ||
        static_cast<int>(ir) > read_buffer_end_ir_) {
        read_chunk(chunk_of(ir));
    }
    const int offset = static_cast<int>(ir) - read_buffer_start_ir_;
    return read_buffer_[offset];
}

// =============================================================================
// Disk I/O helpers
// =============================================================================
std::string PotentialStorage::chunk_filename(int chunk_idx) const {
    return checkpoint_dir_ + "/" + kChunkPrefix + std::to_string(chunk_idx) + kChunkSuffix;
}
std::string PotentialStorage::metadata_filename() const {
    return checkpoint_dir_ + "/" + kMetaName;
}
std::string PotentialStorage::manifest_filename() const {
    return checkpoint_dir_ + "/" + kManifestName;
}
std::string PotentialStorage::success_filename() const {
    return checkpoint_dir_ + "/" + kSuccessName;
}

void PotentialStorage::write_manifest_and_success() {
    if (checkpoint_dir_.empty()) return;   // MEMORY, not on disk
    // manifest.txt (user-readable). Atomic: write to .tmp then rename.
    const std::string mfn     = manifest_filename();
    const std::string mfn_tmp = mfn + ".tmp";
    {
        std::ofstream f(mfn_tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error(
            "PotentialStorage: cannot write " + mfn_tmp);
        f << manifest_;
        if (!f) throw std::runtime_error(
            "PotentialStorage: short write to " + mfn_tmp);
    }
    std::filesystem::rename(mfn_tmp, mfn);

    // __SUCCESS__ marker (empty file) -- LAST so its presence certifies
    // the checkpoint finished cleanly.
    const std::string sfn     = success_filename();
    const std::string sfn_tmp = sfn + ".tmp";
    {
        std::ofstream f(sfn_tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error(
            "PotentialStorage: cannot write " + sfn_tmp);
        f << "OK\n";
    }
    std::filesystem::rename(sfn_tmp, sfn);
}

bool PotentialStorage::verify_manifest_and_success(const std::string& dir) const {
    const std::string sfn = dir + "/" + kSuccessName;
    if (!std::filesystem::exists(sfn)) return false;
    if (manifest_.empty()) return true;  // caller opted out of manifest check
    const std::string mfn = dir + "/" + kManifestName;
    std::ifstream f(mfn, std::ios::binary);
    if (!f) return false;
    std::string got((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());
    return got == manifest_;
}

void PotentialStorage::write_chunk(int chunk_idx, int count)
{
    // Opt-in fast path: parallel pwrite-at-distinct-offsets with atomic
    // temp-file + rename + fsync for crash safety.  Bytes written are
    // byte-identical to the legacy serial path -- ONLY the wall time
    // changes.  Validated by test_storage_parallel_write.
    if (parallel_chunk_write_) {
        write_chunk_parallel_(chunk_idx, count);
        return;
    }

    // Use POSIX write() directly: std::ofstream's default internal buffer
    // (typically 4-8 KB) forces many tiny flushes when writing 10s-100s of
    // MB per matrix. Direct write() lets the OS handle buffering via the
    // page cache, which gave ~2× the throughput for large production
    // matrices in the bench_storage benchmark.
    const std::string fn = chunk_filename(chunk_idx);
    int fd = ::open(fn.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error("PotentialStorage: cannot open " + fn +
                                 " for writing: " + std::strerror(errno));
    }

    // Header (chunk count).
    if (::write(fd, &count, sizeof(int)) != static_cast<ssize_t>(sizeof(int))) {
        ::close(fd);
        throw std::runtime_error("PotentialStorage: short header write to " + fn);
    }

    // SERIAL write: version_0 intentionally kept write single-threaded.
    // Parallel pwrite on a consumer NVMe hurts (single controller + small
    // SLC cache → thread contention eats the parallelism). On striped
    // parallel filesystems (Lustre) it CAN help, but on our bench
    // hardware it was 40% slower than a single sequential write().
    // Keep the simple serial write with a partial-write retry loop.
    auto write_all = [&](const void* buf, std::size_t n) {
        const char* p = static_cast<const char*>(buf);
        while (n > 0) {
            ssize_t got = ::write(fd, p, n);
            if (got < 0) {
                if (errno == EINTR) continue;
                ::close(fd);
                throw std::runtime_error(
                    "PotentialStorage: write error on " + fn +
                    ": " + std::strerror(errno));
            }
            if (got == 0) {
                ::close(fd);
                throw std::runtime_error(
                    "PotentialStorage: zero-byte write on " + fn +
                    " (disk full?)");
            }
            p += got; n -= static_cast<std::size_t>(got);
        }
    };
    if (symmetric_storage_) {
        // SYMMETRIC mode: pack the LOWER triangle of each matrix into a
        // contiguous scratch buffer, then write only those bytes.  ZERO
        // accuracy loss when the input is bit-symmetric (caller's
        // contract): the upper triangle is byte-equal to the transpose
        // of the lower, so it's redundant on disk.
        std::vector<double> packed(packed_lower_doubles_(channels_));
        const std::size_t packed_bytes = packed.size() * sizeof(double);
        for (int i = 0; i < count; ++i) {
            pack_lower_(write_buffer_[i], packed.data());
            write_all(packed.data(), packed_bytes);
        }
    } else {
        // FULL legacy mode: write each N×N matrix verbatim.
        for (int i = 0; i < count; ++i) {
            write_all(write_buffer_[i].data(), bytes_per_matrix_);
        }
    }
    ::close(fd);
}

// ----------------------------------------------------------------------
// Opt-in parallel chunk writer.
//
// CORRECTNESS:
//   * pwrite() at non-overlapping offsets is thread-safe per POSIX.
//     Each worker handles a unique matrix index `i` (assigned via an
//     atomic fetch-add) and writes at offset = header_bytes +
//     i * record_bytes.  Offsets are disjoint by construction.
//   * write_buffer_[i] is read-only during this phase: the caller's
//     contract (store() in DISK mode) is sequential single-threaded
//     fill of write_buffer_, then ONE call to write_chunk to flush.
//     No producer touches write_buffer_ while these workers run.
//   * In symmetric mode each worker has its own thread-local `packed`
//     scratch buffer -- no shared mutation.
//   * Errors are captured via std::exception_ptr + mutex; the first
//     error wins, all workers stop, and the temp file is unlinked
//     during stack unwinding.
//
// CRASH SAFETY:
//   * Writes go to a temp file (O_EXCL) with PID + chunk_idx in the
//     name, then rename() onto the final filename atomically.  Readers
//     see EITHER the old file OR the complete new file, never a torn
//     intermediate.
//   * fsync() is called on both the file (data durability) and its
//     parent directory (rename durability).  This is more conservative
//     than the serial path (which doesn't fsync) and is what makes
//     this version "crash-safe" -- at the cost of some wall time on
//     filesystems with slow fsync.
//
// SPEED:
//   * Workers cap at min(hw_concurrency, count, 4) -- 4 is a safe
//     default for NVMe; on Lustre with stripe count > 1 you may want
//     to bump this manually (one-line change inline below).
// ----------------------------------------------------------------------
void PotentialStorage::write_chunk_parallel_(int chunk_idx, int count)
{
    if (count < 0) {
        throw std::runtime_error("PotentialStorage: negative chunk count");
    }

    const std::string fn  = chunk_filename(chunk_idx);
    // Temp file MUST be on the same filesystem as `fn` for rename() to be
    // atomic.  Constructing the temp name as fn + ".tmp.PID.IDX"
    // guarantees same parent dir.
    const std::string tmp = fn + ".tmp." + std::to_string(::getpid()) +
                                  "."   + std::to_string(chunk_idx);

    const std::size_t header_bytes = sizeof(int);
    const std::size_t record_bytes = bytes_per_matrix_disk_;
    const std::size_t packed_doubles =
        symmetric_storage_ ? packed_lower_doubles_(channels_) : 0;

    if (record_bytes == 0 && count > 0) {
        throw std::runtime_error("PotentialStorage: zero record size");
    }
    {
        const std::uint64_t max_u64 = std::numeric_limits<std::uint64_t>::max();
        if (record_bytes != 0 &&
            static_cast<std::uint64_t>(count) >
                (max_u64 - header_bytes) /
                static_cast<std::uint64_t>(record_bytes)) {
            throw std::runtime_error("PotentialStorage: file size overflow");
        }
    }

    bool renamed = false;
    try {
        unique_fd fd(::open(tmp.c_str(),
                            O_WRONLY | O_CREAT | O_EXCL,
                            0644));
        if (fd.get() < 0) {
            throw std::runtime_error(
                "PotentialStorage: cannot open temp file " + tmp +
                ": " + std::strerror(errno));
        }

        // Header at offset 0.  Written single-threaded BEFORE any worker
        // pwrite, so there's no overlap with worker offsets (which start
        // at header_bytes).
        const int count_to_write = count;
        pwrite_all(fd.get(), &count_to_write, sizeof(int), 0, tmp);

        if (count > 0) {
            std::atomic<int>  next{0};
            std::atomic<bool> stop{false};
            std::exception_ptr first_error = nullptr;
            std::mutex         error_mutex;

            const unsigned hw =
                std::max(1u, std::thread::hardware_concurrency());
            const unsigned workers =
                std::min<unsigned>({hw, static_cast<unsigned>(count), 4u});

            auto worker = [&]() {
                std::vector<double> packed;
                if (symmetric_storage_) packed.resize(packed_doubles);
                while (!stop.load(std::memory_order_relaxed)) {
                    const int i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= count) break;
                    try {
                        const std::uint64_t offset64 =
                            static_cast<std::uint64_t>(header_bytes) +
                            static_cast<std::uint64_t>(i) *
                            static_cast<std::uint64_t>(record_bytes);
                        const off_t offset = checked_off(offset64);
                        if (symmetric_storage_) {
                            // pack_lower_ reads write_buffer_[i] (read-only
                            // here) and writes thread-local `packed`.
                            pack_lower_(write_buffer_[i], packed.data());
                            pwrite_all(fd.get(), packed.data(),
                                       record_bytes, offset, tmp);
                        } else {
                            pwrite_all(fd.get(),
                                       write_buffer_[i].data(),
                                       record_bytes, offset, tmp);
                        }
                    } catch (...) {
                        stop.store(true, std::memory_order_relaxed);
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!first_error) first_error = std::current_exception();
                    }
                }
            };

            std::vector<std::thread> pool;
            pool.reserve(workers);
            for (unsigned t = 0; t < workers; ++t) pool.emplace_back(worker);
            for (auto& th : pool) th.join();

            if (first_error) std::rethrow_exception(first_error);
        }

        // Data durability: fsync ensures every byte is on stable storage
        // BEFORE we rename the file into place.  Conservative (slower)
        // than the legacy serial path but crash-safe.
        if (::fsync(fd.get()) != 0) {
            throw std::runtime_error(
                "PotentialStorage: fsync failed on " + tmp +
                ": " + std::strerror(errno));
        }
        close_checked(fd, tmp);

        // Atomic publication: rename() is atomic on POSIX filesystems
        // when source and destination are on the same filesystem.
        if (::rename(tmp.c_str(), fn.c_str()) != 0) {
            throw std::runtime_error(
                "PotentialStorage: rename " + tmp + " -> " + fn +
                " failed: " + std::strerror(errno));
        }
        renamed = true;

        // Rename durability: fsync the parent dir.  Without this, a
        // crash between the rename() and the next directory-cache flush
        // could leave the new filename pointing at the old inode.
        fsync_parent_dir(fn);

    } catch (...) {
        if (!renamed) ::unlink(tmp.c_str());
        throw;
    }
}

// ---- read_chunk_into_ ------------------------------------------------------
// Buffer-agnostic chunk reader: fills `dst_buf[0..count-1]` with the matrices
// of chunk `chunk_idx`, where `count` is read from the file's header and
// returned via `*count_out`.  Used by both the synchronous read_chunk() path
// (dst_buf = read_buffer_) and the asynchronous prefetch path (dst_buf =
// prefetched_buffer_).  Bytes written into dst_buf are byte-identical to
// what a synchronous read_chunk would have written -- only the SCHEDULING
// changes, so any caller that later swaps dst_buf into read_buffer_ sees
// the same matrix contents as the legacy code path.
//
// `max_omp_workers`: cap on the OMP worker count used INSIDE the parallel
// pread region.  For the synchronous path we keep the legacy default
// (no cap; uses omp_get_max_threads).  For the async prefetch path we
// cap at 4 so the prefetch thread doesn't steal all cores from the
// main-thread compute that's running concurrently.
void PotentialStorage::read_chunk_into_(int chunk_idx,
                                         std::vector<Eigen::MatrixXd>& dst_buf,
                                         int& count_out,
                                         int max_omp_workers)
{
    const std::string fn = chunk_filename(chunk_idx);
    int fd = ::open(fn.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("PotentialStorage: cannot open " + fn + ": " + std::strerror(errno));

    int count = 0;
    if (::read(fd, &count, sizeof(int)) != sizeof(int)) {
        ::close(fd); throw std::runtime_error("PotentialStorage: short header read from " + fn);
    }
    // Guard against corrupt header.
    if (count < 0 || count > chunk_size_) {
        ::close(fd); throw std::runtime_error("PotentialStorage: bad chunk count in " + fn);
    }
    if (static_cast<int>(dst_buf.size()) < count) {
        dst_buf.resize(count);
        for (int i = 0; i < count; ++i) dst_buf[i] = Eigen::MatrixXd(channels_, channels_);
    }
    const off_t base = sizeof(int);
    // PARALLEL read: pread is thread-safe on the same fd (it takes the
    // offset as an argument, not using the fd's internal seek position).
    // Matches version_0's chunk-read strategy. On NVMe and HPC parallel
    // filesystems, dispatching N concurrent preads increases the queue
    // depth seen by the device and boosts throughput substantially.
    // Linux caps every pread()/read() syscall at MAX_RW_COUNT = 0x7FFFF000
    // (~2 GB - 4 KB) regardless of buffer size; see man 2 read.  At
    // l_cont = 100 the W^(-1) / Rinv matrix is 17461^2*8 = 2.27 GB > 2 GB
    // and a single pread short-reads.  Loop until the full matrix is
    // pulled in (or pread genuinely fails).
    auto pread_full = [](int fd_, void* buf, std::size_t n, off_t off) -> ssize_t {
        char* p = static_cast<char*>(buf);
        std::size_t total = 0;
        while (total < n) {
            const ssize_t got = ::pread(fd_, p + total, n - total,
                                        off + static_cast<off_t>(total));
            if (got < 0) {
                if (errno == EINTR) continue;
                return -1;            // hard error
            }
            if (got == 0) break;      // EOF (truncated file)
            total += static_cast<std::size_t>(got);
        }
        return static_cast<ssize_t>(total);
    };

    std::atomic<int> fail_idx{-1};
    std::atomic<ssize_t> fail_got{-1};

    if (symmetric_storage_) {
        // SYMMETRIC mode: parallel pread + unpack.
        //
        // Lustre delivers its rated throughput only with high queue depth;
        // a single-threaded sequential reader saturates around 200 MB/s
        // while N parallel pread()s see multi-GB/s aggregate.  Measured at
        // LRZ Phase 2 (L=100, rinv chunk=39, 1.2 GB packed/matrix): the
        // serial path costs ~225 s/chunk, parallel ~10 s/chunk.  The 21x
        // wall-time savings dwarf the ~K * 1.2 GB transient packed
        // scratch (K workers).
        //
        // Per-thread scratch is `firstprivate` -- each worker owns its
        // own packed[] so the preads target distinct buffers and the
        // unpack+mirror writes its matched output slot without
        // contention.
        //
        // Worker cap = min(count, omp_get_max_threads, kMaxSymReadWorkers)
        // keeps peak transient bounded: at L=100 / N=17461 each thread
        // holds packed_lower_doubles * 8 = ~1.2 GB.  Cap of 8 gives
        // ~10 GB transient, fits the planner's 10% reserve easily on a
        // 503 GB node.  Bump if your stripe count is higher.
        constexpr int kMaxSymReadWorkers = 8;
        int max_workers = count;
#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        const int omp_max = omp_get_max_threads();
        if (omp_max > 0 && omp_max < max_workers) max_workers = omp_max;
#endif
        if (max_workers > kMaxSymReadWorkers) max_workers = kMaxSymReadWorkers;
        if (max_omp_workers > 0 && max_workers > max_omp_workers)
            max_workers = max_omp_workers;
        if (max_workers < 1) max_workers = 1;
        const std::size_t packed_doubles_loc =
            packed_lower_doubles_(channels_);
        const std::size_t packed_bytes = packed_doubles_loc * sizeof(double);
#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        #pragma omp parallel num_threads(max_workers)
#endif
        {
            std::vector<double> packed(packed_doubles_loc);
#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = 0; i < count; ++i) {
                const off_t off = base + static_cast<off_t>(i)
                                       * static_cast<off_t>(bytes_per_matrix_disk_);
                const ssize_t got = pread_full(fd, packed.data(),
                                               packed_bytes, off);
                if (got != static_cast<ssize_t>(packed_bytes)) {
                    int expect = -1;
                    if (fail_idx.compare_exchange_strong(expect, i)) {
                        fail_got.store(got);
                    }
                    continue;
                }
                unpack_lower_and_reflect_(packed.data(), dst_buf[i]);
            }
        }
    } else {
        // FULL legacy mode: parallel pread (each thread reads its own
        // matrix), no unpack.
#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        int worker_cap = (max_omp_workers > 0) ? max_omp_workers : omp_get_max_threads();
        if (worker_cap < 1) worker_cap = 1;
        #pragma omp parallel for schedule(static) num_threads(worker_cap)
#endif
        for (int i = 0; i < count; ++i) {
            const off_t off = base + static_cast<off_t>(i) * static_cast<off_t>(bytes_per_matrix_disk_);
            const ssize_t got = pread_full(fd, dst_buf[i].data(),
                                           bytes_per_matrix_disk_, off);
            if (got != static_cast<ssize_t>(bytes_per_matrix_disk_)) {
                int expect = -1;
                if (fail_idx.compare_exchange_strong(expect, i)) {
                    fail_got.store(got);
                }
            }
        }
    }

    if (fail_idx.load() >= 0) {
        const int bad_i = fail_idx.load();
        const ssize_t got = fail_got.load();
        ::close(fd);
        throw std::runtime_error("PotentialStorage: pread short at i=" +
                                 std::to_string(bad_i) +
                                 " (got=" + std::to_string(got) +
                                 ", want=" + std::to_string(bytes_per_matrix_disk_) + ")");
    }
    ::close(fd);
    count_out = count;
}

// ---- read_chunk: synchronous wrapper that fills read_buffer_ -------------
// First checks for a matching prefetch (zero-cost swap if available);
// otherwise calls read_chunk_into_ on read_buffer_ with default omp threads.
void PotentialStorage::read_chunk(int chunk_idx)
{
    // Prefetch hit: bytes are already in prefetched_buffer_ (or will be when
    // the prefetch thread completes).  Swap into read_buffer_ -- bit-identical
    // to a fresh synchronous read, with the disk wait amortised against the
    // compute that ran while the prefetch was in flight.
    if (prefetched_chunk_idx_ == chunk_idx) {
        wait_prefetch_();   // join the worker thread if still running
        if (prefetch_error_) {
            std::string msg = std::move(prefetch_error_msg_);
            prefetched_chunk_idx_ = -1;
            prefetched_count_     = 0;
            prefetch_error_       = false;
            // Fall through to synchronous read so the caller still gets the
            // chunk (in case the prefetch failed mid-flight for a transient
            // reason).  Throwing here would mask transient I/O hiccups.
            std::cerr << "[PotentialStorage] prefetch failed: " << msg
                      << " -- retrying synchronously\n";
        } else {
            std::swap(read_buffer_, prefetched_buffer_);
            read_buffer_start_ir_ = chunk_idx * chunk_size_;
            read_buffer_end_ir_   = read_buffer_start_ir_ + prefetched_count_ - 1;
            cached_ir_            = -1;
            prefetched_chunk_idx_ = -1;
            prefetched_count_     = 0;
            return;
        }
    }

    // Synchronous fallback.
    int count = 0;
    read_chunk_into_(chunk_idx, read_buffer_, count, /*max_omp_workers=*/-1);
    read_buffer_start_ir_ = chunk_idx * chunk_size_;
    read_buffer_end_ir_   = read_buffer_start_ir_ + count - 1;
    cached_ir_            = -1;
}

// ---- start_prefetch: async chunk read -------------------------------------
// Launches a background thread that fills prefetched_buffer_ with the
// contents of chunk `chunk_idx`.  When a subsequent get() / read_chunk()
// request lands on the same chunk, the bytes are already in RAM and we
// just swap the buffer pointer (zero-copy, O(1)).
//
// No-op cases:
//   * MEMORY mode -- chunks don't exist on disk.
//   * Out-of-range chunk_idx.
//   * Requested chunk is the currently-resident read_buffer_ chunk.
//   * Same chunk is already prefetched / being prefetched.
//
// If a DIFFERENT prefetch is already in flight, we wait for it and discard.
// This keeps "at most one prefetch in flight" -- simpler and avoids
// multiple threads competing for Lustre bandwidth.
//
// Bit-equivalence: the bytes written into prefetched_buffer_ come from the
// same pread() calls (same fd, same offsets, same lengths) as a synchronous
// read of this chunk would have made.  After the swap inside read_chunk(),
// read_buffer_'s contents are byte-identical to the legacy code.
void PotentialStorage::start_prefetch(int chunk_idx)
{
    // Planner-controlled gate: when the StoragePlanner had no budget
    // for an extra chunk-sized buffer it leaves prefetch_allowed_ at
    // false and we short-circuit here.  This is what keeps the C8F8 /
    // l_cont=100 BackPropagator path from OOM-killing the rank when
    // sinv+rinv would otherwise allocate ~190 GB of extra buffers.
    if (!prefetch_allowed_) return;
    if (mode_ != Mode::DISK) return;
    if (chunk_idx < 0 || chunk_idx >= num_chunks_) return;
    if (chunk_of(static_cast<std::size_t>(read_buffer_start_ir_)) == chunk_idx
        && read_buffer_start_ir_ >= 0) return;
    if (prefetched_chunk_idx_ == chunk_idx) return;

    // One prefetch in flight at a time: wait for any prior worker.
    wait_prefetch_();

    // Lazily allocate the prefetched buffer with the same shape as
    // read_buffer_ (chunk_size_ matrices of channels_ × channels_).
    if (static_cast<int>(prefetched_buffer_.size()) < chunk_size_) {
        prefetched_buffer_.resize(chunk_size_);
        for (int i = 0; i < chunk_size_; ++i)
            prefetched_buffer_[i] = Eigen::MatrixXd(channels_, channels_);
    }

    prefetched_chunk_idx_ = chunk_idx;
    prefetched_count_     = 0;
    prefetch_error_       = false;
    prefetch_error_msg_.clear();
    prefetch_running_.store(true, std::memory_order_release);
    prefetch_thread_ = std::thread([this, chunk_idx]() {
        try {
            int count = 0;
            // Cap async-prefetch worker count at 4 so we don't steal all
            // OMP threads from the main-thread compute running concurrently.
            this->read_chunk_into_(chunk_idx, this->prefetched_buffer_,
                                   count, /*max_omp_workers=*/4);
            this->prefetched_count_ = count;
        } catch (const std::exception& e) {
            this->prefetch_error_     = true;
            this->prefetch_error_msg_ = e.what();
        }
        this->prefetch_running_.store(false, std::memory_order_release);
    });
}

void PotentialStorage::wait_prefetch_()
{
    if (prefetch_thread_.joinable()) {
        prefetch_thread_.join();
    }
}

void PotentialStorage::set_prefetch_allowed(bool allowed)
{
    // If we're switching OFF while a worker is mid-flight, join it
    // first so the thread is in a clean state and the prefetched
    // buffer is not silently leaked.  Subsequent get() calls will
    // hit the synchronous read path (prefetched_chunk_idx_ stays
    // valid, so an in-flight chunk that already finished is still
    // usable as a one-shot zero-copy swap; future start_prefetch
    // calls will short-circuit).
    if (!allowed && prefetch_thread_.joinable()) {
        wait_prefetch_();
    }
    prefetch_allowed_ = allowed;
}

void PotentialStorage::set_chunk_rechunk_allowed(bool allowed)
{
    chunk_rechunk_allowed_ = allowed;
}

// ---------------------------------------------------------------------------
// renormalize_chunks_to_  --  transactional, byte-verified re-chunk
// ---------------------------------------------------------------------------
// Called from initialize_from_checkpoint when the on-disk chunk_size_
// (read from metadata.bin) exceeds the runtime-budgeted chunk_size and
// chunk_rechunk_allowed_ is true.  See header for the full algorithm
// + safety contract.
//
// Pre-conditions enforced inside read_metadata() before this is called:
//   * `chunk_size_`       : OLD chunk_size from disk metadata
//   * `num_chunks_`       : OLD chunk count from disk metadata
//   * `symmetric_storage_`: from disk metadata
//   * `bytes_per_matrix_`,
//     `bytes_per_matrix_disk_`: from caller (initialize_from_checkpoint)
//   * `checkpoint_dir_`   : set to a directory containing the OLD chunks
//   * `n_grid_`, `channels_`: from caller
// Post-conditions on success:
//   * `chunk_size_` = `new_chunk_size`
//   * `num_chunks_` = ceil(n_grid_ / new_chunk_size)
//   * On-disk layout matches a fresh build at `new_chunk_size`,
//     byte-for-byte per ir record.
// On failure (statvfs short, write error, byte mismatch on verify):
//   * Throws std::runtime_error
//   * Leaves the OLD chunks untouched on disk
//   * The `_rechunk_pending/` subdir is left behind for forensics; the
//     caller can delete it and retry, or run with --no-checkpoint-rechunk.
void PotentialStorage::renormalize_chunks_to_(int new_chunk_size)
{
    namespace fs = std::filesystem;

    if (checkpoint_dir_.empty()) {
        throw std::runtime_error(
            "PotentialStorage::renormalize_chunks_to_: no checkpoint_dir set");
    }
    if (new_chunk_size <= 0) {
        throw std::runtime_error(
            "PotentialStorage::renormalize_chunks_to_: new_chunk_size must be > 0");
    }
    if (new_chunk_size == chunk_size_) {
        // Nothing to do.  Caller usually filters this case but be defensive.
        return;
    }
    if (n_grid_ == 0 || bytes_per_matrix_disk_ == 0) {
        throw std::runtime_error(
            "PotentialStorage::renormalize_chunks_to_: invalid Nr/bytes_per_matrix_disk");
    }
    const int old_chunk_size = chunk_size_;
    const int old_num_chunks = num_chunks_;
    const int new_num_chunks =
        static_cast<int>((n_grid_ + new_chunk_size - 1) /
                         static_cast<std::size_t>(new_chunk_size));
    const std::size_t bytes_per_record = bytes_per_matrix_disk_;
    const std::size_t total_data_bytes =
        n_grid_ * bytes_per_record;

    std::cout << "[PotentialStorage] re-chunking " << checkpoint_dir_
              << ":  on-disk chunk_size=" << old_chunk_size
              << "  ->  runtime chunk_size=" << new_chunk_size
              << "  (" << old_num_chunks << " chunks -> " << new_num_chunks
              << " chunks; ~"
              << (total_data_bytes / (1024.0 * 1024.0 * 1024.0))
              << " GB per layout, ~2x transiently)\n";
    std::cout.flush();

    // ---- (1) Pre-flight disk-space check via statvfs --------------------
    {
        struct statvfs vfs;
        if (::statvfs(checkpoint_dir_.c_str(), &vfs) == 0) {
            const std::size_t free_bytes =
                static_cast<std::size_t>(vfs.f_bavail) *
                static_cast<std::size_t>(vfs.f_frsize);
            const std::size_t required =
                total_data_bytes + (1ull << 30);   // 1 GiB slack
            if (free_bytes < required) {
                std::ostringstream o;
                o << "PotentialStorage::renormalize_chunks_to_: insufficient "
                     "free space on "
                  << checkpoint_dir_
                  << "  (need ~" << (required / (1024.0 * 1024.0 * 1024.0))
                  << " GB for transient new-chunk staging, have "
                  << (free_bytes / (1024.0 * 1024.0 * 1024.0))
                  << " GB).  Either free space, or rerun with "
                     "--no-checkpoint-rechunk to fall back to rebuild.";
                throw std::runtime_error(o.str());
            }
            std::cout << "[PotentialStorage] re-chunk disk check:  free "
                      << (free_bytes / (1024.0 * 1024.0 * 1024.0))
                      << " GB  >  required "
                      << (required / (1024.0 * 1024.0 * 1024.0)) << " GB  OK\n";
        }
        // If statvfs fails (rare; non-POSIX filesystem) we proceed
        // anyway -- the OS will fail loudly on the first write if
        // truly out of space.
    }

    // ---- (2) Crash-recovery: refuse if a previous re-chunk was committing.
    const std::string committing_marker = checkpoint_dir_ + "/_rechunk_committing";
    if (fs::exists(committing_marker)) {
        throw std::runtime_error(
            "PotentialStorage::renormalize_chunks_to_: found leftover "
            "'_rechunk_committing' marker at " + committing_marker +
            ".  A previous re-chunk crashed during commit; the on-disk "
            "layout is half-old / half-new and cannot be safely loaded.  "
            "Delete " + checkpoint_dir_ + " and rebuild, or restore from "
            "backup.");
    }

    // ---- (3) Create transactional pending directory ---------------------
    const std::string pending_dir = checkpoint_dir_ + "/_rechunk_pending";
    // Wipe any stale pending dir from a previous failed attempt.
    if (fs::exists(pending_dir)) {
        std::error_code ec;
        fs::remove_all(pending_dir, ec);
        if (ec) {
            throw std::runtime_error(
                "PotentialStorage::renormalize_chunks_to_: cannot remove "
                "stale pending dir " + pending_dir + ": " + ec.message());
        }
    }
    fs::create_directories(pending_dir);

    auto pending_chunk_path = [&](int k) {
        return pending_dir + "/" + kChunkPrefix + std::to_string(k) + kChunkSuffix;
    };
    auto old_chunk_path = [&](int k) {
        // The OLD chunk files live in checkpoint_dir_ with the standard naming.
        return checkpoint_dir_ + "/" + kChunkPrefix + std::to_string(k) + kChunkSuffix;
    };

    // Open every OLD chunk file once for pread()-style random access.
    // Each old chunk file layout is:
    //   bytes 0..3                                     : int count
    //   bytes 4..4 + count*bytes_per_record            : count records
    // The "count" field is just the truncation count of the LAST chunk
    // (= old_chunk_size for chunks 0..old_num_chunks-2; <= old_chunk_size
    // for the tail).  We don't trust it blindly here -- we compute the
    // expected count from (n_grid_, old_chunk_size) and assert equality.
    std::vector<int> old_fds(old_num_chunks, -1);
    auto close_all_old_fds = [&]() {
        for (int& fd : old_fds) {
            if (fd >= 0) { ::close(fd); fd = -1; }
        }
    };
    try {
        for (int k = 0; k < old_num_chunks; ++k) {
            const std::string path = old_chunk_path(k);
            int fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) {
                throw std::runtime_error(
                    "PotentialStorage::renormalize_chunks_to_: cannot open "
                    "old chunk " + path + " for read: " + std::strerror(errno));
            }
            int header = 0;
            ssize_t got = ::pread(fd, &header, sizeof(int), 0);
            if (got != static_cast<ssize_t>(sizeof(int))) {
                ::close(fd);
                throw std::runtime_error(
                    "PotentialStorage::renormalize_chunks_to_: short header "
                    "read from " + path);
            }
            const int expect =
                (k == old_num_chunks - 1)
                ? static_cast<int>(n_grid_ - static_cast<std::size_t>(k) * old_chunk_size)
                : old_chunk_size;
            if (header != expect) {
                ::close(fd);
                throw std::runtime_error(
                    "PotentialStorage::renormalize_chunks_to_: old chunk " +
                    std::to_string(k) + " header=" + std::to_string(header) +
                    " expected=" + std::to_string(expect) +
                    " -- corrupt checkpoint, refuse to re-chunk");
            }
            old_fds[k] = fd;
        }

        // Stream pread of one ir record from the old chunks.
        // Per-record offset within an old chunk file:
        //   file_offset = sizeof(int) + (ir % old_chunk_size) * bytes_per_record
        auto pread_record = [&](std::size_t ir, void* dst) {
            const int    old_idx       = static_cast<int>(ir / static_cast<std::size_t>(old_chunk_size));
            const std::size_t offset_in_chunk = ir % static_cast<std::size_t>(old_chunk_size);
            const off_t  file_offset   =
                static_cast<off_t>(sizeof(int)) +
                static_cast<off_t>(offset_in_chunk) *
                static_cast<off_t>(bytes_per_record);
            ssize_t total = 0;
            char* p = static_cast<char*>(dst);
            std::size_t n = bytes_per_record;
            while (n > 0) {
                ssize_t got = ::pread(old_fds[old_idx], p, n,
                                       file_offset + total);
                if (got < 0) {
                    if (errno == EINTR) continue;
                    throw std::runtime_error(
                        "PotentialStorage::renormalize_chunks_to_: pread "
                        "error on old chunk " + std::to_string(old_idx) +
                        " at ir=" + std::to_string(ir) +
                        ": " + std::strerror(errno));
                }
                if (got == 0) {
                    throw std::runtime_error(
                        "PotentialStorage::renormalize_chunks_to_: short "
                        "read at ir=" + std::to_string(ir));
                }
                p += got;
                total += got;
                n -= static_cast<std::size_t>(got);
            }
        };

        // ---- (4) Write each NEW chunk into the pending directory ----
        std::vector<char> new_chunk_buf(
            static_cast<std::size_t>(new_chunk_size) * bytes_per_record);
        std::vector<char> verify_buf(bytes_per_record);

        for (int K = 0; K < new_num_chunks; ++K) {
            const std::size_t ir_lo = static_cast<std::size_t>(K) * new_chunk_size;
            const std::size_t ir_hi = std::min(
                ir_lo + static_cast<std::size_t>(new_chunk_size), n_grid_);
            const int count_K = static_cast<int>(ir_hi - ir_lo);

            // Pull each ir record from the OLD chunks into the new chunk buffer.
            for (int j = 0; j < count_K; ++j) {
                const std::size_t ir = ir_lo + static_cast<std::size_t>(j);
                pread_record(ir,
                             new_chunk_buf.data() +
                             static_cast<std::size_t>(j) * bytes_per_record);
            }

            // Write the new chunk file (header + body) to pending dir.
            const std::string new_path = pending_chunk_path(K);
            int wfd = ::open(new_path.c_str(),
                              O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (wfd < 0) {
                throw std::runtime_error(
                    "PotentialStorage::renormalize_chunks_to_: cannot open "
                    + new_path + " for write: " + std::strerror(errno));
            }
            auto write_all = [&](const void* buf, std::size_t n) {
                const char* p = static_cast<const char*>(buf);
                while (n > 0) {
                    ssize_t got = ::write(wfd, p, n);
                    if (got < 0) {
                        if (errno == EINTR) continue;
                        ::close(wfd);
                        throw std::runtime_error(
                            "PotentialStorage::renormalize_chunks_to_: write "
                            "error on " + new_path + ": " + std::strerror(errno));
                    }
                    if (got == 0) {
                        ::close(wfd);
                        throw std::runtime_error(
                            "PotentialStorage::renormalize_chunks_to_: zero-byte "
                            "write on " + new_path + " (disk full?)");
                    }
                    p += got; n -= static_cast<std::size_t>(got);
                }
            };
            write_all(&count_K, sizeof(int));
            write_all(new_chunk_buf.data(),
                      static_cast<std::size_t>(count_K) * bytes_per_record);
            if (::fsync(wfd) != 0) {
                // fsync failure is rare on POSIX; report but don't abort
                // (close+rename below still gives durability on most
                // filesystems).
                std::cerr << "[PotentialStorage] WARNING: fsync(" << new_path
                          << ") failed: " << std::strerror(errno) << "\n";
            }
            ::close(wfd);

            // Verify: read every record from the new chunk file back
            // and memcmp byte-for-byte against the original old chunk
            // bytes for the same ir.  Zero tolerance; any byte off ->
            // abort.  This is the bit-identity gate the user demanded.
            int rfd = ::open(new_path.c_str(), O_RDONLY);
            if (rfd < 0) {
                throw std::runtime_error(
                    "PotentialStorage::renormalize_chunks_to_: cannot reopen "
                    + new_path + " for verify: " + std::strerror(errno));
            }
            int hdr_back = 0;
            ssize_t hg = ::pread(rfd, &hdr_back, sizeof(int), 0);
            if (hg != static_cast<ssize_t>(sizeof(int)) ||
                hdr_back != count_K)
            {
                ::close(rfd);
                throw std::runtime_error(
                    "PotentialStorage::renormalize_chunks_to_: verify "
                    "FAILED on " + new_path + " (header mismatch)");
            }
            for (int j = 0; j < count_K; ++j) {
                const std::size_t ir = ir_lo + static_cast<std::size_t>(j);
                const off_t off =
                    static_cast<off_t>(sizeof(int)) +
                    static_cast<off_t>(j) * static_cast<off_t>(bytes_per_record);
                ssize_t total = 0;
                char* p = verify_buf.data();
                std::size_t n = bytes_per_record;
                while (n > 0) {
                    ssize_t got = ::pread(rfd, p, n, off + total);
                    if (got < 0) {
                        if (errno == EINTR) continue;
                        ::close(rfd);
                        throw std::runtime_error(
                            "PotentialStorage::renormalize_chunks_to_: "
                            "verify pread error: " + std::string(std::strerror(errno)));
                    }
                    if (got == 0) {
                        ::close(rfd);
                        throw std::runtime_error(
                            "PotentialStorage::renormalize_chunks_to_: "
                            "verify short read at ir=" + std::to_string(ir));
                    }
                    p += got; total += got;
                    n -= static_cast<std::size_t>(got);
                }
                // Compare with the source-of-truth: re-pread the same
                // record from the OLD chunk.  This catches BOTH:
                //   (a) write didn't land what we asked, AND
                //   (b) we asked the wrong thing (off-by-one in ir math).
                std::vector<char> src(bytes_per_record);
                pread_record(ir, src.data());
                if (std::memcmp(verify_buf.data(), src.data(),
                                bytes_per_record) != 0) {
                    ::close(rfd);
                    throw std::runtime_error(
                        "PotentialStorage::renormalize_chunks_to_: "
                        "byte-identity VERIFY FAILED at ir=" +
                        std::to_string(ir) +
                        ".  Old and new on-disk records differ -- aborting "
                        "re-chunk to protect the original checkpoint.");
                }
            }
            ::close(rfd);

            if ((K & 31) == 0 || K == new_num_chunks - 1) {
                std::cout << "[PotentialStorage] re-chunk progress: "
                          << (K + 1) << "/" << new_num_chunks
                          << " new chunks written + verified\n";
                std::cout.flush();
            }
        }

        // ---- (5) Write new metadata.bin in the pending dir ----
        // We write it directly using the same byte layout as
        // write_metadata() but to a pending path; commit phase below
        // renames it into checkpoint_dir_.
        {
            const std::string new_meta = pending_dir + "/" + kMetaName;
            std::ofstream out(new_meta, std::ios::binary);
            if (!out) {
                throw std::runtime_error(
                    "PotentialStorage::renormalize_chunks_to_: cannot write "
                    + new_meta);
            }
            if (symmetric_storage_) {
                out.write(reinterpret_cast<const char*>(&kSymStorageMagic),
                          sizeof(std::int32_t));
                out.write(reinterpret_cast<const char*>(&kSymStorageVersion),
                          sizeof(std::int32_t));
                out.write(reinterpret_cast<const char*>(&n_grid_),
                          sizeof(std::size_t));
                out.write(reinterpret_cast<const char*>(&channels_),
                          sizeof(int));
                out.write(reinterpret_cast<const char*>(&new_chunk_size),
                          sizeof(int));
                out.write(reinterpret_cast<const char*>(&new_num_chunks),
                          sizeof(int));
                const std::int32_t sym_flag = 1;
                out.write(reinterpret_cast<const char*>(&sym_flag),
                          sizeof(std::int32_t));
            } else {
                out.write(reinterpret_cast<const char*>(&n_grid_),
                          sizeof(std::size_t));
                out.write(reinterpret_cast<const char*>(&channels_),
                          sizeof(int));
                out.write(reinterpret_cast<const char*>(&new_chunk_size),
                          sizeof(int));
                out.write(reinterpret_cast<const char*>(&new_num_chunks),
                          sizeof(int));
            }
            if (!out) {
                throw std::runtime_error(
                    "PotentialStorage::renormalize_chunks_to_: short metadata "
                    "write to " + new_meta);
            }
        }

        // ---- (6) COMMIT (atomic from here on) ----
        // Create the recovery marker.  Its presence on any subsequent
        // load is treated as a fatal "half-committed layout" by the
        // caller; we remove it as the very last step.
        {
            std::ofstream m(committing_marker, std::ios::binary | std::ios::trunc);
            if (!m) {
                throw std::runtime_error(
                    "PotentialStorage::renormalize_chunks_to_: cannot write "
                    "commit marker " + committing_marker);
            }
            m << "rechunk old=" << old_chunk_size
              << " new=" << new_chunk_size << "\n";
        }

        // Invalidate the checkpoint while we swap: delete __SUCCESS__.
        // Any reader after this point that doesn't see __SUCCESS__
        // refuses the dir; that's the desired safety.
        const std::string success = checkpoint_dir_ + "/" + kSuccessName;
        std::error_code ec;
        fs::remove(success, ec);   // OK if it doesn't exist (atomic-rename branch already removed it)

        // Delete OLD chunk files (we held them open above; close them
        // first so the unlink frees the inode immediately on filesystems
        // that don't do that automatically for open-deleted files).
        close_all_old_fds();
        for (int k = 0; k < old_num_chunks; ++k) {
            fs::remove(old_chunk_path(k), ec);
            // We tolerate missing files (this should not happen, but
            // a crash + manual cleanup might leave us here).
        }

        // Move new chunks into place.
        for (int K = 0; K < new_num_chunks; ++K) {
            const std::string from = pending_chunk_path(K);
            const std::string to   = checkpoint_dir_ + "/" + kChunkPrefix +
                                     std::to_string(K) + kChunkSuffix;
            fs::rename(from, to);
        }

        // Move new metadata into place.
        fs::rename(pending_dir + "/" + kMetaName, checkpoint_dir_ + "/" + kMetaName);

        // Remove the now-empty pending dir.
        fs::remove(pending_dir, ec);

        // Re-write __SUCCESS__ via the standard atomic-tmp-rename path
        // (also re-writes manifest.txt, which is byte-identical to what's
        // there now -- harmless).
        write_manifest_and_success();

        // Remove the commit marker LAST -- its absence certifies the
        // commit completed cleanly.
        fs::remove(committing_marker, ec);

        // Finally, update in-memory state.
        chunk_size_ = new_chunk_size;
        num_chunks_ = new_num_chunks;

        std::cout << "[PotentialStorage] re-chunk COMMITTED: "
                  << old_chunk_size << " -> " << new_chunk_size
                  << "  (" << new_num_chunks << " chunks resident)\n";
    }
    catch (...) {
        close_all_old_fds();
        throw;
    }
}

void PotentialStorage::write_metadata()
{
    const std::string fn = metadata_filename();
    std::ofstream file(fn, std::ios::binary);
    if (!file) throw std::runtime_error("PotentialStorage: cannot write metadata to " + fn);

    if (symmetric_storage_) {
        // v2 format: magic + version + (legacy fields) + symmetric flag.
        file.write(reinterpret_cast<const char*>(&kSymStorageMagic),   sizeof(std::int32_t));
        file.write(reinterpret_cast<const char*>(&kSymStorageVersion), sizeof(std::int32_t));
        file.write(reinterpret_cast<const char*>(&n_grid_),            sizeof(std::size_t));
        file.write(reinterpret_cast<const char*>(&channels_),          sizeof(int));
        file.write(reinterpret_cast<const char*>(&chunk_size_),        sizeof(int));
        file.write(reinterpret_cast<const char*>(&num_chunks_written_),sizeof(int));
        const std::int32_t sym_flag = 1;
        file.write(reinterpret_cast<const char*>(&sym_flag),           sizeof(std::int32_t));
    } else {
        // v1 (legacy) format: no prefix, exactly as previous releases
        // wrote it.  Bit-for-bit byte-equal to pre-symmetric-storage
        // metadata.bin so old readers still work.
        file.write(reinterpret_cast<const char*>(&n_grid_),            sizeof(std::size_t));
        file.write(reinterpret_cast<const char*>(&channels_),          sizeof(int));
        file.write(reinterpret_cast<const char*>(&chunk_size_),        sizeof(int));
        file.write(reinterpret_cast<const char*>(&num_chunks_written_),sizeof(int));
    }
    if (!file) throw std::runtime_error("PotentialStorage: short metadata write");
}

bool PotentialStorage::read_metadata(std::size_t expected_n_grid, int expected_channels)
{
    const std::string fn = metadata_filename();
    if (!std::filesystem::exists(fn)) return false;
    std::ifstream file(fn, std::ios::binary);
    if (!file) return false;

    // Peek the first 4 bytes.  If they match the v2 magic, parse v2;
    // otherwise rewind and parse v1 (legacy, pre-symmetric).
    std::int32_t first_word = 0;
    file.read(reinterpret_cast<char*>(&first_word), sizeof(std::int32_t));
    if (!file) return false;

    std::size_t stored_n  = 0;
    int         stored_ch = 0, stored_cs = 0, stored_nc = 0;

    if (first_word == kSymStorageMagic) {
        // v2 format.
        std::int32_t version = 0;
        file.read(reinterpret_cast<char*>(&version), sizeof(std::int32_t));
        if (!file || version != kSymStorageVersion) return false;
        file.read(reinterpret_cast<char*>(&stored_n),  sizeof(std::size_t));
        file.read(reinterpret_cast<char*>(&stored_ch), sizeof(int));
        file.read(reinterpret_cast<char*>(&stored_cs), sizeof(int));
        file.read(reinterpret_cast<char*>(&stored_nc), sizeof(int));
        std::int32_t sym_flag = 0;
        file.read(reinterpret_cast<char*>(&sym_flag),  sizeof(std::int32_t));
        if (!file) return false;
        symmetric_storage_ = (sym_flag != 0);
    } else {
        // v1 legacy format: rewind and read as before.
        file.seekg(0);
        file.read(reinterpret_cast<char*>(&stored_n),  sizeof(std::size_t));
        file.read(reinterpret_cast<char*>(&stored_ch), sizeof(int));
        file.read(reinterpret_cast<char*>(&stored_cs), sizeof(int));
        file.read(reinterpret_cast<char*>(&stored_nc), sizeof(int));
        if (!file) return false;
        symmetric_storage_ = false;
    }
    if (stored_n  != expected_n_grid)  return false;
    if (stored_ch != expected_channels)return false;
    chunk_size_ = stored_cs;
    num_chunks_ = stored_nc;
    return true;
}

// =============================================================================
// Utility
// =============================================================================
std::size_t PotentialStorage::memory_bytes() const
{
    std::size_t b = 0;
    if (mode_ == Mode::MEMORY) {
        b += memory_storage_.size() * bytes_per_matrix_;
    } else {
        b += write_buffer_.size() * bytes_per_matrix_;
        b += read_buffer_.size()  * bytes_per_matrix_;
        b += bytes_per_matrix_;
    }
    return b;
}

// =============================================================================
// Hybrid MEMORY<->DISK checkpoint
// =============================================================================
void PotentialStorage::save_to_disk(const std::string& dir, int chunk_size,
                                    bool symmetric_storage,
                                    bool parallel_chunk_write)
{
    if (mode_ != Mode::MEMORY)
        throw std::runtime_error("PotentialStorage::save_to_disk requires MEMORY mode");
    if (!read_ready_)
        throw std::runtime_error("PotentialStorage::save_to_disk: not finalized yet");
    if (dir.empty())
        throw std::runtime_error("PotentialStorage::save_to_disk: empty dir");
    if (chunk_size <= 0)
        throw std::runtime_error("PotentialStorage::save_to_disk: chunk_size must be > 0");

    std::filesystem::create_directories(dir);

    // Temporarily re-bind the disk fields so the existing write_chunk()
    // code works unchanged. Save & restore to keep the MEMORY object's
    // state consistent at the end.
    const std::string saved_dir    = checkpoint_dir_;
    const int         saved_cs     = chunk_size_;
    const int         saved_nc     = num_chunks_;
    const int         saved_nw     = num_chunks_written_;
    auto              saved_buffer = std::move(write_buffer_);
    const bool        saved_sym    = symmetric_storage_;
    const std::size_t saved_bpd    = bytes_per_matrix_disk_;
    const bool        saved_pcw    = parallel_chunk_write_;

    checkpoint_dir_     = dir;
    chunk_size_         = chunk_size;
    num_chunks_         = static_cast<int>((n_grid_ + chunk_size - 1) / chunk_size);
    num_chunks_written_ = 0;
    // Switch to the requested on-disk format for this save.
    symmetric_storage_     = symmetric_storage;
    parallel_chunk_write_  = parallel_chunk_write;
    bytes_per_matrix_disk_ = symmetric_storage_
        ? packed_lower_doubles_(channels_) * sizeof(double)
        : bytes_per_matrix_;

    write_buffer_.resize(chunk_size_);
    for (int i = 0; i < chunk_size_; ++i)
        write_buffer_[i] = Eigen::MatrixXd(channels_, channels_);

    // Walk every matrix, fill buffer, flush per chunk_size.
    int count = 0;
    for (std::size_t ir = 0; ir < n_grid_; ++ir) {
        write_buffer_[count] = memory_storage_[ir];
        ++count;
        if (count == chunk_size_) {
            write_chunk(num_chunks_written_, count);
            ++num_chunks_written_;
            count = 0;
        }
    }
    if (count > 0) {
        write_chunk(num_chunks_written_, count);
        ++num_chunks_written_;
    }

    write_metadata();
    write_manifest_and_success();
    std::cout << "[PotentialStorage] save_to_disk: " << num_chunks_written_
              << " chunks (" << (n_grid_) << " matrices) -> " << dir
              << "  [manifest="
              << (manifest_.empty() ? "<none>" : ("`" + manifest_ + "`"))
              << "]\n";

    // Restore prior disk fields (we're still MEMORY mode logically).
    checkpoint_dir_        = saved_dir;
    chunk_size_            = saved_cs;
    num_chunks_            = saved_nc;
    num_chunks_written_    = saved_nw;
    write_buffer_          = std::move(saved_buffer);
    symmetric_storage_     = saved_sym;
    bytes_per_matrix_disk_ = saved_bpd;
    parallel_chunk_write_  = saved_pcw;
}

bool PotentialStorage::try_load_into_memory(std::size_t Nr, int channels,
                                            const std::string& dir, int chunk_size)
{
    if (dir.empty() || !std::filesystem::is_directory(dir)) return false;

    // Integrity gate.
    if (!verify_manifest_and_success(dir)) {
        std::cout << "[PotentialStorage] try_load_into_memory rejected "
                     "(no SUCCESS or manifest mismatch): " << dir << "\n";
        return false;
    }

    // Set up transient disk fields so read_metadata/read_chunk work.
    clear();
    n_grid_           = Nr;
    channels_         = channels;
    bytes_per_matrix_ = static_cast<std::size_t>(channels) * channels * sizeof(double);
    checkpoint_dir_   = dir;
    chunk_size_       = chunk_size;
    mode_             = Mode::DISK;
    if (!read_metadata(Nr, channels)) {
        clear();
        return false;
    }
    // read_metadata sets symmetric_storage_ from the v2 magic if present.
    bytes_per_matrix_disk_ = symmetric_storage_
        ? packed_lower_doubles_(channels) * sizeof(double)
        : bytes_per_matrix_;
    // Verify every chunk file exists before we commit to loading.
    for (int i = 0; i < num_chunks_; ++i) {
        if (!std::filesystem::exists(chunk_filename(i))) {
            clear();
            return false;
        }
    }

    // Now stream all chunks into memory_storage_.
    memory_storage_.resize(Nr);
    for (std::size_t ir = 0; ir < Nr; ++ir)
        memory_storage_[ir] = Eigen::MatrixXd(channels_, channels_);

    std::size_t ir = 0;
    for (int ci = 0; ci < num_chunks_; ++ci) {
        read_chunk(ci);  // fills read_buffer_ for chunk ci
        const int count = read_buffer_end_ir_ - read_buffer_start_ir_ + 1;
        for (int i = 0; i < count; ++i, ++ir) {
            memory_storage_[ir] = read_buffer_[i];
        }
    }
    if (ir != Nr) {
        clear();
        return false;
    }
    // Switch to MEMORY mode for subsequent reads; drop chunk caches.
    mode_          = Mode::MEMORY;
    write_mode_    = false;
    initialized_   = true;
    read_ready_    = true;
    read_buffer_.clear();  read_buffer_.shrink_to_fit();
    read_buffer_start_ir_ = -1;
    read_buffer_end_ir_   = -1;
    cached_ir_            = -1;
    std::cout << "[PotentialStorage] try_load_into_memory: loaded "
              << Nr << " matrices from " << dir << "\n";
    return true;
}

void PotentialStorage::release_read_buffer()
{
    // MEMORY mode has no chunk cache to free; memory_storage_ is the data.
    if (mode_ != Mode::DISK) return;
    // Join any in-flight prefetch worker first (it may still be writing
    // into prefetched_buffer_) so we don't free the buffer underneath it.
    wait_prefetch_();
    prefetched_buffer_.clear();
    prefetched_buffer_.shrink_to_fit();
    prefetched_chunk_idx_ = -1;
    prefetched_count_     = 0;
    // Drop the chunk cache.  Subsequent get() will trigger read_chunk(),
    // which lazily resizes read_buffer_ back to chunk_size_ matrices and
    // pread_full's the requested chunk file.  All other state is intact:
    //   n_grid_, channels_, chunk_size_, num_chunks_, checkpoint_dir_,
    //   bytes_per_matrix_, mode_, initialized_, read_ready_.
    read_buffer_.clear();
    read_buffer_.shrink_to_fit();
    cached_matrix_.resize(0, 0);
    read_buffer_start_ir_ = -1;
    read_buffer_end_ir_   = -1;
    cached_ir_            = -1;
}

// Free helper: peek metadata.bin to learn the on-disk chunk_size.  See
// PotentialStorage.hpp for the format contract.  Returns 0 on any failure.
//
// Auto-detects v1 (legacy, no prefix) vs v2 (magic-prefix + symmetric flag).
int peek_checkpoint_chunk_size(const std::string& checkpoint_dir)
{
    const std::string fn = checkpoint_dir + "/metadata.bin";
    if (!std::filesystem::exists(fn)) return 0;
    std::ifstream file(fn, std::ios::binary);
    if (!file) return 0;

    std::int32_t first_word = 0;
    file.read(reinterpret_cast<char*>(&first_word), sizeof(std::int32_t));
    if (!file) return 0;

    std::size_t stored_n  = 0;
    int         stored_ch = 0, stored_cs = 0, stored_nc = 0;

    if (first_word == kSymStorageMagic) {
        std::int32_t version = 0;
        file.read(reinterpret_cast<char*>(&version), sizeof(std::int32_t));
        if (!file || version != kSymStorageVersion) return 0;
        file.read(reinterpret_cast<char*>(&stored_n),  sizeof(std::size_t));
        file.read(reinterpret_cast<char*>(&stored_ch), sizeof(int));
        file.read(reinterpret_cast<char*>(&stored_cs), sizeof(int));
        file.read(reinterpret_cast<char*>(&stored_nc), sizeof(int));
    } else {
        file.seekg(0);
        file.read(reinterpret_cast<char*>(&stored_n),  sizeof(std::size_t));
        file.read(reinterpret_cast<char*>(&stored_ch), sizeof(int));
        file.read(reinterpret_cast<char*>(&stored_cs), sizeof(int));
        file.read(reinterpret_cast<char*>(&stored_nc), sizeof(int));
    }
    if (!file) return 0;
    if (stored_cs <= 0) return 0;
    return stored_cs;
}

void PotentialStorage::delete_checkpoint_files()
{
    if (checkpoint_dir_.empty()) return;
    for (int i = 0; i < num_chunks_; ++i) {
        const std::string f = chunk_filename(i);
        if (std::filesystem::exists(f)) std::filesystem::remove(f);
    }
    const std::string m  = metadata_filename();
    const std::string mf = manifest_filename();
    const std::string sf = success_filename();
    for (const auto& f : {m, mf, sf}) {
        if (std::filesystem::exists(f)) std::filesystem::remove(f);
    }
}

}  // namespace scatt
