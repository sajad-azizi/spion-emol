// PotentialStorage.hpp -- in-memory OR chunked-disk storage of V(r).
//
// Port of version_0/src/Potential_checkpoints.hpp, same algorithm, same
// on-disk format so old checkpoints remain loadable. Changes are
// cosmetic + stronger thread-safety contract:
//
//   * Two modes: MEMORY (RAM) and DISK (chunks of CHUNK_SIZE matrices
//     per file, with a metadata.bin header).
//   * Checkpoint reload: if a valid directory with metadata.bin + all
//     chunk files exists and matches (Nr, channels), we skip the rebuild
//     and stream V(r) from disk instead. Matches version_0's
//     initializeFromCheckpoint.
//
// Thread-safety contract (carefully stated to avoid subtle bugs):
//
//   WRITE path:
//     MEMORY: store(ir, M) is safe to call CONCURRENTLY from many threads
//             as long as each thread writes a DISTINCT ir. Multiple
//             threads writing the same ir is UB (the underlying Eigen
//             matrix slot is a single object).
//     DISK:   store(ir, M) MUST be called sequentially from a single
//             thread, in MONOTONICALLY INCREASING order of ir, starting
//             at ir = 0. The buffer `write_buffer_` and the per-chunk
//             write sequence are not guarded by a mutex. We enforce this
//             with an assertion so misuse aborts at test time, not in
//             production with a silently corrupted checkpoint.
//
//   READ path (after finalize / checkpoint load):
//     MEMORY: get(ir) returns a const reference; safe from many threads.
//     DISK:   get(ir) mutates the internal chunk cache AND the
//             cached_matrix_ field. NOT thread-safe. The scattering
//             Numerov sweep is single-threaded in r, so this is fine. If
//             future code needs parallel reads in DISK mode we provide
//             get_copy(ir) which wraps a mutex (added only when needed).
//
// The file format on disk matches version_0 byte-for-byte:
//     pot_chunk_<i>.bin: [int count][count * matrix_bytes]
//     metadata.bin:      [size_t n_grid][int channels][int chunk_size][int num_chunks]

#pragma once

#include <Eigen/Dense>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace scatt {

class PotentialStorage {
public:
    enum class Mode { MEMORY, DISK };

    PotentialStorage()  = default;
    ~PotentialStorage();
    PotentialStorage(const PotentialStorage&)            = delete;
    PotentialStorage& operator=(const PotentialStorage&) = delete;

    // --------- setup ----------------------------------------------------------

    // Begin a fresh build. Discards any prior state.
    //   chunk_size is the #matrices per chunk file (DISK mode only).
    //   checkpoint_dir must be non-empty in DISK mode; created if missing.
    //
    //   symmetric_storage:
    //     If true, on-disk chunk files store ONLY the LOWER triangle of
    //     each N×N matrix (column-major packed: N*(N+1)/2 doubles instead
    //     of N²). On read, the upper triangle is reconstructed by
    //     reflection.  Caller MUST guarantee that every matrix passed to
    //     store() is bit-symmetric (i.e. M(i,j) == M(j,i) byte-for-byte
    //     for all i<j) -- the upper triangle is silently discarded on
    //     write and re-derived from the lower on read.
    //     If the input is not bit-symmetric this option will silently
    //     LOSE the asymmetry, so callers must opt in only when they know
    //     the matrix is exactly symmetric (e.g. pot/sinv/rinv).
    //     Default false preserves the legacy full-matrix on-disk format
    //     bit-for-bit, including metadata layout.
    //     The in-memory representation is ALWAYS full N×N regardless of
    //     this flag -- only the disk path changes.
    //
    //   parallel_chunk_write:
    //     If true, write_chunk uses a multi-threaded pwrite-at-distinct-
    //     offsets implementation with atomic temp-file + rename + fsync
    //     for crash safety.  Bytes written are byte-identical to the
    //     serial path; only the WALL TIME of the write changes (typically
    //     1.5-3x faster on Lustre with stripe count > 1).  Caps at 4
    //     worker threads per chunk file (NVMe-safe default).
    //     Default false preserves the legacy single-threaded write
    //     (no fsync, no temp file) byte-for-byte.
    //     Validated bit-equivalent by test_storage_parallel_write.
    void initialize_for_write(std::size_t Nr, int channels, Mode mode,
                              const std::string& checkpoint_dir = "",
                              int chunk_size = 100,
                              bool symmetric_storage = false,
                              bool parallel_chunk_write = false);

    // Try to read an existing checkpoint on DISK matching (Nr, channels).
    // On success returns true; subsequent get() calls will stream from
    // disk. On failure returns false and leaves the object uninitialized.
    //
    // The on-disk format (legacy full or new symmetric-packed) is detected
    // automatically from the metadata.bin magic prefix; callers do NOT need
    // to know which format the checkpoint was written in.
    bool initialize_from_checkpoint(std::size_t Nr, int channels,
                                    const std::string& checkpoint_dir,
                                    int chunk_size = 100);

    // Release everything. Disk files on the checkpoint are kept.
    void clear();

    // Delete the on-disk chunk and metadata files owned by this object.
    void delete_checkpoint_files();

    // --------- hybrid MEMORY<->DISK checkpoint -------------------------------
    // After a finalized MEMORY-mode build, dump the in-memory state to disk
    // as a checkpoint, using the same on-disk chunk format as DISK mode. A
    // subsequent run may then call try_load_into_memory() and skip the rebuild.
    //
    // This is the "compute-in-RAM, persist-for-reuse" pattern. Safe to call
    // from a single thread; creates `dir` if needed.
    void save_to_disk(const std::string& dir, int chunk_size = 100,
                      bool symmetric_storage   = false,
                      bool parallel_chunk_write = false);

    // Try to load a previously written on-disk checkpoint (via either DISK
    // finalize or save_to_disk) directly into memory_storage_. On success
    // the object ends up in MEMORY mode, read-ready. On failure (missing
    // or mismatched metadata / manifest / success-marker) returns false and
    // leaves the object in its prior state -- call initialize_for_write()
    // to recover.
    bool try_load_into_memory(std::size_t Nr, int channels,
                              const std::string& dir, int chunk_size = 100);

    // -------- manifest / SUCCESS marker (checkpoint integrity) ---------------
    //
    // Call BEFORE initialize_for_write / save_to_disk. The manifest string
    // (e.g. "E=0.5 l_cont=4 Nr=3001 ...") is written to `manifest.txt`
    // inside the checkpoint dir as THE LAST THING when the build finishes.
    // A SECOND file `__SUCCESS__` is also written last; its presence
    // indicates the checkpoint was completed cleanly.
    //
    // On try_load_into_memory / initialize_from_checkpoint:
    //   - `__SUCCESS__` must exist  (else the checkpoint is mid-write or
    //                                  was never finalized → rebuild).
    //   - `manifest.txt` must match EXACTLY the string supplied here
    //                                  (else the checkpoint is from a run
    //                                  with different parameters → rebuild).
    //
    // Pass empty string to disable these checks (NOT recommended for prod).
    void set_manifest(std::string manifest) { manifest_ = std::move(manifest); }
    const std::string& manifest() const { return manifest_; }

    // --------- writing --------------------------------------------------------

    // Store V(r) at radial index ir. See thread-safety contract at top.
    void store(std::size_t ir, const Eigen::MatrixXd& matrix);

    // Finalize the DISK-mode build: flushes last chunk + metadata.
    // No-op in MEMORY.
    void finalize_write();

    // --------- reading --------------------------------------------------------

    // Fetch V(r_ir). See thread-safety contract at top.
    const Eigen::MatrixXd& get(std::size_t ir);

    // Asynchronously prefetch `chunk_idx` into an internal background
    // buffer.  A later `get()` (or explicit `read_chunk` via get's
    // chunk-cross path) for any ir in this chunk consumes the bytes
    // via a zero-copy buffer swap instead of triggering a synchronous
    // pread.  Bit-identical to the legacy code: the bytes written into
    // the prefetch buffer come from the SAME pread calls (same fd /
    // offsets / lengths) as a synchronous read; only the timing changes.
    //
    // No-op when: MEMORY mode, chunk_idx out of range, chunk_idx is the
    // currently-resident chunk, or chunk_idx is already being prefetched.
    //
    // At most ONE prefetch is in flight at a time -- a second call with
    // a different chunk_idx waits for the prior worker first, then
    // launches the new one.  Prefetch worker count is capped at 4 to
    // avoid stealing all OMP threads from the main-thread compute that
    // runs concurrently with the I/O.
    //
    // EARLY-RETURN GATE.  Default-constructed storage has prefetch
    // disabled; the planner enables it explicitly on stages that have
    // budget for the EXTRA chunk-sized buffer (see
    // StoragePlanner.hpp:StagePlan::enable_prefetch).  When disabled
    // start_prefetch is a no-op and the synchronous read path is used
    // -- bit-equivalent to disabling prefetch at compile time.
    void start_prefetch(int chunk_idx);

    // Authorise (or revoke) async chunk prefetching on this storage.
    // The StoragePlanner emits a per-stage enable_prefetch decision
    // based on whether the budget can absorb a second chunk-sized
    // buffer; the orchestrator forwards that decision here.  Default
    // is OFF to preserve the planner-naive memory budget for callers
    // that don't go through the planner (most tests).  Calling with
    // false while a prefetch is in flight waits for it first to keep
    // the worker thread joinable.
    void set_prefetch_allowed(bool allowed);
    bool prefetch_allowed() const { return prefetch_allowed_; }

    // Allow re-chunking an on-disk checkpoint at load time when its
    // on-disk chunk_size exceeds the runtime budget chunk_size.  When
    // ON (default) and `initialize_from_checkpoint` detects that the
    // on-disk chunk_size > runtime budget chunk_size, the storage
    // streams every chunk into a smaller layout under a transactional
    // pending directory, byte-verifies each new chunk against the
    // original ir record, then atomically swaps the new layout into
    // place via rename().  Bit-identical to a fresh build at the
    // smaller chunk_size (proven by test_storage_chunk_rechunk).
    //
    // When OFF, the older safety behaviour is preserved: the
    // checkpoint is rejected with an "exceeds runtime budget" message
    // and the caller must rebuild.
    //
    // Default ON because cross-node-size moves (different machine =
    // different planner output) are common and a re-chunk takes
    // ~minutes of disk I/O vs ~hours of compute rebuild.  Opt-out via
    // --no-checkpoint-rechunk on the main CLI (which threads through
    // SchurInverter / FRP / BackPropagator configs to here).
    //
    // Safety properties:
    //   * Transactional: new chunks live under <dir>/_rechunk_pending/
    //     until commit; old chunks survive until commit-time delete.
    //   * Verified: every new chunk is read back from disk and
    //     memcmp'd against the original ir record before commit.
    //   * Crash-safe: a commit marker file `_rechunk_committing`
    //     is created at commit start and removed at commit end.  Any
    //     subsequent load that sees this marker rejects the checkpoint
    //     with a clear error and asks the user to rebuild -- never
    //     silently load a half-committed layout.
    //   * Pre-flight disk-space check: statvfs the parent filesystem
    //     and require ~2x the on-disk size free before starting.
    void set_chunk_rechunk_allowed(bool allowed);
    bool chunk_rechunk_allowed() const { return chunk_rechunk_allowed_; }

    // Release ONLY the chunk read cache (read_buffer_ + cached_matrix_) of
    // a DISK-mode store, without invalidating any other state.  All disk
    // metadata (n_grid_, channels_, chunk_size_, checkpoint_dir_, etc.) is
    // preserved, so subsequent get() calls work normally -- they will
    // lazily re-allocate read_buffer_ and re-read the requested chunk.
    //
    // No-op in MEMORY mode (memory_storage_ is the sole allocation and
    // must be kept).  Callers are responsible for ensuring no other thread
    // holds a reference returned by get() at the time of the call (the
    // returned reference would be dangling once the buffer is cleared).
    void release_read_buffer();

    // --------- introspection --------------------------------------------------
    bool          is_initialized() const { return initialized_; }
    bool          is_read_ready()  const { return read_ready_;  }
    Mode          mode()           const { return mode_;        }
    std::size_t   N_grid()         const { return n_grid_;      }
    int           channels()       const { return channels_;    }
    bool          is_symmetric_storage() const { return symmetric_storage_; }
    std::size_t   memory_bytes()   const;

    // Chunk-layout introspection (DISK mode).  In MEMORY mode both return
    // 0 — chunks are an on-disk concept only.  Exposed so callers that
    // want to iterate "one chunk at a time" (to keep a chunk pinned in
    // read_buffer_ across many get() calls) can drive the loop themselves.
    int           chunk_size()     const { return mode_ == Mode::DISK ? chunk_size_ : 0; }
    int           num_chunks()     const { return mode_ == Mode::DISK ? num_chunks_ : 0; }

private:
    // General state.
    bool        initialized_ = false;
    bool        read_ready_  = false;
    bool        write_mode_  = false;
    Mode        mode_        = Mode::MEMORY;
    std::size_t n_grid_      = 0;
    int         channels_    = 0;
    std::size_t bytes_per_matrix_ = 0;     // ALWAYS N²·8 (in-memory size)

    // Symmetric-storage mode (DISK only):
    //   * symmetric_storage_ true  -> on-disk = packed lower triangle,
    //                                 N(N+1)/2 doubles per matrix
    //                                 (bytes_per_matrix_disk_ = that)
    //   * symmetric_storage_ false -> on-disk = full N²
    //                                 (bytes_per_matrix_disk_ == bytes_per_matrix_)
    // In-memory representation (memory_storage_, write_buffer_, read_buffer_,
    // cached_matrix_) is ALWAYS full N×N regardless of this flag.
    bool        symmetric_storage_   = false;
    std::size_t bytes_per_matrix_disk_ = 0;

    // Opt-in parallel chunk-write mode (DISK only).  When true,
    // write_chunk dispatches to a multi-threaded pwrite-at-distinct-
    // offsets implementation with atomic temp-file + rename for crash
    // safety.  Default false preserves legacy serial write byte-for-byte.
    bool        parallel_chunk_write_ = false;

    // MEMORY mode.
    std::vector<Eigen::MatrixXd> memory_storage_;

    // DISK mode.
    std::string checkpoint_dir_;
    int         chunk_size_ = 100;
    int         num_chunks_ = 0;

    // DISK write state (SINGLE THREAD during writes).
    std::vector<Eigen::MatrixXd> write_buffer_;
    int                          write_buffer_count_ = 0;
    int                          num_chunks_written_ = 0;
    std::atomic<int>             last_ir_written_    {-1};   // debug guard

    // DISK read state (NOT SHARED between threads).
    std::vector<Eigen::MatrixXd> read_buffer_;
    int                          read_buffer_start_ir_ = -1;
    int                          read_buffer_end_ir_   = -1;
    Eigen::MatrixXd              cached_matrix_;
    int                          cached_ir_ = -1;

    // DISK prefetch state (background thread reads into prefetched_buffer_).
    // At most one prefetch is in flight; main-thread get() / read_chunk()
    // call wait_prefetch_() when a chunk-cross hits the prefetched chunk
    // and then swaps the buffers (zero-copy).  Bit-identical to the
    // synchronous path -- only the SCHEDULING differs.
    std::vector<Eigen::MatrixXd> prefetched_buffer_;
    int                          prefetched_chunk_idx_ = -1;
    int                          prefetched_count_     = 0;
    std::thread                  prefetch_thread_;
    std::atomic<bool>            prefetch_running_{false};
    bool                         prefetch_error_       = false;
    std::string                  prefetch_error_msg_;
    // Planner-controlled gate.  When false, start_prefetch is a no-op
    // (the EXTRA chunk-sized prefetched_buffer_ is never allocated).
    // Default false so callers that bypass the planner stay conservative.
    bool                         prefetch_allowed_     = false;

    // Caller-controlled gate for on-the-fly checkpoint re-chunking.
    // Default ON so cross-node-size loads don't waste compute on a
    // rebuild.  See set_chunk_rechunk_allowed() above for full safety
    // contract and rationale.
    bool                         chunk_rechunk_allowed_ = true;

    // Helpers.
    std::string chunk_filename(int chunk_idx) const;
    std::string metadata_filename()          const;
    std::string manifest_filename()          const;
    std::string success_filename()           const;
    void        write_chunk(int chunk_idx, int count);
    void        write_chunk_parallel_(int chunk_idx, int count);   // opt-in
    void        read_chunk (int chunk_idx);
    // Buffer-agnostic chunk reader used by both the synchronous read_chunk
    // path (dst_buf = read_buffer_) and the async prefetch worker thread
    // (dst_buf = prefetched_buffer_).  `max_omp_workers` caps the inner
    // pread parallelism (-1 = default omp_get_max_threads).
    void        read_chunk_into_(int chunk_idx,
                                  std::vector<Eigen::MatrixXd>& dst_buf,
                                  int& count_out,
                                  int max_omp_workers);
    // Joins the prefetch worker thread if joinable.
    void        wait_prefetch_();
    int         chunk_of(std::size_t ir) const { return static_cast<int>(ir) / chunk_size_; }
    void        write_metadata();
    bool        read_metadata(std::size_t expected_n_grid, int expected_channels);

    // Write manifest.txt + __SUCCESS__ as the last step. Throws if dir
    // doesn't exist. Safe to call multiple times (overwrites).
    void        write_manifest_and_success();
    // Verify manifest / __SUCCESS__ in `dir`. Returns true iff BOTH:
    //   - __SUCCESS__ exists, AND
    //   - manifest.txt content equals manifest_ (or manifest_ is empty).
    bool        verify_manifest_and_success(const std::string& dir) const;

    // Re-chunk an on-disk checkpoint from its current layout (`chunk_size_`
    // and `num_chunks_` already read from metadata.bin) to a NEW
    // chunk_size.  Assumes `Nr` (the grid count), `channels_`,
    // `symmetric_storage_`, and `bytes_per_matrix_disk_` have been set
    // by the caller -- this is invoked from inside
    // `initialize_from_checkpoint` after the metadata read.
    //
    // Algorithm (transactional, bit-identical to a fresh build):
    //   1. statvfs(checkpoint_dir_) check: refuse if free < 2x on-disk size + slack.
    //   2. mkdir <dir>/_rechunk_pending/.
    //   3. For each new chunk K in [0..new_num_chunks):
    //        a) Compute ir-range [K*new_size .. min((K+1)*new_size, Nr)) .
    //        b) pread() each ir's bytes_per_matrix_disk-sized record from
    //           the appropriate old chunk file at file_offset =
    //           sizeof(int) + (ir % old_size) * bytes_per_matrix_disk.
    //        c) Write a new chunk file in <dir>/_rechunk_pending/ with
    //           the standard header + body layout.
    //        d) Re-pread the new chunk's records and memcmp byte-for-byte
    //           against the original old-chunk bytes for the same ir
    //           range.  Any mismatch aborts (leaves pending dir for
    //           forensics; old chunks untouched).
    //   4. Write new metadata.bin to pending dir.
    //   5. Commit phase (atomic):
    //        a) touch <dir>/_rechunk_committing  (recovery marker).
    //        b) Delete __SUCCESS__ from <dir>.
    //        c) Delete OLD chunk files from <dir>.
    //        d) rename(<dir>/_rechunk_pending/pot_chunk_K.bin, <dir>/pot_chunk_K.bin)
    //           for K in [0..new_num_chunks).
    //        e) rename(<dir>/_rechunk_pending/metadata.bin, <dir>/metadata.bin).
    //        f) rmdir <dir>/_rechunk_pending/.
    //        g) Re-write __SUCCESS__ via the standard atomic-rename path.
    //        h) Remove <dir>/_rechunk_committing.
    //
    //   On crash anywhere in (5), a subsequent load that detects
    //   `_rechunk_committing` immediately throws -- never silently
    //   loads a half-committed layout.  The user is told to delete the
    //   directory and rebuild.
    //
    // After successful return: chunk_size_ = new_chunk_size and
    // num_chunks_ is updated to the new chunk count.  metadata.bin on
    // disk matches.
    //
    // Throws std::runtime_error on any failure.
    void        renormalize_chunks_to_(int new_chunk_size);

    std::string manifest_;
};

// Free helper: read just the chunk_size from a checkpoint directory's
// metadata.bin without touching anything else.  Returns 0 if the metadata
// file is missing, malformed, or unreadable.  Used by the StoragePlanner
// to honour an on-disk pot chunk_size that may differ from the runtime
// budget (e.g. pot built on a 1 TB node, loaded on a 503 GB node).
//
// Format must match write_metadata():
//   size_t  n_grid
//   int     channels
//   int     chunk_size
//   int     num_chunks
int peek_checkpoint_chunk_size(const std::string& checkpoint_dir);

}  // namespace scatt
