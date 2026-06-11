// run_mpi.cpp -- thin MPI process orchestrator for the scattering scan.
//
// Distributes the inclusive interval [--ik-min, --ik-max] across MPI ranks
// and runs the existing `scattering` binary once per rank with that rank's
// slice.  No code in main.cpp or the scatt/ pipeline is modified -- MPI
// lives ONLY at the launcher level.
//
// Block distribution (balanced):
//     N  = ik_max - ik_min + 1
//     lo = ik_min + (r     * N) / P
//     hi = ik_min + ((r+1) * N) / P - 1     (inclusive; <lo means idle)
//
// Per-rank stdout/stderr -> <log-dir>/rank_<r>.{out,err}.  The launcher
// reduces every rank's exit code with MPI_MAX so the SLURM job fails iff
// any rank failed.
//
// Usage (one rank per node, full OpenMP per rank):
//     srun --ntasks-per-node=1 ./run_mpi \
//         <preproc.h5> --ik-min N --ik-max M [--dk D] \
//         [--binary <path>] [--log-dir <dir>] \
//         [...any flags forwarded verbatim to scattering...]
//
// Example for 45 nodes covering ik in [11..100] (90 points -> 2/rank):
//     srun -N 45 --ntasks-per-node=1 --cpus-per-task=112 ./run_mpi \
//         /work/c8f8.preproc.h5 --ik-min 11 --ik-max 100 --dk 0.01 \
//         --binary ./build/scattering --log-dir mpi_logs_lc100 \
//         --work $WORK --scratch $SCRATCH --lmax-cont 100 \
//         --omp-threads 112 --scan-id c8f8_lc100_ik11-100 --use-gpu \
//         --bench-out bench_lc100.json

#include <mpi.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <vector>

namespace fs = std::filesystem;

static std::string shquote(const std::string& s) {
    // POSIX-shell single-quote escape: every char is literal inside '...',
    // except the closing quote itself which we splice in as '"'"'.
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\"'\"'";
        else           out += c;
    }
    out += "'";
    return out;
}

static void usage_and_abort(int rank) {
    if (rank == 0) {
        std::cerr <<
"usage: run_mpi <preproc.h5> --ik-min N --ik-max M [--dk D]\n"
"               [--binary path] [--log-dir dir] [extra flags forwarded]\n"
"\n"
"Each rank runs:\n"
"    <binary> <preproc.h5> <lo> <hi> <dk> [extra flags...]\n"
"where lo..hi is this rank's contiguous slice of [ik-min..ik-max].\n";
    }
    MPI_Abort(MPI_COMM_WORLD, 2);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // ---- parse args ------------------------------------------------------
    std::string binary  = "./build/scattering";
    std::string log_dir = "./mpi_logs";
    std::string preproc;
    int    ik_min = -1, ik_max = -1;
    double dk     = 0.01;
    std::vector<std::string> passthrough;

    auto need_value = [&](int& i, const char* opt) -> std::string {
        if (i + 1 >= argc) {
            if (rank == 0) std::cerr << "missing value for " << opt << "\n";
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        return std::string(argv[++i]);
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--ik-min")  ik_min  = std::stoi(need_value(i, "--ik-min"));
        else if (a == "--ik-max")  ik_max  = std::stoi(need_value(i, "--ik-max"));
        else if (a == "--dk")      dk      = std::stod(need_value(i, "--dk"));
        else if (a == "--binary")  binary  = need_value(i, "--binary");
        else if (a == "--log-dir") log_dir = need_value(i, "--log-dir");
        else if (a == "-h" || a == "--help") usage_and_abort(rank);
        else if (preproc.empty() && !a.empty() && a[0] != '-')
            preproc = std::move(a);                     // first positional
        else
            passthrough.push_back(std::move(a));         // forward verbatim
    }

    if (preproc.empty() || ik_min < 0 || ik_max < 0 || ik_max < ik_min)
        usage_and_abort(rank);

    // ---- this rank's slice ----------------------------------------------
    const long long N  = static_cast<long long>(ik_max - ik_min + 1);
    const long long lo = ik_min + (static_cast<long long>(rank)     * N) / nranks;
    const long long hi = ik_min + (static_cast<long long>(rank + 1) * N) / nranks - 1;

    if (rank == 0) {
        fs::create_directories(log_dir);
        std::cout
          << "[run_mpi] world size = " << nranks
          << "  ik range = [" << ik_min << ".." << ik_max
          << "]  (" << N << " points)\n"
          << "[run_mpi] block size ~ " << (N + nranks - 1) / nranks
          << " ik / rank\n"
          << "[run_mpi] binary       = " << binary  << "\n"
          << "[run_mpi] preproc.h5   = " << preproc << "\n"
          << "[run_mpi] log dir      = " << log_dir << "/rank_<r>.{out,err}\n";
    }
    // Make sure log_dir exists on the shared filesystem before any rank writes.
    MPI_Barrier(MPI_COMM_WORLD);

    int exit_code = 0;
    if (lo > hi) {
        std::cout << "[rank " << rank
                  << "] idle (more ranks than ik points)\n";
    } else {
        // Build:  <binary> <preproc> <lo> <hi> <dk> <pass-through...>
        //        > rank_<r>.out  2> rank_<r>.err
        std::ostringstream cmd;
        cmd << shquote(binary)
            << ' ' << shquote(preproc)
            << ' ' << lo << ' ' << hi << ' ' << dk;
        for (const auto& a : passthrough) cmd << ' ' << shquote(a);
        cmd << " > "  << shquote(log_dir + "/rank_" + std::to_string(rank) + ".out");
        cmd << " 2> " << shquote(log_dir + "/rank_" + std::to_string(rank) + ".err");

        std::cout << "[rank " << rank << "] launching ik=["
                  << lo << ".." << hi << "]\n" << std::flush;

        const int rc = std::system(cmd.str().c_str());
        if (rc == -1) {
            std::cerr << "[rank " << rank << "] system() failed to fork\n";
            exit_code = 127;
        } else if (WIFSIGNALED(rc)) {
            std::cerr << "[rank " << rank << "] subprocess killed by signal "
                      << WTERMSIG(rc) << "\n";
            exit_code = 128 + WTERMSIG(rc);
        } else {
            exit_code = WEXITSTATUS(rc);
        }
        std::cout << "[rank " << rank << "] done ik=["
                  << lo << ".." << hi << "]  exit=" << exit_code
                  << "\n" << std::flush;
    }

    int max_exit = 0;
    MPI_Allreduce(&exit_code, &max_exit, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "[run_mpi] all ranks finished; worst exit code = "
                  << max_exit << "\n";
    }
    MPI_Finalize();
    return max_exit;
}
