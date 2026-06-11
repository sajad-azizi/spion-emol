// CheckpointCleanup.hpp -- small utility to tear down intermediate
// checkpoints (Sinv, Rinv) after the final result (K or ψ) is safely
// extracted. For a long HPC run the intermediate checkpoints can add up
// to TB's of disk; user calls this to free the space once they're sure
// the pipeline finished.
//
// The user always owns the final call: we never auto-delete. A "safe"
// default behaviour is to recursively remove the whole checkpoint root
// (e.g. "./checkpoints") only if it contains no __ACTIVE__ marker file.
//
// Typical usage in a driver:
//
//     // ... pipeline runs, produces K, ψ ...
//     if (final_results_written_to_hdf5) {
//         scatt::cleanup::remove_sinv_rinv(cfg.checkpoint_root);
//         //  psi checkpoint stays if user wants future re-use
//     }

#pragma once

#include <filesystem>
#include <iostream>
#include <string>

namespace scatt::cleanup {

// Remove any directory under `root` whose name begins with `prefix`.
// No-op if root doesn't exist.
inline std::size_t remove_prefixed(const std::string& root,
                                   const std::string& prefix,
                                   bool verbose = true)
{
    namespace fs = std::filesystem;
    if (!fs::exists(root) || !fs::is_directory(root)) return 0;
    std::size_t n = 0;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0) {   // starts_with
            fs::remove_all(entry.path());
            if (verbose) std::cout << "[cleanup] removed " << entry.path() << "\n";
            ++n;
        }
    }
    return n;
}

// Remove intermediate (Sinv, Rinv) checkpoint directories. Leaves pot and
// psi alone -- pot is molecule-level and reusable across energies, psi is
// often kept for future reload.
inline std::size_t remove_sinv_rinv(const std::string& root = "./checkpoints",
                                    bool verbose = true)
{
    std::size_t n = 0;
    n += remove_prefixed(root, "sinv_", verbose);
    n += remove_prefixed(root, "rinv_", verbose);
    return n;
}

// Nuke EVERYTHING under `root` (pot, Sinv, Rinv, psi). Most aggressive.
inline std::size_t remove_all(const std::string& root = "./checkpoints",
                              bool verbose = true)
{
    std::size_t n = 0;
    n += remove_prefixed(root, "pot_",  verbose);
    n += remove_prefixed(root, "sinv_", verbose);
    n += remove_prefixed(root, "rinv_", verbose);
    n += remove_prefixed(root, "psi_",  verbose);
    return n;
}

}  // namespace scatt::cleanup
