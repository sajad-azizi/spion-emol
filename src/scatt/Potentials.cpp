#include "scatt/Potentials.hpp"

#include "angular/Gaunt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
#include <omp.h>
#endif

namespace scatt {

// ============================================================================
// Potentials ctor.
// ============================================================================
Potentials::Potentials(const Parameters& params) : params_(params) {}

// ============================================================================
// Build the sigma-vector at one radial point.
//
// V_sigma(r) = V_ee_sigma(r) + V_en_sigma(r)
//
// V_ee_sigma(r) comes from /potential/V_H in the HDF5 (Nlm_sce rows).
// We only consume the first n_exp = (l_exp_max + 1)^2 entries -- anything
// beyond is unused by the Gaunt contraction for channels <= l_max_continuum.
//
// V_en_sigma(r) is computed on-the-fly via the multipole expansion of
// 1/|r - R_i|:
//     V_en(r, n_hat) = -sum_i Z_i sum_{l,m} (4 pi / (2l+1)) *
//                      ( r_<^l / r_>^{l+1} ) Y^R_{l,m}(R_i) Y^R_{l,m}(n_hat)
// Projecting on Y^R_{l,m}(n_hat): V_en_sigma(r) = -sum_i Z_i *
//                                                 (4pi/(2l+1)) r_<^l/r_>^{l+1} *
//                                                 Y^R_{l,m}(R_hat_i).
// ============================================================================
Eigen::VectorXd Potentials::compute_V_sigma_total(
    double r, std::size_t ir, const io::PreprocData& data) const
{
    const int n_exp = params_.n_exp();
    Eigen::VectorXd V_sigma = Eigen::VectorXd::Zero(n_exp);

    // --- V_ee from preprocessing HDF5. V_H is (Nlm_sce, Nr) with
    //     Nlm_sce >= n_exp thanks to Parameters::validate().
    const Eigen::VectorXd& V_ee_col = data.V_H.col(static_cast<Eigen::Index>(ir));
    const int nmax_ee = std::min<int>(n_exp, static_cast<int>(V_ee_col.size()));
    V_sigma.head(nmax_ee) += V_ee_col.head(nmax_ee);

    // --- V_en via multipole expansion. Atoms are already translated so
    //     the SCE origin is at (0,0,0); no offset needed.
    const int l_exp_max = params_.l_exp_max();
    for (const auto& atom : data.atoms) {
        const double Z = static_cast<double>(atom.Z);
        const double Rx = atom.x, Ry = atom.y, Rz = atom.z;
        const double R  = std::sqrt(Rx * Rx + Ry * Ry + Rz * Rz);

        double theta_R = 0.0, phi_R = 0.0;
        if (R > 1e-14) {
            theta_R = std::acos(std::clamp(Rz / R, -1.0, 1.0));
            phi_R   = std::atan2(Ry, Rx);
        }
        const double r_less    = std::min(r, R);
        const double r_greater = std::max(r, R);

        for (int l = 0; l <= l_exp_max; ++l) {
            const double radial = (4.0 * M_PI / (2.0 * l + 1.0))
                                * std::pow(r_less, l)
                                / std::pow(r_greater, l + 1);
            for (int m = -l; m <= l; ++m) {
                double Y_lm_R;
                if (R < 1e-14) {
                    // atom AT origin: only Y^R_{0,0} contributes (which is 1/sqrt(4 pi))
                    Y_lm_R = (l == 0) ? (1.0 / (2.0 * std::sqrt(M_PI))) : 0.0;
                } else {
                    Y_lm_R = angular::real_Ylm(l, m, theta_R, phi_R);
                }
                if (std::abs(Y_lm_R) > 1e-15) {
                    const int sigma = angular::lm_to_idx(l, m);
                    V_sigma(sigma) += -Z * radial * Y_lm_R;
                }
            }
        }
    }
    return V_sigma;
}

// ============================================================================
// Polarization model.
//
// If the preprocessing HDF5 supplied a polarizability tensor (α_ij), we add a
// long-range attractive polarization potential to each continuum channel
// diagonal. Form (isotropic approximation; anisotropic tensor contributions
// would couple (l,m) ↔ (l±2,m') which we leave for a follow-up):
//
//   V_pol(r) = -0.5 · α_iso · f_damp(r) / r^4
//   α_iso   = (α_xx + α_yy + α_zz) / 3          (from data.alpha_iso)
//   f_damp  = (1 - exp(-(r/r_c)^6))^2           (Gianturco-Rodríguez-Ruiz form,
//                                                Thomson-Fabrikant-style cutoff)
//   r_c     = 1.5 bohr (typical for polyatomic photoionization, harmless
//             for our short-range-dominated integrand).
//
// The diagonal-only approximation is exact for a spherically symmetric V_pol
// (isotropic α): V_{μν}(r) = V_pol(r) · δ_{μν} in any orthonormal basis.
//
// Leaving α=0 in the HDF5 -> this function returns zero (back-compat with
// older preproc files that don't populate /polarizability).
// ============================================================================
Eigen::MatrixXd Potentials::compute_U_polarization_at_r(
    double r, const io::PreprocData& data) const
{
    const int channels = params_.channels();
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(channels, channels);
    if (!data.has_polarizability || data.alpha_iso == 0.0) return V;
    if (r < 1.0e-6) return V;   // damping handles finite-r divergence; skip origin

    constexpr double r_c = 1.5;              // bohr
    const double u  = r / r_c;
    const double u6 = u*u*u*u*u*u;
    const double damp = (1.0 - std::exp(-u6));
    const double damp2 = damp * damp;
    const double r4 = r*r*r*r;
    const double V_pol = -0.5 * data.alpha_iso * damp2 / r4;

    for (int mu = 0; mu < channels; ++mu) V(mu, mu) = V_pol;
    return V;
}

// ============================================================================
// Assemble V(r) over the whole radial grid.
// ============================================================================
namespace {
    static std::string human_bytes(std::size_t b) {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int u = 0; double v = static_cast<double>(b);
        while (v > 1024.0 && u < 4) { v /= 1024.0; ++u; }
        char buf[64]; std::snprintf(buf, sizeof(buf), "%.2f %s", v, units[u]);
        return buf;
    }
}

void Potentials::build(const io::PreprocData& data, StorageMode mode,
                       const std::string& checkpoint_dir, bool verbose,
                       bool try_load_checkpoint, bool save_checkpoint,
                       bool symmetric_storage,
                       bool parallel_chunk_write)
{
    params_.validate();

    const int          l_max_c   = params_.l_max_continuum;
    const int          l_exp_max = params_.l_exp_max();
    const int          channels  = params_.channels();
    const std::size_t  Nr        = params_.N_grid;
    const double       dr        = params_.dr;
    const double       r_min     = params_.r_min;

    // --- Build a manifest + auto-generate default dir (V is energy-independent,
    //     so the dir must ONLY encode molecule+grid+angular cutoffs, NOT E).
    //     Compute a compact 64-bit hash of the molecule data that V depends on,
    //     so a checkpoint from a different molecule on the same grid can't be
    //     silently loaded.  FNV-1a over atom Z+xyz + V_H checksum + orbital
    //     characters.  -----------------------------------------------------
    auto fnv1a_update = [](std::uint64_t& h, const void* p, std::size_t n) {
        const auto* b = static_cast<const unsigned char*>(p);
        for (std::size_t i = 0; i < n; ++i) {
            h ^= b[i];
            h *= 0x100000001b3ULL;
        }
    };
    std::uint64_t mol_hash = 0xcbf29ce484222325ULL;
    for (const auto& a : data.atoms) {
        int Z = a.Z; double x = a.x, y = a.y, z = a.z;
        fnv1a_update(mol_hash, &Z, sizeof(Z));
        fnv1a_update(mol_hash, &x, sizeof(x));
        fnv1a_update(mol_hash, &y, sizeof(y));
        fnv1a_update(mol_hash, &z, sizeof(z));
    }
    {
        // V_H summary: row-count, col-count, first-row sum, last-row sum.
        int rows = static_cast<int>(data.V_H.rows());
        int cols = static_cast<int>(data.V_H.cols());
        fnv1a_update(mol_hash, &rows, sizeof(rows));
        fnv1a_update(mol_hash, &cols, sizeof(cols));
        if (data.V_H.size() > 0) {
            double s0 = data.V_H.row(0).sum();
            double s1 = data.V_H.row(rows - 1).sum();
            fnv1a_update(mol_hash, &s0, sizeof(s0));
            fnv1a_update(mol_hash, &s1, sizeof(s1));
        }
    }
    {
        int nsce = data.n_sce;
        fnv1a_update(mol_hash, &nsce, sizeof(nsce));
        fnv1a_update(mol_hash, &data.Lmax_sce, sizeof(data.Lmax_sce));
    }

    char mol_hash_buf[24];
    std::snprintf(mol_hash_buf, sizeof(mol_hash_buf), "%016llx",
                  static_cast<unsigned long long>(mol_hash));
    auto fmt_d = [](double v) {
        char b[32]; std::snprintf(b, sizeof(b), "%.6f", v); return std::string(b);
    };

    const std::string manifest =
        std::string("kind=pot") +
        " mol=" + mol_hash_buf +
        " Nr="  + std::to_string(Nr) +
        " dr="  + fmt_d(dr) +
        " rmin="+ fmt_d(r_min) +
        " ch="  + std::to_string(channels) +
        " lcont=" + std::to_string(l_max_c) +
        " lexp="  + std::to_string(l_exp_max) +
        " Lsce="  + std::to_string(params_.Lmax_sce);

    const std::string dir = checkpoint_dir.empty()
        ? ("./checkpoints/pot_mol"   + std::string(mol_hash_buf) +
           "_Nr" + std::to_string(Nr) +
           "_ch" + std::to_string(channels) +
           "_lc" + std::to_string(l_max_c))
        : checkpoint_dir;
    pot_storage_.set_manifest(manifest);

    // Resolve AUTO mode based on the memory budget.
    const std::size_t v_mat_bytes = static_cast<std::size_t>(channels) * channels * sizeof(double);
    const std::size_t v_all_bytes = v_mat_bytes * Nr;
    if (mode == StorageMode::AUTO) {
        mode = (v_all_bytes <= params_.memory_budget_bytes)
               ? StorageMode::MEMORY
               : StorageMode::DISK;
    }
    mode_     = mode;
    data_ptr_ = (mode == StorageMode::ON_DEMAND) ? &data : nullptr;

    // --- Try checkpoint reload first (regardless of MEMORY vs DISK). ---
    if (try_load_checkpoint && mode != StorageMode::ON_DEMAND) {
        if (mode == StorageMode::MEMORY) {
            if (pot_storage_.try_load_into_memory(Nr, channels, dir,
                                                  params_.chunk_size)) {
                if (verbose) {
                    std::cout << "[Potentials] loaded MEMORY checkpoint from "
                              << dir << "\n";
                }
                return;
            }
        } else {  // DISK
            if (pot_storage_.initialize_from_checkpoint(Nr, channels, dir,
                                                        params_.chunk_size)) {
                if (verbose) {
                    std::cout << "[Potentials] loaded DISK checkpoint from "
                              << dir << "\n";
                }
                return;
            }
        }
    }

    if (verbose) {
        const std::size_t v_H_bytes = static_cast<std::size_t>(data.V_H.rows())
                                    * data.V_H.cols() * sizeof(double);
        const char* mode_str = "MEMORY";
        if (mode == StorageMode::DISK)       mode_str = "DISK";
        if (mode == StorageMode::ON_DEMAND)  mode_str = "ON_DEMAND";
        std::cout << "=== scatt::Potentials::build ===\n"
                  << "  l_max_continuum = " << l_max_c << "\n"
                  << "  Lmax_sce        = " << params_.Lmax_sce << "\n"
                  << "  l_exp_max       = " << l_exp_max << "\n"
                  << "  channels        = " << channels << "\n"
                  << "  Nr              = " << Nr
                  << "  dr=" << dr << "  r_min=" << r_min << "\n"
                  << "  atoms           = " << data.atoms.size() << "\n"
                  << "  mode            = " << mode_str << "\n"
                  << "  memory budget   = " << human_bytes(params_.memory_budget_bytes) << "\n";
        std::cout << "  memory footprint:\n"
                  << "    V_H (loaded)           = " << human_bytes(v_H_bytes) << "\n"
                  << "    one V(r)               = " << human_bytes(v_mat_bytes) << "\n"
                  << "    full V-cube (all r)    = " << human_bytes(v_all_bytes);
        if (mode == StorageMode::MEMORY)     std::cout << "  <-- will allocate in RAM\n";
        else if (mode == StorageMode::DISK)  std::cout << "  <-- will stream to disk chunks\n";
        else                                  std::cout << "  (not stored; ON_DEMAND)\n";
    }

    // -----------------------------------------------------------------
    // 1. Build the sparse Gaunt matrix once. We use |V_sigma|_max across
    //    r and over BOTH V_ee (from HDF5) and an estimate of V_en's
    //    maximum magnitude. In practice V_en has every (l, m) non-zero
    //    for an arbitrary geometry, so keeping all sigmas up to n_exp
    //    is the safe default; build() handles that natively when we
    //    pass an empty max-vector.
    // -----------------------------------------------------------------
    std::vector<double> V_sigma_max(params_.n_exp(), 0.0);
    {
        // From V_ee:
        const Eigen::MatrixXd& VH = data.V_H;
        const int nmax = std::min<int>(params_.n_exp(), static_cast<int>(VH.rows()));
        for (int s = 0; s < nmax; ++s) {
            double m = 0.0;
            for (Eigen::Index k = 0; k < VH.cols(); ++k) m = std::max(m, std::abs(VH(s, k)));
            V_sigma_max[s] = m;
        }
        // From V_en: bound by 4*pi*Z_max / r_min for l=0. Flag all sigmas
        // with Z-weighted angular factor > threshold as significant.
        // Cheap conservative upper bound: Any (l, m) with at least one
        // atom not at origin contributes non-trivially -- so we flag
        // every sigma in [0, n_exp) to "non-negligible". (Equivalent to
        // empty vector in GauntSparseMatrix::build.)
        for (auto& x : V_sigma_max) x = std::max(x, 1.0);  // force-include
    }
    {
        using clk = std::chrono::steady_clock;
        auto tg0 = clk::now();
        gaunt_.build(channels, l_exp_max, V_sigma_max, /*threshold=*/1e-20, verbose);
        stats_.t_gaunt_build_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - tg0).count());
    }

    // Per-r assembly helper with sub-phase timers.  Each thread-local call
    // atomically accumulates into stats_; summed ns / wall gives the
    // effective parallelism of the assembly stage.
    stats_.t_v_sigma_ns = stats_.t_gaunt_matvec_ns = stats_.t_v_pol_ns = 0;
    stats_.t_store_ns   = stats_.n_steps = 0;

    auto assemble_V_at = [&](std::size_t ir) -> Eigen::MatrixXd {
        using clk = std::chrono::steady_clock;
        const double r = r_min + ir * dr;
        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(channels, channels);
        if (r > 1e-14) {
            auto tA = clk::now();
            Eigen::VectorXd V_sigma = compute_V_sigma_total(r, ir, data);
            auto tB = clk::now();
            V = gaunt_.compute_V(V_sigma);
            auto tC = clk::now();
            V += compute_U_polarization_at_r(r, data);
            const double inv_r2 = 1.0 / (r * r);
            for (int mu = 0; mu < channels; ++mu) {
                int l, m; angular::idx_to_lm(mu, l, m);
                V(mu, mu) += l * (l + 1.0) * 0.5 * inv_r2;
            }
            auto tD = clk::now();
            const std::uint64_t dV =
                std::chrono::duration_cast<std::chrono::nanoseconds>(tB - tA).count();
            const std::uint64_t dG =
                std::chrono::duration_cast<std::chrono::nanoseconds>(tC - tB).count();
            const std::uint64_t dP =
                std::chrono::duration_cast<std::chrono::nanoseconds>(tD - tC).count();
        #if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
            #pragma omp atomic
        #endif
            stats_.t_v_sigma_ns      += dV;
        #if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
            #pragma omp atomic
        #endif
            stats_.t_gaunt_matvec_ns += dG;
        #if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
            #pragma omp atomic
        #endif
            stats_.t_v_pol_ns        += dP;
        }
    #if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        #pragma omp atomic
    #endif
        ++stats_.n_steps;
        return V;
    };

    // -----------------------------------------------------------------
    // 2. Storage setup + assembly. The parallelism is GATED by the mode
    //    because DISK storage has serial write_buffer state.
    // -----------------------------------------------------------------
    if (mode == StorageMode::MEMORY) {
        pot_storage_.initialize_for_write(Nr, channels, PotentialStorage::Mode::MEMORY);
        auto t0 = std::chrono::steady_clock::now();
        stats_.parallel_over_ir = true;
#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        #pragma omp parallel for schedule(dynamic, 16)
#endif
        for (std::size_t ir = 0; ir < Nr; ++ir) {
            auto V = assemble_V_at(ir);
            auto tSA = std::chrono::steady_clock::now();
            pot_storage_.store(ir, V);
            const std::uint64_t ds =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - tSA).count();
        #if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
            #pragma omp atomic
        #endif
            stats_.t_store_ns += ds;
        }
        auto t1 = std::chrono::steady_clock::now();
        if (verbose) {
            const double dt = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "  MEMORY: assembled V(r) at " << Nr
                      << " radial points in " << dt << " s   ("
                      << (Nr / dt) << " r/s, parallel over r)\n";
        }
        // Persist MEMORY build for reuse (e.g. across energy scan).
        if (save_checkpoint) {
            pot_storage_.save_to_disk(dir, params_.chunk_size, symmetric_storage,
                                      parallel_chunk_write);
            if (verbose)
                std::cout << "  MEMORY: checkpoint persisted to " << dir
                          << (symmetric_storage ? "  (symmetric on-disk)" : "")
                          << (parallel_chunk_write ? "  (parallel write)" : "")
                          << "\n";
        }

    } else if (mode == StorageMode::DISK) {
        // Note: checkpoint reload already happened at top of build().
        pot_storage_.initialize_for_write(Nr, channels, PotentialStorage::Mode::DISK,
                                          dir, params_.chunk_size, symmetric_storage,
                                          parallel_chunk_write);

        auto t0 = std::chrono::steady_clock::now();
        // SERIAL over ir: the write_buffer state is not thread-safe. This
        // matches version_0's DISK-mode loop. The per-r work itself
        // (gaunt matvec, Eigen unpack) is already parallelized internally
        // by Eigen/BLAS, so we don't give up much throughput.
        stats_.parallel_over_ir = false;
        for (std::size_t ir = 0; ir < Nr; ++ir) {
            auto V = assemble_V_at(ir);
            auto tSA = std::chrono::steady_clock::now();
            pot_storage_.store(ir, V);
            stats_.t_store_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - tSA).count();
            if (verbose && ir > 0 && (ir % 500 == 0)) {
                auto tnow = std::chrono::steady_clock::now();
                const double el = std::chrono::duration<double>(tnow - t0).count();
                const double eta = (Nr - ir - 1.0) * (el / (ir + 1.0));
                std::cout << "    DISK progress: " << ir << " / " << Nr
                          << "   ETA " << eta << " s\n";
            }
        }
        pot_storage_.finalize_write();
        auto t1 = std::chrono::steady_clock::now();
        if (verbose) {
            const double dt = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "  DISK: assembled + streamed in " << dt << " s   ("
                      << (Nr / dt) << " r/s, serial over r)\n";
        }

    } else if (mode == StorageMode::ON_DEMAND) {
        if (verbose && Nr > 1) {
            auto t0 = std::chrono::steady_clock::now();
            auto V  = assemble_V_at(1);
            auto t1 = std::chrono::steady_clock::now();
            const double dt = std::chrono::duration<double>(t1 - t0).count();
            (void)V;
            std::cout << "  ON_DEMAND: one V(r) build = " << dt << " s  "
                      << "(~" << (Nr * dt) << " s for a full serial sweep)\n";
        }
    }
}

// ============================================================================
// V_at / get / apply_V
// ============================================================================
Eigen::MatrixXd Potentials::V_at(std::size_t ir) const {
    if (mode_ == StorageMode::ON_DEMAND) {
        if (!data_ptr_) throw std::runtime_error("V_at: ON_DEMAND without data_ptr_");
        const int channels = params_.channels();
        const double r = params_.r(ir);
        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(channels, channels);
        if (r > 1e-14) {
            Eigen::VectorXd V_sigma = compute_V_sigma_total(r, ir, *data_ptr_);
            V = gaunt_.compute_V(V_sigma);
            V += compute_U_polarization_at_r(r, *data_ptr_);
            const double inv_r2 = 1.0 / (r * r);
            for (int mu = 0; mu < channels; ++mu) {
                int l, m; angular::idx_to_lm(mu, l, m);
                V(mu, mu) += l * (l + 1.0) * 0.5 * inv_r2;
            }
        }
        return V;
    }
    // MEMORY or DISK: fetch from storage. In DISK mode this mutates the
    // chunk cache; caller must serialize.
    return const_cast<PotentialStorage&>(pot_storage_).get(ir);
}

const Eigen::MatrixXd& Potentials::get(std::size_t ir) const {
    if (mode_ == StorageMode::ON_DEMAND)
        throw std::runtime_error("Potentials::get: reference invalid in ON_DEMAND. Use V_at().");
    // In DISK mode the reference is to the internal cached_matrix_, which
    // is overwritten on the next get() call. Caller must copy if needed.
    return const_cast<PotentialStorage&>(pot_storage_).get(ir);
}

void Potentials::apply_V(std::size_t ir, const Eigen::VectorXd& x,
                         Eigen::VectorXd& y) const {
    const int channels = params_.channels();
    if (x.size() != channels)
        throw std::runtime_error("apply_V: x.size() != channels");
    if (y.size() != channels) y.resize(channels);

    if (mode_ == StorageMode::ON_DEMAND) {
        Eigen::MatrixXd V = V_at(ir);
        y.noalias() = V.selfadjointView<Eigen::Upper>() * x;
        return;
    }
    const Eigen::MatrixXd& V = const_cast<PotentialStorage&>(pot_storage_).get(ir);
    y.noalias() = V.selfadjointView<Eigen::Upper>() * x;
}

// ============================================================================
// Symmetry diagnostic. Returns max |V - V^T| over a sample of radial points.
// ============================================================================
double Potentials::max_symmetry_deviation(std::size_t n_samples) const {
    const std::size_t Nr = params_.N_grid;
    if (Nr == 0) return 0.0;
    const std::size_t n = (n_samples == 0) ? Nr : std::min(n_samples, Nr);
    const std::size_t step = std::max<std::size_t>(1, Nr / n);
    double max_dev = 0.0;
    // In MEMORY and DISK mode we use get() (no copy). In ON_DEMAND we
    // have to materialize per sample, so we fall through to V_at().
    const bool use_ref = (mode_ != StorageMode::ON_DEMAND);
    for (std::size_t ir = 1; ir < Nr; ir += step) {   // skip r=0 where V=0
        if (use_ref) {
            const Eigen::MatrixXd& M =
                const_cast<PotentialStorage&>(pot_storage_).get(ir);
            max_dev = std::max(max_dev, (M - M.transpose()).cwiseAbs().maxCoeff());
        } else {
            Eigen::MatrixXd M = V_at(ir);
            max_dev = std::max(max_dev, (M - M.transpose()).cwiseAbs().maxCoeff());
        }
    }
    return max_dev;
}

// ============================================================================
// Cubic-symmetry probe (matches version_0::check_cubic_symmetry). For a
// molecule with octahedral (or higher) symmetry the three p-orbital
// diagonals must be equal; we print them at a few radii for inspection.
// ============================================================================
void Potentials::check_cubic_symmetry() const {
    const int channels = params_.channels();
    if (channels < 4) return;   // need at least up to l = 1
    const std::size_t Nr = params_.N_grid;
    std::cout << "\n=== V-matrix cubic-symmetry probe ===\n";
    std::cout << "  index convention: mu = l*l + l + m, p_y=1, p_z=2, p_x=3\n";
    for (std::size_t ir : {std::size_t(100), std::size_t(500), std::size_t(1000)}) {
        if (ir >= Nr) continue;
        // Use get() in MEMORY/DISK to avoid a ~channels^2 copy per probe.
        const bool use_ref = (mode_ != StorageMode::ON_DEMAND);
        Eigen::MatrixXd M_copy;
        const Eigen::MatrixXd* M_ptr = nullptr;
        if (use_ref) {
            M_ptr = &const_cast<PotentialStorage&>(pot_storage_).get(ir);
        } else {
            M_copy = V_at(ir);
            M_ptr  = &M_copy;
        }
        const auto& M = *M_ptr;
        const double r = params_.r(ir);
        const double py = M(1, 1), pz = M(2, 2), px = M(3, 3);
        std::cout << "  r=" << r
                  << "   V(p_y)=" << py
                  << "   V(p_z)=" << pz
                  << "   V(p_x)=" << px
                  << "   |py-pz|=" << std::abs(py - pz)
                  << "   |px-pz|=" << std::abs(px - pz) << "\n";
    }
}

}  // namespace scatt
