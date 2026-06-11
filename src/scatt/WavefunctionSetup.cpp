#include "scatt/WavefunctionSetup.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
#include <omp.h>
#endif

namespace scatt {

// ===========================================================================
// build_G_vector  --  mirrors version_0 GauntCoefficients::build_G_vector.
// Returns non-zero G^R_{mu, lambda, sigma} with mu in [0, n_mu),
// lambda in [0, n_lambda), sigma in [0, n_sigma).
// ===========================================================================
std::vector<AngTriplet>
WavefunctionSetup::build_G_vector(int n_mu, int n_lambda, int n_sigma, bool verbose)
{
    auto t0 = std::chrono::steady_clock::now();
#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    std::vector<std::vector<AngTriplet>> local(static_cast<std::size_t>(nthreads));
    const std::size_t reserve_per_thread =
        std::max<std::size_t>(64, static_cast<std::size_t>(n_mu) * n_lambda * n_sigma / (20 * nthreads));
    for (auto& t : local) t.reserve(reserve_per_thread);

#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
    #pragma omp parallel
#endif
    {
#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        auto& me = local[static_cast<std::size_t>(tid)];

#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        #pragma omp for schedule(dynamic, 1)
#endif
        for (int mu = 0; mu < n_mu; ++mu) {
            int l_m, m_m; angular::idx_to_lm(mu, l_m, m_m);
            for (int lambda = 0; lambda < n_lambda; ++lambda) {
                int l_l, m_l; angular::idx_to_lm(lambda, l_l, m_l);
                for (int sigma = 0; sigma < n_sigma; ++sigma) {
                    int l_s, m_s; angular::idx_to_lm(sigma, l_s, m_s);
                    const double val = angular::gaunt_real(l_m, m_m, l_l, m_l, l_s, m_s);
                    if (std::abs(val) > 1e-14) {
                        me.push_back({mu, lambda, sigma, val});
                    }
                }
            }
        }
    }

    std::size_t total = 0;
    for (const auto& t : local) total += t.size();
    std::vector<AngTriplet> out;
    out.reserve(total);
    for (auto& t : local) { out.insert(out.end(), t.begin(), t.end()); t.clear(); t.shrink_to_fit(); }

    if (verbose) {
        const double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        const double sparsity = 100.0 * static_cast<double>(out.size())
                              / (static_cast<double>(n_mu) * n_lambda * n_sigma);
        std::cout << "[WavefunctionSetup] build_G_vector("
                  << n_mu << ", " << n_lambda << ", " << n_sigma << ") -> "
                  << out.size() << " triplets (" << sparsity
                  << "% sparsity) in " << dt << " s\n";
    }
    return out;
}

// ===========================================================================
// build_C_vector
// ===========================================================================
std::vector<AngTriplet>
WavefunctionSetup::build_C_vector(int n_sigma, int n_lambda, int n_mu, bool verbose)
{
    auto t0 = std::chrono::steady_clock::now();
#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    std::vector<std::vector<AngTriplet>> local(static_cast<std::size_t>(nthreads));
    const std::size_t reserve_per_thread =
        std::max<std::size_t>(64, static_cast<std::size_t>(n_sigma) * n_lambda * n_mu / (20 * nthreads));
    for (auto& t : local) t.reserve(reserve_per_thread);

#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
    #pragma omp parallel
#endif
    {
#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        auto& me = local[static_cast<std::size_t>(tid)];

#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        #pragma omp for schedule(dynamic, 1)
#endif
        for (int sigma = 0; sigma < n_sigma; ++sigma) {
            int l_s, m_s; angular::idx_to_lm(sigma, l_s, m_s);
            for (int lambda = 0; lambda < n_lambda; ++lambda) {
                int l_l, m_l; angular::idx_to_lm(lambda, l_l, m_l);
                for (int mu = 0; mu < n_mu; ++mu) {
                    int l_m, m_m; angular::idx_to_lm(mu, l_m, m_m);
                    const double val = angular::gaunt_real(l_s, m_s, l_l, m_l, l_m, m_m);
                    if (std::abs(val) > 1e-14) {
                        me.push_back({sigma, lambda, mu, val});
                    }
                }
            }
        }
    }

    std::size_t total = 0;
    for (const auto& t : local) total += t.size();
    std::vector<AngTriplet> out;
    out.reserve(total);
    for (auto& t : local) { out.insert(out.end(), t.begin(), t.end()); t.clear(); t.shrink_to_fit(); }

    if (verbose) {
        const double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        const double sparsity = 100.0 * static_cast<double>(out.size())
                              / (static_cast<double>(n_sigma) * n_lambda * n_mu);
        std::cout << "[WavefunctionSetup] build_C_vector("
                  << n_sigma << ", " << n_lambda << ", " << n_mu << ") -> "
                  << out.size() << " triplets (" << sparsity
                  << "% sparsity) in " << dt << " s\n";
    }
    return out;
}

// ===========================================================================
// load_chi_from_hdf5
//
// chi[ir](i, lambda) = r_ir * psi_lm(i * Nlm_sce + lambda, ir)
//   (multiplies by r because PDF uses the u/r convention -- see
//    memory/project_scattering_equations.md).
// ===========================================================================
ChiRadial WavefunctionSetup::load_chi_from_hdf5(
    const io::PreprocData& data, int n_occ, int n_lambda_cut,
    int n_transition, double dr, bool verbose)
{
    if (n_occ <= 0) throw std::runtime_error("load_chi_from_hdf5: n_occ <= 0");
    if (n_lambda_cut <= 0) throw std::runtime_error("load_chi_from_hdf5: n_lambda_cut <= 0");
    if (n_transition <= 0 || n_transition > static_cast<int>(data.Nr))
        throw std::runtime_error("load_chi_from_hdf5: bad n_transition");

    // Storage extents recorded by preprocess_molden.  Equal to the full
    // (Lmax_sce, Nr) when no truncation was requested; can be smaller
    // if the user passed --orb-lmax / --orb-rmax to save disk + RAM.
    const int Nlm_store = data.Nlm_orb_store > 0
                          ? data.Nlm_orb_store
                          : (data.Lmax_sce + 1) * (data.Lmax_sce + 1);
    const int Nr_store  = data.Nr_orb_store > 0
                          ? data.Nr_orb_store
                          : static_cast<int>(data.Nr);

    if (n_lambda_cut > Nlm_store)
        throw std::runtime_error(
            "load_chi_from_hdf5: n_lambda_cut=" + std::to_string(n_lambda_cut) +
            " exceeds stored Nlm_orb_store=" + std::to_string(Nlm_store) +
            ". Re-run preprocess_molden with --orb-lmax >= " +
            std::to_string(static_cast<int>(std::sqrt(double(n_lambda_cut))) - 1));
    if (n_transition > Nr_store)
        throw std::runtime_error(
            "load_chi_from_hdf5: n_transition=" + std::to_string(n_transition) +
            " exceeds stored Nr_orb_store=" + std::to_string(Nr_store) +
            ". Re-run preprocess_molden with --orb-rmax >= " +
            std::to_string((n_transition - 1) * dr) + " bohr");
    if (data.psi_lm.rows() < static_cast<Eigen::Index>(n_occ) * Nlm_store)
        throw std::runtime_error(
            "load_chi_from_hdf5: psi_lm has fewer rows than "
            "n_occ * Nlm_orb_store");

    ChiRadial chi(static_cast<std::size_t>(n_transition));
    for (auto& M : chi) M = Eigen::MatrixXd::Zero(n_occ, n_lambda_cut);

    for (int ir = 0; ir < n_transition; ++ir) {
        const double r = data.rmin + ir * dr;
        for (int j = 0; j < n_occ; ++j) {
            const Eigen::Index row0 = static_cast<Eigen::Index>(j) * Nlm_store;
            for (int lam = 0; lam < n_lambda_cut; ++lam) {
                chi[static_cast<std::size_t>(ir)](j, lam) =
                    r * data.psi_lm(row0 + lam, ir);
            }
        }
    }

    if (verbose) {
        std::cout << "[WavefunctionSetup] chi loaded: "
                  << "Nr=" << n_transition
                  << "  n_occ=" << n_occ
                  << "  n_lambda_cut=" << n_lambda_cut
                  << "  Nlm_orb_store=" << Nlm_store
                  << "  Nr_orb_store=" << Nr_store << "\n";
    }
    return chi;
}

// ===========================================================================
// prepare
// ===========================================================================
SetupBundle WavefunctionSetup::prepare(const Parameters& params,
                                       const io::PreprocData& data,
                                       double energy,
                                       const Inputs& inputs)
{
    SetupBundle b;

    // ---- decide n_occ --------------------------------------------------
    int n_occ = inputs.n_occ;
    if (n_occ <= 0) n_occ = data.n_occ_alpha;
    if (n_occ <= 0)
        throw std::runtime_error(
            "WavefunctionSetup::prepare: n_occ not provided and HDF5 has no occupied alpha MOs");
    if (n_occ > static_cast<int>(data.orb_occupations.size()))
        throw std::runtime_error("WavefunctionSetup::prepare: n_occ > n_sce in HDF5");

    // ---- angular cutoffs ---------------------------------------------
    const int l_cont = params.l_max_continuum;
    // Default l_exch rule: l_exch = min(l_cont, 10).
    const int l_exch = (inputs.l_max_exchange >= 0)
                       ? inputs.l_max_exchange
                       : std::min(l_cont, 10);
    const int l_orb  = l_cont + l_exch;
    const int n_mu     = (l_cont + 1) * (l_cont + 1);
    const int n_sigma  = (l_exch + 1) * (l_exch + 1);
    const int n_lambda = (l_orb  + 1) * (l_orb  + 1);

    if (n_lambda > (data.Lmax_sce + 1) * (data.Lmax_sce + 1))
        throw std::runtime_error(
            "WavefunctionSetup::prepare: l_max_orbitals > preprocessing Lmax_sce. "
            "Regenerate the HDF5 with a larger SCE cutoff.");
    // Stricter check: the orbitals on disk may have been TRUNCATED below
    // Lmax_sce by --orb-lmax.  We need to error early here rather than
    // letting load_chi_from_hdf5 choke later.
    {
        const int Nlm_store = data.Nlm_orb_store > 0
                              ? data.Nlm_orb_store
                              : (data.Lmax_sce + 1) * (data.Lmax_sce + 1);
        if (n_lambda > Nlm_store) {
            throw std::runtime_error(
                "WavefunctionSetup::prepare: scattering needs n_lambda=" +
                std::to_string(n_lambda) +
                " orbital channels but the HDF5 stored only Nlm_orb_store=" +
                std::to_string(Nlm_store) +
                ". Re-run preprocess_molden with --orb-lmax >= " +
                std::to_string(l_orb));
        }
    }

    // ---- decide n_transition ------------------------------------------
    // Resolution order:
    //   1. If inputs.n_transition > 0, honor it exactly (explicit override).
    //   2. Else if inputs.chi_cutoff > 0, scan psi_lm for the last ir where
    //      any (j, λ) coefficient exceeds chi_cutoff in magnitude across the
    //      occupied subspace and the angular cut, add a 5-bohr / ≥100-pt
    //      buffer, clamp to [1, Nr]. Orbitals are implicitly zero past this
    //      ir (exponential decay + unit-normalised MO ⇒ contribution is
    //      below chi_cutoff by construction).
    //   3. Else full Nr (no truncation -- safest fallback).
    int n_tr;
    const char* n_tr_mode = "explicit";
    if (inputs.n_transition > 0) {
        n_tr = inputs.n_transition;
        n_tr_mode = "explicit (user override)";
    } else if (inputs.chi_cutoff > 0.0) {
        n_tr_mode = "auto from chi_cutoff";
        // Use the actual on-disk storage extents -- if the orbitals
        // were truncated at preprocessing time, scanning past the
        // stored size would index out-of-bounds.
        const int Nlm_store = data.Nlm_orb_store > 0
                              ? data.Nlm_orb_store
                              : (data.Lmax_sce + 1) * (data.Lmax_sce + 1);
        const int Nr_store  = data.Nr_orb_store > 0
                              ? data.Nr_orb_store
                              : static_cast<int>(data.Nr);
        const int n_occ_local   = n_occ;
        const int n_lambda_scan = std::min(n_lambda, Nlm_store);
        int last_ir = 0;
        for (int ir = Nr_store - 1; ir >= 0; --ir) {
            double maxabs = 0.0;
            for (int j = 0; j < n_occ_local; ++j) {
                const Eigen::Index row0 =
                    static_cast<Eigen::Index>(j) * Nlm_store;
                for (int lam = 0; lam < n_lambda_scan; ++lam) {
                    const double v = std::abs(data.psi_lm(row0 + lam, ir));
                    if (v > maxabs) maxabs = v;
                }
            }
            if (maxabs > inputs.chi_cutoff) { last_ir = ir; break; }
        }
        const int buffer = std::max(100, static_cast<int>(5.0 / params.dr));
        // Clip the auto-detected n_transition at BOTH the full radial
        // grid (data.Nr) AND the orbital storage extent (Nr_store).
        // The latter matters when preprocess_molden was run with
        // --orb-rmax(-auto): orbitals past Nr_store are zero by
        // construction, so even if chi_cutoff scan lands on the
        // truncation boundary and asks for a +5-bohr buffer, that
        // buffer would index past the loaded psi_lm.
        n_tr = std::min({ static_cast<int>(data.Nr),
                          Nr_store,
                          last_ir + 1 + buffer });
        if (n_tr < 1) n_tr = 1;
    } else {
        n_tr = static_cast<int>(data.Nr);
        n_tr_mode = "full grid (chi_cutoff = 0)";
    }
    b.params.n_grid              = params.N_grid;
    b.params.dr                  = params.dr;
    b.params.r_min               = params.dr;                  // avoid centrifugal singular r=0
    b.params.asymptotic_start    = std::max(
        0.0, (static_cast<double>(params.N_grid) - 100.0) * params.dr);
    b.params.n_chi_grid          = static_cast<int>(data.Nr);
    b.params.n_transition        = n_tr;
    b.params.energy              = energy;
    b.params.n_mu                = n_mu;
    b.params.n_sigma             = n_sigma;
    b.params.l_max_continuum     = l_cont;
    b.params.l_max_exchange      = l_exch;
    b.params.l_max_orbitals      = l_orb;
    b.params.n_occ               = n_occ;
    b.params.n_threads           = 0;
    b.params.singular_threshold  = inputs.singular_threshold;
    b.params.chi_cutoff          = inputs.chi_cutoff;
    b.params.use_disk_checkpoints = inputs.use_disk_checkpoints;
    b.params.checkpoint_dir       = inputs.checkpoint_dir;
    b.params.chunk_size           = inputs.chunk_size;
    b.params.max_memory_mb        = inputs.max_memory_mb;
    b.params.alpha                = std::sqrt(2.0 * M_PI);     // PDF eqs. 4, 13, 16

    const double r_trans = (n_tr - 1) * params.dr;
    const double r_max   = (static_cast<int>(data.Nr) - 1) * params.dr;
    std::cout << "=== scatt::WavefunctionSetup::prepare ===\n"
              << "  energy       = " << energy << " Ha  (k = " << std::sqrt(2.0 * energy) << ")\n"
              << "  l_cont/exch  = " << l_cont << " / " << l_exch
              << "   (l_orb = " << l_orb
              << (inputs.l_max_exchange < 0 ? "; rule: l_exch = min(l_cont, 10)" : "; user-supplied")
              << ")\n"
              << "  n_mu/sigma   = " << n_mu << " / " << n_sigma
              << "   n_lambda = " << n_lambda << "\n"
              << "  n_occ        = " << n_occ << "\n"
              << "  n_transition = " << n_tr
              << "  (r = " << r_trans << " bohr; "
              << n_tr_mode << "; chi_cutoff=" << inputs.chi_cutoff << ")\n"
              << "  n_chi_grid   = " << b.params.n_chi_grid
              << "  (r_max = " << r_max << " bohr)"
              << "   --> orbitals set to 0 for ir > "
              << (n_tr - 1) << "\n"
              << "  alpha        = sqrt(2*pi) = " << b.params.alpha << "\n";

    // ---- load chi (with the r factor) --------------------------------
    b.chi = load_chi_from_hdf5(data, n_occ, n_lambda, n_tr, params.dr);

    // ---- build G and C triplet lists ---------------------------------
    b.G_coeff = build_G_vector(n_mu,    n_lambda, n_sigma);
    b.C_coeff = build_C_vector(n_sigma, n_lambda, n_mu);

    return b;
}

}  // namespace scatt
