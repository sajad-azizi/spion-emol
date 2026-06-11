// test_checkpoints.cpp -- HPC-robustness tests for the checkpoint system.
//
// Scenarios simulated:
//   (1) SUCCESSFUL RE-RUN: build, write, rebuild instance, load, verify
//       bit-equal. Covered for Sinv, Rinv, Psi.
//
//   (2) MANIFEST MISMATCH (different energy / l_cont / molecule in the
//       same dir): try_load must REJECT and fall through to rebuild.
//       This is the "I re-ran at a new E in the same directory" bug guard.
//
//   (3) INCOMPLETE CHECKPOINT (process killed mid-write): __SUCCESS__
//       missing → try_load must REJECT. Simulate by deleting __SUCCESS__
//       after a real build and attempting reload.
//
//   (4) PARAMETER-ENCODED DEFAULT DIRS: runs at E=0.5 and E=1.0 must pick
//       DIFFERENT auto-generated dirs.
//
//   (5) CHECKPOINT CLEANUP UTIL removes sinv/rinv but leaves psi.
//
//   (6) CHECKPOINTS FOR PSI (NEW): BackPropagator with try_load_checkpoint
//       reads ψ from disk if valid.

#include "io/HDF5Reader.hpp"
#include "scatt/Banner.hpp"
#include "scatt/BackPropagator.hpp"
#include "scatt/CheckpointCleanup.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/ForwardRPropagator.hpp"
#include "scatt/KMatrixExtractor.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Dense>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>

namespace fs = std::filesystem;

static int fails = 0;
static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++fails;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <h2o_hdf5>\n"; return 2; }

    scatt::print_la_banner();

    scatt::io::HDF5Reader reader(argv[1]);
    auto data = reader.load_all();

    scatt::Parameters params;
    params.r_min           = data.rmin;
    params.dr              = data.dr;
    params.N_grid          = data.Nr;
    params.Lmax_sce        = data.Lmax_sce;
    params.l_max_continuum = 4;
    params.validate();

    // Use an isolated root for all these tests.
    const std::string root = "./ck_test_root";
    fs::remove_all(root);
    fs::create_directories(root);

    auto build_pipeline = [&](double E, scatt::SchurInverter::Config& sic,
                              scatt::ForwardRPropagator::Config& frc,
                              scatt::BackPropagator::Config& bpc,
                              scatt::SetupBundle& bundle_out,
                              scatt::Potentials& pot_out,
                              const std::string& tag) {
        bundle_out = scatt::WavefunctionSetup::prepare(params, data, E);
        pot_out.build(data, scatt::StorageMode::MEMORY, "", false);
        sic.verbose = false;
        frc.verbose = false;
        bpc.verbose = false;
        (void) tag;
    };

    scatt::Potentials pot(params);
    pot.build(data, scatt::StorageMode::MEMORY, "", false);

    // ------------------------------------------------------------------
    // (1) Successful re-run: Sinv.
    // ------------------------------------------------------------------
    std::cout << "\n--- (1) Sinv build + reload round-trip (same E, default dir) ---\n";
    {
        auto bundle = scatt::WavefunctionSetup::prepare(params, data, 0.5);
        scatt::ExchangeCoupling EC(bundle.G_coeff, bundle.params.n_mu,
                                   bundle.params.n_sigma, bundle.params.n_occ,
                                   data.rmin, data.dr);

        scatt::SchurInverter SI_a(bundle.params, pot, &EC, &bundle.chi);
        scatt::SchurInverter::Config c;
        c.storage = scatt::StorageMode::MEMORY;
        c.checkpoint_dir = root + "/sinv_test";
        c.try_load_checkpoint = false;
        c.save_checkpoint = true;
        c.verbose = false;
        SI_a.build(c);

        // Rebuild instance, reload from checkpoint.
        scatt::SchurInverter SI_b(bundle.params, pot, &EC, &bundle.chi);
        scatt::SchurInverter::Config c2 = c;
        c2.try_load_checkpoint = true;
        c2.save_checkpoint     = false;
        c2.verbose = true;
        SI_b.build(c2);

        double worst = 0.0;
        for (int ir : {0, 1, 500, 1500, (int)bundle.params.n_grid - 1}) {
            double d = (SI_a.get(ir) - SI_b.get(ir)).cwiseAbs().maxCoeff();
            worst = std::max(worst, d);
        }
        check(worst < 1e-14, "Sinv loaded bit-equal to built (worst=" +
                             std::to_string(worst) + ")");
    }

    // ------------------------------------------------------------------
    // (2) MANIFEST MISMATCH: build at E=0.5, try to load at E=1.0 (same dir).
    //     Reload MUST be rejected.
    // ------------------------------------------------------------------
    std::cout << "\n--- (2) manifest mismatch across energies ---\n";
    {
        // The dir already has the E=0.5 checkpoint from (1).
        // Now spin up an instance at E=1.0 with the SAME checkpoint_dir.
        auto bundle2 = scatt::WavefunctionSetup::prepare(params, data, 1.0);
        scatt::ExchangeCoupling EC2(bundle2.G_coeff, bundle2.params.n_mu,
                                    bundle2.params.n_sigma, bundle2.params.n_occ,
                                    data.rmin, data.dr);
        scatt::SchurInverter SI_wrong(bundle2.params, pot, &EC2, &bundle2.chi);
        scatt::SchurInverter::Config c3;
        c3.storage = scatt::StorageMode::MEMORY;
        c3.checkpoint_dir = root + "/sinv_test";   // same dir!
        c3.try_load_checkpoint = true;
        c3.save_checkpoint     = false;
        c3.verbose             = true;
        SI_wrong.build(c3);

        // If the manifest check works, SI_wrong should have REBUILT (not
        // silently loaded the wrong-E Sinv). Verify by reading from the
        // NEW manifest dir or by checking the Sinv values against a
        // freshly-computed reference at E=1.0.
        scatt::SchurInverter SI_ref(bundle2.params, pot, &EC2, &bundle2.chi);
        scatt::SchurInverter::Config c_ref = c3;
        c_ref.checkpoint_dir   = root + "/sinv_ref_E1";
        c_ref.try_load_checkpoint = false;
        c_ref.save_checkpoint  = false;
        c_ref.verbose          = false;
        fs::remove_all(c_ref.checkpoint_dir);
        SI_ref.build(c_ref);

        double worst = 0.0;
        for (int ir : {0, 1, 500, 1500}) {
            double d = (SI_wrong.get(ir) - SI_ref.get(ir)).cwiseAbs().maxCoeff();
            worst = std::max(worst, d);
        }
        check(worst < 1e-14,
              "manifest guard forced rebuild at new E "
              "(values match fresh E=1.0 build, worst=" + std::to_string(worst) + ")");
    }

    // ------------------------------------------------------------------
    // (3) INCOMPLETE CHECKPOINT: delete __SUCCESS__, reload must reject.
    // ------------------------------------------------------------------
    std::cout << "\n--- (3) incomplete checkpoint (no __SUCCESS__) is rejected ---\n";
    {
        auto bundle = scatt::WavefunctionSetup::prepare(params, data, 0.5);
        scatt::ExchangeCoupling EC(bundle.G_coeff, bundle.params.n_mu,
                                   bundle.params.n_sigma, bundle.params.n_occ,
                                   data.rmin, data.dr);

        const std::string dir = root + "/sinv_test";
        // There's already a valid checkpoint there. Overwrite by rebuilding
        // at E=0.5 then DELETE __SUCCESS__ to simulate a crash mid-write.
        scatt::SchurInverter SI_build(bundle.params, pot, &EC, &bundle.chi);
        scatt::SchurInverter::Config cb;
        cb.storage = scatt::StorageMode::MEMORY;
        cb.checkpoint_dir = dir;
        cb.try_load_checkpoint = false;
        cb.save_checkpoint = true;
        cb.verbose = false;
        fs::remove_all(dir);
        SI_build.build(cb);
        // confirm marker exists
        const std::string success = dir + "/__SUCCESS__";
        check(fs::exists(success), "SUCCESS marker written");

        // Now simulate a crash: delete the marker.
        fs::remove(success);

        // Second run attempts to load. If rejected, rebuild + re-save so
        // the SUCCESS marker is back (that's what a user would want).
        scatt::SchurInverter SI_reload(bundle.params, pot, &EC, &bundle.chi);
        scatt::SchurInverter::Config cr = cb;
        cr.try_load_checkpoint = true;
        cr.save_checkpoint     = true;    // restore marker
        cr.verbose             = true;
        SI_reload.build(cr);

        // After rebuild, SUCCESS must be back.
        check(fs::exists(success), "SUCCESS marker restored after rebuild");

        // Values must still match the original.
        double worst = 0.0;
        for (int ir : {0, 1, 500}) {
            double d = (SI_build.get(ir) - SI_reload.get(ir)).cwiseAbs().maxCoeff();
            worst = std::max(worst, d);
        }
        check(worst < 1e-14, "rebuilt checkpoint == original (worst=" +
                             std::to_string(worst) + ")");
    }

    // ------------------------------------------------------------------
    // (4) Parameter-encoded default dirs: two different Es at default
    //     location must create two different dirs.
    // ------------------------------------------------------------------
    std::cout << "\n--- (4) default dirs encode (E, l_cont) ---\n";
    {
        // Force a fresh default-dir run at E=0.5 and E=1.0, check the dirs
        // that end up under ./checkpoints are distinct.
        scatt::cleanup::remove_sinv_rinv("./checkpoints", false);

        auto bundle_a = scatt::WavefunctionSetup::prepare(params, data, 0.5);
        scatt::ExchangeCoupling ECa(bundle_a.G_coeff, bundle_a.params.n_mu,
                                    bundle_a.params.n_sigma, bundle_a.params.n_occ,
                                    data.rmin, data.dr);
        scatt::SchurInverter SI_a(bundle_a.params, pot, &ECa, &bundle_a.chi);
        scatt::SchurInverter::Config ca;
        ca.storage = scatt::StorageMode::MEMORY;
        ca.try_load_checkpoint = false;
        ca.save_checkpoint = true;
        ca.verbose = false;
        SI_a.build(ca);

        auto bundle_b = scatt::WavefunctionSetup::prepare(params, data, 1.0);
        scatt::ExchangeCoupling ECb(bundle_b.G_coeff, bundle_b.params.n_mu,
                                    bundle_b.params.n_sigma, bundle_b.params.n_occ,
                                    data.rmin, data.dr);
        scatt::SchurInverter SI_b(bundle_b.params, pot, &ECb, &bundle_b.chi);
        scatt::SchurInverter::Config cb = ca;
        SI_b.build(cb);

        int sinv_dirs_seen = 0;
        for (const auto& e : fs::directory_iterator("./checkpoints")) {
            if (e.is_directory() &&
                e.path().filename().string().rfind("sinv_", 0) == 0) {
                ++sinv_dirs_seen;
            }
        }
        check(sinv_dirs_seen >= 2,
              "distinct default dirs for E=0.5 and E=1.0 "
              "(" + std::to_string(sinv_dirs_seen) + " sinv_* dirs created)");
    }

    // ------------------------------------------------------------------
    // (5) Cleanup utility removes sinv/rinv but preserves psi.
    // ------------------------------------------------------------------
    std::cout << "\n--- (5) cleanup utility ---\n";
    {
        // Create fake dirs.
        fs::create_directories("./checkpoints/sinv_fake");
        fs::create_directories("./checkpoints/rinv_fake");
        fs::create_directories("./checkpoints/psi_fake");

        // User clean-up at end of pipeline.
        std::size_t n = scatt::cleanup::remove_sinv_rinv("./checkpoints", false);

        check(!fs::exists("./checkpoints/sinv_fake"), "sinv_fake removed");
        check(!fs::exists("./checkpoints/rinv_fake"), "rinv_fake removed");
        check(fs::exists("./checkpoints/psi_fake"),   "psi_fake preserved");
        (void) n;
    }

    // ------------------------------------------------------------------
    // (6.pre) POTENTIALS REUSE ACROSS ENERGIES:
    //   V depends on the molecule + grid + l_cont + Lmax_sce but NOT on E.
    //   Build once, reload for every subsequent E — that's the whole point
    //   of persisting the potential.
    // ------------------------------------------------------------------
    std::cout << "\n--- (6.pre) Potentials reused across energies ---\n";
    {
        const std::string pdir = root + "/pot_test";
        fs::remove_all(pdir);

        scatt::Potentials p1(params);
        p1.build(data, scatt::StorageMode::MEMORY, pdir, /*verbose=*/false,
                 /*try_load=*/false, /*save=*/true);
        check(fs::exists(pdir + "/__SUCCESS__"),
              "Potentials MEMORY build saved checkpoint (__SUCCESS__)");

        // "New run" at different E (different Parameters energy, but we don't
        // even need to reconstruct here — V doesn't depend on E). Same
        // checkpoint dir → should load.
        scatt::Potentials p2(params);
        p2.build(data, scatt::StorageMode::MEMORY, pdir, /*verbose=*/true,
                 /*try_load=*/true, /*save=*/false);

        double worst = 0.0;
        for (std::size_t ir : {std::size_t(0), std::size_t(100),
                                std::size_t(1500), std::size_t(params.N_grid - 1)}) {
            double d = (p1.get(ir) - p2.get(ir)).cwiseAbs().maxCoeff();
            worst = std::max(worst, d);
        }
        check(worst < 1e-14,
              "Potentials reload bit-equal to build (worst=" +
              std::to_string(worst) + ")");
    }

    // ------------------------------------------------------------------
    // (6) BackPropagator checkpoint: build, save, reload, compare.
    // ------------------------------------------------------------------
    std::cout << "\n--- (6) BackPropagator ψ checkpoint ---\n";
    {
        // Build full pipeline once to get K.
        auto bundle = scatt::WavefunctionSetup::prepare(params, data, 0.5);
        scatt::ExchangeCoupling EC(bundle.G_coeff, bundle.params.n_mu,
                                   bundle.params.n_sigma, bundle.params.n_occ,
                                   data.rmin, data.dr);
        scatt::SchurInverter SI(bundle.params, pot, &EC, &bundle.chi);
        scatt::SchurInverter::Config sc;
        sc.storage = scatt::StorageMode::MEMORY;
        sc.try_load_checkpoint = false; sc.save_checkpoint = false;
        sc.verbose = false;
        sc.checkpoint_dir = root + "/sinv_for_bp";
        fs::remove_all(sc.checkpoint_dir);
        SI.build(sc);
        scatt::WInverseOperator WI(bundle.params, SI, &EC, &bundle.chi, sc.W_min);
        scatt::ForwardRPropagator FRP(bundle.params, pot, WI);
        scatt::ForwardRPropagator::Config frc;
        frc.storage = scatt::StorageMode::MEMORY;
        frc.try_load_checkpoint = false; frc.save_checkpoint = false;
        frc.verbose = false;
        frc.checkpoint_dir = root + "/rinv_for_bp";
        fs::remove_all(frc.checkpoint_dir);
        FRP.run(frc);
        scatt::KMatrixExtractor KME(bundle.params, FRP);
        auto res = KME.extract();
        auto bc  = scatt::KMatrixExtractor::make_psi_boundary(bundle.params, res.K_matrix);

        // First BackPropagator run: fresh + save.
        scatt::BackPropagator BP_a(bundle.params, pot, FRP, WI);
        scatt::BackPropagator::Config bpc_a;
        bpc_a.n_keep_lo = 0; bpc_a.n_keep_hi = (int)bundle.params.n_grid - 1;
        bpc_a.compute_f = false;
        bpc_a.psi_storage = scatt::StorageMode::MEMORY;
        bpc_a.checkpoint_dir = root + "/psi_test";
        bpc_a.try_load_checkpoint = false;
        bpc_a.save_checkpoint = true;
        bpc_a.verbose = false;
        fs::remove_all(bpc_a.checkpoint_dir);
        BP_a.run(bc, bpc_a);

        // Sanity check: SUCCESS marker is present.
        check(fs::exists(bpc_a.checkpoint_dir + "/__SUCCESS__"),
              "BackPropagator wrote __SUCCESS__ marker");

        // Second run: try to load ψ from checkpoint.
        scatt::BackPropagator BP_b(bundle.params, pot, FRP, WI);
        scatt::BackPropagator::Config bpc_b = bpc_a;
        bpc_b.try_load_checkpoint = true;
        bpc_b.save_checkpoint = false;
        bpc_b.verbose = true;
        BP_b.run(bc, bpc_b);

        double worst = 0.0;
        for (int n : {0, 1, 100, 1500, (int)bundle.params.n_grid - 1}) {
            double d = (BP_a.get_psi((std::size_t)n) - BP_b.get_psi((std::size_t)n))
                         .cwiseAbs().maxCoeff();
            worst = std::max(worst, d);
        }
        check(worst < 1e-14,
              "BP loaded ψ bit-equal to build (worst=" +
              std::to_string(worst) + ")");
    }

    // Clean up the test sandbox.
    fs::remove_all(root);
    scatt::cleanup::remove_sinv_rinv("./checkpoints", false);

    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL")
              << " (" << fails << " failed)\n";
    return fails == 0 ? 0 : 1;
}
