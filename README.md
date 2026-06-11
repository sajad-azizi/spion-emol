# static-exchangeHF

Molecular photoionization in the static-exchange Hartree–Fock approximation,
with checkpoint-friendly scattering, full per-stage benchmarking, and a
post-processing pipeline that produces σ(ω), τ(ω), β(ω) and gauge
diagnostics.

Target production system: **C₈F₈ photoionization** (continuum HOMO,
anion-initial-state dipole matrix element) at `l_cont ≈ 100`,
`r_max ≈ 100` bohr on LRZ.

---

## Pipeline

```
  preprocess_molden  →  scattering  →  gather_dipoles.py  →  cross_section_delay.py
   (C++ HDF5)           (C++ HDF5)       (Python .dat)         (Python .dat + .png)
        │                    │
        │                    ├── persistent:   $WORK/pot_<hash>/
        │                    │                 $WORK/dipole_<hash>_<scan>/ikNNNN.h5
        │                    └── per-ik scratch: $SCRATCH/{sinv,rinv,psi}_<hash>_ikNNNN/
        └── output: $WORK/<stem>.preproc.h5
```

Each stage is a separate executable / script with its own `--help`.  Data
flows through HDF5 for the heavy stuff (potentials, orbitals, wavefunctions,
dipole matrix elements) and plain `.dat` for small human-readable outputs
(cross sections, β, τ, diagnostics).

### 1. Preprocessing (`preprocessing/build/preprocess_molden`)

Converts a Psi4 molden file (from an HF/SCF calculation) into the
single-center-expanded HDF5 the scattering code consumes.  Computes:
V_en (multipole), V_H (radial Poisson), V_x LDA (*diagnostic only — the
scattering code ignores it; exchange is handled at the non-local K level*),
and SCE coefficients of all occupied MOs plus the anion SOMO (if provided
as `--initial-state-molden`).

```bash
# neutral C8F8, full SCE expansion
preprocess_molden c8f8.molden \
    --lmax 300 --dr 0.005 --rmax 100 \
    --exchange none --origin com \
    --initial-state-molden c8f8_anion_in_thf.molden
# output: $WORK/c8f8.preproc.h5
```

### 2. Scattering (`build/scattering`)

Energy-scan driver.  Given a preproc HDF5 and `(ik_min, ik_max, dk)`, runs
the full static-exchange pipeline for each ik in the scan:

```
Potentials → SchurInverter → ForwardRPropagator → KMatrixExtractor
           → BackPropagator → AsymptoticAmplitudes → DipoleMatrixElement
           → DipoleWriter (HDF5)
```

Pot is built **once** (energy-independent) and checkpointed to `$WORK`.
Sinv, Rinv, ψ live in `$SCRATCH`.  After each ik is written, sinv and rinv
are deleted; ψ is kept (post-processing optionally reuses it for
continuum–continuum matrix elements).

```bash
scattering $WORK/c8f8.preproc.h5  1 200 0.01  \
    --lmax-cont 100 \
    --work $WORK --scratch $SCRATCH \
    --omp-threads 16 --bench-out $WORK/bench.dat
```

Automatic choices:
- `n_transition` from χ-cutoff scan (orbitals set to zero past this ir)
- `n_keep_hi` (main ψ store) = `n_transition - 1`
- `n_asym = 300` (in-memory tail for asymptotic fit)
- StoragePlanner assigns MEMORY/DISK per stage based on available RAM

#### Closed-channel handling at high ℓ

For high-ℓ channels where `k·r_max < ℓ` the channel sits entirely under
its centrifugal barrier in the fit window — the 2×2 normal matrix of
`AsymptoticAmplitudes` becomes singular and the run **aborts by default**:

```
AsymptoticAmplitudes: singular normal matrix for ℓ=91
  (fit window may be too narrow or on a Bessel node).  Extend the window or shift it.
```

Two safe fixes (no approximation):
1. **Increase `r_max`** so `k·r_max > l_cont` for every `ik` in the scan,
   e.g. `r_max ≥ l_cont / k_min`.  No physics is changed.
2. **Reduce `l_cont`** so all channels are open.

Or, **opt in** to a controlled approximation:

```bash
scattering ... --allow-closed-channels
```

This substitutes `A_{μν} = δ_{μν}, B_{μν} = 0` for the affected ℓ's, so
their K-matrix block is exactly 0.  Result is **equivalent** to running
with `l_cont` reduced to the highest open ℓ value.  Safe IFF the
photoionization observable is already converged in `l_cont` within the
open-ℓ subset — verify by running a smaller `l_cont` reference where
everything is open and comparing σ, τ.  Default is OFF; you opt in
deliberately and own the convergence justification.

### 3. Post-processing

```bash
# collect per-channel .dat files from the HDF5 scan dir
python postprocessing/gather_dipoles.py  $WORK/dipole_<hash>_<scan>/

# compute σ(ω), τ(ω), β(ω), gauge diagnostics + plots
python postprocessing/cross_section_delay.py  $WORK/gathered_<scan>/  \
    --xaxis omega
```

Outputs (`.dat` + `.png`):
- `delay_xsec_len_homo.dat`, `delay_xsec_vel_homo.dat` — σ, τ, σ_raw per polarization
- `beta_asymmetry.dat` — β_L(ω), β_V(ω)
- `gauge_diagnostics.dat` — Q_σ, Δφ_rms
- `cross_section_delay_both.png`, `gauge_diagnostics.png`,
  `gauge_overlay.png`, `beta_asymmetry.png`

---

## Observables

All in atomic units; conversion factors at top of `cross_section_delay.py`.

| quantity | formula | units |
|---|---|---|
| σ_L | (8π ω / c k) Σ\|d_lm^L\|² | bohr² (→ Mb via × 28.003) |
| σ_V | (8π / c k ω) Σ\|d_lm^V\|² | bohr² |
| τ | Im[ Σ d* dd/dE ] / Σ\|d\|² | au of time (→ as via × 24.188) |
| β | 5 ⟨P_2⟩ / σ  (numerical angle integral) | dimensionless, ∈ [-1, 2] |
| Q_σ | Σ\|d^V\|² / (ω² Σ\|d^L\|²) | = 1 for exact H |
| Δφ_rms | weighted-RMS of arg(d^V / d^L) | rad (= 0 for exact H) |

Conventions: real-Y spherical harmonics with q-map x=+1, y=−1, z=0;
u/r convention χ=r·F; incoming-wave Ψ⁻ = (A−iB)⁻†·ψ_β.

---

## Build

### macOS (dev)

```bash
brew install cmake eigen hdf5 gsl libomp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j8
```

No MKL available on macOS (typically) → Eigen native; dense LA is
single-threaded per step; outer-loop parallelism in SchurInverter still
uses all cores.

### LRZ (production)

```bash
module load intel mkl cmake eigen hdf5 gsl
export WORK=/hppfs/work/<project>/<user>/se_work
export SCRATCH=/hppfs/scratch/<project>/<user>/se_scratch
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

The CMake finds MKL via `MKL_DIR` (set by `module load mkl`) and defines
`SCATT_HAS_MKL=1` + `EIGEN_USE_MKL_ALL`.  Eigen routes its dense LA through
MKL (parallel LU/Cholesky/triangular-solve), and the direct LAPACK wrapper
`inverse_general()` in `src/scatt/LapackInverse.hpp` calls
`LAPACKE_dgetrf + LAPACKE_dgetri` directly for the hot per-ir inverses.

The scattering executable prints the linear-algebra state at startup:

```
[threading]  omp_get_max_threads=16  Eigen::nbThreads=16
             EIGEN_HAS_OPENMP=yes    MKL=yes
```

Expect **2–4× total speedup** vs the Eigen-native macOS dev path at the
same thread count, because MKL parallelizes the serial hotspots
(FRP::linsolve_pLU, BP::wi_apply).

### Benchmarking output

At the end of each scan, `bench.dat` (tab-separated) and a formatted table
to stdout give per-stage totals in **wall time + peak RSS + RAPL energy
(Linux only)**.  35+ line items — every hot routine has internal sub-item
timers.  Use the ratio `(summed-ns across threads) / (wall-ns)` to see
effective parallelism per stage.

Typical production bench on LRZ (C₈F₈, l_cont=100, n_grid=20001, MKL
16-thread) will look something like:

```
[bench]  wall=1.4 h  peak_rss=120 GB  omp_threads=16  rapl_energy=820 kJ

ForwardRPropagator::run     ~55%
BackPropagator::run         ~30%
SchurInverter::build        ~5% (parallel over ir)
Potentials::build            <1% (built once, reused across ik)
DipoleMatrixElement          <1%
[bunch of other stages]      <1% each
```

---

## Tests

```bash
cd build && ctest --output-on-failure
```

20 tests, ~2 minutes on macOS.  Key ones:
- `test_freeparticle_dipole` — dipole formula bit-identity check vs analytic
- `test_backprop_asym_buffer` — full-range vs split-range ψ: bit-equal
- `test_lapack_inverse` — MKL-path bit-match Eigen
- `test_dipole_io` — DipoleWriter HDF5 round-trip
- `test_storage_planner` — memory/disk decision logic
- plus all the per-stage end-to-end tests for H₂ and H₂O

Any test failure means do not trust the output; run `--output-on-failure`
to see what broke.

---

## Directory layout

```
static_exchangeHF/
├── preprocessing/        Psi4-molden → HDF5 (separate CMake project)
│   ├── src/main_preprocess.cpp
│   └── build/preprocess_molden
├── src/
│   ├── main.cpp          Scattering driver (CLI + ik loop + bench)
│   ├── io/
│   │   └── HDF5Reader.hpp
│   └── scatt/
│       ├── Parameters.hpp        config + energy grid
│       ├── EnergyGrid.hpp        ik → k → E conversion
│       ├── StoragePlanner.hpp    auto MEMORY/DISK per stage
│       ├── SystemMemory.hpp      RAM detection
│       ├── MoleculeHash.hpp      stable 64-bit hash of molecule+grid
│       ├── Potentials.hpp        V_en + V_H + V_pol; Stats
│       ├── PotentialStorage.hpp  chunked HDF5-compatible backend
│       ├── ExchangeCoupling.hpp  on-fly Q_ψf (no Q_fψ; use .transpose())
│       ├── SchurInverter.hpp     parallel-over-ir; Stats
│       ├── ForwardRPropagator.hpp  serial recursion; Stats
│       ├── WInverseOperator.hpp  W^(-1) application
│       ├── BackPropagator.hpp    main ψ + asym-buffer trick; Stats
│       ├── KMatrixExtractor.hpp  ψ boundary from K-matrix
│       ├── AsymptoticAmplitudes.hpp  A, B least-squares fit
│       ├── DipoleMatrixElement.hpp  length + velocity gauges; Stats
│       ├── DipoleIO.hpp          HDF5 writer + reader
│       ├── LapackInverse.hpp     MKL path with Eigen fallback
│       └── Bench.hpp             BenchReport + ProfileScope
│
├── postprocessing/
│   ├── gather_dipoles.py         HDF5 scan → per-channel .dat
│   └── cross_section_delay.py    .dat → σ(ω), τ(ω), β(ω), diagnostics, plots
│
├── validation/                   regression reports (H₂, H₂O)
└── version_0/                    reference implementation (archived)
```

---

## Known limitations

- **Basis = cc-pVDZ.**  All validation to date uses this small basis, which
  is sufficient for qualitative shape but over-predicts magnitude by ~30–50 %
  vs experiment.  Production runs should use `aug-cc-pVTZ` or larger for
  better continuum support.
- **No electron correlation beyond SCF.**  SE-HF = frozen-core HF + K
  exchange.  Near-threshold features (shake-up, post-Koopmans correlation)
  are not captured.
- **Static polarization only** (`V_pol(r) ∝ −α_iso/(2r⁴)` with Gianturco
  damping).  Dynamic / frequency-dependent polarization is not modeled.
- **No relativistic corrections.**
- **Asymptotic fit requires `k · r_max ≳ 4`.**  Below that, the asymptotic
  LSQ becomes rank-degenerate and `AsymptoticAmplitudes::extract` throws
  with a clear error message.  Use a larger `--rmax` at preproc stage.

---

## Validation summary

See `validation/*/report.md`.

**H₂** (cc-pVDZ, V_pol with α=2.9 au): peak σ_L = 12.6 Mb (lit 8.5 Mb; SE-HF
+48 % — textbook).  Peak ω=21 eV (lit ~18 eV).  β ≈ 2.00 across scan
(atomic s→p limit, matches experiment).  **10/10 validation checks pass.**

**H₂O** (cc-pVDZ, V_pol with α=9.5 au): shape correct in the 20–40 eV
window (±50 % of lit).  Near-threshold under-predicts by ~20× because
1b₁ (l_i=1, m_i=±1) reaches only l_f=2 for x/z polarizations → Wigner-k⁵
suppression.  Known SE-HF limitation, not a code bug.

---

## Credits / provenance

Derived from `version_0/` (archived in this repo).  Re-written ground-up
for memory and checkpoint discipline; physics conventions re-verified
against the project PDF document on the coupled-channel Numerov
formulation (`memory/project_scattering_equations.md`).

License: for academic use within the research group.

---

## Quick sanity command

```bash
# full pipeline for H2, ~1 minute on 4 threads:
preprocess_molden preprocessing/reference_data/h2_ccpvdz_sph.molden \
    --lmax 24 --dr 0.005 --rmax 40 --exchange none \
    --polarizability 2.9 0 0  0 2.9 0  0 0 2.9
build/scattering $WORK/h2_ccpvdz_sph.preproc.h5  5 12 0.1  \
    --lmax-cont 6 --omp-threads 4
python postprocessing/gather_dipoles.py  $WORK/dipole_<hash>_*/
python postprocessing/cross_section_delay.py  $WORK/gathered_*/
```

Expect σ_L peak ≈ 12–15 Mb at ω ≈ 21 eV, β ≈ 2.0, Q_σ ≈ 0.4, fit_residual
≲ 10⁻⁶ at every ik.
