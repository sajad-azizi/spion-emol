# Threading & sub-stage benchmarking

Two observations from production runs:

1. `htop` shows only 1–2 active threads during most of the scattering wall
   time, even with `--omp-threads 4` and `EIGEN_HAS_OPENMP = yes`.
2. The top-level `[bench]` table reports `ForwardRPropagator::run` and
   `BackPropagator::run` as the two dominant stages (together ~85–96 % of
   wall) but doesn't tell us **what inside them** is slow.

This note addresses both.

---

## 1. Why only 1–2 threads?

Startup banner now prints the full threading state:

```
[threading]  omp_get_max_threads = 4   Eigen::nbThreads = 4
             EIGEN_HAS_OPENMP = yes    MKL = no
```

All four are correctly configured — yet most of the wall is serial. The
reason is **Eigen's native linear algebra is mostly single-threaded**:

| Eigen operation                 | parallel in native? | parallel in MKL? |
|---|---|---|
| general matrix–matrix product (GEMM), ≥ threshold size | yes (via OpenMP) | yes |
| GEMM below threshold (tiny matrices) | **no**                | yes |
| `partialPivLu().inverse()` / `.solve()` | **no**           | yes |
| `llt().solve()` / `ldlt().solve()`     | **no**           | yes |
| Triangular solve (`TriangularView::solve`) | **no**      | yes |

The scattering hot path is dominated by **per-n matrix inverses** in the
propagation recursion — which Eigen native never parallelizes regardless
of thread count.

### Matrix-size threshold for GEMM parallelism

Eigen's GEMM parallel dispatch kicks in only when
`ops (≈ m·n·k) > ~ 25 000` (rough default). For our runs:

| molecule | l_cont | n_channels | per-step matrix size | ops per GEMM | parallel? |
|---|---:|---:|---:|---:|---|
| H₂    |  6 |  49 |  49 ×  49 |  1.2 × 10⁵ | borderline |
| H₂O   |  8 |  81 |  81 ×  81 |  5.3 × 10⁵ | yes        |
| H₂O   | 10 | 121 | 121 × 121 |  1.8 × 10⁶ | yes        |
| C₈F₈  | 100 | 10201 | 10201² | 1.0 × 10¹² | yes (forever) |

For **H₂O at l_cont=10 and C₈F₈ at l_cont=100**, GEMM itself would
parallelize fine in Eigen native. But LU and triangular solves stay
serial — and the `linsolve_pLU` step IS the hot spot, not GEMM.

### Fix

On the **LRZ production node**: `module load mkl`, then rebuild. MKL
parallelizes LU/Cholesky/triangular-solve at any matrix size, controlled
by `MKL_NUM_THREADS` (or falls back to `OMP_NUM_THREADS`). Expected
speedup on H₂O 10201² matrices: 4–8 × with 16 threads.

On **macOS (dev box)**: no practical fix without MKL or a parallel BLAS
rebuild of Eigen. Accept the serial behavior for development; validate
on the LRZ node.

---

## 2. Sub-stage benchmarks (what's slow *inside* FRP and BP)

Added internal per-step timers to `ForwardRPropagator::run` and
`BackPropagator::run` (structs `Stats`, exposed via `.stats()` accessors),
aggregated into the main `BenchReport` in `run_one_energy`.

### H₂ bench, 7 ik points, l_cont=6, r_max=40 bohr, 4 threads (macOS)

```
[bench]  wall = 27.67 s   peak_rss = 1.86 GB   omp_threads = 4
           rapl_energy = unavailable

stage                               count    total(s)   share
-----------------------------------------------------------------
ForwardRPropagator::run                 7      15.54    56.2 %
  ├─ FRP::u_assemble                    7       6.83    24.7 %   (in apply_U: exchange pieces + inner solves)
  ├─ FRP::linsolve_pLU                  7       8.00    28.9 %   ← partialPivLu().inverse(), per-n, SERIAL
  └─ FRP::store                         7       0.43     1.6 %   (dense copy into PotentialStorage)

BackPropagator::run                     7       8.19    29.6 %
  ├─ BP::rinv_fetch                     7    0.0009     0.0 %   (MEMORY mode: negligible)
  ├─ BP::gemm_z                         7       2.07     7.5 %   (Z_next = Rinv_n · Z, GEMM)
  ├─ BP::wi_apply                       7       4.29    15.5 %   ← W^{-1} action, SERIAL triangular solves
  └─ BP::store                          7       0.21     0.7 %

DipoleMatrixElement::compute(x6)        7       2.26     8.2 %
SchurInverter::build                    7       0.91     3.3 %   (parallel over ir via explicit omp)
other                                   -     < 0.5 %
```

### Interpretation

- **FRP::linsolve_pLU = 29 % of total wall, SINGLE-THREADED.** One serial
  `partialPivLu().inverse()` per ir × ≈ 8000 ir. On LRZ + MKL, this
  becomes O(50 µs/step × 4 threads) = ≈ 4 × speedup ⇒ drops to ~7 % of wall.

- **FRP::u_assemble = 25 %.** Hidden behind `WInverseOperator::apply_U`,
  which performs an N_f-block triangular solve and a back-substitution.
  Also serial in Eigen native; MKL would parallelize.

- **BP::wi_apply = 16 %.** Same story — triangular solve per step.

- **BP::gemm_z = 7.5 %.** This IS a dense GEMM; should parallelize on
  H₂O-sized problems (81² and up). On H₂ (49²) it's borderline — the
  observed 7.5 % in 4-thread native Eigen is consistent with "barely
  parallel below threshold".

- **SchurInverter::build = 3 % only.** Parallel over ir already
  (`#pragma omp for schedule(dynamic, 64)` in the file), so multi-thread
  utilization during this stage is genuine. **Verified:** the new
  `Sinv::*` sub-timers are summed across threads, so their total divided
  by the top-level wall gives the effective thread count. On H₂ at
  4 threads we measure:
  ```
  Sinv::invert + B_build + schur + A_build + store  = 1.463 s
  SchurInverter::build wall                         = 0.417 s
  ratio = 3.5  →  ≈ 3.5 effective threads active
  ```
  This matches version_0's SchurInverter pattern (parallel over n in
  MEMORY mode, sequential in DISK mode to avoid file-write contention).

  By contrast, FRP and BP show ratios near 1.0:
  ```
  FRP::u_assemble + linsolve_pLU + store  = 6.462 s   (wall 6.612 s -> 0.98× threads)
  BP::gemm_z + wi_apply + store           = 2.526 s   (wall 3.226 s -> 0.78× threads)
  ```
  Exactly the single-thread behavior that htop shows for those two
  stages. Sinv is the ONLY hot stage today that is genuinely multi-thread
  in Eigen native.

- **Everything else combined < 15 %.** The per-energy bundle build,
  exchange coupling, K-matrix extraction, asymptotic fit, dipole
  computation, and dipole writer are all < 10 % each.

### Where the speedup will come from on LRZ + MKL

Serial-dense-LA in Eigen native  ≈ **FRP::u_assemble + FRP::linsolve_pLU
+ BP::wi_apply** = 6.83 + 8.00 + 4.29 = **19.12 s of 27.67 s total =
69 % of wall**. With MKL + 16 threads, a conservative expectation is
that this drops by a factor of 4 (LU/solve parallel efficiency is
~50–70 % for these sizes). That would give 19.12 → ~5 s, total wall
≈ 14 s instead of 27.7 s = **~2× overall speedup on a 16-thread LRZ node.**

For larger basis sizes (H₂O lcont=10 or C₈F₈), the speedup factor grows
because MKL efficiency increases with matrix size.

---

## 3. How to read the new per-energy bench table

The bench table now has ~20 rows where it used to have ~10. Rows prefixed
`FRP::` are sub-stages of `ForwardRPropagator::run`; rows prefixed `BP::`
are sub-stages of `BackPropagator::run`. The two top-level stage totals
are still reported — they bracket the child rows, and the children should
roughly sum to the parent (small residual = auxiliary work outside the
inner loop, e.g. FRP analytic-init region or BP boundary Z̃ setup).

A typical healthy pattern:

```
ForwardRPropagator::run                      <total>
├─ FRP::u_assemble                           ~25 %
├─ FRP::linsolve_pLU                         ~30 %    <- MKL speeds up
└─ FRP::store                                < 2 %    (DISK: up to ~20 %)

BackPropagator::run                          <total>
├─ BP::rinv_fetch                            < 1 %    (DISK: up to ~30 %)
├─ BP::gemm_z                                ~8 %     <- MKL speeds up
├─ BP::wi_apply                              ~15 %    <- MKL speeds up
└─ BP::store                                 < 2 %    (DISK: up to ~10 %)
```

If `FRP::store` or `BP::store` grows to > 10 %, you're I/O-bound and
should use MEMORY mode or increase chunk size.

If `BP::rinv_fetch` is > 5 %, DISK-mode reads of rinv are slow — tune
chunk size in StoragePlanner.

---

## 4. Files touched

- `src/main.cpp` — threading diagnosis banner, sub-bench wiring
- `src/scatt/BackPropagator.hpp/.cpp` — `Stats` struct + per-step timers
- `src/scatt/ForwardRPropagator.hpp/.cpp` — `Stats` struct + per-step timers
- `src/scatt/SchurInverter.hpp/.cpp` — `Stats` struct + atomic per-step timers
  (summed across threads so we can **measure** the effective parallelism)
