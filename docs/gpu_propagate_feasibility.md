# GPU-accelerated forward / backward propagation — feasibility

Audit of version_0's SYCL implementation (in
`version_0/src/propagating_sycl.hpp` + `RenormalizedNumerov.cpp`) and a
practical assessment of porting it onto this branch.

---

## 1. What version_0 actually does on the GPU

Two classes in `propagating_sycl.hpp`:

### `ForwardPropagatorGPU::propagateStep(W_inv, R_prev_inv_out)`

Per radial point `n`:
1. **Upload** `W_inv(n)`  (N_total × N_total × 8 B) from host to device.
2. **On GPU**:
   - `U = 12·W_inv − 10·I` (simple elementwise kernel)
   - `R_current = U − R_prev_inv`  (elementwise)
   - **Invert** `R_current` via `oneapi::mkl::lapack::getrf` + `getri`
   - **Symmetrize** `R_prev_inv = 0.5·(R + Rᵀ)`
3. **Download** `R_prev_inv` to host for checkpointing.
4. Symmetry error computed on host for diagnostics.

All persistent GPU buffers are 7 matrices × N_total² + two LAPACK scratchpads.
For C₈F₈ at l_cont=100 (N_total ≈ 10201): **~6 GB GPU memory**.

### `BackpropagatorGPU::propagateStep(Rinv_n, W_inv_n, psi_n, f_n, compute_f)`

Per radial point `n`:
1. **Upload** `Rinv_n` (N_total × N_total) AND `W_inv_n` (N_total × N_total).
2. **On GPU**:
   - `Z ← Rinv_n · Z`   (oneMKL GEMM, N_total × N_psi matrix as RHS)
   - swap buffer
   - `Y = W_inv_n · Z`  (oneMKL GEMM)
   - extract top N_psi rows (ψ) + bottom N_f rows (f) via custom kernels
3. **Download** ψ (N_psi × N_psi) and optionally f (N_f × N_psi) to host.

### Driving loop (wavefunctions.cpp → RenormalizedNumerov.cpp)

Both FRP and BP are still **sequential across n** (recursion). The GPU
only parallelizes the inner matrix operations — LU+inverse for FRP, two
GEMMs for BP.

Sinv is kept on **host** and fetched per step (with a prefetch helper
`Sinv_storage_.getWithPrefetch`).  Q_psi_f (exchange coupling) is either
precomputed on host or computed on-fly per step.

---

## 2. The honest performance story

### Expected CPU baseline on LRZ (16-thread, our current code)

| stage | op cost | wall (C₈F₈ l_cont=100, ~20 000 n) |
|---|---|---:|
| FRP per step | dgetrf + dgetri on 10201² | **~60 ms** |
| FRP total | 20 000 × 60 ms | **~20 min** |
| BP per step | 2 GEMM (N×N × N×Npsi) | ~30 ms |
| BP total | 20 000 × 30 ms | ~10 min |
| **combined** | — | **~30–45 min per ik** |

### Expected GPU performance (version_0 pattern, Intel PVC)

**Per forward step** (N_total = 10201):
- dgetrf + dgetri on GPU: ~25 ms (Intel PVC H200-class device)
- Host↔device memcpy: upload W_inv 800 MB + download Rinv 800 MB = 1.6 GB / 40 GB/s PCIe = **40 ms**
- **Total per step ≈ 65 ms ≈ 10 % faster than CPU. Not 10×. Not 5×. Not even 2×.**

**Why so small a win**: the problem is PCIe-bandwidth-bound. Each step
roundtrips **1.6 GB across the bus** for W_inv + Rinv, and PCIe Gen4 x16
tops out at ~25 GB/s effective. The actual GPU math (dgetrf at 50
TFLOPS) is fast, but it's starved by the memcpy.

For backprop the story is similar — three N² matrix uploads per step.

### What would actually get a 5–10× speedup

**Keep Sinv and Rinv resident on device** in a ring/chunk cache (with a
prefetcher for the next chunk). Do `W_inv` on-fly on device from the
three on-device tensors (Sinv, B, Dinv). Download only ψ chunks (which
are ~N×Npsi, ~800 MB / 100 = 8 MB, tiny). The GPU then runs 10³–10⁴ steps
between any host touch, and you're actually compute-bound.

That's a **fundamental architecture redesign**, not a straight port.
Version_0 did NOT do this — its code is the "naive 1:1 mirror" pattern
that leaves 80 % of the PCIe traffic in place.

---

## 3. Can we port it to this branch?

### What works with minimal effort (1–2 weeks)

- **Port as-is**: take `propagating_sycl.hpp`, drop it into
  `src/scatt/GpuPropagate.hpp`, add SYCL detection to `CMakeLists.txt`
  (`find_package(IntelSYCL)` + `target_compile_options(-fsycl)`), wire
  into `ForwardRPropagator::run` and `BackPropagator::run` as an
  opt-in path behind `Config::use_gpu = true`.
- **Expected speedup**: 10–20 % on LRZ PVC. Not great.
- **Risk**: low. Always falls back to CPU if SYCL runtime missing.

### What actually helps (4–6 weeks)

- Resident Sinv + Rinv on device (chunked, prefetched).
- On-device W_inv assembly from (Sinv, B, Dinv).
- Async memcpy overlap (device compute + host-to-device next chunk).
- Only download checkpoint windows + psi_asym buffer, not every matrix.
- **Expected speedup**: 5–10× on PVC at C₈F₈ scale. Real win.
- **Risk**: higher — it's real engineering. Testing discipline critical
  (bit-equal vs CPU within 1e-10 at every ir).

### Hard requirements either way

1. **Compiler**: Intel oneAPI DPC++ (`icpx`) — NOT GCC or Clang-Clang.
   `module load intel` on LRZ has this, but the CMake build will need
   `-DCMAKE_CXX_COMPILER=icpx` and `-fsycl` in `target_compile_options`.
2. **GPU hardware**: version_0's code uses oneMKL SYCL, which targets
   **Intel GPUs only** (PVC, Battlemage, ARC). It will NOT run on NVIDIA
   or AMD without rewriting the LAPACK calls to cuBLAS / rocBLAS.
3. **GPU node access**: confirm your LRZ allocation has GPU-enabled
   nodes (SuperMUC-NG is CPU-only; SuperMUC-NG-2 / LRZ AI nodes have
   PVC or A100).
4. **Memory**: 6+ GB VRAM for the persistent buffers at l_cont=100.
   Stadard PVC (48 GB HBM) is fine; a smaller card will OOM.
5. **Regression tests**: CPU vs GPU bit-equality tests (within 1e-10)
   should be added. Numerically they're not identical (different
   LAPACK rounding) but should match to machine precision in the
   observables.

---

## 4. My honest recommendation

**Ask these two questions first:**

1. **Does your LRZ allocation have Intel GPU nodes?**
   If yes → GPU port is worth considering.
   If no (only CPU or only NVIDIA) → stop here. Don't port.

2. **How many C₈F₈-scale scans will you run over the lifetime of this
   project?**
   - **One production scan + a few small studies** → no. 30–45 min per ik
     on 16-thread MKL is tolerable for a one-shot, and a 1-week port with
     only 10 % speedup is a loss.
   - **Dozens of scans (different molecules, solvents, basis
     convergence)** → the naive port gains back 1–2 weeks of total wall
     but costs 1–2 weeks to build. Rough break-even.
   - **Production campaign with hundreds of energy points across many
     molecules** → serious port (option 2 above) justified. 5–10× wall
     reduction over a year is significant.

**If you're in case 1 or 2** (most likely given where you are in the
project): run the C₈F₈ scan on CPU MKL, publish, and come back to the GPU
port when you have a second or third molecule queued up. The CPU code
is already validated and threaded; don't pay for portability/complexity
you don't need yet.

**If you're in case 3**: the naive port (option 1) is the wrong first
step — it leaves the performance sitting on the PCIe. Go straight to the
resident-Sinv/Rinv design (option 2). I can write a detailed design +
staged implementation plan if you decide to pursue it.

---

## 5. What I've done now

- Created branch `gpu-propagate`.
- Audited version_0's GPU code: `ForwardPropagatorGPU`,
  `BackpropagatorGPU`, and the `backpropagate_sycl*` family in
  RenormalizedNumerov.cpp.
- This writeup at `docs/gpu_propagate_feasibility.md`.
- **No production code changed.**

Whenever you decide, let me know which option (if any) to pursue and
I'll either delete the branch or start a concrete port.
