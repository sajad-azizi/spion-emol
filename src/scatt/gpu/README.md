# `src/scatt/gpu/` — multi-vendor GPU backends

The public GPU stepper classes (`scatt::GpuContext`,
`scatt::GpuForwardStepper`, `scatt::GpuSinvStepper`,
`scatt::GpuBackStepper`) live in `src/scatt/GpuPropagate.hpp` and are
**vendor-agnostic by design** (every state-holding member is hidden
behind a `std::unique_ptr<Impl>` pimpl).  The actual GPU kernels for
each vendor live in this directory:

```
src/scatt/gpu/
├── sycl/                     # Intel oneAPI / oneMKL — SCATT_WITH_SYCL=ON
│   └── GpuPropagate.cpp      #   per-step dgemm / dgetrf / dgetri / dsytrf / dsytri
├── cuda/                     # NVIDIA cuBLAS / cuSOLVER — SCATT_WITH_CUDA=ON
│   └── GpuPropagate.cu       #   (currently SCAFFOLD ONLY; see status below)
└── README.md                 # this file
```

Future additions go in their own sibling directories:
- `gpu/kokkos/` — Kokkos Kernels (portable fallback for ad-hoc backends).
- `gpu/hip/`    — AMD HIP / rocBLAS / rocSOLVER (when needed).

## Build flags

| Flag | Default | Backend selected |
|---|---|---|
| `SCATT_WITH_SYCL=ON` | OFF | Intel oneAPI / oneMKL (SYCL DPC++ via `icpx`) |
| `SCATT_WITH_CUDA=ON` | OFF | NVIDIA CUDA / cuBLAS / cuSOLVER |

**Build-time backend selection** today: the two flags are mutually
exclusive (CMake errors if both are ON).  Each backend implements the
same `struct GpuBackStepper::Impl` (and the three sibling classes')
using its own kernels — compiling both at once would yield duplicate-
symbol link errors, so CMake picks exactly one source file (either
`gpu/sycl/GpuPropagate.cpp` or `gpu/cuda/GpuPropagate.cu`) for the
`scatt_potential` library.

CPU-only builds (neither flag set) get the SYCL .cpp's `#else` branch,
which provides no-op host stubs — `GpuContext::gpu_available()` returns
false and the production code falls back to the CPU paths in FRP / BP
/ DipoleMatrixElement (same as today).

**Runtime multi-vendor dispatch** (one binary, both Intel and NVIDIA
devices visible, factory picks at construction) is a follow-up
refactor.  It requires making `struct Impl` an abstract base, deriving
`ImplSycl` and `ImplCuda` from it, and adding a factory that probes
the device vendor.  Deferred until both backends are implemented and
tested independently — see `future_plan.md` item 3 for the queued
work.  For now, build a separate binary per cluster (which is also
what most HPC sites use in practice).

## Status (current commit)

- **SYCL backend**: FULL implementation, production-validated.  Moved
  here from `src/scatt/GpuPropagate.cpp` (file relocate only, zero
  logical change).
- **CUDA backend**: SCAFFOLD ONLY.  The translation unit compiles
  (gated on `SCATT_WITH_CUDA=ON`) but does not yet provide `Impl`
  bodies — building with `-DSCATT_WITH_CUDA=ON` today will fail at
  link time because the public class constructors expect a working
  `struct Impl`.  Real cuBLAS / cuSOLVER kernels arrive in a follow-up
  commit (tracked in `future_plan.md` item 3, "Phase 4: CUDA Impl
  bodies").  Until then, `-DSCATT_WITH_CUDA=ON` is intended for
  reviewing the CMake wiring, not for producing a working binary.

## What does NOT change with future GPU work

- The public classes in `src/scatt/GpuPropagate.hpp`.  Their signatures
  are the contract that FRP / BP / BackPropagator / SchurInverter call
  against, and **must remain identical** while adding backends.
- The call sites in `ForwardRPropagator.cpp`, `BackPropagator.cpp`,
  `SchurInverter.cpp`, `DipoleMatrixElement.cpp` — they all hold
  `std::unique_ptr<GpuXxxStepper>` and dispatch via the public class
  methods.  Adding a new backend means new files under `gpu/<vendor>/`
  and a few lines in the factory; no changes to the call sites.

## Validation

For each backend, the gate tests are:

- `test_gpu_propagate`: the back-prop sweep on a small fixture; output
  ψ checked element-wise against the CPU path tolerance.
- `test_gpu_steppers`: per-step bit-equivalence of the FRP forward
  step.
- `test_gpu_sinv`: per-step bit-equivalence of `Sinv = (A - B·D⁻¹·Bᵀ)⁻¹`.

When the CUDA backend is filled in, the additional gate is **cross-vendor
bit-equivalence to within the dgemm roundoff floor** — running the same
input through SYCL and CUDA backends on the same node should produce
results agreeing to `ε_mach × N` per element (i.e. ULP-level).  This is
the same tolerance the existing tests already use.
