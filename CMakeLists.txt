cmake_minimum_required(VERSION 3.22)
project(static_exchangeHF_scattering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# -----------------------------------------------------------------------------
# Optimization tuning.
#
# We deliberately do NOT enable -ffast-math:
#   * The R-matrix recursion exercises finite/nan checks on LU residuals and
#     needs IEEE-conformant associativity so the gold-standard test stays
#     bit-reproducible across builds.
#   * Eigen/MKL already detect -O3 and -march=native and enable SIMD paths.
# What we DO enable on Release-ish builds:
#   -O3              : full inlining + vectorization
#   -march=native    : SIMD width for this host's CPU (AVX2 / AVX-512 on LRZ)
#   -fno-math-errno  : drop errno plumbing around libm (safe; we never read errno)
#   -funroll-loops   : already implied by -O3 on most compilers but be explicit
# On Intel icpx these get replaced by -xHost (equivalent to -march=native for
# Intel silicon) and -qopt-zmm-usage=high when AVX-512 is present.
#
# Guard by build type so Debug builds stay O0 + -g for gdb stepping.
set(_scatt_is_release $<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>)
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    add_compile_options($<${_scatt_is_release}:-O3>
                        $<${_scatt_is_release}:-xHost>
                        $<${_scatt_is_release}:-fno-math-errno>)
else()
    # GCC / Clang / AppleClang.
    add_compile_options($<${_scatt_is_release}:-O3>
                        $<${_scatt_is_release}:-march=native>
                        $<${_scatt_is_release}:-fno-math-errno>
                        $<${_scatt_is_release}:-funroll-loops>)
endif()

# -----------------------------------------------------------------------------
# Where preprocessed HDF5 files live. If $WORK is set, we point both the
# preprocess_molden output and the scattering tests at it so everything stays
# in one persistent place across jobs. Otherwise we fall back to the legacy
# default (preprocessing/build/) for back-compat with existing workflows.
# -----------------------------------------------------------------------------
if(NOT PREPROC_H5_DIR)
    if(DEFINED ENV{WORK} AND NOT "$ENV{WORK}" STREQUAL "")
        file(MAKE_DIRECTORY "$ENV{WORK}")
        set(PREPROC_H5_DIR "$ENV{WORK}")
    else()
        set(PREPROC_H5_DIR "${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/build")
    endif()
endif()
message(STATUS "PREPROC_H5_DIR = ${PREPROC_H5_DIR}")
# Sanity check: if the reference fixture isn't there, tests will all throw
# `H5Fopen failed`. That's a deployment problem (preprocessing wasn't run
# yet), not a code bug, so we ONLY warn at configure time -- the test
# targets are still built.
if(NOT EXISTS "${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5")
    message(WARNING
        "H2O preproc fixture not found at:\n"
        "   ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5\n"
        "The scattering tests need it.  Build + run preprocessing first:\n"
        "   cmake -S preprocessing -B preprocessing/build && cmake --build preprocessing/build -j\n"
        "   preprocessing/build/preprocess_molden \\\n"
        "       preprocessing/reference_data/h2o_ccpvdz_sph.molden \\\n"
        "       ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5 --lmax 16 --dr 0.005 --rmax 15\n"
        "Or set WORK=... and point PREPROC_H5_DIR at the location of an\n"
        "already-built fixture (-DPREPROC_H5_DIR=...).")
endif()

# All test executables land in build/bin_test/; main executables (scattering,
# and diagnostic tools dump_*/pde_*/bench_*) stay at build/ top-level.
set(SCATT_TEST_BIN_DIR "${CMAKE_BINARY_DIR}/bin_test")
file(MAKE_DIRECTORY "${SCATT_TEST_BIN_DIR}")

# -----------------------------------------------------------------------------
# Dependency discovery. Same pattern as preprocessing/CMakeLists.txt: try
# find_package first; fall back to path HINTS including the LRZ spack env
# vars (EIGEN_BASE, HDF5_BASE, GSL_BASE, GSL_LIBDIR, FFTW_BASE).
# On LRZ user runs `module load gcc cmake eigen hdf5 gsl` and we just pick up.
# -----------------------------------------------------------------------------

# ---------- Eigen ----------
find_package(Eigen3 3.4 CONFIG QUIET)
if(NOT Eigen3_FOUND)
    find_path(EIGEN_INCLUDE_DIR
        NAMES Eigen/Dense Eigen/Core Eigen/Sparse
        HINTS ${EIGEN_ROOT} $ENV{EIGEN_ROOT} $ENV{EIGEN_DIR}
              $ENV{EIGEN_INCLUDE_DIR} $ENV{EIGEN_BASE}
              /opt/homebrew/include/eigen3 /usr/include/eigen3 /usr/local/include/eigen3
        PATH_SUFFIXES eigen3 include/eigen3 include
        REQUIRED)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_INCLUDE_DIR}")
    set(SCATT_EIGEN_STATUS "found (path=${EIGEN_INCLUDE_DIR})")
    message(STATUS "Eigen3     : FOUND via path HINTS  ${EIGEN_INCLUDE_DIR}")
else()
    set(SCATT_EIGEN_STATUS "found (CONFIG, ${Eigen3_DIR})")
    message(STATUS "Eigen3     : FOUND via find_package CONFIG  ${Eigen3_DIR}")
endif()

# ---------- HDF5 ----------
find_package(HDF5 COMPONENTS C QUIET)
if(NOT HDF5_FOUND)
    find_path(HDF5_INCLUDE_DIR NAMES hdf5.h
        HINTS ${HDF5_ROOT} $ENV{HDF5_ROOT} $ENV{HDF5_DIR} $ENV{HDF5_BASE}
              /opt/homebrew/include /usr/include /usr/local/include
        PATH_SUFFIXES include REQUIRED)
    find_library(HDF5_C_LIB NAMES hdf5 hdf5_serial
        HINTS ${HDF5_ROOT} $ENV{HDF5_ROOT} $ENV{HDF5_DIR}
              $ENV{HDF5_BASE} $ENV{HDF5_LIBDIR}
              /opt/homebrew/lib /usr/lib /usr/local/lib
        PATH_SUFFIXES lib lib64 REQUIRED)
    add_library(hdf5::hdf5 UNKNOWN IMPORTED)
    set_target_properties(hdf5::hdf5 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}"
        IMPORTED_LOCATION             "${HDF5_C_LIB}")
    set(SCATT_HDF5_STATUS "found (path HINTS, ${HDF5_C_LIB})")
    message(STATUS "HDF5       : FOUND via path HINTS  ${HDF5_C_LIB}")
else()
    if(NOT TARGET hdf5::hdf5)
        add_library(hdf5::hdf5 INTERFACE IMPORTED)
        set_target_properties(hdf5::hdf5 PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES      "${HDF5_LIBRARIES}")
    endif()
    set(SCATT_HDF5_STATUS "found (CONFIG, version ${HDF5_VERSION})")
    message(STATUS "HDF5       : FOUND via find_package CONFIG  version=${HDF5_VERSION}")
endif()

# ---------- GSL (3j symbols for Gaunt coefficients) ----------
find_package(GSL QUIET)
if(NOT GSL_FOUND)
    find_path(GSL_INCLUDE_DIR NAMES gsl/gsl_sf_coupling.h
        HINTS ${GSL_ROOT} $ENV{GSL_ROOT_DIR} $ENV{GSL_BASE}
              /opt/homebrew/include /usr/include /usr/local/include
        PATH_SUFFIXES include REQUIRED)
    find_library(GSL_LIB NAMES gsl
        HINTS ${GSL_ROOT} $ENV{GSL_ROOT_DIR} $ENV{GSL_BASE} $ENV{GSL_LIBDIR}
              /opt/homebrew/lib /usr/lib /usr/local/lib
        PATH_SUFFIXES lib lib64 REQUIRED)
    find_library(GSL_CBLAS_LIB NAMES gslcblas cblas
        HINTS ${GSL_ROOT} $ENV{GSL_ROOT_DIR} $ENV{GSL_BASE} $ENV{GSL_LIBDIR}
              /opt/homebrew/lib /usr/lib /usr/local/lib
        PATH_SUFFIXES lib lib64)
    add_library(GSL::gsl UNKNOWN IMPORTED)
    set_target_properties(GSL::gsl PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${GSL_INCLUDE_DIR}"
        IMPORTED_LOCATION             "${GSL_LIB}"
        INTERFACE_LINK_LIBRARIES      "${GSL_CBLAS_LIB}")
    set(SCATT_GSL_STATUS "found (path HINTS, ${GSL_LIB})")
    message(STATUS "GSL        : FOUND via path HINTS  ${GSL_LIB}")
else()
    set(SCATT_GSL_STATUS "found (CONFIG, version ${GSL_VERSION})")
    message(STATUS "GSL        : FOUND via find_package CONFIG  version=${GSL_VERSION}")
endif()

option(SCATT_WITH_SYCL "Build SYCL/oneMKL GPU offload path (Intel oneAPI DPC++)." OFF)
option(SCATT_WITH_CUDA "Build CUDA / cuBLAS / cuSOLVER GPU offload path (NVIDIA)." OFF)

# For now the two GPU backends are MUTUALLY EXCLUSIVE per binary.  Each
# implements the SAME GpuPropagate.hpp signatures with its own kernels
# (oneMKL or cuBLAS/cuSOLVER), and CMake picks one via the source-list
# gating below.  A future refactor (see future_plan.md item 3) will
# introduce a runtime factory + abstract Impl so a SINGLE binary can
# dispatch to either backend based on the detected device; deferred until
# we have both backends implemented and tested independently.
if(SCATT_WITH_SYCL AND SCATT_WITH_CUDA)
    message(FATAL_ERROR
        "SCATT_WITH_SYCL and SCATT_WITH_CUDA cannot both be ON in the same\n"
        "build today (runtime multi-vendor dispatch not yet implemented).\n"
        "Pick one for the binary you are about to build:\n"
        "  -DSCATT_WITH_SYCL=ON   # Intel PVC / oneAPI nodes (LRZ Phase 2)\n"
        "  -DSCATT_WITH_CUDA=ON   # NVIDIA nodes (H100, A100, ...)")
endif()


# ---------- Optional Intel MKL ----------
# When MKL is found we enable EIGEN_USE_MKL_ALL, which routes Eigen's dense
# decompositions (partialPivLu, llt, ldlt, bdcsvd, ...) through MKL's LAPACKE
# (dgetrf/dgetri, dpotrf/dpotri, ...). Transparent to call sites -- we keep
# writing plain Eigen code. Fallback is Eigen's native implementations.
#
# IMPORTANT: force LP64 (32-bit MKL_INT) interface.  Eigen 3.4's LAPACKE
# bindings (e.g. ColPivHouseholderQR_LAPACKE.h) hard-code `int*` for IPIV.
# If MKLConfig.cmake picks ILP64 (the LRZ mkl module sometimes defaults to
# ilp64 when MKL_INTERFACE env var is set), you get:
#   error: cannot initialize a variable of type 'long long *' with an
#          rvalue of type 'Scalar *' (aka 'int *')
# The fix is to pin LP64 BEFORE find_package(MKL), and to make sure
# MKL_ILP64 is NOT defined when compiling Eigen headers.
set(MKL_INTERFACE   "lp64"          CACHE STRING "MKL integer interface (lp64/ilp64). Must be lp64 for Eigen.")
set(MKL_THREADING   "intel_thread"  CACHE STRING "MKL threading (sequential/intel_thread/tbb). intel_thread + mkl_set_dynamic(1) gives parallel MKL in serial sections AND 1-thread MKL inside our omp parallel regions -- the right trade-off.")
set(MKL_LINK        "dynamic"       CACHE STRING "MKL linking (dynamic/static).")
#
# Detection order:
#   1. find_package(MKL CONFIG)  -- oneAPI CMake target MKL::MKL
#   2. $ENV{MKLROOT}             -- LRZ `module load mkl` path
#   3. none -> Eigen native
#
# When SCATT_WITH_SYCL=ON we SKIP find_package(MKL) entirely: the icpx
# driver's -qmkl=parallel below handles MKL linkage (LP64 + OpenMP + SYCL
# BLAS/LAPACK variants).  Mixing find_package's MKL::MKL link list with
# -qmkl caused the "undefined reference to sycl::..::submit_with_event_impl"
# ABI mismatch on SuperMUC-NG Phase 2, because find_package picked up a
# generic MKL while libmkl_sycl_blas.so expected the oneAPI libsycl.
set(MKL_FOUND FALSE)
if(NOT SCATT_WITH_SYCL)
    find_package(MKL CONFIG QUIET)
endif()
if(SCATT_WITH_SYCL)
    message(STATUS "MKL        : detection SKIPPED (SCATT_WITH_SYCL=ON; -qmkl=parallel handles linkage, set below)")
    set(SCATT_HAS_MKL FALSE)
    set(SCATT_MKL_STATUS "will link via -qmkl=parallel (see SYCL block below)")
elseif(MKL_FOUND)
    set(SCATT_HAS_MKL TRUE)
    set(SCATT_MKL_STATUS "found (CONFIG, version ${MKL_VERSION}, ${MKL_INTERFACE}/${MKL_THREADING})")
    message(STATUS "MKL        : FOUND via find_package CONFIG  version=${MKL_VERSION}  interface=${MKL_INTERFACE}  threading=${MKL_THREADING}")
elseif(DEFINED ENV{MKLROOT})
    set(MKL_BASE $ENV{MKLROOT})
    find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS ${MKL_BASE}/include REQUIRED)
    find_library(MKL_INTEL_LP64 mkl_intel_lp64
                 HINTS ${MKL_BASE}/lib/intel64 ${MKL_BASE}/lib REQUIRED)
    find_library(MKL_SEQUENTIAL mkl_sequential
                 HINTS ${MKL_BASE}/lib/intel64 ${MKL_BASE}/lib REQUIRED)
    find_library(MKL_CORE mkl_core
                 HINTS ${MKL_BASE}/lib/intel64 ${MKL_BASE}/lib REQUIRED)
    add_library(MKL::MKL INTERFACE IMPORTED)
    set_target_properties(MKL::MKL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES
            "${MKL_INTEL_LP64};${MKL_SEQUENTIAL};${MKL_CORE};pthread;m;dl")
    set(SCATT_HAS_MKL TRUE)
    set(SCATT_MKL_STATUS "found (MKLROOT=${MKL_BASE})")
    message(STATUS "MKL        : FOUND via MKLROOT=${MKL_BASE}")
else()
    set(SCATT_HAS_MKL FALSE)
    set(SCATT_MKL_STATUS "NOT found (fallback: Eigen native LU/GEMM -- SERIAL)")
    message(STATUS "MKL        : NOT FOUND  -- using Eigen native linear algebra (dense LU/GEMM will be SERIAL)")
endif()

# ---------- Optional SYCL / oneMKL GPU offload (Intel PVC, Battlemage) -------
# Turn on with:
#     cmake -DSCATT_WITH_SYCL=ON -DCMAKE_CXX_COMPILER=icpx ...
# Requires Intel oneAPI DPC++.  When enabled, the GpuPropagate.cpp translation
# unit is compiled with -fsycl and linked against oneMKL's SYCL DPC++ variant.
# When disabled (or when the SYCL headers are not found), the stub branch of
# GpuPropagate.cpp compiles with any host compiler and all gpu paths throw
# at runtime.
set(SCATT_HAS_SYCL FALSE)
set(SCATT_SYCL_STATUS "")   # user-facing status string, filled below.
if(SCATT_WITH_SYCL)
    # IntelSYCL provides the `sycl` CMake target (or macros like
    # add_sycl_to_target). Available via `module load intel` on LRZ.
    find_package(IntelSYCL CONFIG QUIET)
    if(NOT IntelSYCL_FOUND)
        # Older oneAPI exports the target through a different package name.
        find_package(SYCL CONFIG QUIET)
    endif()
    if(IntelSYCL_FOUND)
        set(SCATT_HAS_SYCL TRUE)
        set(SCATT_SYCL_STATUS "found (IntelSYCL CONFIG)")
        message(STATUS "SYCL       : FOUND via IntelSYCL CONFIG  -- GPU offload ENABLED")
    elseif(SYCL_FOUND)
        set(SCATT_HAS_SYCL TRUE)
        set(SCATT_SYCL_STATUS "found (SYCL CONFIG)")
        message(STATUS "SYCL       : FOUND via SYCL CONFIG  -- GPU offload ENABLED")
    else()
        message(FATAL_ERROR
            "SCATT_WITH_SYCL=ON but IntelSYCL/SYCL not found.\n"
            "  Hint: set CMAKE_CXX_COMPILER=icpx and ensure Intel oneAPI is loaded\n"
            "        (on LRZ:  module load intel).\n"
            "  The CXX compiler currently points at: ${CMAKE_CXX_COMPILER}")
    endif()
    # oneMKL's SYCL DPC++ target (mkl::mkl_sycl or mkl::mkl_intel_ilp64 +
    # mkl_sycl). When MKL was detected in CONFIG mode above, the target
    # MKL::MKL_DPCPP (or similar) is usually available.
else()
    set(SCATT_SYCL_STATUS "disabled (configure with -DSCATT_WITH_SYCL=ON to enable)")
    message(STATUS "SYCL       : NOT REQUESTED -- CPU-only build "
                   "(pass -DSCATT_WITH_SYCL=ON + CMAKE_CXX_COMPILER=icpx for GPU)")
endif()

# ---------- Optional CUDA / cuBLAS / cuSOLVER backend ----------
# NVIDIA GPU offload path -- sits alongside the SYCL backend.  Both can be
# enabled simultaneously; at runtime the GpuPropagate factory picks the
# backend matching the visible device.
#
# Currently SCAFFOLDED ONLY (the .cu translation unit defines stub Impl
# classes that throw "not implemented yet" at use; the public class API
# and the FRP/BP/BackPropagator call sites are unchanged).  Real
# implementation lands in a follow-up commit -- see future_plan.md item 3.
set(SCATT_HAS_CUDA FALSE)
set(SCATT_CUDA_STATUS "")
if(SCATT_WITH_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(SCATT_HAS_CUDA TRUE)
        set(SCATT_CUDA_STATUS "found (CUDAToolkit ${CUDAToolkit_VERSION})")
        message(STATUS "CUDA       : FOUND CUDAToolkit ${CUDAToolkit_VERSION}  "
                       "(target arch via CMAKE_CUDA_ARCHITECTURES)")
    else()
        message(FATAL_ERROR
            "SCATT_WITH_CUDA=ON but CUDAToolkit not found.\n"
            "  Hint: ensure nvcc is on PATH and CMAKE_CUDA_COMPILER points\n"
            "        at it, OR set CUDAToolkit_ROOT.")
    endif()
else()
    set(SCATT_CUDA_STATUS "disabled (configure with -DSCATT_WITH_CUDA=ON to enable)")
    message(STATUS "CUDA       : NOT REQUESTED  "
                   "(pass -DSCATT_WITH_CUDA=ON to enable NVIDIA GPU backend)")
endif()

# Pick the GPU backend translation unit added to scatt_potential.  The two
# backends (SYCL/oneMKL and CUDA/cuBLAS) provide the SAME public API
# from src/scatt/GpuPropagate.hpp -- compiling both at once would yield
# duplicate-symbol link errors, so CMake selects one based on the
# enable-flags above.  Build-time backend selection is the current
# design; runtime multi-vendor dispatch is future work (see
# future_plan.md item 3, "runtime multi-vendor dispatch").
#
# SYCL is the default backend (and also serves CPU-only builds via its
# own #ifdef-guarded host stubs).  CUDA wins when explicitly requested.
if(SCATT_HAS_CUDA)
    set(SCATT_GPU_BACKEND_SRC
            src/scatt/gpu/cuda/GpuPropagate.cu
            src/scatt/gpu/cuda/GpuDipoleEngine.cu)
    message(STATUS "GPU backend: CUDA  (src/scatt/gpu/cuda/{GpuPropagate.cu, GpuDipoleEngine.cu})")
else()
    set(SCATT_GPU_BACKEND_SRC
            src/scatt/gpu/sycl/GpuPropagate.cpp
            src/scatt/gpu/sycl/GpuDipoleEngine.cpp)
    if(SCATT_HAS_SYCL)
        message(STATUS "GPU backend: SYCL  (src/scatt/gpu/sycl/{GpuPropagate.cpp, GpuDipoleEngine.cpp})")
    else()
        message(STATUS "GPU backend: HOST-STUB  (CPU-only; "
                       "src/scatt/gpu/sycl/*.cpp #else branches)")
    endif()
endif()

# Probe oneMKL on the SYCL build for sytri (inverse from sytrf factors).
# Some Intel oneMKL versions ship sytrf (factor) but NOT sytri (inverse-from-
# factor) for the GPU LAPACK API.  Both are needed for the symmetric-LDLᵀ
# inversion path in GpuForwardStepper; if sytri is missing we fall back to
# the legacy getrf+getri path entirely (no Phase 2 speedup, but builds
# cleanly).  The probe is a small CXX try_compile.
set(SCATT_HAS_ONEMKL_SYTRI FALSE)
if(SCATT_HAS_SYCL)
    include(CheckCXXSourceCompiles)
    set(_old_flags "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "-fsycl -qmkl=parallel -DMKL_ILP64")
    check_cxx_source_compiles([[
        #if __has_include(<sycl/sycl.hpp>)
        #  include <sycl/sycl.hpp>
        #else
        #  include <CL/sycl.hpp>
        #endif
        #include <oneapi/mkl/lapack.hpp>
        int main() {
            sycl::queue q;
            (void) oneapi::mkl::lapack::sytri_scratchpad_size<double>(
                q, oneapi::mkl::uplo::lower,
                static_cast<std::int64_t>(10),
                static_cast<std::int64_t>(10));
            return 0;
        }
    ]] SCATT_HAS_ONEMKL_SYTRI)
    set(CMAKE_REQUIRED_FLAGS "${_old_flags}")
    if(SCATT_HAS_ONEMKL_SYTRI)
        message(STATUS "oneMKL sytri: FOUND -- GPU symmetric-LDLᵀ inversion ENABLED")
    else()
        message(STATUS "oneMKL sytri: NOT FOUND -- GPU FRP will use legacy getrf+getri")
    endif()
endif()

# ---------- OpenMP ----------
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND AND APPLE)
    set(OMP_ROOT "/opt/homebrew/opt/libomp")
    if(EXISTS "${OMP_ROOT}/lib/libomp.dylib")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${OMP_ROOT}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${OMP_ROOT}/lib/libomp.dylib")
        find_package(OpenMP)
    endif()
endif()
if(OpenMP_CXX_FOUND)
    set(SCATT_OPENMP_STATUS "found (flags: ${OpenMP_CXX_FLAGS})")
    message(STATUS "OpenMP     : FOUND  flags=${OpenMP_CXX_FLAGS}")
else()
    set(SCATT_OPENMP_STATUS "NOT found (kernels will run serially)")
    message(WARNING "OpenMP     : NOT FOUND -- kernels will run SERIALLY.")
endif()

# -----------------------------------------------------------------------------
# Core library (header-only interface target).
# -----------------------------------------------------------------------------
add_library(scatt_core INTERFACE)
target_include_directories(scatt_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(scatt_core INTERFACE
    Eigen3::Eigen
    hdf5::hdf5
    GSL::gsl
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(scatt_core INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(scatt_core INTERFACE SCATT_HAS_OPENMP=1)
endif()
if(SCATT_HAS_MKL)
    target_link_libraries(scatt_core INTERFACE MKL::MKL)
    target_compile_definitions(scatt_core INTERFACE
        EIGEN_USE_MKL_ALL
        SCATT_HAS_MKL=1
        MKL_DIRECT_CALL_SEQ_JIT=1)
    # Belt-and-braces: make sure no caller (env var, MKLConfig.cmake quirk)
    # ever tells Eigen or mkl.h to expect 64-bit ints.  If MKL_INTERFACE was
    # set to ilp64 we already errored at find_package level; this line kills
    # any remaining ILP64 defines leaking in via MKL::MKL's INTERFACE_COMPILE_DEFINITIONS.
    target_compile_options(scatt_core INTERFACE -UMKL_ILP64)
endif()

# -----------------------------------------------------------------------------
# Potentials module (keep real .cpp here so compile time scales with code
# size, not with every translation unit including a fat header).
# -----------------------------------------------------------------------------
add_library(scatt_potential STATIC
    src/scatt/Parameters.cpp
    src/scatt/Potentials.cpp
    src/scatt/PotentialStorage.cpp
    src/scatt/WavefunctionSetup.cpp
    src/scatt/ExchangeCoupling.cpp
    src/scatt/SchurInverter.cpp
    src/scatt/WInverseOperator.cpp
    src/scatt/ForwardRPropagator.cpp
    src/scatt/KMatrixExtractor.cpp
    src/scatt/BackPropagator.cpp
    src/scatt/AsymptoticAmplitudes.cpp
    src/scatt/DipoleMatrixElement.cpp
    src/scatt/DipoleIO.cpp
    src/scatt/StoragePlanner.cpp
    src/scatt/Bench.cpp
    ${SCATT_GPU_BACKEND_SRC}
)
target_link_libraries(scatt_potential PUBLIC scatt_core)

# SYCL activation -- compile GpuPropagate.cpp with -fsycl, and let the
# icpx driver pull in the matching libsycl + oneMKL via -qmkl=parallel.
#
# IMPORTANT: do NOT hand-list "-lsycl -lmkl_sycl -lmkl_intel_lp64
# -lmkl_intel_thread -lmkl_core -lOpenCL" even though the oneMKL
# documentation lists them.  On LRZ SuperMUC-NG Phase 2 that picks up a
# SYSTEM libsycl while libmkl_sycl_blas.so was built against the oneAPI
# libsycl, giving the ABI-mismatch error
#     undefined reference to
#       `sycl::_V1::queue::submit_with_event_impl(...)'
# The Intel driver with "-fsycl -qmkl=parallel" knows the right libsycl
# and MKL LP64+OpenMP+DPC++ combination to link.  This is exactly the
# recipe version_0/CMakeLists.txt uses and that production C8F8 runs
# compiled with.
if(SCATT_HAS_SYCL)
    set_source_files_properties(src/scatt/gpu/sycl/GpuPropagate.cpp PROPERTIES
        COMPILE_OPTIONS "-fsycl")
    target_compile_definitions(scatt_potential PUBLIC SCATT_HAS_SYCL=1)
    if(SCATT_HAS_ONEMKL_SYTRI)
        target_compile_definitions(scatt_potential PRIVATE SCATT_HAS_ONEMKL_SYTRI=1)
    endif()
    # version_0/CMakeLists.txt's exact compile + link recipe.  This is the
    # combo that's known to work on LRZ SuperMUC-NG Phase 2 with the
    # intel-toolkit/2025.2.0 module set; deviating from it (-qmkl=sequential,
    # missing -ltbb / -lonemkl, etc.) reproducibly segfaults inside MKL
    # LAPACK or fails to link.
    #
    # Compile flags applied to ALL scatt-side TUs (we propagate via
    # scatt_core's INTERFACE compile options):
    #   -include unistd.h    -- pulls in POSIX decls globally
    #   -O3 -march=native    -- aggressive host-SIMD optimization
    #   -ffast-math          -- yes, even though it breaks bit-repro of the
    #                           gold-standard FRP test.  Matches version_0.
    #   -DNDEBUG             -- strip asserts
    #   -fsycl               -- enable SYCL kernels (already set per-file
    #                           on GpuPropagate.cpp; PUBLIC here is harmless)
    #   -qopenmp             -- libiomp5 OpenMP runtime (matches MKL's
    #                           intel_thread and SYCL's host runtime)
    #   -qmkl=parallel       -- LP64 + intel_thread MKL through icpx driver
    #   -DMKL_ILP64          -- 64-bit MKL_INT.  Required because version_0's
    #                           direct oneapi::mkl::lapack calls expect int64;
    #                           our scatt::inverse_general() uses MKL_INT and
    #                           adapts.  Eigen's LAPACKE bindings hard-code
    #                           int* for IPIV (Eigen 3.4) and would fail to
    #                           compile with ILP64, so we drop EIGEN_USE_MKL_ALL
    #                           below.  Eigen falls back to native LU/QR/...,
    #                           which is fine -- the dense-inverse hot path
    #                           still goes through MKL via inverse_general,
    #                           and the heaviest decompositions (FRP/BP) run
    #                           on the GPU anyway.
    set(_sycl_common_flags
        -include unistd.h
        -O3 -march=native -ffast-math -DNDEBUG
        -fsycl -qopenmp -qmkl=parallel -DMKL_ILP64)
    target_compile_options(scatt_core INTERFACE ${_sycl_common_flags})
    # Link: same compile flags + the oneMKL DPC++ libs and TBB.  These are
    # the libraries the icpx driver doesn't auto-link with -qmkl alone.
    target_link_options(scatt_potential PUBLIC
    -fsycl
    -qopenmp
    -qmkl=parallel
    -ltbb)

    # Level Zero Sysman loader -- needed because Bench.cpp calls
    # zesInit / zesDriverGet / zesPowerGetEnergyCounter directly for GPU
    # energy reporting.  Resolved via find_library() so we pass the FULL
    # path to target_link_libraries() -- this makes CMake place the
    # library AFTER libscatt_potential.a on the link line and avoids the
    # "static lib pulled in but loader -l flag came too early" failure
    # that happens with target_link_options(-lze_loader).
    find_library(ZE_LOADER_LIB
        NAMES ze_loader libze_loader.so.1
        HINTS
            $ENV{CMPLR_ROOT}/lib
            $ENV{CMPLR_ROOT}/linux/lib
            $ENV{ONEAPI_ROOT}/compiler/latest/lib
            $ENV{ONEAPI_ROOT}/compiler/latest/linux/lib
            $ENV{ONEAPI_ROOT}/lib
            /usr/lib64
            /usr/lib
    )
    if(ZE_LOADER_LIB)
        message(STATUS "Level Zero loader found: ${ZE_LOADER_LIB}")
        target_link_libraries(scatt_potential PUBLIC ${ZE_LOADER_LIB})
    else()
        message(STATUS "Level Zero loader NOT found by find_library; "
                "falling back to -lze_loader (link may fail if not on path)")
        target_link_libraries(scatt_potential PUBLIC ze_loader)
    endif()

    if(NOT SCATT_HAS_MKL)
        set(SCATT_HAS_MKL TRUE)
        set(SCATT_MKL_STATUS "linked via -qmkl=parallel + -lonemkl* + -ltbb (icpx, ILP64)")
        message(STATUS "MKL        : version_0 recipe (-qmkl=parallel -DMKL_ILP64 + onemkl/tbb)")
        if(DEFINED ENV{MKLROOT})
            target_include_directories(scatt_core INTERFACE $ENV{MKLROOT}/include)
        endif()
        # SCATT_HAS_MKL routes scatt::inverse_general through LAPACKE_dgetrf.
        # MKL_DIRECT_CALL_SEQ_JIT NOT defined (mismatch with parallel MKL).
        # EIGEN_USE_MKL_ALL NOT defined (Eigen 3.4 LAPACKE bindings can't
        # take ILP64 IPIV).
        target_compile_definitions(scatt_core INTERFACE SCATT_HAS_MKL=1)
    endif()
endif()

# ---------- CUDA backend wiring ----------
# When SCATT_HAS_CUDA is on, the .cu translation unit was already added
# to scatt_potential's source list above (via SCATT_GPU_BACKEND_SRC).
# Link cuBLAS + cuSOLVER + cudart, mark the file as CUDA language, and
# advertise SCATT_HAS_CUDA=1 to call-site code.  The real Impl bodies
# live in src/scatt/gpu/cuda/GpuPropagate.cu -- currently a scaffold
# stub; full implementation arrives in a follow-up (see future_plan.md
# item 3, "Phase 4: CUDA Impl bodies").
if(SCATT_HAS_CUDA)
    set_source_files_properties(src/scatt/gpu/cuda/GpuPropagate.cu
        PROPERTIES LANGUAGE CUDA)
    target_link_libraries(scatt_potential PUBLIC
        CUDA::cudart CUDA::cublas CUDA::cusolver)
    # NVML for GPU energy reporting in Bench.cpp.  Same role as Level Zero
    # Sysman in the SYCL build.  CUDAToolkit ships libnvidia-ml.so in
    # ${CUDAToolkit_LIBRARY_DIR}; nvml.h in ${CUDAToolkit_INCLUDE_DIRS}.
    # CMake exposes both via the CUDA::nvml imported target on
    # CMake >= 3.17 with CUDAToolkit found.
    if(TARGET CUDA::nvml)
        target_link_libraries(scatt_potential PUBLIC CUDA::nvml)
        message(STATUS "CUDA       : NVML wired via CUDA::nvml (GPU energy will report in bench)")
    else()
        # Fallback for environments where CMake didn't synthesize CUDA::nvml.
        # libnvidia-ml.so is always next to libcudart.so on the driver path.
        target_link_libraries(scatt_potential PUBLIC nvidia-ml)
        message(STATUS "CUDA       : NVML wired via direct -lnvidia-ml fallback")
    endif()
    target_compile_definitions(scatt_potential PUBLIC SCATT_HAS_CUDA=1)
endif()

# -----------------------------------------------------------------------------
# Main executable.
# -----------------------------------------------------------------------------
add_executable(scattering src/main.cpp)
target_link_libraries(scattering PRIVATE scatt_potential)

# -----------------------------------------------------------------------------
# Postprocessing: two-photon delays (Baykusheva & Wörner 2017).
# -----------------------------------------------------------------------------
# Continuum-continuum dipole calculator.  Reads the saved psi from two
# energies and computes the back-prop-basis c-c dipole matrix element
# in length gauge.  Sits under postprocessing/two_photon_delay/ but is
# wired in here so it can link against scatt_potential.
add_library(cc_dipole STATIC
    postprocessing/two_photon_delay/cc_dipole/src/CCDipole.cpp
)
target_include_directories(cc_dipole PUBLIC
    postprocessing/two_photon_delay/cc_dipole/include)
target_link_libraries(cc_dipole PUBLIC scatt_potential)

# CLI driver: load saved ψ from disk, iterate over (κ, ν) pairs.
add_executable(cc_dipole_driver
    postprocessing/two_photon_delay/cc_dipole/src/cc_dipole_driver.cpp)
target_link_libraries(cc_dipole_driver PRIVATE cc_dipole hdf5::hdf5)

# MPI distributed driver: opt-in, gated on MPI being available.
# Same math as cc_dipole_driver --reuse-kappa-chunks (validated by the
# cc_dipole_bruteforce / symmetry / two-energy tests); distributes κ
# across MPI ranks.  Build with:
#     cmake -S . -B build -DSCATT_WITH_MPI=ON
find_package(MPI COMPONENTS CXX QUIET)
option(SCATT_WITH_MPI "Build MPI-distributed cc_dipole driver" ${MPI_FOUND})
if(SCATT_WITH_MPI)
    if(NOT MPI_FOUND)
        message(FATAL_ERROR
            "SCATT_WITH_MPI=ON but MPI not found by find_package(MPI).")
    endif()
    add_executable(cc_dipole_mpi
        postprocessing/two_photon_delay/cc_dipole/src/cc_dipole_mpi.cpp)
    target_link_libraries(cc_dipole_mpi PRIVATE
        cc_dipole hdf5::hdf5 MPI::MPI_CXX)
    message(STATUS "cc_dipole_mpi: enabled (MPI ${MPI_CXX_VERSION})")
else()
    message(STATUS "cc_dipole_mpi: SKIPPED (set -DSCATT_WITH_MPI=ON to enable)")
endif()

# -----------------------------------------------------------------------------
# Tests.
# -----------------------------------------------------------------------------
enable_testing()

add_executable(test_gaunt src/tests/test_gaunt.cpp)
target_link_libraries(test_gaunt PRIVATE scatt_core)
add_test(NAME gaunt_known_values COMMAND test_gaunt)

add_executable(test_potential_storage src/tests/test_potential_storage.cpp)
target_link_libraries(test_potential_storage PRIVATE scatt_potential)
add_test(NAME potential_storage COMMAND test_potential_storage)

add_executable(test_storage_symmetric src/tests/test_storage_symmetric.cpp)
target_link_libraries(test_storage_symmetric PRIVATE scatt_potential)
add_test(NAME storage_symmetric COMMAND test_storage_symmetric)

add_executable(test_storage_symmetric_end_to_end src/tests/test_storage_symmetric_end_to_end.cpp)
target_link_libraries(test_storage_symmetric_end_to_end PRIVATE scatt_potential)
add_test(NAME storage_symmetric_end_to_end COMMAND test_storage_symmetric_end_to_end
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_storage_parallel_write src/tests/test_storage_parallel_write.cpp)
target_link_libraries(test_storage_parallel_write PRIVATE scatt_potential)
add_test(NAME storage_parallel_write COMMAND test_storage_parallel_write)

# Direct byte-for-byte gate for the async chunk prefetch.  Writes a
# small DISK checkpoint, reads it back twice (with and without
# prefetch active), and asserts memcmp == 0 on every element.
add_executable(test_storage_prefetch_bit_equivalence
    src/tests/test_storage_prefetch_bit_equivalence.cpp)
target_link_libraries(test_storage_prefetch_bit_equivalence PRIVATE scatt_potential)
add_test(NAME storage_prefetch_bit_equivalence
         COMMAND test_storage_prefetch_bit_equivalence)

add_executable(test_potential_h2o src/tests/test_potential_h2o.cpp)
target_link_libraries(test_potential_h2o PRIVATE scatt_potential)
add_test(NAME potential_h2o_assembly COMMAND test_potential_h2o
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_potential_access_cost src/tests/test_potential_access_cost.cpp)
target_link_libraries(test_potential_access_cost PRIVATE scatt_potential)
add_test(NAME potential_access_cost COMMAND test_potential_access_cost
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_wavefunction_setup src/tests/test_wavefunction_setup.cpp)
target_link_libraries(test_wavefunction_setup PRIVATE scatt_potential)
add_test(NAME wavefunction_setup COMMAND test_wavefunction_setup
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_exchange_coupling src/tests/test_exchange_coupling.cpp)
target_link_libraries(test_exchange_coupling PRIVATE scatt_potential)
add_test(NAME exchange_coupling COMMAND test_exchange_coupling
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_schur_inverter src/tests/test_schur_inverter.cpp)
target_link_libraries(test_schur_inverter PRIVATE scatt_potential)
add_test(NAME schur_inverter COMMAND test_schur_inverter
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_w_inverse_operator src/tests/test_w_inverse_operator.cpp)
target_link_libraries(test_w_inverse_operator PRIVATE scatt_potential)
add_test(NAME w_inverse_operator COMMAND test_w_inverse_operator
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_winv_blockdiag_fastpath src/tests/test_winv_blockdiag_fastpath.cpp)
target_link_libraries(test_winv_blockdiag_fastpath PRIVATE scatt_potential)
add_test(NAME winv_blockdiag_fastpath COMMAND test_winv_blockdiag_fastpath
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_sinv_serial_vs_parallel src/tests/test_sinv_serial_vs_parallel.cpp)
target_link_libraries(test_sinv_serial_vs_parallel PRIVATE scatt_potential)
add_test(NAME sinv_serial_vs_parallel COMMAND test_sinv_serial_vs_parallel
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_sinv_parallel_disk_path src/tests/test_sinv_parallel_disk_path.cpp)
target_link_libraries(test_sinv_parallel_disk_path PRIVATE scatt_potential)
add_test(NAME sinv_parallel_disk_path COMMAND test_sinv_parallel_disk_path
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_forward_r_propagator src/tests/test_forward_r_propagator.cpp)
target_link_libraries(test_forward_r_propagator PRIVATE scatt_potential)
add_test(NAME forward_r_propagator COMMAND test_forward_r_propagator
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_k_matrix_extractor src/tests/test_k_matrix_extractor.cpp)
target_link_libraries(test_k_matrix_extractor PRIVATE scatt_potential)
add_test(NAME k_matrix_extractor COMMAND test_k_matrix_extractor
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_back_propagator src/tests/test_back_propagator.cpp)
target_link_libraries(test_back_propagator PRIVATE scatt_potential)
add_test(NAME back_propagator COMMAND test_back_propagator
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_backprop_asym_buffer src/tests/test_backprop_asym_buffer.cpp)
target_link_libraries(test_backprop_asym_buffer PRIVATE scatt_potential)
add_test(NAME backprop_asym_buffer COMMAND test_backprop_asym_buffer
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_backprop_v_outer_cache src/tests/test_backprop_v_outer_cache.cpp)
target_link_libraries(test_backprop_v_outer_cache PRIVATE scatt_potential)
add_test(NAME backprop_v_outer_cache COMMAND test_backprop_v_outer_cache
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_sinv_symmetric_inverse src/tests/test_sinv_symmetric_inverse.cpp)
target_link_libraries(test_sinv_symmetric_inverse PRIVATE scatt_potential)
add_test(NAME sinv_symmetric_inverse COMMAND test_sinv_symmetric_inverse
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_frp_symmetric_inverse src/tests/test_frp_symmetric_inverse.cpp)
target_link_libraries(test_frp_symmetric_inverse PRIVATE scatt_potential)
add_test(NAME frp_symmetric_inverse COMMAND test_frp_symmetric_inverse
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_checkpoints src/tests/test_checkpoints.cpp)
target_link_libraries(test_checkpoints PRIVATE scatt_potential)
add_test(NAME checkpoints COMMAND test_checkpoints
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_asymptotic_amplitudes src/tests/test_asymptotic_amplitudes.cpp)
target_link_libraries(test_asymptotic_amplitudes PRIVATE scatt_potential)
add_test(NAME asymptotic_amplitudes COMMAND test_asymptotic_amplitudes
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_dipole_matrix_element src/tests/test_dipole_matrix_element.cpp)
target_link_libraries(test_dipole_matrix_element PRIVATE scatt_potential)
add_test(NAME dipole_matrix_element COMMAND test_dipole_matrix_element
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

# Bit-identity gate for the batched DipoleMatrixElement::compute_six API.
# Asserts compute_six(A, B, cfg) is byte-equal to six legacy compute() calls
# with the cached_b_overlap pattern -- the contract that lets main.cpp swap
# the legacy 6x loop for the batched call without any accuracy drift.
add_executable(test_dipole_compute_six src/tests/test_dipole_compute_six.cpp)
target_link_libraries(test_dipole_compute_six PRIVATE scatt_potential)
add_test(NAME dipole_compute_six COMMAND test_dipole_compute_six
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

# ---- Postprocessing: c-c dipole tests ----
add_executable(test_cc_dipole_symmetry
    postprocessing/two_photon_delay/cc_dipole/tests/test_cc_dipole_symmetry.cpp)
target_link_libraries(test_cc_dipole_symmetry PRIVATE cc_dipole scatt_potential)
add_test(NAME cc_dipole_symmetry COMMAND test_cc_dipole_symmetry
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_cc_dipole_two_energy
    postprocessing/two_photon_delay/cc_dipole/tests/test_cc_dipole_two_energy.cpp)
target_link_libraries(test_cc_dipole_two_energy PRIVATE cc_dipole scatt_potential)
add_test(NAME cc_dipole_two_energy COMMAND test_cc_dipole_two_energy
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

# Bit-equivalence gate: hand-summed reference (no BLAS) vs production
# cc_dipole::compute_cc_dipole.  Catches any change to the hot-loop math
# (e.g. switch to sparse-Ang × dense-ψ first product, GEMM tile changes,
# loop reorganizations) that would degrade accuracy beyond the existing
# 1e-12 abs / 1e-10 rel tolerance.
add_executable(test_cc_dipole_bruteforce
    postprocessing/two_photon_delay/cc_dipole/tests/test_cc_dipole_bruteforce.cpp)
target_link_libraries(test_cc_dipole_bruteforce PRIVATE cc_dipole scatt_potential)
add_test(NAME cc_dipole_bruteforce COMMAND test_cc_dipole_bruteforce
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

# Strict per-channel length-velocity gauge identity check on H2O.
# Catches sign-mapping bugs in DipoleMatrixElement::velocity_coef and
# in the (l_mu, l_nu) <-> (raising, lowering) branch logic.  Tighter than
# the cross-section-level check inside test_dipole_matrix_element (#5).
add_executable(test_dipole_gauge_per_channel
               src/tests/test_dipole_gauge_per_channel.cpp)
target_link_libraries(test_dipole_gauge_per_channel PRIVATE scatt_potential)
add_test(NAME dipole_gauge_per_channel COMMAND test_dipole_gauge_per_channel
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_dipole_io src/tests/test_dipole_io.cpp)
target_link_libraries(test_dipole_io PRIVATE scatt_potential)
add_test(NAME dipole_io COMMAND test_dipole_io)

add_executable(test_storage_planner src/tests/test_storage_planner.cpp)
target_link_libraries(test_storage_planner PRIVATE scatt_potential)
add_test(NAME storage_planner COMMAND test_storage_planner)

add_executable(test_bench src/tests/test_bench.cpp)
target_link_libraries(test_bench PRIVATE scatt_potential)
add_test(NAME bench COMMAND test_bench)

add_executable(test_lapack_inverse src/tests/test_lapack_inverse.cpp)
target_link_libraries(test_lapack_inverse PRIVATE scatt_potential)
add_test(NAME lapack_inverse COMMAND test_lapack_inverse)

# Verifies HDF5Reader auto-detects split-vs-combined layout and produces
# identical PreprocData regardless of which path the user hands it.
add_executable(test_hdf5_split_roundtrip src/tests/test_hdf5_split_roundtrip.cpp)
target_link_libraries(test_hdf5_split_roundtrip PRIVATE scatt_potential)
add_test(NAME hdf5_split_roundtrip COMMAND test_hdf5_split_roundtrip
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

# Verifies --orb-lmax / --orb-rmax(-auto) write the same orbital
# coefficients inside the stored extent as a full preprocessing run --
# bit-identical, no recomputation, just a crop.  Needs the truncated
# fixture h2o_ccpvdz_sph_trunc.orbitals.h5 to exist next to the full
# one; the user (or this CMake's add_custom_command if PREPROC_PYTHON
# weren't disabled here) generates it once via:
#   preprocess_molden h2o_ccpvdz_sph.molden h2o_ccpvdz_sph_trunc \
#       --lmax 32 --dr 0.005 --rmax 15 --origin origin_of_file \
#       --orbitals occupied --exchange none --orb-lmax 16
add_executable(test_orbital_truncation src/tests/test_orbital_truncation.cpp)
target_link_libraries(test_orbital_truncation PRIVATE scatt_potential)
add_test(NAME orbital_truncation COMMAND test_orbital_truncation
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph_trunc.orbitals.h5)

# Regression: angular_dipole must implement the REAL-Y selection rule,
# not the complex-Y rule m_mu = m_nu + q.  Catches the σ_y → 1e-17 bug
# that affected C8F8 photoionization runs through 2026-04.
add_executable(test_real_gaunt_dipole src/tests/test_real_gaunt_dipole.cpp)
target_link_libraries(test_real_gaunt_dipole PRIVATE scatt_potential)
add_test(NAME real_gaunt_dipole COMMAND test_real_gaunt_dipole)

# Exhaustive numerical validation of `gaunt_real` for the dipole channel
# (l2 = 1, q = -1/0/+1) and every (l_mu, m_mu, l_nu, m_nu) up to L_MAX
# (default 50). Opt-in (NOT wired into ctest); emits ~M lines and
# allocates a ~400 MB Y_lm grid. Run manually:
#   ./test_real_gaunt_full_l50 [L_MAX]
add_executable(test_real_gaunt_full_l50 src/tests/test_real_gaunt_full_l50.cpp)
target_link_libraries(test_real_gaunt_full_l50 PRIVATE scatt_potential)

add_executable(test_gpu_propagate src/tests/test_gpu_propagate.cpp)
target_link_libraries(test_gpu_propagate PRIVATE scatt_potential)
add_test(NAME gpu_propagate COMMAND test_gpu_propagate
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_gpu_frp_symmetric_inverse src/tests/test_gpu_frp_symmetric_inverse.cpp)
target_link_libraries(test_gpu_frp_symmetric_inverse PRIVATE scatt_potential)
add_test(NAME gpu_frp_symmetric_inverse COMMAND test_gpu_frp_symmetric_inverse
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_gpu_sinv src/tests/test_gpu_sinv.cpp)
target_link_libraries(test_gpu_sinv PRIVATE scatt_potential)
add_test(NAME gpu_sinv COMMAND test_gpu_sinv
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_gpu_steppers src/tests/test_gpu_steppers.cpp)
target_link_libraries(test_gpu_steppers PRIVATE scatt_potential)
add_test(NAME gpu_steppers COMMAND test_gpu_steppers)

add_executable(test_gpu_dme src/tests/test_gpu_dme.cpp)
target_link_libraries(test_gpu_dme PRIVATE scatt_potential)
add_test(NAME gpu_dme COMMAND test_gpu_dme
         ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.orbitals.h5)

add_executable(test_storage_chunk_rechunk src/tests/test_storage_chunk_rechunk.cpp)
target_link_libraries(test_storage_chunk_rechunk PRIVATE scatt_potential)
add_test(NAME storage_chunk_rechunk COMMAND test_storage_chunk_rechunk)

add_executable(test_freeparticle_dipole src/tests/test_freeparticle_dipole.cpp)
target_link_libraries(test_freeparticle_dipole PRIVATE scatt_potential)
add_test(NAME freeparticle_dipole COMMAND test_freeparticle_dipole)

# Informational; not wired into ctest.
add_executable(test_potential_scaling src/tests/test_potential_scaling.cpp)
target_link_libraries(test_potential_scaling PRIVATE scatt_potential)

# -----------------------------------------------------------------------------
# Test environment hardening (mostly for LRZ login nodes).
#
# When we build with SCATT_WITH_SYCL=ON every test binary links libsycl.so
# transitively (because scatt_potential pulls it in).  libsycl eagerly
# enumerates backends at load time; on a login node without a usable
# Level-Zero / i915 driver that enumeration can segfault, even in tests
# that never call a single GPU routine.
#
# Setting ONEAPI_DEVICE_SELECTOR narrows the enumeration scope:
#   "*:cpu"        -- only ever consider CPU backend (always usable)
#   "opencl:gpu"   -- only consider OpenCL GPU devices (typical login node)
# "*:cpu" is the safe default for ctest.  Tests that actually need a GPU
# (gpu_propagate, gpu_steppers) override this to "*:gpu" so they see the
# real device when run via `salloc -p <gpu-partition> ctest`.
#
# KMP_DUPLICATE_LIB_OK=TRUE silences the libiomp/libgomp duplicate-runtime
# warning if it still triggers (shouldn't on this build because we use
# -qmkl=sequential, but keeps the door open).
set(_scatt_ctest_env
    "ONEAPI_DEVICE_SELECTOR=*:cpu"
    "KMP_DUPLICATE_LIB_OK=TRUE")

# Route every test_* executable to bin_test/.
set(_scatt_test_targets
    test_gaunt
    test_potential_storage
    test_potential_h2o
    test_potential_access_cost
    test_wavefunction_setup
    test_exchange_coupling
    test_schur_inverter
    test_w_inverse_operator
    test_forward_r_propagator
    test_k_matrix_extractor
    test_back_propagator
    test_checkpoints
    test_asymptotic_amplitudes
    test_dipole_matrix_element
    test_dipole_io
    test_storage_planner
    test_potential_scaling
    test_bench
    test_freeparticle_dipole
    test_backprop_asym_buffer
    test_lapack_inverse
    test_hdf5_split_roundtrip
    test_orbital_truncation
    test_real_gaunt_dipole
    test_gpu_propagate
    test_gpu_steppers
    test_gpu_frp_symmetric_inverse
    test_gpu_sinv
    test_gpu_dme
    test_storage_chunk_rechunk
)
foreach(_t ${_scatt_test_targets})
    set_target_properties(${_t} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${SCATT_TEST_BIN_DIR}")
endforeach()

# Apply the safe-startup env to every ctest entry.
get_property(_scatt_all_tests DIRECTORY PROPERTY TESTS)
foreach(_t ${_scatt_all_tests})
    set_tests_properties(${_t} PROPERTIES ENVIRONMENT "${_scatt_ctest_env}")
endforeach()
# GPU tests: override so they prefer GPU when one is visible.  On a login
# node that still has no GPU they'll just report [skip].
set_tests_properties(gpu_propagate gpu_steppers gpu_frp_symmetric_inverse gpu_sinv gpu_dme PROPERTIES
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=*:gpu;KMP_DUPLICATE_LIB_OK=TRUE")

add_executable(dump_potential_slices src/tools/dump_potential_slices.cpp)
target_link_libraries(dump_potential_slices PRIVATE scatt_potential)

add_executable(dump_f_psi src/tools/dump_f_psi.cpp)
target_link_libraries(dump_f_psi PRIVATE scatt_potential)

add_executable(pde_residual_check src/tools/pde_residual_check.cpp)
target_link_libraries(pde_residual_check PRIVATE scatt_potential)

add_executable(bench_storage src/tools/bench_storage.cpp)
target_link_libraries(bench_storage PRIVATE scatt_potential)

# -----------------------------------------------------------------------------
# Dependency summary -- printed at the end of every configure run so it's
# the LAST thing you see before the build starts. Makes "did it pick up MKL?
# did it see the GPU?" trivially answerable.
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=============================================================")
message(STATUS " Dependency summary for static_exchangeHF_scattering")
message(STATUS "=============================================================")
message(STATUS "  Eigen3   : ${SCATT_EIGEN_STATUS}")
message(STATUS "  HDF5     : ${SCATT_HDF5_STATUS}")
message(STATUS "  GSL      : ${SCATT_GSL_STATUS}")
message(STATUS "  OpenMP   : ${SCATT_OPENMP_STATUS}")
message(STATUS "  MKL      : ${SCATT_MKL_STATUS}")
message(STATUS "  SYCL     : ${SCATT_SYCL_STATUS}")
message(STATUS "-------------------------------------------------------------")
message(STATUS "  CXX      : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build    : ${CMAKE_BUILD_TYPE}")
if(SCATT_HAS_SYCL)
    message(STATUS "  >> GPU offload path is compiled in. Run with --use-gpu to activate.")
else()
    message(STATUS "  >> CPU-only build. --use-gpu will throw at runtime.")
endif()
if(NOT SCATT_HAS_MKL)
    message(STATUS "  >> No MKL: dense LU/GEMM in the Numerov hot loop will run SERIAL.")
endif()
message(STATUS "=============================================================")
message(STATUS "")
