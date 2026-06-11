cmake_minimum_required(VERSION 3.20)
project(multichannel_full_tdse LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# IEEE associativity (no -ffast-math) -- the Taylor stepper's
# convergence relies on exact accumulation order.
set(WARN_FLAGS "-Wall -Wextra -Wno-unused-parameter")
set(OPT_FLAGS  "-O3 -fno-math-errno")
if(NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    set(OPT_FLAGS "${OPT_FLAGS} -march=native")
endif()
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS}         ${WARN_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")

# --- OpenMP (optional; reference code's Common.hpp #include's omp.h
#     unconditionally, so we provide a stub on systems without OMP).
# On macOS with Apple Clang, system OpenMP detection fails; check for
# Homebrew libomp explicitly.
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND AND APPLE)
    find_path(LIBOMP_INC NAMES omp.h PATHS /opt/homebrew/opt/libomp/include
                                            /usr/local/opt/libomp/include
                                            NO_DEFAULT_PATH)
    find_library(LIBOMP_LIB NAMES omp PATHS /opt/homebrew/opt/libomp/lib
                                             /usr/local/opt/libomp/lib
                                             NO_DEFAULT_PATH)
    if(LIBOMP_INC AND LIBOMP_LIB)
        message(STATUS "Found Homebrew libomp at ${LIBOMP_LIB}")
        add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
        target_include_directories(OpenMP::OpenMP_CXX INTERFACE ${LIBOMP_INC})
        target_compile_options    (OpenMP::OpenMP_CXX INTERFACE -Xpreprocessor -fopenmp)
        target_link_libraries     (OpenMP::OpenMP_CXX INTERFACE ${LIBOMP_LIB})
        set(OpenMP_CXX_FOUND TRUE)
    endif()
endif()

# --- Eigen ---------------------------------------------------------
find_package(Eigen3 3.4 QUIET CONFIG)
if(NOT TARGET Eigen3::Eigen)
    find_path(EIGEN_INC Eigen/Dense
        PATHS /opt/homebrew/include/eigen3 /usr/local/include/eigen3
              /usr/include/eigen3 /opt/intel/oneapi/eigen3
        NO_DEFAULT_PATH)
    if(EIGEN_INC)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN_INC})
        message(STATUS "Eigen3 found at ${EIGEN_INC}")
    else()
        message(FATAL_ERROR "Eigen3 not found.")
    endif()
endif()

# --- reference subset (Stage 4) -----------------------------------
# We pull in JUST the reference SpinAlgebra (atomic-physics layer)
# for Stage 4a -- it's small, well-tested, and self-contained.  More
# reference modules join in 4b (radial Numerov) and 4c (analytic
# square-well validator).
#
# The reference Common.hpp does `#include <omp.h>` unconditionally;
# on systems without OpenMP we put `omp_stub/` on the include path
# (a header-only no-op shim).  When OpenMP IS available, the real
# omp.h wins because the system include path is searched first.
set(REF_DIR "${CMAKE_SOURCE_DIR}/../reference_code")
# Stage 4a needed only SpinAlgebra.  Stage 4b adds the radial/Numerov
# stack: Equations + Eigenvalues + Wavefunctions + Potentials, plus the
# AnalyticSquareWell cross-validator.  Renamed `mc_ref_spin` -> `mc_ref`.
add_library(mc_ref STATIC
    "${REF_DIR}/SpinAlgebra.cpp"
    "${REF_DIR}/Equations.cpp"
    "${REF_DIR}/Eigenvalues.cpp"
    "${REF_DIR}/Wavefunctions.cpp"
    "${REF_DIR}/Potentials.cpp"
    "${REF_DIR}/AnalyticSquareWell.cpp"
)
target_include_directories(mc_ref PUBLIC "${REF_DIR}")
target_link_libraries(mc_ref PUBLIC Eigen3::Eigen)
if(NOT OpenMP_CXX_FOUND)
    target_include_directories(mc_ref PUBLIC "${CMAKE_SOURCE_DIR}/omp_stub")
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(mc_ref PUBLIC OpenMP::OpenMP_CXX)
endif()
add_library(mc_ref_spin ALIAS mc_ref)   # backward-compat in this CMake

# --- core ----------------------------------------------------------
# Stage 1 had only header-only Taylor stepper. Stage 2 adds
# TDSEHamiltonian.cpp.  Stage 4a adds Rb85Spin.cpp (bridge to the
# reference SpinAlgebra with the σ⁺ vertex needed by the recipe).
add_library(mc_tdse_core STATIC
    src/TDSEHamiltonian.cpp
    src/MockMultiblock.cpp
    src/Rb85Spin.cpp
    src/Rb85Halo.cpp
    src/BlockEigenstates.cpp
    src/DipoleAssembler.cpp
    src/PooledBasis.cpp
    src/PooledTDSE.cpp
    src/SpectralReadout.cpp
    src/BasisCache.cpp
    src/PerturbationTheory.cpp
)
target_include_directories(mc_tdse_core PUBLIC src)
target_link_libraries(mc_tdse_core PUBLIC Eigen3::Eigen mc_ref)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mc_tdse_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- main executable -----------------------------------------------
# Production driver for Route A (full TDSE).  Builds pooled basis,
# runs interaction-picture Taylor matrix-vector propagation, writes
# dP/dE per block and a summary file.  CLI knobs documented in the
# .cpp header.
add_executable(run_tdse src/run_tdse.cpp)
target_link_libraries(run_tdse PRIVATE mc_tdse_core)

# --- tests ---------------------------------------------------------
enable_testing()
set(TEST_BIN_DIR "${CMAKE_BINARY_DIR}/bin_test")
file(MAKE_DIRECTORY "${TEST_BIN_DIR}")

file(GLOB TEST_SRC tests/test_*.cpp)
foreach(test_src ${TEST_SRC})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_BIN_DIR}")
    target_link_libraries(${test_name} PRIVATE mc_tdse_core)
    add_test(NAME ${test_name}
             COMMAND "${TEST_BIN_DIR}/${test_name}")
endforeach()

message(STATUS "------- multichannel_tdse / full_tdse ----------------")
message(STATUS "  build       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler    : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Eigen       : found")
message(STATUS "  test bin    : ${TEST_BIN_DIR}")
message(STATUS "------------------------------------------------------")
