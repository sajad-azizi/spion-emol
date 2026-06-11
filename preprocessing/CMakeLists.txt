cmake_minimum_required(VERSION 3.22)
project(static_exchangeHF_preprocessing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# Release-time optimization (same rationale as scattering/CMakeLists.txt:
# -O3, native SIMD, no fast-math so projected densities stay reproducible).
set(_preproc_is_release $<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>)
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    add_compile_options($<${_preproc_is_release}:-O3>
                        $<${_preproc_is_release}:-xHost>
                        $<${_preproc_is_release}:-fno-math-errno>)
else()
    add_compile_options($<${_preproc_is_release}:-O3>
                        $<${_preproc_is_release}:-march=native>
                        $<${_preproc_is_release}:-fno-math-errno>
                        $<${_preproc_is_release}:-funroll-loops>)
endif()

# -----------------------------------------------------------------------------
# Portable dependency discovery.
#
# Order on every external: try `find_package` (CONFIG and MODULE modes) first
# so we respect CMAKE_PREFIX_PATH / environment-module hints on LRZ / any
# other cluster. Fall back to an explicit find_path / find_library with HINTS
# including ENV variables and common system locations.
#
# On LRZ, a typical invocation is something like
#     module load cmake gcc eigen hdf5
#     cmake -S . -B build
# -- the modules set CMAKE_PREFIX_PATH and the first find_package call
# succeeds. If a package isn't loadable via modules, users can point us at
# it with a path: `cmake -DEIGEN_ROOT=... -DHDF5_ROOT=...` or equivalent
# environment variables.
# -----------------------------------------------------------------------------

# ---------- Eigen ----------
# Prefer Eigen3's CMake config (Eigen3::Eigen target). Fall back to a path
# search. HINTS cover: user-provided -DEIGEN_ROOT=, $EIGEN_ROOT / $EIGEN_DIR /
# $EIGEN_INCLUDE_DIR env vars, Homebrew, standard Linux, and LRZ spack trees.
find_package(Eigen3 3.4 CONFIG QUIET)
if(NOT Eigen3_FOUND)
    find_path(EIGEN_INCLUDE_DIR
        NAMES Eigen/Dense Eigen/Core
        HINTS
            ${EIGEN_ROOT}
            $ENV{EIGEN_ROOT}
            $ENV{EIGEN_DIR}
            $ENV{EIGEN_INCLUDE_DIR}
            $ENV{EIGEN_BASE}                    # LRZ spack module sets this
            /opt/homebrew/include/eigen3
            /usr/include/eigen3
            /usr/local/include/eigen3
        PATH_SUFFIXES eigen3 include/eigen3 include
    )
    if(NOT EIGEN_INCLUDE_DIR)
        message(FATAL_ERROR
            "Eigen3 not found.\n"
            "Tried find_package(Eigen3) and explicit HINTS.\n"
            "On LRZ: module load eigen (or similar). Otherwise:\n"
            "  cmake -DEIGEN_ROOT=/path/to/eigen-install ...\n"
            "or set environment EIGEN_ROOT=/path/to/eigen-install.")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_INCLUDE_DIR}")
    set(PREPROC_EIGEN_STATUS "found (path HINTS, ${EIGEN_INCLUDE_DIR})")
    message(STATUS "Eigen3     : FOUND via path HINTS  ${EIGEN_INCLUDE_DIR}")
else()
    set(PREPROC_EIGEN_STATUS "found (CONFIG, ${Eigen3_DIR})")
    message(STATUS "Eigen3     : FOUND via find_package CONFIG  ${Eigen3_DIR}")
endif()

# ---------- HDF5 ----------
# find_package(HDF5) is the canonical path and understands HDF5_ROOT /
# HDF5_DIR env vars. Request only the C component (we don't use C++ bindings).
# Fall back to a path search if the find module is unavailable.
find_package(HDF5 COMPONENTS C QUIET)
if(NOT HDF5_FOUND)
    find_path(HDF5_INCLUDE_DIR
        NAMES hdf5.h
        HINTS
            ${HDF5_ROOT}
            $ENV{HDF5_ROOT}
            $ENV{HDF5_DIR}
            $ENV{HDF5_BASE}                     # LRZ spack module
            /opt/homebrew/include
            /usr/include
            /usr/local/include
        PATH_SUFFIXES include
    )
    find_library(HDF5_C_LIB
        NAMES hdf5 hdf5_serial
        HINTS
            ${HDF5_ROOT}
            $ENV{HDF5_ROOT}
            $ENV{HDF5_DIR}
            $ENV{HDF5_BASE}                     # LRZ spack module
            $ENV{HDF5_LIBDIR}                   # LRZ spack module sets this directly
            /opt/homebrew/lib
            /usr/lib
            /usr/local/lib
        PATH_SUFFIXES lib lib64
    )
    if(NOT HDF5_INCLUDE_DIR OR NOT HDF5_C_LIB)
        message(FATAL_ERROR
            "HDF5 not found.\n"
            "Tried find_package(HDF5) and explicit HINTS.\n"
            "On LRZ: module load hdf5 (or similar). Otherwise:\n"
            "  cmake -DHDF5_ROOT=/path/to/hdf5-install ...\n"
            "or set environment HDF5_ROOT=/path/to/hdf5-install.")
    endif()
    add_library(hdf5::hdf5 UNKNOWN IMPORTED)
    set_target_properties(hdf5::hdf5 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}"
        IMPORTED_LOCATION "${HDF5_C_LIB}")
    set(HDF5_LIBRARIES hdf5::hdf5)
    set(PREPROC_HDF5_STATUS "found (path HINTS, ${HDF5_C_LIB})")
    message(STATUS "HDF5       : FOUND via path HINTS  ${HDF5_C_LIB}")
else()
    # HDF5_LIBRARIES is set by find_package; prefer the hdf5::hdf5 imported
    # target from modern CMake if available.
    if(NOT TARGET hdf5::hdf5)
        add_library(hdf5::hdf5 INTERFACE IMPORTED)
        set_target_properties(hdf5::hdf5 PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES      "${HDF5_LIBRARIES}")
    endif()
    set(PREPROC_HDF5_STATUS "found (CONFIG, version ${HDF5_VERSION})")
    message(STATUS "HDF5       : FOUND via find_package CONFIG  version=${HDF5_VERSION}")
endif()

# ---------- FFTW3 (optional, strongly preferred) ----------
# Used by the SCE projector for the phi-direction DFT. Turns the
# O(N_phi) per-mode inner loop into O(log N_phi), which is the difference
# between minutes and hours at Lmax=300 for C8F8.
# If FFTW is not found, the SCE falls back to an O(N_phi^2) direct DFT.
#
# LRZ: `module load fftw` sets FFTW_BASE and prepends to CMAKE_PREFIX_PATH.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3_PC QUIET fftw3)
endif()
find_path(FFTW3_INCLUDE_DIR
    NAMES fftw3.h
    HINTS
        ${FFTW3_PC_INCLUDE_DIRS}
        ${FFTW_ROOT}
        $ENV{FFTW_ROOT}
        $ENV{FFTW_DIR}
        $ENV{FFTW_BASE}                 # LRZ spack module
        /opt/homebrew/include
        /usr/include
        /usr/local/include
    PATH_SUFFIXES include
)
find_library(FFTW3_LIBRARY
    NAMES fftw3
    HINTS
        ${FFTW3_PC_LIBRARY_DIRS}
        ${FFTW_ROOT}
        $ENV{FFTW_ROOT}
        $ENV{FFTW_DIR}
        $ENV{FFTW_BASE}
        $ENV{FFTW_LIBDIR}               # LRZ spack module
        /opt/homebrew/lib
        /usr/lib
        /usr/local/lib
    PATH_SUFFIXES lib lib64
)
if(FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY)
    add_library(FFTW3::fftw3 UNKNOWN IMPORTED)
    set_target_properties(FFTW3::fftw3 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}"
        IMPORTED_LOCATION             "${FFTW3_LIBRARY}")
    set(PREPROC_HAS_FFTW TRUE)
    set(PREPROC_FFTW_STATUS "found (${FFTW3_LIBRARY})")
    message(STATUS "FFTW3      : FOUND  ${FFTW3_LIBRARY}")
else()
    set(PREPROC_HAS_FFTW FALSE)
    set(PREPROC_FFTW_STATUS "NOT found (fallback: O(N_phi^2) direct DFT -- slow at Lmax > 50)")
    message(STATUS "FFTW3      : NOT FOUND  -- SCE will fall back to O(N_phi^2) direct DFT (slow at Lmax > 50)")
endif()

# ---------- OpenMP ----------
# Try the native find_package first. If it fails AND we're on macOS, try the
# AppleClang + Homebrew libomp workaround. On Linux this just works with the
# standard find_package.
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
    set(PREPROC_OPENMP_STATUS "found (flags: ${OpenMP_CXX_FLAGS})")
    message(STATUS "OpenMP     : FOUND  flags=${OpenMP_CXX_FLAGS}")
else()
    set(PREPROC_OPENMP_STATUS "NOT found (SCE + Hartree will run serially)")
    message(WARNING "OpenMP     : NOT FOUND -- SCE + Hartree will run SERIALLY.")
endif()

# ---------- Python for optional roundtrip tests ----------
#
# The Python roundtrip tests (test_hdf5_roundtrip.py) are convenience
# cross-checks. The C++ tests are the authoritative validation. If the
# Python environment is unusable (missing h5py, numpy<->h5py ABI mismatch,
# etc.) we silently skip the Python tests so ctest stays green.
#
# Override with: -DPREPROC_PYTHON=/path/to/python  or env PREPROC_PYTHON=...
# Skip explicitly with: -DPREPROC_SKIP_PYTHON_TESTS=ON
option(PREPROC_SKIP_PYTHON_TESTS "Skip Python-based HDF5 roundtrip tests" OFF)

set(PREPROC_PYTHON_READY FALSE)
if(NOT PREPROC_SKIP_PYTHON_TESTS)
    # Resolve PREPROC_PYTHON in order of precedence:
    #   1. CMake cache var (user passed -DPREPROC_PYTHON=...)
    #   2. $PREPROC_PYTHON env var
    #   3. find_program search with HINTS (LRZ/conda/Homebrew)
    if(NOT PREPROC_PYTHON AND DEFINED ENV{PREPROC_PYTHON} AND NOT "$ENV{PREPROC_PYTHON}" STREQUAL "")
        set(PREPROC_PYTHON "$ENV{PREPROC_PYTHON}" CACHE FILEPATH "Python interpreter for roundtrip tests")
    endif()
    if(NOT PREPROC_PYTHON)
        find_program(_preproc_python_found
            NAMES python3 python
            HINTS
                $ENV{PSI4_PYTHON}
                $ENV{CONDA_PREFIX}/bin
                /opt/anaconda3/envs/psi4/bin
                /opt/anaconda3/bin
        )
        if(_preproc_python_found)
            set(PREPROC_PYTHON "${_preproc_python_found}" CACHE FILEPATH "Python interpreter for roundtrip tests")
        endif()
    endif()
    if(PREPROC_PYTHON)
        # Probe whether this Python can actually run the roundtrip script.
        execute_process(
            COMMAND ${PREPROC_PYTHON} -c "import h5py, numpy, json"
            RESULT_VARIABLE _probe_rc
            OUTPUT_QUIET
            ERROR_VARIABLE  _probe_err
        )
        if(_probe_rc EQUAL 0)
            set(PREPROC_PYTHON_READY TRUE)
            message(STATUS "Python for roundtrip tests: ${PREPROC_PYTHON}")
        else()
            message(STATUS "Python found (${PREPROC_PYTHON}) but 'import h5py, numpy' failed.")
            message(STATUS "  Skipping Python-based roundtrip tests.")
            message(STATUS "  To enable, point PREPROC_PYTHON at a working interpreter:")
            message(STATUS "     cmake -DPREPROC_PYTHON=/path/to/python ...")
            message(STATUS "  Or force-skip: cmake -DPREPROC_SKIP_PYTHON_TESTS=ON ...")
            message(STATUS "  (probe error was: ${_probe_err})")
        endif()
    else()
        message(STATUS "Python not found; Python-based roundtrip tests will be skipped.")
    endif()
else()
    message(STATUS "PREPROC_SKIP_PYTHON_TESTS=ON: Python roundtrip tests disabled.")
endif()

# -----------------------------------------------------------------------------
# Core library (header-only for now)
# -----------------------------------------------------------------------------
add_library(preproc_core INTERFACE)
target_include_directories(preproc_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(preproc_core INTERFACE Eigen3::Eigen hdf5::hdf5)
if(OpenMP_CXX_FOUND)
    target_link_libraries(preproc_core INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(preproc_core INTERFACE PREPROC_HAS_OPENMP=1)
endif()
if(PREPROC_HAS_FFTW)
    target_link_libraries(preproc_core INTERFACE FFTW3::fftw3)
    target_compile_definitions(preproc_core INTERFACE PREPROC_HAS_FFTW=1)
endif()

# -----------------------------------------------------------------------------
# Output directory for generated .preproc.h5 files.
# If $WORK is set, write into it so scattering can pick them up out of the
# same persistent dir. Otherwise fall back to the build dir (back-compat).
# -----------------------------------------------------------------------------
if(NOT PREPROC_H5_DIR)
    if(DEFINED ENV{WORK} AND NOT "$ENV{WORK}" STREQUAL "")
        file(MAKE_DIRECTORY "$ENV{WORK}")
        set(PREPROC_H5_DIR "$ENV{WORK}")
    else()
        set(PREPROC_H5_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
endif()
message(STATUS "PREPROC_H5_DIR = ${PREPROC_H5_DIR}")

# All test executables go in build/bin_test/; the main one stays at build/.
set(PREPROC_TEST_BIN_DIR "${CMAKE_BINARY_DIR}/bin_test")
file(MAKE_DIRECTORY "${PREPROC_TEST_BIN_DIR}")

# -----------------------------------------------------------------------------
# Main executable
# -----------------------------------------------------------------------------
add_executable(preprocess_molden src/main_preprocess.cpp)
target_link_libraries(preprocess_molden PRIVATE preproc_core)
target_include_directories(preprocess_molden PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)  # tiny_json.hpp

# In-place repair tool: recompute /potential/V_H + V_total from the
# already-stored /rho/rho_lm in a preproc.h5.  Skips the 6-hour orbital
# SCE projection step.  Use after a preprocessing run that produced
# J = NaN due to the (now-fixed) high-Lmax overflow.
add_executable(repair_v_h src/repair_v_h.cpp)
target_link_libraries(repair_v_h PRIVATE preproc_core)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
enable_testing()

add_executable(test_molden_parse tests/test_molden_parse.cpp)
target_link_libraries(test_molden_parse PRIVATE preproc_core)
target_include_directories(test_molden_parse PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_test(NAME h2_ccpvdz_sph_parse
         COMMAND test_molden_parse
                 ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2_ccpvdz_sph.molden
                 ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2_ccpvdz_sph_reference.json)

add_executable(test_angular tests/test_angular.cpp)
target_link_libraries(test_angular PRIVATE preproc_core)
add_test(NAME angular_ylm_grid COMMAND test_angular)

add_executable(test_sce_h2 tests/test_sce_h2.cpp)
target_link_libraries(test_sce_h2 PRIVATE preproc_core)
add_test(NAME sce_h2 COMMAND test_sce_h2
         ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2_ccpvdz_sph.molden)

add_executable(test_sce_scaling tests/test_sce_scaling.cpp)
target_link_libraries(test_sce_scaling PRIVATE preproc_core)
# informational only; not wired into ctest

add_executable(test_v_en_h2 tests/test_v_en_h2.cpp)
target_link_libraries(test_v_en_h2 PRIVATE preproc_core)
target_include_directories(test_v_en_h2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_test(NAME v_en_h2 COMMAND test_v_en_h2
         ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2_ccpvdz_sph.molden
         ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2_ccpvdz_sph_reference.json)

add_executable(test_v_H_pointwise tests/test_v_H_pointwise.cpp)
target_link_libraries(test_v_H_pointwise PRIVATE preproc_core)
target_include_directories(test_v_H_pointwise PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_test(NAME v_H_pointwise COMMAND test_v_H_pointwise
         ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2_ccpvdz_sph.molden
         ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2_ccpvdz_sph_reference.json)

add_executable(test_v_H_h2 tests/test_v_H_h2.cpp)
target_link_libraries(test_v_H_h2 PRIVATE preproc_core)
target_include_directories(test_v_H_h2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_test(NAME v_H_h2 COMMAND test_v_H_h2
         ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2_ccpvdz_sph.molden
         ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2_ccpvdz_sph_reference.json)

# Regression: build_V_H must NOT produce NaN/Inf at high Lmax (the C8F8
# Lmax_sce=300, rmax=100 case).  Self-contained, no fixture file needed.
add_executable(test_v_H_high_lmax tests/test_v_H_high_lmax.cpp)
target_link_libraries(test_v_H_high_lmax PRIVATE preproc_core)
target_include_directories(test_v_H_high_lmax PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_test(NAME v_H_high_lmax COMMAND test_v_H_high_lmax)

add_executable(test_basis_pointwise tests/test_basis_pointwise.cpp)
target_link_libraries(test_basis_pointwise PRIVATE preproc_core)
target_include_directories(test_basis_pointwise PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_test(NAME h2_ccpvdz_sph_pointwise
         COMMAND test_basis_pointwise
                 ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2_ccpvdz_sph.molden
                 ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2_ccpvdz_sph_reference.json)
add_test(NAME h2o_ccpvdz_sph_pointwise
         COMMAND test_basis_pointwise
                 ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2o_ccpvdz_sph.molden
                 ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2o_ccpvdz_sph_reference.json)

# End-to-end pipeline tests: run preprocess_molden and verify the HDF5 with
# a Python script. Only wired in if a suitable Python interpreter was found
# AND it can import h5py + numpy without errors.
if(PREPROC_PYTHON_READY)
    add_test(NAME h2_pipeline_to_hdf5
             COMMAND ${CMAKE_COMMAND} -E env
                     $<TARGET_FILE:preprocess_molden>
                     ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2_ccpvdz_sph.molden
                     ${PREPROC_H5_DIR}/h2_ccpvdz_sph.preproc.h5
                     --lmax 24 --dr 0.005 --rmax 12 --exchange lda)
    add_test(NAME h2_pipeline_verify
             COMMAND ${PREPROC_PYTHON}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hdf5_roundtrip.py
                     ${PREPROC_H5_DIR}/h2_ccpvdz_sph.preproc.h5
                     ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2_ccpvdz_sph_reference.json)
    set_tests_properties(h2_pipeline_verify PROPERTIES DEPENDS h2_pipeline_to_hdf5)

    add_test(NAME h2o_pipeline_to_hdf5
             COMMAND ${CMAKE_COMMAND} -E env
                     $<TARGET_FILE:preprocess_molden>
                     ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2o_ccpvdz_sph.molden
                     ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.preproc.h5
                     --lmax 32 --dr 0.005 --rmax 15 --exchange lda --origin com)
    add_test(NAME h2o_pipeline_verify
             COMMAND ${PREPROC_PYTHON}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hdf5_roundtrip.py
                     ${PREPROC_H5_DIR}/h2o_ccpvdz_sph.preproc.h5
                     ${CMAKE_CURRENT_SOURCE_DIR}/reference_data/h2o_ccpvdz_sph_reference.json)
    set_tests_properties(h2o_pipeline_verify PROPERTIES DEPENDS h2o_pipeline_to_hdf5)
endif()

# -----------------------------------------------------------------------------
# Route every test_* executable to bin_test/.
# -----------------------------------------------------------------------------
set(_preproc_test_targets
    test_molden_parse
    test_angular
    test_sce_h2
    test_sce_scaling
    test_v_en_h2
    test_v_H_pointwise
    test_v_H_h2
    test_v_H_high_lmax
    test_basis_pointwise
)
foreach(_t ${_preproc_test_targets})
    set_target_properties(${_t} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PREPROC_TEST_BIN_DIR}")
endforeach()

# -----------------------------------------------------------------------------
# Dependency summary -- last thing printed by configure so you can tell at a
# glance whether the fast paths (FFTW, OpenMP) are active.
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=============================================================")
message(STATUS " Dependency summary for static_exchangeHF_preprocessing")
message(STATUS "=============================================================")
message(STATUS "  Eigen3   : ${PREPROC_EIGEN_STATUS}")
message(STATUS "  HDF5     : ${PREPROC_HDF5_STATUS}")
message(STATUS "  FFTW3    : ${PREPROC_FFTW_STATUS}")
message(STATUS "  OpenMP   : ${PREPROC_OPENMP_STATUS}")
message(STATUS "-------------------------------------------------------------")
message(STATUS "  CXX      : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build    : ${CMAKE_BUILD_TYPE}")
if(NOT PREPROC_HAS_FFTW)
    message(STATUS "  >> No FFTW: SCE will be very slow for Lmax > 50. "
                   "Load fftw module before configuring.")
endif()
message(STATUS "=============================================================")
message(STATUS "")
