cmake_minimum_required(VERSION 3.20)
project(spherical_3d LANGUAGES CXX)

# C++20 -- needs std::sph_bessel / std::sph_neumann.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# IEEE associativity: NO -ffast-math (matches the parent project's policy
# and lets the bit-reproducible Numerov tests pass).
set(WARN_FLAGS  "-Wall -Wextra -Wno-unused-parameter")
set(OPT_FLAGS   "-O3 -fno-math-errno")
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # icpx -- Intel oneAPI / LRZ build path.
    set(OPT_FLAGS "${OPT_FLAGS} -xHost")
elseif(NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    set(OPT_FLAGS "${OPT_FLAGS} -march=native")
endif()

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS}         ${WARN_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")

# --- Eigen ------------------------------------------------------------
find_package(Eigen3 3.4 QUIET CONFIG)
if(NOT TARGET Eigen3::Eigen)
    # Try a homebrew/macports-style path next to the parent project.
    find_path(EIGEN_INC Eigen/Dense
        PATHS /opt/homebrew/include/eigen3 /usr/local/include/eigen3
              /usr/include/eigen3 /opt/intel/oneapi/eigen3
        NO_DEFAULT_PATH)
    if(EIGEN_INC)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN_INC})
        message(STATUS "Eigen3 found at ${EIGEN_INC}")
    else()
        message(FATAL_ERROR "Eigen3 not found. Install eigen >= 3.4.")
    endif()
endif()

# --- GSL --------------------------------------------------------------
find_package(GSL REQUIRED)

# --- HDF5 -------------------------------------------------------------
# Used for energy-independent bound-state caching and per-energy dipole
# output (which is byte-compatible with the parent project's
# gather_dipoles.py / cross_section_delay.py post-processing).  Only the
# C interface is needed; all our HDF5 calls are libhdf5.{a,so} symbols.
find_package(HDF5 REQUIRED COMPONENTS C)

# --- OpenMP -----------------------------------------------------------
find_package(OpenMP)

# --- MKL (optional; LRZ build path) -----------------------------------
# Falls back silently if MKL is not present; we don't *need* it for this
# code (channel counts are small, Eigen native LU is fine).
set(SPH3D_HAVE_MKL OFF)
if(DEFINED ENV{MKLROOT})
    set(MKL_DIR_HINT "$ENV{MKLROOT}")
    if(EXISTS "${MKL_DIR_HINT}/lib/cmake/mkl/MKLConfig.cmake")
        set(MKL_LINK static)
        set(MKL_THREADING gnu_thread)
        set(MKL_INTERFACE lp64)
        find_package(MKL CONFIG QUIET)
        if(TARGET MKL::MKL)
            set(SPH3D_HAVE_MKL ON)
            message(STATUS "MKL    : found (LP64, ${MKL_LINK})")
        endif()
    endif()
endif()

# --- core library -----------------------------------------------------
add_library(sph3d_core STATIC
    src/Potentials.cpp
    src/Equations.cpp
    src/Eigenvalues.cpp
    src/Wavefunctions.cpp
    src/DipoleMat.cpp
    src/RunDirs.cpp
    src/BoundStateIO.cpp
    src/DipoleIO.cpp
)
target_include_directories(sph3d_core PUBLIC src)
target_link_libraries(sph3d_core PUBLIC Eigen3::Eigen GSL::gsl HDF5::HDF5)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sph3d_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(SPH3D_HAVE_MKL)
    target_link_libraries(sph3d_core PUBLIC MKL::MKL)
    target_compile_definitions(sph3d_core PUBLIC EIGEN_USE_MKL_ALL)
endif()

# --- main executable --------------------------------------------------
# Lives directly in ${CMAKE_BINARY_DIR}/spherical_3d -- this is the only
# binary intended to be invoked by users / cluster scripts.
add_executable(spherical_3d src/main.cpp)
set_target_properties(spherical_3d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(spherical_3d PRIVATE sph3d_core)

# --- tests ------------------------------------------------------------
# Test binaries are sequestered under ${CMAKE_BINARY_DIR}/bin_test so they
# don't clutter the top-level build dir.  ctest is enabled and registers
# every tests/test_*.cpp automatically; run with `ctest --test-dir build`.
enable_testing()
set(SPH3D_TEST_BIN_DIR "${CMAKE_BINARY_DIR}/bin_test")
file(MAKE_DIRECTORY "${SPH3D_TEST_BIN_DIR}")
file(GLOB SPH3D_TEST_SRC tests/test_*.cpp)
foreach(test_src ${SPH3D_TEST_SRC})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${SPH3D_TEST_BIN_DIR}")
    target_link_libraries(${test_name} PRIVATE sph3d_core)
    add_test(NAME ${test_name}
             COMMAND "${SPH3D_TEST_BIN_DIR}/${test_name}")
endforeach()

# --- summary ---------------------------------------------------------
message(STATUS "------- spherical_3d ---------------------------------")
message(STATUS "  build       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler    : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Eigen       : found")
message(STATUS "  GSL         : ${GSL_VERSION}")
message(STATUS "  OpenMP      : ${OpenMP_CXX_FOUND}")
message(STATUS "  MKL         : ${SPH3D_HAVE_MKL}")
message(STATUS "------------------------------------------------------")
