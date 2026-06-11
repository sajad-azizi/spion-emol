cmake_minimum_required(VERSION 3.22)
project(spion_2d LANGUAGES CXX)

# ──────────  Compiler & language  ─────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    message(FATAL_ERROR "Configure with icpx (Intel oneAPI).")
endif()

# ──────────  Global compile & link options  ───────────────────────────────────
add_compile_options(
    -D_GNU_SOURCE
    -include unistd.h
    -O3 -march=native -ffast-math -DNDEBUG
    -fsycl -qopenmp -qmkl=parallel -DMKL_ILP64
    -lonemkl -lonemkl_blas -lonemkl_spblas
    -ltbb
    -w
)
add_link_options(-fsycl -qopenmp -qmkl=parallel -lpthread -lm -ldl)

# ──────────  Third-party include / lib paths  ─────────────────────────────────
#   Eigen headers
include_directories(
    /dss/dsshome1/08/di35ker/Project_physics/general_lib
)

#   GSL
set(GSL_ROOT "/dss/lrzsys/sys/spack/release/24.4.1.sp6/opt/x86_64/gsl/2.8-oneapi-mfqog37")
include_directories("${GSL_ROOT}/include")
link_directories("${GSL_ROOT}/lib")

# ──────────  Main executable & sources  ───────────────────────────────────────
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/Parameters.cpp
    src/Potentials.cpp
    src/Equations.cpp
    src/Eigenvalues.cpp
    src/Wavefunctions.cpp
    src/DipoleMat.cpp
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        "${GSL_ROOT}/lib/libgsl.a"
        "${GSL_ROOT}/lib/libgslcblas.a"
)

# ──────────  Compile-time path macros  ────────────────────────────────────────
target_compile_definitions(${PROJECT_NAME}
    PUBLIC
        GENERAL_LIB_PATH=\"/dss/dsshome1/08/di35ker/Project_physics/general_lib\"
)

# ──────────  RPATH convenience  ───────────────────────────────────────────────
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "${GSL_ROOT}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# ──────────  Create output directory for wavefunction data  ───────────────────
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/basissets)

########cmake .. -DCMAKE_C_COMPILER=mpiicx -DCMAKE_CXX_COMPILER=mpiicpx -DCMAKE_BUILD_TYPE=Release
