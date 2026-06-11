// Common.hpp -- shared headers and types for the 3D spherical model code.
//
// 3D coupled-channel radial Schrödinger solver in real-Y^R basis.
// Conventions match the parent project (static_exchangeHF):
//   * Real spherical harmonics Y^R_{l,m} with q-map x->+1, y->-1, z->0.
//   * u/r convention: psi(r,Ω) = sum_{l,m} (chi_{lm}(r) / r) Y^R_{l,m}(Ω).
//   * Channel index idx = l*l + l + m, m in [-l, +l].
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Dense>

using std::cout;
using std::endl;

using dcompx = std::complex<double>;
inline constexpr dcompx I_unit{0.0, 1.0};
