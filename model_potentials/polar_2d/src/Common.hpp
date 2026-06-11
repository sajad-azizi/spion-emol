#pragma once

#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <complex>
#include <cmath>
#include <omp.h>

#include <Eigen/Dense>

using std::cout;
using std::endl;

typedef std::complex<double> dcompx;

constexpr dcompx I{0.0, 1.0};
constexpr dcompx zero{0.0, 0.0};
