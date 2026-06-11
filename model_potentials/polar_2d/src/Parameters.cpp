#include "Parameters.hpp"

Parameters::Parameters(int N_grid, int channels, double dr, int Nroots, int NTHREADS,
                       int N_phi, double dphi, double Emin, double Emax,
                       int divide, int p, int external_parameter)
    : N_grid(N_grid), channels(channels), dr(dr), Nroots(Nroots),
      NTHREADS(NTHREADS), external_parameter(external_parameter),
      Emin(Emin), Emax(Emax), N_phi(N_phi), dphi(dphi),
      divide(divide), p(p)
{}
