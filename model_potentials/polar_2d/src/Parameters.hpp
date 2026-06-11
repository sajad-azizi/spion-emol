#pragma once

class Parameters {
    public:
        Parameters(int N_grid, int channels, double dr, int Nroots, int NTHREADS,
                int N_phi, double dphi, double Emin, double Emax,
                int divide, int p, int external_parameter);

        int N_grid;
        int channels;
        double dr;
        int Nroots;
        int NTHREADS;
        int external_parameter;
        double Emin;
        double Emax;
        int N_phi;
        double dphi;
        int divide;
        int p;
};
