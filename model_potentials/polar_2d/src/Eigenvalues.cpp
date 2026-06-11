#include "Eigenvalues.hpp"

Eigenvalues::Eigenvalues(Equations *equations, Parameters *parameters)
    : equations(equations), gsEnergy(0.0), i_match(0)
{
    Emin   = parameters->Emin;
    Emax   = parameters->Emax;
    N_grid = parameters->N_grid;
}


void Eigenvalues::groundstate_finder(){

    std::ifstream grin("../ground_energy.dat");
    if (grin.is_open()) {
        grin >> gsEnergy >> i_match;
        return;
    }

    double e_L = Emin;
    double e_H = Emax;
    const int desire_node = 1;

    std::ofstream eout("ground_energy.dat");
    eout << std::fixed << std::setprecision(15);

    double e_tr = e_L;
    while (true) {
        auto [node_counting, node_pos] = equations->OutwardNodeCounting(e_tr);

        if (node_counting < desire_node)
            e_L = e_tr;
        else
            e_H = e_tr;

        cout << e_tr << "\t" << e_L << "\t" << e_H << "\t"
             << node_pos << "\t" << node_counting << "\t" << desire_node << endl;

        if(std::abs(e_H - e_L) < 1e-9){
            i_match  = matching_point_finder(e_tr);
            gsEnergy = e_tr;
            cout << "Done! energy: " << e_tr << " i_match: " << i_match
                 << " node_pos: " << node_pos << " nodes: " << node_counting << endl;
            eout << e_tr << "\t" << i_match << endl;
            break;
        }
        e_tr = (e_L + e_H) / 2.0;
    }
    eout.close();
}


void Eigenvalues::Boundstates_finder(){

    double e_L = Emin;
    double e_H = Emax;
    int i_match_loc;

    std::ofstream eout("eigenvalues.dat");
    eout << std::fixed << std::setprecision(15);

    auto [node_max, node_pos_max] = equations->OutwardNodeCounting(Emax);

    int oo = 0;
    for (int desire_node = 1; desire_node < node_max; desire_node++) {
        e_H = Emax;
        double e_tr = e_L;

        while (true) {
            auto [node_counting, node_pos] = equations->OutwardNodeCounting(e_tr);

            if (node_counting < desire_node)
                e_L = e_tr;
            else
                e_H = e_tr;

            if (std::abs(e_H - e_L) < 1e-12) {
                i_match_loc = (e_tr < 0) ? matching_point_finder(e_tr) : N_grid - 1;

                cout << "Done! e: " << e_tr << "\t" << i_match_loc << "\t"
                     << node_pos << "\t" << node_counting << "\t" << desire_node << endl;
                eout << oo << "\t" << e_tr << "\t" << i_match_loc << endl;
                oo++;
                break;
            }
            e_tr = (e_L + e_H) / 2.0;
        }
        e_L = e_tr;
    }
    eout.close();
}


int Eigenvalues::matching_point_finder(double Energy){

    Eigen::MatrixXcd Rm, Rmp1;
    equations->propagateForward(Energy, N_grid - 1, Rm, true);
    equations->propagateBackward(Energy, 0, Rmp1, false);

    double per_deter = 0.0;
    int i_match_loc = 0;

    for (int i = 1; i < N_grid - 2; i++) {
        dcompx deter = (equations->Rinv_vector[i].inverse() - equations->Rinv_vector_back[i]).determinant();
        if (std::abs(deter) > per_deter && i > 5) {
            i_match_loc = i;
            cout << "i_match: " << i_match_loc << endl;
            break;
        }
        per_deter = std::abs(deter);
    }

    return i_match_loc;
}
