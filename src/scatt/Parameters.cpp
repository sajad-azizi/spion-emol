#include "scatt/Parameters.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

namespace scatt {

void Parameters::validate() const {
    auto bad = [](const std::string& msg) -> void {
        throw std::runtime_error("Parameters::validate: " + msg);
    };
    if (dr <= 0.0)                   bad("dr must be > 0");
    if (N_grid < 3)                  bad("N_grid must be >= 3");
    if (l_max_continuum < 0)         bad("l_max_continuum must be >= 0");
    if (Lmax_sce < 0)                bad("Lmax_sce must be >= 0");
    // By selection rule, Lmax_sce >= l_exp_max() is needed or we truncate
    // the coupling the scattering solver expects.
    if (Lmax_sce < l_exp_max()) {
        std::ostringstream os;
        os << "Lmax_sce (" << Lmax_sce << ") < l_exp_max (" << l_exp_max()
           << " = 2 * l_max_continuum). HDF5 V_sigma will not contain enough"
              " angular components to couple all channels; increase preprocessing"
              " Lmax or decrease l_max_continuum.";
        bad(os.str());
    }
}

}  // namespace scatt
