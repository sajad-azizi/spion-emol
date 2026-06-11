// Molden.hpp — pure-header parser + data model for Psi4-style Molden files.
//
// Goals of this module (Milestone 1):
//   - Read [Atoms], [GTO], [MO] sections.
//   - Detect Cartesian vs spherical AO flags: [5D], [5D7F], [5D10F], [7F], [9G].
//   - Produce a strict in-memory representation with indices matching the
//     shell ordering written to the molden file (which is the same ordering
//     used by the MO coefficient rows).
//   - Atomic units everywhere; if [Atoms] (Angs) is used, convert.
//   - Fail loudly on any malformed / ambiguous input.
//
// This header deliberately contains no angular / basis-set math — it just
// records what the file says. Evaluation and AO normalization live in
// basis/. We keep the parser independent so it can be unit-tested alone.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace preproc::molden {

constexpr double ANG2BOHR = 1.8897261245650618;

// ---------- low-level utilities ---------------------------------------------

// Convert Fortran-style exponents (1.0D-3, 1.0d-3) to C-style (1.0E-3).
// Only replace 'D'/'d' when it's *between two digits* -- otherwise a shell
// letter 'd' would be mangled.
inline std::string sanitize_fortran_exp(std::string s) {
    for (size_t i = 1; i + 1 < s.size(); ++i) {
        if ((s[i] == 'D' || s[i] == 'd')
            && std::isdigit(static_cast<unsigned char>(s[i - 1]))
            && (std::isdigit(static_cast<unsigned char>(s[i + 1]))
             || s[i + 1] == '+' || s[i + 1] == '-')) {
            s[i] = 'E';
        }
    }
    return s;
}

inline std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

inline std::string strip(const std::string& s) {
    size_t i = 0, j = s.size();
    while (i < j && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    while (j > i && std::isspace(static_cast<unsigned char>(s[j - 1]))) --j;
    return s.substr(i, j - i);
}

// Very small element symbol → Z table (extend as needed).
inline int symbol_to_Z(const std::string& sym) {
    static const std::unordered_map<std::string, int> T = {
        {"H", 1},  {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5},  {"C", 6},
        {"N", 7},  {"O", 8},  {"F", 9},  {"Ne", 10},{"Na", 11},{"Mg", 12},
        {"Al", 13},{"Si", 14},{"P", 15}, {"S", 16}, {"Cl", 17},{"Ar", 18},
        {"K", 19}, {"Ca", 20},
    };
    std::string s = sym;
    if (!s.empty()) s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
    for (size_t i = 1; i < s.size(); ++i) s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
    auto it = T.find(s);
    if (it == T.end()) throw std::runtime_error("symbol_to_Z: unknown element '" + sym + "'");
    return it->second;
}

// ---------- data model ------------------------------------------------------

struct Atom {
    std::string symbol;
    int         Z;          // nuclear charge (from molden column OR symbol)
    int         index_1b;   // 1-based index as it appears in the file
    Eigen::Vector3d xyz;    // in Bohr
};

// A "shell" = a contracted Gaussian with a fixed angular momentum l,
// shared among (2l+1) spherical or C(l+2,2) Cartesian real functions.
// We keep exponents + contraction coefficients *exactly as written* in the
// molden file. AO normalization constants are the responsibility of basis/.
struct Shell {
    int    atom_index;               // 0-based into Molecule::atoms
    int    l;                        // 0=s, 1=p, 2=d, ...
    bool   pure;                     // true: spherical (2l+1 comps); false: Cartesian
    std::vector<double> exponents;   // alpha_i
    std::vector<double> coeffs;      // contraction coefficients c_i  (as written)
    // Filled once the parse is complete; first AO index (0-based) this shell owns.
    int    ao_offset = -1;
    int    n_ao      = -1;           // 2l+1 if pure else (l+1)(l+2)/2
};

struct MO {
    std::string sym   = "";
    double      energy = 0.0;
    double      occ    = 0.0;
    char        spin   = 'A';        // 'A' = alpha, 'B' = beta
    std::vector<double> C;           // length = nbf (global AO basis)
};

struct Molecule {
    std::vector<Atom>  atoms;
    std::vector<Shell> shells;
    std::vector<MO>    mos_alpha;
    std::vector<MO>    mos_beta;     // empty for RHF
    int                nbf = 0;      // total AO basis size (shell.n_ao summed)
    // Global spherical/Cartesian flags detected from [5D]/[7F]/[9G] markers.
    // Defaults per Molden spec: Cartesian d (6), Cartesian f (10), Cartesian g (15).
    bool sph_d = false;
    bool sph_f = false;
    bool sph_g = false;
};

// ---------- parser ----------------------------------------------------------

class MoldenParser {
public:
    explicit MoldenParser(bool verbose = true) : verbose_(verbose) {}

    Molecule parse(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("MoldenParser: cannot open '" + path + "'");

        // Slurp file into a line buffer; the format is line-oriented and small.
        std::vector<std::string> lines;
        {
            std::string line;
            while (std::getline(f, line)) lines.push_back(line);
        }
        if (verbose_) std::cerr << "[molden] read " << lines.size() << " lines from " << path << "\n";

        Molecule mol;

        // Locate sections by their [Header] markers.
        auto is_section = [](const std::string& s, std::string& name_out) {
            std::string t = strip(s);
            if (t.size() < 3 || t.front() != '[') return false;
            auto close = t.find(']');
            if (close == std::string::npos) return false;
            name_out = t.substr(1, close - 1);
            return true;
        };

        // Pass 1: detect global spherical/Cartesian markers (they can appear anywhere).
        for (const auto& ln : lines) {
            std::string name;
            if (!is_section(ln, name)) continue;
            std::string u = lower(name);
            // "5d" implies 5d; "5d7f" implies both; "5d10f" implies 5d but Cartesian f; "7f" => spherical f; "9g" => spherical g.
            if (u.find("5d10f") != std::string::npos) { mol.sph_d = true;  mol.sph_f = false; }
            else if (u.find("5d7f") != std::string::npos) { mol.sph_d = true;  mol.sph_f = true; }
            else if (u == "5d")                            { mol.sph_d = true; }
            else if (u == "7f")                            { mol.sph_f = true; }
            else if (u == "9g")                            { mol.sph_g = true; }
        }
        if (verbose_) {
            std::cerr << "[molden] sph flags: d=" << mol.sph_d
                      << " f=" << mol.sph_f << " g=" << mol.sph_g << "\n";
        }

        // Pass 2: parse [Atoms]
        parse_atoms(lines, mol);
        if (verbose_) std::cerr << "[molden] parsed " << mol.atoms.size() << " atoms\n";

        // Pass 3: parse [GTO]
        parse_gto(lines, mol);
        if (verbose_) std::cerr << "[molden] parsed " << mol.shells.size() << " shells, nbf=" << mol.nbf << "\n";

        // Pass 4: parse [MO]
        parse_mo(lines, mol);
        if (verbose_) {
            std::cerr << "[molden] parsed " << mol.mos_alpha.size() << " alpha MOs, "
                      << mol.mos_beta.size() << " beta MOs\n";
        }

        validate(mol);
        return mol;
    }

private:
    bool verbose_;

    // Find the line index (in `lines`) of the section header with the given
    // case-insensitive name. Returns -1 if not found.
    static int find_section(const std::vector<std::string>& lines, const std::string& want) {
        std::string w = lower(want);
        for (int i = 0; i < (int)lines.size(); ++i) {
            std::string t = strip(lines[i]);
            if (t.empty() || t.front() != '[') continue;
            auto close = t.find(']');
            if (close == std::string::npos) continue;
            std::string name = lower(t.substr(1, close - 1));
            // the rest of the line may contain "(AU)" etc.
            if (name == w) return i;
        }
        return -1;
    }

    // True if `t` starts a new [Section]. Used to stop reading a block.
    static bool is_any_section_line(const std::string& s) {
        std::string t = strip(s);
        return !t.empty() && t.front() == '[';
    }

    void parse_atoms(const std::vector<std::string>& lines, Molecule& mol) {
        int idx = find_section(lines, "Atoms");
        if (idx < 0) throw std::runtime_error("molden: [Atoms] section missing");

        // Detect AU vs Angs from the header line.
        std::string header = lower(lines[idx]);
        bool au = header.find("(au)") != std::string::npos
               || header.find("au")  != std::string::npos;
        bool angs = header.find("angs") != std::string::npos;
        if (!au && !angs) {
            // default in molden spec is Angstrom
            angs = true;
        }
        double scale = au ? 1.0 : ANG2BOHR;
        if (verbose_) std::cerr << "[molden] [Atoms] units: " << (au ? "AU(Bohr)" : "Angstrom") << "\n";

        for (int i = idx + 1; i < (int)lines.size(); ++i) {
            const std::string& ln = lines[i];
            if (is_any_section_line(ln)) break;
            std::string t = strip(ln);
            if (t.empty()) continue;
            std::istringstream is(sanitize_fortran_exp(t));
            Atom a;
            int one_based, z_col;
            double x, y, z;
            if (!(is >> a.symbol >> one_based >> z_col >> x >> y >> z))
                throw std::runtime_error("molden: bad [Atoms] line: '" + t + "'");
            a.index_1b = one_based;
            // Prefer the explicit Z column; fallback to symbol lookup if zero.
            a.Z = (z_col > 0) ? z_col : symbol_to_Z(a.symbol);
            a.xyz << x * scale, y * scale, z * scale;
            mol.atoms.push_back(std::move(a));
        }

        if (mol.atoms.empty()) throw std::runtime_error("molden: [Atoms] parsed 0 atoms");
    }

    // Parse a shell type letter into l.  Supports combined "sp" shells too.
    static int letter_to_l(char c) {
        switch (std::tolower(static_cast<unsigned char>(c))) {
            case 's': return 0;
            case 'p': return 1;
            case 'd': return 2;
            case 'f': return 3;
            case 'g': return 4;
            case 'h': return 5;
            case 'i': return 6;
            default:  throw std::runtime_error(std::string("molden: unknown shell letter '") + c + "'");
        }
    }

    void parse_gto(const std::vector<std::string>& lines, Molecule& mol) {
        int idx = find_section(lines, "GTO");
        if (idx < 0) throw std::runtime_error("molden: [GTO] section missing");

        int i = idx + 1;
        const int N = (int)lines.size();

        // Loop: atom-block := "<atom_index> 0" then repeated shells, separated by blank lines.
        while (i < N) {
            // Skip blanks
            while (i < N && strip(lines[i]).empty()) ++i;
            if (i >= N) break;
            if (is_any_section_line(lines[i])) break;

            // Atom header line: "<atom_1based> 0"
            std::istringstream ah(strip(sanitize_fortran_exp(lines[i])));
            int atom_1b = -1, zero = -1;
            if (!(ah >> atom_1b >> zero))
                throw std::runtime_error("molden: bad [GTO] atom header: '" + lines[i] + "'");
            if (atom_1b < 1 || atom_1b > (int)mol.atoms.size())
                throw std::runtime_error("molden: [GTO] atom index out of range");
            int atom_index = atom_1b - 1;
            ++i;

            // Shells until a blank line or next section.
            while (i < N) {
                std::string t = strip(lines[i]);
                if (t.empty())                { ++i; break; }
                if (is_any_section_line(lines[i])) break;

                std::istringstream sh(sanitize_fortran_exp(t));
                std::string letter;
                int n_prim;
                double scale;
                if (!(sh >> letter >> n_prim >> scale))
                    throw std::runtime_error("molden: bad shell header: '" + t + "'");
                if (letter == "sp" || letter == "SP")
                    throw std::runtime_error("molden: combined 'sp' shell not yet supported");

                Shell s;
                s.atom_index = atom_index;
                s.l = letter_to_l(letter[0]);
                bool is_sph = (s.l == 0) || (s.l == 1)
                            || (s.l == 2 && mol.sph_d)
                            || (s.l == 3 && mol.sph_f)
                            || (s.l == 4 && mol.sph_g)
                            || (s.l >= 5);   // Molden spec: h/i always spherical
                s.pure = is_sph;
                s.exponents.reserve(n_prim);
                s.coeffs.reserve(n_prim);
                ++i;
                for (int p = 0; p < n_prim; ++p) {
                    if (i >= N) throw std::runtime_error("molden: EOF inside shell primitives");
                    std::istringstream pl(sanitize_fortran_exp(strip(lines[i])));
                    double alpha, c;
                    if (!(pl >> alpha >> c))
                        throw std::runtime_error("molden: bad primitive line: '" + lines[i] + "'");
                    s.exponents.push_back(alpha);
                    s.coeffs.push_back(c);
                    ++i;
                }
                s.n_ao = s.pure ? (2 * s.l + 1) : ((s.l + 1) * (s.l + 2) / 2);
                s.ao_offset = mol.nbf;
                mol.nbf += s.n_ao;
                (void)scale;   // molden "scale" column is historically ignored
                mol.shells.push_back(std::move(s));
            }
        }
    }

    void parse_mo(const std::vector<std::string>& lines, Molecule& mol) {
        int idx = find_section(lines, "MO");
        if (idx < 0) throw std::runtime_error("molden: [MO] section missing");

        auto starts_with = [](const std::string& s, const char* p) {
            size_t n = std::char_traits<char>::length(p);
            return s.size() >= n && lower(s.substr(0, n)) == lower(std::string(p));
        };

        int i = idx + 1;
        const int N = (int)lines.size();
        while (i < N) {
            std::string t = strip(lines[i]);
            if (t.empty())                { ++i; continue; }
            if (is_any_section_line(lines[i])) break;

            MO m;
            // Header lines: Sym=, Ene=, Spin=, Occup=  (order varies)
            while (i < N) {
                std::string u = strip(lines[i]);
                if (u.empty()) { ++i; continue; }
                if      (starts_with(u, "Sym="))   m.sym    = strip(u.substr(u.find('=') + 1));
                else if (starts_with(u, "Ene="))   m.energy = std::stod(sanitize_fortran_exp(strip(u.substr(u.find('=') + 1))));
                else if (starts_with(u, "Spin="))  {
                    std::string v = lower(strip(u.substr(u.find('=') + 1)));
                    m.spin = v.empty() ? 'A' : (v[0] == 'b' ? 'B' : 'A');
                }
                else if (starts_with(u, "Occup=")) m.occ    = std::stod(sanitize_fortran_exp(strip(u.substr(u.find('=') + 1))));
                else break;     // first coefficient line
                ++i;
            }

            // Coefficient lines: "<ao_index_1based> <coeff>"
            m.C.assign(mol.nbf, 0.0);
            int filled = 0;
            while (i < N) {
                std::string u = strip(lines[i]);
                if (u.empty()) { ++i; break; }
                if (is_any_section_line(lines[i])) break;
                // peek: is this the start of a new MO header?
                if (starts_with(u, "Sym=") || starts_with(u, "Ene=")
                 || starts_with(u, "Spin=") || starts_with(u, "Occup=")) break;
                std::istringstream cl(sanitize_fortran_exp(u));
                int ao_1b; double c;
                if (!(cl >> ao_1b >> c))
                    throw std::runtime_error("molden: bad MO coeff line: '" + u + "'");
                if (ao_1b < 1 || ao_1b > mol.nbf)
                    throw std::runtime_error("molden: MO coeff AO index out of range");
                m.C[ao_1b - 1] = c;
                ++filled;
                ++i;
            }
            if (filled != mol.nbf) {
                std::cerr << "[molden][warn] MO block filled " << filled << "/" << mol.nbf
                          << " coefficients (missing entries assumed zero)\n";
            }
            if (m.spin == 'B') mol.mos_beta.push_back(std::move(m));
            else               mol.mos_alpha.push_back(std::move(m));
        }
    }

    void validate(const Molecule& mol) {
        if (mol.atoms.empty())          throw std::runtime_error("validate: 0 atoms");
        if (mol.shells.empty())         throw std::runtime_error("validate: 0 shells");
        if (mol.nbf <= 0)               throw std::runtime_error("validate: nbf <= 0");
        if (mol.mos_alpha.empty())      throw std::runtime_error("validate: 0 alpha MOs");
        for (const auto& m : mol.mos_alpha)
            if ((int)m.C.size() != mol.nbf) throw std::runtime_error("validate: alpha MO coeff size mismatch");
        for (const auto& m : mol.mos_beta)
            if ((int)m.C.size() != mol.nbf) throw std::runtime_error("validate: beta MO coeff size mismatch");
    }
};

// Pretty summary for debugging.
inline void print_summary(const Molecule& mol, std::ostream& os = std::cerr) {
    os << "=== Molden summary ===\n";
    os << "  atoms: " << mol.atoms.size() << "\n";
    for (size_t a = 0; a < mol.atoms.size(); ++a) {
        const auto& at = mol.atoms[a];
        os << "    [" << a << "] " << at.symbol << " Z=" << at.Z
           << "  xyz(Bohr)=(" << at.xyz.transpose() << ")\n";
    }
    os << "  sph flags: d=" << mol.sph_d << " f=" << mol.sph_f << " g=" << mol.sph_g << "\n";
    os << "  nbf: " << mol.nbf << "   (shells: " << mol.shells.size() << ")\n";
    for (size_t s = 0; s < mol.shells.size(); ++s) {
        const auto& sh = mol.shells[s];
        os << "    shell[" << s << "] atom=" << sh.atom_index
           << " l=" << sh.l << (sh.pure ? " (pure)" : " (cart)")
           << " n_prim=" << sh.exponents.size()
           << " n_ao=" << sh.n_ao
           << " ao_off=" << sh.ao_offset << "\n";
    }
    os << "  alpha MOs: " << mol.mos_alpha.size()
       << "  beta MOs: " << mol.mos_beta.size() << "\n";
    os << "  first 5 alpha eps: ";
    for (size_t i = 0; i < std::min<size_t>(5, mol.mos_alpha.size()); ++i)
        os << mol.mos_alpha[i].energy << " ";
    os << "\n";
}

}  // namespace preproc::molden
