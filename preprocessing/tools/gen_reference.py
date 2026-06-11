#!/usr/bin/env python3
"""
Generate a Psi4 reference calculation for a closed-shell molecule.

Produces, in <out_dir>:
    <name>.molden          Gaussian AOs + MO coefficients + geometry
    <name>_reference.json  Gold-standard numbers to cross-check against:
                             - geometry (Bohr)
                             - basis info (name, cartesian/spherical, nbf)
                             - SCF total / kinetic / nuclear-attraction / two-electron (J, K) energies
                             - orbital energies and occupations
                             - nuclear repulsion
                             - dipole moment (Debye, AU)
                             - Mulliken populations
                             - AO overlap matrix S (flattened, row-major)
                             - MO coefficient matrix C (AO x MO, row-major)

Usage (after `conda activate psi4`):
    python gen_reference.py h2 cc-pvdz              # spherical (default)
    python gen_reference.py h2 cc-pvdz --cartesian  # Cartesian AOs

Molecules known to this script: h2, he, n2, h2o.  Add more as needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import psi4
except ImportError:
    sys.stderr.write("ERROR: psi4 not importable. Run under the psi4 conda env.\n")
    sys.exit(1)


# Geometries in Angstrom at experimental / standard values.
MOLECULES: dict[str, str] = {
    "h2":  "H 0.0 0.0 0.0\nH 0.0 0.0 0.74",
    "he":  "He 0.0 0.0 0.0",
    "n2":  "N 0.0 0.0 0.0\nN 0.0 0.0 1.0977",
    "h2o": (
        "O  0.000000  0.000000  0.117790\n"
        "H  0.000000  0.755453 -0.471161\n"
        "H  0.000000 -0.755453 -0.471161"
    ),
    # Octafluorocubane C8F8 (neutral, O_h symmetry).
    # Geometry from version_0/input/c8f8m_thf_sph.molden, in Bohr:
    #   C at (+-1.47, +-1.47, +-1.47), F at (+-2.9880, +-2.9880, +-2.9880).
    # Input in Angstrom = Bohr / 1.8897261245650618 => 0.7778 A and 1.5812 A.
    "c8f8": "\n".join([
        f"{sym}  {x:+10.6f}  {y:+10.6f}  {z:+10.6f}"
        for sym, R in [("C", 0.7778), ("F", 1.5812)]
        for sx in (-1, +1) for sy in (-1, +1) for sz in (-1, +1)
        for x, y, z in [(sx * R, sy * R, sz * R)]
    ]),
}


# -----------------------------------------------------------------------------
# Minimal independent AO evaluator in molden ordering (s, p, d, f, g).
#
# Parses the molden file back, renormalizes each contracted shell using
# < chi | chi > = 1, and evaluates at 3D points. Each spherical AO is
#
#     chi^sph_{l,m}(r) = |r-A|^l * Y^R_{l,m}(theta,phi)
#                        * scale * sum_a c_a * N_sph(alpha_a, l) * exp(-alpha_a |r-A|^2)
#
# with real Y^R matching the C++ convention in angular/Ylm.hpp.
# -----------------------------------------------------------------------------

def _dfact_odd_neg(n: int) -> float:
    """(2n-1)!! with the convention (-1)!! = 1. Used for Cartesian primitive norm."""
    r = 1.0
    for k in range(1, n + 1):
        r *= (2.0 * k - 1.0)
    return r


def _dfact_pos_odd(l: int) -> float:
    """(2l+1)!! used in spherical primitive norm."""
    r = 1.0
    for k in range(1, l + 1):
        r *= (2.0 * k + 1.0)
    return r


def _sph_prim_norm(alpha: float, l: int) -> float:
    """N_sph(alpha, l), unit-normalized spherical Gaussian primitive."""
    pow2 = 2.0 ** (2.0 * l + 3.5)          # 2^{2l+7/2}
    pow_alpha = alpha ** (l + 1.5)
    return np.sqrt(pow2 * pow_alpha / (np.sqrt(np.pi) * _dfact_pos_odd(l)))


def _prim_overlap_same_shell(alpha: float, beta: float, L: int) -> float:
    p = 2.0 * np.sqrt(alpha * beta) / (alpha + beta)
    return p ** (L + 1.5)


def _sph_legendre_noCS(l: int, m: int, theta: np.ndarray) -> np.ndarray:
    """No-Condon-Shortley normalized associated Legendre S_{l,m}(theta),
    matching the C++ angular/Legendre.hpp implementation exactly."""
    if l < 0 or m < 0 or m > l:
        return np.zeros_like(theta)
    s = np.sin(theta); c = np.cos(theta)
    # Diagonal seed up to S_{m,m}.
    S_mm = np.full_like(theta, 1.0 / np.sqrt(4.0 * np.pi))
    for k in range(1, m + 1):
        S_mm = np.sqrt((2.0 * k + 1.0) / (2.0 * k)) * s * S_mm
    if l == m:
        return S_mm
    # One step up.
    S_lm1 = S_mm
    S_l   = np.sqrt(2.0 * m + 3.0) * c * S_mm
    if l == m + 1:
        return S_l
    # Ascend.
    S_lm2 = S_lm1
    S_lm1 = S_l
    for k in range(m + 2, l + 1):
        a = np.sqrt((4.0 * k * k - 1.0) / (k * k - m * m))
        b = np.sqrt(((k - 1) ** 2 - m * m) / (4.0 * (k - 1) ** 2 - 1.0))
        S_k = a * (c * S_lm1 - b * S_lm2)
        S_lm2 = S_lm1
        S_lm1 = S_k
    return S_lm1


def _real_Ylm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    am = abs(m)
    S = _sph_legendre_noCS(l, am, theta)
    if m == 0:
        return S
    if m > 0:
        return np.sqrt(2.0) * S * np.cos(m * phi)
    return np.sqrt(2.0) * S * np.sin(am * phi)


def _molden_idx_to_m(l: int, idx: int) -> int:
    """Molden in-shell ordering -> m. Mirrors C++ `molden_index_to_m`."""
    if l == 0: return 0
    if l == 1:
        return (+1, -1, 0)[idx]
    if idx == 0: return 0
    half = (idx + 1) // 2
    return +half if (idx & 1) else -half


class MoldenAO:
    """Spherical AO in molden order: chi_{l,m}(r) = r^l * Y^R_{l,m}(theta,phi)
    * sum_a c_tilde_a * exp(-alpha_a r^2).
    """
    __slots__ = ("l", "m", "center", "exponents", "c_tilde")

    def __init__(self, l, m, center, exponents, c_tilde):
        self.l = int(l); self.m = int(m)
        self.center = np.asarray(center, dtype=float)
        self.exponents = np.asarray(exponents, dtype=float)
        self.c_tilde   = np.asarray(c_tilde, dtype=float)

    def eval(self, R: np.ndarray) -> np.ndarray:
        d = R - self.center
        r2 = (d * d).sum(axis=1)
        r  = np.sqrt(r2)
        # ang(r) = r^l * Y^R_{l,m}(theta, phi)   (= 0 at r=0 for l>=1)
        out = np.zeros_like(r)
        nz = r > 1e-300
        if self.l == 0:
            ang = np.full_like(r, 1.0 / np.sqrt(4.0 * np.pi))
        else:
            theta = np.zeros_like(r)
            phi   = np.zeros_like(r)
            theta[nz] = np.arccos(np.clip(d[nz, 2] / r[nz], -1.0, 1.0))
            phi[nz]   = np.arctan2(d[nz, 1], d[nz, 0])
            ang = np.zeros_like(r)
            ang[nz] = (r[nz] ** self.l) * _real_Ylm(self.l, self.m, theta[nz], phi[nz])
        rad = np.exp(-np.outer(r2, self.exponents)) @ self.c_tilde
        out = ang * rad
        return out


def _parse_molden_shells(lines):
    """Return (atoms, shells, sph_flags) from raw molden text lines.
       atoms:  list of (Z, np.array([x,y,z]) in Bohr)
       shells: list of dict {atom: int, l: int, pure: bool, alphas: np.ndarray, coeffs: np.ndarray}
    """
    ANG2BOHR = 1.8897261245650618
    # Pass 1: detect spherical flags, which may appear after [GTO] in the file.
    sph_d = sph_f = sph_g = False
    for ln in lines:
        t = ln.strip()
        if not (t.startswith("[") and "]" in t):
            continue
        low = t.split("]")[0][1:].lower()
        if   low.startswith("5d10f"): sph_d = True
        elif low.startswith("5d7f"):  sph_d = True; sph_f = True
        elif low == "5d":             sph_d = True
        elif low == "7f":             sph_f = True
        elif low == "9g":             sph_g = True
    # Pass 2: parse the sections, using the flags above for per-shell pure-ness.
    section = None
    header_au = False
    atoms = []
    shells = []
    i = 0
    while i < len(lines):
        t = lines[i].strip()
        if t.startswith("["):
            lname = t.split("]")[0][1:].lower()
            if lname.startswith("atoms"):
                section = "atoms"
                header_au = "au" in t.lower()
            elif lname == "gto":
                section = "gto"
            elif lname == "mo":
                section = "mo"
            else:
                section = None
            i += 1
            continue
        if section == "atoms" and t:
            parts = t.split()
            sym = parts[0]; Z = int(parts[2])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            scale = 1.0 if header_au else ANG2BOHR
            atoms.append((Z, np.array([x, y, z]) * scale))
        elif section == "gto" and t:
            # atom header line: "<1based> 0"
            parts = t.split()
            if len(parts) == 2 and parts[1] == "0":
                atom_idx = int(parts[0]) - 1
                i += 1
                while i < len(lines):
                    u = lines[i].strip()
                    if not u:
                        i += 1; break
                    if u.startswith("["):
                        break
                    p = u.split()
                    letter = p[0]
                    n_prim = int(p[1])
                    l_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
                    L = l_map[letter.lower()]
                    alphas = []; coeffs = []
                    i += 1
                    for _ in range(n_prim):
                        pp = lines[i].replace("D", "E").replace("d", "e").split()
                        alphas.append(float(pp[0])); coeffs.append(float(pp[1]))
                        i += 1
                    pure = (L == 0 or L == 1 or
                            (L == 2 and sph_d) or (L == 3 and sph_f) or (L == 4 and sph_g))
                    shells.append({"atom": atom_idx, "l": L, "pure": pure,
                                   "alphas": np.asarray(alphas), "coeffs": np.asarray(coeffs),
                                   "center": atoms[atom_idx][1]})
                continue
        i += 1
    return atoms, shells, (sph_d, sph_f, sph_g)


def _build_molden_aos(shells):
    aos: list[MoldenAO] = []
    for sh in shells:
        L = sh["l"]
        if not sh.get("pure", True) and L >= 2:
            raise RuntimeError(f"Cartesian l={L} shell not supported")
        a = sh["alphas"]; c = sh["coeffs"]
        # Contraction renormalization -- same formula works for any l.
        ov = 0.0
        for p in range(len(a)):
            for q in range(len(a)):
                ov += c[p] * c[q] * _prim_overlap_same_shell(a[p], a[q], L)
        scale = 1.0 / np.sqrt(ov)
        n_ao = 1 if L == 0 else 2 * L + 1
        for idx in range(n_ao):
            m = _molden_idx_to_m(L, idx)
            c_tilde = np.array([scale * c[p] * _sph_prim_norm(a[p], L) for p in range(len(a))])
            aos.append(MoldenAO(L, m, sh["center"], a, c_tilde))
    return aos


def _mo_coeffs_from_molden(lines, nbf):
    """Return list of (energy, occ, spin, C-vector in molden AO order)."""
    mos = []
    cur = None
    in_mo = False
    for ln in lines:
        t = ln.strip()
        if t.startswith("["):
            in_mo = t.lower().startswith("[mo]")
            continue
        if not in_mo or not t:
            continue
        low = t.lower()
        if low.startswith("sym="):
            if cur is not None:
                mos.append(cur)
            cur = {"sym": t.split("=", 1)[1].strip(), "ene": 0.0, "occ": 0.0,
                   "spin": "alpha", "C": np.zeros(nbf)}
        elif low.startswith("ene="):
            cur["ene"] = float(t.split("=", 1)[1].replace("D", "E").replace("d", "e"))
        elif low.startswith("spin="):
            cur["spin"] = t.split("=", 1)[1].strip().lower()
        elif low.startswith("occup="):
            cur["occ"] = float(t.split("=", 1)[1].replace("D", "E").replace("d", "e"))
        else:
            parts = t.replace("D", "E").replace("d", "e").split()
            if len(parts) >= 2:
                idx = int(parts[0]) - 1; val = float(parts[1])
                cur["C"][idx] = val
    if cur is not None:
        mos.append(cur)
    return mos


def _self_check_molden_basis(aos, mos, n_electrons: int) -> dict:
    """Integrate rho and |psi_homo|^2 on a Cartesian grid; report totals."""
    # Crude Cartesian grid on ±5 Bohr, 201 pts per dim (0.05 step). About 8M pts.
    # For an H2 molecule this integrates density to <1e-4 accuracy.
    N = 201
    xs = np.linspace(-5.0, 5.0, N)
    h = xs[1] - xs[0]
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    R = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    nbf = len(aos)
    phi = np.empty((R.shape[0], nbf))
    for m, ao in enumerate(aos):
        phi[:, m] = ao.eval(R)

    # density
    rho = np.zeros(R.shape[0])
    homo = None
    for mo in mos:
        if mo["spin"] != "alpha":
            continue
        psi = phi @ mo["C"]
        if mo["occ"] > 0.0:
            rho += mo["occ"] * psi * psi
            if homo is None or mo["ene"] > homo["ene"]:
                homo = {"ene": mo["ene"], "psi": psi}
    integ_rho = float(rho.sum() * h ** 3)
    integ_homo = float((homo["psi"] ** 2).sum() * h ** 3) if homo is not None else float("nan")
    return {"integral_rho": integ_rho, "expected_N_e": n_electrons,
            "integral_psi_homo_sq": integ_homo, "grid_N": N, "grid_step": h}


def compute_polarizability_cphf(
    name: str, basis: str, cartesian: bool,
    method: str = "scf", verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Gas-phase static dipole polarizability via analytic coupled-
    perturbed HF/KS (CPHF). Fully symmetry-preserving: no external
    perturbation is ever applied, only the linear response equations
    are solved around the unperturbed ground state.

    This calls `psi4.properties(method, properties=['DIPOLE',
    'DIPOLE_POLARIZABILITIES'])`, which in current Psi4 builds reaches
    the analytic polarizability code for both SCF (HF) and DFT
    references without requiring PCM.

    Runs in an ISOLATED Psi4 context; the main unperturbed SCF that
    produced the molden file is already finished and its outputs are
    untouched.
    """
    if name not in MOLECULES:
        raise SystemExit(f"unknown molecule '{name}'")
    geom_body = MOLECULES[name]
    psi4.core.clean()
    psi4.core.clean_variables()
    psi4.geometry(
        "0 1\n" + geom_body +
        "\nsymmetry c1\nno_reorient\nno_com\nunits angstrom\n"
    )
    psi4.set_options({
        "basis":         basis,
        "puream":        not cartesian,
        "reference":     "rhf" if method.lower() in ("scf", "hf", "rhf") else "rks",
        "scf_type":      "df",
        "e_convergence": 1e-10,
        "d_convergence": 1e-8,
        "maxiter":       200,
    })
    e, wfn = psi4.properties(
        method, properties=["DIPOLE", "DIPOLE_POLARIZABILITIES"], return_wfn=True
    )
    keys = {
        (0, 0): "DIPOLE POLARIZABILITY XX",
        (1, 1): "DIPOLE POLARIZABILITY YY",
        (2, 2): "DIPOLE POLARIZABILITY ZZ",
        (0, 1): "DIPOLE POLARIZABILITY XY",
        (0, 2): "DIPOLE POLARIZABILITY XZ",
        (1, 2): "DIPOLE POLARIZABILITY YZ",
    }
    alpha = np.zeros((3, 3))
    for (i, j), key in keys.items():
        try:
            v = float(psi4.variable(key))
        except Exception:
            v = 0.0
        alpha[i, j] = v
        alpha[j, i] = v
    iso = float(np.trace(alpha) / 3.0)
    # Anisotropy magnitude (Kramers): sqrt(1/2 [(a-b)^2 + (b-c)^2 + (c-a)^2 + 6(d^2+e^2+f^2)])
    diag_vals = [alpha[0, 0], alpha[1, 1], alpha[2, 2]]
    off = [alpha[0, 1], alpha[0, 2], alpha[1, 2]]
    aniso = float(np.sqrt(0.5 * (
        (diag_vals[0] - diag_vals[1]) ** 2
        + (diag_vals[1] - diag_vals[2]) ** 2
        + (diag_vals[2] - diag_vals[0]) ** 2
        + 6.0 * sum(x * x for x in off)
    )))
    diag = {
        "iso":   iso,
        "aniso": aniso,
        "xx":    float(alpha[0, 0]), "yy": float(alpha[1, 1]), "zz": float(alpha[2, 2]),
        "xy":    float(alpha[0, 1]), "xz": float(alpha[0, 2]), "yz": float(alpha[1, 2]),
        "alpha_tensor": alpha.flatten().tolist(),
        "method":       f"analytic_cphf_{method.lower()}",
        "no_pcm":       True,
    }
    if verbose:
        print(f"  [pol] method={method}  alpha_iso = {iso:.4f} a0^3  aniso = {aniso:.4f}")
        print(f"  [pol] diag = [{alpha[0,0]:.4f}, {alpha[1,1]:.4f}, {alpha[2,2]:.4f}]")
        print(f"  [pol] off  = [xy {alpha[0,1]: .2e}  xz {alpha[0,2]: .2e}  yz {alpha[1,2]: .2e}]")
    return alpha, diag


def run(name: str, basis: str, cartesian: bool, out_dir: Path,
        polarizability: bool = False,
        pol_method: str = "scf",
        charge: int = 0,
        multiplicity: int = 1,
        pcm_solvent: str | None = None) -> None:
    if name not in MOLECULES:
        raise SystemExit(f"unknown molecule '{name}'. known: {list(MOLECULES)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{name}_{basis.replace('-', '').replace('*', 's')}"
    tag += "_cart" if cartesian else "_sph"
    if charge != 0:
        tag += f"_q{charge:+d}"
    if multiplicity != 1:
        tag += f"_m{multiplicity}"
    if pcm_solvent is not None:
        # compress the solvent name into the tag (e.g. "Tetrahydrofurane" -> "thf")
        short = "".join(ch for ch in pcm_solvent.lower() if ch.isalnum())[:6]
        tag += f"_pcm{short}"

    psi4.core.clean()
    psi4.core.set_output_file(str(out_dir / f"{tag}_psi4.out"), False)
    psi4.set_memory("2 GB")
    psi4.set_num_threads(2)

    # Build geometry. We always use the SAME base geometry for neutral and
    # anion (vertical attachment); only charge/multiplicity change.
    is_open_shell = (multiplicity != 1)
    geom = (f"{charge} {multiplicity}\n" + MOLECULES[name]
            + "\nsymmetry c1\nno_reorient\nno_com\nunits angstrom\n")
    mol = psi4.geometry(geom)

    # Pick scf_type based on size: pk is deterministic for small systems,
    # df is required for large ones (cubane C8F8 cc-pvdz etc.).
    scf_type = "pk" if name not in ("c8f8",) else "df"
    psi4.set_options({
        "basis":          basis,
        "puream":         not cartesian,
        "reference":      "uhf" if is_open_shell else "rhf",
        "scf_type":       scf_type,
        "e_convergence":  1e-10,
        "d_convergence":  1e-8,
        "ints_tolerance": 1e-12,
        "maxiter":        200,
        "print_mos":      False,
    })

    # Optional PCM solvent (needed for the anion workflow).
    if pcm_solvent is not None:
        psi4.set_options({"pcm": True, "pcm_scf_type": "total"})
        pcm_input = (
            "Medium {\n"
            "    SolverType = IEFPCM\n"
            f"    Solvent    = {pcm_solvent}\n"
            "}\n"
            "Cavity {\n"
            "    Type = GePol\n"
            "    RadiiSet = UFF\n"
            "    Scaling = True\n"
            "    Mode = Implicit\n"
            "}\n"
        )
        psi4.pcm_helper(pcm_input)
        print(f"  [pcm] enabled solvent: {pcm_solvent}")

    e_scf, wfn = psi4.energy("scf", return_wfn=True)

    # --- molden file ---
    molden_path = out_dir / f"{tag}.molden"
    psi4.molden(wfn, str(molden_path))

    # --- one-electron energy decomposition ---
    mints = psi4.core.MintsHelper(wfn.basisset())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    S = np.asarray(mints.ao_overlap())
    Ca = np.asarray(wfn.Ca())                # AO x MO (alpha spin)
    Da = np.asarray(wfn.Da())                # AO density (alpha spin)
    if is_open_shell:
        Cb = np.asarray(wfn.Cb())            # AO x MO (beta spin)
        Db = np.asarray(wfn.Db())
        D_tot = Da + Db
    else:
        Cb = None
        Db = None
        D_tot = 2.0 * Da                     # closed-shell total density

    E_kin  = float(np.einsum("ij,ij->", D_tot, T))
    E_nuc_el = float(np.einsum("ij,ij->", D_tot, V))
    E_nn = float(mol.nuclear_repulsion_energy())

    # Two-electron energies. For small closed-shell systems compute J, K
    # directly from the full ERI tensor. For open-shell or large systems
    # we just report E_2e from the total (reference numbers are
    # molden/orbital-focused; J/K breakdown isn't used downstream).
    nbf = int(wfn.basisset().nbf())
    if (nbf <= 80) and (not is_open_shell):
        I = np.asarray(mints.ao_eri())       # (pq|rs) in chemist's notation, shape (nbf,)*4
        J_mat = np.einsum("pqrs,rs->pq", I, D_tot)
        K_mat = np.einsum("prqs,rs->pq", I, D_tot)
        E_J = 0.5 * float(np.einsum("ij,ij->", D_tot, J_mat))
        E_K = -0.25 * float(np.einsum("ij,ij->", D_tot, K_mat))
        E_2e = E_J + E_K
    else:
        # Recover from the total energy; J/K individually marked null.
        E_2e = float(e_scf) - E_nn - (E_kin + E_nuc_el)
        E_J = None
        E_K = None

    E_1e = E_kin + E_nuc_el
    E_scf_check = E_nn + E_1e + E_2e

    # --- orbital energies, occupations ---
    eps_alpha = np.asarray(wfn.epsilon_a())
    nalpha_occ = wfn.nalpha()
    if is_open_shell:
        eps_beta = np.asarray(wfn.epsilon_b())
        nbeta_occ = wfn.nbeta()
        # Open-shell: each spin orbital is singly occupied or empty.
        occ_alpha = np.zeros_like(eps_alpha); occ_alpha[:nalpha_occ] = 1.0
        occ_beta  = np.zeros_like(eps_beta ); occ_beta [:nbeta_occ]  = 1.0
        # For the JSON, keep "orbital_energies_hartree" and "occupations" as
        # the ALPHA set (the SOMO = anion's extra electron lives there); the
        # beta set is dumped separately.
        eps = eps_alpha
        occ = occ_alpha
    else:
        # Closed shell: alpha orbitals doubly occupied.
        occ_alpha = np.zeros_like(eps_alpha); occ_alpha[:nalpha_occ] = 2.0
        eps_beta = None
        occ_beta = None
        eps = eps_alpha
        occ = occ_alpha

    # --- dipole ---
    psi4.oeprop(wfn, "DIPOLE")
    dx = float(wfn.variable("SCF DIPOLE")[0])
    dy = float(wfn.variable("SCF DIPOLE")[1])
    dz = float(wfn.variable("SCF DIPOLE")[2])

    # --- geometry in Bohr ---
    natom = mol.natom()
    atoms = []
    for i in range(natom):
        atoms.append({
            "symbol": mol.symbol(i),
            "Z": int(mol.Z(i)),
            "xyz_bohr": [float(mol.x(i)), float(mol.y(i)), float(mol.z(i))],
        })

    # For large systems, skip the pointwise ESP and the full-C dump to save memory/disk.
    is_large = (nbf > 80)

    # --- parse the molden we just wrote, to get MO coeffs in *molden AO order* ---
    # Psi4's internal Ca() uses CCA spherical ordering (m=-l..+l); molden uses
    # a different per-shell convention (p: x,y,z; d: z2,xz,yz,x2y2,xy; ...).
    # We parse the text we produced so the C++ cross-check can compare apples to apples.
    mo_molden_rows: list[list[float]] = []   # row = MO, col = AO (molden order)
    mo_molden_energies: list[float] = []
    mo_molden_occ: list[float] = []
    mo_molden_spin: list[str] = []
    with open(molden_path) as f:
        lines = f.readlines()
    in_mo = False
    cur_C: list[float] | None = None
    cur_meta: dict = {}
    for ln in lines:
        t = ln.strip()
        if t.startswith("["):
            in_mo = (t.lower().startswith("[mo]"))
            if cur_C is not None:
                mo_molden_rows.append(cur_C); cur_C = None
            continue
        if not in_mo or not t:
            continue
        low = t.lower()
        if low.startswith(("sym=", "ene=", "spin=", "occup=")):
            if low.startswith("sym="):
                if cur_C is not None:
                    mo_molden_rows.append(cur_C)
                    mo_molden_energies.append(cur_meta.get("ene", 0.0))
                    mo_molden_occ.append(cur_meta.get("occup", 0.0))
                    mo_molden_spin.append(cur_meta.get("spin", "alpha"))
                cur_C = [0.0] * int(wfn.basisset().nbf())
                cur_meta = {"sym": t.split("=", 1)[1].strip()}
            elif low.startswith("ene="):
                cur_meta["ene"] = float(t.split("=", 1)[1].replace("D", "E").replace("d", "e"))
            elif low.startswith("spin="):
                cur_meta["spin"] = t.split("=", 1)[1].strip().lower()
            elif low.startswith("occup="):
                cur_meta["occup"] = float(t.split("=", 1)[1].replace("D", "E").replace("d", "e"))
        else:
            parts = t.replace("D", "E").replace("d", "e").split()
            if len(parts) >= 2 and cur_C is not None:
                ao_1b = int(parts[0]); val = float(parts[1])
                cur_C[ao_1b - 1] = val
    if cur_C is not None:
        mo_molden_rows.append(cur_C)
        mo_molden_energies.append(cur_meta.get("ene", 0.0))
        mo_molden_occ.append(cur_meta.get("occup", 0.0))
        mo_molden_spin.append(cur_meta.get("spin", "alpha"))

    basis_obj = wfn.basisset()
    ref = {
        "molecule": name,
        "basis": basis_obj.name(),
        "cartesian": bool(cartesian),
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "pcm_solvent": pcm_solvent if pcm_solvent is not None else "none",
        "is_open_shell": bool(is_open_shell),
        "nbf": int(basis_obj.nbf()),
        "nao_cart": int(basis_obj.nao()),
        "natom": natom,
        "nalpha": int(wfn.nalpha()),
        "nbeta": int(wfn.nbeta()),
        "nelectron": int(wfn.nalpha() + wfn.nbeta()),
        "ndocc": nalpha_occ,
        "atoms": atoms,
        "energies": {
            "E_scf": float(e_scf),
            "E_scf_reconstructed": E_scf_check,
            "E_nuclear_repulsion": E_nn,
            "E_kinetic": E_kin,
            "E_nuclear_electron": E_nuc_el,
            "E_coulomb_J": E_J,
            "E_exchange_K": E_K,
            "E_one_electron": E_1e,
            "E_two_electron": E_2e,
        },
        "orbital_energies_hartree": eps.tolist(),          # alpha set
        "occupations":             occ.tolist(),           # alpha set
        "orbital_energies_beta":   (eps_beta.tolist() if eps_beta is not None else None),
        "occupations_beta":        (occ_beta.tolist() if occ_beta is not None else None),
        "dipole_au": [dx, dy, dz],
        "overlap_S": {"shape": list(S.shape), "rowmajor": S.flatten().tolist()},
        "mo_coeff_C_ao_by_mo": {"shape": list(Ca.shape), "rowmajor": Ca.flatten().tolist()},
        # MOs as they appear in the molden file (molden's AO ordering).
        # Shape: (n_mo, nbf).  Each row matches an alpha-spin MO in file order.
        "mo_coeff_molden": {
            "shape": [len(mo_molden_rows), basis_obj.nbf()],
            "rowmajor": [float(c) for row in mo_molden_rows for c in row],
            "energies": mo_molden_energies,
            "occupations": mo_molden_occ,
            "spins": mo_molden_spin,
        },
        "density_total_D": {"shape": list(D_tot.shape), "rowmajor": D_tot.flatten().tolist()},
        "_provenance": {
            "psi4_version": psi4.__version__,
            "scf_type": "pk",
            "e_convergence": 1e-12,
            "d_convergence": 1e-10,
        },
    }

    # -------------------------------------------------------------------
    # Build an independent Python AO evaluator in molden ordering, then dump
    # psi_HOMO and rho at a set of test points for the C++ cross-check.
    # -------------------------------------------------------------------
    atoms_parsed, shells_parsed, _sph = _parse_molden_shells(lines)
    if is_large:
        print(f"  [note] large basis (nbf={nbf}) — skipping pointwise Python reference.")
    elif any((not sh.get("pure", True)) and sh["l"] >= 2 for sh in shells_parsed):
        print("  [note] Cartesian d/f/g shells present — pointwise reference skipped.")
    else:
        aos_py = _build_molden_aos(shells_parsed)
        mos_py = _mo_coeffs_from_molden(lines, int(wfn.basisset().nbf()))

        # Self-check: integrate rho and |psi_HOMO|^2.
        chk = _self_check_molden_basis(aos_py, mos_py, int(wfn.nalpha() + wfn.nbeta()))
        print(f"  [self-check] integral_rho        = {chk['integral_rho']:.6f}  (expected {chk['expected_N_e']})")
        print(f"  [self-check] integral |psi_HOMO|^2 = {chk['integral_psi_homo_sq']:.6f}  (expected 1)")

        # Choose test points: a line along z through both atoms + some off-axis points + random.
        rng = np.random.default_rng(42)
        line_z = np.stack([np.zeros(21), np.zeros(21), np.linspace(-2.5, 4.0, 21)], axis=1)
        line_x = np.stack([np.linspace(-2.0, 2.0, 11), np.zeros(11), np.full(11, 0.7)], axis=1)
        random = rng.uniform(low=-3.0, high=3.0, size=(40, 3))
        # Include the bond midpoint and a very-far-field point (should be ~0).
        extras = np.array([[0.0, 0.0, 0.5 * 1.3983973328],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.3983973328],
                           [10.0, 0.0, 0.0]])
        pts = np.concatenate([line_z, line_x, random, extras], axis=0)

        # Build nbf x npts AO matrix.
        phi = np.empty((pts.shape[0], len(aos_py)))
        for m, ao in enumerate(aos_py):
            phi[:, m] = ao.eval(pts)
        # rho
        rho_pts = np.zeros(pts.shape[0])
        # identify HOMO (highest occupied)
        homo_mo = None
        for mo in mos_py:
            if mo["spin"] != "alpha" or mo["occ"] == 0.0:
                continue
            if homo_mo is None or mo["ene"] > homo_mo["ene"]:
                homo_mo = mo
        for mo in mos_py:
            if mo["spin"] != "alpha" or mo["occ"] == 0.0:
                continue
            psi = phi @ mo["C"]
            rho_pts += mo["occ"] * psi * psi
        psi_homo_pts = phi @ homo_mo["C"]

        # --- Independent V_H reference via Psi4's libint electrostatic integrals ---
        # mints.electrostatic_potential_value returns V_elec(P) = -V_H(P).
        # (Potential at P due to electrons of charge -1.) We negate below.
        #
        # Use a subset of test points that are comfortably away from the
        # nuclei, so cusp-like pointwise interpolation of V_H^R_{lm}(r) in
        # the C++ side does not mask real errors.
        far_pts = []
        for p in pts:
            # exclude points within 0.15 Bohr of any atom
            d_min = min(float(np.linalg.norm(p - a)) for a in [atoms_parsed[i][1] for i in range(natom)])
            if d_min > 0.15:
                far_pts.append(p)
        far_pts = np.asarray(far_pts)
        charges = psi4.core.Vector.from_array(np.ones(far_pts.shape[0]))
        coords_mat = psi4.core.Matrix.from_array(far_pts)
        D_psi4 = psi4.core.Matrix.from_array(D_tot)
        vh_psi4 = -np.asarray(
            psi4.core.MintsHelper(wfn.basisset())
            .electrostatic_potential_value(charges, coords_mat, D_psi4)
        )

        ref["pointwise_reference"] = {
            "note": "Values from independent Python molden-basis evaluator. "
                    "Use for bit-accurate C++ cross-check.",
            "n_points": int(pts.shape[0]),
            "points_xyz_bohr_rowmajor": pts.flatten().tolist(),
            "rho": rho_pts.tolist(),
            "psi_homo": psi_homo_pts.tolist(),
            "homo_energy_hartree": float(homo_mo["ene"]),
            "integration_self_check": chk,
            # V_H sub-block: independent path via Psi4 libint.
            "v_H": {
                "source": "psi4.MintsHelper.electrostatic_potential_value (negated); uses libint "
                          "analytic <mu|1/|r-P||nu> integrals contracted with density matrix D.",
                "n_points": int(far_pts.shape[0]),
                "points_xyz_bohr_rowmajor": far_pts.flatten().tolist(),
                "V_H_hartree": vh_psi4.tolist(),
            },
        }

    # ------------------------------------------------------------------
    # Polarizability (gas-phase, finite-field). Computed ONLY after the
    # molden + MO coefficients are fully written, so nothing here feeds
    # back into the preprocessing input.
    # ------------------------------------------------------------------
    if polarizability:
        print(f"  [pol] computing gas-phase static dipole polarizability "
              f"via analytic CPHF (method={pol_method}, no PCM)...")
        alpha_tensor, pol_diag = compute_polarizability_cphf(
            name, basis, cartesian, method=pol_method
        )
        ref["polarizability"] = {
            "note": ("Gas-phase static dipole polarizability (atomic units, a0^3). "
                     "Computed by analytic CPHF (coupled-perturbed HF/KS) linear "
                     "response via psi4.properties(method, "
                     "properties=['DIPOLE_POLARIZABILITIES']). Fully symmetry-"
                     "preserving -- no external field is ever applied to the "
                     "Hamiltonian. The main SCF producing the molden file is "
                     "not affected by this step."),
            **pol_diag,
        }

    json_path = out_dir / f"{tag}_reference.json"
    with open(json_path, "w") as f:
        json.dump(ref, f, indent=2)

    # Terse summary to stdout
    print(f"[done] {tag}")
    print(f"  molden : {molden_path}")
    print(f"  json   : {json_path}")
    print(f"  E_scf  = {e_scf:.12f} Ha")
    print(f"  E_scf (reconstructed from integrals) = {E_scf_check:.12f} Ha")
    print(f"  |delta| = {abs(e_scf - E_scf_check):.2e}")
    print(f"  E_nn   = {E_nn:.12f}")
    print(f"  E_kin  = {E_kin:.12f}")
    print(f"  E_Vne  = {E_nuc_el:.12f}")
    print(f"  E_J    = {E_J:.12f}" if E_J is not None else "  E_J    = None (skipped)")
    print(f"  E_K    = {E_K:.12f}" if E_K is not None else "  E_K    = None (skipped)")
    print(f"  eps    = {eps.tolist()}")
    print(f"  occ    = {occ.tolist()}")
    print(f"  dipole (au) = [{dx:.6e}, {dy:.6e}, {dz:.6e}]")
    print(f"  nbf    = {basis_obj.nbf()}   (puream={not cartesian})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("molecule", choices=list(MOLECULES))
    ap.add_argument("basis")
    ap.add_argument("--cartesian", action="store_true", help="Use Cartesian AOs (6d/10f) instead of spherical")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "reference_data"))
    ap.add_argument("--polarizability", action="store_true",
                    help="Also compute gas-phase static dipole polarizability (analytic CPHF)")
    ap.add_argument("--pol-method", default="scf",
                    help="Method for polarizability CPHF (scf/hf/rhf or a DFT functional like b3lyp)")
    # Open-shell / solvent / non-default charge (for anion initial-state runs).
    ap.add_argument("--charge", type=int, default=0,
                    help="Molecular charge (default 0; use -1 for anion)")
    ap.add_argument("--multiplicity", type=int, default=1,
                    help="Spin multiplicity (default 1 = singlet; 2 = doublet for open-shell anion)")
    ap.add_argument("--pcm-solvent", default=None,
                    help="Enable PCM implicit solvent with this PCMSolver solvent name "
                         "(e.g. 'Tetrahydrofurane', 'Water'). Default: gas phase.")
    args = ap.parse_args()
    run(args.molecule, args.basis, args.cartesian, Path(args.out),
        polarizability=args.polarizability, pol_method=args.pol_method,
        charge=args.charge, multiplicity=args.multiplicity,
        pcm_solvent=args.pcm_solvent)


if __name__ == "__main__":
    main()
