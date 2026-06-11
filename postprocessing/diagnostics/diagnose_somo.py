#!/usr/bin/env python3
"""
diagnose_somo.py -- given a molden of an open-shell anion, find the alpha
SOMO (highest-energy alpha occupied MO), decompose its coefficients by
(atom, shell_l, m), and report whether it contains the y-like (m=-1)
channels expected for a fully symmetric t1u state.

If the SOMO has near-zero amplitude in basis functions with m=-1 (p_y,
d_yz, ...) but substantial m=+1 / m=0 / m=±2 / etc., that proves Psi4's
unrestricted SCF picked a single Cartesian-aligned t1u component, which
fully explains the observed sigma_y = 0 in the C8F8 photoionization
output.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


MOLDEN_M_ORDER = {
    # Within a shell of orbital angular momentum l, molden lists
    # the basis functions in this m order:
    0: [0],
    1: [+1, -1, 0],            # p_x, p_y, p_z  (Psi4/molden convention)
    2: [0, +1, -1, +2, -2],
    3: [0, +1, -1, +2, -2, +3, -3],
    4: [0, +1, -1, +2, -2, +3, -3, +4, -4],
    5: [0, +1, -1, +2, -2, +3, -3, +4, -4, +5, -5],
}


def parse_molden(path: Path):
    """Lightweight parser. Returns (atoms, shells, mos)."""
    atoms = []
    shells = []
    mos = []
    section = None
    cur_atom_for_shells = None
    cur_shell = None
    cur_mo = None
    with path.open() as f:
        for raw in f:
            line = raw.rstrip()
            if not line.strip():
                # End of a shell sub-block in [GTO]?
                if section == "GTO" and cur_shell is not None:
                    shells.append(cur_shell)
                    cur_shell = None
                continue
            if line.startswith("["):
                section = line.strip().strip("[]").split()[0].upper()
                if section == "MO":
                    cur_mo = None
                continue
            if section == "ATOMS":
                # "C 1 6 -1.469 -1.469 -1.469"
                parts = line.split()
                atoms.append(dict(
                    sym=parts[0], idx=int(parts[1]), Z=int(parts[2]),
                    xyz=np.array(list(map(float, parts[3:6]))),
                ))
            elif section == "GTO":
                parts = line.split()
                if len(parts) == 2 and parts[0].isdigit():
                    # "<atom_idx> 0"  -- start of a new atom's shell list
                    cur_atom_for_shells = int(parts[0])
                elif len(parts) == 3 and parts[0].isalpha():
                    # "s 5 1.00"  -- new shell header.  Save previous if any.
                    if cur_shell is not None:
                        shells.append(cur_shell)
                    angl = parts[0].lower()
                    L = {"s": 0, "p": 1, "d": 2, "f": 3,
                         "g": 4, "h": 5}[angl]
                    cur_shell = dict(
                        atom=cur_atom_for_shells, l=L,
                        n_prim=int(parts[1]),
                        prims=[],          # list of (exp, coef)
                    )
                else:
                    # "exp coef" primitive line
                    cur_shell["prims"].append(
                        tuple(map(float, parts[:2])))
            elif section == "MO":
                if line.startswith(" Sym"):
                    if cur_mo is not None:
                        mos.append(cur_mo)
                    cur_mo = dict(sym=line.split("=")[1].strip(),
                                  ene=None, spin=None, occ=None, c=[])
                elif " Ene=" in line:
                    cur_mo["ene"] = float(line.split("=")[1])
                elif " Spin=" in line:
                    cur_mo["spin"] = line.split("=")[1].strip()
                elif " Occup=" in line:
                    cur_mo["occ"] = float(line.split("=")[1])
                else:
                    parts = line.split()
                    if len(parts) == 2 and "." in parts[1]:
                        cur_mo["c"].append(float(parts[1]))
    if cur_shell is not None:
        shells.append(cur_shell)
    if cur_mo is not None:
        mos.append(cur_mo)
    return atoms, shells, mos


def basis_function_table(shells):
    """Return list of (atom_idx, l, m, basis_idx) entries in molden order."""
    table = []
    bidx = 0
    for sh in shells:
        L = sh["l"]
        for idx_in_shell, m in enumerate(MOLDEN_M_ORDER[L]):
            table.append(dict(
                atom=sh["atom"], l=L, m=m, idx=bidx,
                shell=sh,
            ))
            bidx += 1
    return table


def main():
    if len(sys.argv) < 2:
        print("usage: diagnose_somo.py <anion.molden>"); sys.exit(2)
    path = Path(sys.argv[1])

    print(f"=== SOMO diagnostic on {path.name} ===\n")
    atoms, shells, mos = parse_molden(path)
    print(f"atoms        : {len(atoms)}")
    print(f"shells       : {len(shells)}")
    print(f"basis funcs  : {sum((sh['l']*0 + (1 if sh['l']==0 else 2*sh['l']+1)) for sh in shells)}")

    bf = basis_function_table(shells)
    Nbf = len(bf)
    print(f"basis funcs (table): {Nbf}\n")

    # Locate the alpha SOMO: highest-energy alpha MO with occ > 0.
    alpha = [m for m in mos if m["spin"] == "Alpha"]
    occupied_alpha = [m for m in alpha if m["occ"] > 0.5]
    somo = max(occupied_alpha, key=lambda m: m["ene"])
    somo_idx = alpha.index(somo)
    print(f"alpha MOs total : {len(alpha)}")
    print(f"alpha occupied  : {len(occupied_alpha)}")
    print(f"SOMO            : alpha MO index {somo_idx}  energy {somo['ene']:.6f} Ha  occ {somo['occ']}")
    print()

    c = np.array(somo["c"])
    if c.size != Nbf:
        print(f"ERROR: SOMO has {c.size} coefficients but basis has {Nbf} functions")
        sys.exit(3)

    # === decomposition per (l, m), summed over atoms ===
    by_lm = defaultdict(float)        # (l, m) -> sum |c|^2
    by_l  = defaultdict(float)        # l -> sum |c|^2
    for k, b in enumerate(bf):
        by_lm[(b["l"], b["m"])] += c[k] ** 2
        by_l [ b["l"]]          += c[k] ** 2
    total = sum(by_lm.values())

    print("Sum |c_i|^2 over basis functions, grouped by (l, m):")
    print(f"  total ||c||^2 = {total:.6f}\n")
    print(f"{'l':>3} {'m':>4}  {'share':>10}     hint")
    for (l, m), s in sorted(by_lm.items()):
        hint = ""
        if l == 1 and m == +1: hint = "<-- p_x (real Y_{1,+1})"
        if l == 1 and m == -1: hint = "<-- p_y (real Y_{1,-1})"
        if l == 1 and m ==  0: hint = "<-- p_z (real Y_{1, 0})"
        if l == 2 and m == +1: hint = "<-- d_xz"
        if l == 2 and m == -1: hint = "<-- d_yz"
        if l == 2 and m ==  0: hint = "<-- d_z2"
        if l == 2 and m == +2: hint = "<-- d_x2-y2"
        if l == 2 and m == -2: hint = "<-- d_xy"
        print(f"{l:3d} {m:+4d}  {s/total*100:>8.2f}%   {hint}")
    print()

    # === per Cartesian direction ===
    # Real-Y q-map: x = m=+1 (and higher m with x-parity),
    # y = m=-1 (and higher),  z = m=0 (only).
    # Strict test: just look at m=+1 vs m=-1 vs m=0 across ALL l.
    s_xlike = sum(s for (l, m), s in by_lm.items() if m == +1)
    s_ylike = sum(s for (l, m), s in by_lm.items() if m == -1)
    s_zlike = sum(s for (l, m), s in by_lm.items() if m ==  0 and l > 0)

    print(f"Aggregated by m (across all l > 0):")
    print(f"  m = +1  (x-like)  : {s_xlike:.6e}    {s_xlike/total*100:6.2f}%")
    print(f"  m = -1  (y-like)  : {s_ylike:.6e}    {s_ylike/total*100:6.2f}%")
    print(f"  m =  0  (z-like)  : {s_zlike:.6e}    {s_zlike/total*100:6.2f}%")
    print()

    # === verdict ===
    print("=== Verdict ===")
    if s_ylike < 1e-6 * total:
        print("  *** The SOMO has effectively ZERO content in m=-1 basis funcs ***")
        print("  *** (p_y, d_yz, f_y*, ...).  This is the smoking gun for     ***")
        print("  *** sigma_y = 0:  the dipole integral <chi_init|y|psi^->     ***")
        print("  *** is identically zero because chi_init has no Y_{l,m=-1}   ***")
        print("  *** content for the operator y to project onto.              ***")
    elif s_ylike < 0.05 * total:
        print(f"  WARNING: y-like share is small ({s_ylike/total*100:.1f}%) but non-zero;")
        print(f"  sigma_y won't be exactly zero, but will be much smaller than sigma_{{x,z}}.")
    else:
        print(f"  y-like share ({s_ylike/total*100:.1f}%) is comparable to x/z;")
        print(f"  sigma_y = 0 is NOT explained by the SOMO basis composition.")
        print(f"  The bug must be downstream (Gaunt / dipole evaluator).")


if __name__ == "__main__":
    main()
