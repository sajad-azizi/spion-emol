#!/usr/bin/env python3
"""
Edit ik_min and ik_max in manifest.h5.

Usage:
  python edit_kgrid.py                    # auto-detect from ik*.h5 files in current dir
  python edit_kgrid.py <ik_min> <ik_max>  # set manually
"""

import sys
import re
import glob
import numpy as np
import h5py


def detect_range():
    files = glob.glob("ik*.h5")
    indices = []
    for f in files:
        m = re.match(r"ik(\d+)\.h5", f)
        if m:
            indices.append(int(m.group(1)))
    if not indices:
        print("No ik*.h5 files found in current directory.")
        sys.exit(1)
    return min(indices), max(indices)


def main():
    if len(sys.argv) == 1:
        ik_min, ik_max = detect_range()
        print(f"Auto-detected range from files: ik_min={ik_min}, ik_max={ik_max}")
    elif len(sys.argv) == 3:
        ik_min, ik_max = int(sys.argv[1]), int(sys.argv[2])
    else:
        print("Usage: python edit_kgrid.py [ik_min ik_max]")
        sys.exit(1)

    if ik_min >= ik_max:
        print(f"Error: ik_min ({ik_min}) must be less than ik_max ({ik_max})")
        sys.exit(1)

    with h5py.File("manifest.h5", "a") as f:
        grp = f["kgrid"]

        old_min = int(grp.attrs["ik_min"])
        old_max = int(grp.attrs["ik_max"])

        # Preserve original dtype (I32 or I64)
        dtype_min = grp.attrs.get("ik_min").dtype
        dtype_max = grp.attrs.get("ik_max").dtype

        grp.attrs["ik_min"] = np.array(ik_min, dtype=dtype_min)
        grp.attrs["ik_max"] = np.array(ik_max, dtype=dtype_max)

        print(f"ik_min: {old_min:>6} → {ik_min}")
        print(f"ik_max: {old_max:>6} → {ik_max}")
        print("Done.")


if __name__ == "__main__":
    main()
