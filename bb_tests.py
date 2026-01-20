from __future__ import annotations

from typing import Sequence, Tuple

import sympy as sp

from bb_code import BBCode


def run_tests(f: sp.Expr, g: sp.Expr, label: str | None = None) -> None:
    bb = BBCode(f, g)
    title = f"[{label}]" if label else ""

    print(f"{title} f:", bb.f)
    print(f"{title} g:", bb.g)

    basis = bb.anyon_space()
    print(f"{title} Anyon basis size:", len(basis))
    print(f"{title} Basis:", basis)

    Lx, Ly = bb.anyon_period()
    print(f"{title} Anyon periods:", (Lx, Ly))

    for i, _ in enumerate(basis):
        vector = [0] * len(basis)
        vector[i] = 1

        s10, t10 = bb.anyon_string(vector, m=1, n=0)
        print(f"{title} String for basis[{i}] with (m,n)=(1,0):")
        print("  s:", s10)
        print("  t:", t10)

        s01, t01 = bb.anyon_string(vector, m=0, n=1)
        print(f"{title} String for basis[{i}] with (m,n)=(0,1):")
        print("  s:", s01)
        print("  t:", t01)
