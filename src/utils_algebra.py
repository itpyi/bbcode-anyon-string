"""Shared algebra helpers for BB code modules."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
import ldpc.mod2 as mod2
from sympy import expand, symbols

x, y = symbols("x y")

_ring_groebner_cache: Dict[Tuple[int, int], sp.GroebnerBasis] = {}


def get_ring_groebner(l: int, m: int) -> sp.GroebnerBasis:
    """Return (and cache) the Groebner basis for the ambient ring relations."""

    key = (l, m)
    gb = _ring_groebner_cache.get(key)
    if gb is None:
        gb = sp.groebner([x**l + 1, y**m + 1], [x, y], modulus=2)
        _ring_groebner_cache[key] = gb
    return gb


def monomial_basis(l: int, m: int) -> List[sp.Expr]:
    """Return ordered basis [x^i y^j] with 0 <= i < l, 0 <= j < m."""

    return [x**i * y**j for i in range(l) for j in range(m)]


def apply_periodic_boundary(poly: sp.Expr, l: int, m: int) -> sp.Expr:
    """Reduce polynomial modulo x^l + 1 and y^m + 1."""

    gb = get_ring_groebner(l, m)
    _, rem = gb.reduce(expand(poly))
    return expand(rem)


def poly_to_vector(
    poly: sp.Expr,
    monomials: Sequence[sp.Expr],
    l: int,
    m: int,
) -> np.ndarray:
    """Convert polynomial to coefficient vector modulo the ambient ring."""

    poly_reduced = apply_periodic_boundary(poly, l, m)
    vector = np.zeros(len(monomials), dtype=np.uint8)
    if poly_reduced == 0:
        return vector
    terms = sp.Add.make_args(poly_reduced) if poly_reduced.is_Add else [poly_reduced]
    monomials_list = list(monomials)
    for term in terms:
        coeff, monom_part = term.as_coeff_Mul()
        if int(coeff) % 2 == 1:
            idx = monomials_list.index(monom_part)
            vector[idx] = 1
    return vector


def vector_to_poly(vector: Sequence[int], monomials: Sequence[sp.Expr]) -> sp.Expr:
    """Convert coefficient vector back to polynomial."""

    poly = 0
    for i, coeff in enumerate(vector):
        if int(coeff) % 2 == 1:
            poly += monomials[i]
    return poly


def vector_to_poly_pair(
    vec: Sequence[int],
    monomials: Sequence[sp.Expr],
    l: int,
    m: int,
) -> Tuple[sp.Expr, sp.Expr]:
    """Split a 2*|B| vector into polynomial pair for the two BB blocks."""

    block_size = l * m
    if len(vec) != 2 * block_size:
        raise ValueError("Vector length does not match 2*l*m")
    monomials_list = list(monomials)
    poly_a = vector_to_poly(vec[:block_size], monomials_list)
    poly_b = vector_to_poly(vec[block_size:], monomials_list)
    return poly_a, poly_b


def to_uint8_array(data: Any) -> np.ndarray:
    """Convert dense/sparse input to a uint8 numpy array modulo 2."""

    if hasattr(data, "toarray"):
        dense = data.toarray()
    else:
        dense = np.asarray(data)
    return (dense.astype(np.uint8) % 2)


def to_uint8_matrix(data: Any) -> np.ndarray:
    """Convert dense/sparse input to a 2D uint8 matrix modulo 2."""

    dense = to_uint8_array(data)
    if dense.ndim == 1:
        if dense.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        dense = dense.reshape(1, -1)
    return dense


def row_basis_uint8(matrix: Any, *, num_cols: Optional[int] = None) -> np.ndarray:
    """Return a row basis for the given matrix over GF(2)."""

    basis = to_uint8_matrix(mod2.row_basis(matrix))
    if basis.size == 0 and num_cols is not None:
        return np.zeros((0, num_cols), dtype=np.uint8)
    return basis
