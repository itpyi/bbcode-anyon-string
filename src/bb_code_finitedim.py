"""Utilities for constructing Bivariate Bicycle (BB) codes.

This module provides helpers to build the BB parity-check matrices (Hx, Hz)
from bivariate polynomial specifications.
"""

from __future__ import annotations

from typing import List, Tuple, Sequence, Union

import numpy as np


def _shift_mat(n: int, k: int) -> np.ndarray:
    """Return n x n cyclic left shift-by-k matrix over GF(2) as uint8.
    
    The definition is consistent with the polynomial to qubit mapping.
    """
    return np.roll(np.identity(n, dtype=np.uint8), -int(k) % n, axis=1)


def _parse_bivariate_terms(
    spec: Union[Sequence[Tuple[int, int]], np.ndarray],
) -> List[Tuple[int, int]]:
    """Normalize polynomial spec into list of (i, j) pairs for x^i y^j.

    Accepts:
    - list of pairs: [(i, j), ...] e.g., [(2,0), (1,1), (0,2)] for x^2 + xy + y^2
    - ndarray shape (k, 2) with integer exponents
    """
    if isinstance(spec, np.ndarray):
        arr = np.asarray(spec, dtype=np.uint8)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return [(int(i), int(j)) for i, j in arr]
        raise ValueError("ndarray spec must have shape (k, 2)")

    if len(spec) > 0 and isinstance(spec[0], (list, tuple)) and len(spec[0]) == 2:
        return [(int(p[0]), int(p[1])) for p in spec]

    raise ValueError("Unsupported polynomial spec format for bivariate terms")


def get_BB_matrix(
    a: Union[Sequence[Tuple[int, int]], np.ndarray], l: int, m: int
) -> np.ndarray:
    """Return the (l*m) x (l*m) binary matrix for polynomial a(x, y).

    The polynomial is specified by exponent pairs for monomials x^i y^j. For
    each (i, j), contribute kron(shift_x(i), shift_y(j)).
    """
    terms = _parse_bivariate_terms(a)
    A = np.zeros((l * m, l * m), dtype=np.uint8)
    for ix, iy in terms:
        term = np.kron(_shift_mat(l, ix), _shift_mat(m, iy)).astype(np.uint8)
        A += term  # XOR accumulates modulo-2 for binary matrices in {0,1}
    return A % 2


def get_BB_Hx_Hz(
    a: Union[Sequence[Tuple[int, int]], np.ndarray],
    b: Union[Sequence[Tuple[int, int]], np.ndarray],
    l: int,
    m: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build BB code Hx, Hz from bivariate polynomials a(x,y), b(x,y).

    Accepts list of exponent pairs [(i, j), ...] or an ndarray of shape (k, 2).
    """
    A = get_BB_matrix(a, l, m)
    B = get_BB_matrix(b, l, m)
    Hx = np.concatenate((A, B), axis=1).astype(np.uint8)
    Hz = np.concatenate((B.T, A.T), axis=1).astype(np.uint8)
    return Hx, Hz


__all__ = ["get_BB_matrix", "get_BB_Hx_Hz"]
