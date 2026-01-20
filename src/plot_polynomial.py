from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import sympy as sp


@dataclass(frozen=True)
class PlaquettePlot:
    points: List[Tuple[int, int]]
    x_range: Tuple[int, int]
    y_range: Tuple[int, int]


@dataclass(frozen=True)
class EdgePlot:
    horizontal: List[Tuple[int, int]]
    vertical: List[Tuple[int, int]]
    x_range: Tuple[int, int]
    y_range: Tuple[int, int]


def _coeff_is_odd(coeff: sp.Expr) -> bool:
    if coeff.is_Number:
        return int(coeff) % 2 == 1
    return bool(coeff)


def _monomial_exponents(
    term: sp.Expr,
    x: sp.Symbol,
    y: sp.Symbol,
    xinv: Optional[sp.Symbol],
    yinv: Optional[sp.Symbol],
) -> Tuple[int, int]:
    powers = term.as_powers_dict()
    x_exp = int(powers.get(x, 0))
    y_exp = int(powers.get(y, 0))
    if xinv is not None:
        x_exp -= int(powers.get(xinv, 0))
    if yinv is not None:
        y_exp -= int(powers.get(yinv, 0))
    return x_exp, y_exp


def polynomial_to_plaquettes(
    expr: sp.Expr,
    x: sp.Symbol,
    y: sp.Symbol,
    xinv: Optional[sp.Symbol] = None,
    yinv: Optional[sp.Symbol] = None,
) -> PlaquettePlot:
    expr = sp.expand(expr)
    terms = expr.as_ordered_terms() if isinstance(expr, sp.Add) else [expr]
    points: List[Tuple[int, int]] = []

    for term in terms:
        coeff, monomial = term.as_coeff_Mul()
        if _coeff_is_odd(coeff):
            points.append(_monomial_exponents(monomial, x, y, xinv, yinv))

    if not points:
        points = [(0, 0)]

    xs = [p[0] for p in points] + [0]
    ys = [p[1] for p in points] + [0]
    x_range = (min(xs), max(xs))
    y_range = (min(ys), max(ys))

    return PlaquettePlot(points=points, x_range=x_range, y_range=y_range)


def polynomial_to_edges(
    s: sp.Expr,
    t: sp.Expr,
    x: sp.Symbol,
    y: sp.Symbol,
    xinv: Optional[sp.Symbol] = None,
    yinv: Optional[sp.Symbol] = None,
) -> EdgePlot:
    s = sp.expand(s)
    t = sp.expand(t)

    def collect_points(expr: sp.Expr) -> List[Tuple[int, int]]:
        terms = expr.as_ordered_terms() if isinstance(expr, sp.Add) else [expr]
        points: List[Tuple[int, int]] = []
        for term in terms:
            coeff, monomial = term.as_coeff_Mul()
            if _coeff_is_odd(coeff):
                points.append(_monomial_exponents(monomial, x, y, xinv, yinv))
        return points

    horizontal = collect_points(s)
    vertical = collect_points(t)

    points = horizontal + vertical
    if not points:
        points = [(0, 0)]

    xs = [p[0] for p in points] + [0]
    ys = [p[1] for p in points] + [0]
    x_range = (min(xs), max(xs))
    y_range = (min(ys), max(ys))

    return EdgePlot(horizontal=horizontal, vertical=vertical, x_range=x_range, y_range=y_range)


def _merge_ranges(
    a: Tuple[int, int], b: Tuple[int, int]
) -> Tuple[int, int]:
    return min(a[0], b[0]), max(a[1], b[1])


def _draw_background_lattice(
    ax: plt.Axes,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    plaquette_size: float,
    lattice_color: str,
) -> None:
    half = plaquette_size / 2
    for a in range(x_range[0], x_range[1] + 1):
        for b in range(y_range[0], y_range[1] + 1):
            rect = plt.Rectangle(
                (a - half, b - half),
                plaquette_size,
                plaquette_size,
                facecolor="white",
                edgecolor=lattice_color,
                linewidth=0.8,
            )
            ax.add_patch(rect)


def plot_plaquettes(
    expr: sp.Expr,
    x: sp.Symbol,
    y: sp.Symbol,
    output_path: str,
    plaquette_size: float = 0.9,
    highlight_color: str = "#4C78A8",
    lattice_color: str = "#DDDDDD",
    xinv: Optional[sp.Symbol] = None,
    yinv: Optional[sp.Symbol] = None,
) -> PlaquettePlot:
    """Plot monomials as lattice plaquettes and save to an SVG/PDF.

    Each monomial x^a y^b is plotted as a square centered at (a, b).
    The (0,0) plaquette is highlighted.
    """
    if xinv is None:
        xinv = sp.Symbol("xinv") if sp.Symbol("xinv") in expr.free_symbols else None
    if yinv is None:
        yinv = sp.Symbol("yinv") if sp.Symbol("yinv") in expr.free_symbols else None

    plot = polynomial_to_plaquettes(expr, x, y, xinv=xinv, yinv=yinv)

    fig, ax = plt.subplots(figsize=(6, 6))
    half = plaquette_size / 2

    _draw_background_lattice(ax, plot.x_range, plot.y_range, plaquette_size, lattice_color)

    # Highlight plaquettes from the polynomial
    for a, b in plot.points:
        rect = plt.Rectangle(
            (a - half, b - half),
            plaquette_size,
            plaquette_size,
            facecolor=highlight_color,
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(rect)

    ax.set_aspect("equal")
    ax.set_xlim(plot.x_range[0] - 1, plot.x_range[1] + 1)
    ax.set_ylim(plot.y_range[0] - 1, plot.y_range[1] + 1)
    ax.set_xlabel("x exponent")
    ax.set_ylabel("y exponent")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return plot


def plot_plaquettes_and_edges(
    expr: sp.Expr,
    s: sp.Expr,
    t: sp.Expr,
    x: sp.Symbol,
    y: sp.Symbol,
    output_path: str,
    plaquette_size: float = 0.9,
    plaquette_color: str = "#4C78A8",
    edge_color: str = "#54A24B",
    lattice_color: str = "#DDDDDD",
    xinv: Optional[sp.Symbol] = None,
    yinv: Optional[sp.Symbol] = None,
) -> Tuple[PlaquettePlot, EdgePlot]:
    if xinv is None:
        xinv = sp.Symbol("xinv") if sp.Symbol("xinv") in expr.free_symbols else None
    if yinv is None:
        yinv = sp.Symbol("yinv") if sp.Symbol("yinv") in expr.free_symbols else None

    plaquettes = polynomial_to_plaquettes(expr, x, y, xinv=xinv, yinv=yinv)
    edges = polynomial_to_edges(s, t, x, y, xinv=xinv, yinv=yinv)

    x_range = _merge_ranges(plaquettes.x_range, edges.x_range)
    y_range = _merge_ranges(plaquettes.y_range, edges.y_range)

    fig, ax = plt.subplots(figsize=(6, 6))
    half = plaquette_size / 2

    _draw_background_lattice(ax, x_range, y_range, plaquette_size, lattice_color)

    for a, b in plaquettes.points:
        rect = plt.Rectangle(
            (a - half, b - half),
            plaquette_size,
            plaquette_size,
            facecolor=plaquette_color,
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(rect)

    for a, b in edges.horizontal:
        ax.plot([a - 0.5, a + 0.5], [b - 0.5, b - 0.5], color=edge_color, linewidth=2)

    for a, b in edges.vertical:
        ax.plot([a - 0.5, a - 0.5], [b - 0.5, b + 0.5], color=edge_color, linewidth=2)

    ax.set_aspect("equal")
    ax.set_xlim(x_range[0] - 1, x_range[1] + 1)
    ax.set_ylim(y_range[0] - 1, y_range[1] + 1)
    ax.set_xlabel("x exponent")
    ax.set_ylabel("y exponent")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return plaquettes, edges


def plot_edges(
    s: sp.Expr,
    t: sp.Expr,
    x: sp.Symbol,
    y: sp.Symbol,
    output_path: str,
    plaquette_size: float = 0.9,
    edge_color: str = "#54A24B",
    lattice_color: str = "#DDDDDD",
    xinv: Optional[sp.Symbol] = None,
    yinv: Optional[sp.Symbol] = None,
) -> EdgePlot:
    if xinv is None:
        xinv = sp.Symbol("xinv") if sp.Symbol("xinv") in (s * t).free_symbols else None
    if yinv is None:
        yinv = sp.Symbol("yinv") if sp.Symbol("yinv") in (s * t).free_symbols else None

    edges = polynomial_to_edges(s, t, x, y, xinv=xinv, yinv=yinv)

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_background_lattice(ax, edges.x_range, edges.y_range, plaquette_size, lattice_color)

    for a, b in edges.horizontal:
        ax.plot([a - 0.5, a + 0.5], [b - 0.5, b - 0.5], color=edge_color, linewidth=2)

    for a, b in edges.vertical:
        ax.plot([a - 0.5, a - 0.5], [b - 0.5, b + 0.5], color=edge_color, linewidth=2)

    ax.set_aspect("equal")
    ax.set_xlim(edges.x_range[0] - 1, edges.x_range[1] + 1)
    ax.set_ylim(edges.y_range[0] - 1, edges.y_range[1] + 1)
    ax.set_xlabel("x exponent")
    ax.set_ylabel("y exponent")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return edges
