## Environment setup

Create a conda environment for computer algebra computations focused on ideals and Groebner bases over Laurent polynomial rings over $\mathbb{F}_2$.

- Use conda-forge packages compatible with macOS (Apple Silicon).
- Primary tools: Singular (and SymPy as a helper library).
- Provide a quick verification example (Groebner basis computation) that runs from the shell.

## Bivariate bicycle (BB) code 

### The class for BB codes

Create a class for BB codes.
It is based on Laurent polynomial ring $R = \mathbb{F}_2[x^{\pm 1}, y^{\pm 1}]$. 
It is specified by two polynomials $f, g \in R$. 

Usage: `from src.bb_code import BBCode; bb = BBCode(f, g)`.

### Anyon space for BB codes

Construct a function computing the anyon space for a BB code.
Mathematically, this is the linear space $R/(f, g)$.
One can do this by first construct Groebner basis for ideal $(f,g)$, and then compute the monomial basis of $R/(f, g)$.
If $(f, g)$ is not maximal, equivalently if the monomial basis is not finite, throw an exception.

Usage: `basis = bb.anyon_space()`.

### Anyon period for BB codes

Construct a function computing the anyon periods $L_x, L_y$ for BB code.
The $L_x$ is defined as the minimal positive integer $L$ such that $x^L + 1 \in (f,g)$.
So is the $L_y$.

Usage: `Lx, Ly = bb.anyon_period()`.

### Compute anyon string

Given an anyon basis $a_i$ of size $n$, a vector $v \in \mathbb{F}_2^n$,
a local anyon is then given by $a=\sum_i a_i v_i$, which is a polynomial.
We want to compute a string that can generate a pair of anyon separated by period (m, n).
That is, we want to compute two polynomials $s,t$ such that $sf + tg = a (x^{m L_x} y^{n L_y}+1)$.
Note that since $L_x, L_y$ are anyon periods, $(x^{m L_x} y^{n L_y}+1) \in (f,g)$ so that it is guaranteed to have a solution.

Usage: `s, t = bb.anyon_string(vector, m, n)`.

## Plot polynomial in a lattice

To visualize the result, we can plot the polynomial in a lattice.

### Plaquette plotting

Given one polynomial $f$, each monomial term $x^a y^b$ can be viewed as a coordinate $(a,b)$.
We can plot this polynomial in a lattice, with the (0,0) plaquette marked.
Range of the lattice should be computed from the polynomial, to put all terms in the figure. 
The output should be an svg or pdf vector graph.

### Edge plotting

Given a pair of polynomials $(s,t)$, view monomials in $s$ as the coordinates for horizontal edges, and $t$ for vertical ones.
Visualise these edges as in the plaquette plotting, using the same colour for horizontal and vertical edges.
Specifically, for an $s$ term $x^a y^b$, the edge is $(a-1/2, b-1/2)$ to $(a+1/2, b-1/2)$. 
For a $t$ term $x^a y^b$, the edge is $(a-1/2, b-1/2)$ to $(a-1/2, b+1/2)$.  