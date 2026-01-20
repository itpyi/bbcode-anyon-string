# Study the path of abelian anyons in 2d translational invariant systems

## Conda environment (computer algebra)

Create the environment for Groebner basis / ideal computations over Laurent polynomial rings (via Singular + SymPy):

- Create: `conda env create -f environment.yml`
- Activate: `conda activate anyon-algebra`

Quick Singular check (modeling Laurent ring by adding inverses and relations):

```
singular -q <<'EOF'
ring r = 2,(x,y,xinv,yinv),lp;
ideal I = x*y + 1, x2 + y, x*xinv - 1, y*yinv - 1;
groebner(I);
EOF
```

