#!/usr/bin/env python3
"""
tensor_verify.py -- exact verifier for matrix-multiplication tensor decompositions.

Target: 100foom Q-7, "a script that verifies the tensor decomposition exactly".

WHAT IS CHECKED
---------------
For <n,m,p> (A is n x m, B is m x p, C = A*B is n x p) the matrix-multiplication
tensor is

    T = sum_{i<n} sum_{j<m} sum_{k<p}  e_ij (x) f_jk (x) g_ik          (*)

living in  R^{n x m} (x) R^{m x p} (x) R^{n x p}.  Component-wise, for index
tuples (i,i') in [n], (j,j') in [m], (k,k') in [p],

    T[(i,j),(j',k),(i',k')] = 1  iff  i=i' and j=j' and k=k',  else 0.

A rank-R decomposition is a list of R terms  (lam_r, U_r, V_r, W_r)  with
U_r (n x m), V_r (m x p), W_r (n x p) and rational scalars lam_r, claiming

    T = sum_{r=1..R} lam_r * U_r (x) V_r (x) W_r .                    (**)

The verifier expands (**) into all n^2 m^2 p^2 components

    sum_r lam_r * U_r[i][j] * V_r[j'][k] * W_r[i'][k']   ?=   T[(i,j),(j',k),(i',k')]

and compares with (*) using exact rational arithmetic (fractions.Fraction).
No floating point is used anywhere in the check, so a PASS is an exact proof
of the identity for that finite decomposition.

NOTE ON STRENGTH
----------------
The 'diagonal' mode (--mode diagonal) only compares the components with
i=i', j=j', k=k'.  That is the weak check that people often do by accident:
it verifies the coefficient of A_ij B_jk C_ik but never looks at the
cross components A_ij B_j'k' C_i'k' with mismatched indices.  A decomposition
can pass 'diagonal' and fail the real identity -- demo_weak_vs_full() below
exhibits exactly such a case.  Only 'full' (the default) is a proof.

Pure stdlib.  Deterministic.  No network.
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from itertools import product

__all__ = [
    "frac", "to_sparse", "verify", "structural_components",
    "schoolbook", "demo_rejects", "demo_weak_vs_full", "main",
]


# --------------------------------------------------------------------------
# exact scalar / matrix normalisation
# --------------------------------------------------------------------------
def frac(x) -> Fraction:
    """Exact conversion to Fraction.  ints and '3/4' strings are exact;
    floats are converted to their exact binary value (use ints/str in practice)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):          # bool is an int subclass -- be explicit
        return Fraction(int(x))
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, str):
        return Fraction(x)
    if isinstance(x, float):
        return Fraction(x)           # exact, but the input float was already lossy
    raise TypeError(f"cannot use {type(x).__name__} as an exact scalar: {x!r}")


def to_sparse(mat, rows, cols, name="matrix"):
    """Normalise a matrix to {(r,c): Fraction} keeping only nonzero entries.

    Accepts a dense nested sequence (list of rows) or a sparse dict
    {(r,c): value}.  Raises on shape mismatch so a mis-typed decomposition
    cannot silently 'pass'.
    """
    out = {}
    if isinstance(mat, dict):
        for key, val in mat.items():
            try:
                r, c = key
            except Exception:
                raise ValueError(f"{name}: sparse key {key!r} is not a pair")
            if not (isinstance(r, int) and isinstance(c, int)):
                raise ValueError(f"{name}: sparse key {key!r} must be ints")
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(f"{name}: sparse key {key!r} out of shape {rows}x{cols}")
            fv = frac(val)
            if fv:
                out[(r, c)] = out.get((r, c), Fraction(0)) + fv
        return out

    if isinstance(mat, (str, bytes)):
        raise ValueError(f"{name}: expected a sequence of rows, got {type(mat).__name__}")
    try:
        nrows = len(mat)
    except TypeError:
        raise ValueError(f"{name}: expected a sequence of rows")
    if nrows != rows:
        raise ValueError(f"{name}: expected {rows} rows, got {nrows}")
    for r, row in enumerate(mat):
        try:
            ncols = len(row)
        except TypeError:
            raise ValueError(f"{name}: row {r} is not a sequence")
        if ncols != cols:
            raise ValueError(f"{name}: row {r} has {ncols} entries, expected {cols}")
        for c, val in enumerate(row):
            fv = frac(val)
            if fv:
                out[(r, c)] = out.get((r, c), Fraction(0)) + fv
    return out


def structural_components(n, m, p):
    """The nonzero components of the matrix-multiplication tensor (*)."""
    return {(i, j, j, k, i, k): Fraction(1)
            for i in range(n) for j in range(m) for k in range(p)}


# --------------------------------------------------------------------------
# the verifier
# --------------------------------------------------------------------------
def verify(n, m, p, terms, mode="full", max_examples=5):
    """Expand a decomposition and compare it to the structural tensor.

    terms: iterable of (lam, U, V, W).  U,V,W dense nested lists or sparse dicts.

    Returns a report dict:
      ok            bool          exact equality of every checked component
      rank          int           number of terms supplied
      n_components  int           number of tensor components compared
      n_mismatch    int           components where actual != expected
      max_abs_diff  Fraction      max |actual - expected| over all components
      examples      list          up to max_examples (key, expected, actual)
    """
    if mode not in ("full", "diagonal"):
        raise ValueError("mode must be 'full' or 'diagonal'")
    if min(n, m, p) < 1:
        raise ValueError("n, m, p must all be >= 1")

    terms = list(terms)
    acc = {}                      # (i,j,j',k,i',k') -> Fraction
    zero_coeff_terms = 0

    for idx, term in enumerate(terms):
        if len(term) != 4:
            raise ValueError(f"term {idx}: expected (coeff, U, V, W), got {len(term)} items")
        lam, U, V, W = term
        lam = frac(lam)
        if lam == 0:
            zero_coeff_terms += 1
            continue
        su = to_sparse(U, n, m, f"term[{idx}].U")
        sv = to_sparse(V, m, p, f"term[{idx}].V")
        sw = to_sparse(W, n, p, f"term[{idx}].W")
        for (i, j), u in su.items():
            for (j2, k), v in sv.items():
                for (i2, k2), w in sw.items():
                    if mode == "diagonal" and not (i == i2 and j == j2 and k == k2):
                        continue
                    key = (i, j, j2, k, i2, k2)
                    acc[key] = acc.get(key, Fraction(0)) + lam * u * v * w

    expected = structural_components(n, m, p)
    if mode == "diagonal":
        # only the index-aligned components are ever populated in this mode
        expected = {k: v for k, v in expected.items()}

    n_components = (n * n) * (m * m) * (p * p) if mode == "full" else (n * m * p)

    mismatch_examples = []
    n_mismatch = 0
    max_abs = Fraction(0)
    for key in set(acc) | set(expected):
        a = acc.get(key, Fraction(0))
        e = expected.get(key, Fraction(0))
        if a != e:
            n_mismatch += 1
            d = abs(a - e)
            if d > max_abs:
                max_abs = d
            if len(mismatch_examples) < max_examples:
                mismatch_examples.append((key, e, a))

    return {
        "ok": n_mismatch == 0,
        "mode": mode,
        "shape": (n, m, p),
        "rank": len(terms),
        "zero_coeff_terms": zero_coeff_terms,
        "n_components": n_components,
        "n_mismatch": n_mismatch,
        "max_abs_diff": max_abs,
        "examples": mismatch_examples,
    }


def format_report(rep, label=""):
    head = f"[{label}] " if label else ""
    st = "PASS" if rep["ok"] else "FAIL"
    lines = [f"{head}{st}: <{rep['shape'][0]},{rep['shape'][1]},{rep['shape'][2]}> "
             f"mode={rep['mode']} rank={rep['rank']} "
             f"components_checked={rep['n_components']} mismatches={rep['n_mismatch']}"]
    if not rep["ok"]:
        lines.append(f"{head}      max|actual-expected| = {rep['max_abs_diff']} "
                     f"(= {float(rep['max_abs_diff']):.6g})")
        for key, e, a in rep["examples"]:
            i, j, j2, k, i2, k2 = key
            lines.append(f"{head}      component[(i={i},j={j}),(j'={j2},k={k}),(i'={i2},k'={k2})]"
                         f"  expected={e}  actual={a}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# reference decompositions used as controls
# --------------------------------------------------------------------------
def schoolbook(n, m, p):
    """The rank n*m*p schoolbook (naive) decomposition: C = sum_j A_ij B_jk.
    Known-correct by construction -- the verifier must accept it."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U = {(i, j): 1}
                V = {(j, k): 1}
                W = {(i, k): 1}
                terms.append((1, U, V, W))
    return terms


def _copy_terms(terms):
    """Deep-ish copy so corruption tests cannot mutate shared state."""
    out = []
    for lam, U, V, W in terms:
        def cp(M):
            if isinstance(M, dict):
                return dict(M)
            return [list(row) for row in M]
        out.append((lam, cp(U), cp(V), cp(W)))
    return out


# --------------------------------------------------------------------------
# demos: the verifier must reject wrong decompositions
# --------------------------------------------------------------------------
def demo_rejects(terms, n, m, p, label="decomposition"):
    """Corrupt a known-good decomposition three ways and prove each is rejected."""
    base = verify(n, m, p, terms)
    ok = base["ok"]
    print(format_report(base, f"{label}: baseline"))

    results = []

    # (1) corrupt one entry of one W matrix (the C-side factor)
    bad = _copy_terms(terms)
    idx = 2 % len(bad)
    lam, U, V, W = bad[idx]
    if isinstance(W, dict):
        W = dict(W)
        W[(0, 0)] = W.get((0, 0), Fraction(0)) + 1
    else:
        W = [list(r) for r in W]
        W[0][0] = frac(W[0][0]) + 1
    bad[idx] = (lam, U, V, W)
    r = verify(n, m, p, bad)
    print(format_report(r, f"{label}: corrupted W entry (+1)"))
    results.append(("corrupted W entry", r))

    # (2) scale one coefficient
    bad = _copy_terms(terms)
    lam, U, V, W = bad[1 % len(bad)]
    bad[1 % len(bad)] = (lam + 1, U, V, W)
    r = verify(n, m, p, bad)
    print(format_report(r, f"{label}: wrong coefficient (+1)"))
    results.append(("wrong coefficient", r))

    # (3) drop one term (rank R-1 truncation)
    bad = _copy_terms(terms)[:-1]
    r = verify(n, m, p, bad)
    print(format_report(r, f"{label}: term dropped (rank {len(bad)})"))
    results.append(("term dropped", r))

    good = ok and all(not rep["ok"] for _, rep in results)
    print("  " + ("PASS" if good else "FAIL") +
          f": {label}: accepted the correct decomposition, rejected all "
          f"{len(results)} corrupted variants")
    return good


def demo_weak_vs_full(n=2, m=2, p=2):
    """Show that the 'diagonal' (index-aligned) check is NOT sufficient.

    Strassen with one off-diagonal entry of W_3 nudged passes the diagonal
    check and fails the full tensor check.  This is why 'full' is the default.
    """
    from strassen import STRASSEN_TERMS
    terms = [t for t in STRASSEN_TERMS]
    ok_base = verify(n, m, p, terms)["ok"]

    bad = []
    for lam, U, V, W in terms:
        bad.append([lam, [list(r) for r in U], [list(r) for r in V], [list(r) for r in W]])
    # W_3 is the 4th term; its (0,1) entry is off the index-aligned diagonal
    bad[3][3][0][1] = bad[3][3][0][1] + 1

    diag = verify(n, m, p, bad, mode="diagonal")
    full = verify(n, m, p, bad, mode="full")
    print(format_report(diag, "weak check on corrupted decomposition"))
    print(format_report(full, "full check on corrupted decomposition"))
    good = ok_base and diag["ok"] and (not full["ok"])
    print("  " + ("PASS" if good else "FAIL") +
          ": 'diagonal' check is fooled by an off-diagonal corruption "
          "while the full tensor check rejects it")
    return good


def load_decomposition(path):
    """Load {"n":..,"m":..,"p":..,"terms":[[coeff,U,V,W],...]} from a JSON file.

    Matrices may be dense nested lists or sparse {"r,c": value} dicts with
    string keys ("0,1") so the file stays plain JSON.
    """
    import json
    with open(path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    for k in ("n", "m", "p", "terms"):
        if k not in spec:
            raise ValueError(f"{path}: missing key {k!r}")
    n, m, p = int(spec["n"]), int(spec["m"]), int(spec["p"])
    terms = []
    for idx, t in enumerate(spec["terms"]):
        if len(t) != 4:
            raise ValueError(f"{path}: term {idx} must be [coeff, U, V, W]")
        lam, U, V, W = t
        def conv(M):
            if isinstance(M, dict):
                out = {}
                for key, val in M.items():
                    r, c = (int(x) for x in str(key).split(","))
                    out[(r, c)] = frac(val)
                return out
            return M
        terms.append((frac(lam), conv(U), conv(V), conv(W)))
    shape = (n, m, p)
    for idx, (lam, U, V, W) in enumerate(terms):
        to_sparse(U, n, m, f"term[{idx}].U")
        to_sparse(V, m, p, f"term[{idx}].V")
        to_sparse(W, n, p, f"term[{idx}].W")
    return shape, terms


def _extra_tests():
    """Rational coefficients, and shape mismatch must be rejected loudly."""
    ok = True

    # rational rescaling: coeff 1/3 with W*3 is still the same decomposition
    sb = schoolbook(2, 2, 2)
    rescaled = []

    def scale(M, k):
        if isinstance(M, dict):
            return {key: k * v for key, v in M.items()}
        return [[k * v for v in row] for row in M]

    for lam, U, V, W in sb:
        rescaled.append((Fraction(1, 3), U, V, scale(W, 3)))
    r = verify(2, 2, 2, rescaled)
    print(format_report(r, "schoolbook with coeff=1/3, W*3 (rationals)"))
    ok &= r["ok"]

    # a mis-shaped matrix must raise, not silently pass
    bad = _copy_terms(sb)
    bad[0] = (bad[0][0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], bad[0][2], bad[0][3])
    try:
        verify(2, 2, 2, bad)
        print("  FAIL: a 3x3 U in a <2,2,2> decomposition was accepted -- shape check broken")
        ok = False
    except ValueError as exc:
        print(f"  PASS: mis-shaped term rejected with ValueError: {exc}")
    return ok


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="exact verifier for <n,m,p> tensor decompositions")
    ap.add_argument("--selftest", action="store_true", help="run the built-in acceptance suite")
    ap.add_argument("--check-file", metavar="JSON",
                    help='verify a decomposition from JSON: {"n":..,"m":..,"p":..,"terms":[[coeff,U,V,W],...]}')
    ap.add_argument("--mode", default="full", choices=["full", "diagonal"],
                    help="'full' (default, a proof) or 'diagonal' (weak, demo only)")
    args = ap.parse_args(argv)

    if args.check_file:
        shape, terms = load_decomposition(args.check_file)
        rep = verify(*shape, terms, mode=args.mode)
        print(f"file: {args.check_file}")
        print(format_report(rep, "file"))
        n, m, p = shape
        weak = " (WEAK: index-aligned components only -- NOT a proof)" if args.mode == "diagonal" else ""
        print(("PASS" if rep["ok"] else "FAIL") +
              f": {args.check_file}: <{n},{m},{p}> rank {len(terms)} mode={args.mode} "
              f"{rep['n_components']} components, "
              f"{rep['n_mismatch']} mismatches{weak}")
        return 0 if rep["ok"] else 1

    if not args.selftest:
        ap.print_help()
        return 0

    print("tensor_verify.py -- exact <n,m,p> tensor-decomposition verifier")
    print("arithmetic: fractions.Fraction (exact rationals), no floats in the check")
    print()
    allgood = True

    # schoolbook rank 8 on 2x2x2 -- known correct, must be accepted
    sb = schoolbook(2, 2, 2)
    r = verify(2, 2, 2, sb)
    print(format_report(r, "schoolbook rank-8"))
    allgood &= r["ok"]

    # 3x3x3 schoolbook (rank 27) -- exercises a larger component count
    r = verify(3, 3, 3, schoolbook(3, 3, 3))
    print(format_report(r, "schoolbook rank-27"))
    allgood &= r["ok"]

    print()
    allgood &= demo_rejects(sb, 2, 2, 2, label="schoolbook rank-8")

    print()
    allgood &= demo_weak_vs_full()

    print()
    allgood &= _extra_tests()

    print()
    if allgood:
        print("PASS: tensor_verify.py self-test (verifier accepts correct, rejects wrong)")
        return 0
    print("FAIL: tensor_verify.py self-test")
    return 1


if __name__ == "__main__":
    sys.exit(main())
