#!/usr/bin/env python3
"""
strassen.py -- Strassen's 2x2 algorithm as an explicit rank-7 tensor decomposition,
verified exactly by tensor_verify, plus a Kronecker-power (tau) ladder.

THE DECOMPOSITION
-----------------
Strassen's seven products, with A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]:

  M1 = (A11+A22)(B11+B22)      C11 = M1 + M4 - M5 + M7
  M2 = (A21+A22) B11           C12 = M3 + M5
  M3 = A11 (B12-B22)           C21 = M2 + M4
  M4 = A22 (B21-B11)           C22 = M1 - M2 + M3 + M6
  M5 = (A11+A12) B22
  M6 = (A21-A11)(B11+B12)
  M7 = (A12-A22)(B21+B22)

Each product is M_r = <U_r, A> * <V_r, B> (Frobenius inner products), and the
contribution of M_r to C is the scalar M_r times the 2x2 matrix W_r.  So

    C = sum_{r=1..7} <U_r,A> <V_r,B> W_r

and at the level of the matmul tensor

    T_<2,2,2> = sum_{r=1..7} U_r (x) V_r (x) W_r ,          all coefficients = 1.

That is a rank-7 decomposition of <2,2,2>, whose naive (schoolbook) rank is 8.
The verifier checks all 2^6 = 64 tensor components of the identity exactly.

tau LADDER
----------
The tau-th Kronecker power of the decomposition is an explicit rank-7^tau
decomposition of <2^tau, 2^tau, 2^tau>.  We verify tau = 1, 2 (and 3 if fast
enough) with the same exact verifier and cross-check the induced algorithms
against naive matmul on random integer matrices.  This is the "value function"
input used by omega.py.
"""

from __future__ import annotations

import random
import sys
import time
from fractions import Fraction

import tensor_verify as tv

# --------------------------------------------------------------------------
# the seven terms: (coeff, U, V, W)
# row/col convention: U is (i,j) on A, V is (j,k) on B, W is (i,k) on C
# --------------------------------------------------------------------------
STRASSEN_TERMS = [
    (1, [[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),    # M1
    (1, [[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),   # M2
    (1, [[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 1]]),   # M3
    (1, [[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [1, 0]]),   # M4
    (1, [[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),   # M5
    (1, [[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),   # M6
    (1, [[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]]),   # M7
]

STRASSEN_LABELS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]


# --------------------------------------------------------------------------
# applying a decomposition as an actual algorithm (exact integers)
# --------------------------------------------------------------------------
def prepare_terms(terms, n):
    """Convert a decomposition to sparse {(r,c): Fraction} form once, so it can be
    applied and re-applied without re-converting dense matrices every time."""
    return [(tv.frac(lam),
             tv.to_sparse(U, n, n, "U"),
             tv.to_sparse(V, n, n, "V"),
             tv.to_sparse(W, n, n, "W"))
            for lam, U, V, W in terms]


def apply_decomposition(terms, A, B, n):
    """Compute A*B (n x n) using the given rank-R decomposition.

    M_r = sum_{i,j} U_r[i][j] A[i][j],  N_r = sum_{j,k} V_r[j][k] B[j][k],
    C   = sum_r M_r N_r W_r .
    Terms may be dense (converted here) or pre-sparsified by prepare_terms().
    Uses exact int/Fraction arithmetic; no floats.
    """
    C = [[Fraction(0)] * n for _ in range(n)]
    for lam, U, V, W in terms:
        ur = U if isinstance(U, dict) else tv.to_sparse(U, n, n, "U")
        vr = V if isinstance(V, dict) else tv.to_sparse(V, n, n, "V")
        wr = W if isinstance(W, dict) else tv.to_sparse(W, n, n, "W")
        Mr = sum(u * A[i][j] for (i, j), u in ur.items())
        Nr = sum(v * B[j][k] for (j, k), v in vr.items())
        if Mr == 0 or Nr == 0:
            continue
        prod = lam * Mr * Nr
        for (i, k), w in wr.items():
            C[i][k] += prod * w
    return C


def naive_matmul(A, B, n):
    return [[sum(A[i][t] * B[t][k] for t in range(n)) for k in range(n)]
            for i in range(n)]


def kron(A, B):
    """Kronecker product of two flat nested lists (exact arithmetic)."""
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    out = []
    for i in range(ra):
        for k in range(rb):
            row = []
            for j in range(ca):
                for l in range(cb):
                    row.append(A[i][j] * B[k][l])
            out.append(row)
    return out


def kron_power_terms(terms, tau, s=2):
    """tau-th Kronecker power decomposition of <s^tau, s^tau, s^tau>.

    The Kronecker power of sum_r U_r (x) V_r (x) W_r is
        sum_{(r1..rtau)} (U_r1 (x)..(x) U_rtau) (x) (V..) (x) (W..)
    i.e. all tuples of tau terms, with coefficient = product of coefficients.
    Rank multiplies: R^tau.
    """
    from itertools import product as iproduct
    cur = [(Fraction(lam), [list(r) for r in U], [list(r) for r in V], [list(r) for r in W])
           for lam, U, V, W in terms]
    for _ in range(tau - 1):
        nxt = []
        for l1, U1, V1, W1 in cur:
            for l2, U2, V2, W2 in terms:
                nxt.append((l1 * Fraction(l2), kron(U1, U2), kron(V1, V2), kron(W1, W2)))
        cur = nxt
    return cur


# --------------------------------------------------------------------------
# random cross-check
# --------------------------------------------------------------------------
def random_case(rng, n, lo=-3, hi=3):
    return ([[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)],
            [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)])


def cross_check(terms, n, n_cases=200, seed=20261126, label=""):
    """Compare the decomposition's output to naive matmul on random integer inputs."""
    rng = random.Random(seed)
    terms = terms if isinstance(terms[0][1], dict) else prepare_terms(terms, n)
    fails = 0
    with_neg = 0
    with_zero = 0
    first_fail = None
    for c in range(n_cases):
        A, B = random_case(rng, n)
        if any(x < 0 for row in A for x in row) or any(x < 0 for row in B for x in row):
            with_neg += 1
        if any(x == 0 for row in A for x in row) or any(x == 0 for row in B for x in row):
            with_zero += 1
        got = apply_decomposition(terms, A, B, n)
        want = naive_matmul(A, B, n)
        if got != [[Fraction(x) for x in row] for row in want]:
            fails += 1
            if first_fail is None:
                first_fail = (c, A, B, got, want)
    ok = fails == 0
    print(f"  {label}cross-check n={n}: {n_cases - fails}/{n_cases} cases match naive matmul "
          f"(exact integer arithmetic, seed={seed}); "
          f"{with_neg} cases had negative entries, {with_zero} had zeros")
    if not ok:
        c, A, B, got, want = first_fail
        print(f"  {label}MISMATCH at case {c}: A={A} B={B}")
        print(f"  {label}  got ={[[str(x) for x in row] for row in got]}")
        print(f"  {label}  want={want}")
    return ok


def hand_picked_cases(terms, n=2):
    """Deterministic edge cases, including negatives and zeros."""
    terms = terms if isinstance(terms[0][1], dict) else prepare_terms(terms, n)
    cases = [
        ("identity * identity", [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        ("zero matrix",         [[0, 0], [0, 0]], [[5, -7], [9, 3]]),
        ("zero * zero",         [[0, 0], [0, 0]], [[0, 0], [0, 0]]),
        ("negatives",           [[-1, -2], [-3, -4]], [[-5, 6], [7, -8]]),
        ("mixed signs + zeros", [[0, -3], [4, 0]], [[-2, 0], [0, 5]]),
        ("one",                 [[1, 1], [1, 1]], [[1, -1], [-1, 1]]),
        ("large ints",          [[123, -456], [789, 0]], [[-11, 12], [13, -14]]),
    ]
    ok = True
    for name, A, B in cases:
        got = apply_decomposition(terms, A, B, n)
        want = naive_matmul(A, B, n)
        good = got == [[Fraction(x) for x in row] for row in want]
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name:<20} A={A} B={B} -> {[[int(x) for x in r] for r in got]}")
    return ok


# --------------------------------------------------------------------------
def main(argv=None):
    t0 = time.perf_counter()
    print("=" * 78)
    print("strassen.py -- Strassen 2x2 as an explicit rank-7 tensor decomposition")
    print("=" * 78)

    print("\n[1] the decomposition, term by term (coeff, U, V, W)")
    for r, ((lam, U, V, W), lab) in enumerate(zip(STRASSEN_TERMS, STRASSEN_LABELS), start=1):
        print(f"  r={r} {lab}  coeff={lam}")
        print(f"        U (on A) = {U}   V (on B) = {V}   W (on C) = {W}")
    print(f"  rank = {len(STRASSEN_TERMS)}")
    print(f"  coefficients = {[t[0] for t in STRASSEN_TERMS]}")

    ok = True

    print("\n[2] exact tensor verification (<2,2,2>, all 64 components)")
    rep = tv.verify(2, 2, 2, STRASSEN_TERMS)
    print("  " + tv.format_report(rep, "strassen").replace("\n", "\n  "))
    ok &= rep["ok"]

    print("\n[3] naive (schoolbook) rank-8 decomposition, same verifier")
    sb = tv.schoolbook(2, 2, 2)
    reps = tv.verify(2, 2, 2, sb)
    print("  " + tv.format_report(reps, "schoolbook").replace("\n", "\n  "))
    ok &= reps["ok"]
    print(f"  rank(strassen) = {len(STRASSEN_TERMS)}  <  rank(schoolbook) = {len(sb)}  "
          f"-> saves {len(sb) - len(STRASSEN_TERMS)} multiplications per 2x2 block")

    print("\n[4] the decomposition must REJECT corrupted variants")
    ok &= tv.demo_rejects(STRASSEN_TERMS, 2, 2, 2, label="strassen rank-7")

    print("\n[5] algorithm cross-check vs naive matmul (exact integers)")
    ok &= hand_picked_cases(STRASSEN_TERMS, 2)
    ok &= cross_check(STRASSEN_TERMS, 2, n_cases=200, seed=20261126, label="tau=1 ")

    print("\n[6] Kronecker-power (tau) ladder: explicit rank-7^tau decompositions of <2^tau,...>")
    ladder = []
    for tau in (1, 2, 3):
        t1 = time.perf_counter()
        pterms = prepare_terms(kron_power_terms(STRASSEN_TERMS, tau), 2 ** tau)
        rank = len(pterms)
        n = 2 ** tau
        try:
            rp = tv.verify(n, n, n, pterms)
            dt = time.perf_counter() - t1
            print(f"  tau={tau}: shape <{n},{n},{n}> rank={rank} "
                  f"({rank}*3 matrices, {n}x{n}) -> "
                  f"{'PASS' if rp['ok'] else 'FAIL'} "
                  f"components_checked={rp['n_components']} "
                  f"mismatches={rp['n_mismatch']}  [{dt:.2f}s]")
            ok &= rp["ok"]
            ladder.append((tau, n, rank, rp["ok"], dt))
        except MemoryError:
            print(f"  tau={tau}: shape <{n},{n},{n}> rank={rank} -> SKIPPED (memory)")
            break
        # cross-check the tau-power as an actual algorithm
        if tau <= 3:
            ok &= cross_check(pterms, n, n_cases=20 if tau == 3 else 50,
                              seed=7 + tau, label=f"tau={tau} ")
        del pterms

    print("\n[7] exponent anchor")
    import omega
    anchor = omega.omega_estimate(tau=1, n=2, m=2, p=2, rank=7)
    import math
    print(f"  omega_estimate(tau=1) for <2,2,2> rank 7 = {anchor!r}")
    print(f"  log2(7)                                  = {math.log2(7)!r}")
    same = abs(anchor - math.log2(7)) < 1e-12
    print(f"  {'PASS' if same else 'FAIL'}: 3*log2(7)/log2(8) == log2(7)  (exact identity, |diff|<1e-12)")
    ok &= same
    ok &= omega.report()

    dt = time.perf_counter() - t0
    print()
    if ok:
        print(f"PASS: strassen.py -- rank 7 verified exactly on <2,2,2>, "
              f"algorithm matches naive matmul on all cross-check cases  [{dt:.2f}s]")
        return 0
    print(f"FAIL: strassen.py  [{dt:.2f}s]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
