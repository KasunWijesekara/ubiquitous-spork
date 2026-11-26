#!/usr/bin/env python3
"""
omega.py -- the tau-symmetrised value function and the exponent it implies.

This is the "9/4-adjacent" piece, deliberately bounded.  It is NOT an attempt
to reproduce any claimed omega = 9/4 result.  It does two honest things:

  1. defines the value / asymptotic-rank functional on a *verified* decomposition
     and the exponent bound it implies, and
  2. recomputes the classical anchor omega <= log2(7) ~= 2.8074 from the rank-7
     Strassen decomposition, and states exactly what a 9/4-type claim would have
     to supply in order to be checkable with tools like these.

THE BOUND USED
--------------
For a rank-R decomposition of <n,m,p>, the standard asymptotic-rank argument
(Strassen; see Bläser, "Fast Matrix Multiplication", or Landsberg's
"Geometry and Complexity Theory") gives

        omega <= 3 * log2(R) / log2(n*m*p).                        (B)

Reason: R <= rank(<n,m,p>) implies omega <= 3 log_{nmp}(R) by taking the
tau-th Kronecker power (the exponent of <n,m,p> tensor powers is
(nmp)^{omega/3}) and letting tau -> infinity.

For <2,2,2> with R = 7 this reads omega <= 3 log2 7 / log2 8 = log2 7, since
log2 8 = 3.  The identity 3*log2(R)/log2(2^3) = log2(R) is why the 2x2x2 case
is the clean anchor.

THE tau-SYMMETRISED FORM
------------------------
The tau-th Kronecker power of a rank-R decomposition is an explicit rank-R^tau
decomposition of <n^tau, m^tau, p^tau> (strassen.kron_power_terms builds it, and
tensor_verify verifies it exactly for small tau).  Define the value

        zeta(tau) = (R_tau)^(1/tau),    R_tau = rank of T^{(x) tau},

so zeta is the "rank per tensor power".  Bound (B) applied to the tau-th power,
and normalised back to the base format, is

        omega <= 3 * log2(zeta(tau)) / log2(n*m*p)      for every tau.   (B')

With the plain Kronecker power R_tau = R^tau we get zeta(tau) = R for all tau,
so (B') is tau-invariant: the Strassen decomposition yields exactly log2(7) at
every tau.  Squaring cannot improve the exponent by itself -- that is the point
of the ladder in section [6] of strassen.py.

Improvements require R_tau < R^tau, i.e. zeta(tau) < R.  Everything in the
fast-matrix-multiplication literature (laser method, asymptotic spectrum,
rectangular formats, "disjoint" combinations) is a way of exhibiting such a
saving for *some* format.  See what_would_be_needed().
"""

from __future__ import annotations

import math

LOG2 = math.log(2.0)


def log2(x: float) -> float:
    return math.log(x) / LOG2


def value_function(rank_tau: float, tau: int = 1) -> float:
    """zeta(tau) = (rank of the tau-th tensor power)^(1/tau).

    "Rank per tensor power" -- the normalised value of a decomposition under
    tau-fold symmetrisation.  zeta(1) is just the rank.
    """
    if tau < 1:
        raise ValueError("tau must be >= 1")
    if rank_tau <= 0:
        raise ValueError("rank must be positive")
    return float(rank_tau) ** (1.0 / tau)


def omega_estimate(tau: int = 1, n: int = 2, m: int = 2, p: int = 2,
                   rank: int = 7, rank_tau: float | None = None) -> float:
    """Exponent bound implied by a decomposition, after tau-fold symmetrisation.

    omega <= 3 * log2(zeta(tau)) / log2(n*m*p),  zeta(tau) = R_tau^(1/tau).

    rank      : rank R of the base decomposition of <n,m,p>
    rank_tau  : explicit rank of the tau-th Kronecker power, if known.
                Defaults to R^tau (the always-available bound, which makes the
                estimate exactly tau-invariant).
    """
    if n < 1 or m < 1 or p < 1:
        raise ValueError("n, m, p must be >= 1")
    R_tau = rank ** tau if rank_tau is None else float(rank_tau)
    zeta = value_function(R_tau, tau)
    return 3.0 * log2(zeta) / log2(float(n) * m * p)


def omega_from_rank(n, m, p, rank):
    """Plain bound (B), tau = 1."""
    return 3.0 * log2(rank) / log2(float(n) * m * p)


# published rank upper bounds, used only as sanity anchors
KNOWN = [
    ("<2,2,2>", 2, 2, 2, 7,  "Strassen 1969 (rank exactly 7)"),
    ("<2,2,3>", 2, 2, 3, 11, "Hopcroft-Kerr / Smirnov"),
    ("<3,3,3>", 3, 3, 3, 23, "Laderman / Smirnov"),
    ("<4,4,4>", 4, 4, 4, 49, "Strassen squared (7^2), verified in strassen.py"),
]


def report() -> bool:
    """Print the tau table, the anchors, and the bounded 9/4 note.  Returns ok."""
    ok = True

    print("\n[8] tau-symmetrised value function, from the verified rank-7^tau ladder")
    print(f"  {'tau':>4} {'format':>14} {'R_tau':>16} {'zeta(tau)':>12} {'omega_est':>12}")
    target = log2(7.0)
    for tau in range(1, 7):
        n = 2 ** tau
        R = 7 ** tau
        z = value_function(R, tau)
        est = omega_estimate(tau=tau, n=2, m=2, p=2, rank=7, rank_tau=R)
        good = abs(est - target) < 1e-12 and abs(z - 7.0) < 1e-9
        ok &= good
        print(f"  {tau:>4} {'<'+str(n)+','+str(n)+','+str(n)+'>':>14} "
              f"{R:>16} {z:>12.6f} {est:>12.9f}   {'ok' if good else 'FAIL'}")
    print(f"  -> tau-invariant, as it must be when R_tau = 7^tau: "
          f"zeta(tau) = 7 and omega_est = log2(7) = {target:.9f}")
    print(f"  {'PASS' if ok else 'FAIL'}: value function reproduces log2(7) for tau = 1..6")

    print("\n[9] anchors: omega implied by published rank upper bounds")
    for name, n, m, p, R, src in KNOWN:
        est = omega_from_rank(n, m, p, R)
        print(f"  {name}  R={R:>3}  omega <= 3*log2({R})/log2({n*m*p}) = {est:.6f}   [{src}]")
    base = omega_from_rank(2, 2, 2, 7)
    print(f"  {'PASS' if abs(base - log2(7.0)) < 1e-12 else 'FAIL'}: "
          f"Strassen anchor omega <= {base:.9f} == log2(7)")

    print()
    ok &= what_would_be_needed()
    return ok


def what_would_be_needed(target: float = 2.25) -> bool:
    """State, numerically, what a 9/4-type claim would have to supply."""
    zeta_needed = 2.0 ** target                      # because omega = log2(zeta) at <2,2,2>
    tau_bits = log2(7.0 / zeta_needed)               # bits saved per tensor power
    print("[10] what a 9/4-type claim would need (we do NOT attempt one)")
    print(f"  target omega = {target}.  Under the <2,2,2> normalisation the bound is "
          f"omega = log2(zeta),")
    print(f"  so omega = {target} requires zeta = 2^{target} = {zeta_needed:.6f} "
          f"instead of zeta = 7.")
    print(f"  Equivalently R_tau must be about {zeta_needed:.6f}^tau rather than 7^tau,")
    print(f"  a saving factor of ({zeta_needed:.6f}/7)^tau = {zeta_needed/7.0:.6f}^tau, i.e. "
          f"{-tau_bits:.6f} bits per tensor power.")
    for tau in (4, 8, 16):
        print(f"    tau={tau:>2}: 7^tau = {7**tau:>24,}   needed ~ 2^{target*tau:g} = "
              f"{int(round(2**(target*tau))):>24,}   ratio {7**tau/2**(target*tau):.2f}x")
    print("  A checkable claim of this shape would have to supply ALL of:")
    print("    (a) an explicit finite decomposition of some <n,m,p> (or a tau-power), with")
    print("        integer/rational entries and its exact rank -- checkable by tensor_verify.py;")
    print("    (b) an argument that R_tau < R^tau genuinely holds (a better base decomposition,")
    print("        or a rectangular/disjoint combination that provably lowers the value);")
    print("    (c) the theory step turning that saving into an asymptotic-rank bound --")
    print("        laser method / asymptotic spectrum / support functionals. This is a")
    print("        mathematical proof, NOT a finite computation, and no decomposition")
    print("        verifier can check it.  That is the boundary of what this artifact does.")
    print("  The plain Kronecker ladder verified in strassen.py shows R_tau = 7^tau exactly,")
    print("  so no saving of this kind is present in Strassen's decomposition itself.")
    return True


def main(argv=None):
    ok = report()
    print()
    print(("PASS" if ok else "FAIL") + ": omega.py -- value function and exponent anchors")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
