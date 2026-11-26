# 100foom Q-7 — exact tensor-decomposition verifier + verified Strassen decomposition

**Ask (verbatim):** *"a script that verifies the tensor decomposition exactly"* — for the
matrix-multiplication exponent.

**Delivered:** a general exact verifier for `<n,m,p>` matrix-multiplication tensor
decompositions, Strassen's 2×2 algorithm as an explicit rank-7 decomposition that the
verifier accepts, three deliberately corrupted variants that it rejects, a Kronecker-power
(τ) ladder up to an explicitly verified rank-343 decomposition of `<8,8,8>`, and the
classical exponent anchor `ω ≤ log₂7 = 2.807355`.

Pure Python 3 standard library (`fractions.Fraction`). No sympy, no numpy, no network,
no external data, no Lean.

---

## Quick start

```bash
cd /root/scripts/foom_work/m2m3
./verify.sh          # full output, ~7 s
./verify.sh -q       # summary table only
./verify.sh; echo $? # 0 iff everything passed
```

Observed on this box (1 core, Linux 6.8.0-124-generic x86_64, CPython 3.11.15) — this is
the summary table of the run saved as `reference_output.txt`:

```
==============================================================
 100foom Q-7 -- exact tensor-decomposition verifier + Strassen
 host: Linux 6.8.0-124-generic x86_64   cores: 1   python: Python 3.11.15
==============================================================
tensor_verify    PASS   0.25s  PASS: tensor_verify.py self-test (verifier accepts correct, rejects wrong)
strassen         PASS   4.18s  PASS: strassen.py -- rank 7 verified exactly on <2,2,2>, algorithm matches naive matmul on all cross-check cases  [4.02s]
omega            PASS   0.10s  PASS: omega.py -- value function and exponent anchors
json-accept      PASS   0.20s  PASS: strassen_2x2x2.json: <2,2,2> rank 7 mode=full 64 components, 0 mismatches
json-reject      PASS   0.21s  rejected as required (exit 1)
--------------------------------------------------------------
total wall time: 4.94s   (per-program logs: ./last_run/)
--------------------------------------------------------------
PASS: all components -- exact verifier, rank-7 Strassen, omega anchor (exit 0)
```

Wall clock varies with load on the shared vCPU: observed 4.9 s to 7.8 s across runs (see
"Runtime" below). All numbers in this README are taken from real runs; `reference_output.txt`
is the complete, unedited stdout of one of them.

---

## Files

| file | purpose |
|---|---|
| `tensor_verify.py` | the general exact verifier + its own acceptance suite (`--selftest`); also a CLI: `--check-file decomposition.json [--mode full\|diagonal]` |
| `strassen.py` | Strassen's 7 terms as an explicit rank-7 decomposition, rejected-variant tests, 200-case integer matmul cross-check, τ ladder up to rank 343 on `<8,8,8>` |
| `omega.py` | the τ-symmetrised value function, the exponent it implies, published-rank anchors, and the bounded "what a 9/4-type claim would need" note |
| `verify.sh` | one command → PASS/FAIL table + exit code; per-program logs in `last_run/` |
| `strassen_2x2x2.json` | the rank-7 decomposition in portable JSON (sparse form) — proves the verifier is general, not hardcoded |
| `strassen_2x2x2_CORRUPT.json` | same file with one entry changed (`-1 → -2`); the verifier **must** reject it, and does |
| `reference_output.txt` | byte-for-byte stdout of one clean `./verify.sh` run |

Everything is deterministic: fixed seed `20261126` for the 200 random cases, and seeded
`7+τ` for the τ-power cross-checks. Two consecutive full runs produce
**byte-identical mathematical output** (verified with `diff` after stripping wall-clock
strings); only the timings differ.

---

## What the verifier actually checks (exactly)

For `<n,m,p>` — `A` is `n×m`, `B` is `m×p`, `C = A·B` is `n×p` — the
matrix-multiplication tensor is

```
T = Σ_{i<n} Σ_{j<m} Σ_{k<p}  e_ij ⊗ f_jk ⊗ g_ik
```

so component-wise, for index tuples `(i,i')∈[n]², (j,j')∈[m]², (k,k')∈[p]²`:

```
T[(i,j),(j',k),(i',k')] = 1  iff  i=i' and j=j' and k=k',   else 0        (2^6 = 64
                                                                           components for 2×2×2)
```

A rank-R decomposition is a list of terms `(λ_r, U_r, V_r, W_r)` with `U_r ∈ Q^{n×m}`,
`V_r ∈ Q^{m×p}`, `W_r ∈ Q^{n×p}` claiming `T = Σ_r λ_r U_r ⊗ V_r ⊗ W_r`. The verifier
expands that claim into **all n²m²p² components**

```
Σ_r λ_r · U_r[i][j] · V_r[j'][k] · W_r[i'][k']   ?=   T[(i,j),(j',k),(i',k')]
```

and compares with exact rational arithmetic. `verify()` returns the component count, the
mismatch count, the maximum `|actual − expected|`, and up to 5 offending components with
their index tuples. There is no floating point anywhere in the comparison path, so a PASS
is an exact statement about that finite decomposition.

Input formats: dense nested lists **or** sparse `{(r,c): value}` dicts, coefficients as
`int`, `"3/4"`, or `Fraction`; shape mismatches raise `ValueError` instead of passing
silently. JSON form:

```json
{"n":2, "m":2, "p":2,
 "terms": [[coeff, U, V, W], ...]}          // matrices dense, or {"0,1": -1} sparse
```

### The verifier is strictly stronger than the "obvious" check

`--mode diagonal` compares only the `i=i', j=j', k=k'` components (8 of 64 for 2×2×2) —
the check you get if you only look at the coefficient of `A_ij B_jk C_ik`. A decomposition
that has been corrupted in an off-diagonal component passes that check and fails the real
one. Observed, from the real run:

```
[weak check on corrupted decomposition] PASS: <2,2,2> mode=diagonal rank=7 components_checked=8  mismatches=0
[full check on corrupted decomposition] FAIL: <2,2,2> mode=full     rank=7 components_checked=64 mismatches=2
[full check]  component[(i=1,j=1),(j'=1,k=0),(i'=0,k'=1)]  expected=0  actual=1
[full check]  component[(i=1,j=1),(j'=0,k=0),(i'=0,k'=1)]  expected=0  actual=-1
```

`--mode full` is the default and the only mode that constitutes a proof.

---

## Observed results (all pasted from real runs)

**1. The structural controls are accepted.**

```
[schoolbook rank-8]  PASS: <2,2,2> mode=full rank=8  components_checked=64  mismatches=0
[schoolbook rank-27] PASS: <3,3,3> mode=full rank=27 components_checked=729 mismatches=0
```

**2. Strassen's rank-7 decomposition is accepted.**

```
[strassen]            PASS: <2,2,2> mode=full rank=7 components_checked=64 mismatches=0
[schoolbook]          PASS: <2,2,2> mode=full rank=8 components_checked=64 mismatches=0
rank(strassen) = 7  <  rank(schoolbook) = 8  -> saves 1 multiplication per 2x2 block
```

The seven terms with all coefficients 1 (printed in full by `strassen.py`):

| r | U (on A) | V (on B) | W (on C) |
|---|---|---|---|
| 1 | `[[1,0],[0,1]]` | `[[1,0],[0,1]]` | `[[1,0],[0,1]]` |
| 2 | `[[0,0],[1,1]]` | `[[1,0],[0,0]]` | `[[0,0],[1,-1]]` |
| 3 | `[[1,0],[0,0]]` | `[[0,1],[0,-1]]` | `[[0,1],[0,1]]` |
| 4 | `[[0,0],[0,1]]` | `[[-1,0],[1,0]]` | `[[1,0],[1,0]]` |
| 5 | `[[1,1],[0,0]]` | `[[0,0],[0,1]]` | `[[-1,1],[0,0]]` |
| 6 | `[[-1,0],[1,0]]` | `[[1,1],[0,0]]` | `[[0,0],[0,1]]` |
| 7 | `[[0,1],[0,-1]]` | `[[0,0],[1,1]]` | `[[1,0],[0,0]]` |

**3. All corrupted variants are rejected** (3 of 3, with the actual mismatch detail):

| corruption | verdict | mismatches | example component |
|---|---|---|---|
| `W₃[0][0] += 1` | FAIL | 2 | `[(0,0),(1,1),(0,0)] expected=0 actual=-1` |
| `λ₂ += 1` | FAIL | 4 | `[(1,0),(0,0),(1,0)] expected=1 actual=2` |
| last term dropped (rank 6) | FAIL | 4 | `[(0,1),(1,0),(0,0)] expected=1 actual=0` |

Same three corruptions applied to the schoolbook rank-8 decomposition: also 3 of 3
rejected (1 mismatch each).

**4. The algorithm, not just the tensor, is checked.** `strassen.py` runs the
decomposition as an actual algorithm (`M_r = ⟨U_r,A⟩⟨V_r,B⟩`, `C = Σ M_r W_r`) in exact
integer arithmetic:

```
tau=1 cross-check n=2: 200/200 cases match naive matmul (exact integer arithmetic, seed=20261126); 196 cases had negative entries, 137 had zeros
```

plus seven hand-picked cases, all `ok`: identity·identity, zero matrix, zero·zero,
all-negative operands (`[[-1,-2],[-3,-4]]·[[-5,6],[7,-8]] = [[-9,10],[-13,14]]`), mixed
signs with zeros (`[[0,-3],[4,0]]·[[-2,0],[0,5]] = [[0,-15],[-8,0]]`), all-ones, and
three-digit integers.

**5. τ ladder — bigger decompositions, still exactly verified.** The τ-th Kronecker power
of the decomposition is an explicit rank-7^τ decomposition of `<2^τ,2^τ,2^τ>`:

```
tau=1: shape <2,2,2> rank=7   -> PASS components_checked=64       mismatches=0
tau=2: shape <4,4,4> rank=49  -> PASS components_checked=4096     mismatches=0
tau=3: shape <8,8,8> rank=343 -> PASS components_checked=262144   mismatches=0
```

Each is also cross-checked as an algorithm against naive matmul (50/50, 50/50, 20/20
cases). The rank-343 / 262 144-component check is the largest object here and costs
≈2.3–2.7 s.

**6. Exponent anchor.**

```
omega_estimate(tau=1) for <2,2,2> rank 7 = 2.807354922057604
log2(7)                                  = 2.807354922057604
PASS: 3*log2(7)/log2(8) == log2(7)  (exact identity, |diff|<1e-12)
```

**7. Negative control — the harness is not a rubber stamp.** On a throwaway copy of this
directory, one sign was flipped in one entry of one `W` matrix (M5's `W[0][0]`,
`-1 → 1`). Result:

```
flipped one sign in M5's W matrix
tensor_verify    FAIL   0.40s  FAIL: tensor_verify.py self-test
strassen         FAIL   5.45s  FAIL: strassen.py  [5.18s]
omega            PASS   0.16s
json-accept      PASS   0.30s
json-reject      PASS   0.33s
FAIL: at least one component failed -- see the table and ./last_run/ (exit 1)
```

`strassen.py` reported `mismatches=2`, `max|actual-expected| = 2`, and exited 1;
`verify.sh` exited 1. (The `tensor_verify` self-test also fails, because its
weak-vs-full demo asserts that the *real* Strassen decomposition is correct.)

**8. General input path.** `python3 tensor_verify.py --check-file strassen_2x2x2.json`
→ `64 components, 0 mismatches`, exit 0; the same command on
`strassen_2x2x2_CORRUPT.json` → `2 mismatches`, exit 1.

---

## The τ / 9⁄4-adjacent piece (deliberately bounded)

For a rank-R decomposition of `<n,m,p>`, the standard asymptotic-rank bound is

```
ω ≤ 3 · log₂R / log₂(n·m·p)                                            (B)
```

which for `<2,2,2>`, R = 7 reads `3·log₂7/log₂8 = log₂7` (because `log₂8 = 3`). The
τ-symmetrised form replaces R by the value `ζ(τ) = (R_τ)^{1/τ}`, where `R_τ` is the rank of
the τ-th tensor power:

```
ω ≤ 3 · log₂ζ(τ) / log₂(n·m·p)      for every τ.                        (B′)
```

With the *plain* Kronecker power `R_τ = R^τ` we get `ζ(τ) = 7` for every τ, so (B′) is
τ-invariant. Observed:

```
 tau         format            R_tau    zeta(tau)    omega_est
   1        <2,2,2>                7     7.000000  2.807354922   ok
   2        <4,4,4>               49     7.000000  2.807354922   ok
   3        <8,8,8>              343     7.000000  2.807354922   ok
   4     <16,16,16>             2401     7.000000  2.807354922   ok
   5     <32,32,32>            16807     7.000000  2.807354922   ok
   6     <64,64,64>           117649     7.000000  2.807354922   ok
PASS: value function reproduces log2(7) for tau = 1..6
```

Sanity anchors against published rank upper bounds:

```
<2,2,2>  R=  7  omega <= 3*log2(7)/log2(8)  = 2.807355   [Strassen 1969 (rank exactly 7)]
<2,2,3>  R= 11  omega <= 3*log2(11)/log2(12)= 2.894952   [Hopcroft-Kerr / Smirnov]
<3,3,3>  R= 23  omega <= 3*log2(23)/log2(27)= 2.854050   [Laderman / Smirnov]
<4,4,4>  R= 49  omega <= 3*log2(49)/log2(64)= 2.807355   [Strassen squared (7^2), verified above]
```

Note `<4,4,4>` gives *the same* number as `<2,2,2>` — squaring buys nothing, which is the
whole content of the τ-invariance above.

**What would be needed for a 9⁄4-type claim** (we did **not** attempt to reproduce
arXiv:2611.09716; nothing here supports or refutes it). Under the `<2,2,2>` normalisation
`ω = log₂ζ`, so `ω = 2.25` requires `ζ = 2^2.25 = 4.756828` instead of `7` — i.e.
`R_τ ≈ 4.756828^τ` rather than `7^τ`, a saving of `(4.756828/7)^τ = 0.679547^τ`, or
`0.557355` bits per tensor power:

```
 tau= 4: 7^tau =                    2,401   needed ~ 2^9  =                      512   ratio 4.69x
 tau= 8: 7^tau =                5,764,801   needed ~ 2^18 =                  262,144   ratio 21.99x
 tau=16: 7^tau =       33,232,930,569,601   needed ~ 2^36 =           68,719,476,736   ratio 483.60x
```

A checkable claim of that shape would have to supply **all** of:

1. an explicit finite decomposition of some `<n,m,p>` (or a τ-power) with integer/rational
   entries and a stated rank — *this* is what `tensor_verify.py` can check;
2. an argument that `R_τ < R^τ` genuinely holds (a better base decomposition, or a
   rectangular/"disjoint" combination that provably lowers the value);
3. the theory step turning that saving into an asymptotic-rank bound (laser method,
   asymptotic spectrum, support functionals). That is a mathematical proof, **not** a
   finite computation, and no decomposition verifier can check it.

Step 3 is the boundary of this artifact.

---

## What this PROVES

* The identity `T_<2,2,2> = Σ_{r=1..7} U_r ⊗ V_r ⊗ W_r` with the table above — **exactly**,
  all 64 components, over `Q`, no floating point.
* The same for `T_<3,3,3>` at rank 27, for `T_<4,4,4>` at rank 49, and for `T_<8,8,8>` at
  rank 343.
* That the induced algorithms compute `A·B` for 200 random 2×2 integer matrices
  (seed `20261126`, 196 of them containing negatives, 137 containing zeros), for 7
  hand-picked edge cases, and for random 4×4 / 8×8 integer matrices via the τ-powers.
* That three specific single-error corruptions of the decomposition are detected, with the
  offending components identified, and that a corrupted input file exits non-zero.
* That `3·log₂7/log₂8 = log₂7 = 2.807354922057604` and that the τ-ladder does not change it.

## What this does NOT prove

* **Not** that rank(`<2,2,2>) = 7. We show rank ≤ 7 and that *our* rank-6 truncation fails;
  the lower bound `rank ≥ 7` needs a separate argument (Bläser/Winograd-style), which is
  not implemented here.
* **Not** `ω ≤ 2.807` from scratch. That needs bound (B) plus the asymptotic (border-rank)
  argument; both are textbook, but they are *assumed*, not verified by code.
* **Not** any 9⁄4 / ω = 2.25 claim, and not any statement at all about arXiv:2611.09716.
  See the bounded note above.
* **Not** the optimality or minimality of the τ-Kronecker construction: `R_τ ≤ 7^τ` is an
  upper bound that happens to be tight here.
* The 200 random cases are strong evidence, **not** a proof, that the algorithm is correct —
  the proof is the tensor identity in item 1. Anyone can run 200 random trials; only the
  exact component check rules out a decomposition that is right on samples and wrong in
  general.
* Floating-point inputs are accepted but converted to their exact binary value; use ints or
  `"a/b"` strings if you care about the exact rational you wrote.

---

## Runtime measured on this box

1 core (AMD/Intel x86_64, `nproc` = 1), Linux 6.8.0-124-generic, CPython 3.11.15,
`fractions.Fraction` arithmetic. 3 consecutive clean `./verify.sh` runs, then a fourth run after the code was finalised:

| run | summed component time | wall clock (`/usr/bin/time`) | user CPU |
|---|---|---|---|
| 1 | 6.29 s | 6.64 s | 1.92 s |
| 2 | 6.72 s | 7.09 s | 2.08 s |
| 3 | 7.48 s | 7.84 s | 2.21 s |
| 4 (final code, saved as `reference_output.txt`) | 4.94 s | — | — |

So: **≈5–8 s wall clock**, of which ≈2 s is actual CPU; the rest is scheduling on the
shared single vCPU. Observed per-component ranges across ~8 runs: `tensor_verify`
0.20–0.40 s, `strassen` 3.4–6.4 s (dominated by the rank-343 `<8,8,8>` exact check at
2.3–2.7 s), `omega` 0.10–0.20 s, `json-accept` 0.20–0.34 s, `json-reject` 0.21–0.34 s.
Well under the 3-minute budget.

---

## Reviewer checklist (5 independent things to confirm)

1. **The decomposition really is Strassen's, not something else.** Derive `M1…M7` from
   Strassen's formulas by hand and check the sign pattern in the `W` table — in particular
   `W₂[1][1] = -1`, `W₅[0][0] = -1`, `W₆ = [[0,0],[0,1]]`. Then confirm the verifier agrees:
   `python3 tensor_verify.py --check-file strassen_2x2x2.json` → `0 mismatches`.
2. **The check is the full 64-component one, not the 8-component shortcut.** Run
   `python3 tensor_verify.py --check-file strassen_2x2x2_CORRUPT.json --mode diagonal`
   (passes, 0 mismatches) vs the same *without* `--mode diagonal` (fails, 2 mismatches).
   If your reviewer's verifier only does the diagonal, they will not catch corruptions like
   this.
3. **No floating point enters the comparison.** Read `verify()` in `tensor_verify.py`: every
   value is `Fraction`, the accumulator is `Fraction`, and `max_abs_diff` is a `Fraction`
   printed both exactly and as a float only for display. `omega.py` is the only module that
   uses floats, and it never touches the pass/fail decision on a decomposition.
4. **The PASS is not hardcoded.** Repeat the negative control: on a copy, flip one sign in
   one entry of one `W` matrix and confirm `python3 strassen.py; echo $?` gives `1` with
   `mismatches=2`. If a single entry change does not break it, the verifier is not checking
   what it claims.
5. **The exponent arithmetic.** Confirm `log₂8 = 3` hence `3·log₂7/log₂8 = log₂7`, and that
   the τ-table is genuinely τ-invariant (`ζ(τ)=7` for τ=1..6) rather than an average that
   hides a trend — i.e. that `7² = 49` for `<4,4,4>` reproduces `2.807355`, which would be
   impossible if a 2.25-type saving were present. Also confirm the `<2,2,2>` rank claim is
   only `≤ 7` (checklist item above), since the lower bound is a different theorem.

---

## Honest notes / limitations of the run

* The three reject tests each produce a *small* mismatch count (1–4 of 64). That is
  expected — a single-entry corruption touches only a few tensor components — but it means
  the demos exercise only local error detection. Structural corruptions (a dropped term,
  a wrong coefficient) are covered too.
* The `"FAIL: ..."` lines printed by `tensor_verify.py` and `strassen.py` during the reject
  demos are **intentional**. `verify.sh` therefore decides pass/fail from the exit code plus
  the program's final `PASS:`/`FAIL:` line, not by grepping the log for the word FAIL.
* `python3` here has no `pip`/sympy (`python3 -c "import sympy"` →
  `ModuleNotFoundError`). Everything is stdlib by design, which is also why it is portable
  and needs no network.
* Runtime and the `<8,8,8>` verification size were chosen to stay comfortably inside the
  3-minute/1-core budget. τ=4 (rank 2401 on `<16,16,16>`, 16⁶ ≈ 16.7 M components) was not
  attempted; the τ=3 object already exercises 262 144 components.
