# Braided-Universe Packing — Measurement Design

**Status:** draft for review, before committing fleet time.
**Purpose:** state precisely *what* we are trying to determine, *how* we will
measure it, and *which assumptions* a reviewer should challenge.

---

## 1. The model

We pack **worldlines** (braids) into a closed loop `z ∈ (0, π)` — a universe that
expands then contracts. Each axis of a path is a two-term sine:

```
x(z) = a·sin(b·z) + a2·sin(z),   b ∈ ℤ,   |a·b| + |a2| = 1   (slope-1 / "speed of light")
```

with one such term per spatial dimension `d` (d = 2 for "2+1", d = 3 for "3+1").
Paths are compared in **comoving** coordinates `X = x / sin(z)`.

**Collision rule.** Sample `z` at `T` evenly spaced timesteps. Two paths collide
if, at *any* timestep, they fall within `CELL = 2/T` (Chebyshev) of each other in
comoving coordinates `X ∈ [−1, 1]`. Non-colliding paths form a valid braid set.
Boundaries are **open** (no wraparound).

> **Open question (Chris, 2026-06-23) — the packing bound.** Chris notes
> `N_sat ≤ (T/2)^d` as a hard cap and that small-T saturated counts exceed it.
> Under the engine's literal rule (collide iff Chebyshev ≤ CELL, space width 2),
> the geometric cap is actually `T^d` (each path excludes radius CELL/2 → `T`
> per axis), and the data sits at ~10–20% of *that* — no violation. `(T/2)^d`
> is the cap only under a 2× coarser exclusion (min-separation `2·CELL`) or a
> width-1 comoving space. Reconciling this convention — and whether the boundary
> should be an *edge wall* (not a wraparound) — is the open modeling question;
> see §7.0. It does **not** change the sub-dimensional finding (the ratio to the
> *correct* cap `T^d` still declines with `T`).

**Generation.** Draw path parameters at random; accept a candidate iff it does
not collide with any already-accepted path. Repeat until no new path can be
added — the set is **jammed**.

---

## 2. What we are trying to determine

**Primary quantity — the growth exponent `D`:**

```
N_sat(T)  ~  C · T^D
```

`N_sat` is the jammed braid count at resolution `T`. `D` answers *"how does the
number of mutually non-intersecting braids grow with scale?"* The open question
is whether `D = d` (trivial — packing fills the grid) or `D < d` (sub-
dimensional / fractal). This also settles whether the apparent `D ≈ 1.6` in 2+1
is the golden ratio (it is not — see §3).

**Secondary quantities — the kinetics:**

- **Cost exponent `k`:** attempts-to-pack `A(T) ~ T^k`. Empirically `k ≈ 10`,
  *far* steeper than `D`. This is the real "complexity growth."
- **Approach (Feder) exponent `d_eff`:** how the acceptance rate decays toward
  jamming, `N_sat − N(t) ∝ t^(−1/d_eff)`.

---

## 3. Why this is an RSA problem (context the reviewer needs)

This is textbook **Random Sequential Adsorption**: irreversible, random, run to
jamming. Three consequences shape the design:

1. **Jamming ≠ optimal packing.** The random-greedy process freezes below the
   densest arrangement (like a parking lot jamming at ~75%). `N_sat` is a
   *dynamical* fact, not a pure count.
2. **The ceiling is the impoverished question.** Almost all structure lives in
   the *approach* `N(t)`, not the endpoint.
3. **Established empirically so far (exploratory, fixed-budget runs):**
   - `(T/2)^d` is a **crossover approximation, not a law**: fully-jammed small
     systems exceed it by up to 1.7× and cross below it near `T ≈ 26`.
   - The cleanest (fully-saturated, T=10–26) data gives **D ≈ 2.44 in 3+1** —
     sub-dimensional, *not* `d = 3`.
   - The frequency band matters: measured `D` shifts with the max frequency.

These motivate a *principled* campaign to replace the exploratory one.

---

## 4. The measurement problem, and the protocol

### 4.1 The bias we must avoid

If every run uses the **same attempt budget**, larger `T` is less converged
(cost `A(T) ~ T^10`), so the undershoot of `N_sat` *grows with `T`*. That is a
**T-dependent bias** that tilts the slope. It cannot be removed by averaging
seeds (it is common-mode), and extrapolating it introduces its own error (our
exploratory `D = 2.65` was an extrapolation overshoot of the truer `~2.45`).

### 4.2 The fix: hold the convergence level constant

Run every `(T, seed)` to the **same fraction `θ` of jamming**. Then

```
N(T, θ) = θ · C · T^D   ⟹   log N = log(θ C) + D · log T
```

The `θ` shifts the **intercept**, not the **slope**. So the log-log slope is an
unbiased estimate of `D` *for any fixed `θ`* — **we never need to reach jamming**.
A modest `θ` is cheap (the steep cost is all near `θ → 1`), which is what makes
this affordable.

### 4.3 Operationalizing "fixed θ" without knowing the ceiling

We stop each run when its **acceptance rate** falls below a fixed threshold
`r*` (same `r*` for all `T`). This is ceiling-free and directly measurable.

> **Key assumption (to review):** fixed `r*` ≈ fixed `θ`. This holds if the
> kinetic curves are *self-similar* across `T` (same shape in `θ`-vs-rescaled-
> attempts coordinates). We **verify** this post-hoc with a curve-collapse test;
> if it fails, the slope picks up a residual `θ(T)` tilt and we revisit.

### 4.4 Seed averaging

At each `T` we run many seeds and average `N`. Bias is common-mode (cancels in
the slope); only variance remains, and averaging `k` seeds cuts it by `√k`.
Error bars on `D` come from **bootstrapping over the seed ensemble**.

### 4.5 Frequency band

We use **safe-full = round(0.85·T)** as the max frequency: effectively the full
band, but below the Nyquist edge where a discrete-sampling artifact (a path
wiggling through another *between* timesteps — a "strobe miss") would spuriously
inflate `N`. Nyquist (`maxfreq = T`) is logged as a variant to quantify that
edge, but safe-full is the headline number.

---

## 5. Estimators (what the analyzer computes)

| Quantity | Method |
|---|---|
| `D` | Weighted least-squares slope of `log⟨N⟩` vs `log T`, weights `1/σ²` |
| `D` error | Bootstrap over seeds (resample within each `T`, refit) |
| Power-law check | Local slopes between adjacent `T` should agree (constant `D`) |
| Cost exponent `k(f)` | Slope of `log A_f` vs `log T`, `A_f` = attempts to reach `f·N` |
| Self-similarity | Curve collapse of `N/N_final` vs `attempts/A(T)` |
| Reference ratio | `N / (T/2)^d` vs `T` (does Chris's formula hold?) |

The **power-law check** is Chris's consistency test: if the local slopes across
timesteps are not constant, `N_sat ~ T^D` is the wrong functional form and the
average is meaningless.

---

## 6. Proposed experimental design (defaults — please critique)

| Parameter | 2+1 | 3+1 | Rationale |
|---|---|---|---|
| Timestep ladder `T` | 20, 28, 40, 56, 80, 112 | 18, 24, 32, 42, 56 | ~×1.4 spacing, ≥5 points for a slope; 3+1 capped lower (grid memory ~T³) |
| Seeds per `T` | 16 | 16 | variance ↓ by 4×; gives a real error bar |
| Band | safe-full (0.85·T) | safe-full | §4.5 |
| Stop `r*` | 1e-7 | 1e-7 | cheap, well-defined θ; window sees ~30 accepts |
| Max-attempts cap | 3e12 | 3e12 | safety only; `r*` should trigger first |

Fleet: 3090 + 2×3080 + 2070S (as available), resumable orchestration — survives
host drops, reboots, and GPU reclaims (all of which happened during exploration).

**Stopping rule for the campaign:** keep adding seeds until the bootstrap error
on `D` is below a target (proposed: ±0.02), then freeze.

---

## 7. Threats to validity (the review checklist)

0. **[BLOCKING] Seam wrap (§1 correction).** The decisive pre-check: re-run the
   fully-saturated small-T ladder with the toroidal collision. If the
   `N/(T/2)^d` ratio **flattens to a constant**, then `D = d` and the prior
   sub-dimensional `D` was the open-boundary artifact — the campaign then
   measures the jamming prefactor `φ_J`, not an exponent. If it **still curves**,
   sub-dimensional packing is real and the campaign proceeds as designed. No
   fleet time is committed until this resolves.
1. **`r*` ≈ fixed θ.** The load-bearing assumption (§4.3). Verified by collapse;
   if curves do not collapse, the unbiased-slope argument weakens.
2. **Finite-size drift.** `D` may *run* with `T` (small systems show boundary /
   discreteness effects). We test by comparing local slopes across the ladder;
   if `D` trends, we report `D(T)` rather than a single number.
3. **Frequency discreteness.** `b` is integer; at small `T` the candidate set is
   small. The ladder starts high enough (`T ≥ 18`) to mitigate.
4. **Process dependence.** RSA jamming is order-dependent (random-greedy MIS).
   `N_sat` is a property of the *process*, not pure geometry — fine, but it means
   `D` describes this generator, and that should be stated, not hidden.
5. **Aliasing residual.** Even safe-full may carry a small edge artifact; the
   nyquist variant bounds it.
6. **Is `D = d` asymptotically?** Our small-T `D ≈ 2.44` could curve back toward
   3 at large `T`. The ladder's top end is where this would show; if the local
   slope climbs, that is the signal.

---

## 8. Deliverables

- `braidlab` tool suite: declarative campaigns, resumable fleet orchestration,
  SQLite result store, analyzer, HTML report.
- A data set: seed ensembles at fixed convergence, both dimensions, both bands.
- A report: `D ± error` (both dims), the cost law `k(f)`, the collapse plot
  (protocol validation), and the `(T/2)^d` ratio — with the §7 threats addressed.

---

## 9. One-paragraph summary

*We are measuring how the number of mutually non-intersecting braids grows with
resolution, `N_sat ~ T^D`, and how the effort to find them grows, `A ~ T^k`. The
subtlety is that this is RSA jamming, where a fixed compute budget biases `D`
because larger systems are less converged. We remove that bias by stopping every
run at the same acceptance rate (a fixed fraction of jamming), which makes the
log-log slope unbiased for any fraction — so we trade a tiny per-run accuracy hit
for a clean, cheap, averageable estimate, then average many seeds for the error
bar. The central assumption — that equal acceptance rate means equal fraction-of-
jamming across sizes — is checked by a curve-collapse test, and the main risk is
that `D` drifts with system size rather than being a single constant.*
