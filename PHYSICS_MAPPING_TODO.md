# Physics-Mapping Extraction — Parked Plan (return to this)

A reviewer proposed mapping the geometric model to physical observables. Logged
here so we can resume after the high-T scaling work.

## The crux that governs everything
`CELL = 2/T` is the simulation **resolution**, not a physical length (→0 as
T→∞). So any *absolute-scale* prediction is resolution-dependent and not
physical unless we (a) calibrate T to a physical scale, or (b) show the quantity
is resolution-*independent* (a universal shape / dimensionless ratio / exponent
in CELL units). **Every extraction below must be run at 2-3 values of T to
separate universal physics from CELL artifact.** This is the first gate.

## Two candidate "dictionaries" (mass weight per worldline)
- **E ∝ b** (quantum wave, E=ħω): weight = frequency integer. Trivial to compute.
- **E ∝ proper length** (string tension): weight = ∫ proper length dz from 0→π/2.

## Triage of the three proposed paths
- **Path B — structure factor (DO FIRST).** g(r) and S(k) at z=π/2 are standard,
  **dictionary-free** (pure geometry), and the jammed correlation hole is robust.
  Defensible prediction = the *universal shape* of small-scale suppression in
  CELL units, NOT an absolute cutoff. Strongest, most refutable, computable now.
- **Path A — equation of state w.** Computable, but the clean CDM/WDM
  two-component story is over-read: (1) the odd/even velocity split is *per axis*,
  and the coprimality constraint allows *at most one even* frequency among
  (bx,by,bw), so a worldline is at rest in ≥2 of 3 axes and the "moving"
  population moves along a single axis — only the all-odd fraction is fully at
  rest, not half. (2) "at rest at the turnaround" ≠ persistently cold (they
  oscillate). Compute w as a number, but flag the interpretation.
- **Path C — unitary Fermi gas (DEPRIORITIZE).** RSA rejection is classical
  hard-core exclusion, not Pauli antisymmetry. Apt analogue is a hard-sphere/RSA
  gas, not a *unitary* gas. Revisit only if the velocity dispersion surprises us.

## Concrete first step (when we resume)
Add an engine dump of accepted-worldline **parameters** at jamming (currently we
dump occupancy histograms + the N-vs-attempts curve, not per-worldline params).
Then offline, at exactly z=π/2 (analytic from params), for T = {18, 32, 60}:
1. g(r) and S(k) — check shape collapses across T in CELL units (Path B).
2. Full joint velocity distribution — quantify the real odd/even/per-axis
   structure, at-rest fraction, dispersion (reality-checks Path A).
3. Attach both mass weights (b and proper length) so ρ, p, w are one step away.
