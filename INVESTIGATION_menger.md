# Future Investigation: Is the braid a Menger-like knot-fractal?

A parked research direction, not a settled result. Came out of the Dq spectrum
(section 11 of PHYSICS_FINDINGS) plus the knot-selection finding (section 10).

## The observation

The **Menger sponge** has fractal dimension `log 20 / log 3 = 2.7268`, is the
canonical *cubic* fractal (carved by removing subcubes, cubic symmetry), is a
**monofractal** (its entire Dq spectrum is flat at 2.7268), and -- per a recent
published result -- is **knot-universal**: every knot can be found on it.

Our braid:

```
our Dq:   D1=2.80  D2=2.78  D3=2.76  D4=2.740  D5=2.723
cube-correlation D2 = 2.73   |   all-cube convergence = 2.73
Menger:   flat 2.7268
```

So our **high-q / cube-geometry** estimates land essentially on the Menger value;
our low-q / box-counting end (~2.80) is higher. We are mildly multifractal where
Menger is mono -- so at best the *dense, knotted cores* are Menger-like, not the
whole object.

## Why it may be more than a coincidence: three shared features

1. **Dimension** -- our dense-core (high-q) and every cube-probe estimate ~ 2.73.
2. **Cubic symmetry** -- Menger is the cube fractal; our model is cube-structured
   (cubic collision, cubic domain [-1,1]^3, axis-aligned Lissajous rectangles,
   the anisotropy residual). It is specifically our *cube* estimates that hit
   2.7268, not the sphere ones (2.79).
3. **Knots** -- Menger is knot-universal; our braid is *built* from knotted
   worldlines and the packing *selects for knottier* ones at small scales
   (section 10).

## The hypothesis

The Menger sponge is a **removal fractal** (remove a self-similar subcube
pattern, recurse). Our model has its own removal rule -- the
**coprimality / mod-p constraints** carve forbidden combinations out of
*frequency* space (the "at most one even", and Chris's mod-3 / mod-5
generalizations). If those number-theoretic exclusions carve frequency space
into a self-similar Cantor-like set, that is a Menger-like construction, and it
could explain the dimension, the cubic symmetry, and the knot-richness *at once*:
the coprimality rules would do to frequency space what subcube-removal does to
the cube.

## Caveats (do not overclaim)

- Mildly multifractal vs Menger's monofractal -- only the dense cores match.
- The match is the high-q / cube end; sphere / low-q is ~2.80.
- This is an analogy / hypothesis, not an identity. The dimension band 2.7 is not
  by itself special; the knot + cubic-symmetry coincidence is what makes it worth
  chasing.

## Tests to run (pure computation unless noted)

1. **Does the high-q dimension sharpen onto 2.7268?** Push D5 / a clean D_infinity
   with more seeds and higher T; does the dense-core dimension converge exactly to
   log20/log3?
2. **Dimension of the allowed-frequency set.** Treat the coprimality-permitted
   frequency triples as a point set in (bx,by,bw) space and compute *its* fractal
   dimension. If the number-theoretic carving gives ~2.73, that is the Menger-like
   mechanism caught in the act. (This is the most direct test of the hypothesis.)
3. **Knot universality.** Do our packed worldlines realize *all* knot types (like
   the Menger sponge), or a restricted family? Survey the torus-knot types present
   in the dumps.

If (1) and (2) both point to log20/log3 via a self-similar frequency carving,
"the braided universe is a Menger-like knot-fractal" becomes a far sharper claim
than "D ~= 2.78" -- worth the effort to confirm or kill.
