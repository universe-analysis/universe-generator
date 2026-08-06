## The question, in-model

Your universe has one measured equation of state: the dust law w = d/(6T) at the turnaround, with the known w(z) shape over the cycle. The natural next question about any measured quantity is: **is it uniform?** Is w one number the whole universe shares, or is it an average over internally different populations — some worldlines hotter, some colder, with the ensemble value just splitting the difference? Real cosmology is the second kind of thing (radiation, matter, and dark energy all coexist with different w's), so this is the obvious place to look for hidden richness in the model.

Chris's subpath question was exactly this probe: the packed universe comes with a built-in, physically meaningful way to divide it — accretion groups, from lone uniques to 45,000-member giants. If any subdivision of the universe carries its own thermodynamics, that one should.

## Finding one: w is uniform, every way we slice it

We cut the packed universe four independent ways and measured w for each piece:

1. **By spectral makeup** — worldlines whose random amplitudes concentrate into few effective terms vs those spread across many. A mild, real effect (concentrated spectra run ~9% hot), but the amplitude simplex self-averages so strongly that no genuinely distinct subpopulation exists.
2. **By accretion history** — sterile groups vs fertile ones, at the turnaround and across the whole cycle. Flat.
3. **By accounting scheme** — Chris's re-weighting where every group carries equal energy, a drastic reallocation of the budget (sterile groups: 3% → 71% of the weight). Total w moved by a third of a percent.
4. **At the extremes** — lone uniques vs the thousand-member giants, seed by seed. Identical to ±2%.

The meaning: **w is a law of the whole universe, not a statistic that happens to average out.** Every worldline, regardless of what it looks like or what structure it joined, obeys the same equation of state. That's a strong homogeneity statement — something like a thermodynamic equivalence principle for the model: how a path is embedded in structure has no bearing on its contribution to the cosmic pressure. Practically, it also armor-plates the dust law itself: a measured number that survives four independent re-slicings and two weighting schemes is a number you can put in a paper without flinching.

## Finding two: uniform thermodynamics, but violently clumpy structure

Here's the twist that makes the uniformity interesting rather than boring. While the *thermodynamics* refused to differentiate, the *organization* of the universe turned out to be extreme:

- **The mass function is heavy-tailed.** Most uniques accrete nothing; a rare few accrete everything (the largest group we can see holds ≥45,000 paths). The group-size distribution has a scale-free-looking middle and a megagroup tail — qualitatively the halo mass function of real structure formation.
- **Groups are coherent objects.** The discovery at the end of the thread: members of a group share essentially one comoving anchor speed (ICC 0.98–0.996 against a shuffled null of zero — statistically, ~99% identical). Each group is a bound-structure analog with its own bulk "peculiar velocity," members buzzing around inside it, and different groups spanning the entire velocity range.
- **Structure and thermodynamics are fully decoupled.** A group's velocity is uncorrelated with its size (corr ≈ 0.00), and neither velocity nor size buys it a different w.

Step back and look at what produced this: the model has **no forces**. Nothing attracts anything. The only rules are "don't touch" and, for subpaths, "stay in contact with exactly one group." Yet that contact rule alone manufactures binding — only a co-moving candidate can shadow one group for the entire cycle while dodging all others — so exclusion *selects* coherent objects into existence. Force-free structure formation: the model grows its halos out of pure geometry.

That combination — **perfectly smooth equation of state on top of violently lumpy structure** — is itself one of the signature features of our actual universe (the CMB is smooth to 10⁻⁵ while matter is organized into clusters and voids). The model reproduces that separation of layers from packing rules alone, which is the genuinely impressive headline of the week.

## The methodological lesson

The road there had a villain worth remembering: **coherence destroys sample size.** A statistics bin holding 93,000 paths behaved like a bin of forty — because group members are internally correlated, the effective number of independent draws is the number of *groups*, not paths. Every apparent trend in the pooled tables (the 1.11×/0.92× split, the T=40 V-shape) was one outlier seed amplified by that shrinkage, and the per-seed decomposition killed each one. Ironically, the noise was the fingerprint of the discovery: the bins scattered like few draws *because* groups are coherent. Going forward, any group-level question gets its statistics counted in groups and seeds — and if finer resolution is needed, the lever is more seeds.

## Where it points next

The subpath phase is now the model's structure-formation sector, and its open measurables are crisp, all in-model: the mass-function tail exponent and its T-scaling; whether the group velocity distribution exactly matches the proposal ensemble or shows accretion selection; whether coherence extends beyond the anchor to the wiggle spectrum; and whether 3+1 grows the same objects (currently unmeasurable — subpaths exist only in the 2+1 engine). *(Update 2026-08-06: both closed — the subpath phase was ported to the 3+1 engine, and the fullsub3d_e6 smoke run answered yes: same sterile-majority mass function, same iso-anchor coherence (ICC 0.981), same w rigidity, with accretion ~10× slower per attempt; see PHYSICS_FINDINGS §17. The dump-row cap is now a campaign knob, `Campaign.dump_rows`.)* The one engineering debt is the 60k dump-row cap, which censors group sizes at small T.

TL;DR: we asked whether the universe's equation of state hides internal differences, and four independent slicings say no — w is a single uniform law, now very well armored. But the same data revealed that this thermodynamically featureless universe is organizationally rich: exclusion alone grows coherent, co-moving, halo-like structures with a scale-free mass spectrum. Smooth physics, lumpy geography — the same layered character as the real cosmos, out of nothing but packing rules.