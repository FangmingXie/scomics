# Methodological notes — ARV, bootstrapping, and the error bar

**Bootstrap distribution.** Resample the cells *with replacement* B times, refit the
statistic each time; the spread of those B values estimates uncertainty without
parametric assumptions. Here `bootstrap_proj_pcha(..., nrepeats=NREPEATS)` does this:
B = `NREPEATS` resamples → B **archetype-position** sets = a bootstrap distribution of
archetype positions.

**ARV (Archetype Relative Variation)** = a single scalar summarizing that
distribution, normalized by archetype spacing:

```
ARV = noise / signal
    = (avg bootstrap scatter of each archetype position)
      / (mean nearest-neighbor distance between consensus archetypes)
```

So ARV ≈ width-of-bootstrap-distribution ÷ archetype-spacing — a scale-free stability
ratio. ARV ≈ 0 → archetypes well-resolved and stable; ARV ≈ 1 → wobble as large as the
gaps between archetypes (too many archetypes requested). **ARV already represents the
bootstrap distribution** — the uncertainty is baked into the single value.

**`NREPEATS` vs `N_OUTER` (the two nested loops).**
- `NREPEATS` (inner) = bootstraps that make up **one** ARV estimate.
- `N_OUTER` (outer) = how many times the whole ARV estimate is repeated to get the
  mean ± std error bar. Since one B-batch yields only one ARV, the outer loop is the
  only way to get multiple ARV values for a band.
- Total PCHA fits per NOC ≈ `N_OUTER × NREPEATS`. Dropping `N_OUTER` 20→3 cut this
  phase ~7× and only coarsens the band, not the ARV value.

**Is the error bar good practice?** Nuanced:
- The ARV *point estimate* is the statistically meaningful quantity (it already
  encodes bootstrap variability of the archetypes).
- The `N_OUTER` mean±std band is **second-order**: it measures how much the ARV
  estimate jitters because B=`NREPEATS` is finite — i.e., **Monte-Carlo / estimator
  noise**, which shrinks toward 0 as `NREPEATS`→∞. It is *not* a confidence interval
  on the biology and should be labeled as estimator precision.
- **But it still carries useful information:** it is a cheap MC **convergence
  diagnostic**. If the band is small relative to the NOC-to-NOC ARV gaps, the current
  bootstrap budget is enough to trust the model-selection ranking; if comparable, you
  need more `NREPEATS` before trusting the call. Our bands are tiny vs the gaps → the
  budget is adequate and the NOC=3 call is reliable.
- **Cleaner alternative:** raise `NREPEATS` for a well-converged single ARV per NOC
  (drop `N_OUTER` to 1), or plot the per-resample bootstrap distribution directly
  (box/violin) rather than mean±std of re-estimates. To improve precision, raise
  `NREPEATS`, not `N_OUTER`.
