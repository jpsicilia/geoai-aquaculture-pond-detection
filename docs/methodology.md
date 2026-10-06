# Methodology & Experiment Log

An extended account of the approach, every experiment run, and the lessons —
including an honest post-mortem on where the ceiling turned out to be.

## 1. Problem framing

Each sample is a 144-dimensional vector: 12 bands × 12 months for one
10 m × 10 m patch. No coordinates, no spatial context. The organisers state the
model is trained on one period and evaluated on another, which makes this an
**out-of-distribution generalisation** problem rather than ordinary
classification.

**Diagnosis 1 — transfer, not discrimination.** A gradient-boosted model reaches
internal CV AUC ≈ 0.99 on the training domain but ~0.80 on the test leaderboard.
The classes are nearly separable *within* the source; the whole gap is
generalisation.

**Diagnosis 2 — near-disjoint domains.** An adversarial classifier (train = 0,
test = 1) separates the two with AUC ≈ 1.0. There is minimal overlap, so
source-only cross-validation is unreliable in absolute terms — confirmed
repeatedly (internal CV read ~0.97 while the leaderboard read ~0.80–0.91).

**Diagnosis 3 — structured missingness.** Test is ~60% `-9999`, seasonally
patterned (≈88% missing in Jan/Dec, ≈36% in Jul); train has none. All bands of a
month are missing jointly — an acquisition mask, not clouds.

## 2. What worked

| Technique | Effect | Mechanism |
|---|---|---|
| NaN-aware indices | 0.78 → 0.80 | stop corrupting indices with `-9999` |
| Invariant feature selection | 0.80 → 0.88 | drop features with high `domain_auc` |
| Percentile descriptors | 0.88 → 0.907 | robust to *which* months are missing |
| Missing-data augmentation | stabilises | train sees test-like partial observations |
| Confident self-training (TTA) | 0.907 → 0.9109 | pull the boundary toward the target |

## 3. What did not work (and why)

- **Raw 144 bands (0.57).** Absolute band values carry the shift directly; the
  model collapses on test, predicting 26% ponds where there are ~60%.
- **CORAL (0.76).** Aligning covariance assumes a linear/second-order shift;
  this one is deeper, so alignment injected noise.
- **DANN (0.70 solo / 0.84 blend).** The principled learned version of invariant
  feature selection, but ~1,700 samples under-train a neural net vs. a
  regularised tree.
- **Temporal Transformer + self-supervised pre-training (0.87 blend).** Genuinely
  diverse (corr 0.80 with the tree) but individually weaker; the blend still fell
  short.
- **Harmonic regression / Shimizu-style Fourier features (0.85).** Only the `a0`
  (mean-level) coefficients survived the transferability ranking — essentially
  re-deriving the means. Amplitude/phase were weak; the goodness-of-fit (RMSE)
  was a domain *trap*, exactly the instability the literature warns of under
  heavy missingness.
- **Probability calibration — Saerens EM & isotonic (0.90–0.91).** Diagnostics
  showed the base probabilities were already near-optimally calibrated at the
  mandatory 0.5 threshold, so recalibration gave nothing.
- **Diverse ensembles / multi-K / 3-algorithm stack.** HistGBM and RandomForest
  correlated 0.98+ with LightGBM on the same invariant features — no diversity to
  exploit; blends diluted the single best model.

## 4. Literature context

Peer-reviewed cross-region aquaculture mapping reports overall accuracy in the
**0.83–0.89** range (e.g. Ottinger et al., Sentinel-1 time series; object-based
Hainan studies) — and reaches it using **pond geometry** (rectangular shape,
dikes, segmentation) that this per-pixel tabular dataset does not provide.
Reaching 0.91 on spectral-temporal invariance alone therefore sits at or above
that benchmark, under a stricter information budget.

For the domain shift itself, the relevant literature (invariant feature
construction, growth-stage/phenology alignment, adversarial validation,
test-time adaptation) was systematically applied; see the tables above for which
techniques helped.

## 5. Honest post-mortem

Final standing: **123 / 604 (top 20%)**, public 0.9109 → private 0.9092 — the
model generalised almost perfectly (negligible public→private drop), which is
exactly the property the design optimised for.

However, the top of the board reached **0.95 on the private split**, and those
scores *held* from public to private (several even rose). That rules out the
"they overfit the public 30%" hypothesis I initially favoured: **there was real,
generalisable signal worth ~0.04 that this approach did not capture.** A 5th-place
solution reached 0.953 in only 24 submissions, so the gap was direction, not
brute force.

Most likely missed levers, to revisit against the published top-5 code:
1. A temporal deep model iterated to convergence (abandoned here after two weak
   attempts) rather than a tree on hand-chosen aggregates.
2. A feature transformation not covered by the indices/percentiles explored.

The lesson kept for next time: when many independent teams converge above you and
*stay* there on the private split, treat it as missing signal to find — not as
overfitting to dismiss.
