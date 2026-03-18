# Volume 5 — Checking Our Work

> *How do we know the model is right? What does "right" even mean for an ideology score?*

---

This volume covers validation — the practice of testing whether our statistical models produce trustworthy results. It moves from the conceptual (what does it mean for an ideology score to be "correct"?) through internal checks (does the model predict what actually happened?) to external benchmarks (do our numbers agree with the gold standard in political science?).

Every model is a simplification of reality. The question isn't whether it's perfect — it's whether the simplifications are acceptable. This volume explains how we test that, and how the pipeline automatically flags results that shouldn't be trusted.

---

## Chapters

| # | Title | What You'll Learn |
|---|-------|-------------------|
| 1 | [What Does Validation Mean?](ch01-what-does-validation-mean.md) | Why ideology scores can't be "right" in the way a thermometer reading can — and what we test instead |
| 2 | [Posterior Predictive Checks: Can the Model Predict Its Own Data?](ch02-posterior-predictive-checks.md) | How we ask the model to regenerate the votes it was fit to, and what a mismatch tells us |
| 3 | [The Gold Standard: Shor-McCarty External Validation](ch03-shor-mccarty.md) | Comparing our Kansas scores to the political science benchmark — and why a 0.95 correlation is both expected and meaningful |
| 4 | [W-NOMINATE: Comparing to the Field Standard](ch04-w-nominate.md) | How the dominant method in political science works, why it's different from our approach, and what we learn from the comparison |
| 5 | [Quality Gates: Automatic Trust Levels](ch05-quality-gates.md) | The three-tier system that decides which model's estimates to publish, and the seven safety checks that catch axis confusion |

---

## Key Terms Introduced in This Volume

| Term | Definition |
|------|-----------|
| Validation | Testing whether a model's outputs are trustworthy — not whether the model is "true" |
| Posterior predictive check (PPC) | Asking the fitted model to generate fake data, then comparing it to the real data |
| Calibration | Whether the model's predicted probabilities match the actual frequencies |
| Classification accuracy | How often the model's most-likely prediction matches what actually happened |
| GMP (Geometric Mean Probability) | A fit metric that harshly penalizes confident wrong predictions — like a teacher who grades harder on the questions you said you were sure about |
| APRE (Aggregate Proportional Reduction in Error) | How much better the model does compared to just guessing the most common outcome every time |
| Bayesian p-value | The fraction of simulated datasets where a summary statistic is as extreme as the observed one — not the same as a classical p-value |
| LOO-CV (Leave-One-Out Cross-Validation) | Estimating how well the model predicts votes it hasn't seen, without actually refitting |
| PSIS (Pareto Smoothed Importance Sampling) | The mathematical trick that makes LOO-CV possible without refitting the model for each left-out observation |
| Pareto k diagnostic | A number that flags observations where LOO-CV's approximation is unreliable |
| Yen's Q3 | A test for local dependence — whether pairs of bills are correlated beyond what a single ideology dimension explains |
| Shor-McCarty scores | The gold-standard ideology estimates for state legislators, created by bridging roll call votes to a common national survey |
| W-NOMINATE | The dominant method in political science for estimating legislator ideal points from roll call votes |
| Optimal Classification (OC) | A nonparametric alternative that finds ideal points by minimizing classification errors without assuming a probability distribution |
| Quality gate | An automated check that determines how much to trust a model's output |
| Tier system | The three-level trust hierarchy: Tier 1 (fully converged, full trust), Tier 2 (point estimates credible, caution on uncertainty), Tier 3 (failed, fall back to simpler model) |
| Party separation (Cohen's d) | The standardized distance between Republican and Democrat average ideal points — a check that the model actually captured ideology, not something else |
| Horseshoe detection | The automated system that identifies sessions where 1D ideal points confuse ideology with establishment loyalty |

---

*Previous: [Volume 4 — Measuring Ideology](../volume-04-measuring-ideology/)*

*Next: [Volume 6 — Finding Patterns](../volume-06-finding-patterns/)*
