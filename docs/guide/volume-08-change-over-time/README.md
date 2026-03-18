# Volume 8 — Change Over Time

> *The Kansas Legislature isn't standing still. Is it more polarized than it was in 2011? When did the shift happen? And did the party move because its members changed their minds — or because moderates were replaced by ideologues?*

---

Volumes 1-7 analyzed the Kansas Legislature one biennium at a time. Each session was a self-contained snapshot: estimate ideology, validate it, find coalitions, predict votes, read the text. But snapshots don't tell you about the movie. A legislature that looks identical in 2011 and 2025 might have gone through dramatic swings in between. A legislature that looks very different might have changed gradually — or all at once.

This volume adds the dimension of time.

Chapter 1 watches the legislature *within* a single session, tracking party cohesion week by week and detecting the moments when the pattern abruptly shifts. Chapter 2 tracks individual legislators *across* sessions, estimating how their ideology drifts over 15 years. Chapter 3 solves the technical problem that makes cross-session comparison possible: putting different sessions on the same measurement scale. Chapter 4 decomposes the *why* behind ideological change — separating the effect of existing members changing their minds from the effect of new members replacing departing ones. Chapter 5 weaves the numbers into a narrative: what fifteen years of data reveal about Kansas politics.

The central question isn't "what does the legislature look like?" — that's what Volumes 1-7 answered. The question is "how did it get here?"

---

## Chapters

| # | Title | What You'll Learn |
|---|-------|-------------------|
| 1 | [Time Series: Detecting When the Legislature Shifts](ch01-time-series.md) | How rolling PCA and the Rice Index track ideology and cohesion through a session, and how PELT changepoint detection finds the moments when the pattern breaks |
| 2 | [Dynamic Ideal Points: The Martin-Quinn Model](ch02-dynamic-ideal-points.md) | A state-space model where each legislator's ideology today is yesterday's plus a random step — tracking 15 years of ideological drift with bridge legislators anchoring the scale |
| 3 | [Cross-Session Alignment: Putting Sessions on the Same Scale](ch03-cross-session-alignment.md) | How affine transformation stretches and shifts one session's IRT scores to match another's, using returning legislators as anchors and robust fitting to handle genuine movers |
| 4 | [Conversion vs. Replacement: Why Does Ideology Change?](ch04-conversion-vs-replacement.md) | Decomposing a party's ideological shift into the effect of returning members changing their minds (conversion) and new members replacing departing ones (replacement) |
| 5 | [Fifteen Years of Kansas Politics (2011-2026)](ch05-fifteen-years.md) | A data-driven narrative of three political eras — the Brownback experiment, the moderate resurgence, and the current polarization — told through the numbers |

---

## Key Terms Introduced in This Volume

| Term | Definition |
|------|-----------|
| Rolling-window PCA | PCA applied to overlapping subsets of votes within a session, producing a time series of ideological positions |
| Rice Index (time series) | Party cohesion measured per vote and aggregated weekly — tracking how unified each party is over the course of a session |
| Desposato correction | A Monte Carlo adjustment that removes the small-group inflation bias from the Rice Index |
| PELT (Pruned Exact Linear Time) | A changepoint detection algorithm that finds the moments when a time series abruptly shifts |
| Changepoint | A point in time where the statistical properties of a series (mean, variance, or both) change suddenly |
| CROPS (Changepoints for a Range of Penalties) | An automated method that scans a range of sensitivity settings and identifies the natural number of changepoints |
| Bai-Perron test | A structural break test that provides formal confidence intervals on when a break occurred |
| Dynamic ideal point | An ideology score that varies over time, estimated via a state-space model |
| State-space model | A model where a hidden state (ideology) evolves over time according to a transition equation and is observed indirectly through data (votes) |
| Random walk | A process where each step is the previous position plus a random increment — the simplest model of drift |
| Evolution variance (tau) | The parameter controlling how fast ideology can change between bienniums — small tau means stability, large tau means volatility |
| Bridge legislator | A lawmaker who serves in multiple bienniums, linking the measurement scales across sessions |
| Non-centered parameterization | A reparameterization trick that helps MCMC samplers explore the posterior efficiently in hierarchical models |
| Affine transformation | A "stretch and shift" operation (y = Ax + B) that maps one session's ideal point scale onto another's |
| Robust fitting | Regression that trims extreme outliers before fitting, preventing genuine movers from distorting the scale alignment |
| Conversion effect | The component of ideological change attributable to returning legislators changing their positions |
| Replacement effect | The component of ideological change attributable to new legislators replacing departing ones |
| KS test (Kolmogorov-Smirnov) | A non-parametric test for whether two distributions differ significantly |
| Population Stability Index (PSI) | A metric for whether the distribution of a variable has shifted meaningfully between two time periods |
| Intraclass correlation (ICC) | A measure of how consistently a metric ranks legislators across two sessions |

---

*Previous: [Volume 7 — Prediction and Text](../volume-07-prediction-and-text/)*

*Next: [Volume 9 — Telling the Story](../volume-09-telling-the-story/)*
