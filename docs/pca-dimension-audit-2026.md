# PCA Dimension Audit: Which Component Is Ideology?

**Date:** 2026-03-26

---

## The Problem

PCA extracts components in order of variance explained, not substantive meaning. In a supermajority chamber where intra-party factionalism generates more variance than the party divide, PC1 can capture the factional axis instead of ideology. This happened in the Kansas Senate for much of the 1999-2012 period.

The pipeline uses a manual override file (`analysis/pca_overrides.yaml`) to tell downstream phases which PC to treat as ideology. This audit re-evaluated all 28 chamber-sessions against the current pipeline configuration.

## What Changed

The `CONTESTED_THRESHOLD` in `analysis/tuning.py` was raised from 0.025 (2.5%) to 0.10 (10%). This removes lopsided votes — the very votes where intra-Republican factional defections are most visible. With fewer lopsided votes in the PCA input matrix, the factional axis loses variance share, and the party divide reclaims PC1 in several previously affected sessions.

The previous overrides were calibrated to the 2.5% threshold and became stale when the threshold changed.

## Methodology

For each of the 14 bienniums (78th-91st), both chambers, we computed:

- **Cohen's d**: Party separation effect size on each PC. Values above 2.0 indicate clear party separation.
- **Point-biserial correlation (rpb)**: Correlation between party (R=1, D=0) and PC scores. The automated `detect_ideology_pc()` function uses this metric.
- **Eigenvalue ratio (λ₁/λ₂)**: How dominant PC1 is over PC2. Ratios below 2.0 indicate the two components capture similar variance and are prone to swapping.

All values computed from the latest pipeline runs using `CONTESTED_THRESHOLD = 0.10`.

## Full Audit Table: Senate

| Session | R% | PC1 d | PC2 d | PC1 rpb | PC2 rpb | Ideology on | Override? |
|---------|-----|-------|-------|---------|---------|-------------|-----------|
| 78th (1999-2000) | 68% | **6.39** | 0.83 | 0.934 | -0.323 | PC1 | No |
| 79th (2001-2002) | 75% | 1.30 | **4.07** | 0.448 | -0.840 | PC2 | **Yes** |
| 80th (2003-2004) | 72% | 0.40 | **3.96** | 0.155 | 0.880 | PC2 | **Yes** |
| 81st (2005-2006) | 75% | **2.44** | 2.18 | 0.668 | 0.640 | PC1 | No |
| 82nd (2007-2008) | 75% | 1.35 | **2.41** | 0.467 | -0.692 | PC2 | **Yes** |
| 83rd (2009-2010) | 78% | 1.26 | **4.53** | 0.400 | 0.856 | PC2 | **Yes** |
| 84th (2011-2012) | 81% | **1.89** | 1.61 | 0.503 | 0.459 | PC1 (weak) | No |
| 85th (2013-2014) | 79% | **6.74** | 0.57 | 0.909 | 0.185 | PC1 | No |
| 86th (2015-2016) | 80% | **5.79** | 0.34 | 0.883 | 0.108 | PC1 | No |
| 87th (2017-2018) | 71% | **2.47** | 1.90 | 0.681 | 0.559 | PC1 | No |
| 88th (2019-2020) | 73% | **4.87** | 1.09 | 0.877 | 0.375 | PC1 | No |
| 89th (2021-2022) | 70% | **8.46** | 0.47 | 0.901 | 0.232 | PC1 | No |
| 90th (2023-2024) | 72% | **7.44** | 0.25 | 0.945 | 0.095 | PC1 | No |
| 91st (2025-2026) | 76% | **10.00** | 0.24 | 0.978 | 0.086 | PC1 | No |

## Full Audit Table: House

All 14 House sessions have PC1 as the clear ideology axis. No overrides needed.

| Session | PC1 d | PC2 d | Ideology on |
|---------|-------|-------|-------------|
| 78th (1999-2000) | **5.12** | 0.53 | PC1 |
| 79th (2001-2002) | **5.42** | 0.21 | PC1 |
| 80th (2003-2004) | **3.76** | 0.89 | PC1 |
| 81st (2005-2006) | **4.06** | 0.87 | PC1 |
| 82nd (2007-2008) | **4.03** | 0.83 | PC1 |
| 83rd (2009-2010) | **3.57** | 0.58 | PC1 |
| 84th (2011-2012) | **5.56** | 0.49 | PC1 |
| 85th (2013-2014) | **5.00** | 0.71 | PC1 |
| 86th (2015-2016) | **3.86** | 1.03 | PC1 |
| 87th (2017-2018) | **3.68** | 1.07 | PC1 |
| 88th (2019-2020) | **8.78** | 0.38 | PC1 |
| 89th (2021-2022) | **8.45** | 0.19 | PC1 |
| 90th (2023-2024) | **7.27** | 0.07 | PC1 |
| 91st (2025-2026) | **10.86** | 0.13 | PC1 |

## Changes Made

Four overrides removed, four retained. Previous count: 8. New count: 4.

### Removed

| Session | Old Override | Why Removed |
|---------|-------------|-------------|
| 78th Senate | PC2 | PC1 d=6.39 at 0.10 threshold. Filtering removed the lopsided factional votes that put ideology on PC2. |
| 81st Senate | PC2 | PC1 d=2.44 > PC2 d=2.18. Borderline at old threshold, PC1 now wins on both Cohen's d and rpb. |
| 84th Senate | PC2 | PC1 d=1.89 > PC2 d=1.61. Previous rationale cited W-NOMINATE validation, but W-NOMINATE shares the same variance-ordering vulnerability (ADR-0127). PC1 has a weak edge. |
| 88th Senate | PC2 | PC1 d=4.87 at 0.10 threshold. With only 71 contested votes, removing lopsided votes eliminated the factional signal entirely. |

### Retained

| Session | Override | Why Retained |
|---------|----------|-------------|
| 79th Senate | PC2 | PC2 d=4.07 >> PC1 d=1.30. The deepest factional split in the dataset — even at 0.10 threshold, intra-R factionalism dominates. This was the peak of the Kansas Republican civil war. |
| 80th Senate | PC2 | PC2 d=3.96 >> PC1 d=0.40. Near-zero party signal on PC1 even after filtering. |
| 82nd Senate | PC2 | PC2 d=2.41 > PC1 d=1.35. Pre-purge factional structure still visible in contested votes. |
| 83rd Senate | PC2 | PC2 d=4.53 >> PC1 d=1.26. Last full session before the 2012 primary purge. Brownback actively organizing primary challenges. |

## The Kansas Story

The axis instability is not a statistical artifact — it reflects genuine political structure. The Kansas Senate experienced a fifteen-year factional war between moderate and conservative Republicans that generated more variance in roll-call voting than the party divide itself.

### Three-Party Politics (1999-2012)

From the late 1990s through 2012, the Kansas Senate functioned as a three-party body: conservative Republicans, moderate Republicans, and Democrats. Moderate Republicans like Senate President Steve Morris (R-Hugoton, 2005-2013), Sandy Praeger, Jean Schodorf, and Jay Emler frequently formed legislative coalitions with Democrats on education funding, healthcare, and fiscal policy. Conservative insurgents like Tim Huelskamp (R-District 38), Dennis Pyle (R-District 1), and their allies defected from the party position on different votes than moderates did.

This produced a voting matrix where the largest source of disagreement was *within* the Republican caucus. With Republicans holding 68-78% of seats, the intra-R factional axis dominated total variance, and PCA correctly ranked it as PC1. The party divide — though substantively important — explained less variance because there were fewer Democrats contributing to it.

In the 79th Senate (2001-2002), the most extreme case, Huelskamp loads at one extreme of PC1 and Praeger at the other. The ten Democrats are scattered through the middle of this axis. PC2, not PC1, is where Democrats and Republicans cleanly separate.

### The 2012 Purge

Governor Sam Brownback, frustrated that moderate Senate Republicans blocked his tax-cut agenda, orchestrated a coordinated primary campaign against moderate incumbents in August 2012. Koch-backed Americans for Prosperity, the Kansas Chamber of Commerce PAC, and Kansans for Life funded conservative challengers. Eight moderate incumbents lost, including Senate President Steve Morris (4,737 to 5,106). Susan Wagle replaced Morris as Senate President.

The purge eliminated the moderate Republican faction from the Kansas Senate nearly overnight. From the 85th Legislature (2013-2014) onward, the remaining Republican caucus was ideologically homogeneous. Intra-R variance collapsed, and the party divide re-emerged as the dominant axis.

### Why the 10% Threshold Matters

The `CONTESTED_THRESHOLD = 0.10` filter removes votes where fewer than 10% of members dissented. These lopsided votes are precisely where factional defections are most visible — a vote passing 35-5 in a 40-member Senate typically features five conservative Republicans voting against a bipartisan majority. These votes carry enormous weight in PCA because they create large individual loadings on the factional axis.

Removing them doesn't eliminate the factional signal entirely (the 79th and 83rd still need PC2 overrides), but it weakens it enough that the party divide reclaims PC1 in sessions where the factional-vs-partisan variance was close (78th, 81st, 88th).

### Session-by-Session Political Context

| Session | Political Context | PCA Result |
|---------|------------------|------------|
| 78th (1999-2000) | Moderate R Senate under Dave Kerr. Three-party era begins. | PC1 at 0.10 threshold |
| 79th (2001-2002) | Peak of R civil war. Huelskamp vs. Praeger. | **PC2** — strongest swap |
| 80th (2003-2004) | Continued factional warfare. | **PC2** |
| 81st (2005-2006) | Morris becomes President. Moderates consolidate. | PC1 (borderline) |
| 82nd (2007-2008) | Pre-purge tension building. KTRM PAC active. | **PC2** |
| 83rd (2009-2010) | Brownback elected governor. Last full pre-purge session. | **PC2** — very strong |
| 84th (2011-2012) | Purge in August 2012. Transitional session. | PC1 (ambiguous) |
| 85th (2013-2014) | Post-purge. Homogeneous R caucus. Brownback tax cuts. | PC1 |
| 86th (2015-2016) | Moderate backlash begins. 2016 primaries unseat 8 conservatives. | PC1 |
| 87th (2017-2018) | Tax cut repeal (veto override). Brownback resigns. | PC1 |
| 88th (2019-2020) | Laura Kelly (D) governor. Partial moderate resurgence. | PC1 at 0.10 threshold |
| 89th-91st | Highly polarized. Party divide dominant. | PC1 |

## Maintenance Notes

- Override file: `analysis/pca_overrides.yaml`
- Contested threshold: `CONTESTED_THRESHOLD` in `analysis/tuning.py`
- Automated fallback: `detect_ideology_pc()` in `analysis/init_strategy.py`
- **If `CONTESTED_THRESHOLD` changes, re-audit all sessions.** The threshold is the single most important parameter affecting PCA axis ordering. A lower threshold brings more lopsided votes into the matrix, strengthening the factional axis. A higher threshold removes them, strengthening the party axis.
- New bienniums (92nd onward) are unlikely to need overrides. The modern Kansas Legislature is highly polarized (91st PC1 d = 10.00), and the factional era ended in 2012.

## References

- ADR-0118: Party separation quality gates across pipeline
- ADR-0127: W-NOMINATE variance-ordering vulnerability
- `docs/pca-ideology-axis-instability.md`: Original deep dive (values at 0.025 threshold)
- `docs/pca-rotation-and-human-intervention.md`: Why rotation methods can't solve this
- Poole, K. T., & Rosenthal, H. (2007). *Ideology and Congress* (2nd ed.).
- de Leeuw, J. (2018). PCA dimensions have no inherent substantive interpretation.
- FiveThirtyEight (2012). "The End of a Kansas Tradition: Moderation."
- NPR (2012). "Conservatives Win in Kansas GOP Senate Primary."
