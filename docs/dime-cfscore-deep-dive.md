# DIME/CFscores Deep Dive

## Overview

This document surveys the Database on Ideology, Money in Politics, and Elections (DIME) as a second external validation source for Tallgrass IRT ideal points. Where Phase 14 validates against Shor-McCarty (roll-call-based ideology), Phase 18 validates against CFscores (campaign-finance-based ideology) — a completely independent construct. The combination provides the strongest possible external validation story: our scores agree with both how legislators *vote* and who *funds* them.

## What CFscores Measure

CFscores estimate candidate ideology from campaign finance records using correspondence analysis (SVD) on the donor-recipient matrix. The key insight is the **donor bridge mechanism**: donors who give to multiple candidates create links between those candidates. If the same donors fund two candidates, those candidates are ideologically similar. The method scales to the full universe of federal and state candidates because FEC and state-level disclosure records provide comprehensive contribution data.

The current database (Version 4.0, December 2024) contains over 850 million itemized contributions, with ideal point estimates for 173,171 candidates, 42,702 committees, 41.5 million individual donors, and 3.3 million organizational donors. V4 improvements include LLM-enhanced candidate-donation matching, improved record linkage for candidates across offices, and new composite ideological scores combining campaign contributions with roll-call votes (see `docs/DIME/dime_codebook_v4.pdf`).

Three flavors of ideological score are available in the recipient database:

| Type | Column | Description |
|------|--------|-------------|
| **Static** (career) | `recipient.cfscore` | Single score across all election cycles. Analogous to Shor-McCarty's career-fixed `np_score`. Primary comparison target. |
| **Dynamic** (per-cycle) | `recipient.cfscore.dyn` | Per-cycle score re-estimated each election cycle while holding contributor scores constant. Captures ideological drift. Secondary comparison. |
| **DW-DIME** | `dwdime` | Supervised ML prediction of roll-call scores from campaign contributions (Bonica 2018). Available for federal candidates. Not used for Phase 18 (state legislators lack roll-call-predicted scores). |

The static CFscore is the more widely used and validated measure. It is estimated from the full donor-recipient matrix across all cycles. The dynamic CFscore provides temporal resolution but may be noisier for candidates with few donors in a given cycle. V4 also adds a `composite.score` combining multiple sources, but we use the raw CFscore for cleaner construct separation.

Bonica's validation compendium (`docs/DIME/dime_validation.pdf`) catalogs three categories of validation: (1) **predictive validity** — vote classification accuracy for 96th-112th Congresses, (2) **external validity** — comparison against DW-NOMINATE, Judicial Common-Space scores, and Turbo-ADA, and (3) **internal validity** — consistency between recipient and contributor scores, donor ideological stability across cycles and geography. A fourth category tests robustness to strategic giving.

## Literature Survey

### Bonica 2014 — The Foundational Paper

Bonica, A. 2014. "Mapping the Ideological Marketplace." *American Journal of Political Science* 58(2): 367-386.

Introduces CFscores and demonstrates their validity against DW-NOMINATE for Congress (r ≈ 0.90) and against Shor-McCarty for state legislatures (r ≈ 0.85). Shows that campaign finance ideology is a distinct but correlated construct from roll-call ideology. The key contribution is coverage: CFscores are available for any candidate who raises money, including challengers, primary losers, and candidates who never take office.

### Bonica 2018 — Supervised Machine Learning Extension

Bonica, A. 2018. "Inferring Roll-Call Scores from Campaign Contributions Using Supervised Machine Learning." *American Journal of Political Science* 62(4): 830-848.

Uses supervised ML to predict roll-call ideology from campaign contributions. Achieves cross-validated r ≈ 0.93 for Congress. Key finding: campaign finance predicts roll-call behavior well overall, but within-party prediction is weaker (r ≈ 0.65-0.70 for Democrats, r ≈ 0.60-0.65 for Republicans at the state level).

### Hill & Huber 2019 — Validation and Limitations

Hill, S.J. and G.A. Huber. 2019. "On the Meaning of Campaign Contributions: New Theory and Evidence." *Journal of Politics* 81(2): 755-762.

Raises important caveats about what campaign contributions actually measure. Argues that contributions reflect a mix of ideology and access-seeking behavior. For state legislatures, access-motivated giving (PACs, business interests) is a larger share of total contributions, which can compress the ideological signal. Implication for our analysis: CFscores may understate within-party ideological differences because access-motivated donors give to both moderates and extremists within the majority party.

### Bonica & Tausanovitch 2022 — Democratic Divergence

Bonica, A. and C. Tausanovitch. 2022. "The Ideological Nationalization of Congressional Campaigns." *Legislative Studies Quarterly* 47(4): 921-953.

Documents a divergence between CFscores and roll-call scores for Democrats starting around 2010. Democratic candidates' donor networks shifted left faster than their voting records, partly because of nationalized small-dollar fundraising. Implication: CFscores may show Democrats as more liberal than their roll-call behavior suggests. This asymmetry should be visible in our scatter plots as a leftward shift for Democrats relative to the regression line.

### Handan-Nader, Myers, & Hall 2022 — State Legislature Limitations

Handan-Nader, C., A. Myers, and A.B. Hall. 2022. "What Do Campaign Contributions Buy? Evidence from State Legislature Roll-Call Votes." Working paper.

Finds that the CFscore-to-roll-call correlation at the state level is lower than at the federal level (r ≈ 0.75-0.85 vs r ≈ 0.90 for Congress). State legislative campaigns have fewer donors, more PAC-heavy funding, and less nationalized donor networks. Kansas is a conservative state with strong party organizations, so PAC contributions may dominate individual giving, potentially weakening the CFscore signal.

## CFscores vs Shor-McCarty Comparison

| Dimension | Shor-McCarty (`np_score`) | CFscore (`recipient.cfscore`) |
|-----------|--------------------------|------------------------------|
| **Construct** | Roll-call voting ideology | Campaign finance ideology |
| **Method** | Bridge-legislator IRT (state-Congress bridge) | Correspondence analysis on donor-recipient matrix |
| **Kansas coverage** | 84th-88th (2011-2020) | 84th-89th (2011-2022) |
| **Temporal resolution** | Career-fixed (one score) | Static (career) + dynamic (per-cycle) |
| **Within-party discrimination** | Good (based on actual votes) | Limited (r ≈ 0.60-0.70 within party) |
| **Coverage breadth** | Legislators who cast votes | Legislators who raise money |
| **Data license** | CC0 (Harvard Dataverse) | ODC-BY (Stanford) |
| **Our expected overall r** | 0.90-0.95 (established) | 0.75-0.90 (literature range) |

## DIME Data Schema

**File:** `data/external/dime_recipients_1979_2024.csv` (144 MB, uncompressed)

The DIME dataset contains 56 columns. Key columns for Phase 18:

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `recipient.cfscore` | float | `-0.67`, `1.08` | Static (career) CFscore — primary comparison |
| `recipient.cfscore.dyn` | float | `-0.783`, `1.068` | Dynamic (per-cycle) CFscore — secondary |
| `lname` | str | `hodge` | Lowercase last name |
| `fname` | str | `timothy` | Lowercase first name |
| `name` | str | `hodge, timothy c tim` | Full name, "last, first middle" |
| `party` | int | `100`=D, `200`=R, `328`=Lib, `500`=Ind | ICPSR party codes (codebook abbreviates 328 as "Ind" but ICPSR 328 = Libertarian; verified from KS data) |
| `state` | str | `KS` | Two-letter abbreviation |
| `seat` | str | `state:lower`, `state:upper` | Office type |
| `district` | str | `KS-72` | State-prefixed district |
| `cycle` | int | `2020` | Election cycle (even year) |
| `ico.status` | str | `I`, `C`, `O` | Incumbent/challenger/open |
| `num.givers` | int | `202` | Unique donor count |
| `bonica.rid` | str | `cand965983` | Unique recipient ID |

## Kansas Coverage Analysis

Verified directly from the DIME CSV:

- **4,500 KS state legislature records** total (1994-2022)
- **100% CFscore coverage** — every KS state leg row has a valid `recipient.cfscore`
- **Last cycle with KS state leg data: 2022** (despite the file name claiming 1979-2024)
- Party distribution: 100=Democrat (828 records), 200=Republican (1423), 328=Libertarian (33), 500=Independent (4)

### Biennium-to-Cycle Mapping

DIME records are indexed by election cycle (even years). Kansas legislators elected in cycle 2020 serve in the 89th biennium (2021-2022). The mapping:

| Biennium | Cycles | SM Coverage | DIME Coverage |
|----------|--------|-------------|---------------|
| 84th (2011-12) | 2010, 2012 | Yes | Yes |
| 85th (2013-14) | 2012, 2014 | Yes | Yes |
| 86th (2015-16) | 2014, 2016 | Yes | Yes |
| 87th (2017-18) | 2016, 2018 | Yes | Yes |
| 88th (2019-20) | 2018, 2020 | Yes | Yes |
| 89th (2021-22) | 2020, 2022 | **No** | Yes |
| 90th (2023-24) | 2022 only | **No** | Partial (stale) |
| 91st (2025-26) | 2022 only | **No** | Partial (stale) |

**Decision:** Validate 84th-89th (6 bienniums). The 89th is the key gain — one extra biennium beyond Shor-McCarty. Skip 90th-91st because using 2022 cycle CFscores for 2023-2026 legislators is too stale to be meaningful validation.

## Methodological Caveats

1. **Within-party discrimination is limited.** The literature consistently finds that CFscores discriminate well between parties (r ≈ 0.85-0.90 overall) but poorly within parties (r ≈ 0.60-0.70). This means our intra-party correlations will likely be lower than with Shor-McCarty — that is expected and not a failure of our IRT scores.

2. **Access-motivated donations compress the signal.** Kansas Republican legislators in safe districts receive significant PAC contributions regardless of ideology. This "access" money pulls CFscores toward the center for both moderate and conservative Republicans, reducing within-party spread.

3. **Democratic divergence (Bonica & Tausanovitch 2022).** Democrats' donor networks have nationalized and shifted left since 2010. Kansas Democrats' CFscores may overstate their liberalism relative to their actual voting behavior. Watch for systematic leftward bias in the Democratic scatter plot points.

4. **Donor count threshold.** Candidates with very few donors (< 5) have unreliable CFscores. The DIME dataset includes these candidates, but their scores are essentially noise. A minimum donor filter (`num.givers >= 5`) is applied during matching.

5. **Incumbent-only matching.** We match against incumbent records (`ico.status == "I"`) because our IRT ideal points are for legislators who actually cast votes. Challengers and open-seat candidates have CFscores but no roll-call record to compare against.

6. **Static vs dynamic CFscore interpretation.** The static CFscore is the career average — directly comparable to Shor-McCarty's career-fixed `np_score`. The dynamic CFscore varies by cycle, which is conceptually closer to our session-specific IRT score, but is noisier for state legislators with few donors in any given cycle.

## Recommendations

1. **Triangulation design.** Report both SM and DIME correlations side-by-side for bienniums 84th-88th (where both are available). This triangulation — roll-call ideology agrees with campaign-finance ideology agrees with our IRT scores — is the strongest validation argument.

2. **Both static and dynamic CFscores.** Report correlations against both `recipient.cfscore` (static) and `recipient.cfscore.dyn` (dynamic). The static score is the primary comparison target; the dynamic score provides a supplementary check on temporal alignment.

3. **Minimum donor filter.** Apply `num.givers >= 5` to exclude candidates with unreliable CFscores. This is conservative — the DIME documentation suggests a threshold of 10 for reliable scores, but Kansas state legislators often have fewer donors than federal candidates.

4. **ODC-BY attribution.** The DIME dataset is released under the Open Data Commons Attribution License (ODC-BY). All reports and publications must credit: Bonica, Adam. 2024. "Database on Ideology, Money in Politics, and Elections (DIME)." Stanford University Libraries. https://data.stanford.edu/dime.

## Local Reference Documents

- `docs/DIME/dime_codebook_v4.pdf` — Version 4.0 codebook (December 2024): all 63 variable descriptions, data sources, seat labels, party codes, scaling algorithm improvements, composite scores
- `docs/DIME/dime_validation.pdf` — Compendium of validation results: predictive validity (vote classification), external validity (DW-NOMINATE, JCS), internal validity (donor stability), robustness to strategic giving

## References

- Bonica, A. 2014. "Mapping the Ideological Marketplace." *American Journal of Political Science* 58(2): 367-386.
- Bonica, A. 2018. "Inferring Roll-Call Scores from Campaign Contributions Using Supervised Machine Learning." *AJPS* 62(4): 830-848.
- Bonica, A. 2024. "Database on Ideology, Money in Politics, and Elections (DIME)." Version 4.0. Stanford University Libraries. https://data.stanford.edu/dime.
- Bonica, A. and M. Sen. 2017. "A Common-Space Scaling of the American Judiciary and Legal Profession." *Political Analysis* 25(1): 114-121.
- Bonica, A. and C. Tausanovitch. 2022. "The Ideological Nationalization of Congressional Campaigns." *Legislative Studies Quarterly* 47(4): 921-953.
- Handan-Nader, C., A. Myers, and A.B. Hall. 2022. "What Do Campaign Contributions Buy? Evidence from State Legislature Roll-Call Votes." Working paper.
- Hill, S.J. and G.A. Huber. 2019. "On the Meaning of Campaign Contributions: New Theory and Evidence." *Journal of Politics* 81(2): 755-762.
