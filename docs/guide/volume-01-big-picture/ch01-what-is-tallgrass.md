# Chapter 1: What Is Tallgrass?

> *"What if you could read the political DNA of every Kansas legislator — not from their campaign speeches, but from every vote they've ever cast?"*

---

## The Elevator Pitch

Tallgrass is an open-source software project that collects every recorded vote from the Kansas Legislature's website, then runs that data through a 28-step statistical analysis pipeline. The result is a detailed, quantitative portrait of each legislator's ideology, each party's cohesion, and the legislature's overall polarization — all derived from the public record of who voted Yea and who voted Nay.

The name comes from the tallgrass prairie ecosystem native to Kansas — vast, deeply rooted, and more complex beneath the surface than it appears.

## What It Produces

At its core, Tallgrass answers a deceptively simple question: *where does each legislator fall on the political spectrum?*

The answer turns out to be anything but simple. A single "ideology score" requires navigating a thicket of statistical problems: How do you weight a unanimous vote versus a nail-biter? How do you handle a legislator who missed half the session? How do you even define "liberal" and "conservative" when the math itself can't tell left from right?

Tallgrass tackles these problems with a combination of techniques drawn from political science, psychometrics (the science of measuring mental traits), and Bayesian statistics (a framework for reasoning under uncertainty). The pipeline produces:

- **Ideal points** — A number for each legislator placing them on a liberal-to-conservative scale. Think of it as a political GPS coordinate: negative values are more liberal, positive values are more conservative, and the distance between two legislators tells you how differently they tend to vote.

- **Polarization measures** — How far apart are the two parties? Is the gap growing or shrinking? Are there factions within each party, or is ideology a smooth continuum?

- **Coalition maps** — Who votes with whom? Are there bridge-builders who connect the two parties? Are there mavericks who break from their own?

- **Per-legislator profiles** — A deep dive on any individual lawmaker: their party loyalty score, their most surprising votes, their voting neighbors, and how they compare to their caucus.

- **Temporal trajectories** — How has a legislator's ideology shifted over their career? Has the legislature as a whole drifted left or right since 2011?

- **Automated narrative reports** — HTML documents that weave all of the above into a readable story about each legislative session, complete with interactive charts and data tables.

## The Scale of the Data

Tallgrass currently holds data from 14 bienniums of the Kansas Legislature, spanning 1999 through 2026 (the 78th through 91st Legislatures). Across all sessions, the dataset contains:

| Metric | Count |
|--------|-------|
| Individual vote records | 1,567,983 |
| Roll call votes | ~14,000 |
| Legislators tracked | ~2,000 unique individuals |
| Years of coverage | 27 (1999-2026) |
| Bienniums analyzed | 14 (78th-91st) |
| Special sessions | 5 (2013, 2016, 2020, 2021, 2024) |

To put that in perspective: if you printed every individual vote record on its own line, the stack of paper would be about 500 feet tall. And every one of those records is a moment where a real human being — an elected representative of Kansas citizens — pressed a button and went on the record.

## What It's Built With

Tallgrass is a Python project, open-source under the MIT license, and available on GitHub. It doesn't require special hardware — the full pipeline runs on a laptop in about 45 minutes per biennium. Here's what's under the hood:

- **Web scraping** — Automated, respectful collection of vote data from kslegislature.gov. The scraper identifies itself, limits its request rate, and caches every page it downloads.

- **Bayesian inference** — The statistical engine uses PyMC (a probabilistic programming library) and nutpie (a fast sampler written in Rust) to estimate ideal points. Instead of producing a single "best guess," Bayesian methods produce a *distribution* — an honest accounting of what the data supports and where uncertainty remains.

- **Dimensionality reduction** — Techniques like Principal Component Analysis (PCA) and UMAP compress the high-dimensional vote matrix into something a human brain can visualize.

- **Machine learning** — Gradient-boosted decision trees (XGBoost) predict individual votes and bill passage outcomes. Natural language processing (BERTopic) identifies policy topics from bill text.

- **Automated reporting** — Every phase generates a self-contained HTML report with plots, tables, and narrative text. No manual chart-making required.

If you're not a programmer, don't worry. This guide will never ask you to read code. When we reference specific files or functions, it's so that technically inclined readers can find the implementation — but the concepts are explained entirely in plain language.

## Who It's For

Tallgrass was built at the intersection of several audiences:

**Researchers** who study state legislatures. The academic literature on legislative ideology is dominated by Congress — 535 members, extensively studied. State legislatures, with their 7,383 members across 50 states, are comparatively under-examined. Tallgrass brings research-grade methodology to one state's data, with an architecture designed to extend to others.

**Journalists** who cover Kansas politics. Campaign ads and floor speeches tell you what a legislator *says* they believe. Voting records tell you what they *do*. Tallgrass makes the voting record legible — not as a raw spreadsheet, but as a statistically grounded analysis that accounts for the fact that not all votes are equally informative.

**Civic organizations** that track legislative accountability. Tallgrass provides objective, reproducible measures. The same data in, the same numbers out, every time. No editorial judgment about which votes "count" — the math decides which votes are informative.

**Curious citizens** who want to understand their legislature. Every Kansan is represented by one House member and one Senator. Tallgrass can tell you where they stand, who they vote with, and how they compare to others in their party.

**Students and self-learners** who want to understand political methodology. This guide exists because the methods used in Tallgrass — IRT, hierarchical Bayesian models, MCMC sampling — are powerful but poorly explained outside of graduate seminars. We believe these ideas deserve a wider audience.

## What It Is Not

A few things Tallgrass explicitly does *not* do:

- **It does not grade legislators.** There is no "A" or "F" rating. Ideology scores are descriptive, not evaluative. Being at +1.5 on the conservative end is not "better" or "worse" than being at -1.2 on the liberal end. The scores tell you *where* someone stands, not *whether* they should stand there.

- **It does not predict elections.** Ideology scores are estimated from voting behavior in office, not from campaign positions or public opinion polls. A legislator with extreme ideal points may win or lose their next election — Tallgrass has nothing to say about it.

- **It does not assign intent.** When a legislator votes Nay on a bill, we record the Nay. We don't know if they voted Nay because they opposed the policy, because they were pressured by leadership, because they were trading favors on a different bill, or because they misread the board. The model works with revealed preferences. Actions, not motives.

- **It does not editorialize.** The narrative reports describe patterns in the data. They do not argue that polarization is good or bad, that mavericks are brave or disloyal, or that any particular policy outcome is desirable. The reader brings the value judgments. Tallgrass brings the evidence.

## How This Guide Is Organized

This guide is a nine-volume series. You're reading Volume 1, which gives you the big picture. The remaining volumes go progressively deeper:

| Volume | What You'll Learn |
|--------|------------------|
| **1. The Big Picture** | What Tallgrass is, how the Kansas Legislature works, what votes reveal |
| **2. Gathering the Data** | How we scrape votes from the web, ensure quality, and store the data |
| **3. Your First Look at the Votes** | How to visualize and summarize 125 legislators x 600 votes |
| **4. Measuring Ideology** | The mathematical core: how we assign each legislator a number on the political spectrum |
| **5. Checking Our Work** | How we validate the results against independent measurements |
| **6. Finding Patterns** | Clustering, networks, and classical political science metrics |
| **7. Prediction and Text** | Machine learning on votes and natural language processing on bills |
| **8. Change Over Time** | Tracking ideology across sessions and decomposing the sources of change |
| **9. Telling the Story** | How raw numbers become readable narrative reports |

Each volume is designed to be readable on its own, but they build on each other. If you read them in order, each one introduces the vocabulary and intuition that the next one needs.

## A Note on Honesty

One theme runs through every volume of this guide: *honesty about uncertainty*.

Statistical models are tools, not oracles. Every ideal point estimate comes with a credible interval — a range of plausible values, not a single definitive answer. Every pipeline phase includes quality gates that flag results the model is less confident about. When the model struggles — and it does, particularly in chambers where one party holds a supermajority — we explain what went wrong, why, and what we do about it.

This is not a limitation to apologize for. It's the whole point. A model that claims more certainty than the data supports is worse than useless — it's misleading. Tallgrass would rather give you a wide range and say "the truth is probably in here somewhere" than give you a precise number and be wrong.

---

## Key Takeaway

Tallgrass turns 1.5 million Kansas Legislature vote records into statistical measures of ideology, polarization, and coalition structure — using methods drawn from psychometrics, political science, and Bayesian statistics. It produces numbers, not judgments. And it's honest about what it knows and what it doesn't.

---

*Terms introduced: ideal point, polarization, base rate, Bayesian inference, credible interval, quality gate*

*Next: [A Quick Tour of the Kansas Legislature](ch02-kansas-legislature.md)*
