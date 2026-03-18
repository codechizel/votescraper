# Volume 3 — Your First Look at the Votes

> *What does a matrix of 125 legislators × 600 votes actually look like? How do we start making sense of it?*

---

Before any statistical model can estimate ideology, someone has to organize the raw data and look at it. This volume covers exploratory data analysis — the techniques that transform thousands of individual vote records into a structured picture of the legislature.

The journey starts with the **vote matrix**: a giant spreadsheet where each row is a legislator, each column is a vote, and each cell says Yea, Nay, or nothing at all. From there, we measure **agreement** between every pair of legislators (correcting for the fact that most votes are Yea), **compress** 600 columns down to two or three meaningful dimensions, and confront the **horseshoe problem** — an artifact that makes the map lie about who's liberal and who's conservative.

By the end of this volume, you'll understand what the data looks like before any models are applied, and why that first look is more subtle than it appears.

---

## Chapters

| # | Title | What You'll Learn |
|---|-------|-------------------|
| 1 | [The Vote Matrix: Ones, Zeros, and Missing Data](ch01-vote-matrix.md) | How raw votes become a structured matrix; why filtering out easy votes matters |
| 2 | [Who Agrees with Whom? (Cohen's Kappa)](ch02-cohens-kappa.md) | Measuring agreement while correcting for the 82% Yea base rate |
| 3 | [Compressing the Data: PCA Explained](ch03-pca-explained.md) | How PCA reduces 600 votes to one or two numbers per legislator |
| 4 | [Alternative Views: MCA and UMAP](ch04-mca-and-umap.md) | Categorical analysis and nonlinear maps — different lenses on the same data |
| 5 | [The Horseshoe Problem (When the Map Lies)](ch05-horseshoe-problem.md) | Why PCA fails in 7 of 14 Kansas Senate sessions, and how the pipeline detects it |

---

## Key Terms Introduced in This Volume

| Term | Definition |
|------|-----------|
| Vote matrix | A table with legislators as rows and roll calls as columns; cells are Yea (1), Nay (0), or missing |
| Contested vote | A roll call where at least 2.5% of voters were in the minority |
| Base rate | The overall frequency of Yea votes (~82% in Kansas) |
| Cohen's Kappa | A measure of agreement that corrects for chance, invented by Jacob Cohen in 1960 |
| PCA (Principal Component Analysis) | A technique that finds the directions of maximum variation in data |
| Eigenvalue | A number that measures how much variation a principal component captures |
| Scree plot | A chart of eigenvalues used to decide how many components matter |
| MCA (Multiple Correspondence Analysis) | PCA's cousin for categorical data, using chi-squared distance |
| Greenacre correction | A fix for MCA's tendency to understate how much it explains |
| UMAP | A nonlinear technique that maps high-dimensional data to a 2D picture |
| Horseshoe effect | A mathematical artifact where a straight spectrum bends into a curve |
| Cohen's d | The standardized difference between two group means, used to detect horseshoes |

---

*Previous: [Volume 2 — Gathering the Data](../volume-02-gathering-data/)*

*Next: [Volume 4 — Measuring Ideology](../volume-04-measuring-ideology/)*
