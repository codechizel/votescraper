# Chapter 4: When One Dimension Isn't Enough: 2D IRT

> *Sometimes the liberal-conservative axis doesn't capture everything. In the Kansas Senate, there's a second dimension: establishment loyalty versus independent contrarianism. This chapter extends IRT from one dimension to two.*

---

## Why One Dimension Falls Short

The 1D IRT model from Chapter 2 makes a strong assumption: all disagreement in the legislature reduces to a single liberal-conservative axis. Every vote that doesn't fit this axis — a moderate Republican who defects on an education bill, a conservative who breaks ranks on a procedural motion — is treated as random noise.

For balanced chambers like the Kansas House, this assumption holds up well. PC1 explains ~35% of the variation, and the 1D model converges cleanly with party separations above 5.0.

But for the Kansas Senate — with its recurring supermajorities and factional wars — one dimension isn't enough. The data is trying to tell us about a **second axis of disagreement** that doesn't line up with party. And when we force a one-dimensional model onto two-dimensional data, we get the convergence problems and contaminated axes we saw in Chapter 3.

### What the Second Dimension Captures

In the Kansas Senate, the second dimension typically separates **establishment loyalists** from **contrarian independents** — legislators who frequently buck their own party's leadership on procedural votes, institutional matters, and issues where party discipline is weak.

This isn't an abstract statistical artifact. It reflects real legislative dynamics:

- During the 78th-83rd Senates (1999-2010), the Republican caucus was split between moderate establishment members (who controlled committee assignments and floor scheduling) and conservative insurgents (who challenged leadership on taxes, education, and social policy). The insurgents didn't just disagree with Democrats — they disagreed with their own party's leadership.

- In the 88th Senate (2019-2020), a similar dynamic returned: some Republicans aligned with the Democratic governor on Medicaid expansion, putting them at odds with their own caucus.

These dynamics create a genuine second dimension of disagreement. A 1D model is forced to project this two-dimensional structure onto a single line, losing information. A 2D model captures both axes simultaneously.

## The 2D Equation

The extension from 1D to 2D is straightforward. Instead of one ideal point per legislator and one discrimination per bill, we have **two** of each.

**Plain English:**

> "The probability of a Yea vote now depends on two things about the legislator — their ideology *and* their establishment loyalty — and two things about the bill — how much it cares about ideology *and* how much it cares about establishment loyalty."

**Equation:**

```
P(Yea) = logistic(β₁·ξ₁ + β₂·ξ₂ − α)
```

where:
- ξ₁ = the legislator's position on Dimension 1 (ideology: liberal ↔ conservative)
- ξ₂ = the legislator's position on Dimension 2 (contrarian ↔ establishment)
- β₁ = how much this bill cares about Dimension 1 (ideological discrimination)
- β₂ = how much this bill cares about Dimension 2 (establishment discrimination)
- α = the bill's overall difficulty

**Worked example:**

Senator Garcia has an ideal point of ξ₁ = +0.6 (moderately conservative) and ξ₂ = −0.8 (contrarian — often bucks party leadership). A bill comes up with β₁ = 1.0 (moderately ideological), β₂ = 0.5 (slightly favors establishment), and α = 0.2 (slightly hard to pass).

1. **Dimension 1 contribution:** β₁ · ξ₁ = 1.0 × 0.6 = 0.60
2. **Dimension 2 contribution:** β₂ · ξ₂ = 0.5 × (−0.8) = −0.40
3. **Subtract difficulty:** 0.60 + (−0.40) − 0.2 = 0.00
4. **Apply logistic:** logistic(0.00) = 0.50
5. **Result: 50% probability of a Yea vote**

Notice what happened: Senator Garcia's conservatism pushes them toward Yea (+0.60), but their contrarianism pushes them toward Nay (−0.40), and these partially cancel. The model captures the internal conflict — this is a legislator who agrees with the bill's ideology but resists the establishment's agenda. In a 1D model, this conflict would be invisible.

Now consider Senator Park with ξ₁ = +0.6 (same ideology) but ξ₂ = +0.8 (establishment loyalist). Same bill:

1. **Dimension 1 contribution:** 1.0 × 0.6 = 0.60
2. **Dimension 2 contribution:** 0.5 × 0.8 = 0.40
3. **Subtract difficulty:** 0.60 + 0.40 − 0.2 = 0.80
4. **Apply logistic:** logistic(0.80) = 0.69
5. **Result: 69% probability of a Yea vote**

Same ideology, different establishment loyalty, different predicted vote. The 2D model can distinguish these two legislators; the 1D model cannot.

## The Rotation Problem

Moving from 1D to 2D introduces a new identification challenge that doesn't exist in one dimension.

### The Map Analogy

Imagine you're looking at a 2D scatter plot of legislator ideal points — a cloud of dots on a plane. Now imagine picking up the entire plane and rotating it 45 degrees. The dots are in new positions on the axes, but their relationships to each other haven't changed. Legislators who were close together are still close. Legislators who were far apart are still far apart.

**Any rotation of the 2D ideal point space gives equally valid predictions.** This is called **rotational invariance**, and it means the model can't tell which direction is "Dimension 1" and which is "Dimension 2" without external help.

In 1D, we had two mirror-image solutions (reflection invariance). In 2D, we have a continuous family of solutions — one for every possible rotation angle. It's a much harder identification problem.

### Why Rotation Matters

Without fixing the rotation, "Dimension 1" and "Dimension 2" are arbitrary labels. The model might put ideology on Dimension 2 and establishment loyalty on Dimension 1 in one run, then reverse them in the next. Or it might put ideology at a 30-degree angle to both axes. The numbers would be different, but the predictions would be identical.

For the model to produce interpretable results — where Dimension 1 reliably means "ideology" and Dimension 2 reliably means "establishment" — we need to pin down the rotation.

## PLT Identification: Pinning the Rotation

Tallgrass uses a technique called **Positive Lower Triangular (PLT) identification** to fix the rotation. The idea is to place constraints on the discrimination matrix (the collection of all β₁ and β₂ values) that uniquely determine the rotation.

### The Intuition

Imagine arranging all the bills' discrimination values in a table:

```
         Dim 1 (β₁)    Dim 2 (β₂)
Bill 1:    free           0         ← rotation anchor
Bill 2:    free          >0         ← positive diagonal
Bill 3:    free          free
Bill 4:    free          free
  ...       ...          ...
```

Three constraints are applied:

1. **Bill 1's β₂ = 0:** The first bill is declared to have zero discrimination on Dimension 2. This pins the rotation — Dimension 1 is defined as "the direction in which Bill 1 discriminates." Think of it as declaring magnetic north: you're choosing a reference direction.

2. **Bill 2's β₂ > 0:** The second bill's Dimension 2 discrimination is forced to be positive. This prevents a reflection of Dimension 2 (which would be like reading the y-axis upside down).

3. **All other β values are free:** The remaining bills can discriminate in any direction on both dimensions, constrained only by the data.

### The Analogy: Setting Up a Map Grid

PLT identification is like placing a **compass rose** on a map. The first constraint (β₂ = 0 for Bill 1) is like saying "this road runs exactly east-west" — it defines the orientation of your grid. The second constraint (β₂ > 0 for Bill 2) is like saying "north is up, not down." Together, they uniquely determine how the grid sits on the landscape.

Once the grid is set, all other features of the map (roads, rivers, towns) are located by the data alone. The constraints don't force the map to show anything in particular — they just ensure the map is oriented consistently.

**Codebase:** `analysis/06_irt_2d/irt_2d.py` — the PLT constraints are applied when constructing the discrimination matrix in `build_irt_2d_graph()`

## Supermajority Adaptive Tuning

The 2D model is harder to fit than the 1D model — more parameters, more complex geometry, weaker signals on Dimension 2. In supermajority chambers, the challenge is even greater: the minority party is so small that the second dimension's signal is faint.

Tallgrass addresses this with **adaptive tuning**: when the majority party holds more than 70% of seats, the sampler gets twice as many tuning steps (4,000 instead of the standard 2,000). This gives nutpie more time to learn the geometry of the posterior before starting to collect samples.

Additionally, the beta (discrimination) initialization for 2D models comes from PCA loadings rather than from 1D IRT. This avoids circular dependency — if the 1D model already captured the wrong axis, using its results to initialize the 2D model would propagate the error.

**Codebase:** `analysis/06_irt_2d/irt_2d.py` (`N_TUNE_SUPERMAJORITY = 4000`, triggered when `majority_frac > 0.70`)

## Dimension Swap Detection

Even with PLT constraints, the model may assign the ideology axis to Dimension 2 and the establishment axis to Dimension 1. PLT fixes the *rotation* but not which axis is which — it's like having a properly oriented map where the street labels might be swapped.

Tallgrass detects this by checking the **party separation** (Cohen's d) on each dimension:

- If Dimension 1 separates parties well (d > 2.0 and d₁ > d₂), everything is fine — ideology is on Dimension 1 as intended.
- If Dimension 2 separates parties better than Dimension 1 (d₂ > d₁ and d₂ > 2.0), the dimensions are swapped. Tallgrass relabels them so that the ideology dimension is always reported as Dimension 1.

This swap detection is one of the seven quality gates described in Volume 3, Chapter 5 — **Gate R7**, specifically designed to catch the horseshoe effect's downstream consequences in 2D models.

## Convergence in 2D

The 2D model has relaxed convergence thresholds compared to the 1D model:

| Diagnostic | 1D Threshold | 2D Threshold | Why Relaxed |
|-----------|-------------|-------------|-------------|
| R-hat | < 1.01 | < 1.05 | Dimension 2 has weaker signal |
| ESS | > 400 | > 200 | Fewer effective samples expected |
| Divergences | < 10 | < 50 | More complex geometry |

These relaxed thresholds reflect the reality that Dimension 2 captures only about 11% of the variation (compared to 35% for Dimension 1). The signal is real but faint, and the sampler needs more latitude.

This is why the canonical routing system (Chapter 7) has a **tiered quality gate** — the model may converge well enough for the Dimension 1 ideal points to be trustworthy (the ideology axis) even when Dimension 2's estimates have wide uncertainty.

## The 79th Senate: The Model Case

The 79th Kansas Senate is the purest example of why 2D IRT matters.

### The 1D Result (Before 2D)

The 1D model with anchor-pca produces:
- Party separation d = 0.28 on the ideal points
- R-hat > 1.5 (poor convergence)
- Tim Huelskamp (a far-right conservative) ranked among the "most liberal"

The axis is wrong. It's capturing establishment-vs-rebel, not liberal-vs-conservative.

### The 2D Result

The 2D model separates the two dimensions:

**Dimension 1 (ideology):** Party separation d = 6.17. Democrats cleanly at the negative end, Republicans at the positive end. Tim Huelskamp correctly at the conservative extreme (+3.26).

**Dimension 2 (establishment):** Party separation d ≈ 0.5 (no party alignment, as expected). Separates establishment Republicans from both contrarian Republicans and Democrats.

The 2D model succeeded precisely where the 1D model failed: by giving the establishment-vs-rebel conflict its own dimension, it freed Dimension 1 to capture pure ideology.

### Visualizing Two Dimensions

```
Establishment (+ξ₂)
    │
    │    ● ● ●    ← Moderate/establishment Republicans
    │  ● ● ●
    │
────┼─────────────── Dimension 1 (ξ₁) →
    │                   (liberal → conservative)
    │  ▲ ▲
    │    ▲ ▲ ▲    ← Conservative rebel Republicans
    │
Contrarian (−ξ₂)

 ■ ■ ■           ← Democrats (liberal, mixed on Dim 2)
```

In this scatter plot:
- **Circles (●):** Establishment Republicans — conservative on Dim 1, high on Dim 2
- **Triangles (▲):** Rebel Republicans — also conservative on Dim 1, but low on Dim 2
- **Squares (■):** Democrats — liberal on Dim 1, scattered on Dim 2

The 1D model collapsed the vertical dimension, projecting all three groups onto the horizontal axis. The rebels and Democrats, both low on Dim 2, ended up blended together. The 2D model preserves their separation.

## When to Use 2D vs. 1D

The 2D model is more powerful but more complex. Not every session needs it:

| Situation | Recommended Model | Why |
|-----------|-------------------|-----|
| Balanced chamber (50-65% majority) | 1D | Single dimension suffices; 2D adds noise |
| Moderate supermajority (65-75%) | Either | Monitor R-hat; use 2D if 1D convergence is poor |
| Strong supermajority (>75%) | 2D | Factional dynamics require a second dimension |
| Small chamber (<30 legislators) | 1D | Not enough data for 2D estimation |

For Kansas specifically:
- **House (all sessions):** 1D works well. The 125-member chamber is large enough that even a 72% Republican majority doesn't produce horseshoe distortion.
- **Senate (normal sessions):** 1D works for the 85th-87th, 89th-91st sessions, where the party divide dominates.
- **Senate (horseshoe sessions):** 2D is essential for the 78th-83rd and 88th sessions, where intra-Republican factionalism dominates the first PCA component.

The pipeline doesn't ask the user to make this choice. The canonical routing system (Chapter 7) automatically selects the best available model based on convergence diagnostics and party separation metrics.

---

## Key Takeaway

The 2D IRT model extends the 1D model by adding a second ideological dimension — typically capturing establishment-vs-contrarian dynamics that are independent of the left-right spectrum. The rotation problem (infinitely many equivalent coordinate systems in 2D) is solved by PLT constraints on the discrimination matrix. The 2D model is essential for Kansas Senate supermajority sessions where factional dynamics within the majority party would otherwise contaminate the ideology axis. Dimension swap detection and adaptive tuning for supermajority chambers ensure the model produces interpretable results even in the hardest cases.

---

*Terms introduced: 2D IRT, Multidimensional 2-Parameter Logistic (M2PL), rotational invariance, Positive Lower Triangular (PLT) identification, rotation anchor, dimension swap, adaptive tuning, supermajority tuning, establishment-contrarian axis*

*Next: [Partial Pooling: Hierarchical Models and Party Structure](ch05-hierarchical-models.md)*
