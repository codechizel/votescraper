# How Ideal Point Estimation Works: Anchors, Mirrors, and Divergences

*February 2026*

This article explains two foundational concepts behind Tallgrass's legislator ideal point estimates: how the model decides which direction is "left" and which is "right" (the identification problem), and what it means when the sampling algorithm reports "divergences" (a warning that the results may be unreliable). Both topics are central to understanding why some of our models produce trustworthy results and others don't — and what we do about it.

## Part 1: The Mirror Image Problem

### What the model actually does

The IRT (Item Response Theory) model estimates where each legislator falls on a liberal-to-conservative spectrum by analyzing their roll call votes. For each bill, the model learns two things: how ideologically divisive the bill is (its *discrimination*), and how likely it is to pass overall (its *difficulty*). For each legislator, it estimates a single number — their *ideal point* — that best predicts their entire voting record.

The math is simple. The probability that legislator *i* votes Yea on bill *j* is:

```
P(Yea) = 1 / (1 + exp(-(β_j × ξ_i - α_j)))
```

where ξ_i is the legislator's ideal point, β_j is the bill's discrimination (how sharply it divides legislators), and α_j is the bill's difficulty (how hard it is to pass). A bill with high discrimination cleanly separates liberals from conservatives. A bill with low discrimination passes or fails for reasons that have nothing to do with ideology.

This formula looks innocent, but it hides a deep problem.

### The mirror in the math

Imagine you've never been to Kansas and someone hands you a spreadsheet of 130 legislators and 300 roll call votes — nothing but ones and zeros, no names, no party labels, no context. You run the model. It converges perfectly and produces a beautiful ranking of legislators from one extreme to the other.

But which end is which?

The model cannot tell you. It can measure *distances* between legislators — it knows that legislator #47 and legislator #102 are far apart — but it has no way to determine which one is conservative and which is liberal. The reason is purely mathematical: if you take every ideal point and flip its sign (multiply by -1), and simultaneously flip every bill's discrimination parameter, the predictions don't change at all. `β × ξ` equals `(-β) × (-ξ)`. The model produces two solutions that are exact mirror images of each other, and both fit the data identically.

This is called *reflection invariance*. Statisticians have known about it for decades, but it's genuinely counterintuitive: the model can perfectly reconstruct the entire structure of legislative voting — who votes with whom, which bills are controversial, which legislators are extreme — but it literally cannot tell left from right.

There are actually three separate ambiguities hiding in the math:

1. **The mirror problem** (reflection): flipping all signs produces an equally valid solution.
2. **The zero problem** (location): shifting every ideal point by the same amount (and adjusting bill parameters to match) produces an equally valid solution. The model has no natural center.
3. **The ruler problem** (scale): stretching or compressing the entire spectrum (and adjusting bill parameters to compensate) produces an equally valid solution. A spectrum from -1 to +1 and a spectrum from -100 to +100 fit the data equally well.

Without solving all three, the model's output is meaningless. The estimates would average to zero across mirror-image solutions, producing the mathematical equivalent of a shrug.

### How we fix it: anchoring two legislators

The solution is to tell the model something it cannot figure out on its own. We pin down two legislators at known positions — one on the left and one on the right.

Think of it like a map without a compass rose. The map accurately shows that Topeka is north of Wichita and that Kansas City is east of both. But without a compass, you could rotate the map any direction and it would still be internally consistent. Anchoring two cities — "I know for a fact that Kansas City is on the right side of the map and Dodge City is on the left" — fixes the orientation. Everything else falls into place.

In practice, we fix the most conservative legislator at ξ = +1 and the most liberal at ξ = -1. This solves all three ambiguities at once:

- **Mirror**: fixed. Conservatives are positive, liberals are negative.
- **Zero**: fixed. The center of the scale is defined by the midpoint of the two anchors.
- **Scale**: fixed. The distance between the two anchors is exactly 2 units, calibrating the ruler.

### How we choose the anchors

The obvious question: if the model can't tell left from right, how do we know who is conservative and who is liberal?

We use PCA — Principal Component Analysis — which runs before the IRT model in the pipeline. PCA performs a simpler version of the same task: it finds the main axis of variation in the vote matrix and assigns each legislator a score along that axis. PCA has the same mirror ambiguity (it can't tell which end is which either), but we resolve it using the sign convention that Democrats score lower than Republicans. Since we know the party labels, we can orient the PCA axis correctly.

The anchor selection algorithm, implemented in `select_anchors()` in `analysis/05_irt/irt.py`, works as follows:

1. **Start with PCA scores.** Every legislator already has a PC1 score from the PCA phase (Phase 2 of the pipeline).

2. **Filter for participation.** A good anchor needs to have voted on most bills. We require at least 50% participation in the filtered vote set. A legislator who missed half the votes would be a poor reference point — the model wouldn't have enough data from them to calibrate the bill parameters.

3. **Pick the extremes.** Among eligible legislators, the one with the *highest* PC1 score becomes the conservative anchor (fixed at ξ = +1.0). The one with the *lowest* PC1 score becomes the liberal anchor (fixed at ξ = -1.0).

That's it. No manual input, no political judgment calls, no hardcoded names. The algorithm adapts automatically to every biennium and every chamber. In the 91st Legislature (2025-26), for example, the House anchors might be the most partisan Republican and the most partisan Democrat. In a biennium where an Independent serves, they'll never be an anchor (Independents are excluded from IRT for different reasons — see ADR-0021).

Once anchored, the remaining ~128 legislators (for a typical House) are free to land anywhere on the spectrum. Their positions are entirely determined by their voting records. The anchors simply provide the coordinate system.

### Other identification strategies

The PCA-based anchor approach described above (`anchor-pca`) is the default for balanced chambers where both parties hold roughly equal seats. But Kansas has a Republican supermajority (~72%), and in supermajority chambers, PCA's first component can be distorted by a **horseshoe effect** — it captures establishment-vs-rebel variation within the majority party rather than the left-right axis.

For these chambers, Tallgrass auto-selects a different strategy: `anchor-agreement`. Instead of PCA scores, it measures how often each legislator agrees with the opposing party on contested votes. The most partisan Republican (lowest agreement with Democrats) and the most partisan Democrat (lowest agreement with Republicans) become anchors. This selects genuine ideological extremes rather than PCA artifacts.

Tallgrass implements seven identification strategies in total, including constraint-based approaches that avoid anchoring on any individual legislator. The `--identification` CLI flag overrides auto-detection. See `docs/irt-identification-strategies.md` for the complete catalog with literature references and auto-detection logic.

### Why not just use party labels?

A reasonable question: if we already know who is a Democrat and who is a Republican, why not just label the scale using party? Because the model's power comes from measuring *within-party* variation. In the Kansas House, where Republicans hold a supermajority (~72% of seats), the interesting question isn't "are Democrats more liberal than Republicans?" (of course they are). It's "which Republicans vote more like moderates?" and "how much ideological variation exists within the majority?" Anchoring to two extreme legislators, rather than to parties, preserves this granularity.

### The hierarchical model uses a different approach

The hierarchical IRT model (Phase 10) doesn't fix any individual legislators. Instead, it uses an *ordering constraint*: it requires the Democratic party mean to be lower than the Republican party mean within each chamber. This approach has a different tradeoff — it allows all legislators to participate in partial pooling toward their party mean, which the flat IRT's hard anchors prevent for the two anchored legislators. But it determines only the *direction* of the scale, not the location or scale, which is why the hierarchical and flat IRT produce ideal points on different scales (requiring linear rescaling for comparison).

## Part 2: What Divergences Mean

### The sampling problem

Once the model is identified (the mirror problem is solved), we still need to actually estimate the parameters. With 130 legislators, 300 bills, and two parameters per bill, there are roughly 730 quantities to estimate simultaneously. We can't solve this with simple algebra — the model is too interconnected. Every legislator's ideal point depends on every bill's parameters, which in turn depend on every other legislator's ideal point.

The standard solution is MCMC — Markov Chain Monte Carlo — which explores the space of possible parameter values by random walking through it. The specific algorithm we use is called NUTS (the No-U-Turn Sampler), which is the state of the art for these kinds of models. Think of it as sending out an explorer to map unknown terrain. The explorer wanders around, spending more time in regions that fit the data well (valleys) and less time in regions that don't (hilltops). After enough wandering, the explorer's trail map tells us where the most plausible parameter values are — and how uncertain we should be about them.

### The physics simulation inside the sampler

NUTS doesn't do a random walk. It uses physics.

Imagine placing a ball on a landscape where the altitude at each point represents how poorly that combination of parameter values fits the data. Valleys are good fits; mountains are bad ones. Now give the ball a random push and let it roll. Because of the physics — momentum, gravity, friction — the ball naturally spends most of its time in the valleys (where the data fits well) and only briefly crosses over ridges into neighboring valleys.

To simulate this rolling, the algorithm uses a *leapfrog integrator*: it takes the ball's current position and velocity, assumes the surface is approximately flat for a small step, and projects where the ball will be after that step. Then it repeats. This is the same kind of numerical integration used to simulate planetary orbits or weather systems.

The crucial assumption is that the surface is approximately flat within each step. On gently rolling terrain, this works well — the ball traces out a smooth path that accurately follows the contours of the landscape. But on terrain with sudden cliffs, sharp valleys, or hairpin turns, the flat-surface assumption breaks down.

### What a divergence is

A divergence is the moment the ball flies off the cliff.

The algorithm takes a step forward, assuming the terrain is gently sloped. But it has stepped across a region where the surface drops sharply or curves suddenly. The ball's projected trajectory diverges wildly from where it would actually go — it overshoots the edge and flies off into mathematically impossible territory. The algorithm detects this (the simulated energy spikes far above what physics allows) and discards the bad step.

The step is thrown away, so it doesn't directly corrupt the results. But the damage is subtler: the algorithm has been deflected *away* from a region of the landscape that contains important information. Consistently being deflected from the same region means the explorer never maps that area. The trail map has a blank spot — and if the blank spot contains a significant valley, the results will be biased.

Think of it like a GPS navigation system that recalculates your route every few seconds by projecting a straight line forward. On a highway, this works perfectly — the road is straight. But on a mountain switchback, the GPS keeps projecting you straight off the cliff. It catches the error and recalculates, but the result is that your recorded track never includes the switchback. Your map shows a straight road where there's actually a winding one, and the elevation profile is wrong because you missed the valley the switchback descended into.

### Why the terrain can be treacherous

Most models produce smooth, gently rolling terrain that the sampler navigates easily. The per-chamber IRT models in Tallgrass — estimating ideal points for 130 House legislators or 40 Senate legislators — have zero divergences across all 16 chamber-biennium combinations we've tested. The terrain is smooth because the model structure is simple: one level of hierarchy, a well-identified scale, and enough data to pin down every parameter.

The joint cross-chamber model is a different story. It attempts to place all 170 legislators on a single scale by estimating them simultaneously. This creates three kinds of treacherous terrain:

**The funnel.** The joint model has a three-level hierarchy: legislators within parties, parties within chambers, chambers within the legislature. The top level has a parameter (σ_chamber) that controls how spread out the chamber-level means are. When this parameter is large, the lower-level parameters can roam freely. When it's small, they're compressed into a tight cluster. The result is a landscape shaped like a funnel — wide at the top and exponentially narrow at the bottom.

The sampler has to pick one step size for the entire landscape. A step size that works for the wide part of the funnel overshoots the narrow throat. A step size safe for the throat is agonizingly slow in the wide part. The ball bounces off the walls of the throat and flies back up, never properly exploring the narrow region. Those bounces are divergences.

This is called *Neal's funnel* after the statistician Radford Neal who described it, and it's the single most common cause of divergences in hierarchical models. With only 2 chambers and 4 party groups informing the top-level parameters, our funnel is especially narrow — the data barely constrains σ_chamber.

**The boundary wall.** When we constrained bill discrimination to be positive (to solve the mirror problem for the joint model), we inadvertently created a wall in the landscape. Some bills — bipartisan procedural votes, near-unanimous motions — genuinely have near-zero discrimination. The model needs to explore close to zero for these bills, but the positivity constraint creates a cliff at the boundary. The curvature of the landscape changes violently near this boundary: the distance between 0.01 and 0.001 in the model's internal coordinate system is 230 times larger than the distance between 1.00 and 0.99.

When we first tried this approach, divergences exploded from 10 to 2,041. We fixed this specific problem by reparameterizing — instead of directly constraining β > 0, we estimated log(β) on an unconstrained scale and then exponentiated. Mathematically identical, but the landscape is smooth everywhere. Divergences dropped from 2,041 to 828.

**The high-dimensional maze.** With ~1,000 parameters, the joint model's landscape is a 1,000-dimensional surface. The sampler must navigate corridors, saddle points, and ridges in this space. Even without funnels or boundaries, the sheer dimensionality slows mixing — the ball ricochets off walls more often in a narrow 1,000-dimensional corridor than in a wide 100-dimensional one.

### Why divergences matter for our results

A few divergences (say, 5-10 out of 8,000 samples) might be tolerable — the blank spots in the map are small and probably don't affect the big picture. But 828 divergences out of 8,000 samples means over 10% of the sampler's exploration attempts hit treacherous terrain. That's not a few potholes on an otherwise smooth road; it's a significant portion of the map that's unreliable.

This is why the per-chamber models (zero divergences) are the primary output of the analysis, and the joint model results come with caveats. It's also why we implemented Stocking-Lord IRT scale linking (ADR-0055) as a production alternative — it achieves cross-chamber comparison using the well-converged per-chamber estimates and a simple 2-parameter optimization, sidestepping the treacherous joint model landscape entirely.

### The bottom line

Divergences are the model's honest admission that it couldn't fully explore some part of the answer. Rather than silently returning biased results, the algorithm flags every instance where its simulation broke down. This is actually a feature — many older statistical methods fail silently. The NUTS sampler's divergence diagnostics are one of the most important practical advances in Bayesian statistics in the last decade, precisely because they tell you when to trust the results and when not to.

When we report that the per-chamber models have zero divergences, it means the sampling algorithm successfully explored every corner of the probability landscape. The ideal point estimates, the uncertainty intervals, and the party-level summaries can all be trusted. When we report 828 divergences in the joint model, it means there are regions of the landscape the algorithm couldn't reach — and the estimates may be missing information from those regions. This is why the joint model results are labeled as experimental, and why the per-chamber ideal points remain the primary output.

---

**Related documents:**
- IRT design choices: `analysis/design/irt.md`
- IRT deep dive: `docs/irt-deep-dive.md` (code audit, ecosystem survey, test gaps)
- IRT field survey: `docs/irt-field-survey.md` (identification problem, Python ecosystem gap)
- Joint model deep dive: `docs/joint-model-deep-dive.md` (concurrent calibration failure, reparameterized beta experiments)
- Hierarchical convergence: `docs/hierarchical-convergence-improvement.md` (funnel theory, improvement plan)
- ADR-0055: Reparameterized LogNormal beta and IRT scale linking
