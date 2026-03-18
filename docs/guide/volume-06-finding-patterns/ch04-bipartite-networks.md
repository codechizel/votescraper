# Chapter 4: Bipartite Networks: Bills That Bridge the Aisle

> *Chapter 3 built a network of legislators connected by shared votes. But there's another actor in every roll call: the bill itself. What if the bill is the protagonist? Which pieces of legislation bring unlikely allies together?*

---

## Flipping the Lens

Every roll call vote involves two kinds of things: **legislators** (who vote) and **bills** (what they vote on). Chapter 3's network connected legislators to legislators, using bills as a medium of comparison. But you can also connect bills to bills, using legislators as the medium. Or — most usefully — you can keep both kinds of entities in the same network.

A network with two distinct types of nodes, where edges only connect across types (never within), is called a **bipartite network**. In the Tallgrass bipartite network:

- **Left nodes:** Legislators (color-coded by party)
- **Right nodes:** Bills (characterized by their IRT discrimination β)
- **Edges:** A legislator is connected to a bill if they voted **Yea**

This structure answers questions that the legislator-only network cannot:

- Which bills attracted the most bipartisan support?
- Which bills were so polarizing that only one party voted for them?
- Which bills serve as bridges — connecting legislators who agree on little else?

## Bill Polarization: The Partisan Divide per Bill

The simplest bipartite-network analysis asks: **how partisan was each bill's vote?**

The **polarization score** measures the gap between Republican and Democratic support:

```
polarization = |pct_R_yea − pct_D_yea|
```

**Step-by-step example:**

A bill receives Yea from 85% of Republicans and 20% of Democrats:

```
polarization = |0.85 − 0.20| = 0.65
```

That's a highly partisan vote. Both parties voted, but they voted opposite ways.

Now consider a bill that passed with 90% of Republicans and 88% of Democrats:

```
polarization = |0.90 − 0.88| = 0.02
```

That's bipartisan — both parties supported it at nearly the same rate.

The polarization distribution for a Kansas session looks like a U-shape with a long right tail. Most bills are either very bipartisan (polarization near 0, typically procedural or noncontroversial) or very partisan (polarization above 0.5, the wedge issues). The middle is thinner — there are fewer bills that genuinely split opinion without splitting along party lines.

**Codebase:** `analysis/12_bipartite/bipartite.py` (`compute_bill_polarization()`, `BILL_POLARIZATION_MIN_VOTERS = 10`)

## Bridge Bills: The Unlikely Alliances

A **bridge bill** is a piece of legislation that connects legislators who otherwise rarely vote together. In network terms, it has high **bipartite betweenness centrality** — it sits on the shortest paths between legislators from different communities.

The analogy: imagine a small town where the teachers and the farmers never interact. Then a bill about school bus routes on farm roads comes up, and both groups attend the same public hearing. That bill is a bridge — it connects two otherwise separate communities.

Tallgrass identifies bridge bills by combining two criteria:

1. **High bipartite betweenness:** The bill connects otherwise disconnected parts of the legislator network
2. **Low polarization:** The bill attracted bipartisan support (if it were purely partisan, it wouldn't bridge anything)

The intersection of these two criteria — high betweenness *and* low polarization — identifies the legislation that genuinely creates cross-party coalitions. These are often policy-specific bills where party ideology is less relevant: infrastructure, local governance, professional licensing, or public safety.

**Codebase:** `analysis/12_bipartite/bipartite.py` (`compute_bipartite_betweenness()`, `identify_bridge_bills()`, `TOP_BRIDGE_BILLS = 20`)

## The Newman Projection: Connecting Bills Through Shared Voters

To find clusters of *bills* that tend to attract the same voters, Tallgrass **projects** the bipartite network onto the bill side. Two bills become connected if they share Yea voters.

But a naive projection has a problem. Consider a legislator who votes Yea on 400 out of 500 bills. They connect nearly every pair of bills to each other — not because those bills are related, but because this legislator votes Yea on almost everything. Their connections are trivial.

**M.E.J. Newman** (2001) solved this in the context of scientific collaboration networks. He observed that two scientists who co-authored a paper with 50 other scientists have a weaker connection than two who co-authored a paper alone. His weighting formula down-weights connections through high-degree nodes:

```
weight(bill_A, bill_B) = sum over shared legislators of: 1 / (k_i − 1)
```

Where k_i is the number of Yea votes legislator *i* cast.

**Plain English:** "If a legislator voted Yea on 100 bills, their contribution to each bill-to-bill connection is divided by 99. If they voted Yea on only 5 bills, each connection they create is worth 1/4 — much more."

The Newman weighting ensures that connections between bills are driven by *selective* shared voters, not by universal yes-voters who connect everything to everything.

**Implementation note:** Tallgrass computes this efficiently using matrix multiplication. If B is the legislator-by-bill adjacency matrix, the Newman-weighted bill projection is B^T × D × B, where D is a diagonal matrix of 1/(k_i − 1) weights.

**Codebase:** `analysis/12_bipartite/bipartite.py` (`build_bill_projection()`, `NEWMAN_PROJECTION = True`)

## Bill Communities: Legislation That Travels Together

Applying the Leiden algorithm to the Newman-weighted bill projection reveals **bill communities** — clusters of legislation that attracted the same voting coalitions.

Each community gets a profile: the average Republican support rate, the average Democratic support rate, and the average polarization. This reveals the legislative landscape from the bill side:

| Community | Mean R %Yea | Mean D %Yea | Mean Polarization | Interpretation |
|-----------|------------|------------|-------------------|---------------|
| 1 | 92% | 85% | 0.08 | Bipartisan bread-and-butter legislation |
| 2 | 88% | 12% | 0.76 | Partisan R-priority bills |
| 3 | 25% | 95% | 0.70 | Partisan D-priority bills (usually doomed) |
| 4 | 65% | 55% | 0.10 | Moderate bills with modest bipartisan support |

Bill communities often map to policy domains: tax policy, education, criminal justice, infrastructure. But the communities are discovered from voting patterns, not policy labels — they emerge from who votes together on what, not from how bills are titled.

**Codebase:** `analysis/12_bipartite/bipartite.py` (`detect_bill_communities()`, `analyze_bill_community_profiles()`, `BILL_CLUSTER_RESOLUTIONS`)

## The BiCM Backbone: What's Statistically Surprising?

### The Problem with Simple Projections

When you project a bipartite network onto one side (say, from legislators × bills to legislators × legislators), the result is dominated by **degree effects**. A legislator who votes Yea on 400 bills will have strong connections to almost every other Yea-heavy legislator — not because they share political views, but because they both vote a lot. The co-voting count between two high-turnout legislators will be high simply because both show up.

You want to know: "Do these two legislators co-vote more than you'd *expect*, given how active each of them is?"

### The BiCM Null Model

The **Bipartite Configuration Model** (Saracco et al., 2015, 2017) provides the null hypothesis. It generates a universe of random bipartite networks that preserve the expected degree of every node on both sides — every legislator keeps their expected number of Yea votes, and every bill keeps its expected number of supporters.

The math uses the **maximum entropy principle**: among all random networks that match the observed degree sequences, the BiCM is the one that makes the fewest additional assumptions. It's the "fairest" random model — the one that says "I only know how active each legislator is and how popular each bill is; everything else is random."

For each pair of legislators, the BiCM computes a **p-value**: the probability that their observed co-voting count (or higher) would occur under the null model. If two legislators co-voted on 150 bills and the BiCM says there's only a 0.001 probability of that happening by chance given their individual activity levels, that's a statistically significant co-voting relationship.

### Constructing the Backbone

Tallgrass keeps only edges with p-values below a significance threshold:

| Chamber | Threshold | Rationale |
|---------|-----------|-----------|
| House | p < 0.01 | Conservative — ~150 legislators create ~11,000 possible pairs |
| Senate | p < 0.05 | Relaxed — ~40 legislators create only ~780 possible pairs |

The result is a **backbone graph**: a sparse network of legislators where every edge represents a co-voting relationship that *cannot be explained by individual activity levels alone.* These are the connections that reflect genuine political alignment, not just showing up to vote.

**Codebase:** `analysis/12_bipartite/bipartite.py` (`extract_bicm_backbone()`, `build_backbone_graph()`, `BICM_SIGNIFICANCE = 0.01`, `BICM_SIGNIFICANCE_SENATE = 0.05`)

## Comparing Backbones: BiCM vs. Kappa

Tallgrass produces two backbone networks for each chamber:

1. **Kappa backbone** (from Chapter 3): edges where Kappa exceeds the threshold, then filtered by the disparity filter
2. **BiCM backbone** (from this chapter): edges where co-voting is statistically significant given individual activity levels

Comparing the two reveals different things:

- **Edges in both:** The core political alliances — strong agreement that's also statistically unexpected
- **Edges in Kappa only:** Pairs who agree beyond chance but whose co-voting isn't surprising given how active they are
- **Edges in BiCM only:** Pairs whose co-voting is surprisingly high given their activity levels, even if their absolute Kappa isn't striking

The most interesting category is BiCM-only edges, especially **cross-party** ones. These are pairs of legislators — one Republican, one Democrat — who co-vote on specific subsets of legislation more than their individual Yea rates predict. These are the hidden alliances: legislators who quietly collaborate on particular policy domains without showing up as allies in overall agreement statistics.

**Codebase:** `analysis/12_bipartite/bipartite.py` (`compare_backbones()`)

## What Bipartite Analysis Reveals About Kansas

### Bridge Bills Are Policy-Specific

The bridge bills — high betweenness, low polarization — tend to cluster in a few policy domains: transportation, local government, professional licensing, and public safety. These are areas where the party divide is weakest, and where pragmatic coalitions form around specific technical needs rather than ideological principles.

### Bill Polarization Tracks IRT Discrimination

Bills with high IRT discrimination (|β| > 1.5) tend to have high polarization. This makes sense: bills that sharply separate ideologies (high β) are the same bills where the parties vote in opposite directions (high polarization). The correlation isn't perfect — some highly discriminating bills split the majority party internally rather than along party lines — but it's strong enough to validate both measures.

### Hidden Alliances Are Real but Narrow

The BiCM backbone typically reveals 5–15 cross-party edges that don't appear in the Kappa backbone. These are genuine alliances on specific policy areas — a rural Republican and a rural Democrat who agree on agriculture bills, or a moderate Republican and a centrist Democrat who cooperate on education. The alliances are narrow (topic-specific) but real (statistically significant).

---

## Key Takeaway

Bipartite network analysis shifts the focus from legislators to legislation. Bill polarization quantifies the partisan divide per bill. Newman-weighted projections reveal bill communities — clusters of legislation that attract the same voting coalitions. The BiCM backbone strips out degree effects to find statistically surprising co-voting relationships — including hidden cross-party alliances invisible to simpler methods. Together, these tools tell you not just who votes together, but *what* they vote together *on*.

---

*Terms introduced: bipartite network, bill polarization, bridge bill, bipartite betweenness, Newman weighting, bill projection, bill community, BiCM (Bipartite Configuration Model), maximum entropy null model, p-value backbone, hidden alliance*

*Next: [Classical Indices: Rice, Party Unity, and the Maverick Score](ch05-classical-indices.md)*
