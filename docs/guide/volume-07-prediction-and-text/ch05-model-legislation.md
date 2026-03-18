# Chapter 5: Model Legislation: Detecting Copy-Paste Policy

> *Sometimes a Kansas bill looks suspiciously similar to a bill in Oklahoma — or to an ALEC template published years earlier. Can we detect that automatically? And what does it tell us about how policy actually spreads?*

---

## The Copy-Paste Problem

State legislatures produce thousands of bills every session. Some are entirely original — drafted by a legislator's staff to address a local concern. But many are adapted from external sources: model legislation written by advocacy organizations, bills that passed in neighboring states, or templates circulated by national policy networks.

This isn't inherently good or bad. Policy diffusion — the spread of ideas from one state to another — is how federalism is supposed to work. Kansas might borrow a successful school funding formula from Colorado, or adopt a criminal sentencing reform that worked in Oklahoma. But when legislation is copied verbatim from a template drafted by a lobbying organization, without local adaptation or disclosure, it raises questions about whose interests the bill actually serves.

The challenge is detection. A legislator won't typically announce "I copied this from ALEC." The bill appears as original Kansas legislation, and only careful comparison to the source material reveals the connection. With 800 bills per session and thousands of potential source documents, manual comparison is impractical.

Tallgrass automates this comparison using the same embedding technology from Chapter 3.

## ALEC and Model Legislation

The most prominent source of model legislation in American politics is the **American Legislative Exchange Council** (ALEC), a membership organization that brings together state legislators and corporate representatives to develop template bills. ALEC doesn't write the bills that appear in state legislatures — technically, legislators introduce them. But ALEC provides the templates, and the degree of textual similarity between ALEC models and introduced bills can be striking.

Tallgrass maintains a corpus of **1,061 ALEC model policies** scraped from publicly available sources. Each model bill has a title, category, and full text. The corpus covers a wide range of policy areas: tax reform, education, criminal justice, healthcare, energy, and more.

The question this analysis answers isn't "is this bill from ALEC?" — that's a political judgment. The question is: "how similar is the text of this Kansas bill to texts in the ALEC corpus?" The reader can then look at the matched pairs and draw their own conclusions.

**Codebase:** `analysis/23_model_legislation/model_legislation.py` (`ALEC_DATA_DIR`), data loaded via `analysis/23_model_legislation/model_legislation_data.py` (`load_alec_corpus()`)

## The Detection Method

### Step 1: Embed Everything

Both the Kansas bills and the ALEC corpus are run through the same embedding model used in Chapter 3 (bge-small-en-v1.5, 384 dimensions). The same preprocessing pipeline applies: strip enacting clauses, normalize statute references, remove boilerplate.

This shared embedding space is critical. Because both corpora are embedded by the same model, their vectors are directly comparable. A Kansas bill about tax credits and an ALEC model about tax credits will have similar 384-dimensional fingerprints, regardless of surface-level differences in wording.

### Step 2: Compute Cosine Similarity

For every Kansas bill, compute the cosine similarity against every ALEC model:

```
similarity(KS_bill, ALEC_model) = (embedding_KS · embedding_ALEC) / (||embedding_KS|| × ||embedding_ALEC||)
```

The result is a matrix of similarities — one number for every Kansas-ALEC pair. A value of 1.0 means the embeddings are identical in direction (the bills have extremely similar semantic content). A value of 0.0 means the bills are about entirely unrelated topics.

### Step 3: Classify the Match Tier

Not every similarity above zero is meaningful. Two bills about healthcare will have nonzero similarity just because they use healthcare vocabulary, even if one is about Medicaid expansion and the other is about insurance regulation. Tallgrass uses three tiers:

| Tier | Cosine Similarity | Interpretation |
|------|------------------|----------------|
| **Near-identical** | >= 0.95 | Likely direct copy or very close adaptation |
| **Strong match** | >= 0.85 | Adapted from the same source, with meaningful modifications |
| **Related** | >= 0.70 | Same policy domain — warrants investigation, but may be coincidence |

These thresholds were chosen conservatively. At 0.95, two bills share enough semantic structure that independent drafting would be a remarkable coincidence. At 0.85, the bills clearly address the same specific policy with similar framing. At 0.70, they're in the same policy neighborhood but might have arrived there independently.

**Codebase:** `analysis/23_model_legislation/model_legislation.py` (`THRESHOLD_NEAR_IDENTICAL = 0.95`, `THRESHOLD_STRONG_MATCH = 0.85`, `THRESHOLD_RELATED = 0.70`)

### Step 4: Confirm with N-gram Overlap

Embedding similarity measures semantic meaning — whether two bills are *about* the same thing. But it can't distinguish between "about the same thing" and "copied from the same source." Two independent bills about property tax caps might have high embedding similarity because they cover the same topic, not because one was copied from the other.

**N-gram overlap** provides the confirmation. An **n-gram** is a sequence of *n* consecutive words. "State of Kansas" is a 3-gram. "Income tax credit for child care" is a 6-gram. If two bills share many 5-grams, it's strong evidence of direct text reuse — the odds of two independent authors producing the same five-word sequences repeatedly are vanishingly small.

The computation:

```
1. Tokenize both texts into words (lowercase)
2. Extract all 5-grams (every sequence of 5 consecutive words)
3. Count how many of Bill A's 5-grams also appear in Bill B
4. Divide by Bill A's total 5-gram count
```

**Plain English:** "What fraction of Bill A's five-word phrases also appear in Bill B?"

An overlap of 0.20 (20%) or higher suggests genuine text sharing. Below that, the similarity is semantic (same topic) rather than textual (same words).

The combination is powerful:

| High Embedding Similarity + High N-gram Overlap | Copy or close adaptation — the bills share both meaning and specific language |
|---|---|
| **High Embedding Similarity + Low N-gram Overlap** | Same topic but independently drafted — similar ideas, different words |
| **Low Embedding Similarity + High N-gram Overlap** | Unlikely, but could indicate shared boilerplate in otherwise different bills |
| **Low Embedding Similarity + Low N-gram Overlap** | Unrelated bills |

N-gram overlap is computed only for strong matches (similarity >= 0.85) to save computation. Below that threshold, the bills are unlikely to be direct copies regardless.

**Codebase:** `analysis/23_model_legislation/model_legislation.py` (`compute_ngram_overlap()`, `NGRAM_SIZE = 5`, `NGRAM_OVERLAP_THRESHOLD = 0.20`)

## Cross-State Comparison: Following the Trail

ALEC is the most prominent source of model legislation, but it's not the only pathway for policy diffusion. Bills can spread state-to-state through governor's associations, legislative staff networks, academic policy recommendations, or simple observation of neighboring states.

Tallgrass compares Kansas bills against legislation from four neighboring states:

| State | Abbreviation | Why Included |
|-------|-------------|-------------|
| Missouri | MO | Eastern neighbor, similar rural-urban dynamics |
| Oklahoma | OK | Southern neighbor, similar political leanings |
| Nebraska | NE | Northern neighbor, unicameral (different legislative structure) |
| Colorado | CO | Western neighbor, different political environment |

The same embedding-and-similarity pipeline runs against each state's bills from the same session period. A Kansas bill that matches an Oklahoma bill at 0.90 similarity with 25% n-gram overlap is strong evidence of cross-state borrowing — the bill may have been adapted from Oklahoma's version (or both may derive from a common source).

### The Policy Diffusion Network

When a Kansas bill matches legislation in multiple states, the picture shifts from individual copying to **policy diffusion** — the systematic spread of policy ideas across state borders.

Tallgrass visualizes this as a network graph:
- **Kansas bills** are blue nodes
- **Source matches** (ALEC, neighboring states) are orange nodes
- **Edges** connect matches, with thickness proportional to similarity

A Kansas bill connected to both an ALEC model and an Oklahoma bill suggests a three-step chain: ALEC published a template, Oklahoma adopted it, and Kansas followed (or all three drew from the same source independently). The network makes these patterns visible at a glance.

**Codebase:** `analysis/23_model_legislation/model_legislation.py` (cross-state via `load_cross_state_texts()`, default states in `DEFAULT_STATES = ["mo", "ok", "ne", "co"]`)

## What the Results Look Like

### The Similarity Distribution

A histogram of maximum ALEC similarity per Kansas bill shows a characteristic shape: a large peak near 0.3-0.5 (most bills are vaguely related to some ALEC model, just by being about state governance), a rapid decline above 0.7, and a thin tail above 0.85.

In a typical session, Tallgrass finds:
- **0-3 near-identical matches** (>= 0.95): Bills that are essentially copied from an ALEC model
- **5-15 strong matches** (>= 0.85): Bills adapted from ALEC with meaningful modifications
- **30-50 related matches** (>= 0.70): Bills in the same policy space, but likely independent

### Near-Identical Detail Cards

For each near-identical match, the report shows a side-by-side comparison: the first 500 characters of the Kansas bill next to the first 500 characters of the ALEC model, with the cosine similarity and n-gram overlap prominently displayed. This allows the reader to see the textual similarity for themselves.

### Topic-Match Heatmap

A heatmap shows which policy topics (from Chapter 3's BERTopic) have the most matches and from which sources. Some topics may have many ALEC matches (e.g., tax policy, criminal justice) while others have more cross-state matches (e.g., education, transportation). The pattern reveals which policy areas are most influenced by model legislation versus state-to-state borrowing.

## The Essential Caveat: Similarity Is Not Causation

This analysis identifies textual similarity, not causal relationships. A high similarity score between a Kansas bill and an ALEC model does not prove that the bill was "from" ALEC. Several alternative explanations exist:

**Common legal conventions.** Bills addressing the same statutory area will inevitably share language because they reference the same existing statutes and use the same legal frameworks. A bill amending K.S.A. 79-3603 (sales tax) will share phrasing with any other bill amending the same statute, including ALEC models.

**Parallel innovation.** Two independent policy shops can arrive at similar solutions to the same problem. If states across the country are grappling with the same issue (e.g., telehealth regulation during a pandemic), their bills may converge in language without any copying.

**Intermediate sources.** A bill might be adapted from a National Conference of State Legislatures (NCSL) recommendation, which itself was influenced by ALEC, but the legislator's direct source was NCSL, not ALEC. The textual similarity to ALEC would be real but the attribution would be misleading.

**Diffuse influence.** ALEC and similar organizations influence the policy conversation broadly. A legislator might read an ALEC white paper, internalize the arguments, and draft a bill that captures similar ideas in similar language — without ever seeing the model bill. The similarity is intellectual, not textual, but the embeddings can't distinguish the two.

Tallgrass reports the similarities and lets the reader investigate. The n-gram overlap provides an additional data point: if 30% of the five-word phrases are shared, independent authorship becomes harder to argue. But the analysis stops at "these texts are very similar" and does not claim "this bill was copied from this source." That's a judgment for journalists, researchers, and citizens to make with full context.

## What Can Go Wrong

### False Positives from Shared Statutory Language

Bills that amend the same section of Kansas law will naturally share the text of that section. Two entirely unrelated bills — one about school finance and one about teacher certification — might both quote K.S.A. 72-6407 and achieve high similarity on the quoted passages. The preprocessing step that replaces K.S.A. references with "STATUTE_REF" mitigates this but doesn't eliminate it entirely.

### Cross-State Data Gaps

Not all neighboring states have bill texts readily available in machine-readable format. Tallgrass uses the OpenStates API as a data source, which depends on each state's legislative website structure. If a state's data is incomplete or unavailable for a particular session, cross-state comparison is skipped for that state, and the analysis reports the gap.

### Embedding Limitations for Legal Text

The bge-small model handles general English well but may not capture fine-grained legal distinctions. Two bills that a lawyer would recognize as substantially different in legal effect might have high embedding similarity because they use similar vocabulary in similar patterns. The n-gram confirmation step catches some of these false positives, but the underlying embedding is general-purpose.

### Corpus Completeness

The ALEC corpus of 1,061 model policies is extensive but not exhaustive. ALEC produces new model legislation regularly, and the scraped corpus represents a point-in-time snapshot. A Kansas bill copied from a recently published ALEC model that isn't yet in the corpus would be missed. Similarly, bills adapted from other organizations' model legislation (Heritage Foundation, National Governors Association, etc.) would not be flagged because those corpora aren't included.

---

## Key Takeaway

Tallgrass detects model legislation by embedding Kansas bills and reference corpora (ALEC, neighboring states) into the same semantic space and measuring cosine similarity. Near-identical matches (>= 0.95) are confirmed with 5-gram overlap to distinguish genuine text reuse from topical similarity. Cross-state comparison reveals policy diffusion networks. The essential caveat: similarity is not causation — high scores identify bills that warrant investigation, not bills proven to be copies.

---

*Terms introduced: model legislation, ALEC (American Legislative Exchange Council), policy diffusion, cosine similarity (in document comparison), n-gram overlap, near-identical match, strong match, related match, cross-state comparison, detail card, common legal conventions, parallel innovation*

*Previous: [Text-Based Ideology and Issue-Specific Scores](ch04-text-ideology.md)*
