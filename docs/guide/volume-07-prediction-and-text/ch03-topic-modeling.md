# Chapter 3: Topic Modeling: What Are the Bills About?

> *So far, every bill has been a number — a Yea/Nay pattern, a difficulty score, a discrimination parameter. But bills are also documents. A tax bill and an education bill might both pass 70-30, yet they're about fundamentally different things. This chapter reads the words.*

---

## Why Text Matters

Volumes 1-6 analyzed the Kansas Legislature entirely through the lens of votes. This was powerful — IRT, clustering, and network analysis extracted a rich picture of ideology and coalition structure from nothing but the pattern of Yeas and Nays. But it was also blind: the analysis didn't know (or care) what the bills were actually about.

That's a significant gap. Two legislators who agree on 85% of votes might agree on very different things — one might cross the aisle on tax issues while the other crosses on social policy. Two bills with identical 70-30 splits might split the chamber along completely different lines. Without knowing *what* each bill addresses, we're left with a map that shows the terrain but not the landmarks.

Topic modeling fills that gap. It reads the text of every bill and assigns it to a policy topic — tax reform, education, criminal justice, healthcare, and so on. Once each bill has a topic label, we can ask new questions: Which topics generate the most partisan conflict? Which topics split the *majority party* internally? Are there policy areas where the usual left-right divide breaks down?

## The BERTopic Pipeline

Tallgrass uses **BERTopic**, a modern topic modeling system created by **Maarten Grootendorst** in 2022. BERTopic is not a single algorithm but a pipeline of four complementary steps. The analogy: imagine you're a librarian who needs to sort 800 unlabeled books into topic sections, but you don't have time to read them all.

### Step 1: Measure the Meaning Fingerprint (Embedding)

Instead of reading each bill word-by-word, BERTopic compresses each bill into a **meaning fingerprint** — a list of 384 numbers that captures its semantic essence. Bills about similar topics get similar fingerprints. Bills about different topics get different fingerprints.

The model that creates these fingerprints is **bge-small-en-v1.5** (Beijing Academy of Artificial Intelligence, 2023), a compact transformer model trained on over a billion text pairs. It was trained to place semantically similar texts near each other in a 384-dimensional space. When it reads "AN ACT concerning taxation; relating to income tax credits for child care," it produces a 384-number fingerprint that lands near other tax-related bills and far from bills about criminal sentencing.

Think of it like a perfume: each bill's "scent" is a blend of 384 base notes, and bills about similar topics smell alike even if they use different words.

**Why this model?** bge-small has 33 million parameters (modest by modern standards) and runs via ONNX Runtime without requiring a GPU or PyTorch installation. It's small enough to run on a laptop but capable enough to rank 51st on the Massive Text Embedding Benchmark (MTEB) — competitive with models 3-4 times its size.

**Codebase:** `analysis/20_bill_text/bill_text.py` (`DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"`)

### Step 2: Reduce Dimensions (UMAP)

384 dimensions is too many for clustering to work well. In high-dimensional spaces, all points tend to be roughly equidistant — a phenomenon called the **curse of dimensionality**. It's like trying to find neighborhoods in a city by measuring distances in 384 directions simultaneously: everything looks the same distance from everything else.

**UMAP** (Uniform Manifold Approximation and Projection), created by **Leland McInnes** in 2018, compresses the 384 dimensions down to 10 while preserving local neighborhood structure. If two bills were close together in 384-D space (similar topics), they'll still be close in 10-D space. If they were far apart, they'll still be far apart.

The analogy: imagine you have a crumpled-up map (384 dimensions). UMAP carefully unfolds it so that nearby cities stay near each other, even though the map is now smaller (10 dimensions). Some global distances might distort, but the local neighborhoods are preserved.

```
Input:  800 bills × 384 dimensions
Output: 800 bills × 10 dimensions
Parameters: n_neighbors=15, min_dist=0.0, metric=cosine
```

Setting `min_dist=0.0` allows clusters to be packed tightly together, which helps the next step (HDBSCAN) find clean boundaries.

**Codebase:** `analysis/20_bill_text/bill_text.py` (`UMAP_N_COMPONENTS = 10`, `UMAP_N_NEIGHBORS = 15`, `UMAP_MIN_DIST = 0.0`)

### Step 3: Find the Clusters (HDBSCAN)

With the bills compressed to 10 dimensions, **HDBSCAN** (the same algorithm from Volume 6, Chapter 1) finds natural groupings. It searches for dense regions of bills — areas where many bills cluster together — separated by sparse gaps.

The key advantage over k-means or other "specify the number of clusters" methods: **HDBSCAN discovers the number of topics automatically.** You don't need to guess whether the legislature is dealing with 5 topics or 25. HDBSCAN examines the data's density structure and decides.

Bills that don't fit any cluster cleanly are labeled as **noise** (topic -1). This is a feature, not a bug — some bills are genuinely idiosyncratic and don't belong to any well-defined topic. A bill about reorganizing the state library system might not group with any of the main policy areas, and that's fine.

```
Parameters: min_cluster_size=15, min_samples=5
```

The `min_cluster_size=15` means a topic must contain at least 15 bills to be recognized — roughly 2% of the typical 800-bill corpus. This prevents tiny, meaningless clusters from appearing.

**Codebase:** `analysis/20_bill_text/bill_text.py` (`MIN_CLUSTER_SIZE = 15`, `MIN_SAMPLES = 5`)

### Step 4: Label the Topics (c-TF-IDF)

HDBSCAN gives us groups of bills, but the groups are just numbered clusters — Topic 0, Topic 1, Topic 2. We need human-readable labels. That's where **c-TF-IDF** (class-based TF-IDF) comes in.

Standard TF-IDF measures how important a word is to a specific document. c-TF-IDF extends this to clusters: it measures how important a word is to *a group of documents* compared to all other groups. If the word "tax" appears frequently in Topic 3 but rarely in other topics, "tax" gets a high c-TF-IDF score for Topic 3. The top-scoring words become the topic's label.

The result: Topic 3 might be labeled **"Tax / Income / Credit / Revenue / Property"** — the five words most distinctive to that cluster.

### The Legislative Stopword Problem

Standard English stopwords ("the," "of," "and") are easy to filter. But legislative text has its own set of meaningless high-frequency words that standard stopword lists don't catch. Without intervention, c-TF-IDF would label topics with words like "shall," "statute," "pursuant," and "therein" — words that appear in nearly every Kansas bill because they're mandatory legal boilerplate, not topical content.

Tallgrass maintains a curated list of 18 **legislative stopwords** that are filtered before c-TF-IDF labeling:

```
Mandatory boilerplate:  shall
Preprocessing artifact: statuteref
Structural markers:     section, subsection, paragraph
Amendatory language:    amendments, amendment, amended, amend
Archaic connectors:     thereto, thereof, therein, herein,
                        hereby, hereof, pursuant, provision, provisions
```

These combine with scikit-learn's 318 standard English stopwords for a total exclusion list of 336 terms.

**Codebase:** `analysis/20_bill_text/bill_text.py` (`LEGISLATIVE_STOPWORDS`, `VECTORIZER_MAX_DF = 0.85`)

## Preprocessing: Cleaning the Legal Language

Before embedding, Tallgrass strips boilerplate that would distort the topic assignments:

1. **Enacting clause** — "Be it enacted by the Legislature of the State of Kansas" appears in every introduced bill. Removed.
2. **Severability clauses** — Formulaic language about what happens if part of the law is struck down. Removed.
3. **Effective date boilerplate** — "This act shall take effect and be in force from and after its publication in the Kansas register." Removed.
4. **K.S.A. references** — Statute citations like "K.S.A. 79-3603" are replaced with a placeholder ("STATUTE_REF") to reduce vocabulary noise while preserving the signal that the bill references existing law.
5. **Section headers** — Simplified to "Section:" to remove numbering artifacts.
6. **Text truncation** — Bills longer than ~8,000 characters (the effective context window of bge-small) are truncated after preprocessing.

### Text Source Preference

Kansas bills come in two forms: the full **introduced text** (dense statutory language with K.S.A. references) and the **supplemental note** (a plain-English summary written by legislative staff). When available, Tallgrass prefers the supplemental note — it's shorter, clearer, and more topically focused. About 60-70% of bills have supplemental notes; the remainder use the introduced text.

**Codebase:** `analysis/20_bill_text/bill_text_data.py` (text loading with source preference)

## CAP Classification: A Standardized Taxonomy

BERTopic discovers topics from the data — it finds whatever structure exists. This is powerful but makes cross-session and cross-state comparisons difficult. If BERTopic discovers 12 topics in the 90th Legislature and 15 in the 91st, are any of them the same?

The **Comparative Agendas Project** (CAP) provides an alternative: a standardized 20-category taxonomy used by political scientists worldwide. Every policy area that a legislature might address falls into one of these categories:

| CAP Code | Category | Kansas Examples |
|----------|----------|----------------|
| Macroeconomics | Budget, taxation, fiscal policy | Income tax cuts, property tax caps |
| Civil Rights | Discrimination, voting, privacy | Voter ID, anti-discrimination ordinances |
| Health | Healthcare, insurance, public health | Medicaid expansion, insurance mandates |
| Education | K-12, higher ed, school finance | School funding formula, charter schools |
| Law & Crime | Criminal justice, sentencing, police | Sentencing reform, concealed carry |
| Agriculture | Farming, rural, water rights | Water appropriation, agricultural subsidies |
| Environment | Conservation, pollution, wildlife | Emissions standards, land management |
| Labor | Employment, wages, unions | Minimum wage, workers' compensation |

The full list includes 20 categories covering the entire range of state policy.

Tallgrass classifies each bill using Claude (the same AI system that powers this analysis) in a **zero-shot classification** task — Claude reads the bill text and assigns the most appropriate CAP category, a confidence score (1-5), and the top 3 most relevant categories. Results are cached by content hash, so each bill is classified once and the result is reused across runs.

CAP classification is optional — it requires an API key and incurs a small cost (~$7-13 for a full session). The phase works fully without it, using only the BERTopic-discovered topics.

**Codebase:** `analysis/20_bill_text/bill_text_classify.py` (Claude Sonnet classification, SHA-256 content caching)

## Caucus-Splitting Analysis: Which Topics Divide the Majority?

The most politically interesting output of topic modeling isn't the topics themselves — it's what happens when you cross-reference topics with voting patterns.

For each topic, Tallgrass computes the **Rice Index** (the party cohesion measure from Volume 6, Chapter 5) separately for each party. A Rice Index of 1.0 means the party was unanimous on bills in that topic. A Rice Index of 0.0 means the party was evenly split.

The **caucus-splitting score** inverts the majority party's Rice Index:

```
Caucus-splitting score = 1 - Rice_Index(majority party)
```

**Plain English:** "How much does this topic divide the majority party internally?"

A score near 0 means the majority party votes as a bloc on bills in that topic — the standard partisan dynamic. A score above 0.30 means the topic generates significant internal dissent within the majority. These are the cross-cutting issues: policy areas where the simple left-right model breaks down because factions within the supermajority disagree.

### A Kansas Example

In the 91st Legislature (2025-2026), the Kansas House Republican caucus holds a comfortable supermajority. On most topics — budget, criminal justice, government operations — the caucus votes together with Rice Indices above 0.80 (caucus-splitting scores below 0.20).

But on topics related to education funding and certain social policy areas, the caucus-splitting score rises above 0.30. Moderate suburban Republicans break from rural conservatives on school finance formulas and certain healthcare votes. These are the same fissures that the clustering analysis (Volume 6, Chapter 1) couldn't find as discrete groups but that the network analysis (Volume 6, Chapter 3) identified through betweenness centrality. Topic modeling adds the missing dimension: it tells you *what* these legislators disagree about, not just *that* they disagree.

**Codebase:** `analysis/20_bill_text/bill_text.py` (Rice Index and caucus-splitting computation per topic)

## Bill Similarity: Finding Duplicates and Companions

A secondary analysis computes **cosine similarity** between all pairs of bill embeddings. Two bills with cosine similarity above 0.80 are remarkably similar in semantic content.

```
similarity(bill_A, bill_B) = (embedding_A · embedding_B) / (||embedding_A|| × ||embedding_B||)
```

**Plain English:** "How similar are the meaning fingerprints of these two bills?"

High-similarity pairs often turn out to be:
- **Companion bills** — a House version and Senate version of the same legislation
- **Competing bills** — multiple proposals addressing the same issue with different approaches
- **Reintroductions** — bills from a previous session introduced again with minor modifications

The top 30 similar pairs are reported in the analysis, with a clustered heatmap showing the similarity structure of the 50 most-connected bills. This provides a visual map of the legislative agenda's redundancy and interconnection.

**Codebase:** `analysis/20_bill_text/bill_text.py` (`SIMILARITY_THRESHOLD = 0.80`, `TOP_SIMILAR_PAIRS = 30`)

## What Can Go Wrong

### Small Corpus Effects

BERTopic works best with hundreds of documents. In chambers with fewer than 100 bills with full text, the density-based clustering may fail to find clean topics. HDBSCAN might assign most bills to noise (topic -1) and produce only 2-3 meaningful topics. The analysis degrades gracefully — it reports what it finds, with honest caveats — but the results are less informative than in larger corpora.

### Supplemental Note Bias

When the analysis uses supplemental notes (which it prefers), the topics reflect the staff summary rather than the actual statutory language. This is usually an advantage (cleaner, more topical text), but it means the analysis captures how legislative staff *describe* the bills, not necessarily what the bills *do* in legal terms.

### Embedding Limitations

The bge-small model was trained on general English, not legal text. It may not capture fine-grained distinctions between different types of statutory amendments or technical regulatory language. BERTopic's c-TF-IDF layer mitigates this by adding domain-specific keyword extraction on top of the general embeddings, but the underlying semantic representation is general-purpose.

---

## Key Takeaway

BERTopic sorts 800 bills into policy topics by measuring semantic fingerprints, reducing dimensions, finding clusters, and extracting keywords. The most interesting downstream analysis is caucus-splitting: which policy topics divide the majority party internally? These cross-cutting issues reveal the fissures that clustering and network analysis detected but couldn't explain. The optional CAP classification provides a standardized 20-category taxonomy for cross-session comparison.

---

*Terms introduced: BERTopic, embedding, bge-small-en-v1.5, UMAP (Uniform Manifold Approximation and Projection), curse of dimensionality, HDBSCAN (in topic modeling context), c-TF-IDF (class-based TF-IDF), legislative stopwords, supplemental note, Comparative Agendas Project (CAP), zero-shot classification, caucus-splitting score, cosine similarity, companion bills*

*Previous: [Bill Passage: Can We Forecast Outcomes?](ch02-bill-passage.md)*

*Next: [Text-Based Ideology and Issue-Specific Scores](ch04-text-ideology.md)*
