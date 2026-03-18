# Volume 7 — Prediction and Text

> *Can we predict how a legislator will vote? What do the words of a bill tell us about its politics?*

---

Volumes 1-6 measured ideology, validated the measurements, and mapped the coalition structure of the Kansas Legislature. This volume asks a different kind of question: **can we predict what happens next?**

Prediction is a stricter test than measurement. A model that measures ideology after the fact is explaining past behavior. A model that predicts a vote before it happens is proving it understands the underlying mechanics well enough to anticipate outcomes. The difference matters. An explanation can be wrong in ways that are hard to detect (it might just be overfitting to noise). A prediction either works or it doesn't.

This volume also adds something entirely new: **the words of the bills themselves**. Volumes 1-6 treated every bill as a binary event — Yea or Nay, pass or fail. But a bill is also a document, with language that carries meaning. A tax cut bill and an education bill might both pass 70-30, but they're about fundamentally different things. By analyzing the text, we can ask questions that votes alone can't answer: What are the bills about? Do legislators who support similar bills share similar language? And when a Kansas bill reads suspiciously like a template from a national lobbying organization, can we detect that?

---

## Chapters

| # | Title | What You'll Learn |
|---|-------|-------------------|
| 1 | [Predicting Votes: What XGBoost Learns](ch01-predicting-votes.md) | How a prediction model achieves 0.98 AUC using ideology, bill parameters, and network position — and why that number is both impressive and misleading |
| 2 | [Bill Passage: Can We Forecast Outcomes?](ch02-bill-passage.md) | A harder prediction task with fewer features, smaller samples, and an honest test of session-wide generalization |
| 3 | [Topic Modeling: What Are the Bills About?](ch03-topic-modeling.md) | How BERTopic sorts 800 bills into policy topics using meaning fingerprints, and which topics split the majority party |
| 4 | [Text-Based Ideology and Issue-Specific Scores](ch04-text-ideology.md) | Deriving ideology from bill content, running separate IRT models per policy topic, and discovering which issues cross-cut the party divide |
| 5 | [Model Legislation: Detecting Copy-Paste Policy](ch05-model-legislation.md) | Using embedding similarity and n-gram overlap to find Kansas bills that match ALEC templates or neighboring states' legislation |

---

## Key Terms Introduced in This Volume

| Term | Definition |
|------|-----------|
| XGBoost | A gradient-boosted decision tree algorithm — the workhorse of tabular prediction competitions |
| Gradient boosting | An ensemble strategy where each new model corrects the errors of the previous one |
| AUC (Area Under the ROC Curve) | A metric that measures how well a model ranks positive examples above negative ones, regardless of threshold |
| ROC curve | A plot of true positive rate vs. false positive rate at every classification threshold |
| SHAP (SHapley Additive exPlanations) | A method from game theory that attributes each prediction to the contribution of each input feature |
| Calibration | Whether a model's predicted probabilities match the actual frequencies of outcomes |
| Brier score | The mean squared error of predicted probabilities — a single number that captures both discrimination and calibration |
| Temporal validation | Testing a model by training on the first part of a time series and testing on the last part |
| NMF (Non-negative Matrix Factorization) | A method that decomposes a document-term matrix into topics, where all weights are non-negative |
| BERTopic | A topic modeling pipeline that combines transformer embeddings, UMAP, HDBSCAN, and c-TF-IDF |
| Embedding | A vector representation of text where similar meanings map to nearby points in high-dimensional space |
| UMAP | A dimensionality reduction method that preserves local neighborhood structure |
| HDBSCAN | A density-based clustering algorithm that finds groups of varying sizes and can label outliers as noise |
| c-TF-IDF | Class-based Term Frequency-Inverse Document Frequency — a variant of TF-IDF that extracts representative words per topic cluster |
| Comparative Agendas Project (CAP) | A standardized 20-category policy taxonomy used in political science for cross-country and cross-time comparisons |
| Legislative stopwords | Domain-specific words ("shall," "pursuant," "thereto") that appear in nearly every bill and carry no topical information |
| Rice Index (topic-level) | The party cohesion measure from Volume 6, applied per policy topic to identify which issues split the majority party |
| Caucus-splitting score | 1 minus the majority party's Rice Index on a topic — higher scores mean more internal party disagreement |
| Text-based ideal point | An ideology score derived from the content of bills a legislator supports, rather than from the vote pattern alone |
| Vote-weighted embedding profile | A legislator's average bill embedding, weighted by their votes (+1 Yea, -1 Nay) |
| Issue-specific IRT | A separate IRT model fit to votes on bills within a single policy topic |
| Cross-topic correlation | The Pearson correlation between two sets of issue-specific ideal points — revealing which policy areas sort legislators the same way |
| Cosine similarity | A measure of the angle between two vectors — 1.0 means identical direction, 0.0 means unrelated |
| N-gram overlap | The fraction of consecutive word sequences (n-grams) shared between two texts — a measure of direct text reuse |
| ALEC (American Legislative Exchange Council) | An organization that produces model legislation adopted by state legislatures across the country |
| Policy diffusion | The process by which legislation spreads from one state to another |
| Near-identical match | Cosine similarity >= 0.95 between a Kansas bill and a reference text |
| Strong match | Cosine similarity >= 0.85 — adapted from the same source but with meaningful modifications |
| Related match | Cosine similarity >= 0.70 — same policy domain, warrants qualitative investigation |

---

*Previous: [Volume 6 — Finding Patterns](../volume-06-finding-patterns/)*

*Next: [Volume 8 — Change Over Time](../volume-08-change-over-time/)*
