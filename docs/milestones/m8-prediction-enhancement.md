# M8: Prediction Enhancement

Enhance Phase 8 bill passage prediction with sponsor party features, SHAP for passage model, stratified accuracy, and feature importance visualization.

**Roadmap item:** R26 (bill outcome prediction model enhancement)
**Estimated effort:** 1-2 sessions
**Dependencies:** None

---

## Current State

Phase 8 (`analysis/08_prediction/prediction.py`) already has two prediction tasks:

1. **Vote prediction** — per-legislator vote (Yea/Nay) using IRT ideal points, party, network centrality, PCA scores. AUC ~0.98.
2. **Bill passage prediction** — per-bill outcome (passed/failed) using bill discrimination, vote type, bill prefix, optional NLP topics.

### Existing Bill Passage Features (`build_bill_features()`, line 411)

| Feature | Type | Source |
|---------|------|--------|
| `beta_mean` | float | IRT bill discrimination |
| `is_veto_override` | bool | Vote type |
| `day_of_session` | int | Temporal position |
| `vt_*` | bool (one-hot) | Vote type (motion category) |
| `pfx_*` | bool (one-hot) | Bill prefix (SB, HB, SCR, etc.) |
| `topic_*` | float (optional) | NLP topic features |

**Excluded (leakage):** `alpha_mean` (bill difficulty encodes outcome), vote counts.

### Existing Training (`train_passage_models()`, line 648)

- Same 3 models as vote prediction: Logistic Regression, XGBoost, Random Forest
- 5-fold stratified CV + 20% holdout
- Same metrics: accuracy, AUC, precision, recall, F1, Brier, log-loss
- Baselines: majority class, historical passage rate

---

## Enhancements

### 1. Sponsor Party Feature

**Problem:** The sponsor's party affiliation is an obvious predictor of passage in a supermajority legislature but is not currently included.

**Data path:**
- `rollcalls.csv` contains `sponsor` column (legislator full name)
- `legislators.csv` contains `full_name` and `party` columns
- Match sponsor name to legislator record to get party

```python
# In build_bill_features(), after existing feature construction:
def _add_sponsor_features(
    bill_features: pl.DataFrame,
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
) -> pl.DataFrame:
    """Add sponsor party as a bill-level feature.

    Matches rollcalls.sponsor to legislators.full_name to get party.
    Bills with unmatched or missing sponsors get sponsor_party_R = 0.
    """
    # Build sponsor → party lookup
    sponsor_party = (
        legislators
        .select("full_name", "party")
        .unique(subset=["full_name"])
        .rename({"full_name": "sponsor"})
    )

    # Join on sponsor name
    enriched = (
        bill_features
        .join(
            rollcalls.select("vote_id", "sponsor").unique(subset=["vote_id"]),
            on="vote_id",
            how="left",
        )
        .join(sponsor_party, on="sponsor", how="left")
        .with_columns(
            (pl.col("party") == "Republican").cast(pl.Int8).fill_null(0).alias("sponsor_party_R"),
        )
        .drop("sponsor", "party")
    )
    return enriched
```

**Note:** Some bills have multiple sponsors (semicolon-separated in the CSV). Use the first sponsor for the party feature. Bills without a matching legislator (e.g., committee-introduced bills) get `sponsor_party_R = 0`.

### 2. SHAP Beeswarm for Passage Model

**Problem:** Vote prediction already has SHAP beeswarm plots, but the passage model does not. Adding SHAP shows which features drive passage predictions.

**Implementation:** The existing SHAP infrastructure in `prediction.py` (`_compute_shap_values()`) already works with any sklearn/XGBoost model. Just call it for the passage model too.

```python
# After training passage models:
if passage_best_model is not None:
    shap_values = _compute_shap_values(passage_best_model, X_test_passage)
    _plot_shap_beeswarm(shap_values, X_test_passage, out_path=plots_dir / f"shap_passage_{chamber}.png")
    _plot_shap_bar(shap_values, out_path=plots_dir / f"shap_bar_passage_{chamber}.png")
```

### 3. Stratified Accuracy by Bill Prefix

**Problem:** Bill passage rates vary dramatically by prefix (HB, SB, HCR, SCR, etc.). Overall accuracy masks potential weaknesses on minority categories.

```python
def compute_stratified_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bill_prefixes: list[str],
) -> pl.DataFrame:
    """Compute accuracy, count, and passage rate per bill prefix.

    Returns DataFrame with columns: prefix, count, accuracy, passage_rate.
    """
    df = pl.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "prefix": bill_prefixes,
    })
    return (
        df.group_by("prefix")
        .agg(
            pl.len().alias("count"),
            (pl.col("y_true") == pl.col("y_pred")).mean().alias("accuracy"),
            pl.col("y_true").mean().alias("passage_rate"),
        )
        .sort("count", descending=True)
    )
```

### 4. Feature Importance Visualization

Add a horizontal bar chart of feature importance (from the best passage model) to complement SHAP:

```python
def plot_passage_feature_importance(
    model,
    feature_names: list[str],
    out_path: Path,
) -> None:
    """Horizontal bar chart of passage model feature importances."""
    import matplotlib.pyplot as plt

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    # Sort by importance
    indices = np.argsort(importances)[-15:]  # top 15
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title("Bill Passage Prediction — Top Features")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
```

---

## Report Updates

In `analysis/08_prediction/prediction_report.py`, the bill passage section starts at line ~108. Add new sections after the existing passage model table/ROC:

```python
# After existing passage sections:

# SHAP beeswarm for passage
_add_figure_if_exists(report, plots_dir, f"shap_passage_{chamber}.png",
                      f"shap-passage-{chamber.lower()}",
                      f"SHAP Beeswarm — {chamber} Bill Passage")

# SHAP bar for passage
_add_figure_if_exists(report, plots_dir, f"shap_bar_passage_{chamber}.png",
                      f"shap-bar-passage-{chamber.lower()}",
                      f"SHAP Feature Importance — {chamber} Bill Passage")

# Feature importance
_add_figure_if_exists(report, plots_dir, f"passage_importance_{chamber}.png",
                      f"passage-importance-{chamber.lower()}",
                      f"Feature Importance — {chamber} Bill Passage")

# Stratified accuracy table
stratified_path = plots_dir.parent / f"stratified_accuracy_{chamber}.parquet"
if stratified_path.exists():
    stratified = pl.read_parquet(stratified_path)
    report.add(InteractiveTableSection(
        id=f"stratified-accuracy-{chamber.lower()}",
        title=f"Accuracy by Bill Type — {chamber}",
        df=stratified,
        caption="Passage prediction accuracy stratified by bill prefix. "
                "Low-count categories may have unreliable accuracy estimates.",
    ))
```

---

## Tests

Add to `tests/test_prediction.py`:

```python
class TestSponsorFeature:
    def test_adds_sponsor_party_column(self):
        """sponsor_party_R column is added to bill features."""
        # Build minimal rollcalls with sponsor names
        # Build minimal legislators with matching names
        # Verify column exists and has correct values

    def test_missing_sponsor_defaults_to_zero(self):
        """Bills with no matched sponsor get sponsor_party_R = 0."""

    def test_multi_sponsor_uses_first(self):
        """Semicolon-separated sponsors use the first name."""


class TestStratifiedAccuracy:
    def test_all_prefixes_present(self):
        """Each unique prefix appears in output."""
        y_true = np.array([1, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 1])
        prefixes = ["HB", "HB", "SB", "SB"]
        result = compute_stratified_accuracy(y_true, y_pred, prefixes)
        assert result.height == 2
        assert set(result["prefix"].to_list()) == {"HB", "SB"}

    def test_accuracy_correct(self):
        """Accuracy is computed correctly per prefix."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        prefixes = ["HB", "HB", "HB", "SB"]
        result = compute_stratified_accuracy(y_true, y_pred, prefixes)
        hb_row = result.filter(pl.col("prefix") == "HB")
        assert abs(hb_row["accuracy"][0] - 2 / 3) < 0.01


class TestPassageFeatureImportance:
    def test_plot_created(self, tmp_path):
        """Feature importance plot is created for XGBoost model."""
        # Train a minimal XGBoost model
        # Call plot_passage_feature_importance
        # Verify PNG file exists
```

---

## Key Considerations

### Sample Size

Bill passage prediction operates on ~250-500 bills per chamber per biennium. The existing `MIN_VOTES_RELIABLE=10` threshold (ADR-0076 fix A7) doesn't apply to bill-level prediction but similar caution is warranted. The stratified accuracy table will naturally expose low-count prefixes.

### Leakage Prevention

The sponsor party feature does **not** introduce leakage:
- Sponsor identity is known at bill introduction (before any vote occurs)
- Unlike `alpha_mean` (IRT difficulty), sponsor party is exogenous to the outcome
- This is comparable to the existing `pfx_*` prefix features (known at introduction)

### Existing Audit Findings

- **A5 (in-sample surprising bills):** Already fixed — evaluates on holdout test set only
- **A7 (per-legislator min threshold):** Already fixed — `MIN_VOTES_RELIABLE=10`
- No additional leakage concerns from these enhancements

---

## Verification

```bash
just test -k "test_prediction" -v     # existing + new tests pass
just lint-check                       # formatting
just pipeline 2025-26                 # full pipeline run
# Open prediction report, verify: SHAP passage plots, stratified table, sponsor feature
```

## Documentation

- Update `docs/roadmap.md` item R26 to "Done"
- No ADR needed for Phase 08 changes (enhancement to existing phase, no architectural decision)

### Downstream Integration (ADR-0081)

The `sponsor_slugs` column introduced by M8 is also consumed by:

- **Phase 11 (Synthesis):** `_compute_sponsor_summary()` splits `sponsor_slugs`, computes `n_bills_sponsored` and `sponsor_passage_rate` per legislator, LEFT JOINs onto `leg_dfs`. Displayed as "Bills Sponsored" in the unified scorecard.
- **Phase 12 (Profiles):** `compute_sponsorship_stats()` identifies bills where a target legislator appears in the sponsor list. Marks primary vs co-sponsor. Report section shows sponsored bills table with passage rate. Defection tables include `sponsor` column for context.

All downstream consumers degrade gracefully when `sponsor_slugs` is absent (pre-89th data, committee sponsors).

## Commit

```
feat(infra): enhance bill passage prediction — sponsor party, SHAP, stratified accuracy [vYYYY.MM.DD.N]
```
