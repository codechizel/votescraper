# EDA Design Choices

**Script:** `analysis/01_eda/eda.py`
**Constants defined at:** `analysis/01_eda/eda.py:149-168`

## Assumptions

1. **Two-party system.** All analysis assumes exactly Republican and Democrat parties. Independents, if any, would be unhandled. Kansas currently has none.

2. **Yea/Nay binary is sufficient.** "Present and Passing" (~22 instances, almost all Senate) is treated as a non-vote — excluded from the binary matrix entirely. This is a deliberate abstention, not a missing data point and not a Nay. Downstream models never see it.

3. **Absences are uninformative.** "Absent and Not Voting" and "Not Voting" become null in the binary matrix. The implicit assumption is that absence does not systematically correlate with ideology. This is probably wrong for strategic absences but is the standard assumption in the literature.

4. **Chambers are independent.** House and Senate are filtered and analyzed separately because they vote on different bills. Mixing them creates a block-diagonal matrix of nulls that distorts all pairwise measures.

5. **Mid-session replacements are valid legislators.** The EDA flags them but does not remove them. Both the original and replacement get their own row in the vote matrix. This means some districts temporarily have "two senators" in the data.

## Parameters & Constants

| Constant | Value | Justification | Code location |
|----------|-------|---------------|---------------|
| `CONTESTED_THRESHOLD` | 0.025 (2.5%) | VoteView standard (defined in `analysis/tuning.py`). Votes where the minority side is < 2.5% carry no ideological signal. | `eda.py:153` |
| `MIN_VOTES` | 20 | Legislators with < 20 substantive votes produce unreliable ideal-point estimates. Threshold chosen to retain mid-session replacements with partial records while excluding empty-seat placeholders. | `eda.py:154` |
| `VOTE_CATEGORIES` | 5 categories | Yea, Nay, Present and Passing, Absent and Not Voting, Not Voting. Exactly these five; anything else triggers an integrity warning. | `eda.py:155-157` |
| `MIN_SHARED_VOTES` | 10 | Minimum shared votes between two legislators to compute pairwise agreement. Below this, agreement is too noisy to be meaningful. | `eda.py:168` |
| `HOUSE_SEATS` / `SENATE_SEATS` | 125 / 40 | Constitutional seat counts. Used as integrity guardrails — exceeding these indicates mid-session replacements or scraping bugs. | `eda.py:163-164` |
| `RICE_BOOTSTRAP_ITERATIONS` | 100 | Iterations for Desposato small-party Rice correction bootstrap. | `eda.py:169` |
| `STRATEGIC_ABSENCE_RATIO` | 2.0 | Threshold for flagging strategic absences (absence on party-line votes / overall absence). | `eda.py:170` |
| `ITEM_TOTAL_CORRELATION_THRESHOLD` | 0.1 | Point-biserial correlation below which a roll call is flagged as non-discriminating. | `eda.py:171` |

## Methodological Choices

### Binary encoding: Yea=1, Nay=0, else=null

**Decision:** Only Yea and Nay are encoded as binary values. All other categories (Present and Passing, Absent and Not Voting, Not Voting) become null.

**Alternatives considered:**
- Encode "Present and Passing" as 0.5 (between Yea and Nay) — rejected because PCA and IRT expect binary inputs
- Encode absences as a third category — rejected because it would require multinomial models, not binary IRT
- Encode "Present and Passing" as Nay (it doesn't help the bill pass) — rejected because the legislator deliberately chose not to oppose

**Impact:** ~22 "Present and Passing" votes are invisible to all downstream models. This is a tiny fraction (<0.03%) and will not meaningfully affect results.

### Filtering order: unanimous votes first, then low-participation legislators

**Decision:** Two sequential filters applied per chamber:
1. Drop votes where minority side < `CONTESTED_THRESHOLD` (2.5%)
2. Drop legislators with < `MIN_VOTES` (20) substantive votes on the remaining contested bills

**Alternatives considered:**
- Filter legislators first, then votes — rejected because it could retain votes that became unanimous after removing low-participation legislators
- Joint filtering (iterate until stable) — rejected as unnecessary; the current order is stable in practice

**Impact:** This order means a legislator's vote count is measured against contested bills only. A legislator who voted on 100 unanimous bills and 15 contested bills gets dropped. This is intentional — unanimous votes don't contribute information.

### Agreement metric: Cohen's Kappa over raw agreement

**Decision:** Both raw agreement and Cohen's Kappa are computed, but Kappa is the preferred metric for downstream use (clustering, network analysis).

**Why:** The ~82% Yea base rate in Kansas means two random legislators would agree ~70% of the time just by both voting Yea on most bills. Raw agreement inflates apparent similarity. Kappa corrects for this chance agreement: K = (observed - expected) / (1 - expected).

**Impact:** Kappa values are lower and have more spread than raw agreement. A Kappa of 0.5 is actually strong agreement; 0.0 is chance-level. Downstream clustering should use Kappa, not raw agreement.

### Additional Diagnostics (Added 2026-02-24)

Five functions were added based on the literature review in `docs/eda-deep-dive.md`:

| Function | Reference | Output |
|----------|-----------|--------|
| `compute_party_unity_scores()` | Carey Legislative Voting Data Project | Per-legislator loyalty rate on party-line votes. Saved to `party_unity_{chamber}.parquet`. |
| `compute_eigenvalue_preview()` | Standard PCA pre-check | Top 5 eigenvalues + lambda1/lambda2 ratio. Saved to manifest. |
| `compute_strategic_absence()` | Rosas & Shomer 2008 | Per-legislator absence rate on party-line vs all votes. Saved to `strategic_absence_{chamber}.parquet`. |
| `compute_desposato_rice_correction()` | Desposato 2005 | Bootstrap-corrected Rice for cross-party comparison. Saved to manifest. |
| `compute_item_total_correlations()` | Classical psychometrics | Point-biserial correlation per roll call. Saved to `item_total_{chamber}.parquet`. |

All five are purely additive diagnostics — they do not change the vote matrices, filtering, or any downstream data. They run between party-line classification and the agreement matrix computation in `main()`.

The `numpy_matrix_to_polars()` helper was extracted from `main()` to module level for testability.

### Party-line classification: 90% threshold

**Decision:** A vote is "party-line" if both parties have > 90% of their members voting in opposite directions. "Bipartisan" if both > 90% in the same direction. "Mixed" otherwise.

**Impact:** This is a descriptive statistic only — not used by downstream models.

### Hierarchical clustering: Ward linkage

**Decision:** Agreement heatmaps use Ward linkage (minimizes within-cluster variance) with Euclidean distance.

**Alternatives:** Complete linkage, average linkage, single linkage. Ward was chosen because it produces the most balanced dendrograms and best reveals block structure.

**Impact:** The heatmap ordering is purely visual. Downstream clustering phases should choose their own linkage method.

## Downstream Implications

### For PCA (Phase 2)
- PCA reads `vote_matrix_{house,senate}_filtered.parquet` — the filtered matrices this phase produces.
- **Nulls remain in the filtered matrices.** PCA must impute them (row-mean is the PCA default). The null fraction is typically 15-30% per chamber.
- The 2.5% contested threshold has already removed near-unanimous votes. PCA should not re-filter, but the sensitivity analysis re-filters at 10% from the full matrix.

### For IRT (Phase 3)
- IRT also reads the filtered matrices but handles nulls natively (absences absent from the likelihood — no imputation needed).
- **The binary encoding (Yea=1, Nay=0) is the IRT input.** "Present and Passing" votes are already gone.
- Legislators near the `MIN_VOTES` = 20 threshold will have wide credible intervals. IRT handles this gracefully but the uncertainty should be reported.
- Mid-session replacements (e.g., Sen. Miller) may appear in both chambers if they served in the House previously. The EDA does not cross-reference — IRT should treat them as independent per-chamber observations unless explicitly building a joint model.

### For Clustering (Phase 5) and Network (Phase 6)
- Use **Kappa matrices**, not raw agreement matrices, as input.
- The `MIN_SHARED_VOTES` = 10 threshold means some pairs have NaN in the agreement matrix. Handle before clustering.
