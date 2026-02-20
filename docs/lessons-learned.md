# Lessons Learned

Hard-won debugging insights from building this scraper. These are real bugs that cost time to diagnose — don't repeat them.

---

## Bug 1: The h2/h3/h4 Tag Mismatch

**Symptom**: `vote_date`, `chamber`, `motion`, and `bill_title` were ALL empty strings in every row of every output CSV. Vote counts and legislator data worked fine.

**Root cause**: The parser searched `<h2>` tags for vote metadata, but the actual HTML uses a non-obvious tag hierarchy:

```
<h2>  → bill number (e.g., "SB 1")
<h4>  → bill title (e.g., "AN ACT exempting the state of Kansas...")
<h3>  → chamber/motion/date (e.g., "Senate - Emergency Final Action - Passed as amended; - 03/20/2025")
<h3>  → vote category headings (e.g., "Yea - (33):")
```

You'd expect the title to be in `<h2>` or `<h3>`, not `<h4>`. And you'd expect chamber/date/motion to be in a higher-level heading than the vote categories, but they're both `<h3>`.

**Fix**: Search `<h4>` for bill title, `<h3>` for chamber/date/motion.

**Lesson**: Always inspect the actual HTML. Don't assume heading hierarchy follows semantic conventions. The KS Legislature website was likely built with a CMS that doesn't enforce consistent heading levels.

---

## Bug 2: Every Legislator Tagged as Republican

**Symptom**: All 172 legislators in the output had `party = "Republican"`, including known Democrats.

**Root cause**: The code did `re.search(r"\bRepublican\b", page.get_text())` on each legislator's member page. But every member page contains a party filter dropdown:

```html
<select>
  <option value="republican">Republican</option>
  <option value="democrat">Democrat</option>
</select>
```

Since `page.get_text()` flattens all text including form elements, "Republican" appears on every page before the actual party info. The regex always matched Republican first.

**Fix**: Instead of searching full page text, target the specific `<h2>` that contains the district info: `<h2>District 27 - Republican</h2>`. Use `soup.find("h2", string=re.compile(r"District\s+\d+"))`.

**Lesson**: Never search full page text for structured data. Form elements, navigation, filters, and footers all contain text that can create false positives. Always target the most specific containing element.

---

## Bug 3: KLISS API Response Format Inconsistency

**Symptom**: API pre-filtering occasionally returned zero bills with votes, falling back to the full scan (slower but functional).

**Root cause**: The API sometimes returns a raw JSON array `[{...}, {...}]` and sometimes wraps it in `{"content": [{...}, {...}]}`. Code that only handles one format silently gets an empty list.

**Fix**: `content = data if isinstance(data, list) else data.get("content", [])`.

**Lesson**: Government APIs are not versioned or documented consistently. Always handle multiple response shapes and fail gracefully.

---

## Design Decision: Why bill_metadata Exists

The KLISS API is already called for pre-filtering (to avoid fetching ~800 bill pages that have no votes). This same API response contains `SHORTTITLE` and `ORIGINAL_SPONSOR` — valuable enrichment data that would otherwise require parsing unreliable HTML.

Rather than discard this data and fetch it again later, `_filter_bills_with_votes()` returns both the filtered URLs and a metadata dict. This adds zero HTTP requests and provides cleaner data than HTML parsing would.

---

## Design Decision: Two-Phase Fetch/Parse

The scraper never mutates shared state during concurrent fetches. The pattern is always:

1. **Fetch phase** (concurrent): `ThreadPoolExecutor` fetches URLs → returns `dict[url, html]`
2. **Parse phase** (sequential): Single-threaded loop processes HTML, mutates `self.rollcalls`, `self.individual_votes`, `self.legislators`

This avoids race conditions without locks on the data structures. The rate-limiting lock only protects the HTTP request timing, not the parsed data.

---

## Design Decision: vote_datetime from vote_id

The vote_id string (e.g., `je_20250320203513_2584`) encodes a precise timestamp. Extracting this gives sub-second vote ordering and a robust deduplication key (`vote_datetime` + `legislator_slug`), which is more reliable than the human-readable `vote_date` (MM/DD/YYYY, no time component).

---

## Gotcha: Cache Key Collisions

Cache filenames are constructed by replacing `/`, `:`, `?` in URLs with `_`, then truncating to 200 chars. For very long URLs, truncation could theoretically cause collisions. In practice this hasn't been an issue because KS Legislature URLs are well under 200 chars, but it's worth knowing about.

---

## Gotcha: session.py CURRENT_BIENNIUM_START

This constant must be manually updated when the KS Legislature starts a new biennium (odd years, e.g., 2027). If it's wrong, the "current" session will use historical URL prefixes, which may or may not work depending on when the legislature updates their site.

---

## Gotcha: Special Session URL Patterns

Special sessions use a completely different URL scheme (`/li_2024s/` instead of `/li_2024/b2023_24/`). They also don't have a biennium code. New special sessions must be manually added to `SPECIAL_SESSION_YEARS` in session.py.

---

## Insight 1: Legislator Count ≠ Seat Count (Mid-Session Replacements)

**Discovered during**: EDA Phase 1 (2026-02)

**Symptom**: The 2025-26 session data contains 130 House legislators and 42 Senate legislators, but the Kansas House has exactly 125 seats and the Senate has exactly 40 seats.

**Root cause**: Mid-session resignations and replacements. Legislators who resign or are appointed to another office are replaced by new members. Both the outgoing and incoming members appear in the data with non-overlapping vote records. In the 2025-26 session:

- **5 House replacements** (Districts 5, 33, 70, 85, 86): Original members served 2025-01-30 → 2025-04-11 (~333 votes), replacements started 2026-01-15 (~151 votes)
- **2 Senate replacements** (Districts 24, 25): Original members served through 2025-04-11, replacements started 2026-01-27

Two legislators (Scott Hill and Silas Miller) appear in both chambers — they left the House and were elected/appointed to the Senate.

**Validation**: Service windows were confirmed non-overlapping. No duplicate votes (same legislator + same rollcall) were found. The data is correct.

**Lesson**: Always validate legislator counts against known chamber sizes. The Kansas House has **125** seats and the Senate has **40** seats — these are constitutional constants. Any count above those numbers signals mid-session turnover and requires verifying that service windows don't overlap (which would indicate a scraping bug). Downstream analysis must account for partial-session legislators: their participation rates will be low by design, and ideal point estimates will be less stable.

---

## Bug 4: "Not passed" Classified as Passed

**Discovered during**: EDA report review (2026-02-19)

**Symptom**: The EDA report showed 100% passage rate for Final Action votes. Manual inspection of the motion text revealed 4 rollcalls with "Not passed" or "Not  passed" (double space) in the motion, all incorrectly marked `passed=True`.

**Root cause**: `_derive_passed()` checked the positive regex `\b(passed)\b` before the negative patterns. Since "Not passed" contains the word "passed", the positive match fired first and returned `True`.

**Fix**: Reorder the checks — test failure patterns (`\b(not\s+passed|failed|rejected)\b`) first, then positive patterns.

**Impact**: 4 rollcalls (1 Final Action, 3 Emergency Final Action) had `passed=True` when they should have been `False`. This corrupted passage rate statistics. All individual vote records for these rollcalls were unaffected (votes were correctly counted), only the rollcall-level `passed` boolean was wrong.

**Lesson**: When regex-matching result text, always check negations first. String containment is not semantic — "not passed" *contains* "passed". This is a textbook order-of-evaluation bug.

---

## Bug 5: get_text(strip=True) Drops Spaces Around Inline Tags

**Discovered during**: Scraper code review (2026-02-19)

**Symptom**: 53 motions in the rollcalls CSV had mangled text like `"Amendment bySenator Franciscowas rejected"` instead of `"Amendment by Senator Francisco was rejected"`.

**Root cause**: BeautifulSoup's `get_text(strip=True)` strips whitespace from each text node individually, then concatenates them **without any separator**. When an `<h3>` tag contains inline `<a>` elements (as Committee of the Whole amendment motions do), the spaces between text nodes and the `<a>` tag are lost:

```html
<h3>... Amendment by <a href="..."> Senator Francisco</a> was rejected ...</h3>
```

Text nodes: `"Amendment by "`, `" Senator Francisco"`, `" was rejected"` → after strip each → `"Amendment by"`, `"Senator Francisco"`, `"was rejected"` → concatenated → `"Amendment bySenator Franciscowas rejected"`.

**Fix**: Use `get_text(separator=" ", strip=True)` which inserts a space between text nodes, then collapse multiple spaces with `" ".join(text.split())`. Extracted as a `_clean_text()` helper.

**Impact**: All 53 affected motions were Committee of the Whole amendment votes. The mangling was cosmetic — vote_type and passed classification still worked because the prefix "Committee of the Whole" and keyword "rejected" were in the non-mangled parts. But the data was ugly and would cause issues for text-based searches.

**Lesson**: `get_text(strip=True)` is almost never what you want for elements with inline children. Always use `separator=" "` when the element might contain `<a>`, `<span>`, `<em>`, or other inline tags.

---

## Gotcha: Legislative Day vs Wall Clock Time

**Discovered during**: Scraper code review (2026-02-19)

**Symptom**: 3 rollcalls have `vote_datetime` (from the vote_id timestamp) on March 18 but `vote_date` (from the h3 on the page) on March 17.

**Root cause**: These are late-night House votes where the legislative session extended past midnight. The legislature considers it still the "March 17 session" even though the clock says March 18. The h3 date reflects the legislative day; the vote_id timestamp reflects the wall clock.

**Not a bug** — both values are correct for their respective meanings. `vote_date` is the authoritative legislative day. `vote_datetime` is the precise wall clock time for ordering.

**Lesson**: Legislative bodies operate on "legislative days" that don't always align with calendar days. When votes extend past midnight, the session date stays the same. Our data correctly preserves both values.

---

## Gotcha: Null bill_title on Amendment Vote Pages

**Discovered during**: Scraper code review (2026-02-19)

15 rollcalls (all Senate Committee of the Whole amendment votes) have null `bill_title` because their vote pages have no `<h4>` tag and the `<h2>` contains only `[`. These pages simply don't include the bill title — it's a quirk of how the KS Legislature website renders amendment-specific vote pages.

`bill_number` is always present and can be used to look up the title from other rollcalls for the same bill. This is a cosmetic gap, not a data integrity issue.

---

## Lesson 6: Positive-Constrained β Prior Silences Half the Ideological Signal

**Discovered during:** IRT audit (2026-02-20)

**Symptom:** The IRT model's beta (discrimination) distribution was bimodal: a cluster at β ≈ 7 (party-line R-Yea bills) and a cluster at β ≈ 0.17 (all other contested bills). All 37 bills where Democrats voted Yea had β < 0.39 — the model treated them as uninformative noise.

**Root cause:** The LogNormal(0.5, 0.5) prior constrains β > 0. With `P(Yea) = logit⁻¹(β·ξ - α)` and β > 0, the probability of voting Yea always increases with ξ (conservatism). Bills where Democrats vote Yea need β < 0 to be modeled correctly. The positive constraint makes this impossible, so the model assigns near-zero discrimination.

The design doc incorrectly claimed "alpha handles directionality." Alpha shifts the probability curve up/down (threshold) but cannot flip its slope (direction). Only the sign of β controls direction.

**Why the standard recommendation was wrong for us:** The LogNormal prior is a standard recommendation for IRT models using *soft identification* (priors only). Our model uses *hard anchors* (two legislators fixed at ξ = ±1), which already solve the sign-switching problem. The positive constraint was redundant and actively harmful — it solved a problem the anchors had already solved, at the cost of discarding 12.5% of bills.

**Fix:** Switch to `pm.Normal("beta", mu=0, sigma=1)`. Experiment results (500/300 draws):

| Metric | LogNormal(0.5,0.5) | Normal(0,1) |
|---|---|---|
| Holdout accuracy | 90.8% | **94.3%** (+3.5%) |
| Holdout AUC-ROC | 0.954 | **0.979** |
| D-Yea \|β\| mean | 0.186 | **2.384** |
| ξ ESS min | 21 | **203** (10× better) |
| Divergences | 0 | 0 |
| PCA Pearson r | 0.950 | **0.972** |

Zero sign-switching. Zero divergences. Better on every metric.

**Lesson:** When using hard anchors in Bayesian IRT, an unconstrained Normal prior on discrimination is both simpler and better than a positive-constrained LogNormal. The anchors provide identification; the prior should provide regularization, not constraints. Don't blindly follow "standard" priors without checking whether their assumptions (soft identification) match your model (hard identification).

**See also:** `analysis/design/beta_prior_investigation.md` for the full investigation, experiment protocol, and plots.

---

## Bug 7: Empty DataFrame Indexing in Party Loyalty Summary

**Discovered during:** Clustering code review and test writing (2026-02-20)

**Symptom:** `compute_party_loyalty()` in `analysis/clustering.py` crashes with `IndexError: index 0 is out of bounds for sequence of length 0` when there are no contested votes.

**Root cause:** After computing party loyalty, the function prints the lowest and highest loyalty legislators by indexing into a sorted DataFrame: `sorted_loyalty['full_name'][0]`. When all votes are unanimous (no contested votes), the loyalty DataFrame is empty (0 rows), and indexing position 0 raises `IndexError`.

**Why this matters:** The bug never triggered in production because real Kansas Legislature data always has contested votes. But it's a latent crash path that would surface with synthetic data, subset analyses, or future sessions with different voting patterns.

**Fix:** Guard the summary print statements with `if loyalty.height > 0`.

**Lesson:** Any function that indexes into a DataFrame or Series must either (a) guarantee the collection is non-empty via an upstream check, or (b) guard the indexing operation. Print/logging statements are particularly easy to forget — they feel "safe" because they don't affect the return value, but they can still crash the function.
