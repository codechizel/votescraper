# Data Dictionary

## Output Files

The scraper produces three CSV files per session in `data/{output_name}/` (e.g., `data/91st_2025-2026/`).

---

## votes CSV (`{output_name}_votes.csv`)

One row per legislator per roll call. The largest file (~68K rows for a full session).

| Field | Type | Description |
|---|---|---|
| `session` | str | Session label, e.g., "91st (2025-2026)" or "2024 Special" |
| `bill_number` | str | Bill identifier, e.g., "SB 1", "HB 2124", "SCR 1601" |
| `bill_title` | str | Full bill title from vote page `<h4>`, truncated to 500 chars |
| `vote_id` | str | Unique vote identifier from URL, e.g., "je_20250320203513_2584" |
| `vote_datetime` | str | ISO 8601 datetime parsed from vote_id, e.g., "2025-03-20T20:35:13" |
| `vote_date` | str | Date as displayed on page, MM/DD/YYYY format |
| `chamber` | str | "Senate" or "House" |
| `motion` | str | Full motion text, e.g., "Emergency Final Action - Passed as amended" |
| `legislator_name` | str | Legislator last name as shown on vote page |
| `legislator_slug` | str | URL slug, e.g., "sen_claeys" or "rep_smith" |
| `vote` | str | One of: "Yea", "Nay", "Present and Passing", "Absent and Not Voting", "Not Voting" |

**Composite key**: `vote_id` + `legislator_slug` (or `vote_datetime` + `legislator_slug`)

---

## rollcalls CSV (`{output_name}_rollcalls.csv`)

One row per roll call vote. Summary-level data (~500 rows for a full session).

| Field | Type | Description |
|---|---|---|
| `session` | str | Session label |
| `bill_number` | str | Bill identifier |
| `bill_title` | str | Full bill title, truncated to 500 chars |
| `vote_id` | str | Unique vote identifier |
| `vote_url` | str | Full URL to the vote page |
| `vote_datetime` | str | ISO 8601 datetime from vote_id |
| `vote_date` | str | Date as displayed, MM/DD/YYYY |
| `chamber` | str | "Senate" or "House" |
| `motion` | str | Full motion text |
| `vote_type` | str | Classified type (see Vote Types below) |
| `result` | str | Outcome text after vote_type prefix, e.g., "Passed as amended" |
| `short_title` | str | Clean 1-sentence description from KLISS API |
| `sponsor` | str | Original sponsor(s) from KLISS API, semicolon-separated |
| `yea_count` | int | Number of Yea votes |
| `nay_count` | int | Number of Nay votes |
| `present_passing_count` | int | Number of Present and Passing votes |
| `absent_not_voting_count` | int | Number of Absent and Not Voting |
| `not_voting_count` | int | Number of Not Voting |
| `total_votes` | int | Sum of all 5 categories |
| `passed` | bool/None | True if passed/adopted/prevailed/concurred, False if failed/rejected/sustained, None if unclear |

**Primary key**: `vote_id`

---

## legislators CSV (`{output_name}_legislators.csv`)

One row per legislator (~172 rows for a full session).

| Field | Type | Description |
|---|---|---|
| `name` | str | Last name as shown on vote pages |
| `full_name` | str | Full name from member page `<h1>`, title prefix stripped |
| `slug` | str | URL slug, e.g., "sen_claeys". Prefix encodes chamber. |
| `chamber` | str | "Senate" or "House" (inferred from slug prefix) |
| `party` | str | "Republican" or "Democrat" (parsed from District h2) |
| `district` | str | District number |
| `member_url` | str | Full URL to legislator's member page |

**Primary key**: `slug`

---

## Vote Types

| vote_type | Trigger | Typical for |
|---|---|---|
| Emergency Final Action | Motion starts with "Emergency Final Action" | Bills needing 2/3 majority |
| Final Action | Motion starts with "Final Action" | Standard passage votes |
| Committee of the Whole | Motion starts with "Committee of the Whole" | Floor amendment votes |
| Consent Calendar | Motion starts with "Consent Calendar" | Non-controversial bills |
| Veto Override | Motion contains "override" and "veto" | Governor veto overrides |
| Conference Committee | Motion contains "conference committee" | Bicameral reconciliation |
| Concurrence | Motion contains "concur" | Accepting other chamber's amendments |
| Procedural Motion | Motion starts with "motion" or "citing rule" | Rules/procedure votes |
| *(empty)* | No pattern matched | Unclassified motions |

---

## Joins

- votes → rollcalls: join on `vote_id`
- votes → legislators: join on `legislator_slug` = `slug`
- rollcalls → legislators: indirect via votes table

---

## Notes for Analysis

- `vote_datetime` is more precise than `vote_date` and better for deduplication and temporal ordering
- `total_votes` enables quick participation rate calculation: `(yea_count + nay_count) / total_votes`
- For passage analysis, filter to `vote_type` in ("Final Action", "Emergency Final Action") to exclude procedural/committee votes
- `short_title` from the API is generally cleaner and more concise than `bill_title` from HTML
- The 5 vote categories are exhaustive — every legislator falls into exactly one per roll call
