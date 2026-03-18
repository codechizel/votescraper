# Chapter 4: Cross-Validation: Trust but Verify (KanFocus)

> *A dataset is only as good as your ability to verify it. The best way to check a scraper's accuracy is to compare its output against a completely independent source.*

---

## Why a Second Source Matters

Here's a scenario that keeps data engineers up at night: you build a scraper, it runs without errors, the output looks reasonable, and you proceed to analyze 95,000 vote records. Six months later, you discover that a parsing bug silently misclassified "Nay" votes as "Absent and Not Voting" in about 3% of cases. None of your downstream results can be trusted. Six months of analysis, invalidated by a quiet bug.

The problem with scraper bugs is that they often produce plausible output. A legislator who appears to have been absent from a vote they actually attended — that's not obviously wrong in any single record. It only becomes visible when you aggregate thousands of records and compare them against something you know is correct.

This is where **cross-validation** comes in. If you have two independent sources of the same data, collected by different methods, and they agree — that's strong evidence that both are correct. Where they disagree, you've found either a bug in one source or a genuine ambiguity in the underlying data.

**Think of it this way.** Imagine you're proofreading a transcript of a speech. You could read it carefully, and you'd catch most errors. But if someone else independently transcribed the same speech, you could compare the two transcripts word by word. Every place they agree, you can be confident. Every place they disagree, you know exactly where to look for the error.

## What Is KanFocus?

**KanFocus** is a subscription-based legislative tracking service specifically for the Kansas Legislature. It has been described as "the only full legislative tracking service in Kansas" and has operated for more than 20 years, provided through the National Online Legislative Association (NOLA).

KanFocus serves every member of the Kansas Legislature, dozens of state agencies, universities, major lobbying firms, and many civic organizations. Its customers use it to track bills, read committee summaries, receive real-time alerts, and — critically for our purposes — look up individual roll call vote tallies.

KanFocus maintains its own database of vote records, collected and formatted independently of the legislature's website. This makes it an ideal cross-validation source: same underlying data (who voted what on which bill), different collection method, different storage format, different organization.

### Coverage

KanFocus data extends further back than kslegislature.gov's digital vote records:

| Source | Coverage | Vote ID Format |
|--------|----------|----------------|
| kslegislature.gov | 2011–2026 (84th–91st) | `je_` prefix (timestamp-based) |
| KanFocus | 1999–2026 (78th–91st) | `kf_` prefix (sequential) |

The overlap (2011–2026) enables cross-validation. The extra years (1999–2010) extend the dataset to 27 years of coverage — invaluable for studying long-term trends.

**Codebase:** `src/tallgrass/kanfocus/` (the full KanFocus subpackage)

## How the KanFocus Scraper Works

KanFocus's scraper operates quite differently from the main kslegislature.gov scraper. The key differences reflect the fact that KanFocus is a *paid, shared service*, not a government's public website.

### Conservative Rate Limiting

Where the main scraper waits 0.15 seconds between requests, the KanFocus scraper waits **7 seconds** — nearly 50 times slower. During business hours, the recommended delay is 12 seconds. The reasoning is straightforward: KanFocus is a commercial service used by real people — lobbyists, legislators, legislative staff — who depend on it for their work. Overwhelming it would affect paying customers.

The KanFocus scraper also runs single-threaded (one request at a time), while the main scraper uses up to five simultaneous connections.

### Authentication

KanFocus is a subscription service, so accessing it requires a login. Rather than storing credentials in configuration files (which would be a security risk), the KanFocus scraper extracts session cookies directly from the user's Chrome browser. If you're logged into KanFocus in Chrome, the scraper piggybacks on that active session.

This approach has a practical benefit: the scraper never has to store or transmit passwords, and it automatically inherits whatever access level the user has.

### Sequential Enumeration

KanFocus vote pages use sequential IDs. The scraper doesn't need to discover which votes exist — it simply counts up from 1, requesting each vote page in order. When it encounters 20 consecutive empty pages (meaning those vote IDs don't exist), it stops.

Each biennium has four "streams" to enumerate: two years times two chambers. The scraper works through each stream independently:
- 2025 House votes
- 2025 Senate votes
- 2026 House votes
- 2026 Senate votes

### What the Data Looks Like

KanFocus vote pages have a consistent format across all 27 years:

```
Vote #: 33    Date: 02/03/2011    Bill Number: SB 13
Question: Shall the bill pass?    Result: Passed

          For  Against  Present  Not Voting
          66   53       0        6

Yea (66):
  Steve Abrams, R-32nd
  Jim Barnett, R-12th
  ...

Nay (53):
  Marci Francisco, D-2nd
  Laura Kelly, D-18th
  ...
```

Each legislator entry includes their name, party abbreviation, and district — more metadata than the main scraper gets from a vote page (which has only names and slugs). The parser extracts all of this, generating standardized slugs from the "Name, Party-District" format.

**Codebase:** `src/tallgrass/kanfocus/parser.py` (vote page parsing), `src/tallgrass/kanfocus/fetcher.py` (HTTP client with authentication)

## The Cross-Validation Process

The cross-validation module is a read-only diagnostic tool. It reads existing data files — it never downloads anything, and it never modifies any data. Its job is strictly to compare.

### Step 1: Match Roll Calls

The first challenge is figuring out which KanFocus vote corresponds to which kslegislature.gov vote. The two sources use completely different identifiers (`kf_33_2025_H` vs. `je_20250320203513_1`), so matching requires finding common ground.

The primary match key is: **(bill number, chamber, vote date)**

Most votes can be uniquely identified by this combination — there's usually only one recorded vote on SB 13 in the Senate on February 3rd. But sometimes a bill has multiple votes on the same day (an amendment vote and a final passage vote, for example). In those cases, the cross-validator uses a secondary key: the **vote tally** (yea count, nay count, not-voting total). If SB 13 had two Senate votes on February 3rd — one with 35-5 and one with 27-13 — the tallies disambiguate.

### Step 2: Compare Individual Votes

For each matched roll call, the cross-validator compares every individual legislator's vote:

1. **Slug matching:** Try to match each KanFocus legislator to a kslegislature.gov legislator by slug. Since the two sources generate slugs differently, this sometimes fails.
2. **Name fallback:** When slugs don't match, try matching by normalized full name — lowercased, with punctuation and suffixes removed.
3. **Last-name fallback:** As a final resort, match by last name only. This fails for ambiguous cases (two legislators with the same last name), which are flagged rather than guessed.

### Step 3: Identify Discrepancies

For each matched pair of votes, the cross-validator categorizes every legislator's record:

| Category | Meaning |
|----------|---------|
| **Match** | Both sources agree on the vote |
| **Mismatch** | The sources disagree (e.g., one says Yea, the other says Nay) |
| **Absent ambiguity** | One source says "Absent and Not Voting," the other says "Not Voting" |
| **Unmatched** | A legislator appears in one source but not the other |

### The ANV/NV Ambiguity

The most common discrepancy type isn't actually an error — it's a genuine ambiguity in the data. The Kansas Legislature records two categories of non-participation:

- **Absent and Not Voting (ANV)** — the legislator was not present
- **Not Voting (NV)** — the legislator was present but did not vote

In practice, the boundary between these categories is sometimes blurry. KanFocus and kslegislature.gov occasionally categorize the same non-vote differently. The cross-validator flags these as "absent ambiguity" rather than counting them as errors, because neither source is clearly wrong — they're applying different interpretations to an inherently fuzzy situation.

### Step 4: Report

The cross-validation produces a Markdown report listing:
- Total roll calls compared
- Match rate (what percentage of individual votes agree)
- Every discrepancy, organized by bill and vote
- Summary statistics by discrepancy type

In practice, the match rate between KanFocus and kslegislature.gov is extremely high — typically above 99.5%. The discrepancies that do exist are almost entirely ANV/NV ambiguities, not genuine errors in either source.

```bash
# Run cross-validation for the 91st Legislature
just kanfocus 2025 --mode crossval
```

**Codebase:** `src/tallgrass/kanfocus/crossval.py` (the full cross-validation logic)

## What Cross-Validation Catches

The value of cross-validation isn't just the final match rate — it's the bugs found along the way. During development, cross-validation against KanFocus revealed:

- **A parsing bug in the "Sub for" prefix.** Kansas bills that are substituted in committee get a modified bill number (e.g., "Sub for SB 55"). The main scraper and KanFocus handle this prefix differently, which initially caused false mismatches. The fix was a normalization function that strips "Sub for" before comparing.

- **ODT-era deduplication issues.** Before deduplication was added, some ODT-era votes appeared twice in the main dataset. Cross-validation flagged these as "extra" votes that KanFocus didn't have, leading to the deduplication fix.

- **Name encoding inconsistencies.** Legislators with apostrophes in their names (like "O'Neal") were sometimes encoded differently across sources. A normalization layer now handles these cases.

None of these bugs produced obviously wrong output. The individual vote records looked fine in isolation. Only by comparing against an independent source did the problems become visible.

## Extending Coverage to 1999

Beyond validation, KanFocus serves a second purpose: it extends the dataset backward in time.

The kslegislature.gov website provides digital vote records starting from the 84th Legislature (2011–2012). For the 78th through 83rd Legislatures (1999–2010), the website may have journal archives but no structured vote data in a scrapeable format.

KanFocus has individual vote records going back to 1999. By scraping KanFocus for these earlier sessions and converting them into the standard CSV format, Tallgrass gains access to six additional bienniums — 12 more years of data.

The conversion process is handled by the KanFocus output module, which transforms KanFocus-specific data structures into the same `IndividualVote` and `RollCall` format used by the main scraper. From the analysis pipeline's perspective, a KanFocus-sourced session looks identical to a kslegislature.gov-sourced session.

The only caveat: KanFocus data uses `kf_` prefixed vote IDs instead of `je_` prefixed ones, so it's always clear which source a particular vote record came from.

**Codebase:** `src/tallgrass/kanfocus/output.py` (format conversion), `src/tallgrass/kanfocus/cli.py` (entry point)

## Data Archiving

Because KanFocus is a paid service and its data isn't guaranteed to remain available, the KanFocus scraper includes an archiving system. After each successful session scrape, the raw HTML files are copied to a permanent archive directory:

```
data/kanfocus_archive/
  78th_1999-2000/
  79th_2001-2002/
  ...
  91st_2025-2026/
```

The `--clear-cache` flag (which re-downloads everything from scratch) is blocked unless an archive already exists for that session. This prevents accidental data loss — you can't delete the cache unless you've already saved a backup.

The cache itself is restart-safe: each page is stored in a file named by the SHA-256 hash of its URL. If a scrape is interrupted halfway through, restarting it picks up where it left off — pages already in the cache aren't re-downloaded.

---

## Key Takeaway

Cross-validation against KanFocus confirms the main scraper's accuracy (99.5%+ match rate) and catches subtle bugs that plausible-looking output would hide. As a bonus, KanFocus extends coverage back to 1999 — doubling the historical depth of the dataset and enabling long-term trend analysis across 27 years of Kansas legislative history.

---

*Terms introduced: cross-validation, KanFocus, match key, absent ambiguity (ANV/NV), slug matching, data archiving, sequential enumeration*

*Next: [Bill Text, ALEC Detection, and the Database](ch05-bill-text-alec-database.md)*
