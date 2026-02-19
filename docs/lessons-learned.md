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
