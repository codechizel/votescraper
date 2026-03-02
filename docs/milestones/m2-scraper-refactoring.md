# M2: Scraper Function Splits — COMPLETE

Extracted helper methods from `_parse_vote_page()` (193 lines) and `enrich_legislators()` (64 lines) to improve readability and testability.

**Roadmap items:** R8 (`_parse_vote_page` split), R9 (`enrich_legislators` split)
**Completed:** 2026-03-02 (1 session)
**Result:** 265 scraper tests pass (264 existing + 1 new). Net -80 lines (219 added, 299 removed).

---

## Part A: `_parse_vote_page()` Split (R8)

### Current State

`_parse_vote_page()` lives at `src/tallgrass/scraper.py:903-1095` (193 lines). It handles bill title extraction, chamber/motion/date parsing, vote category parsing, legislator registry updates, and RollCall/IndividualVote construction in a single method.

### Extractions

#### 1. `_extract_bill_title(soup) -> str`

**Source lines:** 915-936 (22 lines)
**Logic:** 3-tier fallback searching `<h4>` tags:
1. Regex match: `AN ACT|A CONCURRENT|A RESOLUTION|A JOINT` (case-insensitive)
2. Scan `<h4>` for text starting with "AN ACT" or length > 50
3. Scan `<h4>` for text > 30 chars not starting with `("SB", "HB", "On roll", "Yea", "Nay", "Senate", "House")`

```python
@staticmethod
def _extract_bill_title(soup: BeautifulSoup) -> str:
    """Extract bill title from vote page HTML using 3-tier h4 fallback.

    Pitfall #1: <h4> = bill title (not <h2> which is bill number).
    """
    ...
```

**Returns:** Bill title string, or `""` if not found.

#### 2. `_extract_chamber_motion_date(soup) -> tuple[str, str, str]`

**Source lines:** 938-971 (34 lines)
**Logic:** 2-tier fallback searching `<h3>` tags:
1. Strict regex: `r"(Senate|House)\s*-\s*(.+?)\s*-\s*(\d{2}/\d{2}/\d{4})$"` on `<h3>` text
2. Loose parse: scan `<h3>` for text starting with "Senate"/"House", extract date via separate regex, remove both from text

```python
@staticmethod
def _extract_chamber_motion_date(soup: BeautifulSoup) -> tuple[str, str, str]:
    """Extract chamber, motion text, and vote date from h3 headers.

    Pitfall #1: <h3> contains chamber/date/motion AND vote category headings.
    Returns: (chamber, motion, vote_date) — date in MM/DD/YYYY format.
    """
    ...
```

**Key detail:** Motion text stripped of trailing `" -;"` (line 955). Uses `_clean_text()` to preserve spaces around inline `<a>` tags (Pitfall #5).

#### 3. `_parse_vote_categories(soup) -> tuple[dict[str, list[dict]], dict[str, dict]]`

**Source lines:** 985-1024 (40 lines)
**Logic:** Iterates `soup.find_all(["h2", "h3", "a"])` in document order (Pitfall #3). When `<h2>`/`<h3>` text matches a `VOTE_CATEGORIES` entry, sets `current_category`. When `<a>` with `/members/` in href is found, extracts name and slug. Also builds new legislator entries.

```python
@staticmethod
def _parse_vote_categories(
    soup: BeautifulSoup,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, str]]]:
    """Parse vote categories and member lists from vote page HTML.

    Pitfall #3: Must scan BOTH <h2> and <h3> for category headings.

    Returns:
        (vote_categories, new_legislators) where:
        - vote_categories maps category name to list of {"name": str, "slug": str}
        - new_legislators maps slug to {"name", "slug", "chamber", "member_url"}
    """
    ...
```

**Critical change:** Currently modifies `self.legislators` as a side effect. The extracted version returns `new_legislators` as a separate dict, and `_parse_vote_page()` merges them. This eliminates the mutation-during-parse side effect.

### Rewired `_parse_vote_page()`

After extraction, the method becomes a thin coordinator:

```python
def _parse_vote_page(self, soup: BeautifulSoup, vote_link: VoteLink) -> None:
    bill_number = self._extract_bill_number(soup, vote_link)
    bill_title = self._extract_bill_title(soup)
    chamber, motion, vote_date = self._extract_chamber_motion_date(soup)
    vote_categories, new_legislators = self._parse_vote_categories(soup)

    # Merge new legislators into registry
    for slug, info in new_legislators.items():
        if slug not in self.legislators:
            self.legislators[slug] = info

    # Build RollCall and IndividualVote records (existing logic, ~60 lines)
    ...
```

---

## Part B: `enrich_legislators()` Split (R9)

### Current State

`enrich_legislators()` lives at `src/tallgrass/scraper.py:1158-1221` (64 lines). The party/district extraction logic has 2 fallback patterns buried in the sequential parse loop.

### Extraction

#### `_extract_party_and_district(soup) -> dict[str, str]`

**Source lines:** 1184-1219 (28 lines in the parse loop)
**Logic:**
1. **Full name** from `<h1>` matching `r"^(Senator|Representative)\s+"` (Pitfall #2b), strip title prefix and leadership suffix
2. **Post-2015** (Tier 1): `<h2>` matching `r"District\s+\d+"` — extract district number and party from same element (Pitfall #2)
3. **Pre-2015** (Tier 2): `<h3>` containing `"Party:"` text — extract party (Pitfall #5 in CLAUDE.md section)

```python
@staticmethod
def _extract_party_and_district(soup: BeautifulSoup) -> dict[str, str]:
    """Extract full name, party, and district from a legislator page.

    Two fallback patterns:
    - Post-2015: <h2>District N - Republican</h2>
    - Pre-2015: <h3>Party: Republican</h3>

    Pitfall #2: Must parse the specific <h2> containing "District \\d+",
    NOT full page text (which always matches "Republican" via dropdown).
    Pitfall #2b: First <h1> is nav heading, not member name.

    Returns: {"full_name": str, "party": str, "district": str}
    """
    ...
```

### Rewired `enrich_legislators()`

```python
def enrich_legislators(self) -> None:
    to_fetch = {slug: info for slug, info in self.legislators.items() if "party" not in info}
    results = self._fetch_many([info["member_url"] for info in to_fetch.values()])

    for slug, info in to_fetch.items():
        result = results.get(info["member_url"])
        if not result or not result.ok:
            continue
        soup = BeautifulSoup(result.html, "html.parser")
        parsed = self._extract_party_and_district(soup)
        info.update(parsed)
```

---

## Existing Test Coverage

`tests/test_scraper_html.py` has 31 tests across 8 classes that directly test the parsing logic:

| Test Class | Line | Tests | Maps To |
|-----------|------|-------|---------|
| `TestCleanText` | 26 | 3 | `_clean_text()` helper (Pitfall #5) |
| `TestExtractBillNumber` | 49 | 3 | `_extract_bill_number()` (already separate) |
| `TestExtractSponsor` | 75 | 3 | Sponsor from portlet structure |
| `TestVoteCategoryParsing` | 116 | 6 | `_parse_vote_categories()` — category detection from h2/h3, member link extraction |
| `TestLegislatorParsing` | 214 | 6 | `_extract_party_and_district()` — h1 name, h2 party/district |
| `TestVotePageMetadata` | 332 | 5 | `_extract_bill_title()` + `_extract_chamber_motion_date()` |
| `TestPreFifteenLegislatorParsing` | 413 | 4 | `_extract_party_and_district()` — h3 fallback |
| `TestOdtViewLinkDetection` | 499 | 3 | ODT routing (unrelated) |

### Test Migration

After extraction, update tests to call the static methods directly:

```python
# Before (tests construct full scraper + mock page):
class TestVotePageMetadata:
    def test_title_from_h4(self, scraper):
        soup = BeautifulSoup(HTML, "html.parser")
        scraper._parse_vote_page(soup, vote_link)
        assert scraper.rollcalls[-1].bill_title == "AN ACT concerning..."

# After (call static method directly):
class TestVotePageMetadata:
    def test_title_from_h4(self):
        soup = BeautifulSoup(HTML, "html.parser")
        title = KSVoteScraper._extract_bill_title(soup)
        assert title == "AN ACT concerning..."
```

This makes tests faster (no scraper instantiation) and more focused (test one function, not the whole pipeline).

---

## Pitfall Preservation Checklist

These documented bugs (CLAUDE.md) **must not regress**:

- [x] Pitfall #1: Tag hierarchy — `<h2>` = bill number, `<h4>` = bill title, `<h3>` = chamber/date/motion + vote category headings
- [x] Pitfall #2: Party from `<h2>` District line, not full page text
- [x] Pitfall #2b: Name from second `<h1>` (Senator/Representative prefix), not nav heading
- [x] Pitfall #3: Vote categories from both `<h2>` and `<h3>` (`find_all(["h2", "h3", "a"])`)
- [x] Pitfall #5: `_clean_text()` preserves spaces around inline `<a>` tags

---

## Verification

```bash
just test-scraper            # 264 tests — must all pass
just test                    # full suite
just lint-check              # formatting
```

## Documentation

- Update `docs/roadmap.md` items R8 and R9 to "Done"
- No ADR needed (pure refactoring, no behavior change)

## Commit

```
refactor(scraper): extract helpers from _parse_vote_page and enrich_legislators [vYYYY.MM.DD.N]
```
