---
paths:
  - "src/tallgrass/scraper.py"
  - "src/tallgrass/odt_parser.py"
  - "src/tallgrass/bills.py"
---

# HTML Parsing Pitfalls (Hard-Won Lessons)

These are real bugs that were found and fixed. Do NOT regress on them:

1. **Tag hierarchy on vote pages is NOT what you'd expect.** `<h2>` = bill number, `<h4>` = bill title, `<h3>` = chamber/date/motion AND vote category headings.

2. **Party detection via full page text will always match "Republican".** Every legislator page has a party filter dropdown. Must parse the specific `<h2>` containing "District \d+".

2b. **Legislator `<h1>` is NOT the member name.** First `<h1>` is a nav heading. Must use `soup.find("h1", string=re.compile(r"^(Senator|Representative)\s+"))`. Also strip leadership suffixes.

3. **Vote category parsing requires scanning BOTH h2 and h3.** Uses `soup.find_all(["h2", "h3", "a"])` — do not simplify.

4. **KLISS API response structure varies.** Raw list or `{"content": [...]}`. Always handle both.

5. **Pre-2015 party detection uses `<h3>Party: Republican</h3>`** instead of `<h2>District N - Republican</h2>`.

6. **Pre-2021 bill lists are JavaScript-rendered.** JS fallback fetches `bills_li_{end_year}.js` data files.

6b. **Pre-2021 JS data uses two key formats.** 88th uses quoted JSON keys; 87th and earlier use unquoted JS object literal syntax.

6c. **JS data files live at `/m/` not `/s/` for all sessions except the 88th.**

7. **Pre-2015 vote pages are ODT files, not HTML.** ZIP archives with `content.xml`. House/Senate use different vote category names.

8. **Pre-2021 member directories are JavaScript-rendered.** Same unquoted-key issue as bill data.

9. **KS Legislature server returns HTML error pages with HTTP 200 for binary URLs.** `_get()` checks `content[:5]` for `<html` prefix.

10. **84th session ODTs often lack individual vote data.** ~30% are committee-of-the-whole (tally-only). Not a parser bug.

## Session URL Logic

- Current (2025-26): `/li/b2025_26/...`
- Historical (2023-24): `/li_2024/b2023_24/...`
- Special (2024): `/li_2024s/...`
- API: current uses `/li/api/v13/rev-1`, historical uses `/li_{end_year}/api/v13/rev-1`
- `CURRENT_BIENNIUM_START` in session.py must be updated when a new biennium becomes current (next: 2027).
