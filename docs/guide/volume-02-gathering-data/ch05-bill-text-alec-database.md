# Chapter 5: Bill Text, ALEC Detection, and the Database

> *Votes tell you what legislators did. Bill text tells you what they were voting on. Combining the two opens the door to questions neither could answer alone.*

---

## Beyond Votes

The first four chapters of this volume focused on collecting vote data — who voted Yea, who voted Nay, on which bill, when. This is the core dataset that drives the entire analysis pipeline. But votes are only half the story.

A vote on "SB 55" means nothing without knowing what SB 55 does. Is it a tax cut? A school funding increase? A criminal sentencing reform? The vote record alone can't tell you. For that, you need the bill's text.

And bill text opens up a whole new category of questions:

- **What topics does the legislature spend its time on?** Are more votes cast on education or healthcare? Has the mix changed over time?
- **Are some bills copied from external templates?** State legislatures across the country frequently consider bills drafted by national organizations. Can we detect when a Kansas bill was based on a template?
- **Can the words of a bill predict how it will be voted on?** If you read a bill's text, can you predict whether it will pass along party lines or attract bipartisan support?

These questions are addressed in Volumes 7 and 8. This chapter covers the infrastructure that makes them possible: retrieving bill text, detecting model legislation, and loading everything into a database.

## Bill Text Retrieval

Every bill introduced in the Kansas Legislature has its full text available as a PDF document on kslegislature.gov. The bill text subpackage downloads these PDFs and extracts the text inside them.

### What Gets Downloaded

For each bill, up to two documents are available:

**Introduced text** — the bill as originally introduced, in formal legislative language. These tend to be long and dense:

> *AN ACT concerning taxation; relating to income tax rates and brackets; rates of tax on individual income; amending K.S.A. 2024 Supp. 79-32,110 and repealing the existing section.*

**Supplemental notes** — a plain-English summary written by legislative staff. These are shorter and much more readable:

> *The bill would reduce the number of individual income tax brackets from three to two and lower the top marginal rate from 5.7% to 5.15%.*

The supplemental notes are particularly valuable for natural language processing, because they use ordinary language rather than legal boilerplate.

### How It Works

The bill text subpackage uses a **state adapter pattern** — a design that separates "how to find bills" from "how to download and extract them." The Kansas-specific adapter knows the URL patterns for Kansas bill PDFs. A future adapter for, say, Missouri would implement the same interface with Missouri-specific URLs. The download-and-extract machinery is shared.

For Kansas, bill PDF URLs follow a deterministic pattern:
```
Introduced text:
  {session_prefix}/measures/documents/{bill_code}_00_0000.pdf

Supplemental note:
  {session_prefix}/measures/documents/supp_note_{bill_code}_00_0000.pdf
```

This means the subpackage doesn't need to scrape bill pages to find PDF links — it constructs the URLs mathematically from the bill number and session.

### PDF Extraction

Downloaded PDFs are processed by `pdfplumber`, a Python library that extracts text from PDF files. The raw extracted text goes through several cleaning steps:

1. **Ligature repair** — PDFs sometimes encode common letter combinations ("fi", "fl", "ff") as single characters called ligatures. These need to be converted back to regular letters.
2. **Page number removal** — Legislative PDFs include page numbers that would confuse text analysis.
3. **Line number stripping** — Introduced texts include line numbers in the margin for reference during floor debate. These are stripped from the extracted text.
4. **Whitespace normalization** — Extra spaces, tabs, and newlines are collapsed into clean paragraph breaks.

The result is clean, readable text for each bill — ready for topic modeling, similarity analysis, and other NLP tasks.

```bash
# Retrieve bill texts for the current session
just text 2025
```

**Codebase:** `src/tallgrass/text/` (the complete bill text subpackage)

## ALEC and Model Legislation

### What Is ALEC?

The **American Legislative Exchange Council (ALEC)** is a nonprofit organization founded in 1973 that brings together state legislators and private-sector representatives — corporations, trade associations, and think tanks — to collaboratively draft legislation. The result is a library of **model bills**: pre-written draft legislation that legislators can introduce, with minimal modification, in their own state legislatures.

ALEC organizes its work through issue-specific "task forces," each co-chaired by a state legislator and a corporate representative. Once a task force approves a model bill, it becomes available to ALEC's legislative members across all 50 states.

The scale is significant. Research from the Brookings Institution found that between 2010 and 2018, ALEC-drafted model legislation was introduced in state legislatures approximately 2,900 times, with over 600 bills enacted into law. The Center for Public Integrity documented more than 10,000 instances of legislators introducing bills based on model templates from various organizations.

### Why Detection Matters

Model legislation is a legitimate part of the legislative process — organizations of all political orientations produce template bills. But detecting when a state bill was derived from a template is valuable for transparency and analysis:

- **For researchers:** Knowing that a bill's origin is external (rather than locally drafted) provides context for understanding voting patterns. A bill introduced as-is from a national template may produce different coalition dynamics than one crafted for Kansas-specific concerns.
- **For journalists:** Identifying model legislation is a standard accountability practice. Voters may want to know whether their legislator is advancing locally developed policy or importing templates from national organizations.
- **For the pipeline:** Text-based similarity between Kansas bills and ALEC model bills is one of the features used in the text analysis phases (Volume 7).

### The ALEC Scraper

Tallgrass includes a dedicated scraper for ALEC's publicly available model legislation at `alec.org/model-policy/`. This scraper is completely independent of the vote scraper — it doesn't touch the Kansas Legislature's website at all.

**How it works:**

1. **Enumerate listings:** The scraper paginates through ALEC's model policy listing pages, extracting metadata for each model bill (title, category, task force, date).
2. **Fetch full text:** For each model bill, the scraper downloads the full page and extracts the bill text from the article body.
3. **Export:** The results are saved as a CSV file.

Each model bill record contains:

| Field | Example |
|-------|---------|
| Title | Occupational Licensing Relief and Job Creation Act |
| Text | (full bill text) |
| Category | Labor and Employment |
| Bill type | Model Policy |
| Date finalized | 2019-08-15 |
| URL | (source link) |
| Task force | Commerce, Insurance and Economic Development |

The scraper currently collects over 1,000 model bills from ALEC's public archive.

**Rate limiting:** The ALEC scraper uses even more conservative settings than the KanFocus scraper — only three simultaneous connections and standard rate limiting. ALEC's website is not a government service and has no obligation to serve automated requests, so extra caution is warranted.

```bash
# Scrape ALEC model legislation
just alec
```

**Codebase:** `src/tallgrass/alec/` (the ALEC scraper subpackage)

### How Detection Works (Preview)

The actual detection of model legislation influence is performed in Phase 23 of the analysis pipeline (covered in Volume 7). The basic approach is text similarity: for each Kansas bill, compute how similar its text is to each ALEC model bill. Bills with high similarity scores are flagged as potential matches.

This is not as simple as checking for identical text. Legislators often modify model bills — changing dollar amounts, adjusting definitions, adding Kansas-specific provisions. The detection system uses embedding-based similarity (converting text into numerical representations that capture meaning, not just exact words) to catch these modified versions.

## The Database

Up to this point, all data has been stored as CSV files on disk. CSV files are simple, portable, and human-readable — you can open them in Excel or any text editor. But they have limitations:

- **No relationships:** CSV files can't express that a vote record belongs to a specific roll call, which belongs to a specific bill. You have to enforce these relationships in your code.
- **No queries:** To answer "how many times did Senator Smith vote Nay on tax bills?", you'd need to load the votes file, filter by legislator, join with the roll calls file, filter by bill topic, and count. In a database, it's one query.
- **No concurrent access:** If two programs try to read and write the same CSV file simultaneously, one of them will get corrupted data.

Tallgrass addresses these limitations with a **PostgreSQL database** managed through Django (a Python web framework).

### The Database Schema

The database stores the same data as the CSV files, but in relational tables with proper foreign keys:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│ legislators │     │  rollcalls   │     │  bill_actions  │
│─────────────│     │──────────────│     │───────────────│
│ slug (PK)   │     │ vote_id (PK) │     │ bill_number   │
│ full_name   │     │ bill_number  │     │ action_code   │
│ party       │     │ motion       │     │ chamber       │
│ district    │     │ result       │     │ status        │
│ ocd_id      │     │ yea_count    │     │ date          │
└──────┬──────┘     │ nay_count    │     └───────────────┘
       │            └──────┬───────┘
       │                   │
       │    ┌──────────────┤
       │    │              │
       ▼    ▼              │
  ┌─────────────┐          │
  │    votes    │          │
  │─────────────│          │
  │ legislator  │──FK──────┘
  │ rollcall    │──FK
  │ vote        │
  └─────────────┘
```

The `votes` table has foreign keys pointing to both `legislators` and `rollcalls`, enforcing that every vote record belongs to a real legislator and a real roll call. The database won't let you insert a vote for a legislator who doesn't exist — a guarantee CSV files can't make.

### Loading Data

The database loading process is designed to be seamless. The `--auto-load` flag on the main scraper triggers it automatically:

```bash
# Scrape and load in one step
just scrape 2025 --auto-load
```

Under the hood, this invokes a Django management command that:

1. Reads the CSV files produced by the scraper
2. Loads legislators, roll calls, and individual votes into the database
3. Loads bill actions and bill texts
4. Handles duplicates gracefully (re-loading a session updates existing records)

The loading is done via a subprocess — the main scraper doesn't import Django at all. This keeps the scraper lightweight and dependency-free. If the database isn't running or Django isn't installed, the scraper still works fine; it just produces CSV files without loading them.

### What the Database Enables

With all data loaded, the database currently contains:

| Table | Rows |
|-------|------|
| Individual votes | 649,000+ |
| Roll calls | 8,000+ |
| Legislators | 2,000+ |
| Bill actions | 23,000+ |
| Bill texts | 1,600+ |
| ALEC model bills | 1,000+ |

The database powers a REST API (at `/api/v1/`) that makes this data available to web applications, dashboards, and third-party tools. It also enables SQL queries that would be cumbersome with CSV files:

```sql
-- How many Nay votes did each party cast in the 91st Legislature?
SELECT l.party, COUNT(*) AS nay_votes
FROM votes v
JOIN legislators l ON v.legislator_id = l.id
JOIN rollcalls r ON v.rollcall_id = r.id
WHERE r.session = '91st (2025-2026)' AND v.vote = 'Nay'
GROUP BY l.party;
```

### Running the Database

The database runs in Docker (via OrbStack on macOS):

```bash
# Start the database
just db-up

# Run migrations (create/update tables)
just db-migrate

# Load all sessions
just db-load-all

# Open the Django admin interface
just db-admin
```

The CSV files remain the primary data format — the analysis pipeline reads CSVs, not the database. The database is a supplementary layer for querying, the REST API, and web-based access.

**Codebase:** `src/tallgrass/db_hook.py` (post-scrape loader), `src/web/` (Django project, database models, REST API)

## The Complete Data Ecosystem

Pulling everything together, here's the full picture of how data flows through the Tallgrass collection system:

```
kslegislature.gov                  KanFocus                    ALEC
      │                               │                         │
      ▼                               ▼                         ▼
  Main Scraper                  KanFocus Scraper           ALEC Scraper
  (4-phase pipeline)            (sequential + auth)        (2-phase)
      │                               │                         │
      ▼                               ▼                         ▼
  CSV files (per session)       CSV files (per session)    ALEC CSV
  ├── votes                     ├── votes                  └── model_bills
  ├── rollcalls                 ├── rollcalls
  ├── legislators               └── legislators
  ├── bill_actions
  └── bill_texts ◄── Bill Text Retriever
                        │
                   PDF download
                   + extraction

All CSVs ──────────────────────► PostgreSQL Database
                                      │
                                      ▼
                                  REST API
                                  (/api/v1/)
```

Three scrapers feed three types of data into a common CSV format. The database unifies everything. Cross-validation connects the main scraper to KanFocus. Bill text connects votes to content. ALEC connects Kansas bills to national templates.

This ecosystem was not designed all at once. It grew organically over the project's development, each piece added when a new analytical question demanded data the existing system didn't have. The CSV format serves as the common language that lets each component operate independently while contributing to a unified dataset.

---

## Key Takeaway

Bill text retrieval, ALEC model legislation scraping, and the PostgreSQL database extend Tallgrass beyond pure vote counting. Bill text enables topic modeling and text-based ideology measurement. ALEC data enables model legislation detection. The database enables efficient querying and powers a REST API. Together with the main scraper and KanFocus, they form a complete data ecosystem — three independent scrapers, a text extractor, a database, and an API, all producing consistently structured data for the 28-phase analysis pipeline.

---

*Terms introduced: supplemental note, state adapter pattern, model legislation, ALEC, text similarity, embedding, PostgreSQL, foreign key, REST API, auto-load*

*Next: [Volume 3 — Your First Look at the Votes](../volume-03-first-look/)*
