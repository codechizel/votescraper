# Volume 2: Gathering the Data

*Before analysis, you need data. Here's how we get it — reliably, reproducibly, and respectfully.*

This volume explains how Tallgrass collects every recorded vote from the Kansas Legislature's website, retrieves the full text of every bill, cross-validates against a second data source, and loads everything into a database. It covers the ethics and mechanics of web scraping, the four-phase scraper pipeline, historical data challenges stretching back to 2011, and the quality assurance practices that ensure the data is worth analyzing.

By the end of this volume, you'll understand how 95,000 vote records per session travel from a government website to a structured dataset ready for statistical analysis — and why every design decision along the way prioritizes correctness over speed.

---

## Chapters

1. **[Web Scraping: The Polite Robot](ch01-web-scraping-polite-robot.md)**
   What web scraping is, why ethics matter, and how Tallgrass collects public data without being a bad neighbor.

2. **[The Four-Phase Scraper Pipeline](ch02-four-phase-scraper-pipeline.md)**
   Discover bills, filter to those with votes, parse every roll call, and enrich with legislator metadata — the assembly line that turns a website into a dataset.

3. **[Historical Sessions and the ODT Challenge](ch03-historical-sessions-odt.md)**
   The Kansas Legislature changed its website format multiple times between 2011 and 2026. Here's how we handle every era.

4. **[Cross-Validation: Trust but Verify (KanFocus)](ch04-cross-validation-kanfocus.md)**
   A second, independent data source lets us verify the scraper's accuracy — and extend coverage back to 1999.

5. **[Bill Text, ALEC Detection, and the Database](ch05-bill-text-alec-database.md)**
   Beyond votes: retrieving the full text of every bill, detecting model legislation, and loading everything into PostgreSQL.

---

## Key Terms Introduced

| Term | Definition |
|------|-----------:|
| **Web scraping** | Automated collection of data from websites by programmatically reading web pages |
| **Rate limiting** | Deliberately slowing requests to avoid overloading a server |
| **Caching** | Storing previously fetched pages locally to avoid redundant requests |
| **robots.txt** | A file on websites that tells automated programs which pages they may access |
| **User agent** | A string that identifies who or what is making a web request |
| **ODT** | OpenDocument Text format — a file format used for Kansas vote records in 2011–2014 |
| **KLISS API** | The Kansas Legislature's programmatic data interface |
| **Cross-validation** | Verifying data accuracy by comparing against an independent source |
| **KanFocus** | A subscription legislative tracking service with vote data back to 1999 |
| **ALEC** | American Legislative Exchange Council — an organization that drafts template legislation for state legislatures |
| **Model legislation** | Pre-written draft bills designed to be introduced across multiple state legislatures |

---

*Previous: [Volume 1 — The Big Picture](../volume-01-big-picture/)*

*Next: [Volume 3 — Your First Look at the Votes](../volume-03-first-look/)*
