# Chapter 1: Web Scraping: The Polite Robot

> *The Kansas Legislature publishes every roll call vote online. The catch? They're buried across thousands of web pages, in formats that change every few years.*

---

## The Problem

The Kansas Legislature's website at [kslegislature.gov](https://www.kslegislature.gov) is a treasure trove of public data. Every bill introduced, every committee hearing, every roll call vote — it's all there, published as a matter of law. The Kansas Constitution requires both chambers to publish a journal for every day they are in session, and the website makes good on that requirement.

But "published" does not mean "organized for analysis." The vote data is spread across thousands of individual web pages, one per roll call. A typical session produces around 1,100 roll calls, each on its own page, each linked from its parent bill's page, each bill listed on a session index page. To collect every vote from a single two-year session, you would need to visit roughly 2,500 web pages — the index pages, the bill pages, and the individual vote pages.

No human is going to do that by hand. Or rather, a human *could*, but it would take weeks of tedious copying and pasting, and the result would be riddled with transcription errors. What we need is a program that can do what a very patient research assistant would do: visit every relevant page, read the data, write it down accurately, and move on to the next one.

That program is called a **web scraper**.

## What Web Scraping Is

Web scraping is the practice of using software to automatically read web pages and extract structured data from them. When you visit a web page in your browser, your browser sends a request to the server, receives HTML (the code that describes the page's content and layout), and renders it as the visual page you see. A web scraper does the same first two steps — sends a request, receives HTML — but instead of rendering a visual page, it reads the HTML and pulls out specific pieces of information.

**Think of it this way.** Imagine you hired a research assistant to go to the library and copy every vote record from every volume of the Kansas Legislative Journal. The assistant would:

1. Find the index of volumes (the session listing page)
2. Open each volume (each bill's page)
3. Find the vote tallies (the vote pages)
4. Copy down who voted Yea, who voted Nay, and who was absent
5. Organize everything into a spreadsheet

A web scraper does exactly this — except it reads web pages instead of physical books, it works around the clock, and it never makes a transcription error. It also works fast: what would take a human weeks takes the scraper about 15 minutes per session.

## Why Ethics Matter

Speed is exactly the problem. A scraper can send hundreds of requests per second if you let it. A government website designed to serve a few hundred human visitors at a time can buckle under that load, slowing down or crashing entirely — which means real people trying to look up their representative's voting record can't access the site.

This is why responsible web scraping follows a set of ethical principles. Think of it as the difference between a library patron who browses the shelves and a person who backs a truck up to the loading dock and starts throwing books in. Both are accessing public materials, but one of them is ruining the experience for everyone else.

### The Four Principles of Polite Scraping

**1. Rate Limiting: Don't Hog the Server**

A polite scraper adds deliberate pauses between requests, ensuring it never sends more traffic than the server can comfortably handle. Tallgrass waits at least 0.15 seconds between requests during normal operation — about six or seven requests per second. That's fast enough to collect a full session in 15 minutes, but gentle enough that a human visitor would never notice the scraper's presence.

When the server shows signs of stress (returning errors or timing out), Tallgrass backs off further — waiting 0.5 seconds between requests and reducing the number of simultaneous connections from five to two. If problems persist, it waits 90 seconds before trying again. The goal is to be the quietest patron in the library.

**2. Caching: Never Ask Twice**

Every page the scraper downloads is saved locally. If you run the scraper again tomorrow, it reads the cached copy instead of re-downloading. This means the server only ever sees each request once, no matter how many times you re-run the pipeline.

This is more than just politeness — it's also good science. Cached pages create a permanent record of exactly what the scraper saw. If someone questions a data point, we can go back to the original HTML and verify it.

**3. Identification: Say Who You Are**

When a web browser visits a page, it sends a "User-Agent" string — a short message identifying itself (e.g., "Chrome/125.0"). Many scrapers disguise themselves as browsers to avoid detection. Tallgrass does the opposite. It identifies itself honestly:

```
Tallgrass/2026.2.25 (Research project; collecting public roll call vote data;
https://github.com/codechizel/tallgrass)
```

This tells the server operator exactly what's hitting their site, why, and where to find more information. If the operator has concerns, they can visit the project's GitHub page or contact the maintainers. Hiding behind a fake user agent would be like sneaking into a library wearing a disguise — unnecessary and suspicious.

**4. Scope: Take Only What You Need**

Tallgrass downloads only the pages it needs — bill listings, vote pages, and legislator profiles. It doesn't crawl the entire website, download images, or follow links into unrelated sections. The scraper is a specialist, not a hoarder.

### The robots.txt Standard

Most websites include a file called `robots.txt` at their root URL (e.g., `example.com/robots.txt`). This file tells automated programs which parts of the site they're welcome to access and which are off-limits.

The robots.txt standard has a surprisingly long history. It was proposed in 1994 by Martijn Koster on an internet mailing list, back when badly behaved early web crawlers were causing servers to crash. For 28 years it remained a voluntary convention — crawlers could read it, but nothing forced them to obey. In September 2022, the Internet Engineering Task Force published RFC 9309, giving robots.txt its first official status as an internet standard.

The key word here is *voluntary*. robots.txt is a polite request, not a locked door. A well-behaved scraper reads it and complies. A malicious one ignores it entirely. Some bad actors have even used robots.txt as a roadmap, targeting the pages a site specifically asked bots to avoid.

Tallgrass checks robots.txt and respects its directives. As of this writing, the Kansas Legislature's website does not block any of the pages Tallgrass accesses.

## Is This Legal?

A reasonable question. The short answer for U.S. government data: yes.

**Federal government works are in the public domain.** Under 17 U.S.C. Section 105, works prepared by federal employees as part of their official duties cannot be copyrighted. They belong to the public.

**State government data varies.** Section 105 applies only to the federal government. State and local government works *may* be subject to copyright, depending on the state. However, Kansas legislative records — journals, votes, bill text — are public records, and the state's open records laws support public access to this information.

**The Computer Fraud and Abuse Act (CFAA)** is the federal law most relevant to web scraping. Three landmark cases have shaped the legal landscape:

- **Van Buren v. United States (2021):** The Supreme Court held that the CFAA's "exceeds authorized access" provision applies only when someone accesses areas of a computer that are completely off-limits — not when they access public areas for purposes the operator might not have intended. This significantly narrowed the CFAA's reach.

- **hiQ Labs v. LinkedIn (2017–2022):** The Ninth Circuit ruled that scraping publicly accessible data does not violate the CFAA, because on a public website there are no access gates to circumvent.

- **Meta v. Bright Data (2024):** A court found that scraping public, non-login-protected data remains legal.

The bottom line: accessing publicly available data on a government website, without circumventing any login or access controls, is on very solid legal ground in the United States. Tallgrass accesses only public pages, identifies itself honestly, and does not circumvent any technical barriers.

None of this constitutes legal advice. If you're planning your own scraping project, consult an attorney for your specific situation.

## The Landscape: kslegislature.gov

Before we dive into the scraper itself, let's take a quick tour of the data source.

The Kansas Legislature's website organizes content by **biennium** — the two-year session. The current session (2025–2026, the 91st Legislature) lives at URL paths starting with `/li/b2025_26/`. Historical sessions use a slightly different pattern: the 2023–2024 session (90th Legislature) uses `/li_2024/b2023_24/`. Special sessions get their own prefix: the 2024 special session uses `/li_2024s/`.

Within each biennium, the structure looks like this:

```
/li/b2025_26/
  measures/
    bills/          ← index of all bills
      sb55/          ← one bill's page
        vote_view/   ← links to individual vote pages
  members/
    sen_smith_john/  ← one legislator's profile page
```

Each bill page lists the votes that occurred on that bill — amendments, committee reports, final passage. Each vote page shows the full roll call: every legislator's name and how they voted (Yea, Nay, Present and Passing, Not Voting, or Absent and Not Voting).

The site also provides a programmatic interface — the **KLISS API** (Kansas Legislature Information Systems and Services) — which returns bill metadata in JSON format. This is how the scraper efficiently identifies which of the ~1,200 bills introduced each session actually had recorded votes (about half of them).

### What Has Changed Over 15 Years

The website has not stayed the same. Between 2011 and 2026, three major changes affected how data is stored:

| Era | Sessions | Bill Discovery | Vote Format |
|-----|----------|---------------|-------------|
| **Early** | 2011–2014 (84th–85th) | JavaScript data files | ODT (OpenDocument) files |
| **Middle** | 2015–2020 (86th–88th) | JavaScript data files | HTML pages |
| **Modern** | 2021–2026 (89th–91st) | HTML listing pages | HTML pages |

This is why the scraper is more complex than you might expect. It's not one scraper — it's effectively three, unified behind a common interface that detects which era a session belongs to and routes accordingly. Chapter 3 covers the historical formats in detail.

## Tallgrass's Approach

Here's how Tallgrass puts the four principles of polite scraping into practice:

| Principle | Implementation | Codebase |
|-----------|---------------|----------|
| **Rate limiting** | 0.15s between requests (normal), 0.5s (retry waves) | `src/tallgrass/config.py` |
| **Caching** | SHA-256 URL hash → local files; never re-fetches | `src/tallgrass/scraper.py` (`_get()`) |
| **Identification** | Descriptive user agent with project URL | `src/tallgrass/config.py` (`USER_AGENT`) |
| **Scope** | Only bill pages, vote pages, and legislator profiles | `src/tallgrass/scraper.py` |

The scraper also uses a **concurrent-but-courteous** model. It sends up to five requests simultaneously (like five research assistants in the library at once), but each one respects the rate limit independently. The total load on the server never exceeds about 30 requests per second, and in practice it's usually much lower because parsing time creates natural gaps between requests.

The full scrape of a single biennium — discovering bills, filtering, parsing votes, and enriching legislator data — typically completes in about 15 minutes. That's fast enough to be practical, slow enough to be polite.

---

## Key Takeaway

Web scraping is how we get from "data exists on a website" to "data exists in a spreadsheet." Tallgrass scrapes the Kansas Legislature's public website using deliberate rate limits, comprehensive caching, honest identification, and minimal scope — the four principles of a polite robot. The data is public, the access is legal, and the process is designed to be invisible to human visitors.

---

*Terms introduced: web scraping, rate limiting, caching, user agent, robots.txt, KLISS API, CFAA*

*Next: [The Four-Phase Scraper Pipeline](ch02-four-phase-scraper-pipeline.md)*
