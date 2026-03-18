# Chapter 2: A Quick Tour of the Kansas Legislature

> *Before you can analyze a legislature, you need to understand how it works. This chapter is a field guide to the institution that produces the data.*

---

## Two Chambers, One Building

The Kansas Legislature is bicameral — it has two chambers that meet in the Kansas State Capitol in Topeka.

**The House of Representatives** has 125 members, each representing a single district. House members serve two-year terms, and all 125 seats are on the ballot in every even-year election. This means the entire House can turn over in a single cycle, though in practice most incumbents win re-election.

**The Senate** has 40 members, also from single-member districts, but serving four-year terms. Unlike the U.S. Senate, Kansas Senate elections are not staggered — all 40 seats are on the ballot in the same election, every four years, in even-numbered non-presidential years (2022, 2026, 2030, etc.). A Senator elected in 2022 won't face voters again until 2026. This longer term insulates Senators from short-term political winds and contributes to a different dynamic than the House.

Neither chamber has term limits. A legislator can serve as long as voters keep electing them.

## The Biennium

Kansas organizes its legislative work in two-year cycles called **bienniums**. The 91st Legislature, for example, covers 2025 and 2026. The legislature convenes on the second Monday of January each year, and in even-numbered years the regular session is limited to 90 calendar days (though this can be extended by a two-thirds vote).

Each biennium gets a number — the 91st Legislature in 2025-2026, the 90th in 2023-2024, and so on back to the 1st Territorial Legislature in 1855. The formula is straightforward: take the start year, subtract 1879, divide by 2, and add 18. Tallgrass uses this numbering as its primary organizational unit. When we say "the 79th Legislature" or "the 84th-91st," we're referencing specific two-year windows of legislative activity.

Here's the full range of Tallgrass data:

| Legislature | Years | House (R-D) | Senate (R-D) | Notes |
|------------|-------|-------------|--------------|-------|
| 78th | 1999-2000 | — | — | KanFocus data only |
| 79th | 2001-2002 | — | — | KanFocus data only |
| 80th | 2003-2004 | — | — | KanFocus data only |
| 81st | 2005-2006 | — | — | KanFocus data only |
| 82nd | 2007-2008 | — | — | KanFocus data only |
| 83rd | 2009-2010 | — | — | KanFocus data only |
| 84th | 2011-2012 | 92-33 | 31-9 | First kslegislature.gov session; ODT vote files |
| 85th | 2013-2014 | 92-33 | 31-9 | Brownback era; ODT vote files |
| 86th | 2015-2016 | 97-28 | 31-9 | Peak Republican supermajority in House |
| 87th | 2017-2018 | 85-40 | 31-9 | Moderate resurgence |
| 88th | 2019-2020 | 86-39 | 29-11 | COVID special session |
| 89th | 2021-2022 | 86-39 | 29-11 | Full HTML scraping begins |
| 90th | 2023-2024 | 85-40 | 31-9 | 2024 special session |
| 91st | 2025-2026 | 87-37 | 31-9 | Current legislature |

A few things jump out from this table. First, Republicans have controlled both chambers for the entire period — and it's not close. The House has ranged from a comfortable majority (85-40) to a near-veto-proof supermajority (97-28). The Senate has been even more lopsided, with Republicans holding 29 to 31 of 40 seats in every session.

This matters enormously for the statistical analysis. When one party holds 75% or more of the seats, the mathematics of ideology estimation become harder. We'll return to this problem — called the "horseshoe effect" — repeatedly throughout this guide.

## How a Bill Becomes a Vote

A simplified version of the Kansas legislative process:

1. **Introduction.** A legislator (or committee) introduces a bill. It receives a number — SB for Senate Bills, HB for House Bills, SCR for Senate Concurrent Resolutions, and so on.

2. **Committee.** The bill is referred to a standing committee (there are about 26 in the House and 16 in the Senate). The committee holds hearings, may amend the bill, and decides whether to send it to the full chamber. Many bills die in committee and never receive a floor vote.

3. **Floor Action.** Bills that survive committee reach the full chamber for debate and voting. This is where roll call data comes from. A bill may face multiple votes: a committee report vote, amendments, final action, emergency final action (which bypasses the normal scheduling delay), and more.

4. **Second Chamber.** If a bill passes one chamber, it crosses to the other and goes through the same committee-and-floor process. If the second chamber amends the bill, it may go to a conference committee to reconcile differences.

5. **Governor.** Bills that pass both chambers go to the Governor, who can sign them into law or veto them. A veto can be overridden by a two-thirds vote in both chambers — and veto override votes are some of the most informative data points in the pipeline, because they force legislators to go on record against their own party's Governor (or in favor of the opposing party's).

Not every step in this process produces a recorded vote. Committee deliberations, voice votes ("all in favor say aye"), and leadership negotiations are invisible to Tallgrass. What we capture is the *recorded roll call* — the moment when every legislator's position is individually documented.

## Types of Votes

Not all roll calls are created equal. Tallgrass classifies each vote into a type based on the motion text:

| Vote Type | What It Is | Why It Matters |
|-----------|-----------|---------------|
| **Final Action** | The chamber's up-or-down vote on a bill | The canonical passage vote; the one most people think of |
| **Emergency Final Action** | Final passage bypassing the normal scheduling delay | Same simple-majority threshold as regular Final Action, but expedited procedure |
| **Committee of the Whole** | Amendment votes during floor debate | Often more revealing than final passage — amendments are where the real fights happen |
| **Consent Calendar** | Non-controversial bills passed in bulk; any single member can object and pull a bill off | Near-zero information content; expected to be unanimous |
| **Veto Override** | Overriding the Governor's veto | Extremely revealing; forces cross-party coalition building |
| **Conference Committee** | Accepting the reconciled version | Reveals willingness to compromise |
| **Concurrence** | Accepting the other chamber's changes | Reveals inter-chamber dynamics |
| **Procedural Motion** | Rules changes, adjournment, etc. | Sometimes informative (e.g., motion to table a bill), sometimes not |

The pipeline doesn't treat all of these equally. Near-unanimous votes (where less than 2.5% of the chamber dissents) are filtered out early, because a 120-5 vote tells you almost nothing about ideology — everyone agreed. The contested votes — the 60-65 nailbiters, the 80-45 party-line splits — are where the statistical signal lives.

## The Players

A few structural features of the Kansas Legislature shape the data in ways that matter for analysis:

### Party Leadership

The Speaker of the House and the Senate President wield significant power over the legislative agenda. They control which bills reach the floor, the order of business, and committee assignments. In a body where one party holds a supermajority, leadership's preferences strongly shape which votes even occur.

This means the roll call record is not a random sample of possible legislation. It is a curated set of votes that leadership *chose* to bring to the floor. This selection effect is invisible in the data but always present.

### The Moderate-Conservative Divide

Kansas Republican politics has been marked by a persistent tension between moderate and conservative factions — particularly during the Brownback era (2011-2018), when the Governor's aggressive tax-cut agenda split the party. In the 86th Legislature (2015-2016), the House had 97 Republicans and only 28 Democrats. The real legislative battles were fought within the Republican caucus, not between the parties.

This is exactly the scenario that makes ideology estimation difficult. When the minority party is small enough, the dominant axis of conflict shifts from "Republican vs. Democrat" to "moderate Republican vs. conservative Republican." Some of our most sophisticated statistical machinery exists to handle this case.

### Turnover and Redistricting

Members come and go. A typical biennium sees 15-25% turnover in the House (thanks to two-year terms). Senate turnover is concentrated in election-year bienniums — all 40 seats are up at once every four years, with only special-election vacancies in between. Redistricting after the 2010 and 2020 censuses reshuffled district boundaries, complicating cross-session comparisons.

Tallgrass tracks individual legislators across sessions using a combination of URL-based "slugs" (like `rep_smith_john_1`) and OpenStates unique identifiers. This allows us to follow a legislator's ideological trajectory from their first term to their last — one of the most revealing analyses the pipeline performs.

## The Numbers in Context

Here's a snapshot of the 91st Legislature (2025-2026) — the most recent session in the dataset — to give you a sense of scale:

| Metric | House | Senate | Total |
|--------|-------|--------|-------|
| Members | 125 | 40 | 165 |
| Republicans | ~87 | 31 | ~118 |
| Democrats | ~37 | 9 | ~46 |
| Roll call votes | ~600 | ~500 | ~1,105 |
| Individual vote records | ~55,000 | ~40,000 | ~95,000 |
| Bills with recorded votes | ~350 | ~250 | ~400 (many cross chambers) |
| Yea votes (% of total) | ~82% | ~82% | ~82% |

That last row — 82% Yea — is one of the most important numbers in the entire project. It means that on a typical vote, more than four out of five legislators vote Yes. This is normal for a state legislature. Most bills that reach a floor vote have already survived committee, leadership gatekeeping, and informal negotiations. By the time they're voted on, they have broad support.

But it creates a statistical headache. If two legislators both vote Yea 82% of the time and you just count how often they agree, you'll get about 70% agreement *by pure chance* — even if they have nothing in common ideologically. The pipeline uses a statistic called Cohen's Kappa to correct for this, but the base rate is a constant background challenge. We'll dig into it in Volume 3.

## Why Kansas?

A reasonable question: why build all of this for Kansas?

Several reasons. Kansas has clean, publicly accessible roll call data going back to the late 1990s. Its legislature is large enough to support meaningful statistical analysis (165 members total) but small enough that the full pipeline runs in under an hour. Its persistent Republican supermajority creates exactly the kind of challenging statistical environment where simple methods fail and sophisticated ones earn their keep. And its internal party dynamics — the moderate-conservative split, the Brownback tax experiment, the suburban political realignment — make it a genuinely interesting subject of study.

But the architecture of Tallgrass is not Kansas-specific. The scraper uses a modular "state adapter" pattern, and the analysis pipeline operates on a generic vote matrix format. Extending coverage to other states is an engineering task, not a statistical one.

## What Comes Next

Now that you have a mental picture of the Kansas Legislature — its size, its structure, its party dynamics, and the types of votes it produces — we can turn to the data itself. The next chapter examines what roll call votes actually reveal about a legislator's beliefs, and equally important, what they *don't*.

---

## Key Takeaway

The Kansas Legislature has 125 House members and 40 Senators, organized into two-year bienniums, with Republicans holding supermajorities throughout the 1999-2026 period. This persistent one-party dominance shapes every aspect of the statistical analysis — from which votes are informative, to which mathematical models are needed, to how we interpret the results.

---

*Terms introduced: biennium, bicameral, supermajority, horseshoe effect (preview), selection effect, turnover, redistricting*

*Next: [What Roll Call Votes Tell Us (and What They Don't)](ch03-roll-call-votes.md)*
