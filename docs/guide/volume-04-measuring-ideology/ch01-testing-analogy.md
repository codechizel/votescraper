# Chapter 1: The Testing Analogy: Bills as Questions, Legislators as Students

> *Imagine a standardized test where the questions are bills and the students are legislators. A "hard" question is a bill that even your allies might vote against. A "discriminating" question is one that sharply separates liberals from conservatives.*

---

## From Counting to Measuring

In Volume 3, we compressed 600 votes into a few numbers per legislator using PCA. That was a good start — PC1 separated Democrats from Republicans with striking clarity. But PCA has limits.

PCA gives you a **point estimate** and nothing else. It says "Senator Smith is at +0.82 on PC1" but offers no way to express uncertainty. It treats every vote equally — a 98-2 blowout counts as much as a 21-19 nail-biter. It can't handle missing data gracefully (what about the legislator who missed a third of the session?). And it doesn't tell you anything about the *bills themselves* — which ones are partisan wedges, and which ones are procedural nothings?

Item Response Theory solves all of these problems. It gives every legislator an ideology score *with* uncertainty bounds, automatically weights informative votes more heavily, handles absences natively, and simultaneously characterizes every bill. It's the difference between taking a photograph (PCA) and building a 3D model (IRT) — both show you something, but the model lets you walk around and examine the subject from every angle.

## The Framework: Testing as a Metaphor

IRT was originally developed not for politics but for **psychometrics** — the science of measuring mental traits. Its first major use was in standardized testing, where researchers needed to answer two questions at once: *how able is each student?* and *how difficult is each question?*

The breakthrough insight was that these two questions are intertwined. A student who aces every question is probably able, but you can only be sure if the questions vary in difficulty. A question that everyone gets right is probably easy, but you can only know that if the students vary in ability. **IRT estimates both simultaneously, each informing the other.**

The leap to politics was first made by the political scientists **Simon Jackman** (2000) and **Joshua Clinton, Simon Jackman, and Douglas Rivers** (2004), who realized that roll call voting has exactly the same structure as a test:

| Testing | Legislation |
|---------|-------------|
| Students | Legislators |
| Test questions | Roll call votes on bills |
| Ability (a latent trait) | Ideology (a latent trait) |
| Getting a question "right" | Voting Yea |
| Question difficulty | Bill difficulty |
| Question discrimination | Bill discrimination |
| The answer pattern reveals ability | The voting pattern reveals ideology |

The word "latent" is key. In both cases, the thing we want to measure — ability, ideology — is **unobservable**. You can't stick a thermometer in someone's brain and read their IQ. You can't scan a legislator and read their ideology score. All you can observe are the *responses* — answers to questions, votes on bills. IRT works backward from the responses to infer the hidden trait.

### A Brief History

**Georg Rasch**, a Danish mathematician, laid the foundations in 1960 with a model for reading comprehension tests. Rasch's key insight was that the probability of a correct answer depends on the *difference* between a student's ability and a question's difficulty. If the student is more able than the question is difficult, they'll probably get it right. If not, they'll probably get it wrong.

**Allan Birnbaum** (1968) extended Rasch's model by adding a second parameter: **discrimination**. Some questions are better at distinguishing strong students from weak ones — they have steep response curves. Others are poor discriminators — nearly everyone gets them right (or wrong) regardless of ability.

The two-parameter model (2PL) that Birnbaum formalized is the one Tallgrass uses. Political scientists adopted it because bills, like test questions, vary enormously in how well they separate liberals from conservatives. A bill to rename a post office has near-zero discrimination (everyone votes Yea). A bill to expand Medicaid has high discrimination (the vote splits cleanly along party lines).

## The Three Parameters

Every IRT model revolves around three quantities. Let's meet them through the testing analogy before introducing any notation.

### The Ideal Point (the "Ability")

In a standardized test, each student has an **ability level** — a single number that captures how strong they are in the subject being tested. A student with high math ability will get most math questions right; a student with low math ability will struggle.

In legislative voting, the equivalent is the **ideal point** — a single number that captures where a legislator falls on the ideological spectrum. A legislator with a high (positive) ideal point is conservative; one with a low (negative) ideal point is liberal. A legislator near zero is moderate — they sometimes vote with Republicans, sometimes with Democrats.

Think of the ideal point as a **GPS coordinate on a one-dimensional political map**:

```
Very Liberal    Moderate    Very Conservative
    -3    -2    -1    0    +1    +2    +3
     ←—————————————|———————————————→
```

Just as GPS tells you where you are in physical space, the ideal point tells you where a legislator is in ideological space. And just as two GPS coordinates close together mean two locations near each other, two ideal points close together mean two legislators who vote similarly.

The Greek letter **ξ** (xi, pronounced "zy" or "ksee") is the standard notation. Each legislator *i* has their own ideal point, written ξ_i.

### The Difficulty (the "Item Difficulty")

In testing, each question has a **difficulty** — a number that captures how hard it is. Easy questions have low difficulty (most students get them right). Hard questions have high difficulty (most students get them wrong).

In legislation, difficulty captures **how hard a bill is to pass** — or more precisely, where on the ideology spectrum a legislator needs to be for the bill to have a 50/50 chance of getting their Yea vote. A bill with low difficulty passes easily; even legislators at the liberal end of the spectrum vote Yea. A bill with high difficulty requires strong conservative conviction to support.

The Greek letter **α** (alpha) is the standard notation. Each bill *j* has its own difficulty, written α_j.

Think of difficulty as the **tipping point** on the ideological map — the position where a legislator is equally likely to vote Yea or Nay. Everyone to the right of the tipping point (more conservative) probably votes Yea. Everyone to the left (more liberal) probably votes Nay.

```
All Nay ←————|————→ All Yea
              α
         (tipping point)
```

### The Discrimination (the "Item Discrimination")

This is the parameter that makes IRT more powerful than simply counting votes.

In testing, a **discriminating** question is one that reliably separates strong students from weak ones. If students who know the material almost always get it right and students who don't almost always get it wrong, the question has high discrimination. But if strong students and weak students perform equally on a question — perhaps because it's confusingly worded, or tests something unrelated to the subject — then the question has low discrimination. It's noise, not signal.

In legislation, discrimination captures **how sharply a bill separates liberals from conservatives**. A bill with high discrimination is a partisan scalpel — nearly all conservatives vote one way and nearly all liberals vote the other. A bill with low discrimination is a blunt instrument — the vote doesn't track ideology at all (maybe it's a procedural motion, or a bipartisan spending bill).

The Greek letter **β** (beta) is the standard notation. Each bill *j* has its own discrimination, written β_j.

Here's why discrimination matters so much: **IRT automatically upweights discriminating bills and downweights non-discriminating ones.** A 98-2 vote on a resolution honoring veterans has near-zero discrimination — it tells you nothing about ideology, so IRT essentially ignores it. A 55-45 vote on Medicaid expansion has high discrimination — it cleanly separates the ideological spectrum, so IRT pays close attention. This happens naturally from the math, without any human judgment about which votes "count."

| Bill Type | Typical Vote Split | Discrimination | IRT Weight |
|-----------|-------------------|----------------|------------|
| Resolution honoring veterans | 98 Yea, 2 Nay | Very low (~0.1) | Essentially ignored |
| Routine budget authorization | 85 Yea, 15 Nay | Low (~0.5) | Minimal influence |
| Tax reform bill | 70 Yea, 30 Nay | Moderate (~1.0) | Contributes meaningfully |
| Medicaid expansion | 55 Yea, 45 Nay | High (~1.5) | Strong influence |
| Abortion restriction | 52 Yea, 48 Nay | Very high (~2.0+) | Maximum influence |

## Why Not Just Count Votes?

Before we go deeper into IRT, it's worth asking: why do we need all this mathematical machinery? Why not just count how often each legislator votes with their party?

The answer is that **raw vote counts mislead** in at least three important ways.

### Problem 1: Not All Votes Are Equally Informative

If a legislator votes Yea on a bill that passed 120-5, that tells you almost nothing. Everyone voted Yea. But if they voted Nay on a bill that passed 63-62, that's enormously informative — they were one of a handful who opposed a measure that barely passed. Raw vote counts treat both of these votes the same way. IRT does not.

### Problem 2: The 82% Base Rate

As we saw in Volume 3, about 82% of all votes in the Kansas Legislature are Yea. This means that two legislators who vote completely randomly would still agree about 70% of the time, just by the mathematics of probability. Raw agreement rates are inflated by this base rate. IRT accounts for it through the difficulty parameter — bills that nearly everyone passes get low discrimination, and bills that split the chamber get high discrimination.

### Problem 3: Missing Data

Legislators miss votes. Some miss a few; some miss dozens. If you count a legislator's Yea percentage, a legislator who voted Yea on 80 out of 100 votes and one who voted Yea on 8 out of 10 votes both show 80% — but you're much more confident about the first number. IRT handles this naturally: with only 10 votes, the ideal point estimate will have a **wide credible interval** (high uncertainty), while with 100 votes, the interval will be narrow. The model knows how much it knows.

## The Analogy Extended: A Concrete Example

Let's make this tangible with a miniature example. Imagine three legislators and three bills:

**The legislators:**
- **Rep. Adams** — a moderate Republican
- **Rep. Baker** — a conservative Republican
- **Sen. Chen** — a progressive Democrat

**The bills:**
1. **HB 100** — a noncontroversial infrastructure bill (low discrimination)
2. **HB 200** — a tax cut bill (moderate discrimination)
3. **SB 300** — a social policy bill (high discrimination)

Here's how they voted:

| | HB 100 (infrastructure) | HB 200 (tax cut) | SB 300 (social policy) |
|--|------------------------|-------------------|----------------------|
| **Rep. Adams** (moderate R) | Yea | Yea | Nay |
| **Rep. Baker** (conservative R) | Yea | Yea | Yea |
| **Sen. Chen** (progressive D) | Yea | Nay | Nay |

Now watch what IRT does with this:

- **HB 100** (everyone voted Yea) gets assigned **low discrimination**: it doesn't separate anyone. IRT learns almost nothing about ideology from this vote.
- **HB 200** (2 Yea, 1 Nay) gets **moderate discrimination**: it separates the Democrat from both Republicans, but it doesn't distinguish between the moderate and conservative Republican.
- **SB 300** (1 Yea, 2 Nay) gets **high discrimination**: it separates all three legislators. Only the most conservative legislator voted Yea.

From this pattern, IRT infers the ideal points:
- Sen. Chen ≈ −1.0 (liberal)
- Rep. Adams ≈ +0.5 (moderate conservative)
- Rep. Baker ≈ +1.5 (strong conservative)

It also infers the bill parameters:
- HB 100: low α (easy to pass), low β (doesn't discriminate)
- HB 200: moderate α, moderate β
- SB 300: high α (hard to pass), high β (strongly discriminating)

The crucial point: **IRT figured out both the legislators' positions and the bills' properties at the same time**, from nothing but the pattern of Yea and Nay votes. Nobody told the model that SB 300 was a social policy bill or that Rep. Baker was conservative. The model *discovered* these relationships from the data.

## What IRT Gives You That PCA Doesn't

Let's be concrete about the upgrade:

| Feature | PCA | IRT |
|---------|-----|-----|
| Ideology score per legislator | Yes (PC1 score) | Yes (ideal point ξ) |
| Uncertainty for that score | No | Yes (credible interval) |
| Bill-level characterization | Loadings (hard to interpret) | Difficulty α and discrimination β (directly meaningful) |
| Handling of missing votes | Requires imputation | Native (absences simply absent from likelihood) |
| Automatic vote weighting | Implicit (via variance) | Explicit (via discrimination β) |
| Probabilistic interpretation | No | Yes: P(Yea) for any legislator-bill pair |
| Extensible to multiple dimensions | Yes (PC2, PC3...) | Yes (2D IRT, hierarchical models) |

The last row is worth highlighting. PCA gives you PC1, PC2, PC3 as separate, independent scores. IRT's 2D model (Chapter 4) gives you a **joint** two-dimensional estimate where both dimensions inform each other — and it gives you the uncertainty for both dimensions simultaneously.

## Where IRT Lives in the Code

The IRT models are implemented across four pipeline phases:

- `analysis/05_irt/irt.py` — the 1D flat IRT model (Chapters 2-3)
- `analysis/06_irt_2d/irt_2d.py` — the 2D flat IRT model (Chapter 4)
- `analysis/07_hierarchical/hierarchical.py` — the hierarchical 1D model (Chapter 5)
- `analysis/07b_hierarchical_2d/hierarchical_2d.py` — the hierarchical 2D model (Chapter 5)
- `analysis/canonical_ideal_points.py` — the routing system that picks the best score (Chapter 7)

## The Road Ahead

The rest of this volume walks through IRT in detail:

- **Chapter 2** builds the 1D model step by step, from the equation to Bayesian estimation to a fully worked Kansas example.
- **Chapter 3** explains why the model can't tell left from right without help, and how "anchoring" solves this.
- **Chapter 4** extends to 2D, adding a second dimension that captures establishment-versus-maverick dynamics.
- **Chapter 5** introduces hierarchical models, which use party membership as an informative starting point.
- **Chapter 6** dives into the seven identification strategies — different ways to tell the model which direction is "conservative."
- **Chapter 7** explains how the pipeline chooses the single best ideology score from the multiple models it fits.

Each chapter builds on the last. By the end, you'll understand not just the number that Tallgrass assigns to each legislator, but the entire chain of reasoning that produces it.

---

## Key Takeaway

Item Response Theory treats legislative voting like a standardized test: bills are questions, legislators are students, and the pattern of Yea/Nay reveals both where each legislator stands on the ideology spectrum and how informative each bill is. Unlike simpler approaches, IRT automatically weights informative votes more heavily, handles missing data gracefully, and provides honest uncertainty bounds on every score.

---

*Terms introduced: Item Response Theory (IRT), psychometrics, ideal point (ξ), difficulty (α), discrimination (β), latent trait, two-parameter logistic model (2PL), Rasch model*

*Next: [The 1D IRT Model — Step by Step](ch02-1d-irt-model.md)*
