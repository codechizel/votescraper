"""Centralized tuning parameters for the analysis pipeline.

Single source of truth for thresholds that affect vote filtering,
legislator inclusion, and sensitivity analysis across all phases.
Change values here to tune the entire pipeline. When we stand up the
Django dashboard, these will be read from the database instead.
"""

# ── Vote Filtering ────────────────────────────────────────────────────────

CONTESTED_THRESHOLD = 0.10
"""Drop votes where the minority side is below this fraction.

A vote with 97.5% Yea / 2.5% Nay has minority_frac = 0.025.  At the
default 0.025, votes must have at least 2.5% dissent to be included.
Raising this (e.g., to 0.10) keeps only more competitive votes.

VoteView uses 2.5% as the standard; 10% is the aggressive alternative
used in sensitivity analyses.
"""

SENSITIVITY_THRESHOLD = 0.10
"""Alternative contested threshold for mandatory sensitivity analysis.

Every core model re-runs at this threshold and compares ideal points /
loadings against the default.  Correlation > 0.95 = robust.
"""

# ── Legislator Filtering ──────────────────────────────────────────────────

MIN_VOTES = 20
"""Minimum substantive votes for a legislator to be included in analysis.

Legislators with fewer than this many non-missing votes are dropped.
Too low → noisy estimates; too high → losing short-tenure members.
"""

# ── Visual Styling ────────────────────────────────────────────────────────

PARTY_COLORS: dict[str, str] = {
    "Republican": "#E81B23",
    "Democrat": "#0015BC",
    "Independent": "#999999",
}
"""Standard party color scheme used across all analysis visualizations."""

# ── Supermajority Detection ───────────────────────────────────────────────

SUPERMAJORITY_THRESHOLD = 0.70
"""Majority party fraction that triggers adaptive MCMC tuning.

When the majority party exceeds this fraction of the chamber, 2D IRT
phases double N_TUNE (ADR-0112) and use tighter beta initialization
from PCA loadings.
"""

# ── Bill Discrimination ───────────────────────────────────────────────────

HIGH_DISC_THRESHOLD = 1.5
"""Bills with |beta| above this are highly discriminating (party-line splits)."""

LOW_DISC_THRESHOLD = 0.5
"""Bills with |beta| below this have low discrimination (bipartisan or trivial)."""

# ── External Validation ───────────────────────────────────────────────────

STRONG_CORRELATION = 0.90
"""Pearson r threshold for strong agreement with external scores (SM, DIME, W-NOMINATE)."""

GOOD_CORRELATION = 0.85
"""Pearson r threshold for good agreement with external scores."""

CONCERN_CORRELATION = 0.70
"""Pearson r below this flags potential methodology concerns."""

WNOM_GATE_DELTA: float = 0.10
"""Minimum improvement in |Pearson r| vs W-NOMINATE to trigger a canonical dimension swap."""

WNOM_GATE_MIN_R: float = 0.70
"""Minimum W-NOMINATE correlation for any candidate canonical source."""
