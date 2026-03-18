"""
Kansas Legislature — Beta-Binomial Bayesian Party Loyalty (Phase 7b)

Applies Bayesian shrinkage to CQ-standard party unity scores using a
Beta-Binomial conjugate model. Legislators with few party votes are pulled
toward their party's average loyalty — producing more reliable estimates with
calibrated uncertainty. Uses empirical Bayes (method of moments) for instant,
closed-form posteriors. No MCMC required.

Usage:
  uv run python analysis/beta_binomial.py [--session 2025-26]

Outputs (in results/<session>/beta_binomial/<date>/):
  - data/:   Parquet files (posterior_loyalty per chamber)
  - plots/:  PNG visualizations (shrinkage arrows, CIs, posteriors, scatter)
  - filtering_manifest.json, run_info.json, run_log.txt
  - beta_binomial_report.html
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir

try:
    from analysis.beta_binomial_report import build_beta_binomial_report
except ModuleNotFoundError:
    from beta_binomial_report import build_beta_binomial_report  # type: ignore[no-redef]

try:
    from analysis.phase_utils import print_header, save_fig
except ImportError:
    from phase_utils import print_header, save_fig  # type: ignore[no-redef]

try:
    from analysis.tuning import PARTY_COLORS
except ImportError:
    from tuning import PARTY_COLORS  # type: ignore[no-redef]

# ── Primer ───────────────────────────────────────────────────────────────────

BETA_BINOMIAL_PRIMER = """\
# Beta-Binomial Bayesian Party Loyalty

## Purpose

Produces more reliable party loyalty estimates by applying Bayesian shrinkage to
the raw CQ-standard party unity scores from the indices phase. Legislators with
few party votes get pulled toward their party's average — preventing small samples
from producing misleadingly extreme scores.

## Method

The Beta-Binomial model is a natural fit for "number of successes out of N trials"
data. For each party-chamber group:

1. **Estimate the group prior.** Method of moments fits a Beta(alpha, beta) to
   the observed loyalty rates within the group, capturing the party's average
   loyalty and how spread out its members are.

2. **Update each legislator.** The Beta prior combined with each legislator's
   Binomial data yields a Beta posterior — available in closed form (no MCMC).

3. **Compute summaries.** Posterior mean, median, 95% credible interval, and
   shrinkage factor for each legislator.

The posterior mean is a weighted average of the raw rate and the party mean:
- Legislators with many votes → posterior ≈ raw rate (data dominates)
- Legislators with few votes → posterior ≈ party mean (prior dominates)

## Inputs

Reads from `results/<session>/indices/latest/data/`:
- `party_unity_{chamber}.parquet` — CQ-standard unity scores with vote counts

## Outputs

All outputs land in `results/<session>/beta_binomial/<date>/`:

### `data/` — Parquet intermediates

| File | Description |
|------|-------------|
| `posterior_loyalty_{chamber}.parquet` | Per-legislator Bayesian loyalty estimates |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `shrinkage_arrows_{chamber}.png` | "How Much Did Bayesian Analysis Change Each Estimate?" |
| `credible_intervals_{chamber}.png` | "How Certain Are We About Each Legislator's Loyalty?" |
| `posterior_distributions_{chamber}.png` | "Three Legislators, Three Levels of Certainty" |
| `raw_vs_bayesian_{chamber}.png` | "Before and After: Raw vs. Bayesian Loyalty" |

## Interpretation Guide

- **Posterior mean** is the best single estimate of a legislator's true loyalty.
- **Credible interval width** reflects uncertainty — wider = less certain.
- **Shrinkage** ranges from 0 (no change from raw) to 1 (fully pulled to party mean).
  Typical values are 0.05-0.20.
- Legislators who still appear as mavericks after shrinkage are reliably independent —
  their low loyalty isn't a small-sample artifact.

## Caveats

- Empirical Bayes underestimates hyperparameter uncertainty (treats the prior as
  known). For formal inference, use a full hierarchical model (roadmap item #3).
- Exchangeability assumption: treats all legislators within a party-chamber group
  as draws from the same distribution. This is approximately true but ignores
  district-level or caucus-level structure.
- Priors are estimated per party per chamber (4 groups). With only ~10 Senate
  Democrats, the Democratic prior may be imprecise.
"""

# ── Constants ────────────────────────────────────────────────────────────────

MIN_PARTY_VOTES = 3
CI_LEVEL = 0.95
FALLBACK_ALPHA = 1.0
FALLBACK_BETA = 1.0


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature Beta-Binomial Party Loyalty")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--indices-dir", default=None, help="Override indices results directory")
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    return parser.parse_args()


# ── Core: Empirical Bayes ────────────────────────────────────────────────────


def estimate_beta_params(
    y: np.ndarray,
    n: np.ndarray,
) -> tuple[float, float]:
    """Estimate Beta(alpha, beta) prior from observed party-line counts.

    Uses method of moments on the observed rates y_i / n_i.

    Returns (alpha, beta). Falls back to Beta(1, 1) (uniform) if variance
    is too high for the Beta family or if data is degenerate.
    """
    if len(y) < 2:
        return (FALLBACK_ALPHA, FALLBACK_BETA)

    rates = y / n
    mu = float(np.mean(rates))
    var = float(np.var(rates, ddof=1))

    # Degenerate cases: mu at boundary makes Beta undefined
    if mu <= 0 or mu >= 1:
        return (FALLBACK_ALPHA, FALLBACK_BETA)

    if var < 1e-12:
        # All legislators have identical rates — use a tight prior around mu
        # kappa ~ 100 gives a tight Beta centered on mu
        kappa = 100.0
        return (max(mu * kappa, 0.5), max((1 - mu) * kappa, 0.5))

    # Method of moments: var = mu*(1-mu)/(alpha+beta+1)
    # So alpha+beta = mu*(1-mu)/var - 1
    if var >= mu * (1 - mu):
        # Variance exceeds Beta maximum — use weakly informative prior
        return (FALLBACK_ALPHA, FALLBACK_BETA)

    common = mu * (1 - mu) / var - 1
    alpha = mu * common
    beta = (1 - mu) * common

    # Clamp to minimum 0.5 to avoid degenerate Beta distributions
    clamped_alpha = max(alpha, 0.5)
    clamped_beta = max(beta, 0.5)
    if alpha < 0.5 or beta < 0.5:
        import warnings

        warnings.warn(
            f"Beta-Binomial MoM produced alpha={alpha:.4f}, beta={beta:.4f} "
            f"(clamped to {clamped_alpha:.4f}, {clamped_beta:.4f}). "
            f"This occurs when party loyalty is very high (mu={mu:.4f}).",
            stacklevel=2,
        )
    return (clamped_alpha, clamped_beta)


def tarone_test(y: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """Tarone's score test for Beta-Binomial overdispersion.

    Tests H0: data are Binomial (no overdispersion) against
    H1: data are Beta-Binomial (overdispersion present).

    Returns (z_statistic, p_value). Large positive Z rejects H0.
    Reference: Tarone (1979), Biometrika 66(3), 585-590.
    """
    p_hat = y.sum() / n.sum()
    expected = n * p_hat
    variance = n * p_hat * (1 - p_hat)
    numerator = float(((y - expected) ** 2 - variance).sum())
    denominator = float(np.sqrt(2 * (variance**2).sum()))
    if denominator < 1e-12:
        return (0.0, 1.0)
    z = numerator / denominator
    p_value = float(1 - sp_stats.norm.cdf(z))
    return (z, p_value)


def compute_bayesian_loyalty(
    unity_df: pl.DataFrame,
    chamber: str,
) -> pl.DataFrame:
    """Compute Bayesian loyalty posteriors for all legislators in one chamber.

    Fits per-party priors via empirical Bayes (method of moments), then
    computes closed-form Beta posteriors for each legislator.

    Returns a DataFrame with posterior stats, one row per legislator.
    """
    if unity_df.height == 0:
        return pl.DataFrame()

    ci_tail = (1 - CI_LEVEL) / 2
    results: list[dict] = []

    for party in ["Republican", "Democrat"]:
        party_df = unity_df.filter(
            (pl.col("party") == party) & (pl.col("party_votes_present") >= MIN_PARTY_VOTES)
        )
        if party_df.height < 2:
            print(f"  {chamber} {party}: only {party_df.height} legislators — skipping")
            continue

        y = party_df["votes_with_party"].to_numpy().astype(float)
        n = party_df["party_votes_present"].to_numpy().astype(float)

        alpha_prior, beta_prior = estimate_beta_params(y, n)
        prior_mean = alpha_prior / (alpha_prior + beta_prior)
        prior_kappa = alpha_prior + beta_prior

        # Overdispersion test: is Beta-Binomial justified over plain Binomial?
        tarone_z, tarone_p = tarone_test(y.astype(int), n.astype(int))

        print(
            f"  {chamber} {party}: alpha={alpha_prior:.2f}, beta={beta_prior:.2f}, "
            f"prior_mean={prior_mean:.4f}, kappa={prior_kappa:.1f} "
            f"(n={party_df.height} legislators)"
        )
        print(
            f"    Tarone's overdispersion test: Z={tarone_z:.2f}, p={tarone_p:.4f}"
            f"{' — overdispersion confirmed' if tarone_p < 0.05 else ' — not significant'}"
        )

        for row in party_df.iter_rows(named=True):
            yi = float(row["votes_with_party"])
            ni = float(row["party_votes_present"])

            alpha_post = alpha_prior + yi
            beta_post = beta_prior + (ni - yi)

            post_mean = alpha_post / (alpha_post + beta_post)
            post_median = float(sp_stats.beta.median(alpha_post, beta_post))
            ci_lower, ci_upper = sp_stats.beta.ppf([ci_tail, 1 - ci_tail], alpha_post, beta_post)

            raw_rate = yi / ni if ni > 0 else float("nan")
            shrinkage = (alpha_prior + beta_prior) / (alpha_prior + beta_prior + ni)

            results.append(
                {
                    "legislator_slug": row["legislator_slug"],
                    "party": party,
                    "full_name": row.get("full_name", ""),
                    "district": row.get("district", ""),
                    "raw_loyalty": raw_rate,
                    "posterior_mean": post_mean,
                    "posterior_median": post_median,
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "ci_width": float(ci_upper - ci_lower),
                    "shrinkage": shrinkage,
                    "votes_with_party": int(yi),
                    "n_party_votes": int(ni),
                    "alpha_prior": alpha_prior,
                    "beta_prior": beta_prior,
                    "prior_mean": prior_mean,
                    "prior_kappa": prior_kappa,
                }
            )

    if not results:
        return pl.DataFrame()

    df = pl.DataFrame(results).sort("posterior_mean")
    return df


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_shrinkage_arrows(
    loyalty_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Arrow plot: raw → posterior, Y-axis = sample size.

    The signature visualization — shows how much each estimate moved and that
    low-N legislators move the most.
    """
    if loyalty_df.height == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    for party in ["Republican", "Democrat"]:
        sub = loyalty_df.filter(pl.col("party") == party)
        if sub.height == 0:
            continue

        color = PARTY_COLORS[party]
        raw = sub["raw_loyalty"].to_numpy()
        post = sub["posterior_mean"].to_numpy()
        n_votes = sub["n_party_votes"].to_numpy()

        # Draw arrows
        for r, p, nv in zip(raw, post, n_votes):
            ax.annotate(
                "",
                xy=(p, nv),
                xytext=(r, nv),
                arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.5, "linewidth": 1.2},
            )

        # Raw dots
        ax.scatter(raw, n_votes, color=color, alpha=0.4, s=20, zorder=3)
        # Posterior dots
        ax.scatter(
            post,
            n_votes,
            color=color,
            alpha=0.8,
            s=30,
            edgecolors="white",
            linewidth=0.5,
            zorder=4,
            label=party,
        )

    # Prior mean line
    prior_means = loyalty_df["prior_mean"].unique().to_list()
    for pm in prior_means:
        ax.axvline(pm, color="#888888", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xlabel("Party Loyalty Rate")
    ax.set_ylabel("Number of Party Votes (sample size)")
    ax.set_title(
        f"{chamber} — How Much Did Bayesian Analysis Change Each Estimate?\n"
        "Arrows show movement from raw rate to Bayesian estimate. "
        "Low-sample legislators move the most.",
        fontsize=12,
        fontweight="bold",
    )

    # Annotation
    ax.text(
        0.02,
        0.98,
        "Faded dots = raw rate\nBright dots = Bayesian estimate\nLonger arrows = more shrinkage",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "alpha": 0.9},
    )

    ax.legend(fontsize=10, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"shrinkage_arrows_{chamber.lower()}.png")


def plot_credible_intervals(
    loyalty_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Forest plot: all legislators with 95% credible interval bars, party-colored."""
    if loyalty_df.height == 0:
        return

    sorted_df = loyalty_df.sort("posterior_mean")
    names = sorted_df["full_name"].to_list()
    post_mean = sorted_df["posterior_mean"].to_numpy()
    ci_lo = sorted_df["ci_lower"].to_numpy()
    ci_hi = sorted_df["ci_upper"].to_numpy()
    parties = sorted_df["party"].to_list()
    colors = [PARTY_COLORS.get(p, "#888888") for p in parties]

    fig_height = max(8, len(names) * 0.20)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y = np.arange(len(names))

    # CI bars
    for i in range(len(names)):
        ax.plot([ci_lo[i], ci_hi[i]], [y[i], y[i]], color=colors[i], linewidth=2, alpha=0.6)
    # Posterior means
    ax.scatter(post_mean, y, c=colors, s=30, zorder=5, edgecolors="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Party Loyalty (posterior mean with 95% credible interval)")
    ax.set_title(
        f"{chamber} — How Certain Are We About Each Legislator's Loyalty?",
        fontsize=13,
        fontweight="bold",
    )

    # Annotate widest and narrowest CI
    ci_widths = ci_hi - ci_lo
    widest_idx = int(np.argmax(ci_widths))
    narrowest_idx = int(np.argmin(ci_widths))

    ax.annotate(
        f"Widest CI: {ci_widths[widest_idx]:.3f}",
        (post_mean[widest_idx], y[widest_idx]),
        fontsize=7,
        fontweight="bold",
        xytext=(10, 0),
        textcoords="offset points",
    )
    ax.annotate(
        f"Narrowest: {ci_widths[narrowest_idx]:.3f}",
        (post_mean[narrowest_idx], y[narrowest_idx]),
        fontsize=7,
        fontweight="bold",
        xytext=(10, 0),
        textcoords="offset points",
    )

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"credible_intervals_{chamber.lower()}.png")


def plot_posterior_distributions(
    loyalty_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Beta PDF curves for selected legislators: most shrunk, least shrunk, lowest loyalty."""
    if loyalty_df.height < 3:
        return

    # Pick 3 interesting legislators
    most_shrunk = loyalty_df.sort("shrinkage", descending=True).row(0, named=True)
    least_shrunk = loyalty_df.sort("shrinkage").row(0, named=True)
    lowest_loyalty = loyalty_df.sort("posterior_mean").row(0, named=True)

    # Deduplicate (in case same legislator appears in multiple categories)
    selected = {}
    for label, row in [
        ("Most shrunk", most_shrunk),
        ("Least shrunk", least_shrunk),
        ("Lowest loyalty", lowest_loyalty),
    ]:
        slug = row["legislator_slug"]
        if slug not in selected:
            selected[slug] = (label, row)

    if len(selected) < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.linspace(0, 1, 300)
    colors_cycle = ["#E81B23", "#0015BC", "#2CA02C", "#D62728"]

    for i, (slug, (label, row)) in enumerate(selected.items()):
        alpha_post = row["alpha_prior"] + row["votes_with_party"]
        beta_post = row["beta_prior"] + (row["n_party_votes"] - row["votes_with_party"])

        pdf = sp_stats.beta.pdf(x, alpha_post, beta_post)
        name = row["full_name"] or slug
        color = colors_cycle[i % len(colors_cycle)]

        ax.plot(
            x, pdf, color=color, linewidth=2, label=f"{name} — {label} (n={row['n_party_votes']})"
        )
        ax.fill_between(x, pdf, alpha=0.1, color=color)

    ax.set_xlabel("Party Loyalty Rate")
    ax.set_ylabel("Density")
    ax.set_title(
        f"{chamber} — Three Legislators, Three Levels of Certainty\n"
        "Wider curves = less certain; taller and narrower = more data, more certain",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"posterior_distributions_{chamber.lower()}.png")


def plot_raw_vs_bayesian(
    loyalty_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Scatter: raw (x) vs posterior mean (y), size = n_party_votes, 45-degree reference."""
    if loyalty_df.height == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    for party in ["Republican", "Democrat"]:
        sub = loyalty_df.filter(pl.col("party") == party)
        if sub.height == 0:
            continue

        color = PARTY_COLORS[party]
        raw = sub["raw_loyalty"].to_numpy()
        post = sub["posterior_mean"].to_numpy()
        sizes = 20 + sub["n_party_votes"].to_numpy() * 0.8

        ax.scatter(
            raw,
            post,
            c=color,
            s=sizes,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            label=party,
        )

    # 45-degree reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="No change line")

    ax.set_xlabel("Raw Party Loyalty (CQ standard)")
    ax.set_ylabel("Bayesian Posterior Mean")
    ax.set_title(
        f"{chamber} — Before and After: Raw vs. Bayesian Loyalty\n"
        "Points below the line were pulled down (toward party mean); "
        "above = pulled up. Larger dots = more votes.",
        fontsize=12,
        fontweight="bold",
    )

    ax.text(
        0.02,
        0.98,
        "Dot size = number of party votes\n"
        "Points near the diagonal barely changed;\n"
        "points far from it were heavily shrunk.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "alpha": 0.9},
    )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"raw_vs_bayesian_{chamber.lower()}.png")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from tallgrass.session import KSSession

    ks = KSSession.from_session_string(args.session)
    results_root = ks.results_dir

    indices_dir = resolve_upstream_dir(
        "13_indices",
        results_root,
        args.run_id,
        Path(args.indices_dir) if args.indices_dir else None,
    )

    with RunContext(
        session=args.session,
        analysis_name="14_beta_binomial",
        params=vars(args),
        primer=BETA_BINOMIAL_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"KS Legislature Beta-Binomial Party Loyalty — Session {args.session}")
        print(f"Indices:  {indices_dir}")
        print(f"Output:   {ctx.run_dir}")

        # ── Load data ──
        print_header("LOADING DATA")
        chambers_found: list[str] = []
        chamber_results: dict[str, dict] = {}

        for chamber in ["House", "Senate"]:
            ch = chamber.lower()
            unity_path = indices_dir / "data" / f"party_unity_{ch}.parquet"
            if not unity_path.exists():
                print(f"  {chamber}: No unity parquet at {unity_path} — skipping")
                continue

            unity_df = pl.read_parquet(unity_path)
            print(f"  {chamber}: {unity_df.height} legislators loaded")
            chambers_found.append(chamber)
            chamber_results[chamber] = {"unity_df": unity_df}

        if not chambers_found:
            print("  No data found — exiting")
            return

        # ── Compute posteriors per chamber ──
        all_loyalty: dict[str, pl.DataFrame] = {}

        for chamber in chambers_found:
            print_header(f"BAYESIAN POSTERIORS — {chamber}")
            unity_df = chamber_results[chamber]["unity_df"]
            loyalty_df = compute_bayesian_loyalty(unity_df, chamber)

            if loyalty_df.height == 0:
                print(f"  {chamber}: No posteriors computed")
                continue

            # Summary stats
            print(f"  {chamber}: {loyalty_df.height} posterior estimates")
            print(f"    Mean shrinkage: {float(loyalty_df['shrinkage'].mean()):.4f}")
            print(f"    Mean CI width:  {float(loyalty_df['ci_width'].mean()):.4f}")

            # Who moved the most?
            most_shrunk = loyalty_df.sort(
                (pl.col("raw_loyalty") - pl.col("posterior_mean")).abs(),
                descending=True,
            ).head(5)
            print("    Most affected by shrinkage:")
            for row in most_shrunk.iter_rows(named=True):
                delta = row["posterior_mean"] - row["raw_loyalty"]
                print(
                    f"      {row['full_name']} ({row['party'][0]}): "
                    f"raw={row['raw_loyalty']:.3f} → post={row['posterior_mean']:.3f} "
                    f"(delta={delta:+.3f}, n={row['n_party_votes']})"
                )

            # Save parquet
            loyalty_df.write_parquet(ctx.data_dir / f"posterior_loyalty_{chamber.lower()}.parquet")
            ctx.export_csv(
                loyalty_df,
                f"posterior_loyalty_{chamber.lower()}.csv",
                f"Bayesian posterior party loyalty for {chamber}",
            )
            all_loyalty[chamber] = loyalty_df
            chamber_results[chamber]["loyalty_df"] = loyalty_df

        # ── Plots ──
        for chamber, loyalty_df in all_loyalty.items():
            print_header(f"PLOTS — {chamber}")
            plot_shrinkage_arrows(loyalty_df, chamber, ctx.plots_dir)
            plot_credible_intervals(loyalty_df, chamber, ctx.plots_dir)
            plot_posterior_distributions(loyalty_df, chamber, ctx.plots_dir)
            plot_raw_vs_bayesian(loyalty_df, chamber, ctx.plots_dir)

        # ── Filtering manifest ──
        print_header("FILTERING MANIFEST")
        manifest: dict = {
            "analysis": "beta_binomial",
            "session": args.session,
            "constants": {
                "MIN_PARTY_VOTES": MIN_PARTY_VOTES,
                "CI_LEVEL": CI_LEVEL,
                "FALLBACK_ALPHA": FALLBACK_ALPHA,
                "FALLBACK_BETA": FALLBACK_BETA,
            },
        }

        for chamber, loyalty_df in all_loyalty.items():
            ch = chamber.lower()
            manifest[f"{ch}_n_legislators"] = loyalty_df.height
            manifest[f"{ch}_mean_shrinkage"] = float(loyalty_df["shrinkage"].mean())
            manifest[f"{ch}_mean_ci_width"] = float(loyalty_df["ci_width"].mean())

            for party in ["Republican", "Democrat"]:
                party_sub = loyalty_df.filter(pl.col("party") == party)
                if party_sub.height > 0:
                    manifest[f"{ch}_{party.lower()}_alpha"] = float(party_sub["alpha_prior"][0])
                    manifest[f"{ch}_{party.lower()}_beta"] = float(party_sub["beta_prior"][0])
                    manifest[f"{ch}_{party.lower()}_prior_mean"] = float(party_sub["prior_mean"][0])
                    manifest[f"{ch}_{party.lower()}_prior_kappa"] = float(
                        party_sub["prior_kappa"][0]
                    )
                    manifest[f"{ch}_{party.lower()}_n"] = party_sub.height

                    # Tarone's overdispersion test per group
                    y_arr = party_sub["votes_with_party"].to_numpy()
                    n_arr = party_sub["n_party_votes"].to_numpy()
                    t_z, t_p = tarone_test(y_arr, n_arr)
                    manifest[f"{ch}_{party.lower()}_tarone_z"] = t_z
                    manifest[f"{ch}_{party.lower()}_tarone_p"] = t_p

        manifest_path = ctx.run_dir / "filtering_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  Saved: {manifest_path.name}")

        # ── HTML report ──
        print_header("HTML REPORT")
        build_beta_binomial_report(
            ctx.report,
            chamber_results=chamber_results,
            all_loyalty=all_loyalty,
            plots_dir=ctx.plots_dir,
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
