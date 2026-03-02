"""Pure data logic for TSA R enrichment (CROPS + Bai-Perron).

All functions are pure (no I/O, no subprocess calls). Signal preparation,
result parsing, elbow detection, and PELT/BP merge logic live here so they
can be tested with synthetic data.
"""

from datetime import date

import polars as pl

# ── Signal Preparation ──────────────────────────────────────────────────────


def prepare_rice_signal_csv(weekly: pl.DataFrame, party: str) -> pl.DataFrame:
    """Filter weekly Rice to one party, select mean_rice column, sorted chronologically.

    Returns a single-column DataFrame suitable for writing to CSV as R input.
    """
    filtered = weekly.filter(pl.col("party") == party).sort("week_start")
    if filtered.height == 0:
        return pl.DataFrame(schema={"mean_rice": pl.Float64})
    return filtered.select("mean_rice")


# ── CROPS Parsing ───────────────────────────────────────────────────────────


def parse_crops_result(crops_json: dict) -> pl.DataFrame | None:
    """Parse CROPS JSON output from R into a DataFrame.

    Expected JSON structure:
        {"penalties": [p1, p2, ...], "n_changepoints": [n1, n2, ...]}
    or:
        {"error": "..."}

    Returns DataFrame with columns {penalty, n_changepoints} or None on error.
    """
    if "error" in crops_json:
        return None

    penalties = crops_json.get("penalties", [])
    n_cps = crops_json.get("n_changepoints", [])

    if not penalties or not n_cps or len(penalties) != len(n_cps):
        return None

    return pl.DataFrame(
        {
            "penalty": [float(p) for p in penalties],
            "n_changepoints": [int(n) for n in n_cps],
        }
    ).sort("penalty")


def find_crops_elbow(crops_df: pl.DataFrame) -> float | None:
    """Find the CROPS elbow — penalty where marginal cost increase is largest.

    The elbow is the penalty value where adding one more changepoint (decreasing
    penalty) produces the steepest jump in cost. This is the point of diminishing
    returns.

    Returns the penalty value at the elbow, or None if insufficient data.
    """
    if crops_df is None or crops_df.height < 2:
        return None

    sorted_df = crops_df.sort("penalty")
    penalties = sorted_df["penalty"].to_list()
    n_cps = sorted_df["n_changepoints"].to_list()

    # Find the largest jump in n_changepoints between adjacent penalties
    max_jump = 0
    elbow_penalty = None

    for i in range(len(penalties) - 1):
        jump = abs(n_cps[i] - n_cps[i + 1])
        if jump > max_jump:
            max_jump = jump
            # The elbow is at the higher penalty (where adding more changepoints
            # starts to require a big penalty reduction)
            elbow_penalty = penalties[i + 1]

    return elbow_penalty


# ── Bai-Perron Parsing ──────────────────────────────────────────────────────


def parse_bai_perron_result(
    bp_json: dict,
    weekly_dates: list[date],
) -> pl.DataFrame | None:
    """Parse Bai-Perron JSON output from R into a DataFrame with dates.

    Expected JSON structure:
        {
            "breakpoints": [idx1, idx2, ...],  (R 1-based indices)
            "ci_lower": [lo1, lo2, ...],
            "ci_upper": [hi1, hi2, ...]
        }
    or:
        {"error": "..."}

    Converts R 1-based indices to 0-based, maps to dates from weekly_dates.

    Returns DataFrame with columns:
        {break_index, break_date, ci_lower_index, ci_upper_index,
         ci_lower_date, ci_upper_date}
    or None on error / no breaks.
    """
    if "error" in bp_json:
        return None

    breakpoints = bp_json.get("breakpoints", [])
    if not breakpoints:
        return None

    ci_lower = bp_json.get("ci_lower", [])
    ci_upper = bp_json.get("ci_upper", [])

    n = len(weekly_dates)
    rows = []

    for i, bp in enumerate(breakpoints):
        # Convert R 1-based to 0-based
        bp_idx = int(bp) - 1
        lo_idx = int(ci_lower[i]) - 1 if i < len(ci_lower) else bp_idx
        hi_idx = int(ci_upper[i]) - 1 if i < len(ci_upper) else bp_idx

        # Clamp to valid range
        bp_idx = max(0, min(bp_idx, n - 1))
        lo_idx = max(0, min(lo_idx, n - 1))
        hi_idx = max(0, min(hi_idx, n - 1))

        rows.append(
            {
                "break_index": bp_idx,
                "break_date": weekly_dates[bp_idx],
                "ci_lower_index": lo_idx,
                "ci_upper_index": hi_idx,
                "ci_lower_date": weekly_dates[lo_idx],
                "ci_upper_date": weekly_dates[hi_idx],
            }
        )

    return pl.DataFrame(rows)


# ── Merge Logic ─────────────────────────────────────────────────────────────


def merge_bai_perron_with_pelt(
    pelt_dates: list[str],
    bp_df: pl.DataFrame,
    max_days_apart: int = 14,
) -> pl.DataFrame:
    """Cross-reference PELT break dates with Bai-Perron breaks + CIs.

    Returns a DataFrame with columns:
        {pelt_date, bp_confirmed, bp_date, ci_lower_date, ci_upper_date, ci_window_days}

    A PELT break is "confirmed" if a BP break falls within max_days_apart.
    """
    rows = []

    for pelt_str in pelt_dates:
        try:
            pelt_d = date.fromisoformat(pelt_str[:10])
        except ValueError, TypeError:
            continue

        best_match = None
        best_dist = max_days_apart + 1

        for bp_row in bp_df.iter_rows(named=True):
            bp_d = bp_row["break_date"]
            if isinstance(bp_d, str):
                bp_d = date.fromisoformat(bp_d[:10])

            dist = abs((pelt_d - bp_d).days)
            if dist <= max_days_apart and dist < best_dist:
                best_dist = dist
                best_match = bp_row

        if best_match is not None:
            ci_lo = best_match["ci_lower_date"]
            ci_hi = best_match["ci_upper_date"]
            if isinstance(ci_lo, str):
                ci_lo = date.fromisoformat(ci_lo[:10])
            if isinstance(ci_hi, str):
                ci_hi = date.fromisoformat(ci_hi[:10])
            ci_window = (ci_hi - ci_lo).days

            rows.append(
                {
                    "pelt_date": pelt_str[:10],
                    "bp_confirmed": True,
                    "bp_date": str(best_match["break_date"])[:10],
                    "ci_lower_date": str(best_match["ci_lower_date"])[:10],
                    "ci_upper_date": str(best_match["ci_upper_date"])[:10],
                    "ci_window_days": ci_window,
                }
            )
        else:
            rows.append(
                {
                    "pelt_date": pelt_str[:10],
                    "bp_confirmed": False,
                    "bp_date": "",
                    "ci_lower_date": "",
                    "ci_upper_date": "",
                    "ci_window_days": 0,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "pelt_date": pl.Utf8,
                "bp_confirmed": pl.Boolean,
                "bp_date": pl.Utf8,
                "ci_lower_date": pl.Utf8,
                "ci_upper_date": pl.Utf8,
                "ci_window_days": pl.Int64,
            }
        )

    return pl.DataFrame(rows)
