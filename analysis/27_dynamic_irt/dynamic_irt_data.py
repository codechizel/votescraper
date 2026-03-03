"""Pure data logic for dynamic ideal point estimation.

Session enumeration, global legislator roster construction, long-format vote
stacking across bienniums, and bridge coverage analysis.  No I/O, no plotting
— all functions take DataFrames/dicts in and return DataFrames/dicts out.
"""

from pathlib import Path

import numpy as np
import polars as pl

try:
    from analysis.phase_utils import normalize_name
except ModuleNotFoundError:
    from phase_utils import normalize_name  # type: ignore[no-redef]

# ── Constants ────────────────────────────────────────────────────────────────

BIENNIUM_SESSIONS: list[str] = [
    "2011-12",
    "2013-14",
    "2015-16",
    "2017-18",
    "2019-20",
    "2021-22",
    "2023-24",
    "2025-26",
]
"""Ordered list of all scraped biennium session strings (84th–91st)."""

BIENNIUM_LABELS: list[str] = [
    "84th",
    "85th",
    "86th",
    "87th",
    "88th",
    "89th",
    "90th",
    "91st",
]
"""Human-readable labels for each biennium (same order as BIENNIUM_SESSIONS)."""

SESSION_TO_LABEL: dict[str, str] = dict(zip(BIENNIUM_SESSIONS, BIENNIUM_LABELS))
"""Map from session string to label, e.g. '2025-26' -> '91st'."""

LABEL_TO_SESSION: dict[str, str] = dict(zip(BIENNIUM_LABELS, BIENNIUM_SESSIONS))
"""Map from label to session string, e.g. '91st' -> '2025-26'."""

MIN_BRIDGE_OVERLAP: int = 5
"""Minimum shared legislators between adjacent bienniums for a valid bridge."""

# ── Global Legislator Roster ────────────────────────────────────────────────


def build_global_roster(
    all_legislators: dict[int, pl.DataFrame],
    chamber: str,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Build a unified legislator roster across all bienniums for one chamber.

    When OCD IDs are available, groups by ``ocd_id`` first (correctly
    separating same-name legislators like two Mike Thompsons).  Falls back
    to ``name_norm`` grouping for legislators without OCD IDs or older CSVs.

    Args:
        all_legislators: Mapping from time_idx (0-based biennium index) to
            legislator DataFrame.  Each must have ``full_name``,
            ``legislator_slug``, ``party``, ``chamber`` columns.
            Optional: ``ocd_id``.
        chamber: ``"House"`` or ``"Senate"``.

    Returns:
        (roster_df, name_to_global_idx) where:
          - roster_df has columns: global_idx, name_norm, full_name,
            parties (comma-joined), first_period, last_period, n_periods,
            slugs (comma-joined per period)
          - name_to_global_idx maps name_norm → int
    """
    # Collect (name_norm, full_name, party, time_idx, slug, ocd_id) tuples
    records: list[dict] = []
    for t, leg_df in sorted(all_legislators.items()):
        # Filter to the requested chamber
        if "chamber" in leg_df.columns:
            ch_df = leg_df.filter(pl.col("chamber") == chamber)
        else:
            ch_df = leg_df

        has_ocd = "ocd_id" in ch_df.columns

        for row in ch_df.iter_rows(named=True):
            slug_col = "legislator_slug" if "legislator_slug" in ch_df.columns else "slug"
            name = row.get("full_name", "")
            ocd_id = row.get("ocd_id", "") if has_ocd else ""
            records.append(
                {
                    "name_norm": normalize_name(name),
                    "full_name": name,
                    "party": row.get("party", ""),
                    "time_idx": t,
                    "slug": row.get(slug_col, ""),
                    "ocd_id": ocd_id or "",
                }
            )

    if not records:
        empty = pl.DataFrame(
            schema={
                "global_idx": pl.Int64,
                "name_norm": pl.Utf8,
                "full_name": pl.Utf8,
                "parties": pl.Utf8,
                "first_period": pl.Int64,
                "last_period": pl.Int64,
                "n_periods": pl.Int64,
            }
        )
        return empty, {}

    raw = pl.DataFrame(records)

    # Determine grouping key: use ocd_id when available, fall back to name_norm.
    # Records with non-empty ocd_id group by ocd_id; records without group by name_norm.
    ocd_records = raw.filter(pl.col("ocd_id") != "")
    no_ocd_records = raw.filter(pl.col("ocd_id") == "")

    roster_rows: list[dict] = []
    name_to_global: dict[str, int] = {}
    global_idx = 0

    def _process_group(group: pl.DataFrame, identity_key: str) -> None:
        nonlocal global_idx
        nn = group["name_norm"].to_list()[-1]  # most recent name_norm
        full_names = group["full_name"].to_list()
        canonical_name = full_names[-1]

        parties = group["party"].unique().sort().to_list()
        party_str = ", ".join(p for p in parties if p)
        if not party_str:
            party_str = "Independent"

        periods = sorted(group["time_idx"].unique().to_list())
        first_period = periods[0]
        last_period = periods[-1]
        n_periods = len(periods)

        slug_strs = []
        for t in periods:
            t_slugs = group.filter(pl.col("time_idx") == t)["slug"].to_list()
            slug_strs.extend(t_slugs)

        name_to_global[nn] = global_idx
        roster_rows.append(
            {
                "global_idx": global_idx,
                "name_norm": nn,
                "full_name": canonical_name,
                "parties": party_str,
                "first_period": first_period,
                "last_period": last_period,
                "n_periods": n_periods,
                "periods_served": periods,
            }
        )
        global_idx += 1

    # Group OCD records by ocd_id (handles same-name legislators correctly)
    if ocd_records.height > 0:
        for _ocd_id, group in ocd_records.group_by("ocd_id"):
            _process_group(group, "ocd_id")

    # Group remaining records by name_norm (fallback for old CSVs)
    if no_ocd_records.height > 0:
        for name_norm, group in no_ocd_records.group_by("name_norm"):
            nn = name_norm[0] if isinstance(name_norm, tuple) else name_norm
            # Skip if this name_norm was already added via OCD ID
            if nn in name_to_global:
                continue
            _process_group(group, "name_norm")

    roster = pl.DataFrame(roster_rows)
    return roster, name_to_global


def _get_party_for_legislator(
    name_norm: str,
    all_legislators: dict[int, pl.DataFrame],
    chamber: str,
) -> str:
    """Get the most recent party for a legislator across bienniums."""
    for t in sorted(all_legislators.keys(), reverse=True):
        leg_df = all_legislators[t]
        if "chamber" in leg_df.columns:
            ch_df = leg_df.filter(pl.col("chamber") == chamber)
        else:
            ch_df = leg_df

        for row in ch_df.iter_rows(named=True):
            nn = normalize_name(row.get("full_name", ""))
            if nn == name_norm:
                return row.get("party", "")
    return ""


# ── Long-Format Vote Stacking ──────────────────────────────────────────────


def stack_bienniums(
    chamber: str,
    all_irt_data: dict[int, dict],
    name_to_global: dict[str, int],
    all_legislators: dict[int, pl.DataFrame],
) -> dict:
    """Stack IRT data across bienniums into a single long-format dataset.

    Args:
        chamber: ``"House"`` or ``"Senate"``.
        all_irt_data: Mapping from time_idx to per-biennium IRT data dicts
            (from ``prepare_irt_data()``).
        name_to_global: Mapping from name_norm to global legislator index.
        all_legislators: Mapping from time_idx to legislator DataFrames.

    Returns:
        Dict with keys:
          - leg_global_idx: np.ndarray of global legislator indices
          - bill_idx: np.ndarray of globally unique bill indices
          - time_idx: np.ndarray of biennium time indices (per observation)
          - y: np.ndarray of binary votes
          - bill_to_time: np.ndarray mapping each bill to its biennium
          - n_legislators: total unique legislators
          - n_bills: total bills across all bienniums
          - n_obs: total observations
          - n_time: number of bienniums
          - leg_names: list of name_norms in global_idx order
          - bill_ids: list of globally unique bill IDs
          - leg_periods: list of lists — which periods each legislator served
          - party_idx: np.ndarray of party indices per global legislator (most recent)
          - party_names: list of party names
    """
    all_leg_global = []
    all_bill_idx = []
    all_time = []
    all_y = []

    bill_offset = 0
    all_bill_ids: list[str] = []
    bill_to_time: list[int] = []

    # Build reverse mapping: global_idx -> name_norm
    idx_to_name = {v: k for k, v in name_to_global.items()}
    n_global = len(name_to_global)

    # Track which periods each legislator served
    leg_periods: list[list[int]] = [[] for _ in range(n_global)]

    for t in sorted(all_irt_data.keys()):
        data_t = all_irt_data[t]
        leg_slugs_t = data_t["leg_slugs"]
        vote_ids_t = data_t["vote_ids"]
        n_bills_t = data_t["n_votes"]

        # Get legislator DataFrame for this biennium
        leg_df_t = all_legislators[t]
        if "chamber" in leg_df_t.columns:
            leg_df_t = leg_df_t.filter(pl.col("chamber") == chamber)

        # Build slug -> name_norm mapping for this biennium
        slug_col = "legislator_slug" if "legislator_slug" in leg_df_t.columns else "slug"
        slug_to_name: dict[str, str] = {}
        for row in leg_df_t.iter_rows(named=True):
            slug = row.get(slug_col, "")
            nn = normalize_name(row.get("full_name", ""))
            slug_to_name[slug] = nn

        # Map local leg_idx -> global leg_idx
        local_to_global: dict[int, int] = {}
        for local_idx, slug in enumerate(leg_slugs_t):
            nn = slug_to_name.get(slug)
            if nn is not None and nn in name_to_global:
                local_to_global[local_idx] = name_to_global[nn]
                # Track periods served
                gidx = name_to_global[nn]
                if t not in leg_periods[gidx]:
                    leg_periods[gidx].append(t)

        # Remap observations
        leg_idx_local = data_t["leg_idx"]
        vote_idx_local = data_t["vote_idx"]
        y_local = data_t["y"]

        for i in range(len(y_local)):
            local_leg = int(leg_idx_local[i])
            if local_leg in local_to_global:
                all_leg_global.append(local_to_global[local_leg])
                all_bill_idx.append(int(vote_idx_local[i]) + bill_offset)
                all_time.append(t)
                all_y.append(int(y_local[i]))

        # Global bill IDs (prefixed with biennium label for uniqueness)
        label = BIENNIUM_LABELS[t] if t < len(BIENNIUM_LABELS) else f"t{t}"
        for vid in vote_ids_t:
            all_bill_ids.append(f"{label}:{vid}")
            bill_to_time.append(t)

        bill_offset += n_bills_t

    # Build party index (most recent party assignment)
    parties_seen: set[str] = set()
    leg_party: list[str] = []
    for gidx in range(n_global):
        nn = idx_to_name[gidx]
        party = _get_party_for_legislator(nn, all_legislators, chamber)
        if not party:
            party = "Independent"
        leg_party.append(party)
        parties_seen.add(party)

    party_names = sorted(parties_seen)
    party_map = {p: i for i, p in enumerate(party_names)}
    party_idx = np.array([party_map[p] for p in leg_party], dtype=np.int64)

    # Sort periods for each legislator
    for periods in leg_periods:
        periods.sort()

    # Build ordered name list
    leg_names = [idx_to_name[i] for i in range(n_global)]

    n_time = max(all_time) + 1 if all_time else 0

    return {
        "leg_global_idx": np.array(all_leg_global, dtype=np.int64),
        "bill_idx": np.array(all_bill_idx, dtype=np.int64),
        "time_idx": np.array(all_time, dtype=np.int64),
        "y": np.array(all_y, dtype=np.int64),
        "bill_to_time": np.array(bill_to_time, dtype=np.int64),
        "n_legislators": n_global,
        "n_bills": len(all_bill_ids),
        "n_obs": len(all_y),
        "n_time": n_time,
        "leg_names": leg_names,
        "bill_ids": all_bill_ids,
        "leg_periods": leg_periods,
        "party_idx": party_idx,
        "party_names": party_names,
    }


# ── Bridge Coverage Analysis ───────────────────────────────────────────────


def compute_bridge_coverage(
    leg_periods: list[list[int]],
    n_time: int,
    labels: list[str] | None = None,
) -> pl.DataFrame:
    """Compute pairwise bridge coverage between all biennium pairs.

    For each pair (t_a, t_b), counts how many legislators served in both.

    Returns:
        DataFrame with columns: period_a, period_b, label_a, label_b,
        shared_count, total_a, total_b, overlap_pct
    """
    if labels is None:
        labels = BIENNIUM_LABELS[:n_time]

    # Count legislators per period and shared across pairs
    period_counts: dict[int, int] = {}
    for t in range(n_time):
        period_counts[t] = sum(1 for periods in leg_periods if t in periods)

    rows: list[dict] = []
    for t_a in range(n_time):
        for t_b in range(t_a + 1, n_time):
            shared = sum(1 for periods in leg_periods if t_a in periods and t_b in periods)
            total_a = period_counts[t_a]
            total_b = period_counts[t_b]
            denom = min(total_a, total_b) if min(total_a, total_b) > 0 else 1
            rows.append(
                {
                    "period_a": t_a,
                    "period_b": t_b,
                    "label_a": labels[t_a] if t_a < len(labels) else f"t{t_a}",
                    "label_b": labels[t_b] if t_b < len(labels) else f"t{t_b}",
                    "shared_count": shared,
                    "total_a": total_a,
                    "total_b": total_b,
                    "overlap_pct": 100.0 * shared / denom,
                }
            )

    return pl.DataFrame(rows)


def compute_adjacent_bridges(
    leg_periods: list[list[int]],
    n_time: int,
    labels: list[str] | None = None,
) -> pl.DataFrame:
    """Compute bridge coverage for adjacent biennium pairs only.

    This is the critical chain: the Markov assumption requires sufficient
    overlap between consecutive periods.

    Returns:
        DataFrame with columns: pair, shared_count, total_a, total_b,
        overlap_pct, sufficient (bool)
    """
    if labels is None:
        labels = BIENNIUM_LABELS[:n_time]

    rows: list[dict] = []
    for t in range(n_time - 1):
        t_a, t_b = t, t + 1
        shared = sum(1 for periods in leg_periods if t_a in periods and t_b in periods)
        total_a = sum(1 for periods in leg_periods if t_a in periods)
        total_b = sum(1 for periods in leg_periods if t_b in periods)
        denom = min(total_a, total_b) if min(total_a, total_b) > 0 else 1
        overlap = 100.0 * shared / denom

        label_a = labels[t_a] if t_a < len(labels) else f"t{t_a}"
        label_b = labels[t_b] if t_b < len(labels) else f"t{t_b}"

        rows.append(
            {
                "pair": f"{label_a}→{label_b}",
                "shared_count": shared,
                "total_a": total_a,
                "total_b": total_b,
                "overlap_pct": overlap,
                "sufficient": shared >= MIN_BRIDGE_OVERLAP,
            }
        )

    return pl.DataFrame(rows)


# ── emIRT Interface ─────────────────────────────────────────────────────────


def prepare_emirt_csv(
    data: dict,
    output_path: Path,
) -> None:
    """Write vote data in the format expected by the emIRT R script.

    Columns: legislator_id, bill_id, vote, time_period
    """
    df = pl.DataFrame(
        {
            "legislator_id": data["leg_global_idx"],
            "bill_id": data["bill_idx"],
            "vote": data["y"],
            "time_period": data["time_idx"],
        }
    )
    df.write_csv(output_path)


def load_emirt_results(result_path: Path) -> pl.DataFrame | None:
    """Load emIRT point estimates from CSV output.

    Returns None if the file doesn't exist.
    """
    if not result_path.exists():
        return None
    return pl.read_csv(result_path)
