"""Common Space Ideal Points — HTML report builder."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl

matplotlib.use("Agg")

try:
    from analysis.phase_utils import save_fig
    from analysis.report import (
        FigureSection,
        InteractiveSection,
        InteractiveTableSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        make_gt,
        make_interactive_table,
    )
    from analysis.tuning import PARTY_COLORS
except ModuleNotFoundError:
    from phase_utils import save_fig  # type: ignore[no-redef]
    from report import (  # type: ignore[no-redef]
        FigureSection,
        InteractiveSection,
        InteractiveTableSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        make_gt,
        make_interactive_table,
    )
    from tuning import PARTY_COLORS  # type: ignore[no-redef]

from analysis.common_space_data import MIN_SERVED_FOR_TRAJECTORY

# ---------------------------------------------------------------------------
# Helper: short session label (e.g., "79th" from "79th_2001-2002")
# ---------------------------------------------------------------------------


def _short(session: str) -> str:
    return session.split("_")[0]


def _year(session: str) -> str:
    parts = session.split("_")
    return parts[1] if len(parts) > 1 else session


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_common_space_report(
    report: ReportBuilder,
    *,
    all_results: dict,
    bridge_matrix: pl.DataFrame,
    sessions: list[str],
    reference: str,
    plots_dir: Path,
) -> None:
    """Add all report sections."""

    # ---- 1. Key Findings ----
    findings: list[str] = []
    for chamber, r in all_results.items():
        n_pass = sum(1 for g in r["gates"] if g.passed)
        n_total = len(r["gates"])
        findings.append(
            f"{chamber}: {len(r['sessions'])} bienniums aligned, "
            f"{n_pass}/{n_total} quality gates passed"
        )
        traj = r["trajectory"]
        if traj.height >= 2:
            first = traj.row(0, named=True)
            last = traj.row(-1, named=True)
            gap_change = last["party_gap"] - first["party_gap"]
            direction = "widened" if gap_change > 0 else "narrowed"
            findings.append(
                f"{chamber} party gap {direction} by {abs(gap_change):.2f} "
                f"({_short(first['session'])} → {_short(last['session'])})"
            )
    findings.append(f"Reference scale: {reference}")
    report.add(KeyFindingsSection(findings=findings))

    # ---- 2. Bridge Coverage Heatmap ----
    _add_bridge_heatmap(report, bridge_matrix, sessions, plots_dir)

    # ---- Per-chamber sections ----
    for chamber, r in all_results.items():
        # ---- 3. Ideal Points Table (searchable/sortable) ----
        _add_ideal_points_table(report, r, chamber)

        # ---- 4. Linking Coefficients ----
        _add_linking_coefficients(report, r, chamber, plots_dir)

        # ---- 5. Polarization Trajectory ----
        _add_polarization_trajectory(report, r, chamber)

        # ---- 6. Party Separation ----
        _add_party_separation(report, r, chamber, plots_dir)

        # ---- 7. Top Movers ----
        _add_top_movers(report, r, chamber)

        # ---- 8. Career Trajectories ----
        _add_career_trajectories(report, r, chamber)

        # ---- 9. Quality Gates ----
        _add_quality_gates(report, r, chamber)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _add_ideal_points_table(
    report: ReportBuilder,
    r: dict,
    chamber: str,
) -> None:
    """Searchable/sortable table of all legislators on the common scale."""
    transformed = r["transformed"]
    if transformed.height == 0:
        return

    display_cols = [
        "full_name",
        "party",
        "session",
        "xi_common",
        "xi_common_sd",
        "xi_common_lo",
        "xi_common_hi",
    ]
    available = [c for c in display_cols if c in transformed.columns]
    df = transformed.select(available).sort("xi_common", descending=True)

    # Round numeric columns
    for col in ["xi_common", "xi_common_sd", "xi_common_lo", "xi_common_hi"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).round(3))

    html = make_interactive_table(
        df,
        title=(
            f"{chamber} — Common Space Ideal Points "
            f"({df.height} legislator-sessions, positive = conservative)"
        ),
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "session": "Session",
            "xi_common": "Score",
            "xi_common_sd": "Std Dev",
            "xi_common_lo": "95% CI Low",
            "xi_common_hi": "95% CI High",
        },
        number_formats={
            "xi_common": ".3f",
            "xi_common_sd": ".3f",
            "xi_common_lo": ".3f",
            "xi_common_hi": ".3f",
        },
        caption=(
            "Combined uncertainty from IRT posterior + alignment bootstrap. "
            "Search by name or session. Sort by any column."
        ),
    )
    report.add(
        InteractiveTableSection(
            id=f"ideal_points_{chamber.lower()}",
            title=f"{chamber} — Common Space Ideal Points",
            html=html,
        )
    )


def _add_bridge_heatmap(
    report: ReportBuilder,
    bridge_matrix: pl.DataFrame,
    sessions: list[str],
    plots_dir: Path,
) -> None:
    labels = [_short(s) for s in sessions]
    n = len(sessions)
    mat = np.zeros((n, n))

    for row in bridge_matrix.iter_rows(named=True):
        try:
            i = sessions.index(row["session_a"])
            j = sessions.index(row["session_b"])
            mat[i, j] = row["n_bridges"]
            mat[j, i] = row["n_bridges"]
        except ValueError:
            continue

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Bridge Legislators Between Bienniums")
    fig.colorbar(im, ax=ax, label="Shared legislators")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if mat[i, j] > 0:
                ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=6)

    path = plots_dir / "bridge_heatmap.png"
    save_fig(fig, path)
    report.add(
        FigureSection.from_file(
            id="bridge_heatmap",
            title="Bridge Coverage Between Bienniums",
            path=path,
            alt_text="Heatmap showing number of shared legislators between each pair of bienniums",
        )
    )


def _add_linking_coefficients(
    report: ReportBuilder,
    r: dict,
    chamber: str,
    plots_dir: Path,
) -> None:
    coef_df = r["coef_df"]
    if coef_df.height == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sessions = coef_df["session"].to_list()
    labels = [_short(s) for s in sessions]
    A_vals = coef_df["A"].to_numpy()
    B_vals = coef_df["B"].to_numpy()
    A_lo = coef_df["A_lo"].to_numpy()
    A_hi = coef_df["A_hi"].to_numpy()
    B_lo = coef_df["B_lo"].to_numpy()
    B_hi = coef_df["B_hi"].to_numpy()

    x = range(len(sessions))

    ax1.errorbar(x, A_vals, yerr=[A_vals - A_lo, A_hi - A_vals], fmt="o", capsize=3, color="#333")
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Scale (A)")
    ax1.set_title(f"{chamber} — Scale Coefficients")

    ax2.errorbar(x, B_vals, yerr=[B_vals - B_lo, B_hi - B_vals], fmt="o", capsize=3, color="#333")
    ax2.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Shift (B)")
    ax2.set_title(f"{chamber} — Shift Coefficients")

    fig.tight_layout()
    path = plots_dir / f"linking_coefficients_{chamber.lower()}.png"
    save_fig(fig, path)
    report.add(
        FigureSection.from_file(
            id=f"linking_coefs_{chamber.lower()}",
            title=f"{chamber} — Linking Coefficients with 95% Bootstrap CIs",
            path=path,
            alt_text=(
                f"Scale and shift coefficients for {chamber} alignment with confidence intervals"
            ),
        )
    )


def _add_polarization_trajectory(
    report: ReportBuilder,
    r: dict,
    chamber: str,
) -> None:
    traj = r["trajectory"]
    if traj.height == 0:
        return

    sessions = traj["session"].to_list()
    labels = [_year(s) for s in sessions]
    r_means = traj["r_mean"].to_list()
    d_means = traj["d_mean"].to_list()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=r_means,
            mode="lines+markers",
            name="Republican Mean",
            line={"color": PARTY_COLORS.get("Republican", "#E81B23")},
            marker={"size": 8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=d_means,
            mode="lines+markers",
            name="Democrat Mean",
            line={"color": PARTY_COLORS.get("Democrat", "#0015BC")},
            marker={"size": 8},
        )
    )
    fig.update_layout(
        title=f"{chamber} — Party Mean Ideology on Common Scale",
        xaxis_title="Biennium",
        yaxis_title="Common Space Ideal Point",
        template="plotly_white",
        height=450,
    )

    report.add(
        InteractiveSection(
            id=f"polarization_{chamber.lower()}",
            title=f"{chamber} — Polarization Trajectory",
            html=fig.to_html(include_plotlyjs="cdn", full_html=False),
            aria_label=f"Interactive line chart showing {chamber} party mean ideology over time",
        )
    )


def _add_party_separation(
    report: ReportBuilder,
    r: dict,
    chamber: str,
    plots_dir: Path,
) -> None:
    gates = r["gates"]
    if not gates:
        return

    sessions = [g.session for g in gates]
    labels = [_short(s) for s in sessions]
    d_vals = [g.party_d for g in gates]
    colors = ["#2ecc71" if g.passed else "#e74c3c" for g in gates]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(sessions)), d_vals, color=colors)
    ax.axhline(1.5, color="gray", linestyle="--", alpha=0.5, label="Quality threshold (d=1.5)")
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cohen's d (party separation)")
    ax.set_title(f"{chamber} — Party Separation per Biennium")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = plots_dir / f"party_separation_{chamber.lower()}.png"
    save_fig(fig, path)
    report.add(
        FigureSection.from_file(
            id=f"party_sep_{chamber.lower()}",
            title=f"{chamber} — Party Separation (Cohen's d) per Biennium",
            path=path,
            alt_text=(
                f"Bar chart showing party separation on aligned scale for each {chamber} biennium"
            ),
        )
    )


def _add_top_movers(
    report: ReportBuilder,
    r: dict,
    chamber: str,
) -> None:
    transformed = r["transformed"]
    if transformed.height == 0:
        return

    # Find legislators who served in multiple sessions and moved the most
    multi = (
        transformed.group_by("name_norm")
        .agg(
            [
                pl.col("xi_common").min().alias("xi_min"),
                pl.col("xi_common").max().alias("xi_max"),
                pl.col("session").n_unique().alias("n_sessions"),
                pl.col("full_name").first(),
                pl.col("party").first(),
            ]
        )
        .filter(pl.col("n_sessions") >= 2)
        .with_columns((pl.col("xi_max") - pl.col("xi_min")).alias("range"))
        .sort("range", descending=True)
        .head(20)
    )

    if multi.height == 0:
        return

    display = multi.select(
        pl.col("full_name"),
        pl.col("party"),
        pl.col("n_sessions"),
        pl.col("xi_min").round(3),
        pl.col("xi_max").round(3),
        pl.col("range").round(3),
    )

    gt = make_gt(display, title=f"{chamber} — Legislators with Largest Ideological Range")
    report.add(
        TableSection(
            id=f"top_movers_{chamber.lower()}",
            title=f"{chamber} — Top Movers (Largest Ideological Range Across Bienniums)",
            html=gt,
        )
    )


def _add_career_trajectories(
    report: ReportBuilder,
    r: dict,
    chamber: str,
) -> None:
    transformed = r["transformed"]
    if transformed.height == 0:
        return

    # Long-serving legislators
    service_counts = (
        transformed.group_by("name_norm")
        .agg(
            [
                pl.col("session").n_unique().alias("n_sessions"),
                pl.col("full_name").first(),
                pl.col("party").first(),
            ]
        )
        .filter(pl.col("n_sessions") >= MIN_SERVED_FOR_TRAJECTORY)
        .sort("n_sessions", descending=True)
        .head(30)
    )

    if service_counts.height == 0:
        return

    fig = go.Figure()
    target_names = set(service_counts["name_norm"].to_list())

    for name_norm in sorted(target_names):
        leg_data = transformed.filter(pl.col("name_norm") == name_norm).sort("session")
        if leg_data.height == 0:
            continue

        row0 = leg_data.row(0, named=True)
        party = row0["party"]
        color = PARTY_COLORS.get(party, "#999999")
        name_display = row0["full_name"]

        fig.add_trace(
            go.Scatter(
                x=[_year(s) for s in leg_data["session"].to_list()],
                y=leg_data["xi_common"].to_list(),
                mode="lines+markers",
                name=f"{name_display} ({party[0]})",
                line={"color": color, "width": 1.5},
                marker={"size": 5},
                opacity=0.7,
            )
        )

    fig.update_layout(
        title=f"{chamber} — Career Trajectories ({MIN_SERVED_FOR_TRAJECTORY}+ Bienniums)",
        xaxis_title="Biennium",
        yaxis_title="Common Space Ideal Point",
        template="plotly_white",
        height=600,
        showlegend=True,
        legend={"font": {"size": 8}},
    )

    report.add(
        InteractiveSection(
            id=f"career_traj_{chamber.lower()}",
            title=f"{chamber} — Career Trajectories (Long-Serving Legislators)",
            html=fig.to_html(include_plotlyjs="cdn", full_html=False),
            aria_label=(
                f"Interactive line chart showing ideological trajectories "
                f"of long-serving {chamber} members"
            ),
        )
    )


def _add_quality_gates(
    report: ReportBuilder,
    r: dict,
    chamber: str,
) -> None:
    gates = r["gates"]
    if not gates:
        return

    rows = []
    for g in gates:
        rows.append(
            {
                "Session": _short(g.session),
                "Party Separation (d)": round(g.party_d, 2),
                "Sign (R > D)": "Yes" if g.sign_ok else "No",
                "Status": "PASS" if g.passed else "FAIL",
            }
        )

    df = pl.DataFrame(rows)
    gt = make_gt(df, title=f"{chamber} — Quality Gate Results")
    report.add(
        TableSection(
            id=f"quality_gates_{chamber.lower()}",
            title=f"{chamber} — Quality Gate Results",
            html=gt,
        )
    )
