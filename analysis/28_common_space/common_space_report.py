"""Common Space Ideal Points — HTML report builder.

Stub for Task 1. Full implementation in Task 4.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def build_common_space_report(
    report: object,
    *,
    all_results: dict,
    bridge_matrix: pl.DataFrame,
    sessions: list[str],
    reference: str,
    plots_dir: Path,
) -> None:
    """Add sections to the report builder.

    Stub — will be implemented with all 9 sections in Task 4.
    """
    # Task 4 will implement:
    # 1. Key Findings
    # 2. Bridge Coverage heatmap
    # 3. Linking Coefficients scatter with bootstrap CIs
    # 4. Polarization Trajectory (Plotly)
    # 5. Party Separation (Cohen's d per biennium)
    # 6. Top Movers
    # 7. Career Trajectories (Plotly)
    # 8. External Validation
    # 9. Quality Gates
    pass
