"""Session-level dashboard index generator.

Scans a pipeline run directory for phase reports and generates an index.html
with sidebar navigation + iframe embedding. Each phase's existing HTML report
is loaded in the main content area when clicked.

Usage:
    from analysis.dashboard import generate_dashboard
    generate_dashboard(session_dir, run_id)

Or via just:
    just dashboard 2025-26
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from jinja2 import Environment

_CT = ZoneInfo("America/Chicago")

# Phase display order and labels
PHASE_ORDER = [
    ("01_eda", "Exploratory Data Analysis"),
    ("02_pca", "Principal Component Analysis"),
    ("02c_mca", "Multiple Correspondence Analysis"),
    ("03_umap", "UMAP Visualization"),
    ("04_irt", "IRT Ideal Points"),
    ("04c_ppc", "Posterior Predictive Checks"),
    ("05_clustering", "Clustering"),
    ("05b_lca", "Latent Class Analysis"),
    ("06_network", "Network Analysis"),
    ("06b_network_bipartite", "Bipartite Network"),
    ("07_indices", "Legislative Indices"),
    ("08_prediction", "Vote Prediction"),
    ("09_beta_binomial", "Beta-Binomial"),
    ("10_hierarchical", "Hierarchical IRT"),
    ("11_synthesis", "Synthesis"),
    ("12_profiles", "Legislator Profiles"),
    ("15_tsa", "Time Series Analysis"),
]


def generate_dashboard(
    session_dir: Path,
    run_id: str,
    git_hash: str = "unknown",
) -> Path:
    """Generate a dashboard index.html for a pipeline run.

    Args:
        session_dir: Path to session results (e.g. results/kansas/91st_2025-2026/).
        run_id: The run ID directory name (e.g. "91-260301.1").
        git_hash: Git commit hash for display.

    Returns:
        Path to the generated index.html.
    """
    run_dir = session_dir / run_id
    if not run_dir.exists():
        msg = f"Run directory not found: {run_dir}"
        raise FileNotFoundError(msg)

    # Discover phases with reports
    phases = []
    total_elapsed = 0.0

    for phase_key, phase_label in PHASE_ORDER:
        phase_dir = run_dir / phase_key
        if not phase_dir.exists():
            continue

        report_file = phase_dir / f"{phase_key}_report.html"
        if not report_file.exists():
            continue

        # Read run_info.json if available
        info_path = phase_dir / "run_info.json"
        elapsed = ""
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            elapsed = info.get("elapsed_display", "")
            total_elapsed += info.get("elapsed_seconds", 0.0)

        phases.append(
            {
                "key": phase_key,
                "label": phase_label,
                "report_path": f"{phase_key}/{phase_key}_report.html",
                "elapsed": elapsed,
            }
        )

    if not phases:
        msg = f"No phase reports found in {run_dir}"
        raise FileNotFoundError(msg)

    # Format total elapsed
    if total_elapsed < 60:
        total_display = f"{total_elapsed:.1f}s"
    elif total_elapsed < 3600:
        m, s = divmod(int(total_elapsed), 60)
        total_display = f"{m}m {s}s"
    else:
        h, rem = divmod(int(total_elapsed), 3600)
        m, s = divmod(rem, 60)
        total_display = f"{h}h {m}m {s}s"

    # Session label from directory name
    session_label = session_dir.name.replace("_", " ")
    now = datetime.now(_CT).strftime("%Y-%m-%d %H:%M %Z")

    template = _get_dashboard_template()
    html = template.render(
        session=session_label,
        run_id=run_id,
        git_hash=git_hash,
        total_elapsed=total_display,
        generated_at=now,
        phases=phases,
        n_phases=len(phases),
    )

    index_path = run_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    print(f"  Dashboard: {index_path}")
    return index_path


DASHBOARD_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ session }} — Pipeline Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      height: 100vh; display: flex; flex-direction: column; }
    .header {
      background: #1a1a2e; color: #e0e0e0; padding: 12px 20px; display: flex;
      align-items: center; justify-content: space-between; flex-shrink: 0;
    }
    .header h1 { font-size: 18px; font-weight: 600; color: #fff; }
    .header .meta { font-size: 12px; color: #aaa; }
    .header .meta span { margin-left: 16px; }
    .main { display: flex; flex: 1; overflow: hidden; }
    .sidebar {
      width: 280px; background: #f5f5f5; border-right: 1px solid #ddd;
      overflow-y: auto; flex-shrink: 0; padding: 8px 0;
    }
    .sidebar a {
      display: flex; align-items: center; padding: 10px 16px;
      text-decoration: none; color: #333; font-size: 13px;
      border-left: 3px solid transparent; transition: all 0.15s;
    }
    .sidebar a:hover { background: #e8e8e8; }
    .sidebar a.active { background: #dde8f7; border-left-color: #2563eb;
      color: #1a3a5c; font-weight: 600; }
    .sidebar .phase-num { color: #888; font-size: 11px; width: 28px; flex-shrink: 0; }
    .sidebar .phase-label { flex: 1; }
    .sidebar .phase-time { color: #999; font-size: 11px; margin-left: 8px; }
    .content { flex: 1; background: #fff; }
    .content iframe { width: 100%; height: 100%; border: none; }
    .content .placeholder {
      display: flex; align-items: center; justify-content: center;
      height: 100%; color: #999; font-size: 16px;
    }
    @media (max-width: 768px) {
      .sidebar { width: 200px; }
      .sidebar a { padding: 8px 12px; font-size: 12px; }
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>{{ session }} — Pipeline Dashboard</h1>
    <div class="meta">
      <span>Run: <strong>{{ run_id }}</strong></span>
      <span>Phases: {{ n_phases }}</span>
      <span>Total: {{ total_elapsed }}</span>
      {% if git_hash and git_hash != "unknown" %}<span>Git:
        <code>{{ git_hash[:8] }}</code></span>{% endif %}
      <span>{{ generated_at }}</span>
    </div>
  </div>
  <div class="main">
    <nav class="sidebar">
      {% for phase in phases %}
      <a href="#" data-report="{{ phase.report_path }}"
         {% if loop.first %}class="active"{% endif %}
         onclick="loadReport(this, '{{ phase.report_path }}'); return false;">
        <span class="phase-num">{{ phase.key.split('_')[0] }}</span>
        <span class="phase-label">{{ phase.label }}</span>
        {% if phase.elapsed %}<span class="phase-time">{{ phase.elapsed }}</span>{% endif %}
      </a>
      {% endfor %}
    </nav>
    <div class="content" id="content">
      <iframe id="report-frame" src="{{ phases[0].report_path }}"></iframe>
    </div>
  </div>
  <script>
    function loadReport(el, path) {
      document.getElementById('report-frame').src = path;
      document.querySelectorAll('.sidebar a').forEach(a => a.classList.remove('active'));
      el.classList.add('active');
    }
  </script>
</body>
</html>"""


def _get_dashboard_template():
    env = Environment(autoescape=False)
    return env.from_string(DASHBOARD_TEMPLATE)


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description="Generate pipeline dashboard")
    parser.add_argument("session", help="Session identifier (e.g. 2025-26)")
    parser.add_argument("--run-id", default=None, help="Run ID (defaults to latest)")
    args = parser.parse_args()

    from tallgrass.session import STATE_DIR, KSSession

    ks = KSSession.from_session_string(args.session)
    session_dir = Path("results") / STATE_DIR / ks.output_name

    run_id = args.run_id
    if run_id is None:
        latest = session_dir / "latest"
        if latest.is_symlink():
            run_id = latest.resolve().name
        else:
            msg = f"No latest symlink found in {session_dir}. Provide --run-id."
            raise FileNotFoundError(msg)

    git_hash = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):  # fmt: skip
        pass

    generate_dashboard(session_dir, run_id, git_hash)
