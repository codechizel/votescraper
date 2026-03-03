# M5: Bill Lifecycle Scraper + Sankey

Capture KLISS API bill lifecycle data (currently discarded) and visualize bill flow through legislative stages as a Sankey diagram.

**Roadmap item:** R22 (Sankey diagrams for bill flow)
**Estimated effort:** 2-3 sessions (scraper changes need careful testing)
**Dependencies:** None (M4 hemicycle is independent)

---

## Phase A: Scraper — Capture Bill Lifecycle Data

### Problem

`_filter_bills_with_votes()` in `src/tallgrass/scraper.py:474-544` fetches the full KLISS API response for each bill, but only extracts `BILLNO`, `SHORTTITLE`, and `ORIGINAL_SPONSOR`. The `HISTORY` array — containing every committee referral, hearing, report, floor action, and governor action — is discarded.

### KLISS API `HISTORY` Field Structure

Verified from live API responses:

```json
{
  "BILLNO": "SB 1",
  "SHORTTITLE": "Reducing the state rate of sales tax on food",
  "HISTORY": [
    {
      "action_code": "cr_rsc_282",
      "chamber": "Senate",
      "committee_names": ["Committee on Assessment and Taxation"],
      "occurred_datetime": "2025-01-13T14:00:00",
      "session_date": "2025-01-13",
      "status": "Referred to Committee on Assessment and Taxation",
      "journal_page_number": "42"
    },
    {
      "action_code": "cr_rsc_282",
      "chamber": "Senate",
      "committee_names": ["Committee on Assessment and Taxation"],
      "occurred_datetime": "2025-02-05T10:30:00",
      "session_date": "2025-02-05",
      "status": "Committee Report recommending bill be passed as amended by",
      "journal_page_number": "156"
    }
  ]
}
```

### Data Availability

| Biennium | KLISS API Available | Notes |
|----------|-------------------|-------|
| 91st (2025-26) | Yes | Current — full HISTORY |
| 90th (2023-24) | Yes | Historical API path `/li_2024/api/v13/rev-1` |
| 89th (2021-22) | Yes | Historical API path `/li_2022/api/v13/rev-1` |
| 88th (2019-20) | Likely | Need to verify API endpoint |
| 84th-87th (2011-18) | Unknown | Pre-KLISS; JS fallback doesn't include HISTORY |

### New Dataclass: `BillAction`

Add to `src/tallgrass/models.py`:

```python
@dataclass(frozen=True)
class BillAction:
    """One action in a bill's legislative history."""

    session: str
    bill_number: str
    action_code: str
    chamber: str
    committee_names: list[str]
    occurred_datetime: str
    session_date: str
    status: str
    journal_page_number: str
```

### Scraper Changes

In `_filter_bills_with_votes()` (`scraper.py:474-544`), expand the metadata extraction loop:

```python
# Current (line 520-527):
for bill in data:
    bill_code = self._normalize_bill_code(bill.get("BILLNO", ""))
    if has_votes(bill):
        filtered_urls.append(url)
        metadata[bill_code] = {
            "short_title": bill.get("SHORTTITLE", ""),
            "sponsor": "; ".join(bill.get("ORIGINAL_SPONSOR", [])),
        }

# New (expand to capture HISTORY):
for bill in data:
    bill_code = self._normalize_bill_code(bill.get("BILLNO", ""))
    history_raw = bill.get("HISTORY", [])
    actions = [
        BillAction(
            session=self.session.output_name,
            bill_number=bill_code,
            action_code=entry.get("action_code", ""),
            chamber=entry.get("chamber", ""),
            committee_names=entry.get("committee_names", []),
            occurred_datetime=entry.get("occurred_datetime", ""),
            session_date=entry.get("session_date", ""),
            status=entry.get("status", ""),
            journal_page_number=str(entry.get("journal_page_number", "")),
        )
        for entry in history_raw
    ]
    if has_votes(bill):
        filtered_urls.append(url)
        metadata[bill_code] = {
            "short_title": bill.get("SHORTTITLE", ""),
            "sponsor": "; ".join(bill.get("ORIGINAL_SPONSOR", [])),
            "actions": actions,
        }
```

Store actions on the scraper instance and pass through to output:

```python
# In KSVoteScraper.__init__():
self.bill_actions: list[BillAction] = []

# After _filter_bills_with_votes():
for code, meta in self.bill_metadata.items():
    self.bill_actions.extend(meta.pop("actions", []))
```

### New CSV Export

In `src/tallgrass/output.py`, add a fourth CSV:

```python
def _save_bill_actions_csv(
    output_dir: Path,
    output_name: str,
    bill_actions: list[BillAction],
) -> None:
    """Export bill lifecycle actions to CSV."""
    if not bill_actions:
        return
    filepath = output_dir / f"{output_name}_bill_actions.csv"
    fieldnames = [
        "session", "bill_number", "action_code", "chamber",
        "committee_names", "occurred_datetime", "session_date",
        "status", "journal_page_number",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for action in bill_actions:
            row = dataclasses.asdict(action)
            row["committee_names"] = "; ".join(row["committee_names"])
            writer.writerow(row)
```

Update `save_csvs()` to accept and save actions:

```python
def save_csvs(
    output_dir: Path,
    output_name: str,
    individual_votes: list[IndividualVote],
    rollcalls: list[RollCall],
    legislators: dict[str, dict],
    bill_actions: list[BillAction] | None = None,  # NEW
) -> None:
    ...
    if bill_actions:
        _save_bill_actions_csv(output_dir, output_name, bill_actions)
```

---

## Phase B: Sankey Visualization

### Bill Lifecycle Stages

Map `status` text to canonical lifecycle stages:

```python
LIFECYCLE_STAGES = {
    "Introduced": ["introduced"],
    "Committee Referral": ["referred to committee"],
    "Hearing": ["hearing", "scheduled for hearing"],
    "Committee Report": ["committee report recommending"],
    "Committee of the Whole": ["committee of the whole"],
    "Floor Vote": ["final action", "emergency final action", "roll call"],
    "Cross-Chamber": ["received by", "transmitted to"],
    "Governor": ["enrolled and presented", "approved by governor", "vetoed"],
    "Signed into Law": ["approved by governor"],
    "Vetoed": ["vetoed by governor", "line item veto"],
    "Died": [],  # inferred: no further action after committee stage
}

def classify_action(status_text: str) -> str:
    """Map a KLISS status string to a canonical lifecycle stage."""
    status_lower = status_text.lower()
    for stage, keywords in LIFECYCLE_STAGES.items():
        if any(kw in status_lower for kw in keywords):
            return stage
    return "Other"
```

### Sankey Construction

Follow the existing Plotly Sankey pattern from `analysis/26_cross_session/cross_session.py:573-631`:

```python
import plotly.graph_objects as go

def plot_bill_lifecycle_sankey(
    actions_df: pl.DataFrame,
    title: str = "Bill Lifecycle Flow",
) -> go.Figure:
    """Sankey diagram showing bill flow through legislative stages.

    Each bill traces a path: Introduced → Committee → Report → Floor → ...
    Link thickness = number of bills flowing between stages.
    """
    # Classify each action
    actions_df = actions_df.with_columns(
        pl.col("status").map_elements(classify_action, return_dtype=pl.Utf8).alias("stage")
    )

    # Build transition counts (consecutive stages per bill)
    transitions = (
        actions_df
        .sort("bill_number", "occurred_datetime")
        .group_by("bill_number")
        .agg(pl.col("stage"))
        # Deduplicate consecutive same-stage entries
        ...
    )

    # Count transitions between stage pairs
    # source → target with value = count of bills making that transition

    labels = list(LIFECYCLE_STAGES.keys())
    label_idx = {label: i for i, label in enumerate(labels)}

    sources, targets, values = [], [], []
    for (src, tgt), count in transition_counts.items():
        if src in label_idx and tgt in label_idx:
            sources.append(label_idx[src])
            targets.append(label_idx[tgt])
            values.append(count)

    fig = go.Figure(data=[go.Sankey(
        node={"pad": 15, "thickness": 20, "label": labels,
              "color": ["#3498db", "#2ecc71", "#f39c12", "#e74c3c",
                        "#9b59b6", "#1abc9c", "#e67e22", "#27ae60",
                        "#c0392b", "#7f8c8d"][:len(labels)]},
        link={"source": sources, "target": targets, "value": values,
              "color": "rgba(200,200,200,0.4)"},
    )])
    fig.update_layout(title=title, width=900, height=600)
    return fig
```

### EDA Report Integration

Add to `analysis/01_eda/eda.py` and `analysis/01_eda/eda_report.py`:

```python
# In eda.py main(), after existing analysis:
actions_path = ctx.data_dir / f"{ctx.session.output_name}_bill_actions.csv"
if actions_path.exists():
    actions_df = pl.read_csv(actions_path)
    fig = plot_bill_lifecycle_sankey(actions_df)
    html_path = ctx.plots_dir / "bill_lifecycle_sankey.html"
    fig.write_html(html_path, include_plotlyjs="cdn")

# In eda_report.py:
def _add_bill_lifecycle_sankey(report, plots_dir):
    path = plots_dir / "bill_lifecycle_sankey.html"
    if path.exists():
        report.add(InteractiveSection(
            id="bill-lifecycle-sankey",
            title="Bill Lifecycle Flow",
            html=path.read_text(),
            caption="Sankey diagram showing how bills flow through legislative stages. "
                    "Width of each link is proportional to the number of bills.",
        ))
```

---

## Tests

### Scraper Tests

Add to `tests/test_scraper_http.py` or a new `tests/test_bill_actions.py`:

```python
class TestBillActionParsing:
    def test_history_extraction(self):
        """KLISS HISTORY array is parsed into BillAction objects."""

    def test_empty_history(self):
        """Bills without HISTORY produce no actions."""

    def test_committee_names_joined(self):
        """Multiple committee names are semicolon-joined in CSV."""

    def test_actions_csv_written(self, tmp_path):
        """bill_actions.csv is created when actions exist."""

    def test_no_actions_csv_when_empty(self, tmp_path):
        """No file created when no actions collected."""
```

### Analysis Tests

```python
class TestBillLifecycleSankey:
    def test_classify_action_referral(self):
        assert classify_action("Referred to Committee on Taxation") == "Committee Referral"

    def test_classify_action_floor(self):
        assert classify_action("Emergency Final Action") == "Floor Vote"

    def test_classify_action_governor(self):
        assert classify_action("Approved by Governor") == "Signed into Law"

    def test_sankey_produces_figure(self):
        """Smoke test: valid actions produce a Plotly figure."""
```

---

## Verification

```bash
# Phase A:
just scrape 2025                       # re-scrape to capture actions
ls data/kansas/91st_2025-2026/*_bill_actions.csv  # verify new CSV
just test-scraper                      # 264+ tests pass
just test                              # full suite

# Phase B:
just eda 2025-26                       # regenerate EDA with Sankey
# Open HTML report, verify Sankey diagram
```

## Documentation

- Update CLAUDE.md "Output" section to mention the 4th CSV
- Update CLAUDE.md "Data Model" section with BillAction fields
- Update `docs/roadmap.md` item R22 to "Done"
- Consider an ADR for the scraper schema expansion

## Commits

```
# Phase A:
feat(scraper): capture KLISS bill lifecycle HISTORY in bill_actions.csv [vYYYY.MM.DD.N]

# Phase B:
feat(infra): bill lifecycle Sankey diagram in EDA report [vYYYY.MM.DD.N]
```
