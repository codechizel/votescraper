"""CAP policy classification via Claude API (Phase 18, optional).

Classifies bills into the 20 Comparative Agendas Project (CAP) major topic
categories using Claude Sonnet.  Responses are cached by content hash for
reproducibility — subsequent runs skip API calls for already-classified bills.

Graceful degradation: if ANTHROPIC_API_KEY is not set, all functions return
empty results and the main script skips CAP sections in the report.

Cost estimate: ~800 bills × ~5K tokens in + ~100 tokens out ≈ $13 standard,
~$7 with Batch API (50% discount).
"""

import hashlib
import json
import os
from pathlib import Path

import polars as pl

# ── CAP Categories ───────────────────────────────────────────────────────────

CAP_CATEGORIES: dict[str, str] = {
    "macroeconomics": "taxes, budgets, fiscal policy, government spending, debt",
    "civil_rights": "discrimination, voting rights, civil liberties, privacy",
    "health": "healthcare, insurance, mental health, substance abuse, public health",
    "agriculture": "farming, livestock, food safety, rural development",
    "labor": "employment, wages, workers compensation, unions, workplace safety",
    "education": "schools, universities, K-12 funding, teacher pay, student aid",
    "environment": "pollution, conservation, water quality, wildlife, climate",
    "energy": "oil, gas, electricity, renewable energy, utilities",
    "immigration": "immigration, refugees, border, citizenship",
    "transportation": "roads, highways, public transit, aviation, rail",
    "law_crime": "criminal justice, courts, corrections, policing, sentencing",
    "social_welfare": "poverty, social services, disability, child welfare",
    "housing": "housing, homelessness, zoning, property, landlord-tenant",
    "banking": "banking, finance, insurance regulation, securities",
    "defense": "military, veterans, national guard, emergency management",
    "technology": "telecommunications, internet, cybersecurity, data privacy",
    "foreign_trade": "trade, tariffs, exports, international commerce",
    "government_operations": "government efficiency, elections, public records, state agencies",
    "public_lands": "parks, water resources, land management, public property",
    "cultural_policy": "arts, sports, recreation, gambling, alcohol, tobacco",
}

CAP_CATEGORY_LABELS: dict[str, str] = {
    "macroeconomics": "Macroeconomics",
    "civil_rights": "Civil Rights",
    "health": "Health",
    "agriculture": "Agriculture",
    "labor": "Labor",
    "education": "Education",
    "environment": "Environment",
    "energy": "Energy",
    "immigration": "Immigration",
    "transportation": "Transportation",
    "law_crime": "Law & Crime",
    "social_welfare": "Social Welfare",
    "housing": "Housing",
    "banking": "Banking & Finance",
    "defense": "Defense",
    "technology": "Technology",
    "foreign_trade": "Foreign Trade",
    "government_operations": "Government Operations",
    "public_lands": "Public Lands",
    "cultural_policy": "Cultural Policy",
}

DEFAULT_MODEL = "claude-sonnet-4-5-20241022"
"""Claude model for CAP classification."""


# ── API Key Check ────────────────────────────────────────────────────────────


def has_api_key() -> bool:
    """Check if ANTHROPIC_API_KEY is available."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ── Content Hashing ──────────────────────────────────────────────────────────


def _content_hash(text: str, model: str) -> str:
    """SHA-256 hash of bill text + model name for cache key."""
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(text.encode())
    return h.hexdigest()[:20]


# ── Prompt Construction ──────────────────────────────────────────────────────


def _build_classification_prompt(bill_text: str, bill_number: str) -> str:
    """Build the CAP classification prompt for a single bill."""
    categories_desc = "\n".join(f"- {key}: {desc}" for key, desc in CAP_CATEGORIES.items())

    return f"""Classify this Kansas state legislature bill into one of the \
20 Comparative Agendas Project (CAP) major topic categories.

## Categories
{categories_desc}

## Bill
Bill number: {bill_number}

Text (may be truncated):
{bill_text[:6000]}

## Instructions
Respond with a JSON object containing:
- "primary_category": the single best CAP category key from the list above
- "confidence": integer 1-5 (1=uncertain, 5=very confident)
- "top3": array of the 3 most relevant category keys, in order of relevance
- "summary": a single sentence summarizing what this bill does (plain English, no jargon)

Respond ONLY with the JSON object, no other text."""


# ── Classification ───────────────────────────────────────────────────────────


def _load_cache(cache_path: Path) -> dict[str, dict]:
    """Load cached API responses."""
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_cache(cache_path: Path, cache: dict[str, dict]) -> None:
    """Save API response cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def _parse_response(response_text: str) -> dict:
    """Parse Claude's JSON response, handling markdown code fences."""
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {
            "primary_category": "government_operations",
            "confidence": 1,
            "top3": ["government_operations"],
            "summary": "Classification failed — could not parse API response.",
        }
    return result


def classify_bills_cap(
    texts: list[str],
    bill_numbers: list[str],
    cache_path: Path,
    model: str = DEFAULT_MODEL,
) -> pl.DataFrame:
    """Classify bills into CAP categories via Claude API with response caching.

    Args:
        texts: Bill texts (preprocessed or raw).
        bill_numbers: Corresponding bill numbers.
        cache_path: Path to JSON cache file.
        model: Claude model to use.

    Returns:
        DataFrame with columns: bill_number, cap_category, cap_label,
        cap_confidence, cap_top3, bill_summary.
    """
    if not has_api_key():
        print("  WARNING: ANTHROPIC_API_KEY not set — skipping CAP classification")
        return pl.DataFrame(
            schema={
                "bill_number": pl.Utf8,
                "cap_category": pl.Utf8,
                "cap_label": pl.Utf8,
                "cap_confidence": pl.Int64,
                "cap_top3": pl.Utf8,
                "bill_summary": pl.Utf8,
            }
        )

    cache = _load_cache(cache_path)
    client = None  # Lazy — created only when an API call is needed

    results: list[dict] = []
    n_cached = 0
    n_api = 0

    for text, bill_number in zip(texts, bill_numbers, strict=True):
        content_key = _content_hash(text, model)

        if content_key in cache:
            parsed = cache[content_key]
            n_cached += 1
        else:
            if client is None:
                import anthropic

                client = anthropic.Anthropic()

            prompt = _build_classification_prompt(text, bill_number)
            response = client.messages.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text
            parsed = _parse_response(response_text)
            cache[content_key] = parsed
            n_api += 1

            # Save cache periodically (every 50 API calls)
            if n_api % 50 == 0:
                _save_cache(cache_path, cache)

        category = parsed.get("primary_category", "government_operations")
        results.append(
            {
                "bill_number": bill_number,
                "cap_category": category,
                "cap_label": CAP_CATEGORY_LABELS.get(category, category),
                "cap_confidence": parsed.get("confidence", 1),
                "cap_top3": "; ".join(parsed.get("top3", [category])),
                "bill_summary": parsed.get("summary", ""),
            }
        )

    # Final cache save
    _save_cache(cache_path, cache)
    print(f"  CAP classification: {n_cached} cached, {n_api} API calls")

    return pl.DataFrame(results)


def classify_bills_cap_batch(
    texts: list[str],
    bill_numbers: list[str],
    cache_path: Path,
    model: str = DEFAULT_MODEL,
) -> str:
    """Submit bills for batch classification (50% cost discount).

    Returns the batch ID.  Results must be retrieved separately once the batch
    completes (typically <1 hour).

    Only sends uncached bills; cached results are preserved.
    """
    if not has_api_key():
        msg = "ANTHROPIC_API_KEY required for batch classification"
        raise RuntimeError(msg)

    import anthropic

    client = anthropic.Anthropic()
    cache = _load_cache(cache_path)

    # Build requests for uncached bills only
    requests = []
    for text, bill_number in zip(texts, bill_numbers, strict=True):
        content_key = _content_hash(text, model)
        if content_key not in cache:
            prompt = _build_classification_prompt(text, bill_number)
            requests.append(
                {
                    "custom_id": f"{bill_number}|{content_key}",
                    "params": {
                        "model": model,
                        "max_tokens": 300,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                }
            )

    if not requests:
        print("  All bills already cached — no batch needed")
        return ""

    print(f"  Submitting batch: {len(requests)} bills (uncached)")
    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch ID: {batch.id}")
    return batch.id
