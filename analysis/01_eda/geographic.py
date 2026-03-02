"""Kansas legislative district choropleth maps via Folium.

Downloads TIGER/Line shapefiles from the US Census Bureau, converts to GeoJSON,
and generates interactive Folium maps colored by party, ideology, or unity.

GeoJSON files are cached to data/external/ after first download. If the files
are not available (no internet, Census server down), all functions gracefully
return None so the EDA report still generates without maps.

Usage (called from eda.py or standalone):
    from analysis.geographic import create_district_maps, download_kansas_districts

    geojson = download_kansas_districts("house")
    maps = create_district_maps(geojson, legislators, ideal_points, "house")
"""

import json
import zipfile
from io import BytesIO
from pathlib import Path

import polars as pl

# ── Constants ────────────────────────────────────────────────────────────────

EXTERNAL_DATA_DIR = Path("data/external")

TIGER_BASE_URL = "https://www2.census.gov/geo/tiger/TIGER2024/SLDL"
TIGER_SENATE_URL = "https://www2.census.gov/geo/tiger/TIGER2024/SLDU"

# Kansas FIPS code
KANSAS_FIPS = "20"

HOUSE_SHAPEFILE = f"tl_2024_{KANSAS_FIPS}_sldl.zip"
SENATE_SHAPEFILE = f"tl_2024_{KANSAS_FIPS}_sldu.zip"

HOUSE_GEOJSON = EXTERNAL_DATA_DIR / "kansas_house_districts.geojson"
SENATE_GEOJSON = EXTERNAL_DATA_DIR / "kansas_senate_districts.geojson"

PARTY_COLORS = {
    "Republican": "#E81B23",
    "Democrat": "#0015BC",
    "Independent": "#999999",
}


# ── GeoJSON Download ─────────────────────────────────────────────────────────


def download_kansas_districts(chamber: str) -> Path | None:
    """Download and cache Kansas legislative district GeoJSON.

    Args:
        chamber: "house" or "senate".

    Returns:
        Path to cached GeoJSON file, or None if download fails.
    """
    if chamber.lower() == "house":
        geojson_path = HOUSE_GEOJSON
        url = f"{TIGER_BASE_URL}/{HOUSE_SHAPEFILE}"
        dist_field = "SLDLST"
    elif chamber.lower() == "senate":
        geojson_path = SENATE_GEOJSON
        url = f"{TIGER_SENATE_URL}/{SENATE_SHAPEFILE}"
        dist_field = "SLDUST"
    else:
        return None

    # Return cached if available
    if geojson_path.exists():
        return geojson_path

    try:
        import geopandas as gpd
        import requests

        print(f"  Downloading {chamber} district shapefile from Census Bureau...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        # Extract shapefile from zip
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extractall(tmpdir)
                # Find the .shp file
                shp_files = list(Path(tmpdir).glob("*.shp"))
                if not shp_files:
                    print(f"  WARNING: No .shp file found in {url}")
                    return None

                gdf = gpd.read_file(shp_files[0])

        # Convert district field to integer for joining
        gdf["district"] = gdf[dist_field].astype(int)

        # Reproject to WGS84 for Folium
        gdf = gdf.to_crs(epsg=4326)

        # Save as GeoJSON
        EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"  Cached: {geojson_path}")
        return geojson_path

    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: Could not download district data: {e}")
        return None


def load_geojson(path: Path) -> dict | None:
    """Load a GeoJSON file and return as dict."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Map Generation ───────────────────────────────────────────────────────────


def create_district_maps(
    geojson_path: Path,
    legislators: pl.DataFrame,
    ideal_points: pl.DataFrame | None,
    chamber: str,
) -> str | None:
    """Create a Folium map with party + ideology layers for one chamber.

    Args:
        geojson_path: Path to GeoJSON file with district boundaries.
        legislators: DataFrame with district, party, full_name columns.
        ideal_points: Optional DataFrame with legislator_slug, xi_mean columns.
        chamber: "house" or "senate".

    Returns:
        HTML string of the Folium map, or None if Folium is unavailable.
    """
    try:
        import folium
    except ImportError:
        return None

    geojson = load_geojson(geojson_path)
    if geojson is None:
        return None

    # Filter legislators to this chamber
    chamber_filter = "House" if chamber.lower() == "house" else "Senate"
    leg_df = legislators.filter(pl.col("chamber") == chamber_filter)

    if leg_df.height == 0:
        return None

    # Build district → legislator lookup
    district_data: dict[int, dict] = {}
    for row in leg_df.iter_rows(named=True):
        dist = row.get("district")
        if dist is None:
            continue
        try:
            dist_int = int(dist)
        except ValueError, TypeError:
            continue
        district_data[dist_int] = {
            "full_name": row.get("full_name", "Unknown"),
            "party": row.get("party", "Unknown"),
            "slug": row.get("legislator_slug", row.get("slug", "")),
        }

    # Join with ideal points if available
    if ideal_points is not None and "xi_mean" in ideal_points.columns:
        slug_col = "legislator_slug" if "legislator_slug" in ideal_points.columns else "slug"
        for dist, info in district_data.items():
            slug = info.get("slug", "")
            xi_row = ideal_points.filter(pl.col(slug_col) == slug)
            if xi_row.height > 0:
                info["xi_mean"] = float(xi_row["xi_mean"][0])

    # Calculate map center from GeoJSON bounds
    import geopandas as gpd

    gdf = gpd.read_file(geojson_path)
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    # Party layer
    party_group = folium.FeatureGroup(name="Party")
    _add_party_layer(party_group, geojson, district_data)
    party_group.add_to(m)

    # Ideology layer (if ideal points available)
    if any("xi_mean" in info for info in district_data.values()):
        ideology_group = folium.FeatureGroup(name="Ideology", show=False)
        _add_ideology_layer(ideology_group, geojson, district_data)
        ideology_group.add_to(m)

    folium.LayerControl().add_to(m)
    return m._repr_html_()


def _add_party_layer(
    group: object,
    geojson: dict,
    district_data: dict[int, dict],
) -> None:
    """Add party-colored districts to a Folium FeatureGroup."""
    import folium

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        dist_field = props.get("SLDLST") or props.get("SLDUST") or props.get("district")
        if dist_field is None:
            continue
        try:
            dist = int(dist_field)
        except ValueError, TypeError:
            continue

        info = district_data.get(dist, {})
        party = info.get("party", "Unknown")
        name = info.get("full_name", "Vacant")
        color = PARTY_COLORS.get(party, "#cccccc")

        tooltip = f"District {dist}: {name} ({party})"

        folium.GeoJson(
            feature,
            style_function=lambda _x, c=color: {
                "fillColor": c,
                "color": "#333",
                "weight": 1,
                "fillOpacity": 0.6,
            },
            tooltip=tooltip,
        ).add_to(group)


def _add_ideology_layer(
    group: object,
    geojson: dict,
    district_data: dict[int, dict],
) -> None:
    """Add ideology-colored districts (RdBu diverging) to a Folium FeatureGroup."""
    import folium

    # Compute xi range for color scaling
    xi_values = [info["xi_mean"] for info in district_data.values() if "xi_mean" in info]
    if not xi_values:
        return

    import numpy as np

    xi_max = max(abs(min(xi_values)), abs(max(xi_values)))
    if xi_max == 0:
        xi_max = 1.0

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        dist_field = props.get("SLDLST") or props.get("SLDUST") or props.get("district")
        if dist_field is None:
            continue
        try:
            dist = int(dist_field)
        except ValueError, TypeError:
            continue

        info = district_data.get(dist, {})
        name = info.get("full_name", "Vacant")
        xi = info.get("xi_mean")

        if xi is not None:
            # Map xi to RdBu: negative=blue, positive=red
            norm = np.clip(xi / xi_max, -1, 1)
            if norm < 0:
                r = int(255 * (1 + norm))
                g = int(255 * (1 + norm))
                b = 255
            else:
                r = 255
                g = int(255 * (1 - norm))
                b = int(255 * (1 - norm))
            color = f"#{r:02x}{g:02x}{b:02x}"
            tooltip = f"District {dist}: {name} (xi={xi:+.2f})"
        else:
            color = "#cccccc"
            tooltip = f"District {dist}: {name} (no IRT data)"

        folium.GeoJson(
            feature,
            style_function=lambda _x, c=color: {
                "fillColor": c,
                "color": "#333",
                "weight": 1,
                "fillOpacity": 0.7,
            },
            tooltip=tooltip,
        ).add_to(group)
