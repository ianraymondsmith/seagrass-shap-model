"""
download_dbhydro_wq.py
=======================
Standalone script to discover SFWMD DBHYDRO water-quality monitoring stations
in Florida Bay and download their historical water-quality time series.

IMPORTANT -- RUN THIS OUTSIDE THE AGENT SANDBOX:
    This script must be run from a normal terminal with real internet access
    (e.g. a plain PowerShell/Anaconda Prompt window). It will NOT run inside
    the coding-agent's sandboxed shell, which has no outbound DNS/network
    access.

WHY THIS SCRIPT EXISTS:
    DBHYDRO has no official public REST/JSON API. The only documented
    programmatic access pattern is the one reverse-engineered by the
    ropensci/dbhydroR R package (https://github.com/ropensci/dbhydroR),
    which drives the same legacy PL/SQL web endpoints used by the DBHYDRO
    Browser web UI. This script re-implements that same access pattern in
    Python: no account/API key is required for these endpoints.

WHAT IT DOES:
    1. Queries the station-discovery endpoint for every station tagged
       category="WQ" (water quality), and keeps only stations whose
       reported coordinates fall inside the Florida Bay bounding box
       (24.85-25.25 N, -81.20 to -80.25 W).
    2. Queries the WQ data-export endpoint for those stations across the
       configured date range and saves the raw (unfiltered) result.

WHY test_name IS NOT FILTERED:
    The data-export endpoint's WHERE clause supports "test_name IN (...)",
    but DBHYDRO's internal test_name strings for chlorophyll-a, TP, TN,
    salinity, etc. are undocumented and NOT verified here. Rather than
    guess and silently return zero rows on a typo, this script pulls ALL
    parameters for the matched stations. After the first successful run,
    inspect the distinct `Test Name` values printed at the end and use
    them to populate config/config.yaml's `ingestion.column_rename_map`.

KNOWN UNCERTAINTIES (flagged so a failed run is easy to diagnose):
    - The discovery endpoint returns an HTML page; this script assumes the
      *third* <table> on the page (index 2) is the results grid, matching
      dbhydroR's behaviour. If DBHYDRO's page layout has changed, the
      diagnostic table-shape printout below will make that obvious.
    - Latitude/Longitude parsing: dbhydroR applies a decimal-point
      reinsertion because DBHYDRO returns coordinates as compact digit
      strings. This script does the same, but ALSO prints the raw values
      for the first few matched rows so you can visually confirm they land
      in Florida Bay (~25 N, -80.5 W) before trusting the filtered set.

OUTPUTS:
    data/raw/dbhydro_stations_florida_bay.csv   -- discovered station metadata
    data/raw/wq_florida_bay_raw.csv             -- raw WQ time series (all parameters)

USAGE:
    pip install requests lxml pandas
    python scripts/download_dbhydro_wq.py

AUTHOR:   [Author placeholder]
CREATED:  2026
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Configuration -- edit these if the study period or study area changes.
# -----------------------------------------------------------------------------
DATE_MIN = "2000-01-01"
DATE_MAX = "2026-01-01"

FLORIDA_BAY_BOUNDS = {
    "lon_min": -81.20,
    "lon_max": -80.25,
    "lat_min": 24.85,
    "lat_max": 25.25,
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
STATIONS_CSV = OUTPUT_DIR / "dbhydro_stations_florida_bay.csv"
WQ_CSV = OUTPUT_DIR / "wq_florida_bay_raw.csv"

STATION_DISCOVERY_URL = "https://my.sfwmd.gov/dbhydroplsql/show_dbkey_info.show_dbkeys_matched"
WQ_DATA_URL = "https://my.sfwmd.gov/dbhydroplsql/water_quality_data.report_full"

STATION_BATCH_SIZE = 40  # keep the generated URL length reasonable
REQUEST_TIMEOUT_S = 60
SECONDS_BETWEEN_REQUESTS = 1.0  # be polite to a shared government server

SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "seagrass-shap-pipeline-research/1.0 (contact: ianraymondsmith@gmail.com)"}
)


def _get(url: str, params: dict) -> requests.Response:
    resp = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp


def _parse_packed_coordinate(raw: str, is_longitude: bool) -> float | None:
    """
    Convert DBHYDRO's packed coordinate string (no reliable decimal point)
    into a decimal-degree float, mirroring dbhydroR's format_coords().

    Parameters
    ----------
    raw : str
        Raw coordinate field as scraped from the HTML table.
    is_longitude : bool
        If True, the result is forced negative (Florida Bay is west of
        the prime meridian; DBHYDRO does not encode the sign).

    Returns
    -------
    float or None
        Parsed decimal-degree coordinate, or None if unparseable.
    """
    if raw is None:
        return None
    digits = str(raw).replace(".", "").strip()
    if len(digits) < 4 or not digits.replace("-", "").isdigit():
        return None
    try:
        value = float(f"{digits[:2]}.{digits[3:]}")
    except ValueError:
        return None
    return -value if is_longitude else value


def discover_stations() -> pd.DataFrame:
    """Query DBHYDRO for every station tagged category=WQ."""
    params = {
        "v_category": "WQ",
        "v_station": "%",
        "v_js_flag": "Y",
        "v_order_by": "STATION",
        "v_dbkey_list_flag": "Y",
        "display_quantity": "100000",
    }
    print(f"Requesting station list from {STATION_DISCOVERY_URL} ...")
    resp = _get(STATION_DISCOVERY_URL, params)

    tables = pd.read_html(io.StringIO(resp.text))
    print(f"Found {len(tables)} <table> elements on the discovery page:")
    for i, t in enumerate(tables):
        print(f"  [{i}] shape={t.shape} columns={list(t.columns)[:8]}")

    if len(tables) < 3:
        raise RuntimeError(
            "Expected at least 3 tables on the DBHYDRO discovery page "
            "(dbhydroR uses index 2 as the results grid) but found "
            f"{len(tables)}. DBHYDRO's page layout may have changed -- "
            "inspect the printed table shapes above and adjust the index."
        )

    df = tables[2].copy()
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    return df


def filter_to_florida_bay(stations: pd.DataFrame) -> pd.DataFrame:
    """Keep only stations whose parsed coordinates fall inside Florida Bay."""
    lat_col = next((c for c in stations.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in stations.columns if "long" in c.lower()), None)
    if lat_col is None or lon_col is None:
        raise RuntimeError(
            "Could not find latitude/longitude columns in the discovery "
            f"table. Columns present: {list(stations.columns)}"
        )

    print(f"\nRaw coordinate samples (column '{lat_col}' / '{lon_col}'):")
    print(stations[[lat_col, lon_col]].head(5).to_string(index=False))

    lat = stations[lat_col].map(lambda v: _parse_packed_coordinate(v, is_longitude=False))
    lon = stations[lon_col].map(lambda v: _parse_packed_coordinate(v, is_longitude=True))

    mask = (
        lat.between(FLORIDA_BAY_BOUNDS["lat_min"], FLORIDA_BAY_BOUNDS["lat_max"])
        & lon.between(FLORIDA_BAY_BOUNDS["lon_min"], FLORIDA_BAY_BOUNDS["lon_max"])
    )
    result = stations[mask].copy()
    result["latitude_parsed"] = lat[mask]
    result["longitude_parsed"] = lon[mask]
    print(
        f"\n{int(mask.sum())} / {len(stations)} stations fall inside the "
        f"Florida Bay bounding box {FLORIDA_BAY_BOUNDS}"
    )
    return result


def download_wq(station_ids: list[str]) -> pd.DataFrame:
    """Pull raw WQ time-series data for the given station IDs (all parameters)."""
    date_min_fmt = pd.Timestamp(DATE_MIN).strftime("%d-%b-%Y").upper()
    date_max_fmt = pd.Timestamp(DATE_MAX).strftime("%d-%b-%Y").upper()

    frames = []
    n_batches = (len(station_ids) + STATION_BATCH_SIZE - 1) // STATION_BATCH_SIZE
    for batch_num, i in enumerate(range(0, len(station_ids), STATION_BATCH_SIZE), start=1):
        batch = station_ids[i : i + STATION_BATCH_SIZE]
        station_list = "(" + ",".join(f"'{s}'" for s in batch) + ")"
        where_clause = (
            f"where date_collected > '{date_min_fmt}' "
            f"and date_collected < '{date_max_fmt}' "
            f"and station_id in {station_list}"
        )
        params = {
            "v_where_clause": where_clause,
            "v_target_code": "file_csv",
            "v_exc_flagged": "Y",  # exclude QA/QC-flagged records
            "v_exc_qc": "N",       # keep field QC replicate samples
        }
        print(f"\nBatch {batch_num}/{n_batches}: requesting {len(batch)} stations...")
        resp = _get(WQ_DATA_URL, params)

        if not resp.text.strip():
            print("  -> empty response, skipping this batch")
            continue
        try:
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception as e:
            print(f"  -> could not parse response as CSV ({e})")
            print(f"  -> first 300 chars of response: {resp.text[:300]!r}")
            continue

        print(f"  -> {len(df)} rows, {len(df.columns)} columns")
        frames.append(df)
        time.sleep(SECONDS_BETWEEN_REQUESTS)

    if not frames:
        raise RuntimeError("No data retrieved for any station batch. See warnings above.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Step 1: discovering DBHYDRO water-quality stations")
    print("=" * 70)
    stations = discover_stations()
    fb_stations = filter_to_florida_bay(stations)

    if fb_stations.empty:
        print(
            "\nNo stations matched the Florida Bay bounding box. This most "
            "likely means the coordinate-parsing assumption in "
            "_parse_packed_coordinate() does not match the current DBHYDRO "
            "page format -- check the raw coordinate samples printed above "
            "and adjust the parser."
        )
        sys.exit(1)

    fb_stations.to_csv(STATIONS_CSV, index=False)
    print(f"\nSaved station metadata -> {STATIONS_CSV}")

    station_id_col = next(c for c in fb_stations.columns if "station" in c.lower())
    station_ids = sorted(fb_stations[station_id_col].dropna().astype(str).unique().tolist())
    print(f"{len(station_ids)} unique station IDs: {station_ids}")

    print("\n" + "=" * 70)
    print(f"Step 2: downloading WQ time series ({DATE_MIN} to {DATE_MAX})")
    print("=" * 70)
    wq = download_wq(station_ids)
    wq.to_csv(WQ_CSV, index=False)
    print(f"\nSaved {len(wq)} rows -> {WQ_CSV}")

    test_col = next((c for c in wq.columns if "test" in c.lower() and "name" in c.lower()), None)
    if test_col:
        print("\nDistinct Test Name values found (use these to fill in")
        print("config/config.yaml's ingestion.column_rename_map):")
        for t in sorted(wq[test_col].dropna().unique()):
            print(" -", t)
    else:
        print(f"\nNo 'Test Name' column found. Columns present: {list(wq.columns)}")


if __name__ == "__main__":
    main()
