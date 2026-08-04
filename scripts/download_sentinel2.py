"""
download_sentinel2.py
======================
Standalone script to discover and download Sentinel-2 L2A imagery over
Florida Bay from the Copernicus Data Space Ecosystem (CDSE).

IMPORTANT -- RUN THIS OUTSIDE THE AGENT SANDBOX:
    Run this from a normal terminal with real internet access. It will NOT
    run inside the coding-agent's sandboxed shell, which has no outbound
    network access (confirmed while building the DBHYDRO downloader).

WHY THIS SCRIPT'S DESIGN:
    Rather than searching for one scene per water-quality sample date
    (376 unique dates in the current WQ export -- see
    data/interim/wq_florida_bay_clean.parquet), this script queries the
    full study period ONCE, filters by cloud cover, and keeps the best
    (lowest-cloud) scene within each ~30-day window
    (config.yaml: ingestion.max_date_offset_days). This decouples Stage 2
    (build a spectral-index time series for the study area) from Stage 3
    (match individual WQ records to the nearest scene) and avoids
    downloading near-duplicate scenes for WQ dates that are close together.

AUTHENTICATION:
    CDSE uses a Keycloak OAuth2 "password" grant with the public client
    id `cdse-public` -- no client secret needed, just your account
    username/password (verified against the official CDSE token-endpoint
    documentation: https://documentation.dataspace.copernicus.eu/APIs/Token.html).
    Per that same documentation: "Please do not hardcode the username and
    password in the application code" -- this script prompts for them
    interactively (or reads CDSE_USERNAME / CDSE_PASSWORD env vars if set)
    rather than storing them in source.

    Access tokens are typically valid for ~30 minutes (confirmed via a
    live run: `expires_in` was 1800 seconds), though this script reads
    the actual value from each token response rather than assuming a
    fixed duration. It re-authenticates automatically before any request
    if the current token is close to expiring, since a full run
    (up to ~120 scene downloads) can exceed a single token's lifetime.

KNOWN UNCERTAINTIES (flagged so a failed run is easy to diagnose):
    - Band asset key names: verified via CDSE community documentation
      that assets are named like "B02_10m" / "B04_10m" / "B08_10m" for a
      sentinel-2-l2a STAC item, but this was not tested against a live
      response. The script prints the full asset-key list for the first
      matched scene so you can confirm/correct BAND_ASSET_KEYS below if
      the download step fails with a KeyError.
    - Asset href format: CDSE STAC assets may expose an `s3://` URI as
      the primary `href` with an HTTPS download link nested under an
      `alternate` property, per the STAC "alternate assets" extension
      convention CDSE is known to use -- but this was not confirmed
      against a live response either. The script tries the plain `href`
      first and falls back to `alternate.download.href` if the former
      isn't an http(s) URL; if both fail, it prints the raw asset dict.

OUTPUTS:
    data/raw/sentinel2/<scene_id>/B02.jp2, B04.jp2, B08.jp2  -- one
        subfolder per selected scene
    data/raw/sentinel2/scene_manifest.csv                     -- selected
        scene IDs, dates, cloud cover, and the 30-day bin they represent

USAGE:
    pip install requests
    python scripts/download_sentinel2.py

AUTHOR:   [Author placeholder]
CREATED:  2026
"""

from __future__ import annotations

import getpass
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATE_MIN = "2016-01-01"
DATE_MAX = "2026-01-01"  # matches the WQ export's date coverage

FLORIDA_BAY_BBOX = [-81.20, 24.85, -80.25, 25.25]  # [min_lon, min_lat, max_lon, max_lat]

# Florida Bay spans a 2x2 grid of Sentinel-2 MGRS tiles (T17RMH, T17RMJ,
# T17RNH, T17RNJ). Picking "lowest cloud cover across the whole bbox"
# without regard to which tile won meant most windows ended up with a
# scene that only covers a handful of the 22 WQ stations -- verified by
# checking each tile's actual raster bounds against real station
# coordinates: T17RNH alone covers 21/22 stations (all but FLAB25), far
# more than any other single tile. So every window is now searched for
# THIS specific tile rather than the bbox's overall best-cloud scene.
TARGET_TILE = "T17RNH"

CLOUD_COVER_MAX = 20.0     # percent
WINDOW_DAYS = 30           # matches config.yaml: ingestion.max_date_offset_days

BAND_ASSET_KEYS = ["B02_10m", "B04_10m", "B08_10m"]  # blue, red, NIR -- see docstring caveat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sentinel2"
MANIFEST_CSV = OUTPUT_DIR / "scene_manifest.csv"

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
STAC_COLLECTION = "sentinel-2-l2a"

REQUEST_TIMEOUT_S = 60
TOKEN_REFRESH_MARGIN_S = 60  # re-auth if less than this long left on the token


class CdseSession:
    """Holds a CDSE access token and refreshes it automatically before expiry."""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self._token: Optional[str] = None
        self._expires_at: float = 0.0

    def _authenticate(self) -> None:
        print("Requesting a new CDSE access token...")
        resp = requests.post(
            TOKEN_URL,
            data={
                "client_id": "cdse-public",
                "username": self.username,
                "password": self.password,
                "grant_type": "password",
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._token = payload["access_token"]
        self._expires_at = time.time() + payload.get("expires_in", 600)
        print(f"  -> token acquired, valid for {payload.get('expires_in', '?')} seconds")

    def token(self) -> str:
        if self._token is None or time.time() > self._expires_at - TOKEN_REFRESH_MARGIN_S:
            self._authenticate()
        return self._token

    def get(self, url: str, **kwargs) -> requests.Response:
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self.token()}"
        return requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_S, **kwargs)


def search_scenes(session: CdseSession) -> List[Dict[str, Any]]:
    """
    Query the CDSE STAC API for all sentinel-2-l2a scenes over Florida Bay
    within the study date range and below the cloud-cover threshold.

    Returns
    -------
    list of dict
        One dict per matched STAC item: id, datetime, cloud_cover, assets.
    """
    items: List[Dict[str, Any]] = []
    body = {
        "collections": [STAC_COLLECTION],
        "bbox": FLORIDA_BAY_BBOX,
        "datetime": f"{DATE_MIN}T00:00:00Z/{DATE_MAX}T00:00:00Z",
        "filter": {
            "op": "<=",
            "args": [{"property": "eo:cloud_cover"}, CLOUD_COVER_MAX],
        },
        "limit": 100,
    }

    request_url = STAC_SEARCH_URL
    request_method = "POST"
    request_body: Optional[Dict[str, Any]] = body
    page = 1

    while request_url:
        print(f"STAC search page {page}...")
        headers = {"Authorization": f"Bearer {session.token()}"}
        if request_method == "POST":
            resp = requests.post(request_url, json=request_body, headers=headers, timeout=REQUEST_TIMEOUT_S)
        else:
            resp = requests.get(request_url, headers=headers, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        payload = resp.json()

        for feat in payload.get("features", []):
            items.append(
                {
                    "id": feat["id"],
                    "datetime": feat["properties"]["datetime"],
                    "cloud_cover": feat["properties"].get("eo:cloud_cover"),
                    "assets": feat["assets"],
                }
            )

        # STAC pagination: the "next" link may specify its own HTTP method
        # and body (commonly POST, per the STAC API pagination extension)
        # rather than being a plain GET -- honour whatever it specifies
        # instead of assuming GET.
        next_link = next((l for l in payload.get("links", []) if l.get("rel") == "next"), None)
        if next_link is None:
            request_url = None
        else:
            request_url = next_link["href"]
            request_method = next_link.get("method", "GET").upper()
            if request_method == "POST":
                next_body = next_link.get("body", {})
                request_body = {**request_body, **next_body} if next_link.get("merge") else next_body
        page += 1

    print(f"Total scenes found (cloud_cover <= {CLOUD_COVER_MAX}%): {len(items)}")
    if items:
        print("Asset keys on first matched scene (verify BAND_ASSET_KEYS against this):")
        print(" ", list(items[0]["assets"].keys()))
    return items


def select_best_per_window(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep the lowest-cloud-cover TARGET_TILE scene within each WINDOW_DAYS bin.

    Filtering to one specific MGRS tile first (rather than picking the
    single best-cloud scene across the whole bbox regardless of tile) is
    what actually gives usable station coverage -- see the TARGET_TILE
    comment above.
    """
    tile_items = [item for item in items if f"_{TARGET_TILE}_" in item["id"]]
    print(f"{len(tile_items)} / {len(items)} scenes are on tile {TARGET_TILE}")

    date_min = datetime.fromisoformat(DATE_MIN)
    bins: Dict[int, Dict[str, Any]] = {}

    for item in tile_items:
        dt = datetime.fromisoformat(item["datetime"].replace("Z", "+00:00")).replace(tzinfo=None)
        bin_idx = (dt - date_min).days // WINDOW_DAYS
        current_best = bins.get(bin_idx)
        if current_best is None or (item["cloud_cover"] or 100) < (current_best["cloud_cover"] or 100):
            bins[bin_idx] = item

    selected = sorted(bins.values(), key=lambda x: x["datetime"])
    n_bins_total = (datetime.fromisoformat(DATE_MAX) - date_min).days // WINDOW_DAYS
    print(
        f"Selected {len(selected)} scenes across {n_bins_total} possible "
        f"{WINDOW_DAYS}-day windows ({n_bins_total - len(selected)} windows have no "
        f"tile-{TARGET_TILE} scene below the cloud-cover threshold)"
    )
    return selected


def _resolve_download_url(asset: Dict[str, Any]) -> str:
    """
    Return an HTTPS URL for an asset.

    CDSE STAC assets expose an `s3://` URI as the primary `href` (requiring
    separate S3 credentials we don't have) plus an `alternate` mapping with
    an OIDC-authenticated HTTPS mirror -- confirmed via a live response to
    live under `alternate.https.href` (auth:refs: ["oidc"], i.e. the same
    Bearer token used everywhere else in this script).
    """
    href = asset.get("href", "")
    if href.startswith("http"):
        return href
    alternates = asset.get("alternate", {})
    https_alt = alternates.get("https", {}).get("href")
    if https_alt and https_alt.startswith("http"):
        return https_alt
    # Fall back: scan any other alternate entry for an http(s) href in case
    # the key name varies across asset/band types.
    for alt in alternates.values():
        candidate = alt.get("href", "")
        if candidate.startswith("http"):
            return candidate
    raise RuntimeError(
        f"Could not find an HTTPS download URL in asset: {asset}. "
        "Inspect this structure and adjust _resolve_download_url()."
    )


def download_scene(session: CdseSession, item: Dict[str, Any]) -> None:
    """Download the configured bands for one scene into its own subfolder."""
    scene_dir = OUTPUT_DIR / item["id"]
    scene_dir.mkdir(parents=True, exist_ok=True)

    for band_key in BAND_ASSET_KEYS:
        asset = item["assets"].get(band_key)
        if asset is None:
            print(f"    WARNING: asset key '{band_key}' not found on scene {item['id']} -- skipping")
            continue

        band_name = band_key.split("_")[0]  # e.g. "B02_10m" -> "B02"
        out_path = scene_dir / f"{band_name}.jp2"
        if out_path.exists():
            print(f"    {out_path.name} already downloaded, skipping")
            continue

        url = _resolve_download_url(asset)
        print(f"    downloading {band_name} from {url[:80]}...")
        resp = session.get(url)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        print(f"    -> saved {out_path} ({len(resp.content) / 1e6:.1f} MB)")


def main() -> None:
    username = os.environ.get("CDSE_USERNAME") or input("CDSE username/email: ")
    password = os.environ.get("CDSE_PASSWORD") or getpass.getpass("CDSE password: ")
    session = CdseSession(username, password)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Step 1: searching for Sentinel-2 L2A scenes over Florida Bay")
    print("=" * 70)
    items = search_scenes(session)
    if not items:
        print("No scenes found -- check FLORIDA_BAY_BBOX, date range, and cloud threshold.")
        return

    print("\n" + "=" * 70)
    print(f"Step 2: selecting best scene per {WINDOW_DAYS}-day window")
    print("=" * 70)
    selected = select_best_per_window(items)

    manifest_rows = []
    print("\n" + "=" * 70)
    print(f"Step 3: downloading bands {BAND_ASSET_KEYS} for {len(selected)} scenes")
    print("=" * 70)
    for i, item in enumerate(selected, start=1):
        print(f"\n[{i}/{len(selected)}] {item['id']} ({item['datetime']}, cloud={item['cloud_cover']}%)")
        try:
            download_scene(session, item)
            manifest_rows.append(item)
        except Exception as e:
            print(f"  FAILED: {e}")

    import csv

    with MANIFEST_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "datetime", "cloud_cover"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({k: row[k] for k in ["id", "datetime", "cloud_cover"]})

    print(f"\nDone. {len(manifest_rows)} scenes downloaded. Manifest -> {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
