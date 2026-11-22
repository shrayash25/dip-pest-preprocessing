"""Download a public stand-in dataset for the six pest classes.

Source: iNaturalist (https://www.inaturalist.org), research-grade observations
with Creative Commons photo licences. One photo is taken from each observation
so that near-duplicate frames are not included.

The first run asks the iNaturalist API for photos and writes a manifest
(data/raw/ATTRIBUTION.csv). Later runs read the manifest, so the dataset is
the same each time.

Usage:
    python download_dataset.py            # 100 images per class
    python download_dataset.py --per-class 50
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "dataset-fetch-script/1.0"
LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"
MIN_SIDE = 800  # skip photos smaller than this on the short side

WORKERS = 8     # parallel image downloads

# class folder name -> list of (iNaturalist taxon id, scientific name).
# The taxa are searched in order until the class has enough photos.
CLASSES = {
    "green_aphid": [(243745, "Myzus persicae"), (132882, "Acyrthosiphon pisum")],
    "yellow_aphid": [(71154, "Aphis nerii")],
    "black_aphid": [(1584763, "Aphis fabae (complex)")],
    "whitefly": [(154784, "Aleyrodidae")],
    "red_mite": [(203334, "Tetranychus")],
    "mealybug": [(49574, "Pseudococcidae")],
}

FIELDS = ["class", "file", "photo_id", "observation_url", "taxon", "license",
          "attribution", "photo_url"]


def get_json(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def find_photos(cls: str, taxon_id: int, wanted: int) -> list[dict]:
    """Return manifest rows for `wanted` photos of one class."""
    rows: list[dict] = []
    id_above = 0
    while len(rows) < wanted:
        data = get_json({
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photo_license": LICENSES,
            "photos": "true",
            "per_page": 200,
            "order_by": "id",
            "order": "asc",
            "id_above": id_above,
        })
        results = data.get("results", [])
        if not results:
            break
        for obs in results:
            id_above = obs["id"]
            photo = obs["photos"][0] if obs.get("photos") else None
            if not photo or not photo.get("license_code"):
                continue
            dims = photo.get("original_dimensions") or {}
            if min(dims.get("width", 0), dims.get("height", 0)) < MIN_SIDE:
                continue
            rows.append({
                "class": cls,
                "file": f"{cls}_{photo['id']}.jpg",
                "photo_id": photo["id"],
                "observation_url": f"https://www.inaturalist.org/observations/{obs['id']}",
                "taxon": obs["taxon"]["name"],
                "license": photo["license_code"],
                "attribution": photo["attribution"],
                "photo_url": photo["url"].replace("/square.", "/original."),
            })
            if len(rows) == wanted:
                break
        time.sleep(1.1)  # stay well under the API rate limit
    return rows


def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
            return True
        except (urllib.error.URLError, TimeoutError) as err:
            print(f"    retry {attempt + 1}: {err}")
            time.sleep(2 * (attempt + 1))
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "ATTRIBUTION.csv"

    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        print(f"Using existing manifest: {manifest} ({len(rows)} photos)")
    else:
        rows = []
        for cls, taxa in CLASSES.items():
            found: list[dict] = []
            for taxon_id, name in taxa:
                if len(found) >= args.per_class:
                    break
                print(f"Searching {cls} ({name}) ...", flush=True)
                found += find_photos(cls, taxon_id, args.per_class - len(found))
            print(f"  found {len(found)}", flush=True)
            rows.extend(found)
        with manifest.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote manifest: {manifest}")

    def fetch(row: dict) -> bool:
        dest = args.out / row["class"] / row["file"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            return True
        tmp = dest.with_suffix(".part")
        if download(row["photo_url"], tmp):
            tmp.replace(dest)
            return True
        print(f"  FAILED {row['photo_url']}", flush=True)
        return False

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(fetch, rows))
    done = sum(results)
    print(f"Finished: {done} present, {len(results) - done} failed, folder {args.out}")


if __name__ == "__main__":
    main()
