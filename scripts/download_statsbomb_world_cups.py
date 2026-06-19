import csv
import json
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT = Path("data/external/statsbomb_open_data")

DIRS = [
    OUT,
    OUT / "matches",
    OUT / "events",
    OUT / "lineups",
    OUT / "three-sixty",
]

for d in DIRS:
    d.mkdir(parents=True, exist_ok=True)

manifest = []

counts = {
    "competitions": 0,
    "matches": 0,
    "events": 0,
    "lineups": 0,
    "three_sixty": 0,
    "failed_skipped": 0,
}

def download_file(url, local_path, file_type, competition_id=None, season_id=None, match_id=None, allow_404_skip=False):
    status = "failed"

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30) as response:
            content = response.read()

        if not content:
            print(f"Empty download skipped: {url}")
            counts["failed_skipped"] += 1
            status = "empty_skipped"
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Only overwrite if downloaded content is non-empty
            with open(local_path, "wb") as f:
                f.write(content)

            status = "downloaded"
            counts[file_type] += 1

    except HTTPError as e:
        if e.code == 404 and allow_404_skip:
            status = "not_available_404_skipped"
        else:
            print(f"Failed download: {url} — HTTP {e.code}")
            status = f"failed_http_{e.code}"
        counts["failed_skipped"] += 1

    except URLError as e:
        print(f"Failed download: {url} — {e}")
        status = "failed_url_error"
        counts["failed_skipped"] += 1

    except Exception as e:
        print(f"Failed download: {url} — {e}")
        status = "failed_exception"
        counts["failed_skipped"] += 1

    manifest.append({
        "source": url,
        "competition_id": competition_id or "",
        "season_id": season_id or "",
        "match_id": match_id or "",
        "file_type": file_type,
        "local_path": str(local_path),
        "status": status,
    })

    return status


# 1. competitions.json
download_file(
    f"{BASE_URL}/competitions.json",
    OUT / "competitions.json",
    "competitions",
)

world_cups = [
    {"competition_id": 43, "season_id": 106},  # 2022
    {"competition_id": 43, "season_id": 3},    # 2018
]

all_match_ids = []

# 2–3. Match files
for wc in world_cups:
    competition_id = wc["competition_id"]
    season_id = wc["season_id"]

    local_match_path = OUT / "matches" / str(competition_id) / f"{season_id}.json"
    url = f"{BASE_URL}/matches/{competition_id}/{season_id}.json"

    status = download_file(
        url,
        local_match_path,
        "matches",
        competition_id=competition_id,
        season_id=season_id,
    )

    if status == "downloaded" or local_match_path.exists():
        with open(local_match_path, "r", encoding="utf-8") as f:
            matches = json.load(f)

        for match in matches:
            all_match_ids.append({
                "match_id": match["match_id"],
                "competition_id": competition_id,
                "season_id": season_id,
            })

# Remove duplicates just in case
seen = set()
unique_matches = []
for m in all_match_ids:
    if m["match_id"] not in seen:
        unique_matches.append(m)
        seen.add(m["match_id"])

# 4–6. Events, lineups, 360
for m in unique_matches:
    match_id = m["match_id"]
    competition_id = m["competition_id"]
    season_id = m["season_id"]

    download_file(
        f"{BASE_URL}/events/{match_id}.json",
        OUT / "events" / f"{match_id}.json",
        "events",
        competition_id=competition_id,
        season_id=season_id,
        match_id=match_id,
    )

    download_file(
        f"{BASE_URL}/lineups/{match_id}.json",
        OUT / "lineups" / f"{match_id}.json",
        "lineups",
        competition_id=competition_id,
        season_id=season_id,
        match_id=match_id,
    )

    download_file(
        f"{BASE_URL}/three-sixty/{match_id}.json",
        OUT / "three-sixty" / f"{match_id}.json",
        "three_sixty",
        competition_id=competition_id,
        season_id=season_id,
        match_id=match_id,
        allow_404_skip=True,
    )

# Manifest
manifest_path = OUT / "download_manifest.csv"

with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "source",
            "competition_id",
            "season_id",
            "match_id",
            "file_type",
            "local_path",
            "status",
        ],
    )
    writer.writeheader()
    writer.writerows(manifest)

print("\nStatsBomb Open Data download summary")
print("-----------------------------------")
print(f"Competitions files downloaded: {counts['competitions']}")
print(f"Match files downloaded:        {counts['matches']}")
print(f"Event files downloaded:        {counts['events']}")
print(f"Lineup files downloaded:       {counts['lineups']}")
print(f"360 files downloaded:          {counts['three_sixty']}")
print(f"Failed/skipped downloads:      {counts['failed_skipped']}")
print(f"Manifest saved to:             {manifest_path}")
