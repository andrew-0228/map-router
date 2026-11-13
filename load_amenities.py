"""Downloads places from OpenStreetMap (pharmacies, schools, parks, ...) for the Access map and Neighbourhood score pages.

Run it once:              python load_amenities.py
Only some kinds:          python load_amenities.py pharmacy school
Download again:           python load_amenities.py --refresh

Each kind is saved on its own in the places/ folder, and kinds that are already there are skipped. If the free
servers are busy and a kind fails, run the command again later: it carries on with what is missing.
It needs the internet and the same map.pkl the app uses, and can take 10 to 30 minutes in total.
"""
import csv
import os
import sys

import osmnx as ox

from graph_io import load_graph
from load_hospitals import TIMEOUT, TILES, fetch_piece, merge_close, pieces, rows_from_features

PLACES_DIR = "places"
OLD_HOSPITAL_FILE = "hospitals.csv"  # made by load_hospitals.py: the app uses it for hospitals
# key, label, OpenStreetMap tags, and how close (in metres) two entries must be to count as the same place
CATEGORIES = [
    ("hospital", "hospitals", {"amenity": "hospital"}, 150),
    ("pharmacy", "pharmacies", {"amenity": "pharmacy"}, 30),
    ("fire_station", "fire stations", {"amenity": "fire_station"}, 100),
    ("police", "police stations", {"amenity": "police"}, 100),
    ("station", "metro and railway stations", {"railway": "station"}, 150),
    ("school", "schools", {"amenity": "school"}, 80),
    ("park", "parks", {"leisure": "park"}, 300),
    ("supermarket", "supermarkets", {"shop": "supermarket"}, 50),
]


def download(key, label, tags, merge_metres, bounds):
    """Download one kind of place. Returns the rows, or None if a piece of the map could not be downloaded."""
    one = {"hospital": "hospital", "pharmacy": "pharmacy", "fire_station": "fire station", "police": "police station",
           "station": "station", "school": "school", "park": "park", "supermarket": "supermarket"}[key]
    fetch = lambda polygon: ox.features_from_polygon(polygon, tags=tags)
    rows = []
    for piece_label, area in pieces(*bounds, n=TILES):
        try:
            tables = fetch_piece(area, f"{label}, {piece_label}", fetch=fetch)
        except RuntimeError as error:
            # the servers are busy: stop here instead of waiting through every other piece as well
            print("  " + str(error))
            print(f"  {label}: stopped, nothing was saved for it. Try again later.")
            return None
        found = [r for gdf in tables if gdf is not None and len(gdf) for r in rows_from_features(gdf)]
        rows += found
        print(f"  {label}, {piece_label}: {len(found)}")
    rows = merge_close(rows, merge_metres)
    rows.sort(key=lambda r: (r["name"] == "", r["name"]))
    for r in rows:
        r["name"] = r["name"] or f"Unnamed {one}"
    return rows


def save(key, rows):
    os.makedirs(PLACES_DIR, exist_ok=True)
    with open(os.path.join(PLACES_DIR, f"{key}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "lat", "lon", "emergency"])
        for r in rows:
            writer.writerow([r["name"], f"{r['lat']:.6f}", f"{r['lon']:.6f}", "yes" if r["emergency"] else ""])


def main(argv):
    refresh = "--refresh" in argv
    wanted = [a for a in argv if not a.startswith("--")]
    unknown = [a for a in wanted if a not in {c[0] for c in CATEGORIES}]
    if unknown:
        print(f"Unknown kind(s): {', '.join(unknown)}. Choose from: {', '.join(c[0] for c in CATEGORIES)}")
        sys.exit(2)
    ox.settings.requests_timeout = TIMEOUT
    G = load_graph()
    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]
    bounds = (min(xs), min(ys), max(xs), max(ys))
    failed, saved = [], []
    for key, label, tags, merge_metres in CATEGORIES:
        if wanted and key not in wanted:
            continue
        have = os.path.exists(os.path.join(PLACES_DIR, f"{key}.csv"))
        if key == "hospital" and not have and os.path.exists(OLD_HOSPITAL_FILE) and not refresh:
            print(f"hospitals: using the {OLD_HOSPITAL_FILE} you already made.")
            continue
        if have and not refresh:
            print(f"{label}: already downloaded, skipping. (Use --refresh to download again.)")
            continue
        print(f"\nDownloading {label}...")
        rows = download(key, label, tags, merge_metres, bounds)
        if rows is None:
            failed.append(label)
            continue
        save(key, rows)
        saved.append(label)
        print(f"Saved {len(rows)} {label} to {PLACES_DIR}/{key}.csv")
    print("\nDone." if not failed else f"\nNot downloaded: {', '.join(failed)}.")
    if failed:
        print("The free servers are busy. Wait 10 to 15 minutes and run this command again. It only downloads what is missing.")
        sys.exit(1)
    print("Restart the app to use the new places.")


if __name__ == "__main__":
    main(sys.argv[1:])
