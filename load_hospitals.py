"""Downloads the hospitals inside your map from OpenStreetMap and saves them to hospitals.csv.

Run it once:   python load_hospitals.py
It needs the internet and the same map.pkl the app uses. It can take a minute or two.
"""
import csv
import sys
import time

import numpy as np
import osmnx as ox
from shapely.geometry import box

from graph_io import load_graph

MERGE_METRES = 150  # buildings of one hospital campus are often mapped separately: merge those closer than this
OUT_FILE = "hospitals.csv"
TAGS = {"amenity": "hospital"}
TILES = 3                    # the map is split into TILES x TILES smaller requests: the free servers accept those more easily
WAITS = [20, 40]             # last resort: seconds to wait between rounds of retrying a tiny piece
MAX_SPLITS = 2               # a piece the servers refuse is cut into 4 smaller ones, up to this many times
TIMEOUT = 90                 # seconds to wait for one server before giving up on it
ENDPOINTS = [                # free public OpenStreetMap servers, tried in this order
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
]


def clean(value):
    """OpenStreetMap values can be missing (NaN) or not text. Return a tidy string or ''."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def merge_close(rows, metres=MERGE_METRES):
    """rows: list of dicts with name, lat, lon, emergency. Keeps one row per cluster of nearby ones
    (prefers a named one) and remembers if any of them has an emergency department."""
    rows = sorted(rows, key=lambda r: (r["name"] == "", r["name"]))  # named hospitals first
    kept = []
    for r in rows:
        for k in kept:
            dy = (r["lat"] - k["lat"]) * 111_000
            dx = (r["lon"] - k["lon"]) * 111_000 * np.cos(np.radians(r["lat"]))
            if dx * dx + dy * dy < metres * metres:
                k["emergency"] = k["emergency"] or r["emergency"]
                break
        else:
            kept.append(dict(r))
    return kept


def rows_from_features(gdf):
    """Turn the table OpenStreetMap gives us into simple rows of name, lat, lon, emergency."""
    points = gdf.geometry.representative_point()  # a point inside buildings too (centroid can fall outside)
    rows = []
    for i in range(len(gdf)):
        row = gdf.iloc[i]
        name = clean(row.get("name:en")) or clean(row.get("name"))
        rows.append({"name": name, "lat": float(points.iloc[i].y), "lon": float(points.iloc[i].x),
                     "emergency": clean(row.get("emergency")).lower() == "yes"})
    return rows


def set_endpoint(url):
    ox.settings.overpass_url = url       # osmnx 2.x
    ox.settings.overpass_endpoint = url  # osmnx 1.x


def pieces(minx, miny, maxx, maxy, n=TILES, prefix="piece"):
    """A rectangle cut into an n x n grid. Returns (label, rectangle) pairs."""
    xs = np.linspace(minx, maxx, n + 1)
    ys = np.linspace(miny, maxy, n + 1)
    return [(f"{prefix} {r * n + c + 1} of {n * n}", box(xs[c], ys[r], xs[c + 1], ys[r + 1])) for r in range(n) for c in range(n)]


def try_servers(area, label, fetch):
    """One pass over the servers. Returns ("ok", table), ("empty", None) or ("fail", last error)."""
    last = None
    for url in ENDPOINTS:
        set_endpoint(url)
        try:
            return "ok", fetch(area)
        except Exception as error:
            if "InsufficientResponse" in type(error).__name__:  # osmnx says this when an area has no hospitals
                return "empty", None
            last = error
            print(f"  {label}: {url.split('/')[2]} did not answer ({type(error).__name__}).")
    return "fail", last


def fetch_piece(area, label, fetch=None, sleep=time.sleep, depth=0):
    """Download the hospitals inside one rectangle. Returns a list of tables (it can be empty).
    A rectangle the servers refuse is usually just too big (the middle of the city has many hospitals), so it is cut
    into four smaller ones. Raises RuntimeError if even the smallest pieces keep failing."""
    fetch = fetch or (lambda polygon: ox.features_from_polygon(polygon, tags=TAGS))
    status, value = try_servers(area, label, fetch)
    if status == "ok":
        return [value]
    if status == "empty":
        return []
    if depth < MAX_SPLITS:
        print(f"  {label}: too big or too busy, splitting it into 4 smaller pieces...")
        out = []
        for sub_label, sub in pieces(*area.bounds, n=2, prefix=f"{label}, part"):
            out += fetch_piece(sub, sub_label, fetch, sleep, depth + 1)
        return out
    for wait in WAITS:
        print(f"  {label}: waiting {wait} s, then trying the servers again...")
        sleep(wait)
        status, value = try_servers(area, label, fetch)
        if status == "ok":
            return [value]
        if status == "empty":
            return []
    raise RuntimeError(f"{label} failed on every server. Last error: {str(value)[:200]}")


def main():
    ox.settings.requests_timeout = TIMEOUT
    G = load_graph()
    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]
    print("Downloading hospitals from OpenStreetMap in small pieces. This can take a few minutes. Please wait...")
    rows, failed = [], []
    for label, area in pieces(min(xs), min(ys), max(xs), max(ys)):
        try:
            tables = fetch_piece(area, label)
        except RuntimeError as error:
            print("  " + str(error))
            failed.append(label)
            continue
        found = [r for gdf in tables if gdf is not None and len(gdf) for r in rows_from_features(gdf)]
        rows += found
        print(f"  {label}: {len(found)} hospitals")
    if failed:
        print(f"\nNo file was saved because {len(failed)} piece(s) could not be downloaded ({', '.join(failed)}).")
        print("The free servers are busy. Wait 10 to 15 minutes and run this again. It is safe to repeat: finished pieces are remembered.")
        sys.exit(1)
    rows = merge_close(rows)  # also removes the duplicates that appear where two pieces touch
    rows.sort(key=lambda r: (r["name"] == "", r["name"]))
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "lat", "lon", "emergency"])
        for r in rows:
            writer.writerow([r["name"] or "Unnamed hospital", f"{r['lat']:.6f}", f"{r['lon']:.6f}", "yes" if r["emergency"] else ""])
    named = sum(1 for r in rows if r["name"])
    print(f"Saved {len(rows)} hospitals to {OUT_FILE} ({named} with names, {sum(r['emergency'] for r in rows)} marked as having an emergency department).")
    print("Restart the app and open the Hospital access page.")


if __name__ == "__main__":
    main()
