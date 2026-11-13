"""Checks the meeting-point search against NetworkX and a brute-force answer."""
import random
import time

import networkx as nx
import numpy as np

from algorithms import all_distances, build_adjacency, meeting_point
from graph_io import load_graph

G = load_graph()
ids = list(G.nodes)
adj = build_adjacency(G, "travel_time")
random.seed(1)
print(f"Graph: {len(ids)} nodes")

# 1. all_distances must agree with NetworkX
for _ in range(5):
    s = random.choice(ids)
    mine, _ = all_distances(adj, s)
    ref = nx.single_source_dijkstra_path_length(G, s, weight="travel_time")
    assert mine.keys() == ref.keys()
    assert all(abs(mine[n] - ref[n]) < 1e-6 for n in ref)
print("all_distances matches NetworkX")

# 2. meeting_point must match a plain loop over every place, for groups of 2 to 8 people
for size in (2, 3, 5, 8):
    people = random.sample(ids, size)
    t0 = time.time()
    costs = np.full((size, len(ids)), np.inf)
    index = {n: i for i, n in enumerate(ids)}
    for row, p in enumerate(people):
        d, _ = all_distances(adj, p)
        for n, v in d.items():
            costs[row, index[n]] = v
    took = time.time() - t0
    for goal in ("fair", "total"):
        k, per_person = meeting_point(costs, goal)
        key = (lambda c: max(c)) if goal == "fair" else (lambda c: sum(c))
        best = min(key(costs[:, j]) for j in range(len(ids)))
        assert abs(key(per_person) - best) < 1e-6, (size, goal)
    print(f"{size} people OK ({took:.1f}s for {size} full searches)")
print("All meeting-point checks passed")
