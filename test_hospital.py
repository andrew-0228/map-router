"""Checks the nearest-hospital search against NetworkX."""
import random

import networkx as nx

from algorithms import (build_reverse_adjacency, nearest_source_distances, route_to_nearest, path_cost)
from graph_io import load_graph

G = load_graph()
ids = list(G.nodes)
random.seed(2)
for weight in ("travel_time", "length"):
    radj = build_reverse_adjacency(G, weight)
    for count in (1, 5, 40):
        hospitals = random.sample(ids, count)
        dist, toward, origin = nearest_source_distances(radj, hospitals)
        # NetworkX: search the reversed graph from all hospitals at once
        ref = nx.multi_source_dijkstra_path_length(G.reverse(copy=False), hospitals, weight=weight)
        assert dist.keys() == ref.keys()
        assert all(abs(dist[n] - ref[n]) < 1e-6 for n in ref), (weight, count)
        # the route we return must really be a drivable route, end at the claimed hospital, and cost dist[n]
        for n in random.sample(ids, 30):
            path = route_to_nearest(toward, n)
            assert path[0] == n and path[-1] == origin[n] and path[-1] in hospitals
            assert all(G.has_edge(a, b) for a, b in zip(path, path[1:]))
            assert abs(path_cost(G, path, weight) - dist[n]) < 1e-6
        print(f"{weight:11s} {count:2d} hospitals: OK")
print("All nearest-hospital checks passed")
