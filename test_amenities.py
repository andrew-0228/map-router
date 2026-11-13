"""Checks the walking search (used for amenities) against NetworkX."""
import random

import networkx as nx

from algorithms import (WALK_MPS, build_walk_adjacency, nearest_source_distances, route_to_nearest, walk_length)
from graph_io import load_graph

G = load_graph()
ids = list(G.nodes)
random.seed(3)
adj = build_walk_adjacency(G)
undirected = G.to_undirected()
for count in (1, 8, 60):
    places = random.sample(ids, count)
    dist, toward, origin = nearest_source_distances(adj, places)
    ref = nx.multi_source_dijkstra_path_length(undirected, places, weight="length")
    assert dist.keys() == ref.keys()
    assert all(abs(dist[n] - ref[n] / WALK_MPS) < 1e-6 for n in ref), count
    for n in random.sample(ids, 30):
        path = route_to_nearest(toward, n)
        assert path[0] == n and path[-1] == origin[n] and path[-1] in places
        assert all(G.has_edge(a, b) or G.has_edge(b, a) for a, b in zip(path, path[1:]))
        assert abs(walk_length(G, path) / WALK_MPS - dist[n]) < 1e-6
    print(f"walking, {count:2d} places: OK")
# walking can only be as slow as, or faster in distance than, the one-way-limited driving distance
print("All walking checks passed")
