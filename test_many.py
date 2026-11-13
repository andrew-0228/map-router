import random
import networkx as nx

from graph_io import load_graph
from algorithms import astar, bidirectional_dijkstra, dijkstra

G = load_graph()
nodes = list(G.nodes)
max_speed = max(d["speed_kph"] for _, _, d in G.edges(data=True)) / 3.6

random.seed(0)
checked = 0
for _ in range(100):
    s, t = random.sample(nodes, 2)
    try:
        truth = nx.shortest_path_length(G, s, t, weight="travel_time")
    except nx.NetworkXNoPath:
        continue
    _, c1, _ = dijkstra(G, s, t)
    _, c2, _ = astar(G, s, t, max_speed=max_speed)
    _, c3, _ = bidirectional_dijkstra(G, s, t)
    assert abs(c1 - truth) < 1e-6, ("dijkstra wrong", s, t)
    assert abs(c2 - truth) < 1e-6, ("astar wrong", s, t)
    assert abs(c3 - truth) < 1e-6, ("bidirectional wrong", s, t)
    checked += 1
print("All good on", checked, "random trips (Dijkstra, A* and Bidirectional all match NetworkX)")
