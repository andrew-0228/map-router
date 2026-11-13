import time
import folium
import networkx as nx
import osmnx as ox

from graph_io import load_graph
from algorithms import dijkstra, astar

G = load_graph()

# Two places: near Indiranagar metro (start) and Domlur (end), roughly
orig = ox.distance.nearest_nodes(G, X=77.6385, Y=12.9783)
dest = ox.distance.nearest_nodes(G, X=77.6401, Y=12.9609)

max_speed = max(d["speed_kph"] for _, _, d in G.edges(data=True)) / 3.6  # m/s

# 1. Ground truth from the library, to check your code
truth = nx.shortest_path_length(G, orig, dest, weight="travel_time")

# 2. Your algorithms
t0 = time.perf_counter()
path_d, cost_d, seen_d = dijkstra(G, orig, dest)
t_d = time.perf_counter() - t0

t0 = time.perf_counter()
path_a, cost_a, seen_a = astar(G, orig, dest, max_speed=max_speed)
t_a = time.perf_counter() - t0

print(f"NetworkX : {truth:.1f} s")
print(f"Dijkstra : {cost_d:.1f} s | {seen_d} nodes visited | {t_d*1000:.1f} ms")
print(f"A*       : {cost_a:.1f} s | {seen_a} nodes visited | {t_a*1000:.1f} ms")

# 3. Draw the route
coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path_a]
m = folium.Map(location=coords[0], zoom_start=15)
folium.PolyLine(coords, weight=6, color="crimson").add_to(m)
folium.Marker(coords[0], tooltip="Start").add_to(m)
folium.Marker(coords[-1], tooltip="End").add_to(m)
m.save("route.html")
print("Open route.html in your browser")
