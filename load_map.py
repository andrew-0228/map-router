import networkx as nx
import osmnx as ox

from graph_io import save_graph

# How much of Bengaluru to download:
#   "east"    = Indiranagar to HSR Layout (about 12 km square), a few minutes
#   "central" = central Bengaluru (about 16 km square), 10 to 20 minutes
#   "city"    = the whole city, can take 20 to 60 minutes and needs several GB of RAM
MAP_SIZE = "city"

ox.settings.log_console = True  # print progress so you can see it is working

if MAP_SIZE == "east":
    G = ox.graph_from_point((12.945, 77.644), dist=6000, network_type="drive")
elif MAP_SIZE == "central":
    G = ox.graph_from_point((12.9716, 77.5946), dist=8000, network_type="drive")
else:
    G = ox.graph_from_place("Bengaluru, Karnataka, India", network_type="drive")

print("Downloaded. Adding speeds and travel times...")
G = ox.add_edge_speeds(G)          # adds speed_kph from road type
G = ox.add_edge_travel_times(G)    # adds travel_time in seconds

# Keep only the largest piece where you can drive from any intersection to any other,
# so every random trip has a route.
biggest = max(nx.strongly_connected_components(G), key=len)
G = G.subgraph(biggest).copy()

# Drop everything we never use (street names, shapes, ...) to save memory and disk space.
for _, _, d in G.edges(data=True):
    for key in list(d):
        if key not in ("length", "speed_kph", "travel_time"):
            del d[key]
for _, d in G.nodes(data=True):
    for key in list(d):
        if key not in ("x", "y"):
            del d[key]

save_graph(G)  # writes map.pkl, loaded quickly by the other scripts
print("Saved map.pkl")
print("Nodes (intersections):", len(G.nodes))
print("Edges (road segments):", len(G.edges))
