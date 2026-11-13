import folium
import osmnx as ox

from graph_io import load_graph
from algorithms import astar, bidirectional_dijkstra, dijkstra

G = load_graph()

# Same start and end as main.py. Change these to try other trips.
orig = ox.distance.nearest_nodes(G, X=77.6385, Y=12.9783)
dest = ox.distance.nearest_nodes(G, X=77.6401, Y=12.9609)

max_speed = max(d["speed_kph"] for _, _, d in G.edges(data=True)) / 3.6  # m/s

runs = {}
fwd, back = [], []
runs["Dijkstra"] = (*dijkstra(G, orig, dest, explored=fwd), fwd, [])
fwd, back = [], []
runs["A*"] = (*astar(G, orig, dest, max_speed=max_speed, explored=fwd), fwd, [])
fwd, back = [], []
runs["Bidirectional"] = (*bidirectional_dijkstra(G, orig, dest, explored=fwd, explored_back=back), fwd, back)
ROUTE_COLOUR = {"Dijkstra": "#6b4bd8", "A*": "#d6336c", "Bidirectional": "#0c8599"}


def blend(c1, c2, t):
    """Mix two hex colours. t=0 gives c1, t=1 gives c2."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def dots(m, points, c1, c2):
    points = points[::max(1, len(points) // 2500)]   # at most about 2500 dots
    n = len(points)
    for i, node in enumerate(points):
        folium.CircleMarker(
            (G.nodes[node]["y"], G.nodes[node]["x"]),
            radius=4, weight=0, fill=True,
            fill_color=blend(c1, c2, i / max(n - 1, 1)), fill_opacity=0.9,
        ).add_to(m)


def draw(name):
    path, cost, visited, explored, explored_back = runs[name]
    m = folium.Map(location=(G.nodes[orig]["y"], G.nodes[orig]["x"]), zoom_start=15)
    dots(m, explored, "#ffe08a", "#5b1a6e")            # forward search: yellow to purple
    dots(m, explored_back, "#a5f3fc", "#0b4f6c")       # backward search: cyan to blue
    coords = [(G.nodes[x]["y"], G.nodes[x]["x"]) for x in path]
    folium.PolyLine(coords, weight=5, color=ROUTE_COLOUR[name]).add_to(m)
    folium.Marker(coords[0], tooltip="Start").add_to(m)
    folium.Marker(coords[-1], tooltip="End").add_to(m)
    return m._repr_html_()


columns = "".join(
    f'<div class="col"><h2>{name}: {runs[name][2]} nodes explored</h2><div>{draw(name)}</div></div>'
    for name in runs
)

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Dijkstra vs A* vs Bidirectional</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #fafafa; }}
h1 {{ margin: 0 0 4px; }}
p {{ margin: 0 0 16px; color: #555; }}
.row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.col {{ flex: 1 1 380px; min-width: 0; }}
.col h2 {{ margin: 0 0 4px; font-size: 20px; }}
</style></head><body>
<h1>Which intersections did each algorithm explore?</h1>
<p>Dots are intersections the search visited. Yellow to purple = forward search, cyan to blue = backward search (Bidirectional only). The line is the final route.</p>
<div class="row">{columns}</div>
</body></html>"""

with open("search.html", "w", encoding="utf-8") as f:
    f.write(page)

for name, run in runs.items():
    print(f"{name:14}: explored {run[2]} nodes (route cost {run[1]:.1f} s)")
print("Open search.html in your browser")
