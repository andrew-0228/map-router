import random
import time

import matplotlib
matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt

from graph_io import load_graph
from algorithms import astar, bidirectional_dijkstra, dijkstra

NUM_TRIPS = 100  # lower this if your map is very large

G = load_graph()
nodes = list(G.nodes)
max_speed = max(d["speed_kph"] for _, _, d in G.edges(data=True)) / 3.6  # m/s

random.seed(42)  # same trips every run, so results are comparable

NAMES = ["Dijkstra", "A*", "Bidirectional"]
results = {n: {"visited": [], "ms": []} for n in NAMES}


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    path, cost, visited = fn(*args, **kwargs)
    return path, visited, (time.perf_counter() - t0) * 1000


done = 0
while done < NUM_TRIPS:
    s, t = random.sample(nodes, 2)

    path, v, ms = timed(dijkstra, G, s, t)
    if path is None:
        continue  # no route between these two points, pick another pair
    results["Dijkstra"]["visited"].append(v)
    results["Dijkstra"]["ms"].append(ms)

    _, v, ms = timed(astar, G, s, t, max_speed=max_speed)
    results["A*"]["visited"].append(v)
    results["A*"]["ms"].append(ms)

    _, v, ms = timed(bidirectional_dijkstra, G, s, t)
    results["Bidirectional"]["visited"].append(v)
    results["Bidirectional"]["ms"].append(ms)

    done += 1
    if done % 20 == 0:
        print(f"  {done}/{NUM_TRIPS} trips done...")


def avg(xs):
    return sum(xs) / len(xs)


avg_visited = [avg(results[n]["visited"]) for n in NAMES]
avg_ms = [avg(results[n]["ms"]) for n in NAMES]

print(f"Average over {NUM_TRIPS} random trips")
for n, v, m in zip(NAMES, avg_visited, avg_ms):
    print(f"{n:14}: {v:9.0f} nodes visited | {m:8.2f} ms | {avg_visited[0] / v:4.1f}x fewer nodes than Dijkstra")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
colors = ["#6b4bd8", "#d6336c", "#0c8599"]
ax1.bar(NAMES, avg_visited, color=colors)
ax1.set_title("Average nodes visited")
ax2.bar(NAMES, avg_ms, color=colors)
ax2.set_title("Average time per trip (ms)")
fig.suptitle(f"Dijkstra vs A* vs Bidirectional on {len(nodes)} intersections, {NUM_TRIPS} random trips")
fig.tight_layout()
fig.savefig("benchmark.png", dpi=150)
print("Saved benchmark.png")
