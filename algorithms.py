import heapq
import math

import numpy as np


def edge_weight(G, u, v, weight, penalty=None):
    """A road graph can have several parallel roads between two nodes.
    Pick the cheapest one. penalty is an optional {(u, v): multiplier} dictionary
    that makes chosen roads more expensive (used to find alternative routes)."""
    w = min(data[weight] for data in G[u][v].values())
    if penalty:
        w *= penalty.get((u, v), 1.0)
    return w


def build_path(prev, source, target):
    """Walk backwards from target to source using the 'prev' dictionary."""
    if target != source and target not in prev:
        return None  # target was never reached
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def dijkstra(G, source, target, weight="travel_time", explored=None, penalty=None):
    """If you pass a list as explored, it is filled with the nodes in the order
    the search visited them (used for drawing)."""
    dist = {source: 0.0}
    prev = {}
    queue = [(0.0, source)]
    visited = 0

    while queue:
        d, u = heapq.heappop(queue)
        if d > dist[u]:
            continue  # stale entry: we already found a better way to u
        visited += 1
        if explored is not None:
            explored.append(u)
        if u == target:
            break
        for v in G[u]:
            new_d = d + edge_weight(G, u, v, weight, penalty)
            if new_d < dist.get(v, math.inf):
                dist[v] = new_d
                prev[v] = u
                heapq.heappush(queue, (new_d, v))

    path = build_path(prev, source, target)
    return path, dist.get(target), visited


def haversine(lat1, lon1, lat2, lon2):
    """Straight-line distance in metres between two lat/lon points."""
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def astar(G, source, target, weight="travel_time", max_speed=None, explored=None, penalty=None):
    """max_speed (metres/second) is only needed when weight='travel_time'."""
    ty, tx = G.nodes[target]["y"], G.nodes[target]["x"]

    def h(n):
        straight = haversine(G.nodes[n]["y"], G.nodes[n]["x"], ty, tx)
        return straight / max_speed if weight == "travel_time" else straight

    dist = {source: 0.0}
    prev = {}
    queue = [(h(source), source)]
    visited = 0
    done = set()

    while queue:
        _, u = heapq.heappop(queue)
        if u in done:
            continue
        done.add(u)
        visited += 1
        if explored is not None:
            explored.append(u)
        if u == target:
            break
        for v in G[u]:
            new_d = dist[u] + edge_weight(G, u, v, weight, penalty)
            if new_d < dist.get(v, math.inf):
                dist[v] = new_d
                prev[v] = u
                heapq.heappush(queue, (new_d + h(v), v))

    path = build_path(prev, source, target)
    return path, dist.get(target), visited


def bidirectional_dijkstra(G, source, target, weight="travel_time", explored=None, explored_back=None):
    """Dijkstra from both ends at once. One search goes forward from the source,
    the other goes backward from the target (following one-way roads in reverse).
    They stop when they can prove no shorter route exists.

    explored      : filled with the nodes the forward search visited
    explored_back : filled with the nodes the backward search visited
                    (if you leave this out, both go into explored)
    """
    if explored_back is None:
        explored_back = explored

    if source == target:
        if explored is not None:
            explored.append(source)
        return [source], 0.0, 1

    # index 0 = forward search, index 1 = backward search
    dist = ({source: 0.0}, {target: 0.0})
    prev = ({}, {})
    queues = ([(0.0, source)], [(0.0, target)])
    settled = (set(), set())
    sinks = (explored, explored_back)

    best = math.inf   # cost of the best complete route found so far
    meet = None       # the node where the two searches joined up for that route
    visited = 0

    while True:
        # throw away heap entries for nodes that are already finished
        for s in (0, 1):
            while queues[s] and queues[s][0][1] in settled[s]:
                heapq.heappop(queues[s])
        if not queues[0] or not queues[1]:
            break
        # Stop: any route not found yet must cost at least top_forward + top_backward
        if queues[0][0][0] + queues[1][0][0] >= best:
            break

        # expand the side whose next node is cheaper to reach
        side = 0 if queues[0][0][0] <= queues[1][0][0] else 1
        d, u = heapq.heappop(queues[side])
        settled[side].add(u)
        visited += 1
        if sinks[side] is not None:
            sinks[side].append(u)

        # forward looks at roads leaving u, backward looks at roads entering u
        neighbours = G[u] if side == 0 else G.pred[u]
        for v in neighbours:
            w = edge_weight(G, u, v, weight) if side == 0 else edge_weight(G, v, u, weight)
            new_d = d + w
            if new_d < dist[side].get(v, math.inf):
                dist[side][v] = new_d
                prev[side][v] = u
                heapq.heappush(queues[side], (new_d, v))
            # has the other search already reached v? then we have a complete route
            if v in dist[1 - side]:
                total = dist[side][v] + dist[1 - side][v]
                if total < best:
                    best, meet = total, v

    if meet is None:
        return None, None, visited

    # forward half: source -> meet
    path = [meet]
    while path[-1] != source:
        path.append(prev[0][path[-1]])
    path.reverse()
    # backward half: meet -> target
    node = meet
    while node != target:
        node = prev[1][node]
        path.append(node)
    return path, best, visited



def path_cost(G, path, weight="travel_time"):
    """Total cost of a path (no penalties)."""
    return sum(edge_weight(G, u, v, weight) for u, v in zip(path, path[1:]))


def _edge_lengths(G, path):
    """{(u, v): length in metres} for every road in a path."""
    out = {}
    for u, v in zip(path, path[1:]):
        out[(u, v)] = min(d["length"] for d in G[u][v].values())
    return out


def alternative_routes(G, source, target, base_path, weight="travel_time", max_speed=None,
                       count=2, penalty_factor=1.6, max_stretch=1.5, max_overlap=0.65, tries=8):
    """Find up to `count` genuinely different routes besides base_path (the best one).

    The penalty method: make the roads of the routes found so far more expensive,
    search again, and keep the new route only if
      - it is not much slower than the best route (cost <= max_stretch x best), and
      - it does not share more than max_overlap of its length with a route we already kept.
    Returns a list of paths (it can be shorter than `count` if the map has no good alternatives).
    """
    base_cost = path_cost(G, base_path, weight)
    kept = [base_path]
    kept_edges = [set(_edge_lengths(G, base_path))]
    penalty = {}

    def add_penalty(path):
        for edge in zip(path, path[1:]):
            penalty[edge] = penalty.get(edge, 1.0) * penalty_factor

    add_penalty(base_path)
    alternatives = []
    for _ in range(tries):
        cand, _, _ = astar(G, source, target, weight=weight, max_speed=max_speed, penalty=penalty)
        if cand is None:
            break
        if cand in kept:
            add_penalty(cand)
            continue
        ok = path_cost(G, cand, weight) <= max_stretch * base_cost
        if ok:
            lengths = _edge_lengths(G, cand)
            total = sum(lengths.values()) or 1.0
            for edges in kept_edges:
                shared = sum(length for e, length in lengths.items() if e in edges)
                if shared / total > max_overlap:
                    ok = False
                    break
        if ok:
            kept.append(cand)
            kept_edges.append(set(_edge_lengths(G, cand)))
            alternatives.append(cand)
            if len(alternatives) >= count:
                break
        add_penalty(cand)  # push the next search somewhere new, even when we rejected this one
    return alternatives


# ---------- meeting point for a group ----------
def build_adjacency(G, weight="travel_time"):
    """A plain {node: [(neighbour, cost), ...]} version of the graph.
    Parallel roads are collapsed to the cheapest one. Building it once makes the
    full searches below much faster than asking the graph for every edge."""
    return {u: [(v, min(d[weight] for d in edges.values())) for v, edges in G[u].items()] for u in G}


def all_distances(adj, source):
    """Dijkstra that never stops early: the cost from source to EVERY node it can reach.
    Returns (dist, prev) dictionaries."""
    dist = {source: 0.0}
    prev = {}
    queue = [(0.0, source)]
    while queue:
        d, u = heapq.heappop(queue)
        if d > dist[u]:
            continue  # stale entry
        for v, w in adj[u]:
            new_d = d + w
            if new_d < dist.get(v, math.inf):
                dist[v] = new_d
                prev[v] = u
                heapq.heappush(queue, (new_d, v))
    return dist, prev


def meeting_point(costs, goal="fair"):
    """Pick the best place for a group to meet.

    costs[i][k] is how much it costs person i to travel to candidate place k
    (so each row comes from one all_distances search started at that person).

    goal = "fair"  : minimise the LONGEST trip (nobody has to travel far)
    goal = "total" : minimise the SUM of everyone's trips
    Ties are broken by the other measure. Returns (k, [cost of each person]).
    """
    costs = np.asarray(costs, dtype=float)
    worst = costs.max(axis=0)   # the longest trip if the group meets at k
    total = costs.sum(axis=0)   # the combined trip length if the group meets at k
    first, second = (worst, total) if goal == "fair" else (total, worst)
    k = int(np.lexsort((second, first))[0])  # sorts by first, then second; take the best
    return k, costs[:, k].tolist()


# ---------- nearest hospital (or any set of places) ----------
def build_reverse_adjacency(G, weight="travel_time"):
    """Like build_adjacency, but every road is flipped. Searching outward over the flipped roads
    from a hospital gives the cost for people to travel TO it (it matters because of one-way streets)."""
    radj = {u: [] for u in G}
    for u in G:
        for v, edges in G[u].items():
            radj[v].append((u, min(d[weight] for d in edges.values())))
    return radj


def nearest_source_distances(radj, sources):
    """One search that starts from ALL the sources at the same time (multi-source Dijkstra).

    radj    : the flipped adjacency from build_reverse_adjacency
    sources : the nodes where the hospitals are
    Returns three dictionaries, one entry for every node that can reach a source:
      dist[n]   cost for someone at n to reach their nearest source
      toward[n] the next node to drive to from n to get there (follow it to get the route)
      origin[n] which source node that nearest one is
    """
    dist, toward, origin = {}, {}, {}
    queue = []
    for s in sources:
        if s not in dist:
            dist[s] = 0.0
            origin[s] = s
            queue.append((0.0, s))
    heapq.heapify(queue)
    while queue:
        d, u = heapq.heappop(queue)
        if d > dist[u]:
            continue  # stale entry
        for v, w in radj[u]:  # v can drive to u, and u already leads to a source
            new_d = d + w
            if new_d < dist.get(v, math.inf):
                dist[v] = new_d
                toward[v] = u
                origin[v] = origin[u]
                heapq.heappush(queue, (new_d, v))
    return dist, toward, origin


def route_to_nearest(toward, node):
    """The list of nodes from `node` to its nearest source (the source itself returns [node])."""
    path = [node]
    while path[-1] in toward:
        path.append(toward[path[-1]])
    return path


# ---------- walking ----------
WALK_MPS = 1.4  # walking speed in metres per second (about 5 km/h)


def build_walk_adjacency(G, speed=WALK_MPS):
    """Walking graph: people on foot ignore one-way signs, so every road works in both directions.
    The cost of a road is the seconds it takes to walk it. Because every road goes both ways, the same graph
    answers 'how long to walk from A' and 'how long to walk to A'."""
    adj = {u: [] for u in G}
    for u in G:
        for v, edges in G[u].items():
            seconds = min(d["length"] for d in edges.values()) / speed
            adj[u].append((v, seconds))
            adj[v].append((u, seconds))
    return adj


def walk_length(G, path):
    """Metres along a path when walking (either direction of a road may be used)."""
    total = 0.0
    for u, v in zip(path, path[1:]):
        lengths = []
        if G.has_edge(u, v):
            lengths += [d["length"] for d in G[u][v].values()]
        if G.has_edge(v, u):
            lengths += [d["length"] for d in G[v][u].values()]
        total += min(lengths)
    return total
