"""Loads and saves the road graph.

A big city graph takes minutes to read from a .graphml file but only seconds
from a pickle file, so load_map.py saves map.pkl and everything else loads it.
"""
import os
import pickle

import osmnx as ox

PICKLE = "map.pkl"
GRAPHML = "map.graphml"  # older maps downloaded before this change


def map_exists():
    return os.path.exists(PICKLE) or os.path.exists(GRAPHML)


def save_graph(G):
    with open(PICKLE, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph():
    if os.path.exists(PICKLE):
        with open(PICKLE, "rb") as f:
            return pickle.load(f)
    G = ox.load_graphml(GRAPHML)  # old map: convert it once for faster loading next time
    save_graph(G)
    return G
