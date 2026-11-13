import csv
import html
import math
import os
import random
import re
import time

import folium
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

from algorithms import (all_distances, alternative_routes, astar, bidirectional_dijkstra, build_adjacency,
                        build_reverse_adjacency, build_walk_adjacency, dijkstra, haversine, meeting_point,
                        nearest_source_distances, route_to_nearest, walk_length, WALK_MPS)
from graph_io import load_graph, map_exists

st.set_page_config(page_title="Wayfinder | A route lab for Bengaluru", page_icon="🗺️", layout="wide")

MAX_DOTS = 2500  # most dots drawn on the map, so big searches do not slow the page
DIJKSTRA_COLOUR = "#6b4bd8"
ASTAR_COLOUR = "#d6336c"
BIDIR_COLOUR = "#0c8599"

# Place search uses the free OpenStreetMap geocoder (Nominatim). Please keep usage light:
# it is meant for small projects (about one request per second, results are cached below).
GEOCODER_URL = os.environ.get("MARGA_GEOCODER_URL", "https://nominatim.openstreetmap.org/search")
VIEWBOX = "77.35,13.25,77.85,12.70"  # west, north, east, south: keeps results inside Bengaluru
USER_AGENT = {"User-Agent": "Wayfinder-route-lab/1.0 (student project)"}

# Meet in the middle
MAX_PEOPLE = 20            # every person needs one full search of the city
OFF_MAP_LIMIT = 2000       # metres: a person further than this from any road on the map is not added
PERSON_COLOURS = ["#e03131", "#1971c2", "#2f9e44", "#f08c00", "#9c36b5", "#0c8599",
                  "#d6336c", "#5f3dc4", "#e8590c", "#66a80f", "#364fc7", "#c2255c"]
FAIR_COLOUR, TOTAL_COLOUR = "#d6336c", "#6b4bd8"
EXAMPLE_GROUP = [("Indiranagar", 12.9784, 77.6408), ("Jayanagar", 12.9308, 77.5838),
                 ("Hebbal", 13.0358, 77.5970), ("Koramangala", 12.9352, 77.6245)]
PAGE_ROUTES, PAGE_MEET, PAGE_ACCESS, PAGE_SCORE = "Route lab", "Meet in the middle", "Access map", "Neighbourhood score"

# A small custom component (the geo_button folder next to this file) that asks the browser for your location.
GEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geo_button")
geo_button = components.declare_component("marga_geo_button", path=GEO_DIR) if os.path.isdir(GEO_DIR) else None

# Approximate coordinates (latitude, longitude). Each is snapped to the nearest intersection.
# Trips that fall outside the map you downloaded are hidden automatically.
PRESETS = [
    ("Indiranagar → Domlur", (12.9783, 77.6385), (12.9609, 77.6401)),
    ("Indiranagar → HSR Layout", (12.9784, 77.6408), (12.9116, 77.6474)),
    ("Koramangala → MG Road", (12.9352, 77.6245), (12.9756, 77.6068)),
    ("Hebbal → Jayanagar", (13.0358, 77.5970), (12.9308, 77.5838)),
    ("Majestic → Electronic City", (12.9767, 77.5713), (12.8456, 77.6603)),
    ("MG Road → Whitefield", (12.9756, 77.6068), (12.9698, 77.7500)),
    ("Yeshwanthpur → Banashankari", (13.0285, 77.5409), (12.9255, 77.5468)),
]

# ---------- styling ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&display=swap');
.block-container { padding-top: 4rem; max-width: 1400px; }
#MainMenu, footer { visibility: hidden; }
.hero { background: linear-gradient(120deg, #4a2c7a 0%, #b03a6f 55%, #f08a5d 100%);
        border-radius: 26px; padding: 38px 44px; color: #fff; margin-bottom: 22px; }
.hero h1 { margin: 0 0 8px 0; padding: 0; font-size: 2.6rem; line-height: 1.1; color: #fff; }
.hero p { margin: 0; font-size: 1.08rem; opacity: .93; max-width: 760px; }
.chips { margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; }
.chip { background: rgba(255,255,255,.17); border: 1px solid rgba(255,255,255,.32);
        padding: 5px 14px; border-radius: 999px; font-size: .85rem; font-weight: 600; }
.section-title { font-size: 1.05rem; font-weight: 700; margin: 18px 0 8px 0; letter-spacing: .01em; }
.panel { border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.07);
         border-radius: 18px; padding: 16px 20px; height: 100%; }
.pin { display: flex; align-items: center; gap: 14px; }
.pin-dot { width: 16px; height: 16px; border-radius: 50%; flex: none; box-shadow: 0 0 0 5px rgba(128,128,128,.18); }
.pin-label { font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; opacity: .65; }
.pin-text { font-size: 1.05rem; font-weight: 600; }
.hint { font-size: .98rem; line-height: 1.5; }
.algo { border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.07);
        border-radius: 20px; padding: 20px 24px; height: 100%; }
.algo.winner { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent); }
.algo-head { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.algo-dot { width: 14px; height: 14px; border-radius: 50%; background: var(--accent); flex: none; }
.algo-name { font-size: 1.35rem; font-weight: 800; }
.badge { margin-left: auto; background: var(--accent); color: #fff; font-size: .72rem; font-weight: 700;
         padding: 4px 12px; border-radius: 999px; letter-spacing: .04em; text-transform: uppercase; }
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 14px 18px; }
.stat-label { font-size: .78rem; opacity: .65; font-weight: 600; }
.stat-value { font-size: 1.2rem; font-weight: 700; line-height: 1.25; }
.bar { height: 10px; border-radius: 999px; background: rgba(128,128,128,.22); margin-top: 18px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--accent); border-radius: 999px; }
.bar-label { font-size: .75rem; opacity: .65; margin-top: 6px; }
.verdict { margin-top: 16px; border-radius: 16px; padding: 14px 20px; font-size: 1.02rem;
           background: rgba(33,160,100,.14); border: 1px solid rgba(33,160,100,.45); }
.notice { margin-top: 12px; border-radius: 14px; padding: 12px 18px; font-size: .95rem;
          background: rgba(240,170,40,.15); border: 1px solid rgba(240,170,40,.5); }
.stButton button { border-radius: 999px; font-weight: 600; border: 1px solid rgba(128,128,128,.4);
                     transition: transform .15s ease, border-color .15s ease; }
.stButton button:hover { transform: translateY(-2px); border-color: #d6336c; color: #d6336c; }
/* ---------- living background ---------- */
.stApp { isolation: isolate; }
[data-testid="stHeader"] { background: transparent; }
.stApp::before { content: ""; position: fixed; inset: -20%; z-index: -1; pointer-events: none;
  background:
    radial-gradient(circle at 18% 24%, rgba(214,51,108,.30) 0, transparent 36%),
    radial-gradient(circle at 82% 18%, rgba(107,75,216,.28) 0, transparent 38%),
    radial-gradient(circle at 72% 82%, rgba(240,138,93,.26) 0, transparent 36%),
    radial-gradient(circle at 22% 86%, rgba(47,158,68,.16) 0, transparent 34%);
  animation: blobDrift 26s ease-in-out infinite alternate; }
.stApp::after { content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
  background-image: radial-gradient(rgba(128,128,128,.30) 1.3px, transparent 1.4px);
  background-size: 36px 36px;
  -webkit-mask-image: radial-gradient(ellipse at center, #000 15%, transparent 78%);
  mask-image: radial-gradient(ellipse at center, #000 15%, transparent 78%);
  animation: gridMove 70s linear infinite; }
.route-bg { position: fixed; inset: 0; width: 100vw; height: 100vh; z-index: -1; pointer-events: none; opacity: .5; }
.route-glow { fill: none; stroke: rgba(214,51,108,.14); stroke-width: 12; stroke-linecap: round; }
.route-line { fill: none; stroke: #d6336c; stroke-width: 3; stroke-linecap: round; stroke-dasharray: 2 16;
  animation: flow 3s linear infinite; }
.route-line.alt { stroke: #6b4bd8; animation-duration: 4.5s; }
.route-pin { fill: #d6336c; transform-box: fill-box; transform-origin: center; animation: pinPulse 3.2s ease-in-out infinite; }
.route-pin.alt { fill: #6b4bd8; animation-delay: 1.2s; }

@keyframes blobDrift { 0% { transform: translate3d(0,0,0) scale(1); }
  50% { transform: translate3d(4%,-3%,0) scale(1.08) rotate(3deg); }
  100% { transform: translate3d(-4%,3%,0) scale(1.04) rotate(-3deg); } }
@keyframes gridMove { to { background-position: 360px 360px; } }
@keyframes flow { to { stroke-dashoffset: -36; } }
@keyframes pinPulse { 0%,100% { transform: scale(1); opacity: .9; } 50% { transform: scale(1.9); opacity: .3; } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: none; } }
@keyframes heroShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes growBar { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes pinRing { 0% { outline: 0 solid currentColor; outline-offset: 0; opacity: 1; } 70% { outline: 3px solid transparent; outline-offset: 10px; } 100% { outline: 0 solid transparent; outline-offset: 0; } }

.hero { background-size: 200% 200%; animation: heroShift 16s ease infinite, fadeUp .8s ease both; }
.chip { transition: transform .2s ease, background .2s ease; }
.chip:hover { transform: translateY(-2px); background: rgba(255,255,255,.28); }
.panel, .algo, .verdict, .notice { -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
  animation: fadeUp .6s ease both; }
.algo { transition: transform .25s ease, box-shadow .25s ease; }
.algo:hover { transform: translateY(-4px); }
[data-testid="stColumn"]:nth-child(2) .panel, [data-testid="stColumn"]:nth-child(2) .algo { animation-delay: .1s; }
[data-testid="stColumn"]:nth-child(3) .panel { animation-delay: .2s; }
.bar-fill { transform-origin: left; animation: growBar 1s cubic-bezier(.2,.7,.2,1) both; }
.pin-dot { animation: pinRing 2.4s ease-out infinite; }
.section-title { animation: fadeUp .5s ease both; }

/* ---------- liquid glass ---------- */
.stApp::before { background:
    radial-gradient(circle at 18% 24%, rgba(214,51,108,.42) 0, transparent 36%),
    radial-gradient(circle at 82% 18%, rgba(107,75,216,.40) 0, transparent 38%),
    radial-gradient(circle at 72% 82%, rgba(240,138,93,.38) 0, transparent 36%),
    radial-gradient(circle at 22% 86%, rgba(12,133,153,.28) 0, transparent 34%); }
.orb { position: fixed; z-index: -1; pointer-events: none; filter: blur(2px);
  animation: morph 14s ease-in-out infinite, orbFloat 22s ease-in-out infinite alternate; }
.orb.o1 { width: 380px; height: 380px; left: 6%; top: 52%;
  background: radial-gradient(circle at 30% 28%, rgba(255,255,255,.75), rgba(214,51,108,.55) 38%, rgba(214,51,108,.05) 72%); }
.orb.o2 { width: 460px; height: 460px; right: 4%; top: 8%; animation-delay: -5s, -8s;
  background: radial-gradient(circle at 30% 28%, rgba(255,255,255,.7), rgba(107,75,216,.52) 40%, rgba(107,75,216,.05) 72%); }
.orb.o3 { width: 320px; height: 320px; right: 26%; bottom: 4%; animation-delay: -9s, -3s;
  background: radial-gradient(circle at 30% 28%, rgba(255,255,255,.7), rgba(12,133,153,.5) 40%, rgba(12,133,153,.05) 72%); }
@keyframes morph {
  0%, 100% { border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; }
  33% { border-radius: 30% 60% 70% 40% / 50% 60% 30% 60%; }
  66% { border-radius: 50% 50% 40% 60% / 40% 50% 60% 50%; }
}
@keyframes orbFloat { from { transform: translate3d(0,0,0) rotate(0deg); } to { transform: translate3d(5vw,-6vh,0) rotate(40deg); } }

[data-testid="stSidebar"] { background: rgba(255,255,255,.12) !important;
  -webkit-backdrop-filter: blur(26px) saturate(180%); backdrop-filter: blur(26px) saturate(180%);
  border-right: 1px solid rgba(255,255,255,.35);
  box-shadow: inset -1px 0 0 rgba(255,255,255,.25), 8px 0 40px rgba(60,40,120,.12); }
[data-testid="stSidebarContent"] { background: transparent !important; }

.panel, .algo, .verdict, .notice, .chip {
  position: relative; overflow: hidden; isolation: isolate;
  background: linear-gradient(135deg, rgba(255,255,255,.30) 0%, rgba(255,255,255,.10) 55%, rgba(255,255,255,.18) 100%);
  -webkit-backdrop-filter: blur(22px) saturate(185%) brightness(1.04);
  backdrop-filter: blur(22px) saturate(185%) brightness(1.04);
  border: 1px solid rgba(255,255,255,.42);
  box-shadow:
    inset 0 1.5px 0 rgba(255,255,255,.75),
    inset 0 -1.5px 0 rgba(255,255,255,.12),
    inset 1.5px 0 0 rgba(255,255,255,.25),
    inset 0 0 28px rgba(255,255,255,.10),
    0 14px 44px rgba(50,30,110,.20); }
.panel::before, .algo::before, .verdict::before, .notice::before { content: ""; position: absolute; inset: 0; z-index: -1;
  border-radius: inherit; pointer-events: none;
  background: radial-gradient(120% 90% at 0% 0%, rgba(255,255,255,.38) 0%, rgba(255,255,255,0) 52%); }
.panel::after, .algo::after { content: ""; position: absolute; top: 0; bottom: 0; left: 0; width: 60%; z-index: -1;
  pointer-events: none; transform: translateX(-130%) skewX(-18deg);
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.38), transparent);
  transition: transform 1s cubic-bezier(.2,.7,.2,1); }
.panel:hover::after, .algo:hover::after { transform: translateX(260%) skewX(-18deg); }
.algo:hover { box-shadow:
    inset 0 1.5px 0 rgba(255,255,255,.85), inset 0 -1.5px 0 rgba(255,255,255,.15),
    inset 0 0 36px rgba(255,255,255,.14), 0 22px 56px rgba(50,30,110,.28); }
.algo.winner { border-color: rgba(255,255,255,.6);
  box-shadow: inset 0 1.5px 0 rgba(255,255,255,.85), inset 0 -1.5px 0 rgba(255,255,255,.15), inset 0 0 40px rgba(255,255,255,.12),
    0 0 0 1.5px var(--accent), 0 0 46px color-mix(in srgb, var(--accent) 50%, transparent), 0 18px 50px rgba(50,30,110,.24); }
.verdict { background: linear-gradient(135deg, rgba(60,200,130,.30), rgba(60,200,130,.10)); border-color: rgba(120,230,170,.55); }
.notice { background: linear-gradient(135deg, rgba(255,190,70,.32), rgba(255,190,70,.10)); border-color: rgba(255,210,120,.6); }

.hero { position: relative; overflow: hidden; isolation: isolate;
  background: linear-gradient(120deg, rgba(74,44,122,.88) 0%, rgba(176,58,111,.82) 55%, rgba(240,138,93,.85) 100%);
  background-size: 200% 200%;
  -webkit-backdrop-filter: blur(20px) saturate(170%); backdrop-filter: blur(20px) saturate(170%);
  border: 1px solid rgba(255,255,255,.45);
  box-shadow: inset 0 2px 0 rgba(255,255,255,.55), inset 0 -2px 0 rgba(255,255,255,.12), inset 0 0 50px rgba(255,255,255,.10),
    0 24px 70px rgba(90,40,120,.38); }
.hero::before { content: ""; position: absolute; inset: 0; z-index: -1; pointer-events: none; border-radius: inherit;
  background: radial-gradient(90% 120% at 8% -10%, rgba(255,255,255,.34) 0%, rgba(255,255,255,0) 55%); }
.chip { background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.5); overflow: visible;
  -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.6), 0 4px 14px rgba(0,0,0,.10); }

div.stButton button { background: linear-gradient(135deg, rgba(255,255,255,.34), rgba(255,255,255,.10)) !important;
  -webkit-backdrop-filter: blur(16px) saturate(180%); backdrop-filter: blur(16px) saturate(180%);
  border: 1px solid rgba(255,255,255,.55) !important;
  box-shadow: inset 0 1.5px 0 rgba(255,255,255,.75), inset 0 -1px 0 rgba(255,255,255,.12), 0 8px 24px rgba(50,30,110,.16);
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
div.stButton button:hover { transform: translateY(-3px); border-color: rgba(255,255,255,.9) !important;
  box-shadow: inset 0 1.5px 0 rgba(255,255,255,.9), inset 0 0 22px rgba(255,255,255,.22), 0 14px 32px rgba(214,51,108,.30); }
div.stButton button:active { transform: translateY(0) scale(.98); }

[data-testid="stSidebar"] [data-testid="stButtonGroup"] button { -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
  border-color: rgba(255,255,255,.45); }
[data-testid="stExpander"] details { background: linear-gradient(135deg, rgba(255,255,255,.28), rgba(255,255,255,.08));
  -webkit-backdrop-filter: blur(20px) saturate(180%); backdrop-filter: blur(20px) saturate(180%);
  border: 1px solid rgba(255,255,255,.42) !important; border-radius: 20px;
  box-shadow: inset 0 1.5px 0 rgba(255,255,255,.7), 0 12px 36px rgba(50,30,110,.16); }
iframe[title="streamlit_folium.st_folium"] { border-radius: 26px; border: 1px solid rgba(255,255,255,.55);
  box-shadow: 0 0 0 6px rgba(255,255,255,.14), 0 24px 60px rgba(50,30,110,.30); }

.algo-dot, .pin-dot { position: relative; box-shadow: inset -2px -3px 6px rgba(0,0,0,.20), inset 2px 2px 4px rgba(255,255,255,.55); }
.algo-dot::after, .pin-dot::after { content: ""; position: absolute; top: 2px; left: 3px; width: 35%; height: 30%;
  border-radius: 50%; background: rgba(255,255,255,.8); filter: blur(.6px); }
.bar { background: rgba(255,255,255,.22); box-shadow: inset 0 1px 3px rgba(0,0,0,.18); }
.bar-fill { background-image: linear-gradient(180deg, rgba(255,255,255,.55) 0%, rgba(255,255,255,0) 60%);
  background-color: var(--accent); box-shadow: 0 0 14px color-mix(in srgb, var(--accent) 60%, transparent); }
.badge { box-shadow: inset 0 1px 0 rgba(255,255,255,.55), 0 4px 12px color-mix(in srgb, var(--accent) 45%, transparent); }

/* selected option in the sidebar toggles: solid glass pill with white text so it stays readable */
button[aria-checked="true"], [data-testid="stBaseButton-segmented_controlActive"] {
  background: linear-gradient(135deg, rgba(214,51,108,.95), rgba(176,58,111,.85)) !important;
  color: #fff !important; border-color: rgba(255,255,255,.65) !important; font-weight: 700;
  box-shadow: inset 0 1.5px 0 rgba(255,255,255,.55), 0 6px 18px rgba(214,51,108,.35); }
button[aria-checked="true"] *, [data-testid="stBaseButton-segmented_controlActive"] * { color: #fff !important; }
button[aria-checked="false"], [data-testid="stBaseButton-segmented_control"] {
  background: rgba(255,255,255,.16) !important; }

/* ---------- title ---------- */
.hero .eyebrow { font-size: .8rem; font-weight: 700; letter-spacing: .32em; text-transform: uppercase; opacity: .85; margin-bottom: 4px; }
.hero h1 { font-family: 'DM Serif Display', Georgia, 'Times New Roman', serif; font-weight: 400; font-size: 4.6rem;
  line-height: 1; letter-spacing: -.02em; margin: 0 0 10px 0; text-shadow: 0 4px 30px rgba(0,0,0,.22); }
@media (max-width: 640px) { .hero h1 { font-size: 3.2rem; } }

/* ---------- route option cards ---------- */
.route-card { --accent: #1971c2; padding: 14px 20px; }
.route-card .route-swatch { width: 46px; height: 7px; border-radius: 999px; background: var(--accent); margin-bottom: 10px;
  box-shadow: 0 0 14px color-mix(in srgb, var(--accent) 65%, transparent); }
.route-card .route-label { font-size: .78rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; opacity: .72; }
.route-card .route-time { font-size: 1.55rem; font-weight: 800; line-height: 1.25; }
.route-card .route-sub { font-size: .92rem; opacity: .82; }
.route-card.selected { border-color: rgba(255,255,255,.65);
  box-shadow: inset 0 1.5px 0 rgba(255,255,255,.85), inset 0 0 34px rgba(255,255,255,.12),
    0 0 0 2px var(--accent), 0 0 40px color-mix(in srgb, var(--accent) 45%, transparent), 0 16px 44px rgba(50,30,110,.22); }

/* ---------- search inputs ---------- */
[data-baseweb="input"], [data-baseweb="base-input"], [data-testid="stTextInputRootElement"], [data-baseweb="select"] > div { border-radius: 999px !important;
  background: rgba(255,255,255,.18) !important; border: 1px solid rgba(255,255,255,.5) !important;
  -webkit-backdrop-filter: blur(14px); backdrop-filter: blur(14px);
  box-shadow: inset 0 1.5px 0 rgba(255,255,255,.6), 0 6px 20px rgba(50,30,110,.12); }
[data-baseweb="input"] input, [data-baseweb="base-input"] input { padding-left: 18px !important; background: transparent !important; }
[data-testid="stForm"] { padding: 0 !important; border: none !important; }
[data-testid="stFormSubmitButton"] button { border-radius: 999px; font-weight: 600;
  background: linear-gradient(135deg, rgba(255,255,255,.34), rgba(255,255,255,.10)) !important;
  -webkit-backdrop-filter: blur(16px) saturate(180%); backdrop-filter: blur(16px) saturate(180%);
  border: 1px solid rgba(255,255,255,.55) !important;
  box-shadow: inset 0 1.5px 0 rgba(255,255,255,.75), inset 0 -1px 0 rgba(255,255,255,.12), 0 8px 24px rgba(50,30,110,.16);
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-3px); border-color: rgba(255,255,255,.9) !important;
  box-shadow: inset 0 1.5px 0 rgba(255,255,255,.9), inset 0 0 22px rgba(255,255,255,.22), 0 14px 32px rgba(214,51,108,.30); }
.pin > div:last-child { min-width: 0; }
.pin-text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pin-sub { font-size: .8rem; opacity: .7; }

/* ---------- meet in the middle ---------- */
.meet-spot { --accent: #d6336c; padding: 20px 26px; }
.meet-spot .spot-label { font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; opacity: .7; }
.meet-spot .spot-name { font-size: 1.7rem; font-weight: 800; line-height: 1.2; margin: 4px 0 6px 0; }
.meet-spot .spot-sub { font-size: .98rem; opacity: .85; }
.meet-row { display: grid; grid-template-columns: 22px minmax(0, 1.3fr) 118px 84px minmax(70px, 1.2fr);
  gap: 12px; align-items: center; padding: 9px 0; border-bottom: 1px solid rgba(128,128,128,.2); }
.meet-row:last-child { border-bottom: none; }
.meet-row.head { font-size: .74rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; opacity: .65; padding-top: 0; }
.meet-row .pin-dot { animation: none; width: 14px; height: 14px; }
.meet-row .who { font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.meet-row .num { font-weight: 700; font-variant-numeric: tabular-nums; }
.meet-row .bar { margin-top: 0; height: 8px; }
.meet-row.longest .who::after { content: "longest"; margin-left: 8px; font-size: .68rem; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; padding: 2px 8px; border-radius: 999px; background: rgba(214,51,108,.85); color: #fff; }
@media (max-width: 760px) { .meet-row { grid-template-columns: 20px minmax(0, 1fr) 100px 70px; } .meet-row .bar { display: none; } }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 4px 0 10px 0; font-size: .85rem; }
.legend .item { display: inline-flex; align-items: center; gap: 7px; }
.legend i { width: 14px; height: 14px; border-radius: 4px; display: inline-block; border: 1px solid rgba(255,255,255,.8); }
.person-chip { display: flex; align-items: center; gap: 10px; padding: 8px 4px; min-height: 40px; }
.person-chip .pin-dot { animation: none; width: 14px; height: 14px; }
.person-chip .who { font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.person-chip .where { font-size: .78rem; opacity: .65; }
[data-testid="stTabs"] [role="tab"] { font-weight: 600; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
<div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div>
<svg class="route-bg" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
<path class="route-glow" d="M -50 700 C 250 620, 330 420, 620 470 S 980 760, 1200 520 S 1500 220, 1680 260"/>
<path class="route-line" d="M -50 700 C 250 620, 330 420, 620 470 S 980 760, 1200 520 S 1500 220, 1680 260"/>
<path class="route-glow" d="M -50 180 C 200 260, 420 120, 640 230 S 1000 380, 1220 250 S 1500 400, 1680 640"/>
<path class="route-line alt" d="M -50 180 C 200 260, 420 120, 640 230 S 1000 380, 1220 250 S 1500 400, 1680 640"/>
<circle class="route-pin" cx="620" cy="470" r="7"/>
<circle class="route-pin" cx="1200" cy="520" r="7"/>
<circle class="route-pin alt" cx="640" cy="230" r="7"/>
<circle class="route-pin alt" cx="1220" cy="250" r="7"/>
</svg>
""",
    unsafe_allow_html=True,
)

if not map_exists():
    st.error("No map found. Run `python load_map.py` first, then restart this app.")
    st.stop()


# ---------- data ----------
@st.cache_resource(show_spinner="Loading the map of Bengaluru. The first load can take a little while...")
def load_data():
    G = load_graph()
    ids = list(G.nodes)
    lats = np.array([G.nodes[n]["y"] for n in ids])
    lngs = np.array([G.nodes[n]["x"] for n in ids])
    max_speed = max(d["speed_kph"] for _, _, d in G.edges(data=True)) / 3.6  # m/s
    index = {n: i for i, n in enumerate(ids)}  # node id -> position in ids / lats / lngs
    return G, ids, lats, lngs, max_speed, index


G, IDS, LATS, LNGS, MAX_SPEED, INDEX = load_data()
BOUNDS = [[float(LATS.min()), float(LNGS.min())], [float(LATS.max()), float(LNGS.max())]]


# ---------- helpers ----------
def nearest_node(lat, lng):
    """Closest intersection to a point, and how many metres away it is."""
    d2 = (LATS - lat) ** 2 + ((LNGS - lng) * math.cos(math.radians(lat))) ** 2
    i = int(d2.argmin())
    return IDS[i], haversine(lat, lng, float(LATS[i]), float(LNGS[i]))


@st.cache_data(ttl=3600, show_spinner="Searching Bengaluru...")
def geocode(query):
    """Look up a place name in Bengaluru. Returns a list of {label, name, lat, lon}."""
    resp = requests.get(
        GEOCODER_URL,
        params={"q": query, "format": "jsonv2", "limit": 6, "viewbox": VIEWBOX, "bounded": 1, "countrycodes": "in"},
        headers=USER_AGENT,
        timeout=10,
    )
    resp.raise_for_status()
    places = []
    for item in resp.json():
        parts = item.get("display_name", "").split(", ")
        places.append({"label": ", ".join(parts[:3]) or query, "name": parts[0] or query,
                       "lat": float(item["lat"]), "lon": float(item["lon"])})
    return places


@st.cache_data(ttl=3600, show_spinner=False, max_entries=200)
def reverse_geocode(lat, lon):
    """Name of the area around a point, e.g. 'Koramangala 5th Block, Bengaluru'. Raises if the service is down."""
    url = GEOCODER_URL.rsplit("/", 1)[0] + "/reverse"
    resp = requests.get(url, params={"lat": round(lat, 5), "lon": round(lon, 5), "format": "jsonv2", "zoom": 16},
                        headers=USER_AGENT, timeout=8)
    resp.raise_for_status()
    parts = (resp.json().get("display_name") or "").split(", ")
    return ", ".join(parts[:3])


def node_latlng(n):
    return (G.nodes[n]["y"], G.nodes[n]["x"])


ALGORITHMS = ("Dijkstra", "A*", "Bidirectional")


@st.cache_data(show_spinner="Searching for the best route...", max_entries=30)
def search_all(start, end, weight):
    out = {}
    for name in ALGORITHMS:
        explored, explored_back = [], []
        t0 = time.perf_counter()
        if name == "Dijkstra":
            path, cost, visited = dijkstra(G, start, end, weight=weight, explored=explored)
        elif name == "A*":
            path, cost, visited = astar(G, start, end, weight=weight,
                                        max_speed=MAX_SPEED, explored=explored)
        else:
            path, cost, visited = bidirectional_dijkstra(G, start, end, weight=weight,
                                                         explored=explored, explored_back=explored_back)
        ms = (time.perf_counter() - t0) * 1000
        out[name] = {"path": path, "cost": cost, "visited": visited, "ms": ms,
                     "explored": explored, "explored_back": explored_back}
    return out


@st.cache_data(show_spinner="Finding alternative routes...", max_entries=30)
def search_alternatives(start, end, weight, base_path):
    max_speed = MAX_SPEED if weight == "travel_time" else None
    return alternative_routes(G, start, end, list(base_path), weight=weight, max_speed=max_speed)


@st.cache_resource(show_spinner="Preparing the road network for group searches...")
def get_adjacency(weight):
    return build_adjacency(G, weight)


@st.cache_resource(max_entries=80, show_spinner=False)
def person_costs(node, weight):
    """Cost for someone standing at `node` to reach EVERY intersection: one full Dijkstra search,
    stored as an array in the same order as IDS. Cached, so adding one person to a group
    only costs one new search."""
    dist, _ = all_distances(get_adjacency(weight), node)
    costs = np.full(len(IDS), np.inf)
    for n, d in dist.items():
        costs[INDEX[n]] = d
    return costs


@st.cache_data(show_spinner="Searching outward from everyone's location...", max_entries=40)
def meet(nodes, weight):
    """Best meeting place for the people standing at `nodes`, for both goals."""
    costs = np.vstack([person_costs(n, weight) for n in nodes])
    out = {}
    for goal in ("fair", "total"):
        k, per_person = meeting_point(costs, goal)
        spot = IDS[k]
        paths = [bidirectional_dijkstra(G, n, spot, weight=weight)[0] for n in nodes]
        out[goal] = {"node": spot, "costs": per_person, "paths": paths}
    return out


def route_totals(path, weight):
    """Total driving time (s) and distance (m) along a path."""
    seconds = metres = 0.0
    for u, v in zip(path, path[1:]):
        edge = min(G[u][v].values(), key=lambda d: d[weight])
        seconds += edge["travel_time"]
        metres += edge["length"]
    return seconds, metres


def blend(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def fmt_time(seconds):
    m, s = divmod(round(seconds), 60)
    return f"{m} min {s:02d} s"


def fmt_delta(seconds):
    sign = "+" if seconds >= 0 else "-"
    secs = abs(round(seconds))
    return sign + (f"{secs} s" if secs < 60 else fmt_time(secs))


def available_presets():
    out = []
    for label, a, b in PRESETS:
        na, da = nearest_node(*a)
        nb, db = nearest_node(*b)
        if da < 600 and db < 600 and na != nb:
            out.append((label, na, nb))
    return out


PRESETS_OK = available_presets()

# ---------- state ----------
ss = st.session_state
if "start" not in ss:
    ss.start, ss.end = (PRESETS_OK[0][1], PRESETS_OK[0][2]) if PRESETS_OK else (None, None)
    ss.start_name, ss.end_name = PRESETS_OK[0][0].split(" → ") if PRESETS_OK else (None, None)
    ss.last_click = None
    ss.notice = ""
    for side in ("from", "to"):
        ss[f"res_{side}"], ss[f"err_{side}"] = [], ""


def set_trip(a, b, names=(None, None)):
    ss.start, ss.end, ss.notice = a, b, ""
    ss.start_name, ss.end_name = names


def random_trip():
    ss.start, ss.end = random.sample(IDS, 2)
    ss.start_name = ss.end_name = None
    ss.notice = ""


def swap_trip():
    ss.start, ss.end = ss.end, ss.start
    ss.start_name, ss.end_name = ss.end_name, ss.start_name


def clear_trip():
    ss.start, ss.end, ss.notice = None, None, ""
    ss.start_name = ss.end_name = None
    for side in ("from", "to"):
        ss[f"res_{side}"], ss[f"err_{side}"] = [], ""


def apply_place(side, place):
    """Use a search result as the start ("from") or end ("to") point."""
    node, off_by = nearest_node(place["lat"], place["lon"])
    if side == "from":
        ss.start, ss.start_name = node, place["name"]
    else:
        ss.end, ss.end_name = node, place["name"]
    if off_by > 600:
        ss.notice = (f"{html.escape(place['name'])} is {off_by / 1000:.1f} km outside the map you downloaded, so the "
                     "nearest intersection was used. To cover more of Bengaluru, set MAP_SIZE in load_map.py and run it again.")
    else:
        ss.notice = ""


def do_search(side):
    query = (ss.get(f"q_{side}") or "").strip()
    ss[f"res_{side}"], ss[f"err_{side}"] = [], ""
    if not query:
        return
    try:
        found = geocode(query)
    except Exception:
        ss[f"err_{side}"] = "Could not reach the place search. Check your internet connection and try again."
        return
    if not found:
        ss[f"err_{side}"] = f"No places found for \u201c{query}\u201d in Bengaluru. Try a nearby landmark or area name."
        return
    ss[f"res_{side}"] = found
    ss[f"pick_{side}"] = 0
    apply_place(side, found[0])  # go to the top match straight away, like most map apps


def pick_place(side):
    apply_place(side, ss[f"res_{side}"][ss[f"pick_{side}"]])


def tiles_for(map_style):
    if map_style == "Satellite":
        return dict(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, Maxar, Earthstar Geographics",
        )
    return dict(tiles="OpenStreetMap")


# ---------- meet in the middle: state and actions ----------
if "people" not in ss:
    ss.people = []  # [{"id", "name", "node", "colour"}]
    ss.m_counter = 0
    ss.m_notice = ""
    ss.m_res, ss.m_err = [], ""
    ss.m_last_click = None
    ss.m_last_geo = None


def add_person(default_name, lat, lon):
    """Put a person on the nearest intersection. The Name box wins over default_name. Returns True if added."""
    typed = " ".join((ss.get("m_name") or "").split())
    name = typed or default_name
    if len(ss.people) >= MAX_PEOPLE:
        ss.m_notice = f"The group is full ({MAX_PEOPLE} people at most, because every person needs one full search of the city)."
        return False
    node, off_by = nearest_node(lat, lon)
    if off_by > OFF_MAP_LIMIT:
        who = html.escape(name) if name else "That spot"
        ss.m_notice = (f"{who} is {off_by / 1000:.1f} km from the nearest road on the map you downloaded, so it was not added. "
                       "Pick a spot inside the map, or set MAP_SIZE in load_map.py to cover more of Bengaluru.")
        return False
    ss.m_counter += 1
    ss.people.append({"id": ss.m_counter, "name": name or f"Person {ss.m_counter}", "node": node,
                      "colour": PERSON_COLOURS[(ss.m_counter - 1) % len(PERSON_COLOURS)]})
    if typed:
        ss.m_clear_name = True  # empty the Name box on the next run, ready for the next person
    ss.m_notice = ""
    if off_by > 600:
        ss.m_notice = (f"{html.escape(ss.people[-1]['name'])} is {off_by / 1000:.1f} km from the nearest road on the map, "
                       "so the closest intersection was used.")
    return True


def m_search():
    query = (ss.get("m_q") or "").strip()
    ss.m_res, ss.m_err = [], ""
    if not query:
        return
    try:
        found = geocode(query)
    except Exception:
        ss.m_err = "Could not reach the place search. Check your internet connection and try again."
        return
    if not found:
        ss.m_err = f"No places found for “{query}” in Bengaluru. Try a nearby landmark or area name."
        return
    ss.m_res = found
    ss.m_pick = 0


def m_add_search():
    place = ss.m_res[ss.m_pick]
    if add_person(place["name"], place["lat"], place["lon"]):
        ss.m_res, ss.m_err, ss.m_q = [], "", ""


def m_add_coords():
    text = ss.get("m_coords") or ""
    pin = re.search(r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)", text)  # the exact pin inside a long Google Maps link
    numbers = [float(pin.group(1)), float(pin.group(2))] if pin else [float(x) for x in re.findall(r"-?\d+\.\d+", text)]
    if len(numbers) < 2:
        ss.m_notice = "Could not find two coordinates. Paste them like 12.9716, 77.5946 (latitude first)."
        return
    lat, lon = numbers[0], numbers[1]
    if abs(lat) > 60 and abs(lon) <= 60:  # pasted longitude first: swap
        lat, lon = lon, lat
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        ss.m_notice = "Those numbers are not valid coordinates. Paste them like 12.9716, 77.5946 (latitude first)."
        return
    if add_person("", lat, lon):
        ss.m_coords = ""


def m_remove(pid):
    ss.people = [p for p in ss.people if p["id"] != pid]
    ss.m_notice = ""


def m_clear():
    ss.people, ss.m_notice = [], ""
    ss.m_res, ss.m_err = [], ""


def m_example():
    ss.people, ss.m_counter, ss.m_notice = [], 0, ""
    for name, lat, lon in EXAMPLE_GROUP:
        add_person(name, lat, lon)


def fmt_cost(cost):
    """A route cost in the units of the chosen route type."""
    return fmt_time(cost) if weight == "travel_time" else f"{cost / 1000:.2f} km"


def meeting_page():
    if ss.pop("m_clear_name", False):
        ss.m_name = ""

    st.markdown('<div class="section-title">Meet in the middle</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel hint">Add two or more people, as many as you like. Wayfinder searches outward from every '
                'person at once and finds the intersection that works best for the whole group, using the same '
                'road network and algorithms as the route lab.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Add people</div>', unsafe_allow_html=True)
    st.text_input("Name (optional)", key="m_name", placeholder="e.g. Priya. Leave empty to use the place name.", max_chars=28)
    tab_search, tab_coords, tab_geo = st.tabs(["Search a place", "Paste coordinates", "My current location"])
    with tab_search:
        with st.form("m_form", border=False):
            st.text_input("Place in Bengaluru", key="m_q", placeholder="e.g. Koramangala 5th Block, Indiranagar Metro")
            st.form_submit_button("Search", on_click=m_search, use_container_width=True)
        if ss.m_res:
            st.selectbox("Matches (pick the right one)", options=list(range(len(ss.m_res))),
                         format_func=lambda i: ss.m_res[i]["label"], key="m_pick")
            st.button("Add to the group", on_click=m_add_search, use_container_width=True, key="m_add_search_btn")
        if ss.m_err:
            st.caption(ss.m_err)
    with tab_coords:
        st.caption("A friend can send you their spot: in Google Maps they press and hold on the place, then copy the two "
                   "numbers shown at the top (like 12.9716, 77.5946). Paste them here.")
        with st.form("m_coord_form", border=False):
            st.text_input("Latitude, longitude", key="m_coords", placeholder="12.9716, 77.5946")
            st.form_submit_button("Add by coordinates", on_click=m_add_coords, use_container_width=True)
    with tab_geo:
        st.caption("Adds the place this device is at right now. Your browser asks for permission first. Your location "
                   "is only used inside this app and is not stored anywhere.")
        if geo_button is None:
            st.caption("This needs the geo_button folder next to app.py. Copy it from the zip and restart the app.")
        else:
            loc = geo_button(label="\U0001F4CD Add my current location", key="m_geo", default=None)
            if loc and loc.get("ts") != ss.m_last_geo:
                ss.m_last_geo = loc.get("ts")
                if add_person("Me", loc["lat"], loc["lon"]) and loc.get("accuracy", 0) > 1500:
                    ss.m_notice = (f"Your device only knows your location to about {loc['accuracy'] / 1000:.1f} km, so the "
                                   "spot may be off. Add the exact place by search if it matters.")
                st.rerun()
    if ss.m_notice:
        st.markdown(f'<div class="notice">{ss.m_notice}</div>', unsafe_allow_html=True)

    # ----- the group -----
    people = ss.people
    st.markdown(f'<div class="section-title">The group ({len(people)})</div>', unsafe_allow_html=True)
    if not people:
        st.markdown('<div class="panel hint">Nobody added yet. Search for a place, paste coordinates, use your current '
                    'location, or click the map below.</div>', unsafe_allow_html=True)
    for row in range(0, len(people), 3):
        for col, p in zip(st.columns(3), people[row:row + 3]):
            with col:
                chip, remove = st.columns([6, 1], vertical_alignment="center")
                lat, lon = node_latlng(p["node"])
                chip.markdown(
                    f'<div class="person-chip"><span class="pin-dot" style="background:{p["colour"]};color:{p["colour"]}"></span>'
                    f'<div style="min-width:0"><div class="who">{html.escape(p["name"])}</div>'
                    f'<div class="where">{lat:.4f}, {lon:.4f}</div></div></div>', unsafe_allow_html=True)
                remove.button("✕", key=f"rm_{p['id']}", on_click=m_remove, args=(p["id"],), help=f"Remove {p['name']}")
    buttons = st.columns(4)
    buttons[0].button("Try an example group", on_click=m_example, use_container_width=True, key="m_example_btn")
    buttons[1].button("Clear group", on_click=m_clear, use_container_width=True, key="m_clear_btn", disabled=not people)

    # ----- the answer -----
    result = chosen = None
    goal = "fair"
    if len(people) >= 2:
        goal_label = st.segmented_control(
            "What should the meeting spot optimise?", ["Fairest", "Lowest total"], default="Fairest", key="m_goal",
            help="Fairest: the longest trip anyone makes is as short as possible. "
                 "Lowest total: everyone's trips added together are as short as possible.") or "Fairest"
        goal = "fair" if goal_label == "Fairest" else "total"
        result = meet(tuple(p["node"] for p in people), weight)
        chosen = result[goal]

        spot_lat, spot_lon = node_latlng(chosen["node"])
        try:
            place = reverse_geocode(spot_lat, spot_lon)
        except Exception:
            place = ""
        title = f"Near {html.escape(place)}" if place else f"{spot_lat:.4f}, {spot_lon:.4f}"
        worst, total = max(chosen["costs"]), sum(chosen["costs"])
        accent = FAIR_COLOUR if goal == "fair" else TOTAL_COLOUR
        st.markdown(
            f'<div class="panel meet-spot" style="--accent:{accent}"><div class="spot-label">Best meeting spot &middot; '
            f'{"Fairest" if goal == "fair" else "Lowest total"}</div><div class="spot-name">{title}</div>'
            f'<div class="spot-sub">Longest trip <b>{fmt_cost(worst)}</b> &middot; all trips together '
            f'<b>{fmt_cost(total)}</b> &middot; average <b>{fmt_cost(total / len(people))}</b></div></div>',
            unsafe_allow_html=True)

        # each person's trip
        longest = max(chosen["costs"])
        rows = ['<div class="meet-row head"><span></span><span>Person</span><span>Travel time</span>'
                '<span>Distance</span><span>Compared with the longest</span></div>']
        for p, path, cost in zip(people, chosen["paths"], chosen["costs"]):
            secs, metres = route_totals(path, weight)
            share = 100 * cost / longest if longest > 0 else 0
            cls = "meet-row longest" if longest > 0 and cost >= longest - 1e-9 else "meet-row"
            rows.append(
                f'<div class="{cls}"><span class="pin-dot" style="background:{p["colour"]};color:{p["colour"]}"></span>'
                f'<span class="who">{html.escape(p["name"])}</span><span class="num">{fmt_time(secs)}</span>'
                f'<span class="num">{metres / 1000:.1f} km</span>'
                f'<div class="bar" style="--accent:{p["colour"]}"><div class="bar-fill" style="width:{share:.0f}%"></div></div></div>')
        st.markdown('<div class="section-title">Everyone\'s trip</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="panel">{"".join(rows)}</div>', unsafe_allow_html=True)

        # the two goals side by side
        st.markdown('<div class="section-title">Fairest vs lowest total</div>', unsafe_allow_html=True)
        for col, g, label, colour in zip(st.columns(2), ("fair", "total"), ("Fairest", "Lowest total"),
                                         (FAIR_COLOUR, TOTAL_COLOUR)):
            costs = result[g]["costs"]
            cls = "panel route-card selected" if g == goal else "panel route-card"
            col.markdown(
                f'<div class="{cls}" style="--accent:{colour}"><div class="route-swatch"></div>'
                f'<div class="route-label">{label}</div><div class="route-time">{fmt_cost(max(costs))}</div>'
                f'<div class="route-sub">longest trip &middot; total {fmt_cost(sum(costs))}</div></div>',
                unsafe_allow_html=True)
        if result["fair"]["node"] == result["total"]["node"]:
            st.caption("For this group both goals choose the same spot.")
    elif people:
        st.markdown('<div class="notice">Add at least one more person to find a meeting spot.</div>', unsafe_allow_html=True)

    # ----- the map -----
    st.markdown('<div class="section-title">Map</div>', unsafe_allow_html=True)
    click_mode = st.toggle("Add a person with each map click", value=True, key="m_click_mode")
    tile_args = tiles_for(style)
    if result:
        fmap = folium.Map(zoom_start=13, **tile_args)
        fmap.fit_bounds([node_latlng(n) for path in chosen["paths"] for n in path], padding=(40, 40))
    elif len(people) == 1:
        fmap = folium.Map(location=node_latlng(people[0]["node"]), zoom_start=14, **tile_args)
    else:
        fmap = folium.Map(zoom_start=11, **tile_args)
        fmap.fit_bounds(BOUNDS)

    if result:
        for p, path in zip(people, chosen["paths"]):
            folium.PolyLine([node_latlng(n) for n in path], weight=5, color=p["colour"], opacity=0.85,
                            tooltip=f"{p['name']}'s route").add_to(fmap)
    for i, p in enumerate(people):
        circle = (f'<div style="width:30px;height:30px;border-radius:50%;background:{p["colour"]};color:#fff;font:800 14px sans-serif;'
                  f'display:flex;align-items:center;justify-content:center;border:3px solid #fff;box-shadow:0 3px 10px rgba(0,0,0,.45)">{i + 1}</div>')
        folium.Marker(node_latlng(p["node"]), tooltip=p["name"],
                      icon=folium.DivIcon(html=circle, icon_size=(30, 30), icon_anchor=(15, 15))).add_to(fmap)
    if result:
        star = ('<div style="width:40px;height:40px;border-radius:50%;background:#f59f00;color:#fff;font:700 22px sans-serif;'
                'display:flex;align-items:center;justify-content:center;border:3px solid #fff;box-shadow:0 4px 14px rgba(0,0,0,.55)">★</div>')
        folium.Marker(node_latlng(chosen["node"]), tooltip="Meeting spot", z_index_offset=1000,
                      icon=folium.DivIcon(html=star, icon_size=(40, 40), icon_anchor=(20, 20))).add_to(fmap)

    out = st_folium(fmap, height=620, use_container_width=True, returned_objects=["last_clicked"], key="meet_map")
    st.caption("Numbered circles are people and the gold star is the meeting spot. Each coloured line is that person's "
               "route to it. " + ("Click the map to add a person there." if click_mode else "Map clicks are off."))

    click = out.get("last_clicked") if out else None
    if click and click != ss.m_last_click:
        ss.m_last_click = click
        if click_mode:
            add_person("", click["lat"], click["lng"])
            st.rerun()

    with st.expander("How does this work?"):
        st.markdown(
            """
- Every person is snapped to the nearest intersection of the road graph.
- From each person, Wayfinder runs **Dijkstra with no destination**: it keeps going until it knows the cheapest travel cost from that person to *every* intersection in the city. That is one full search per person.
- Putting those results side by side gives, for every intersection, how long each person would need to get there. Then it is just a choice of goal:
  - **Fairest** picks the intersection where the *longest* trip is smallest (in graph theory, the *centre* of the group).
  - **Lowest total** picks the intersection where the *sum* of all trips is smallest (the *median*).
- If two intersections tie, the other goal breaks the tie.
- The routes drawn are the cheapest road routes from each person to the chosen spot (found with bidirectional Dijkstra).
- Travel is one-way, from each person to the meeting spot, so one-way streets and slow roads are respected in each direction.
- The spot is an intersection, not a venue. Use it as the area to look for a café or restaurant.
- Adding one more person only needs one new search, because the earlier ones are remembered.
"""
        )

# ---------- hero ----------
st.markdown(
    f"""
<div class="hero">
<div class="eyebrow">A route lab for Bengaluru</div>
<h1>Wayfinder</h1>
<p>Watch three shortest-path algorithms race across real Bengaluru roads, and see how much less work the smarter ones do to find the same route.</p>
<div class="chips">
<span class="chip">{len(G.nodes):,} intersections</span>
<span class="chip">{len(G.edges):,} road segments</span>
<span class="chip">Dijkstra</span>
<span class="chip">A*</span>
<span class="chip">Bidirectional</span>
<span class="chip">Meet in the middle</span>
<span class="chip">Access map</span>
<span class="chip">Neighbourhood score</span>
</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- amenity access: nearest place of any kind, and the neighbourhood score ----------
PLACES_DIR = "places"
OLD_HOSPITAL_FILE = "hospitals.csv"   # made by load_hospitals.py; the app still uses it for hospitals
PLACE_OFF_MAP = 1000                  # metres: a place further than this from any road on the map is ignored
MAX_CELLS = 2500                      # the coloured patches on the maps
CATS = {                              # the kinds of place the app knows about, in the order they are listed
    "hospital": {"one": "hospital", "many": "hospitals", "emoji": "🏥", "colour": "#e03131"},
    "pharmacy": {"one": "pharmacy", "many": "pharmacies", "emoji": "💊", "colour": "#0ca678"},
    "fire_station": {"one": "fire station", "many": "fire stations", "emoji": "🚒", "colour": "#f76707"},
    "police": {"one": "police station", "many": "police stations", "emoji": "🚓", "colour": "#1c7ed6"},
    "station": {"one": "metro or railway station", "many": "metro and railway stations", "emoji": "🚇", "colour": "#7048e8"},
    "school": {"one": "school", "many": "schools", "emoji": "🏫", "colour": "#f59f00"},
    "park": {"one": "park", "many": "parks", "emoji": "🌳", "colour": "#5c940d"},
    "supermarket": {"one": "supermarket", "many": "supermarkets", "emoji": "🛒", "colour": "#d6336c"},
}
TRAVEL = {"Driving": "travel_time", "Walking": "walk"}
TIME_BANDS = [(300, "Under 5 min"), (600, "5 to 10 min"), (900, "10 to 15 min"),
              (1200, "15 to 20 min"), (1800, "20 to 30 min"), (math.inf, "Over 30 min")]
GOOD = {"travel_time": 600, "walk": 900}   # what counts as "close" for the headline figure (seconds)
BAND_COLOURS = ["#2f9e44", "#94d82d", "#fcc419", "#ff922b", "#f03e3e", "#862e9c"]
SCORE_COLOURS = ["#862e9c", "#f03e3e", "#ff922b", "#fcc419", "#94d82d", "#2f9e44"]
SCORE_LABELS = ["Under 20%", "20 to 40%", "40 to 60%", "60 to 80%", "80 to 95%", "95 to 100%"]

for _key, _default in (("h_point", None), ("h_last_click", None), ("h_suggest", False), ("h_notice", ""),
                       ("s_point", None), ("s_last_click", None), ("s_notice", ""), ("s_names", {}), ("s_spots", [])):
    if _key not in ss:
        ss[_key] = _default


@st.cache_resource(show_spinner=False)
def get_reverse_adjacency(mode):
    """Roads for 'how long does it take to get TO the place': flipped for driving (one-way streets), both ways for walking."""
    return build_walk_adjacency(G) if mode == "walk" else build_reverse_adjacency(G, mode)


def place_stamp(key):
    """(file, change time) of the places file for a kind, or None if it has not been downloaded."""
    path = os.path.join(PLACES_DIR, f"{key}.csv")
    if not os.path.exists(path) and key == "hospital" and os.path.exists(OLD_HOSPITAL_FILE):
        path = OLD_HOSPITAL_FILE
    return (path, os.path.getmtime(path)) if os.path.exists(path) else None


@st.cache_resource(show_spinner=False)
def category_sites(key, stamp):
    """The places of one kind, each moved to its nearest intersection. Returns (list, number outside the map).
    `stamp` changes when the file does, so a new download is picked up automatically."""
    if stamp is None:
        return None, 0
    placed, outside = [], 0
    with open(stamp[0], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                lat, lon = float(row["lat"]), float(row["lon"])
            except (KeyError, ValueError):
                continue
            node, off_by = nearest_node(lat, lon)
            if off_by > PLACE_OFF_MAP:
                outside += 1
                continue
            placed.append({"name": row.get("name") or f"Unnamed {CATS[key]['one']}", "lat": lat, "lon": lon,
                           "emergency": (row.get("emergency") or "").strip().lower() == "yes", "node": node})
    return placed, outside


def pick_spread(order, clat, clng, count, min_km):
    """From cells listed best-first, take `count` of them that are at least min_km apart."""
    chosen = []
    for i in order:
        if all(haversine(clat[i], clng[i], clat[j], clng[j]) >= min_km * 1000 for j in chosen):
            chosen.append(int(i))
            if len(chosen) == count:
                break
    return chosen


@st.cache_resource(show_spinner=False)
def city_grid():
    """The city cut into small square patches (for drawing). `inv` says which patch every intersection is in."""
    mid = (BOUNDS[0][0] + BOUNDS[1][0]) / 2
    height_m = (BOUNDS[1][0] - BOUNDS[0][0]) * 111_000
    width_m = (BOUNDS[1][1] - BOUNDS[0][1]) * 111_000 * math.cos(math.radians(mid))
    cell_m = max(250.0, math.sqrt(height_m * width_m / MAX_CELLS))
    dlat, dlng = cell_m / 111_000, cell_m / (111_000 * math.cos(math.radians(mid)))
    iy = ((LATS - BOUNDS[0][0]) // dlat).astype(int)
    ix = ((LNGS - BOUNDS[0][1]) // dlng).astype(int)
    width = int(ix.max()) + 1
    uniq, inv = np.unique(iy * width + ix, return_inverse=True)
    lat0 = BOUNDS[0][0] + (uniq // width) * dlat
    lng0 = BOUNDS[0][1] + (uniq % width) * dlng
    return {"inv": inv, "counts": np.bincount(inv), "lat0": lat0, "lng0": lng0, "dlat": dlat, "dlng": dlng,
            "clat": lat0 + dlat / 2, "clng": lng0 + dlng / 2}


def node_in_cell(grid, values, c, pick):
    """An intersection inside patch c: the one with the highest value, or the one nearest the patch centre."""
    members = np.nonzero(grid["inv"] == c)[0]
    if pick == "max":
        return IDS[int(members[values[members].argmax()])]
    d2 = (LATS[members] - grid["clat"][c]) ** 2 + (LNGS[members] - grid["clng"][c]) ** 2
    return IDS[int(members[d2.argmin()])]


@st.cache_resource(max_entries=24, show_spinner="Searching outward from every place at once...")
def access_search(nodes, mode):
    """One multi-source search from all the places of a kind: how long from every intersection to the nearest one.
    Reused for every click on the map. `mode` is "travel_time" (driving) or "walk"."""
    dist, toward, origin = nearest_source_distances(get_reverse_adjacency(mode), nodes)
    times = np.full(len(IDS), np.inf)
    for n, d in dist.items():
        times[INDEX[n]] = d
    grid = city_grid()
    mean = np.bincount(grid["inv"], weights=times) / grid["counts"]
    far_cells = pick_spread(np.argsort(-mean), grid["clat"], grid["clng"], 5, 2.0)                    # worst-served places
    pain_cells = pick_spread(np.argsort(-(mean * grid["counts"])), grid["clat"], grid["clng"], 12, 2.0)  # far AND many junctions
    return {"times": times, "toward": toward, "origin": origin, "cell_mean": mean,
            "farthest": [node_in_cell(grid, times, c, "max") for c in far_cells],
            "candidates": [node_in_cell(grid, times, c, "centre") for c in pain_cells]}


@st.cache_resource(max_entries=6, show_spinner="Trying possible sites for a new place. This takes a little while...")
def suggest_site(nodes, mode):
    """Try each candidate place as an extra one. A new place only helps the junctions that are closer to it
    than to their current nearest one. Keep the candidate that saves the most travel time in total."""
    cov = access_search(nodes, mode)
    now = cov["times"]
    limit = GOOD[mode]
    best = None
    for node in cov["candidates"]:
        dist, _ = all_distances(get_reverse_adjacency(mode), node)  # cost for everyone to reach this candidate
        to_new = np.full(len(IDS), np.inf)
        for n, d in dist.items():
            to_new[INDEX[n]] = d
        after = np.minimum(now, to_new)
        saved = float((now - after).sum())
        if best is None or saved > best["saved"]:
            best = {"node": node, "saved": saved, "avg_before": float(now.mean()), "avg_after": float(after.mean()),
                    "good_before": float((now <= limit).mean()), "good_after": float((after <= limit).mean())}
    return best


def route_stats(path, mode):
    """Time (s) and distance (m) along a route to a place, driving or walking."""
    if mode == "walk":
        metres = walk_length(G, path)
        return metres / WALK_MPS, metres
    return route_totals(path, "travel_time")


def fmt_long(seconds):
    """Minutes and seconds, or hours and minutes once it is over an hour (walking across the whole city takes hours)."""
    if seconds < 3600:
        return fmt_time(seconds)
    h, m = divmod(round(seconds / 60), 60)
    return f"{h} h {m:02d} min"


def box(label, value):
    return f'<div><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>'


def draw_cells(fmap, grid, bands, colours):
    for lat0, lng0, b in zip(grid["lat0"], grid["lng0"], bands):
        folium.Rectangle(bounds=[[lat0, lng0], [lat0 + grid["dlat"], lng0 + grid["dlng"]]], stroke=False, fill=True,
                         fill_color=colours[int(b)], fill_opacity=0.55).add_to(fmap)


def draw_places(fmap, places, key):
    """Emoji pins for a few places, small dots for many (so the map stays fast)."""
    cat, n = CATS[key], len(places)
    if n <= 120:
        pin_html = (f'<div style="width:26px;height:26px;border-radius:50%;background:#fff;border:2px solid {cat["colour"]};'
                    'display:flex;align-items:center;justify-content:center;font-size:14px;box-shadow:0 2px 8px rgba(0,0,0,.45)">'
                    f'{cat["emoji"]}</div>')
        for p in places:
            folium.Marker((p["lat"], p["lon"]), tooltip=html.escape(p["name"]),
                          icon=folium.DivIcon(html=pin_html, icon_size=(26, 26), icon_anchor=(13, 13))).add_to(fmap)
    else:
        for p in places:
            folium.CircleMarker((p["lat"], p["lon"]), radius=5 if n <= 1500 else 3, color="#fff", weight=1, fill=True,
                                fill_color=cat["colour"], fill_opacity=0.95,
                                tooltip=html.escape(p["name"]) if n <= 1500 else None).add_to(fmap)


def numbered_pin(fmap, latlng, number, colour, text_colour, tooltip):
    dot = (f'<div style="width:24px;height:24px;border-radius:50%;background:{colour};color:{text_colour};font:800 13px sans-serif;'
           f'display:flex;align-items:center;justify-content:center;border:2px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.45)">{number}</div>')
    folium.Marker(latlng, tooltip=tooltip, icon=folium.DivIcon(html=dot, icon_size=(24, 24), icon_anchor=(12, 12))).add_to(fmap)


def h_show_suggestion():
    ss.h_suggest = True


def h_hide_suggestion():
    ss.h_suggest = False


def h_reset_point():
    ss.h_point, ss.h_notice = None, ""


def s_reset_point():
    ss.s_point, ss.s_notice = None, ""


def s_name_spots():
    """Look up the names of the best-served spots (one request a second, as the free service asks)."""
    for lat, lon in ss.s_spots:
        if (lat, lon) in ss.s_names:
            continue
        try:
            ss.s_names[(lat, lon)] = reverse_geocode(lat, lon)
        except Exception:
            break
        time.sleep(1.1)


NO_DATA = ('<div class="notice">There are no places downloaded yet. In the terminal (venv active, in the map-router folder) run '
           '<b>python load_amenities.py</b> and wait for it to finish (it can take 10 to 30 minutes and you can run it again '
           'if the free servers are busy), then restart this app. For just hospitals, <b>python load_hospitals.py</b> is enough.</div>')


def access_page():
    st.markdown('<div class="section-title">How close is the nearest place?</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel hint">Pick a kind of place and see how far every part of Bengaluru is from its nearest one. '
                'One search starts from all of them at the same time, so every intersection learns its closest one in a single '
                'pass. Click the map to see the route from any spot.</div>', unsafe_allow_html=True)
    avail = [k for k in CATS if place_stamp(k)]
    if not avail:
        st.markdown(NO_DATA, unsafe_allow_html=True)
        return
    c1, c2 = st.columns([3, 1])
    with c1:
        key = st.segmented_control("What are you looking for?", avail, default=avail[0], key="a_cat", on_change=h_hide_suggestion,
                                   format_func=lambda k: f"{CATS[k]['emoji']} {CATS[k]['many'].capitalize()}") or avail[0]
    with c2:
        travel_label = st.segmented_control("Travel by", list(TRAVEL), default="Driving", key="a_travel",
                                            on_change=h_hide_suggestion) or "Driving"
    travel = TRAVEL[travel_label]
    cat = CATS[key]
    one, many = cat["one"], cat["many"]
    missing = [CATS[k]["many"] for k in CATS if k not in avail]
    if missing:
        st.caption(f"Not downloaded yet: {', '.join(missing)}. Run python load_amenities.py to add them.")

    placed, outside = category_sites(key, place_stamp(key))
    if not placed:
        st.markdown(f'<div class="notice">The file for {many} has nothing inside your map. Run <b>python load_amenities.py {key} --refresh</b>.</div>',
                    unsafe_allow_html=True)
        return
    use = placed
    emergency_count = sum(p["emergency"] for p in placed)
    if key == "hospital" and emergency_count >= 5:
        if st.toggle(f"Only hospitals tagged with an emergency department ({emergency_count})", value=False, key="h_emergency",
                     help="OpenStreetMap volunteers add this tag, so many emergency hospitals may be missing from it."):
            use = [p for p in placed if p["emergency"]]
    nodes = tuple(sorted({p["node"] for p in use}))
    by_node = {}
    for p in use:
        by_node.setdefault(p["node"], []).append(p["name"])
    cov = access_search(nodes, travel)
    times = cov["times"]
    limits = [b[0] for b in TIME_BANDS[:-1]]
    band_of = np.digitize(times, limits)
    shares = np.bincount(band_of, minlength=len(TIME_BANDS)) / len(times)
    good = float((times <= GOOD[travel]).mean())
    good_label = "15 minutes" if travel == "walk" else "10 minutes"
    verb = "walk" if travel == "walk" else "trip"

    st.markdown('<div class="section-title">How well is the city covered?</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel"><div class="stats" style="grid-template-columns: repeat(auto-fit, minmax(150px, 1fr))">'
        f'{box(many.capitalize() + " used", f"{len(use):,}")}{box(f"Within {good_label}", f"{good * 100:.0f}%")}'
        f'{box("Average " + verb, fmt_long(float(times.mean())))}{box("Median " + verb, fmt_long(float(np.median(times))))}'
        f'{box("Longest " + verb, fmt_long(float(times.max())))}</div></div>', unsafe_allow_html=True)
    rows = [f'<div class="meet-row head"><span></span><span>Time to the nearest {one}</span><span>Share</span>'
            '<span>Junctions</span><span></span></div>']
    for i, (_, label) in enumerate(TIME_BANDS):
        n = int((band_of == i).sum())
        rows.append(
            f'<div class="meet-row"><span class="pin-dot" style="background:{BAND_COLOURS[i]};color:{BAND_COLOURS[i]}"></span>'
            f'<span class="who">{label}</span><span class="num">{shares[i] * 100:.1f}%</span><span class="num">{n:,}</span>'
            f'<div class="bar" style="--accent:{BAND_COLOURS[i]}"><div class="bar-fill" style="width:{shares[i] * 100:.0f}%"></div></div></div>')
    st.markdown(f'<div class="panel">{"".join(rows)}</div>', unsafe_allow_html=True)
    notes = ("Driving times use each road's typical speed with no traffic, so real trips (especially in rush hour) will be longer. "
             "Walking times assume about 5 km/h and allow every road in both directions, including some where walking is not "
             "really possible. Places come from OpenStreetMap, which can be incomplete, and each one counts the same whatever its "
             "size. Each junction counts equally, not each person.")
    if outside:
        notes = f"{outside} place(s) were outside your map and ignored. " + notes
    st.caption(notes)

    st.markdown(f'<div class="section-title">Farthest from a {one}</div>', unsafe_allow_html=True)
    far_rows = ['<div class="meet-row head"><span></span><span>Place (latitude, longitude)</span><span>Time</span>'
                f'<span>Distance</span><span>Nearest {html.escape(one)}</span></div>']
    for n in cov["farthest"]:
        path = route_to_nearest(cov["toward"], n)
        secs, metres = route_stats(path, travel)
        name = by_node.get(cov["origin"][n], [one.capitalize()])[0]
        lat, lon = node_latlng(n)
        far_rows.append(
            f'<div class="meet-row"><span class="pin-dot" style="background:#343a40;color:#343a40"></span>'
            f'<span class="who">{lat:.4f}, {lon:.4f}</span><span class="num">{fmt_long(secs)}</span>'
            f'<span class="num">{metres / 1000:.1f} km</span><span class="who">{html.escape(name)}</span></div>')
    st.markdown(f'<div class="panel">{"".join(far_rows)}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-title">Where would one more {one} help most?</div>', unsafe_allow_html=True)
    cols = st.columns([1, 1, 2])
    cols[0].button("Suggest a site", on_click=h_show_suggestion, use_container_width=True, key="h_suggest_btn")
    cols[1].button("Hide suggestion", on_click=h_hide_suggestion, use_container_width=True, key="h_hide_btn",
                   disabled=not ss.h_suggest)
    best = None
    if ss.h_suggest:
        best = suggest_site(nodes, travel)
        lat, lon = node_latlng(best["node"])
        try:
            place = reverse_geocode(lat, lon)
        except Exception:
            place = ""
        title = f"Near {html.escape(place)}" if place else f"{lat:.4f}, {lon:.4f}"
        drop = 100 * (best["avg_before"] - best["avg_after"]) / best["avg_before"] if best["avg_before"] > 0 else 0
        st.markdown(
            f'<div class="panel meet-spot" style="--accent:#2f9e44"><div class="spot-label">Suggested site for a new {html.escape(one)}</div>'
            f'<div class="spot-name">{title}</div><div class="spot-sub">The average {verb} drops from <b>{fmt_long(best["avg_before"])}</b> '
            f'to <b>{fmt_long(best["avg_after"])}</b> ({drop:.1f}% shorter). Junctions within {good_label} of one go from '
            f'<b>{best["good_before"] * 100:.0f}%</b> to <b>{best["good_after"] * 100:.0f}%</b>.</div></div>', unsafe_allow_html=True)
        st.caption(f"How it was found: Wayfinder tried {len(cov['candidates'])} candidate places, chosen from areas that are both far from one "
                   "and have many junctions, and kept the one that cut total travel time the most. It is a quick "
                   "search, not a guaranteed best site, and junctions are not the same as people.")

    st.markdown('<div class="section-title">Map</div>', unsafe_allow_html=True)
    legend = "".join(f'<span class="item"><i style="background:{BAND_COLOURS[i]}"></i>{label}</span>' for i, (_, label) in enumerate(TIME_BANDS))
    st.markdown(f'<div class="legend">{legend}<span class="item">{cat["emoji"]} {html.escape(one)}</span></div>', unsafe_allow_html=True)
    fmap = folium.Map(zoom_start=11, **tiles_for(style))
    path = None
    if ss.h_point is not None:
        path = route_to_nearest(cov["toward"], ss.h_point)
        fmap.fit_bounds([node_latlng(n) for n in path], padding=(60, 60))
    else:
        fmap.fit_bounds(BOUNDS)
    draw_cells(fmap, city_grid(), np.digitize(cov["cell_mean"], limits), BAND_COLOURS)
    if path is not None:
        folium.PolyLine([node_latlng(n) for n in path], weight=6, color="#1c7ed6", opacity=0.95).add_to(fmap)
    draw_places(fmap, use, key)
    for i, n in enumerate(cov["farthest"], start=1):
        numbered_pin(fmap, node_latlng(n), i, "#343a40", "#fff", f"Far from a {one} (#{i})")
    if best is not None:
        plus = ('<div style="width:34px;height:34px;border-radius:50%;background:#2f9e44;color:#fff;font:900 24px/1 sans-serif;'
                'display:flex;align-items:center;justify-content:center;border:3px solid #fff;box-shadow:0 3px 12px rgba(0,0,0,.55)">+</div>')
        folium.Marker(node_latlng(best["node"]), tooltip=f"Suggested new {one}", z_index_offset=900,
                      icon=folium.DivIcon(html=plus, icon_size=(34, 34), icon_anchor=(17, 17))).add_to(fmap)
    if ss.h_point is not None:
        folium.CircleMarker(node_latlng(ss.h_point), radius=9, color="#fff", weight=3, fill=True, fill_color="#1c7ed6",
                            fill_opacity=1, tooltip="Your spot").add_to(fmap)
        secs, metres = route_stats(path, travel)
        name = by_node.get(cov["origin"][ss.h_point], [one.capitalize()])[0]
        st.markdown(
            f'<div class="panel meet-spot" style="--accent:#1c7ed6"><div class="spot-label">Nearest {html.escape(one)} from your spot</div>'
            f'<div class="spot-name">{html.escape(name)}</div><div class="spot-sub"><b>{fmt_long(secs)}</b> {"on foot" if travel == "walk" else "by road"} &middot; '
            f'<b>{metres / 1000:.1f} km</b></div></div>', unsafe_allow_html=True)
        st.button("Clear my spot", on_click=h_reset_point, key="h_clear_btn")
    if ss.h_notice:
        st.markdown(f'<div class="notice">{ss.h_notice}</div>', unsafe_allow_html=True)
    out = st_folium(fmap, height=620, use_container_width=True, returned_objects=["last_clicked"], key="access_map")
    note = "Colours show the average time to the nearest one in each small patch of the city. Blank patches have no roads. "
    if len(use) > 120:
        note += f"The {len(use):,} places are shown as small dots. "
    st.caption(note + "Numbered dark circles are the farthest-served places. Click the map to see the route to the nearest one.")
    click = out.get("last_clicked") if out else None
    if click and click != ss.h_last_click:
        ss.h_last_click = click
        node, off_by = nearest_node(click["lat"], click["lng"])
        if off_by > OFF_MAP_LIMIT:
            ss.h_notice = f"That click was {off_by / 1000:.1f} km from the nearest road on the map, so it was ignored."
        else:
            ss.h_point, ss.h_notice = node, ""
        st.rerun()

    with st.expander("How does this work?"):
        st.markdown(
            """
- Each place is moved to its nearest road intersection.
- A normal route search starts at one place. This one starts at **every place of the kind at once** (a *multi-source Dijkstra*),
  always expanding the cheapest option next. The first time the search reaches an intersection, that is its nearest one.
- For **driving** the roads are searched **backwards**, because the question is how long it takes to drive *to* the place, and
  one-way streets make that different from driving away from it. For **walking** every road works both ways.
- That is one search for the whole city, instead of one search for every intersection.
- *Suggest a site* tries a handful of candidate places as an extra one. It adds each as a new starting point, sees how much every
  intersection's time shrinks, and keeps the candidate with the biggest total saving.
"""
        )


def score_page():
    st.markdown('<div class="section-title">Neighbourhood score</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel hint">How many of the everyday essentials are close by? Pick the kinds of place that matter and a '
                'time limit. Every part of Bengaluru is scored by how many of them are within reach. Click the map to see the '
                'breakdown for any spot.</div>', unsafe_allow_html=True)
    avail = [k for k in CATS if place_stamp(k)]
    if len(avail) < 2:
        st.markdown(NO_DATA, unsafe_allow_html=True)
        return
    c1, c2, c3 = st.columns([3, 1, 1])
    chosen = c1.multiselect("Essentials to include", avail, default=avail, key="s_cats",
                            format_func=lambda k: f"{CATS[k]['emoji']} {CATS[k]['many'].capitalize()}")
    travel_label = c2.segmented_control("Travel by", list(TRAVEL), default="Walking", key="s_travel") or "Walking"
    minutes = c3.select_slider("Within (minutes)", options=[5, 10, 15, 20, 30], value=15, key="s_minutes")
    travel, limit = TRAVEL[travel_label], minutes * 60
    data = {}
    for k in chosen:
        placed, _ = category_sites(k, place_stamp(k))
        if not placed:
            continue
        by_node = {}
        for p in placed:
            by_node.setdefault(p["node"], []).append(p["name"])
        data[k] = {"cov": access_search(tuple(sorted(by_node)), travel), "by_node": by_node, "n": len(placed)}
    if not data:
        st.markdown('<div class="notice">Choose at least one kind of place above.</div>', unsafe_allow_html=True)
        return
    K = len(data)
    within = {k: d["cov"]["times"] <= limit for k, d in data.items()}
    score = sum(w.astype(int) for w in within.values())
    way = "on foot" if travel == "walk" else "by car"

    st.markdown('<div class="section-title">How well is the city covered?</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel"><div class="stats" style="grid-template-columns: repeat(auto-fit, minmax(150px, 1fr))">'
        f'{box("Essentials counted", K)}{box(f"All {K} within {minutes} min", f"{(score == K).mean() * 100:.0f}%")}'
        f'{box("Average covered", f"{score.mean():.1f} of {K}")}{box("Nothing within reach", f"{(score == 0).mean() * 100:.0f}%")}'
        f'</div></div>', unsafe_allow_html=True)
    rows = [f'<div class="meet-row head"><span></span><span>Within {minutes} min {way}</span><span>Share</span><span>Places</span><span></span></div>']
    for k, d in data.items():
        share = float(within[k].mean())
        c = CATS[k]
        rows.append(
            f'<div class="meet-row"><span class="pin-dot" style="background:{c["colour"]};color:{c["colour"]}"></span>'
            f'<span class="who">{c["emoji"]} {c["many"].capitalize()}</span><span class="num">{share * 100:.0f}%</span>'
            f'<span class="num">{d["n"]:,}</span><div class="bar" style="--accent:{c["colour"]}">'
            f'<div class="bar-fill" style="width:{share * 100:.0f}%"></div></div></div>')
    st.markdown(f'<div class="panel">{"".join(rows)}</div>', unsafe_allow_html=True)
    st.caption("The share is the part of the city's junctions that have at least one of these within reach. "
               "Each junction counts equally, not each person, and places come from OpenStreetMap, which can be incomplete.")

    grid = city_grid()
    cell_mean = np.bincount(grid["inv"], weights=score) / grid["counts"]
    cell_band = np.digitize(cell_mean / K, [0.2, 0.4, 0.6, 0.8, 0.95])
    order = np.argsort(-cell_mean)
    best_cells = pick_spread(order, grid["clat"], grid["clng"], 5, 2.0)
    best_nodes = [node_in_cell(grid, score, c, "max") for c in best_cells]
    ss.s_spots = [tuple(round(v, 5) for v in node_latlng(n)) for n in best_nodes]

    st.markdown('<div class="section-title">Best-served spots</div>', unsafe_allow_html=True)
    spot_rows = ['<div class="meet-row head"><span></span><span>Place</span><span>Essentials</span><span></span><span></span></div>']
    for i, n in enumerate(best_nodes):
        lat, lon = ss.s_spots[i]
        label = html.escape(ss.s_names[(lat, lon)]) if (lat, lon) in ss.s_names else f"{lat:.4f}, {lon:.4f}"
        spot_rows.append(
            f'<div class="meet-row"><span class="pin-dot" style="background:#e8a317;color:#e8a317"></span>'
            f'<span class="who">{label}</span><span class="num">{int(score[INDEX[n]])} of {K}</span><span></span><span></span></div>')
    st.markdown(f'<div class="panel">{"".join(spot_rows)}</div>', unsafe_allow_html=True)
    st.button("Look up the names of these places", on_click=s_name_spots, key="s_name_btn",
              help="Asks the free OpenStreetMap search service, one place per second.")

    st.markdown('<div class="section-title">Map</div>', unsafe_allow_html=True)
    legend = "".join(f'<span class="item"><i style="background:{SCORE_COLOURS[i]}"></i>{label}</span>' for i, label in enumerate(SCORE_LABELS))
    st.markdown(f'<div class="legend">{legend}<span class="item">of the essentials within {minutes} min {way}</span></div>', unsafe_allow_html=True)
    fmap = folium.Map(zoom_start=11, **tiles_for(style))
    paths = {}
    if ss.s_point is not None:
        for k, d in data.items():
            paths[k] = route_to_nearest(d["cov"]["toward"], ss.s_point)
        fmap.fit_bounds([node_latlng(n) for p in paths.values() for n in p], padding=(60, 60))
    else:
        fmap.fit_bounds(BOUNDS)
    draw_cells(fmap, grid, cell_band, SCORE_COLOURS)
    for k, p in paths.items():
        folium.PolyLine([node_latlng(n) for n in p], weight=4, color=CATS[k]["colour"], opacity=0.95).add_to(fmap)
    for i, n in enumerate(best_nodes, start=1):
        numbered_pin(fmap, node_latlng(n), i, "#e8a317", "#fff", f"Best-served spot (#{i})")
    if ss.s_point is not None:
        folium.CircleMarker(node_latlng(ss.s_point), radius=9, color="#fff", weight=3, fill=True, fill_color="#1c7ed6",
                            fill_opacity=1, tooltip="Your spot").add_to(fmap)
        idx = INDEX[ss.s_point]
        reach = int(score[idx])
        lines = [f'<div class="meet-row head"><span></span><span>Nearest</span><span>Time</span><span>Distance</span><span></span></div>']
        for k, d in data.items():
            secs, metres = route_stats(paths[k], travel)
            name = d["by_node"].get(d["cov"]["origin"][ss.s_point], [CATS[k]["one"]])[0]
            ok = secs <= limit
            lines.append(
                f'<div class="meet-row"><span class="pin-dot" style="background:{CATS[k]["colour"]};color:{CATS[k]["colour"]}"></span>'
                f'<span class="who">{CATS[k]["emoji"]} {html.escape(name)}</span><span class="num">{fmt_long(secs)}</span>'
                f'<span class="num">{metres / 1000:.1f} km</span><span class="who">{"Within reach" if ok else "Too far"}</span></div>')
        st.markdown(
            f'<div class="panel meet-spot" style="--accent:#1c7ed6"><div class="spot-label">Your spot</div>'
            f'<div class="spot-name">{reach} of {K} essentials within {minutes} min {way}</div></div>'
            f'<div class="panel">{"".join(lines)}</div>', unsafe_allow_html=True)
        st.button("Clear my spot", on_click=s_reset_point, key="s_clear_btn")
    if ss.s_notice:
        st.markdown(f'<div class="notice">{ss.s_notice}</div>', unsafe_allow_html=True)
    out = st_folium(fmap, height=620, use_container_width=True, returned_objects=["last_clicked"], key="score_map")
    st.caption("Each patch is coloured by the share of the chosen essentials that are within reach. Blank patches have no roads. "
               "Numbered gold circles are the best-served spots. Click the map for a spot's breakdown, with a line to the nearest "
               "of each kind.")
    click = out.get("last_clicked") if out else None
    if click and click != ss.s_last_click:
        ss.s_last_click = click
        node, off_by = nearest_node(click["lat"], click["lng"])
        if off_by > OFF_MAP_LIMIT:
            ss.s_notice = f"That click was {off_by / 1000:.1f} km from the nearest road on the map, so it was ignored."
        else:
            ss.s_point, ss.s_notice = node, ""
        st.rerun()

    with st.expander("How does this work?"):
        st.markdown(
            """
- For each kind of place there is one **multi-source Dijkstra** search that starts from all of them at once, giving every
  intersection its time to the nearest one.
- A junction's **score** is how many of the chosen kinds are within your time limit. A patch is coloured by the average score of
  the junctions in it, as a share of the kinds you chose.
- It is the idea behind the "15-minute city": can you reach what you need in a short walk?
- Tip: switch off the kinds that do not matter to you (say, schools) to see how the best areas change.
"""
        )


# ---------- mode switch ----------
page = st.segmented_control("Mode", [PAGE_ROUTES, PAGE_MEET, PAGE_ACCESS, PAGE_SCORE], default=PAGE_ROUTES, key="page",
                            label_visibility="collapsed") or PAGE_ROUTES

# ---------- sidebar ----------
view, show_explored, show_alts = "A*", True, True
with st.sidebar:
    st.markdown("### Settings")
    mode = "Fastest"
    if page in (PAGE_ROUTES, PAGE_MEET):
        mode = st.segmented_control("Route type", ["Fastest", "Shortest"], default="Fastest",
                                    help="Fastest minimises travel time. Shortest minimises distance.") or "Fastest"
    weight = "travel_time" if mode == "Fastest" else "length"
    if page == PAGE_ROUTES:
        view = st.segmented_control("Show the search of", ["A*", "Dijkstra", "Bidirectional"], default="A*") or "A*"
        show_explored = st.toggle("Show explored intersections", value=True)
        show_alts = st.toggle("Show alternative routes", value=True)
    style = st.segmented_control("Map style", ["Street", "Satellite"], default="Street") or "Street"
    st.divider()
    if page == PAGE_ROUTES:
        st.caption("Search for places, click the map, or use the quick trips.")
    elif page == PAGE_MEET:
        st.caption("Add people by search, coordinates, your current location, or by clicking the map.")
    elif page == PAGE_ACCESS:
        st.caption("Pick a kind of place and how you travel, then click the map to see the route to the nearest one.")
    else:
        st.caption("Choose the essentials and a time limit. Click the map to see what is within reach of any spot.")

if page == PAGE_MEET:
    meeting_page()
    st.stop()
if page == PAGE_ACCESS:
    access_page()
    st.stop()
if page == PAGE_SCORE:
    score_page()
    st.stop()

# ---------- search ----------
st.markdown('<div class="section-title">Search places in Bengaluru</div>', unsafe_allow_html=True)
search_cols = st.columns(2)
for col, side, label, example in (
    (search_cols[0], "from", "From", "e.g. Lalbagh, Koramangala, Majestic"),
    (search_cols[1], "to", "To", "e.g. Whitefield, Phoenix Marketcity, Hebbal"),
):
    with col:
        with st.form(f"form_{side}", border=False):
            st.text_input(label, key=f"q_{side}", placeholder=example)
            st.form_submit_button(f"Find {'start' if side == 'from' else 'end'} point", on_click=do_search,
                                  args=(side,), use_container_width=True)
        found = ss.get(f"res_{side}", [])
        if found:
            st.selectbox("Matches (pick another if the first is not right)", options=list(range(len(found))),
                         format_func=lambda i, f=found: f[i]["label"], key=f"pick_{side}",
                         on_change=pick_place, args=(side,))
        if ss.get(f"err_{side}"):
            st.caption(ss[f"err_{side}"])

# ---------- quick trips ----------
st.markdown('<div class="section-title">Quick trips</div>', unsafe_allow_html=True)
actions = [(label, set_trip, (a, b, tuple(label.split(" → ")))) for label, a, b in PRESETS_OK]
actions += [("🎲 Random trip", random_trip, ()), ("⇄ Swap", swap_trip, ()), ("Clear", clear_trip, ())]
for row_start in range(0, len(actions), 4):
    cols = st.columns(4)
    for col, (label, fn, args) in zip(cols, actions[row_start:row_start + 4]):
        col.button(label, on_click=fn, args=args, use_container_width=True, key=f"btn_{label}")


# ---------- trip summary ----------
def pin(label, node, colour, name=None):
    if node is None:
        text, sub = "Not set", ""
    else:
        coords = "{:.4f}, {:.4f}".format(*node_latlng(node))
        text, sub = (html.escape(name), coords) if name else (coords, "")
    sub_html = f'<div class="pin-sub">{sub}</div>' if sub else ""
    return (f'<div class="pin"><span class="pin-dot" style="background:{colour};color:{colour}"></span>'
            f'<div><div class="pin-label">{label}</div><div class="pin-text">{text}</div>{sub_html}</div></div>')


if ss.start is None:
    hint = "Click anywhere on the map to set the <b>start</b> point."
elif ss.end is None:
    hint = "Now click the map to set the <b>end</b> point."
else:
    hint = "Click the map again to start a new trip. The next click sets a new start."

c1, c2, c3 = st.columns([1, 1, 1.4])
c1.markdown(f'<div class="panel">{pin("Start", ss.start, "#2f9e44", ss.start_name)}</div>', unsafe_allow_html=True)
c2.markdown(f'<div class="panel">{pin("End", ss.end, "#e03131", ss.end_name)}</div>', unsafe_allow_html=True)
c3.markdown(f'<div class="panel hint">{hint}</div>', unsafe_allow_html=True)
if ss.notice:
    st.markdown(f'<div class="notice">{ss.notice}</div>', unsafe_allow_html=True)

# ---------- results ----------
results = None
if ss.start is not None and ss.end is not None:
    results = search_all(ss.start, ss.end, weight)
    if results["Dijkstra"]["path"] is None:
        st.warning("There is no route between these two points. Pick other points.")
        results = None


def stat(label, value):
    return f'<div><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>'


def algo_card(name, colour, r, winner, share):
    secs, metres = route_totals(r["path"], weight)
    badge = '<span class="badge">Fewest explored</span>' if winner else ""
    cls = "algo winner" if winner else "algo"
    visited = f"{r['visited']:,}"
    search_ms = f"{r['ms']:.1f} ms"
    dist = f"{metres / 1000:.2f} km"
    return (
        f'<div class="{cls}" style="--accent:{colour}">'
        f'<div class="algo-head"><span class="algo-dot"></span><span class="algo-name">{name}</span>{badge}</div>'
        f'<div class="stats">{stat("Travel time", fmt_time(secs))}{stat("Distance", dist)}'
        f'{stat("Intersections explored", visited)}{stat("Search time", search_ms)}</div>'
        f'<div class="bar"><div class="bar-fill" style="width:{share:.0f}%"></div></div>'
        f'<div class="bar-label">Search effort (shorter bar = less work)</div>'
        f"</div>"
    )


COLOURS = {"Dijkstra": DIJKSTRA_COLOUR, "A*": ASTAR_COLOUR, "Bidirectional": BIDIR_COLOUR}


def vs_dijkstra(name):
    """'3.2x fewer' or '1.4x more' intersections explored than Dijkstra."""
    ratio = results["Dijkstra"]["visited"] / max(results[name]["visited"], 1)
    if ratio >= 1:
        return f"{name} explored <b>{ratio:.1f}x fewer</b>"
    return f"{name} explored <b>{1 / ratio:.1f}x more</b>"


if results:
    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
    visits = {n: results[n]["visited"] for n in ALGORITHMS}
    fewest = min(visits.values())
    biggest = max(max(visits.values()), 1)
    for col, name in zip(st.columns(3), ALGORITHMS):
        winner = visits[name] == fewest and list(visits.values()).count(fewest) == 1
        col.markdown(algo_card(name, COLOURS[name], results[name], winner, 100 * visits[name] / biggest),
                     unsafe_allow_html=True)
    costs = [results[n]["cost"] for n in ALGORITHMS]
    same = max(costs) - min(costs) < 1e-6
    verdict = f"{vs_dijkstra('A*')} and {vs_dijkstra('Bidirectional')} intersections than Dijkstra"
    verdict += ". All three found a route with exactly the same cost." if same else "."
    st.markdown(f'<div class="verdict">{verdict}</div>', unsafe_allow_html=True)

# ---------- route options (best route + alternatives) ----------
ROUTE_COLOURS = ["#1971c2", "#e8590c", "#2f9e44"]
routes = []  # (label, path, colour)
route_note = ""
if results:
    best_label = "Fastest" if weight == "travel_time" else "Shortest"
    routes.append((best_label, results[view]["path"], ROUTE_COLOURS[0]))
    if show_alts:
        alts = search_alternatives(ss.start, ss.end, weight, tuple(results[view]["path"]))
        for i, alt_path in enumerate(alts):
            routes.append((f"Alternative {i + 1}", alt_path, ROUTE_COLOURS[i + 1]))
        if len(alts) == 0:
            route_note = "No other route is meaningfully different for this trip."
        elif len(alts) == 1:
            route_note = "Only one other route is meaningfully different for this trip."


def route_card(label, colour, path, best_totals, selected):
    secs, metres = route_totals(path, weight)
    if best_totals is None:
        sub = "Best route"
    else:
        sub = f"{fmt_delta(secs - best_totals[0])}, {(metres - best_totals[1]) / 1000:+.1f} km"
    cls = "panel route-card selected" if selected else "panel route-card"
    return (f'<div class="{cls}" style="--accent:{colour}"><div class="route-swatch"></div>'
            f'<div class="route-label">{label}</div><div class="route-time">{fmt_time(secs)}</div>'
            f'<div class="route-sub">{metres / 1000:.1f} km &middot; {sub}</div></div>')


pick = None
if routes:
    st.markdown('<div class="section-title">Route options</div>', unsafe_allow_html=True)
    labels = [r[0] for r in routes]
    pick = (st.segmented_control("Highlight on the map", labels, default=labels[0]) or labels[0]) if len(labels) > 1 else labels[0]
    best_totals = route_totals(routes[0][1], weight)
    for col, (label, path, colour) in zip(st.columns(3), routes):
        col.markdown(route_card(label, colour, path, None if label == routes[0][0] else best_totals, label == pick),
                     unsafe_allow_html=True)
    if route_note:
        st.caption(route_note)

# ---------- map ----------
st.markdown('<div class="section-title">Map</div>', unsafe_allow_html=True)
tile_args = tiles_for(style)

if results:
    view_result = results[view]
    all_coords = [node_latlng(n) for _, path, _ in routes for n in path]
    fmap = folium.Map(zoom_start=15, **tile_args)
    fmap.fit_bounds(all_coords, padding=(40, 40))
elif ss.start is not None:
    fmap = folium.Map(location=node_latlng(ss.start), zoom_start=15, **tile_args)
else:
    fmap = folium.Map(zoom_start=11, **tile_args)
    fmap.fit_bounds(BOUNDS)

sample_note = ""


def add_dots(points, first_colour, last_colour):
    """Draw explored intersections, colour by order. Returns True if a sample was drawn."""
    step = max(1, len(points) // MAX_DOTS)  # draw a sample if there are too many
    shown = points[::step]
    n = len(shown)
    for i, node in enumerate(shown):
        folium.CircleMarker(
            node_latlng(node), radius=3, weight=0, fill=True,
            fill_color=blend(first_colour, last_colour, i / max(n - 1, 1)), fill_opacity=0.85,
        ).add_to(fmap)
    return step > 1


if results and show_explored:
    sampled = add_dots(view_result["explored"], "#ffe08a", "#5b1a6e")
    sampled = add_dots(view_result["explored_back"], "#a5f3fc", "#0b4f6c") or sampled
    if sampled:
        sample_note = " Showing a sample of the explored intersections, so the map stays fast."
if results:
    # draw the highlighted route last so it sits on top
    for label, path, colour in [r for r in routes if r[0] != pick] + [r for r in routes if r[0] == pick]:
        chosen = label == pick
        folium.PolyLine([node_latlng(n) for n in path], weight=8 if chosen else 5, color=colour,
                        opacity=1.0 if chosen else 0.75, tooltip=label).add_to(fmap)
if ss.start is not None:
    folium.Marker(node_latlng(ss.start), tooltip="Start",
                  icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(fmap)
if ss.end is not None:
    folium.Marker(node_latlng(ss.end), tooltip="End",
                  icon=folium.Icon(color="red", icon="flag", prefix="fa")).add_to(fmap)

out = st_folium(fmap, height=650, use_container_width=True, returned_objects=["last_clicked"], key="map")

if results and show_explored:
    if view == "Bidirectional":
        legend = ("Yellow to purple dots: the forward search from the start. "
                  "Cyan to blue dots: the backward search from the end. They meet in the middle.")
    else:
        legend = "Dots are intersections the search visited: yellow = early, purple = late."
    st.caption(legend + sample_note)

# ---------- handle a click ----------
click = out.get("last_clicked") if out else None
if click and click != ss.last_click:
    ss.last_click = click
    node, off_by = nearest_node(click["lat"], click["lng"])
    if off_by > 600:
        ss.notice = (f"That click was {off_by / 1000:.1f} km outside the map you downloaded, so the nearest "
                     "intersection was used. To cover more of Bengaluru, set MAP_SIZE in load_map.py and run it again.")
    else:
        ss.notice = ""
    if ss.start is None or ss.end is not None:
        ss.start, ss.end = node, None
        ss.start_name = ss.end_name = None
    else:
        ss.end, ss.end_name = node, None
    st.rerun()

# ---------- explanation ----------
with st.expander("How does this work?"):
    st.markdown(
        """
- The road network is a **graph**: intersections are nodes and road segments are edges, weighted by travel time or length.
- **Dijkstra** always expands the closest unexplored intersection, so it spreads out in every direction.
- **A\\*** does the same but adds a guess of the remaining cost (the straight-line distance to the destination), so it leans toward the target and explores fewer intersections.
- **Bidirectional Dijkstra** runs two searches at once: one forward from the start and one backward from the end (following one-way roads in reverse). Each only needs to cover about half the trip, and they stop when they can prove no shorter route exists.
- **Alternative routes** use the *penalty method*: after the best route is found, its roads are made more expensive and the search runs again. A new route is kept only if it is different enough from the routes already shown and not much slower.
- All three return the same optimal cost. The smarter ones just do less work. For *Fastest* routes the A* guess is weak (it has to assume the fastest speed on the map), so try *Shortest* to see A* help more.
- All three algorithms are written from scratch in `algorithms.py`.
"""
    )
