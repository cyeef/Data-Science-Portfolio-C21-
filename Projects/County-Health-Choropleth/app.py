"""
Bivariate Choropleth — Child Poverty x Low Birthweight (U.S. Counties)
=====================================================================
Streamlit app that maps two CHR 2025 variables at once:

    children_in_poverty_pct  (x)   vs.   low_birthweight_pct  (y)

Each county is placed on a 3x3 bivariate colour grid. Counties that are
HIGH on BOTH measures light up in the darkest corner colour — the
"pockets" where the two burdens concentrate.

Data is pulled at runtime from the published Hugging Face dataset:
    RaayGunz/county-health-maternal-clean

Run locally:   python -m streamlit run app.py
"""

import json
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(page_title="Poverty x Low Birthweight", layout="wide")

HF_CSV = ("https://huggingface.co/datasets/RaayGunz/"
          "county-health-maternal-clean/resolve/main/"
          "county_health_maternal_clean.csv")

COUNTIES_GEOJSON = ("https://raw.githubusercontent.com/plotly/datasets/"
                    "master/geojson-counties-fips.json")

POV = "children_in_poverty_pct"
LBW = "low_birthweight_pct"

# 3x3 bivariate palette (Stevens green-blue).
# Key "p-l": p = poverty tier (1 low..3 high), l = low-birthweight tier.
# The 3-3 corner (both high) is the darkest.
BIVAR_COLORS = {
    "1-1": "#e8e8e8", "1-2": "#b5c0da", "1-3": "#6c83b5",
    "2-1": "#b8d6be", "2-2": "#90b2b3", "2-3": "#567994",
    "3-1": "#73ae80", "3-2": "#5a9178", "3-3": "#2a5a5b",
}
TIER_LABEL = {1: "Low", 2: "Med", 3: "High"}


# --------------------------------------------------------------------------
# Data loading (cached so it only fetches once)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading CHR data...")
def load_data():
    df = pd.read_csv(HF_CSV, dtype={"fipscode": str})
    df["fipscode"] = df["fipscode"].str.zfill(5)          # keep leading zeros
    df = df.dropna(subset=[POV, LBW]).copy()
    # 3 tiers per variable via tertiles
    df["pov_tier"] = pd.qcut(df[POV], 3, labels=[1, 2, 3]).astype(int)
    df["lbw_tier"] = pd.qcut(df[LBW], 3, labels=[1, 2, 3]).astype(int)
    df["bivar"] = df["pov_tier"].astype(str) + "-" + df["lbw_tier"].astype(str)
    return df


@st.cache_data(show_spinner="Loading county boundaries...")
def load_geojson():
    with urlopen(COUNTIES_GEOJSON) as resp:
        return json.load(resp)


df = load_data()
counties = load_geojson()

# --------------------------------------------------------------------------
# Sidebar — brief + controls
# --------------------------------------------------------------------------
st.sidebar.title("About this map")
st.sidebar.markdown(
    """
This map shows U.S. counties by **child poverty** and **low birthweight**
(County Health Rankings 2025).

Analysis of this data found **poverty is the dominant predictor** of low
birthweight (regression coefficient 1.45 — about 4x any other factor),
while **air pollution showed a negligible effect**. The darkest counties —
high in *both* poverty and low birthweight — are where the two burdens
concentrate.

Use the controls below to explore.
"""
)

mode = st.sidebar.radio(
    "Map mode",
    ["Bivariate (3x3)", "Threshold highlight"],
    help="Bivariate shades every county on a 3x3 grid. "
         "Threshold highlight lets you set your own cut-offs.",
)

states = ["All US"] + sorted(df["state"].dropna().unique().tolist())
state_pick = st.sidebar.selectbox("Zoom to state", states)

view = df if state_pick == "All US" else df[df["state"] == state_pick]

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("Child Poverty x Low Birthweight — U.S. Counties")

if mode == "Bivariate (3x3)":
    view = view.copy()
    view["color"] = view["bivar"].map(BIVAR_COLORS)
    view["pov_lbl"] = view["pov_tier"].map(TIER_LABEL)
    view["lbw_lbl"] = view["lbw_tier"].map(TIER_LABEL)
    view["hover"] = (
        view["county"] + ", " + view["state"]
        + "<br>Poverty: " + view[POV].round(1).astype(str) + "%"
        + " (" + view["pov_lbl"] + ")"
        + "<br>Low birthweight: " + view[LBW].round(1).astype(str) + "%"
        + " (" + view["lbw_lbl"] + ")"
    )

    # go.Choropleth cannot take per-feature hex directly, so we draw one
    # trace per bivariate class (up to 9) — each trace is a single flat colour.
    fig = go.Figure()
    for cls, color in BIVAR_COLORS.items():
        sub = view[view["bivar"] == cls]
        if sub.empty:
            continue
        fig.add_trace(go.Choropleth(
            geojson=counties,
            locations=sub["fipscode"],
            z=[1] * len(sub),
            text=sub["hover"],
            hoverinfo="text",
            colorscale=[[0, color], [1, color]],
            showscale=False,
            marker_line_width=0.1,
            marker_line_color="white",
        ))

else:  # Threshold highlight
    c1, c2 = st.columns(2)
    pov_cut = c1.slider("Poverty threshold (%)",
                        float(df[POV].min()), float(df[POV].max()),
                        float(df[POV].median()))
    lbw_cut = c2.slider("Low-birthweight threshold (%)",
                        float(df[LBW].min()), float(df[LBW].max()),
                        float(df[LBW].median()))

    view = view.copy()
    view["hit"] = ((view[POV] >= pov_cut) & (view[LBW] >= lbw_cut)).astype(int)
    n_hit = int(view["hit"].sum())
    view["hover"] = (
        view["county"] + ", " + view["state"]
        + "<br>Poverty: " + view[POV].round(1).astype(str) + "%"
        + "<br>Low birthweight: " + view[LBW].round(1).astype(str) + "%"
        + "<br>" + np.where(view["hit"] == 1,
                            "MEETS both thresholds", "below threshold")
    )

    st.markdown(
        f"**{n_hit}** counties are at or above **{pov_cut:.1f}%** poverty "
        f"*and* **{lbw_cut:.1f}%** low birthweight"
        + ("" if state_pick == "All US" else f" in {state_pick}") + "."
    )

    fig = go.Figure(go.Choropleth(
        geojson=counties,
        locations=view["fipscode"],
        z=view["hit"],
        text=view["hover"],
        hoverinfo="text",
        colorscale=[[0, "#e8e8e8"], [1, "#2a5a5b"]],
        showscale=False,
        marker_line_width=0.1,
        marker_line_color="white",
    ))

# Fit the map to the selection
if state_pick == "All US":
    fig.update_geos(scope="usa")
else:
    fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=600)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Legend for bivariate mode (a small 3x3 grid drawn as HTML)
# --------------------------------------------------------------------------
if mode == "Bivariate (3x3)":
    st.markdown("#### Legend — read toward the dark corner")
    rows = ""
    for p in [3, 2, 1]:                       # high poverty on top
        cells = ""
        for l in [1, 2, 3]:                   # low->high LBW left to right
            cells += (f"<td style='background:{BIVAR_COLORS[f'{p}-{l}']};"
                      f"width:46px;height:46px;'></td>")
        label = TIER_LABEL[p]
        rows += f"<tr><td style='padding-right:6px'>{label} pov.</td>{cells}</tr>"
    footer = ("<tr><td></td><td style='text-align:center'>Low</td>"
              "<td style='text-align:center'>Med</td>"
              "<td style='text-align:center'>High</td></tr>"
              "<tr><td></td><td colspan='3' style='text-align:center'>"
              "Low birthweight &rarr;</td></tr>")
    st.markdown(f"<table>{rows}{footer}</table>", unsafe_allow_html=True)
    st.caption("Darkest = high poverty AND high low birthweight — the pockets.")

# --------------------------------------------------------------------------
# Data source note
# --------------------------------------------------------------------------
st.caption(
    "Source: County Health Rankings & Roadmaps 2025 (Univ. of Wisconsin "
    "Population Health Institute), cleaned dataset at "
    "huggingface.co/datasets/RaayGunz/county-health-maternal-clean. "
    "Non-commercial use (CC-BY-NC-4.0). County-level associations only; "
    "not causal or individual-level claims."
)
