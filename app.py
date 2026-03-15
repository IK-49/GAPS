import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd

st.set_page_config(page_title="GAPS Map", layout="wide")

TOUR_STEPS = [
    ("🗺️ Welcome to GAPS", "This tool maps high-probability zones for undiscovered shipwrecks along the North Carolina Outer Banks, also known as the 'Graveyard of the Atlantic'. Click Next to learn how it works."),
    ("📊 The Grid", "The study area is divided into 91,980 tiles at 500×500m resolution. Each tile is scored by a Random Forest model trained on 102 confirmed wreck locations from NOAA nautical chart data."),
    ("🔴 Red Circles", "Red circles are high-risk tiles or the top N% most likely to contain an undiscovered wreck. Opacity indicates confidence: brighter red = higher ML probability according to the model."),
    ("🟡 Yellow Dots", "Yellow dots are confirmed wreck locations from NOAA ENC charts and the NOAA ENC Direct API. These were used to train and validate the model."),
    ("🎚️ The Slider", "Use the sidebar slider to adjust how many tiles are shown. At just 1%, the model captures 82% of known wrecks while reducing the search area by 99%."),
    ("📈 The Metrics", "Total Tiles Analyzed = full study area. High Risk Zones = tiles above your threshold. Known Wrecks Captured = how many confirmed wrecks fall inside high-risk zones."),
    ("✅ Ready", "You're all set. Zoom into the Outer Banks to explore high-risk clusters. Hover over the red circles for depth and probability details."),
]

if 'tour_active' not in st.session_state:
    st.session_state.tour_active = False
if 'tour_step' not in st.session_state:
    st.session_state.tour_step = 0

if st.sidebar.button("📖 Take the Tour"):
    st.session_state.tour_active = True
    st.session_state.tour_step = 0

if st.session_state.tour_active:
    step = st.session_state.tour_step
    title, text = TOUR_STEPS[step]

    with st.container():
        st.info(f"**{title}** ({step + 1}/{len(TOUR_STEPS)})\n\n{text}")
        col_prev, col_next, col_close = st.columns([1, 1, 1])

        with col_prev:
            if step > 0:
                if st.button("← Back"):
                    st.session_state.tour_step -= 1
                    st.rerun()

        with col_next:
            if step < len(TOUR_STEPS) - 1:
                if st.button("Next →"):
                    st.session_state.tour_step += 1
                    st.rerun()
            else:
                if st.button("✅ Finish"):
                    st.session_state.tour_active = False
                    st.rerun()

        with col_close:
            if st.button("✕ Close"):
                st.session_state.tour_active = False
                st.rerun()

st.title("Graveyard of the Atlantic Predictive Safeguarding (GAPS)")
st.markdown("Identifying high-risk zones for undiscovered shipwrecks along the North Carolina coast using NOAA nautical chart data.")

# load data from gpkg
@st.cache_data
def load_data():
    gdf = gpd.read_file('./data_sources/grid_with_risk.gpkg')
    gdf = gdf.dropna(subset=['DRVAL1', 'DRVAL2', 'depth_range', 'obstruction_distance'])
    gdf = gdf.to_crs(epsg=4326)
    # precompute centroids once
    gdf['cx'] = gdf.geometry.centroid.x
    gdf['cy'] = gdf.geometry.centroid.y
    return gdf

with st.spinner("Loading NOAA nautical chart data..."): # loading icon
    gdf = load_data()       

# Controls sidebar
st.sidebar.header("Controls")

top_percentile = st.sidebar.slider(
    "Show Top % High Risk Tiles",
    min_value=1,
    max_value=10,
    value=1,
    step=1,
    help="Show only the top N% highest scoring tiles"
)

show_wrecks = st.sidebar.checkbox("Show confirmed wrecks", value=True)
show_high_risk = st.sidebar.checkbox("Show high-risk zones", value=True)

# Statistics
col1, col2, col3, col4 = st.columns(4)

threshold    = gdf['wreck_prob'].quantile(1 - top_percentile / 100)
high_risk    = gdf[gdf['wreck_prob'] >= threshold]
total_wrecks = gdf[gdf['target'] == 1]
captured     = high_risk[high_risk['target'] == 1]

col1.metric("Total Tiles Analyzed", f"{len(gdf):,}")
col2.metric("High Risk Zones", f"{len(high_risk):,}")
col3.metric("Known Wrecks Captured", f"{len(captured)}/{len(total_wrecks)}")
col4.metric("Search Area Reduction", f"{100 - top_percentile}%")

# Map Rendering w/ Folium
with st.spinner("Rendering map..."):
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=9, tiles='CartoDB dark_matter')

    if show_high_risk:
        folium.GeoJson(
            high_risk[['geometry', 'wreck_prob', 'DRVAL1']].to_json(),
            style_function=lambda f: {
                'fillColor': 'red',
                'color': 'red',
                'weight': 0,
                'fillOpacity': f['properties']['wreck_prob']
            },
            tooltip=folium.GeoJsonTooltip(fields=['wreck_prob', 'DRVAL1'],
                                           aliases=['ML Probability', 'Depth (m)'])
        ).add_to(m)

    if show_wrecks:
        folium.GeoJson(
            total_wrecks[['geometry']].to_json(),
            style_function=lambda f: {
                'fillColor': 'yellow',
                'color': 'yellow',
                'weight': 1,
                'fillOpacity': 0.9
            },
            tooltip=folium.GeoJsonTooltip(fields=[], aliases=[])
        ).add_to(m)

    st_folium(m, width=1200, height=500)

# Table Rendering
with st.expander("Feature Analysis"):
    st.markdown("#### Mean feature values by class")
    feat_means = gdf.groupby('target')[['DRVAL1', 'DRVAL2', 'depth_range', 'obstruction_distance']].mean().round(2)
    feat_means.index = ['No Wreck', 'Wreck']
    st.dataframe(feat_means)

# Methodology
with st.expander("Methodology"):
    st.markdown("""
    **Overview**
    
    GAPS divides the North Carolina coastline into a grid of 91,980 tiles, each 500×500 meters, and uses Random Forest Classification to score every tile by its likelihood of containing an undiscovered shipwreck.

    **Data**
    
    All data comes from NOAA's Electronic Navigational Charts, which contain decades of hydrographic survey data for U.S. coastal waters. GAPS pulls confirmed wreck locations, water depth measurements, and charted obstruction points for the Outer Banks and Pamlico Sound region. To supplement the local dataset, the NOAA ENC Direct API was queried for additional harbour-scale wreck records, bringing our total confirmed wreck count to 102 from 59.

    **How the model works**
    
    Each tile has four key features for training: minimum depth, maximum depth, depth variability, and distance to the nearest charted obstruction. A common pattern observed is that wrecks occur significantly more in shallower water (37m average) compared to the surrounding ocean (162m average). The features were then inputted into the model in order to distinguish wreck-likely tiles from the rest.

    **Results**
    
    At the default 1% threshold, the model flags just 930 tiles out of 91,980 while still capturing 82% of all known wreck sites. This reduces the effective search area by 99% compared to an uninformed survey.

    **Limitations**
    
    102 confirmed wrecks is a small training set for a dataset of this size, which caps the model's precision. The Outer Banks is also uniformly shallow, so depth alone can't pinpoint exact wreck locations. Incorporating seafloor composition, historical shipping routes, and sediment transport data would meaningfully improve accuracy and represents the best way to expand GAPS for further use.
    """)

st.divider()
st.caption("Made by Izad Khokhar for SMathHacks 2026")