import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd

st.set_page_config(page_title="Wreck Risk Map", layout="wide")

TOUR_STEPS = [
    ("🗺️ Welcome to GAPS", "This tool maps high-probability zones for undiscovered shipwrecks along the North Carolina Outer Banks — the 'Graveyard of the Atlantic'. Click Next to learn how it works."),
    ("📊 The Grid", "The study area is divided into 91,980 tiles at 500×500m resolution. Each tile is scored by a Random Forest model trained on 102 confirmed wreck locations from NOAA nautical chart data."),
    ("🔴 Red Circles", "Red circles are high-risk tiles — the top N% most likely to contain an undiscovered wreck. Opacity indicates confidence: brighter red = higher ML probability."),
    ("🟡 Yellow Dots", "Yellow dots are confirmed wreck locations from NOAA ENC charts and the NOAA ENC Direct API. These were used to train and validate the model."),
    ("🎚️ The Slider", "Use the sidebar slider to adjust how many tiles are shown. At 10%, the model captures 87% of known wrecks while reducing the search area by 90%."),
    ("📈 The Metrics", "Total Tiles Analyzed = full study area. High Risk Zones = tiles above your threshold. Known Wrecks Captured = how many confirmed wrecks fall inside high-risk zones."),
    ("✅ Ready", "You're all set. Zoom into the Outer Banks to explore high-risk clusters. Hover over any circle for depth and probability details."),
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

st.title("🚢 Graveyard of the Atlantic — Undiscovered Wreck Risk Map")
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
    **Data Sources**
    - NOAA Electronic Navigational Charts (ENC) — wreck points, wreck areas, depth areas, obstruction points
    - NOAA ENC Direct API — additional harbour-scale wreck points (106 additional records)
    - Study area: North Carolina coast (Outer Banks / Pamlico Sound)
    - Grid: 91,980 tiles at 500×500m resolution

    **Features**
    - `DRVAL1`: Minimum water depth — strongest signal (37m at wrecks vs 162m elsewhere)
    - `DRVAL2`: Maximum water depth — strong signal (214m vs 542m)
    - `depth_range`: Seafloor depth variability (DRVAL2 − DRVAL1)
    - `obstruction_distance`: Distance to nearest charted obstruction

    **Model**
    - Random Forest Classifier (300 trees, class_weight='balanced')
    - ROC-AUC: 0.63 | Positive samples: 102 out of 91,980 tiles
    - Top 10% of tiles by ML probability captures 87% of known wrecks

    **Limitations**
    - 102 confirmed wreck tiles is a severe class imbalance
    - Additional data (historical shipping lanes, seafloor composition) would improve accuracy
    """)