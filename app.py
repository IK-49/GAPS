import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd

st.set_page_config(page_title="Wreck Risk Map", layout="wide")

st.title("🚢 Graveyard of the Atlantic — Undiscovered Wreck Risk Map")
st.markdown("Identifying high-risk zones for undiscovered shipwrecks along the North Carolina coast using NOAA nautical chart data.")

# load data from gpkg
@st.cache_data
def load_data():
    gdf = gpd.read_file('./data_sources/grid_with_risk.gpkg')
    gdf = gdf.dropna(subset=['DRVAL1', 'DRVAL2', 'depth_range', 'obstruction_distance'])
    gdf = gdf.to_crs(epsg=4326)
    return gdf

with st.spinner("Loading NOAA nautical chart data..."): # loading icon
    gdf = load_data()       

# Controls sidebar
st.sidebar.header("Controls")

top_percentile = st.sidebar.slider(
    "Show Top % High Risk Tiles",
    min_value=1,
    max_value=20,
    value=10,
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
        for _, row in high_risk.iterrows():
            centroid = row.geometry.centroid
            folium.CircleMarker(
                location=[centroid.y, centroid.x],
                radius=3,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=row['wreck_prob'],
                tooltip=f"ML Probability: {row['wreck_prob']:.3f} | Depth: {row['DRVAL1']:.0f}m"
            ).add_to(m)

    if show_wrecks:
        for _, row in total_wrecks.iterrows():
            centroid = row.geometry.centroid
            folium.CircleMarker(
                location=[centroid.y, centroid.x],
                radius=6,
                color='yellow',
                fill=True,
                fill_color='yellow',
                fill_opacity=0.9,
                tooltip="Confirmed wreck"
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