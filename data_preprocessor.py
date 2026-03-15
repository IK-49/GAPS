import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box
import requests
import io

# File Paths
gdb_path    = "./data_sources/NOAA_coastal_data.gdb"
sand_path   = "./data_sources/sand_resources.geojson"
output_path = "./data_sources/processed_grid.gpkg"

FEATURES = [
    # 'shore_distance',
    'DRVAL1',
    'obstruction_distance',
    'DRVAL2',
    'depth_range',
    # 'sand_distance',
]

# Load wreck layers
wrecks = gpd.read_file(gdb_path, layer='Coastal_Wreck_point').to_crs(epsg=4326)
wreck_area = gpd.read_file(gdb_path, layer='Coastal_Wreck_area').to_crs(epsg=4326)

def fetch_awois_wrecks():
    import urllib3
    urllib3.disable_warnings()
    bbox = "-76.5,34.5,-75.0,36.5"
    url = (
        "https://encdirect.noaa.gov/arcgis/rest/services/encdirect/enc_harbour/MapServer/36/query"
        f"?where=1=1&geometry={bbox}&geometryType=esriGeometryEnvelope"
        "&inSR=4326&spatialRel=esriSpatialRelIntersects"
        "&outFields=*&returnGeometry=true&f=geojson"
    )
    r = requests.get(url, verify=False)
    return gpd.read_file(io.StringIO(r.text)).to_crs(epsg=4326)

awois_wrecks = fetch_awois_wrecks()
print(f"AWOIS wrecks fetched: {len(awois_wrecks)}")

potential_wreck_targets = gpd.GeoDataFrame(
    pd.concat([wrecks, wreck_area, awois_wrecks], ignore_index=True), # combines wreck points and wreck areas into a single dataframe
    crs="EPSG:4326"
)

# Load environmental factors used for sand buffers / dredge zones
sand = gpd.read_file(sand_path).to_crs(epsg=4326)

# Creates a 500x500m grid in 32618 then reprojects to 4326
wrecks_m = wrecks.to_crs(epsg=32618)
xmin, ymin, xmax, ymax = wrecks_m.total_bounds

resolution = 500  # true 500m in EPSG:32618

x_steps = np.arange(xmin, xmax, resolution)
y_steps = np.arange(ymin, ymax, resolution)

squares = []
for x in x_steps:
    for y in y_steps:
        squares.append(box(x, y, x + resolution, y + resolution))

tiles = gpd.GeoDataFrame(geometry=squares, crs="EPSG:32618").to_crs(epsg=4326)

# Label tiles (1 if intersects known wreck/obstruction)
joined_df = gpd.sjoin(tiles, potential_wreck_targets, how='left', predicate='intersects')
joined_df['target'] = joined_df['index_right'].notnull().astype(int)
joined_df = joined_df.groupby(joined_df.index).first()
joined_df = gpd.GeoDataFrame(joined_df, geometry='geometry', crs='EPSG:4326')

# Go back to meters projection
joined_df_m = joined_df.to_crs(epsg=32618)

LOADERS = {}

def load_shore_distance(grid):
    # Calculate shore distance
    coastline = gpd.read_file(gdb_path, layer='Coastal_Coastline_line').to_crs(epsg=32618)
    return grid.geometry.centroid.distance(coastline.union_all())
LOADERS['shore_distance'] = load_shore_distance

def load_DRVAL1(grid): # min depth
    min_depth = gpd.read_file(gdb_path, layer='Coastal_Depth_Area').to_crs(epsg=32618)
    temp = gpd.sjoin(grid[['geometry']], min_depth[['DRVAL1', 'geometry']], how='left')
    return temp.groupby(temp.index).first()['DRVAL1'].fillna(-1).astype(float)
LOADERS['DRVAL1'] = load_DRVAL1

def load_DRVAL2(grid): # max depth
    min_depth = gpd.read_file(gdb_path, layer='Coastal_Depth_Area').to_crs(epsg=32618)
    temp = gpd.sjoin(grid[['geometry']], min_depth[['DRVAL2', 'geometry']], how='left')
    return temp.groupby(temp.index).first()['DRVAL2'].fillna(-1).astype(float)
LOADERS['DRVAL2'] = load_DRVAL2

def load_depth_range(grid): # max depth - min depth
    return (grid['DRVAL2'] - grid['DRVAL1']).clip(lower=0)
LOADERS['depth_range'] = load_depth_range

def load_sand_distance(grid): # 
    sand = gpd.read_file(sand_path).to_crs(epsg=32618)
    return joined_df_m.geometry.centroid.distance(sand.union_all())
LOADERS['sand_distance'] = load_sand_distance

def load_obstruction_distance(grid):
    obs = gpd.read_file(gdb_path, layer='Coastal_Obstruction_point').to_crs(epsg=32618)
    return grid.geometry.centroid.distance(obs.union_all())
LOADERS['obstruction_distance'] = load_obstruction_distance

for feat in FEATURES:
    joined_df_m[feat] = LOADERS[feat](joined_df_m)

#sand_m = sand.to_crs(epsg=32618)
#sand_buffer = sand_m.buffer(5000).union_all()  # 5km buffer around actual dredge zones
#joined_df_m = joined_df_m[joined_df_m.geometry.intersects(sand_buffer)].copy()

# Information
print(f"Total tiles in coastal zone: {len(joined_df_m)}")
print(f"Confirmed wreck tiles (target=1): {joined_df_m['target'].sum()}")
print("\nFeature means by class:")
print(joined_df_m.groupby('target')[FEATURES].mean())

# Export
joined_df_m.to_file(output_path, driver="GPKG")
print(f"\nProcessed grid exported to: {output_path}")

print(joined_df_m['target'].value_counts())