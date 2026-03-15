import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

gdf = gpd.read_file('./data_sources/processed_grid.gpkg')

FEATURES = ['DRVAL1', 'DRVAL2', 'depth_range', 'obstruction_distance']

gdf = gdf.dropna(subset=FEATURES)

# Calculate risk score using MinMaxScaler to scale the features
scaler = MinMaxScaler()
scaled = scaler.fit_transform(gdf[FEATURES])

gdf['risk_score'] = (
    (1 - scaled[:, 0]) * 0.45 +  # DRVAL1; shallow = high risk
    (1 - scaled[:, 1]) * 0.30 +  # DRVAL2; low max depth = high risk
    (1 - scaled[:, 2]) * 0.10 +  # depth_range; low depth range
    (1 - scaled[:, 3]) * 0.15    # obstruction_distance; close to obstructions = high risk
)

# Validation
print("Mean risk score at known wreck tiles:", gdf[gdf['target']==1]['risk_score'].mean())
print("Mean risk score at non-wreck tiles:  ", gdf[gdf['target']==0]['risk_score'].mean())

threshold = gdf['risk_score'].quantile(0.90)
top_tiles = gdf[gdf['risk_score'] >= threshold]
wreck_capture = top_tiles[top_tiles['target']==1].shape[0]
total_wrecks = gdf[gdf['target']==1].shape[0]
print(f"Top 10% of tiles captures {wreck_capture}/{total_wrecks} known wrecks ({100*wreck_capture/total_wrecks:.1f}%)")

# Random Forest Classification
print("\n── Random Forest ──")

x = gdf[FEATURES].values
y = gdf['target'].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=67, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    min_samples_leaf=2,
    random_state=67,
    n_jobs=-1
)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]

print(classification_report(y_test, predictions, target_names=['No Wreck', 'Wreck']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Export layers

# Add ML probabilities to grid
gdf['wreck_prob'] = model.predict_proba(gdf[FEATURES].values)[:, 1]

# Full grid with risk score (for heatmap coloring)
gdf.to_file('./data_sources/grid_with_risk.gpkg', driver='GPKG')

# Top 5% highest risk tiles as centroids (circles on map)
top5 = gdf[gdf['wreck_prob'] >= gdf['wreck_prob'].quantile(0.95)].copy()
top5['geometry'] = top5.geometry.centroid
top5.to_file('./data_sources/high_risk_points.gpkg', driver='GPKG')

# Known wreck locations for reference overlay
known_wrecks = gdf[gdf['target'] == 1].copy()
known_wrecks['geometry'] = known_wrecks.geometry.centroid
known_wrecks.to_file('./data_sources/known_wrecks.gpkg', driver='GPKG')

print(f"\nHigh risk points (top 5%): {len(top5)} tiles")
print("Exported → grid_with_risk.gpkg")
print("Exported → high_risk_points.gpkg")
print("Exported → known_wrecks.gpkg")