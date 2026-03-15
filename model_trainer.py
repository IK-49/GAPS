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

# Confirm ML capture rate at 1% threshold
threshold_ml = gdf['wreck_prob'].quantile(0.99)
top_ml = gdf[gdf['wreck_prob'] >= threshold_ml]
ml_capture = top_ml[top_ml['target']==1].shape[0]
print(f"Top 1% by ML probability captures {ml_capture}/{total_wrecks} known wrecks ({100*ml_capture/total_wrecks:.1f}%)")

# Export full grid with wreck_prob, risk_score, and dredge_overlap for app.py
gdf.to_file('./data_sources/grid_with_risk.gpkg', driver='GPKG')
print("Exported → grid_with_risk.gpkg")