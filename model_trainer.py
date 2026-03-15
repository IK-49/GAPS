from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score

gdf = gpd.read_file('./data_sources/processed_grid.gpkg')
FEATURES = ['DRVAL1', 'DRVAL2', 'obstruction_distance', 'depth_range']

x = gdf[FEATURES].values
y = gdf['target'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=67, stratify=y)

#smote = SMOTE(random_state=67)
#x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

model = RandomForestClassifier(n_estimators=100, random_state=67, class_weight='balanced', min_samples_leaf=2)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

y_proba = model.predict_proba(x_test)[:, 1]

print(classification_report(y_test, predictions, target_names=['No Wreck', 'Wreck']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")