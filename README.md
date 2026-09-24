# GAPS: Graveyard of the Atlantic Predictive Safeguarding

An end-to-end geospatial hazard assessment pipeline and machine learning model designed to evaluate maritime navigation risk along North Carolina's Outer Banks using historical NOAA shipwreck records, bathymetric depth charts, and automated spatial feature engineering.

---

## Technical Architecture

> **Data Pipeline:**
> 1. Raw NOAA Maritime & Bathymetry Data
> 2. Spatial Discretization (`geopandas`, `pandas`, `Shapely`)
> 3. Feature Pipeline (Shoal Proximity, Depth Gradients, Coordinate Grids)
> 4. Predictive Classification (`scikit-learn` Random Forest)
> 5. Interactive UI (`Streamlit` Dashboard & `Folium` Mapping)

---

## Engineering Highlights

* **Geospatial Feature Engineering:** Cleaned, reprojected (EPSG:4326), and transformed raw NOAA coastal point datasets into structured feature arrays using `geopandas` and `pandas`.
* **Predictive Hazard Scoring:** Implemented a `Random Forest Classifier` via `scikit-learn` to identify risk-factor correlations across historical wreck densities and bathymetric depth contours.
* **Percentile-Grid Architecture:** Addressed static-chart baseline limitations by precalculating spatial risk grids into a normalized relative hazard index, avoiding live inference bottlenecks.
* **Interactive UI:** Built a lightweight `Streamlit` dashboard allowing users to input oceanic coordinates and visualize regional hazard layers in real time.

---

## Tech Stack

* **Language:** Python 3.10+
* **Data & Machine Learning:** scikit-learn, pandas, NumPy, geopandas, Shapely
* **Visualization & Interface:** Streamlit, Folium, Matplotlib

---

## Local Setup & Reproduction

### 1. Clone Repository & Create Virtual Environment
```bash
git clone [https://github.com/your-username/GAPS.git](https://github.com/your-username/GAPS.git)
cd GAPS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
