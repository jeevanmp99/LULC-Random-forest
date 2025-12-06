🌍 Land Use/Land Cover (LULC) Classification using Random Forest

### **Python | Rasterio | Scikit-Learn | GeoPandas**

This project performs a **supervised Land Use/Land Cover (LULC) classification** using a **Random Forest machine learning model** applied to multispectral Landsat imagery.
Digital Elevation Model (DEM) and slope layers are incorporated as **auxiliary predictor variables** to improve classification accuracy.

The workflow extracts pixel values under point-based training samples, builds a classification model, evaluates its performance, and generates a classified LULC raster.
This methodology is well suited for applications in:

* Remote sensing
* Environmental and ecological mapping
* Hydrological studies
* Urban and regional planning
* Agricultural monitoring

---

# ✅ Features

* Load multispectral imagery (Landsat)
* Integrate DEM and slope layers as additional predictors
* Extract training data from point-based shapefile
* Train a Random Forest classifier
* Perform train/test accuracy assessment
* Predict LULC for the full study area
* Export a classified GeoTIFF raster
* Visualize the final LULC map

---

# 📦 Prerequisites

### ✔ Required Python Libraries

```bash
pip install geopandas rasterio numpy pandas scikit-learn matplotlib
```

---

# 📁 Input Data Requirements

All input datasets must follow these conditions:

---

## **1. Coordinate Reference System (CRS)**

The following must use the **same CRS**:

* Landsat composite raster
* DEM raster
* Slope raster
* Training point shapefile

**CRS mismatches will result in incorrect pixel extraction.**

---

## **2. Spatial Resolution Requirements**

DEM and slope layers must be:

* **Resampled** to match the Landsat pixel resolution (~30 m)
* **Aligned** so that all rasters share the same grid structure

---

## **3. Spatial Extent**

DEM and slope must be:

* Clipped to the Landsat extent
* Having matching **rows**, **columns**, and **geotransform**

This ensures all layers stack correctly.

---

# 🎯 Training Sample Requirements (Point-Based Sampling)

The classification workflow uses **point training samples**, where each point represents a known land-cover class.

### ✔ Training Data Format

* **Shapefile (.shp)**
* **Geometry: POINT**
* **Attribute field:**

  * **Class** → integer code representing the land-cover class
* **CRS identical to all rasters**

### Example Attribute Table

| Point_ID | Class          | Geometry    |
| -------- | -------------- | ----------- |
| 1        | 1 (Water)      | POINT(x, y) |
| 2        | 3 (Vegetation) | POINT(x, y) |
| 3        | 5 (Barren)     | POINT(x, y) |

Each point is used to extract:

```
Band1, Band2, Band3, Band4, Band5, Band6, Band7, DEM, Slope
```

### Resulting Training Table Format

| Class      | Band1 | Band2 | Band3 | Band4 | Band5 | Band6 | Band7 | DEM | Slope |
| ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ----- |
| Barren     | 123   | 98    | 76    | 45    | 23    | 12    | 5     | 255 | 14    |
| Vegetation | 45    | 63    | 88    | 123   | 145   | 110   | 95    | 300 | 9     |
| Water      | 10    | 20    | 25    | 30    | 15    | 10    | 8     | 201 | 2     |

This feature matrix is built automatically by the script.

---

# 🧭 Why Use Point Samples?

* Avoids boundary errors from polygons
* Ideal for pixel-based machine learning workflows
* Ensures exact spatial correspondence between training data and raster grid
* Faster and more memory-efficient
* Works cleanly with Rasterio and NumPy

---

# 📁 Recommended Project Structure

```
project/
├── data/
│   ├── landsat_composite.tif
│   ├── dem_resampled.tif
│   ├── slope_resampled.tif
│   ├── training_samples.shp
│   └── ...
├── src/
│   └── classify_lulc.py
└── README.md
```

---

# ▶️ Running the Classification Script

### 1. Update file paths:

```python
landsat_path = '/path/to/landsat.tif'
train_shp_path = '/path/to/training_points.shp'
dem_path = '/path/to/dem_resampled.tif'
slope_path = '/path/to/slope_resampled.tif'
```

### 2. Execute the script:

```bash
python classify_lulc.py
```

### 3. Output Files Generated:

* **LULC_2016.tif** → Classified raster
* Accuracy metrics (precision, recall, F1-score)
* Matplotlib visualization of classification

---

# 📊 Output Example

* GeoTIFF classified map saved to disk
* Displayed map using a color-coded scheme
* Accuracy evaluation for model validation

---

# 🧠 Why Random Forest?

Random Forest is a powerful classifier for remote sensing because:

* Handles nonlinear relationships
* Works well with high-dimensional data
* Robust against noise
* Requires no assumptions about data distribution
* Provides high accuracy for mixed Land Cover data

---

# 💡 Potential Extensions

* Add vegetation/water indices (NDVI, NDWI, SAVI)
* Add terrain-based predictors (TWI, TRI, HAND)
* Add GLCM texture metrics
* Tune model using GridSearchCV
* Generate probability/confidence maps
* Add lineament density as a predictor
* Perform temporal LULC change analysis

---
