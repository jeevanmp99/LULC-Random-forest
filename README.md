🌍 Land Use/Land Cover (LULC) Classification using Random Forest

### **Python | Rasterio | Scikit-Learn | GeoPandas**

This project performs a **supervised Land Use/Land Cover (LULC) classification** using a **Random Forest classifier** on multispectral Landsat imagery, combined with **DEM and slope** as additional predictor variables.
The workflow extracts training pixels from a ground-truth shapefile, trains a machine learning model, evaluates accuracy, and generates a classified LULC raster.

This workflow is designed for **remote sensing, environmental analysis, hydrological studies, and urban planning applications**.

---

# ✅ Features

* Load multispectral satellite imagery (Landsat)
* Load DEM & slope rasters as auxiliary features
* Extract training data from vector shapefile
* Train Random Forest model
* Split training/testing data for accuracy assessment
* Predict LULC for entire study area
* Export classified GeoTIFF raster
* Visualize outputs

---

# 📦 Prerequisites

### ✔ **Software / Libraries**

Install required Python libraries:

```bash
pip install geopandas rasterio numpy pandas scikit-learn matplotlib
```

### ✔ **Input Data Requirements**

All input datasets must meet the following conditions:

#### **1. Same Coordinate Reference System (CRS)**

* Landsat raster
* DEM raster
* Slope raster
* Training shapefile

All must share **identical CRS** (e.g., EPSG:4326 or UTM Zone).

#### **2. Same Spatial Resolution (for DEM & Slope)**

* DEM and slope must be **resampled** to match Landsat pixel size
  (e.g., ~30 meters)

#### **3. Same Spatial Extent / Alignment**

* DEM and slope should be **clipped and aligned** to the Landsat image
* No shift in pixel alignment
* The rasters must have the **same pixel dimensions (rows/columns)**

#### **4. Training Shapefile Requirements**

Your training dataset must:

* Contain point or polygon samples
* Include a **Class** field (integer labels)
* Fall entirely within the raster extent

---

# 📁 Input File Structure Example

```
project/
├── composite16_cliped.tif      # Landsat image
├── TRAIN_2016_3.shp            # Training samples
├── dem_resamp.tif              # DEM resampled to Landsat resolution
├── slope_resamp.tif            # Slope map (derived from DEM)
└── classify.py                 # Main script
```

---

# ▶️ **How to Run the Script**

### 1. Update file paths in the script:

```python
landsat_path = '/path/to/composite16_cliped.tif'
train_shp_path = '/path/to/TRAIN_2016_3.shp'
dem_path = '/path/to/dem_resamp.tif'
slope_path = '/path/to/slope_resamp.tif'
```

### 2. Run the script:

```
python classify.py
```

### 3. Output:

* Classified raster saved as:
  `LULC_2016.tif`
* Accuracy report printed in terminal
* Visualization displayed with matplotlib

---

# 📊 Output Example

* **GeoTIFF** raster
* Classified map displayed using a color scheme

---

# 📘 Code Explanation (High-Level)

The script:

1. **Loads multi-band raster data** (Landsat)
2. **Loads DEM & Slope** as additional predictors
3. **Reads training shapefile**
4. **Extracts pixel-based feature vectors**
5. **Trains Random Forest classifier**
6. **Evaluates accuracy** (precision, recall, F1-score)
7. **Predicts classes for all pixels**
8. **Saves a new classified raster**

---

# 🧠 Why Random Forest?

* Handles high-dimensional satellite data well
* Resistant to noise
* Works without strict assumptions
* Provides high accuracy for LULC classification

---

# 💡 Optional Enhancements

You may extend this repository by adding:

* NDVI, NDWI, SAVI as additional bands
* Texture metrics (GLCM)
* TWI, TRI, or lineament density layers
* Hyperparameter tuning (GridSearchCV)
* Confusion matrix visualization
* Exporting probability maps
