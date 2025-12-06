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

Here is the **corrected and improved description** for your GitHub README, based on your new requirement:

✅ **Training data must be POINTS only**
✅ **Left side = Land class label**
✅ **Right side = Pixel values (Bands, DEM, Slope) extracted at each point**


---

# 🎯 **4. Training Sample Requirements (Point-Based Sampling)**

The classification workflow uses **point-based training data**, where each point represents a known land-cover class.
During processing, the script extracts the raster pixel values under each point and links them to the land-cover class.

### 🔹 **Training Sample Format**

* File type: **Shapefile (.shp)**
* Geometry: **Points only**
* Projection: **Same CRS as the rasters**
* Attribute field:

  * **Class** → integer representing the land cover class

Example:

| Point_ID | Class          | Geometry    |
| -------- | -------------- | ----------- |
| 1        | 1 (Water)      | POINT(x, y) |
| 2        | 3 (Vegetation) | POINT(x, y) |
| 3        | 5 (Barren)     | POINT(x, y) |

---

# 📌 **How the Training Data Will Be Used**

For every point, the script extracts:

```
Band1, Band2, Band3, Band4, Band5, Band6, Band7, DEM, Slope
```

So the extracted dataset (training table) will look like:

| Class      | Band1 | Band2 | Band3 | Band4 | Band5 | Band6 | Band7 | DEM | Slope |
| ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ----- |
| Barren     | 123   | 98    | 76    | 45    | 23    | 12    | 5     | 255 | 14    |
| Vegetation | 45    | 63    | 88    | 123   | 145   | 110   | 95    | 300 | 9     |
| Water      | 10    | 20    | 25    | 30    | 15    | 10    | 8     | 201 | 2     |

**Left side = land class (label)**
**Right side = pixel values extracted from rasters**

This table is automatically created during the script execution.

---

# ✔ Training Data Requirements Summary

### Your shapefile must contain:

* **Points** (NOT polygons)
* A field named **Class**
* Coordinates within the raster boundary
* Same CRS as:

  * Landsat/Sentinel composite
  * DEM
  * Slope raster

### Points should be well distributed across:

* Water
* Vegetation
* Built-up
* Agriculture
* Barren
* Forest
* Any classes you include

---

# 🧭 Why Point Samples?

Point-based sampling is preferred because:

* It avoids polygon boundary errors
* Ensures 1-to-1 pixel–class mapping
* Works perfectly with rasterio & machine learning models
* Faster, cleaner, less RAM usage

---

# 📁 Input File Structure Example

```
project/
├── data/
│   ├── satellite_image.tif        # Multispectral raster
│   ├── dem_resampled.tif          # DEM matched to satellite resolution
│   ├── slope_resampled.tif        # Slope derived from DEM
│   ├── training_samples.shp       # LULC training data
│   └── (other optional layers)
├── src/
│   └── lulc_classification.py     # Main script
└── README.md

---

# ▶️ **How to Run the Script**

### 1. Update file paths in the script:

```python
landsat composite image path 
training sample path
dem path 
slope path
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
