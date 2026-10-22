# Land Use/Land Cover (LULC) Classification using Random Forest

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A supervised, pixel-based Land Use/Land Cover (LULC) classification pipeline using Scikit-Learn's Random Forest classifier on multispectral Landsat imagery. Auxiliary terrain variables (Digital Elevation Model and Slope) are integrated alongside spectral bands to resolve spectral confusion across heterogeneous landscapes.

The repository includes a self-contained, pre-aligned sample dataset in `data/sample/` so the complete pipeline executes out-of-the-box.

---

## Repository Structure

```text
.
├── data/
│   └── sample/
│       ├── composite16_cliped.tif   # Multi-band Landsat surface reflectance clip
│       ├── dem_resamp.tif           # Aligned DEM raster (~30m)
│       ├── slope_resamp.tif         # Aligned Slope raster (degrees)
│       └── TRAIN_2016_3.shp         # Point vector with ground-truth classes
├── classify_lulc.py                 # Core extraction, training, and inference script
├── requirements.txt                 # Pinned dependencies
└── README.md
```

---

## Installation & Setup

Spatial dependencies (`rasterio`, `geopandas`) rely on underlying system libraries (GDAL, PROJ, GEOS). Installing via Conda is recommended to avoid binary wheel mismatches:

```bash
# Recommended installation via Conda
conda create -n lulc-rf python=3.10 -c conda-forge geopandas rasterio scikit-learn pandas matplotlib
conda activate lulc-rf
```

Alternatively, install via pip:

```bash
pip install -r requirements.txt
```

---

## Quick Start (Running with Included Sample Data)

The script defaults to the bundled subset in `data/sample/` so you can verify the workflow immediately:

```bash
python classify_lulc.py
```

**Outputs Generated:**

- **`LULC_2016_corrected.tif`**: Single-band, LZW-compressed classified GeoTIFF matching the input raster grid.
- **Validation Metrics**: Overall accuracy, class-wise precision, recall, and F1-scores printed to the console.
- **Feature Importance Plot**: Horizontal bar chart showing relative Gini predictor contributions.
- **Visual Preview**: Color-mapped diagnostic display of classified terrain (background/NoData masked).

---

## Methodology & Workflow

1. **CRS & Extent Verification:** Checks that all raster datasets share identical dimensions, geotransforms, and projections, automatically reprojecting point vectors if CRS mismatches occur.
2. **Point-Based Feature Extraction:** Extracts exact spectral and terrain values under coordinate points using `rasterio.sample`, eliminating edge contamination common to polygon boundary pixels.
3. **Stratified Sampling:** Uses stratified train/test splitting to preserve minority class representation.
4. **Balanced Random Forest Training:** Employs cost-sensitive weighting (`class_weight='balanced'`) and tree depth limits to reduce majority-class bias and noise memorization.
5. **Memory-Safe Windowed Inference:** Processes large rasters in $512 \times 512$ tile windows via `rasterio.windows.Window`, keeping RAM usage constant regardless of scene size.
6. **NoData & Border Masking:** Detects unmeasured background pixels and zero-fill boundaries, assigning them to Class 0 (NoData) rather than forcing arbitrary classifications.

---

## Dataset Schema & Class Distribution

The ground-truth dataset contains $N = 597$ point samples distributed across six land-cover classes, evaluated using an 80/20 stratified split ($n_{\text{test}} = 120$):

| Class ID | Land Cover Type (Example) | Total Points (N=597) | Proportion | Test Support ($n_{\text{test}}$) |
|:--------:|---------------------------|:--------------------:|:----------:|:--------------------------------:|
| 1 | Barren land | 108 | 18.1% | 22 |
| 2 | Fallow land | 108 | 18.1% | 22 |
| 3 | Vegetation | 129 | 21.6% | 26 |
| 4 | Built-up / Urban | 109 | 18.3% | 22 |
| 5 | Water | 33 | 5.5% | 6 |
| 6 | Mountain-Forest | 110 | 18.4% | 22 |

---

## Model Evaluation & Performance

The classifier was evaluated on the independent holdout test split ($n = 120$).

### Performance Summary

- **Overall Accuracy:** 95.00%
- **Macro Average F1-Score:** 0.955
- **Weighted Average F1-Score:** 0.950

### Stratified Validation Report

| Class ID | Precision | Recall | F1-Score | Support ($n_{\text{test}}$) |
|:--------:|:---------:|:------:|:--------:|:---------------------------:|
| 1 | 0.870 | 0.909 | 0.889 | 22 |
| 2 | 0.950 | 0.864 | 0.905 | 22 |
| 3 | 0.963 | 1.000 | 0.981 | 26 |
| 4 | 0.957 | 1.000 | 0.978 | 22 |
| 5 | 1.000 | 1.000 | 1.000 | 6 |
| 6 | 1.000 | 0.955 | 0.977 | 22 |

> **Methodological Note on Class Imbalance:**
> Water (Class 5) represents an underrepresented minority class in the study region (33 total points; 6 test samples). While the model achieved an F1-score of 1.000 on these 6 instances, individual point errors in small test samples heavily impact class metrics, so this figure should not be over-interpreted. Using `class_weight='balanced'` makes the ensemble penalize misclassifications proportionally, to prevent minority class omission.

> **Validation Caveat:**
> The holdout split is random at the point level. Spatially adjacent samples may appear in both train and test sets, so reported accuracy may be optimistic. See *Potential Pipeline Extensions* for spatial block cross-validation.

---

## Feature Importance Analysis

Relative predictor contributions were evaluated using Mean Decrease in Impurity (Gini Importance):

| Rank | Feature | Importance Score | Remote Sensing Interpretation |
|:----:|---------|:----------------:|-------------------------------|
| 1 | Band 7 (SWIR-2) | ~0.147 | Highest discriminator. Highly sensitive to moisture differences, soil exposure, and impervious surfaces. |
| 2 | DEM | ~0.143 | Second highest predictor. Resolves elevational gradients; no single terrain variable exceeds 20% importance. |
| 3 | Band 1 (Coastal/Blue) | ~0.130 | Strong discriminator for clear vs. turbid water bodies and dark target surfaces. |
| 4 | Band 4 (Red) | ~0.125 | Captures the chlorophyll absorption trough, isolating vegetative from non-vegetative terrain. |
| 5 | Band 5 (NIR) | ~0.112 | Key marker for internal leaf structure reflectance and open-water absorption. |
| 6 | Band 6 (SWIR-1) | ~0.106 | Moisture sensitivity and built-up structure differentiation. |
| 7 | Band 2 (Blue) | ~0.088 | Secondary optical reflectance band. |
| 8 | Slope | ~0.079 | Differentiates steep uncultivated terrain from flat agricultural/urban floodplains. |
| 9 | Band 3 (Green) | ~0.068 | Lowest relative contribution due to spectral collinearity with adjacent optical bands. |

> Gini importance can be biased toward high-cardinality continuous features and is affected by correlated predictors. Treat rankings as indicative, not definitive.

---

## Input Data Specifications (Custom Scenes)

When applying this script to custom scenes, ensure input datasets meet these geometric requirements:

- **Identical Coordinate Reference System (CRS):** All rasters must share the exact same projected CRS (e.g., UTM Zone).
- **Grid Harmonization:** Auxiliary layers (`dem_resamp.tif`, `slope_resamp.tif`) must be resampled to match Landsat pixel resolution (~30 m) and snapped to the master scene's bounding box and geotransform.
- **Training Schema:** Point shapefiles must contain an integer attribute field named `Class` corresponding to the target categories.

---

## Potential Pipeline Extensions

- [ ] Add spectral indices directly into the feature stack (NDVI, NDWI, NDBI, MNDWI).
- [ ] Implement spatial block cross-validation to explicitly test for spatial autocorrelation.
- [ ] Export class probability/confidence rasters using `clf.predict_proba` to visualize classification uncertainty.
- [ ] Incorporate radar backscatter (e.g., Sentinel-1 VV/VH) for structural surface differentiation.
