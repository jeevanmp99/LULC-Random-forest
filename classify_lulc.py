"""
Supervised Land Use/Land Cover (LULC) Classification using Random Forest.
Integrates multispectral imagery (Landsat) with auxiliary topographic predictors (DEM, Slope).
"""

import argparse
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def parse_arguments():
    """Parses command-line arguments with sensible default paths."""
    parser = argparse.ArgumentParser(
        description="Supervised LULC classification using Random Forest and auxiliary terrain data."
    )
    parser.add_argument(
        "--landsat",
        type=str,
        default="data/sample/composite16_cliped.tif",
        help="Path to Landsat multi-band GeoTIFF",
    )
    parser.add_argument(
        "--train_shp",
        type=str,
        default="data/sample/TRAIN_2016_3.shp",
        help="Path to training point shapefile or GeoPackage",
    )
    parser.add_argument(
        "--dem",
        type=str,
        default="data/sample/dem_resamp.tif",
        help="Path to aligned DEM raster",
    )
    parser.add_argument(
        "--slope",
        type=str,
        default="data/sample/slope_resamp.tif",
        help="Path to aligned Slope raster",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="LULC_2016_classified.tif",
        help="Destination path for output GeoTIFF",
    )
    parser.add_argument(
        "--class_col",
        type=str,
        default="Class",
        help="Target attribute column name in shapefile",
    )
    return parser.parse_args()


def load_and_verify_data(landsat_path, dem_path, slope_path, shp_path, class_col):
    """Verifies file existence, spatial alignment, and CRS consistency."""
    for file_path in [landsat_path, dem_path, slope_path, shp_path]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Missing required input file: {file_path}")

    samples = gpd.read_file(shp_path)
    if class_col not in samples.columns:
        raise ValueError(
            f"Target column '{class_col}' missing from {shp_path}. Found: {list(samples.columns)}"
        )

    with rasterio.open(landsat_path) as src_l:
        meta = src_l.meta.copy()
        crs = src_l.crs
        height, width = src_l.height, src_l.width

    with rasterio.open(dem_path) as src_d, rasterio.open(slope_path) as src_s:
        if (src_d.height, src_d.width) != (height, width) or (src_s.height, src_s.width) != (height, width):
            raise ValueError(
                f"Dimension mismatch! Landsat is ({height}, {width}), "
                f"DEM is ({src_d.height}, {src_d.width}), "
                f"Slope is ({src_s.height}, {src_s.width}). Ensure rasters are pre-aligned."
            )

    # Reproject points if CRS does not match imagery
    if samples.crs != crs:
        print(f"[Notice] Reprojecting training vector from {samples.crs} to {crs}...")
        samples = samples.to_crs(crs)

    return samples, meta


def extract_point_features(landsat_path, dem_path, slope_path, samples, class_col):
    """Samples pixel values beneath ground-truth points without polygon edge contamination."""
    coords = [(geom.x, geom.y) for geom in samples.geometry]

    with rasterio.open(landsat_path) as src_l, \
         rasterio.open(dem_path) as src_d, \
         rasterio.open(slope_path) as src_s:

        landsat_pts = list(src_l.sample(coords))
        dem_pts = list(src_d.sample(coords))
        slope_pts = list(src_s.sample(coords))
        num_bands = src_l.count

    train_data = []
    labels = samples[class_col].values

    for i in range(len(coords)):
        l_vals = landsat_pts[i]
        d_val = dem_pts[i][0]
        s_val = slope_pts[i][0]

        # Filter out NaN or coordinates outside valid data zones
        if not np.isnan(d_val) and not np.isnan(s_val):
            train_data.append(list(l_vals) + [d_val, s_val, labels[i]])

    columns = [f"Band_{i+1}" for i in range(num_bands)] + ["DEM", "Slope", "Class"]
    df = pd.DataFrame(train_data, columns=columns)

    if df.empty:
        raise ValueError("Extraction yielded 0 valid points. Ensure shapefile points overlap raster extents.")

    print(f"Extracted {len(df)} samples across {len(columns) - 1} predictors.")
    print("Class sample counts:\n", df["Class"].value_counts().sort_index())
    return df


def train_classifier(df):
    """Performs stratified train/test split and trains a balanced Random Forest model."""
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratified split ensures minority classes maintain representative proportions in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining Random Forest model (n_estimators=200, class_weight='balanced')...")
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=20,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\nOverall Validation Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

    return clf


def predict_and_export_raster(clf, landsat_path, dem_path, slope_path, output_path, meta):
    """
    Infers classes in 512x512 tile windows to keep memory usage constant,
    masks unmeasured background pixels as 0, and writes a single-band GeoTIFF.
    """
    out_meta = meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint8",
        "nodata": 0,
        "compress": "lzw"
    })

    print(f"\nClassifying full scene via windowed inference. Writing to {output_path}...")
    with rasterio.open(landsat_path) as src_l, \
         rasterio.open(dem_path) as src_d, \
         rasterio.open(slope_path) as src_s, \
         rasterio.open(output_path, "w", **out_meta) as dst:

        landsat_nodata = src_l.nodata
        block_size = 512

        for row in range(0, src_l.height, block_size):
            for col in range(0, src_l.width, block_size):
                w = min(block_size, src_l.width - col)
                h = min(block_size, src_l.height - row)
                window = Window(col, row, w, h)

                tile_l = src_l.read(window=window)
                tile_d = src_d.read(1, window=window)
                tile_s = src_s.read(1, window=window)

                # Identify valid ground pixels (ignore zero-fill boundaries and NaNs)
                valid_mask = (tile_l != 0).any(axis=0) & (~np.isnan(tile_d)) & (~np.isnan(tile_s))
                if landsat_nodata is not None:
                    valid_mask &= (tile_l[0] != landsat_nodata)

                tile_pred = np.zeros((h, w), dtype=np.uint8)

                if np.any(valid_mask):
                    flat_l = tile_l[:, valid_mask].T
                    flat_d = tile_d[valid_mask][:, np.newaxis]
                    flat_s = tile_s[valid_mask][:, np.newaxis]

                    tile_X = np.hstack([flat_l, flat_d, flat_s])
                    tile_pred[valid_mask] = clf.predict(tile_X).astype(np.uint8)

                dst.write(tile_pred, 1, window=window)

    print("Raster export completed successfully.")


def main():
    args = parse_arguments()

    # 1. Validation & Input Loading
    samples, meta = load_and_verify_data(
        args.landsat, args.dem, args.slope, args.train_shp, args.class_col
    )

    # 2. Extract Training Features
    df = extract_point_features(
        args.landsat, args.dem, args.slope, samples, args.class_col
    )

    # 3. Model Training & Evaluation
    clf = train_classifier(df)

    # 4. Memory-Safe Inference & GeoTIFF Export
    predict_and_export_raster(
        clf, args.landsat, args.dem, args.slope, args.output, meta
    )


if __name__ == "__main__":
    main()
