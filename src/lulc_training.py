import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rasterio

from affine import Affine
from glob import glob
from math import ceil
from rasterio.merge import merge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sentinelhub import (
    BBox,
    bbox_to_dimensions,
    CRS,
    DataCollection,
    SHConfig
)

from utils import split_bbox, merge_rasters, get_raster_image, compute_indices, vector_dataset

import warnings
warnings.filterwarnings('ignore')

def train_lulc_model(dates, satellite_data, cloud_coverage, shapefile, training_data, base_path):
    """
    Trains a Land Use Land Cover (LULC) classification model using Sentinel-2 satellite imagery and ground truth data.
    
    Parameters:
        dates (tuple): Time range for satellite data retrieval.
        satellite_data (DataCollection): Sentinel-2 data collection.
        cloud_coverage (int): Maximum allowable cloud coverage percentage.
        shapefile (str): Path to the Region of Interest (ROI) shapefile.
        training_data (str): Path to ground truth training data (GeoPackage).
        base_path (str): Base directory for storing outputs.
    """
    # Create necessary directories
    os.makedirs(os.path.join(base_path, "tif"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "gpkg"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "png"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "csv"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "model"), exist_ok=True)

    # Load ROI and define bounding box
    roi_gdf = gpd.read_file(shapefile)
    bounds = list(roi_gdf.total_bounds)
    roi_bbox = BBox(bbox=bounds, crs=CRS.WGS84)
    
    # Determine image size and split if necessary
    image_size = bbox_to_dimensions(roi_bbox, resolution=10)
    width, height = image_size
    cols, rows = ceil(width / 2500), ceil(height / 2500)

    # Generate raster data
    raster_dir = os.path.join(base_path, "tif")
    raster_path = os.path.join(raster_dir, "raster_image.tif")

    if cols > 1 or rows > 1:
        split_bboxs = split_bbox(rows, cols, bounds)
        for split_bounds in split_bboxs:
            bbox = BBox(bbox=split_bounds, crs=CRS.WGS84)
            size = bbox_to_dimensions(bbox, resolution=10)
            ras_split_img = get_raster_image(dates, cloud_coverage, bbox, size, config, satellite_data, raster_dir)
            ras_split_img.get_data(save_data=True)[0]

        raster_files = glob(os.path.join(raster_dir, "**", "*.tiff"))
        raster_image = merge_rasters(raster_files, raster_path)
    else:
        raster_image = get_raster_image(dates, cloud_coverage, roi_bbox, image_size, config, satellite_data)
        raster_image.get_data(save_data=True)[0]
        raster_files = glob(os.path.join(base_path, "**", "*.tiff"))
        raster_image = merge_rasters(raster_files, raster_path)

    # Compute spectral indices
    indices_image = compute_indices(raster_path)

    # Extract RGB channels and normalize for visualization
    red, green, blue = raster_image[3], raster_image[2], raster_image[1]
    rgb_image = np.stack([red, green, blue], axis=-1)
    normalize_rgb = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    
    plt.figure(figsize=(10, 10))
    plt.imshow(normalize_rgb, extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
    plt.axis("off")
    plt.title("RGB Image")
    plt.savefig(os.path.join(base_path, "png", "rgb_image.png"))
    plt.close()

    # Load ground truth data
    data_gdf = gpd.read_file(training_data)
    coords = [(x, y) for x, y in zip(data_gdf["geometry"].x, data_gdf["geometry"].y)]

    # Extract spectral signatures from raster data
    with rasterio.open(raster_path) as src:
        ras_arr = src.read().transpose(1, 2, 0)
        height, width, _ = ras_arr.shape
        no_data_val = src.nodata
        bands_extract = np.stack([x for x in src.sample(coords)])

    bands = ["B02", "B03", "B04", "B05", "B08", "B09", "B11", "B12", "NDVI", "GCVI", "NDWI", "SAVI", "NDBI"]
    data_gdf[bands] = bands_extract

    # Split data into train and test sets
    X = data_gdf[bands].values
    y = data_gdf['class_id'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(base_path, "model", "LULC_rf_model_scaler.pkl"))

    # Define scoring metrics for model evaluation
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': make_scorer(precision_score, average='macro', zero_division=1),
        'recall_macro': make_scorer(recall_score, average='macro', zero_division=1),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=1)
    }

    # Hyperparameter tuning
    cv_params = {
        'max_depth': [2, 3, 4, 5, None],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [2, 3, 4],
        'max_features': [2, 3, 4],
        'n_estimators': [75, 100, 125, 150]
    }

    rf_model = RandomForestClassifier(random_state=42)
    rf_cv = GridSearchCV(rf_model, cv_params, scoring=scoring, cv=5, refit='f1_macro', n_jobs=-1)
    rf_cv.fit(X_train_scaled, y_train)

    # Validate model
    y_pred = rf_cv.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.xticks(rotation=30)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(base_path, "png", "confusion_matrix.png"))
    plt.close()

    # Report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(2)
    report_df.to_csv(os.path.join(base_path, "csv", "classification_report.csv"))

    # Save trained model
    joblib.dump(rf_cv, os.path.join(base_path, "model", "LULC_rf_model.pkl"))



if __name__ == "__main__" :
    config = SHConfig()
    config.sh_client_id = "554be802-fc4b-47d7-a2c2-15aa73e46cae"
    config.sh_client_secret = "YdhKashxgseNhF0G2usMXGNeSKRMSTSZ"

    time_interval = (f"2024-01-01", f"2025-01-01")
    cloud_coverage = 10
    resolution = 10
    collection = DataCollection.SENTINEL2_L2A

    base_path = r"C:\assignment"
    data_file = rf"{base_path}\gpkg\LULC_ground_data.gpkg"
    roi_file = rf"{base_path}\gpkg\viti_island.gpkg"

    train_lulc_model(time_interval, collection, cloud_coverage, roi_file, data_file, base_path)