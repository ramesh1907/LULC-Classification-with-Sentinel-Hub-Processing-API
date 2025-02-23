import geopandas as gpd
import joblib
import numpy as np
import os

from math import ceil
from shapely.ops import unary_union

from sentinelhub import (
    BBox,
    bbox_to_dimensions,
    CRS,
    DataCollection,
    SHConfig
)

from utils_old import split_bbox, merge_images, get_raster_image, compute_indices, images_check, save_image, vector_dataset

import warnings
warnings.filterwarnings('ignore')


def classify_image(config, collection, dates, cloud_percentage, pixel_size, roi_file, base_path):
    """
    Classifies Land Use Land Cover (LULC) using a pre-trained model on Sentinel-2 imagery.

    Parameters:
        config (SHConfig): Sentinel Hub API configuration.
        collection (DataCollection): Sentinel-2 data collection.
        dates (tuple): Time range for satellite data retrieval.
        cloud_percentage (int): Maximum allowable cloud coverage percentage.
        pixel_size (int): Satellite image resolution in meters.
        roi_file (str): Path to the Region of Interest (ROI) shapefile.
        base_path (str): Base directory for storing outputs.
    """

    roi_gdf = gpd.read_file(roi_file)
    roi = unary_union(roi_gdf['geometry'])
    bounds = list(roi_gdf.total_bounds)
    roi_bbox = BBox(bbox=bounds, crs=CRS.WGS84)

    check = images_check(config, collection, roi_bbox, dates, cloud_percentage)
    if not check:
        return

    image_size = bbox_to_dimensions(roi_bbox, resolution=pixel_size)
    width, height = image_size
    cols, rows = ceil(width / 2500), ceil(height / 2500)

    if cols > 1 or rows > 1:
        split_bboxs = split_bbox(rows, cols, bounds)
        rgb_split_images = []
        for split_bounds in split_bboxs:
            bbox = BBox(bbox=split_bounds, crs=CRS.WGS84)
            size = bbox_to_dimensions(bbox, resolution=pixel_size)
            ras_split_img = get_raster_image(dates, cloud_percentage, bbox, size, config, collection)
            ras_split_img = ras_split_img.get_data()[0]
            rgb_split_images.append(ras_split_img)
        
        raster_image = merge_images(rgb_split_images, rows, cols)
    else:
        raster_image = get_raster_image(dates, cloud_percentage, roi_bbox, image_size, config, collection)
        raster_image = raster_image.get_data()[0]

    # Process and classify the raster image
    raster_image = raster_image.transpose(2, 0, 1)
    indices_image = compute_indices(raster_image, file=False).transpose(1, 2, 0)
    height, width, num_bands = indices_image.shape

    bands = ["B02", "B03", "B04", "B05", "B08", "B09", "B11", "B12", 
             "NDVI", "GCVI", "NDWI", "SAVI", "NDBI"]

    # Load trained model and scaler
    model_path = os.path.join(base_path, "model", "LULC_rf_model.pkl")
    scaler_path = os.path.join(base_path, "model", "LULC_rf_model_scaler.pkl")

    rf_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Flatten and preprocess the image for classification
    flat_arr = indices_image.reshape(-1, len(bands))
    flat_arr[np.isinf(flat_arr) | np.isnan(flat_arr)] = 0
    scaled_arr = scaler.transform(flat_arr)

    # Predict classes
    classified_arr = rf_model.predict(scaled_arr)
    classified_image = classified_arr.reshape(height, width).astype(int)

    # Save the classified raster
    raster_out_path = os.path.join(base_path, "tif", "classified_image.tif")
    save_image(classified_image, raster_out_path, bounds, roi)

    # Convert raster to vector dataset
    lulc_vector_path = os.path.join(base_path, "gpkg", "lulc_vector_dataset.gpkg")
    vector_dataset(raster_out_path, lulc_vector_path)

    print(f"Classification completed")


if __name__ == "__main__":
    config = SHConfig()
    config.sh_client_id = os.getenv("SH_CLIENT_ID", "your-client-id-here")
    config.sh_client_secret = os.getenv("SH_CLIENT_SECRET", "your-client-secret-here")

    satellite_collection = DataCollection.SENTINEL2_L2A
    time_interval = ("2025-01-01", "2025-02-01")
    cloud_coverage = 10
    resolution = 10  # Pixel resolution for image acquisition

    base_path = r"C:\lulc_classification"
    shapefile = os.path.join(base_path, "gpkg", "viti_island.gpkg")

    classify_image(config, satellite_collection, time_interval, cloud_coverage, resolution, shapefile, base_path)

