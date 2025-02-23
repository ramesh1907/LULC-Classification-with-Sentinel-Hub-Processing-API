import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from affine import Affine
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask, shapes

from shapely.geometry import shape

from sentinelhub import (
    MimeType,
    MosaickingOrder,
    SentinelHubCatalog,
    SentinelHubRequest,
)

# Dictionary mapping class IDs to land cover types
class_dict = {
    1: 'water',
    2: 'mangrove',
    3: 'barren',
    4: 'urban',
    5: 'cropland',
    6: 'grassland',
    7: 'shrubland',
    8: 'forest'
}


def split_bbox(rows, cols, bounds):
    """
    Splits a given bounding box into smaller tiles.

    Parameters:
        rows (int): Number of rows to split into.
        cols (int): Number of columns to split into.
        bounds (list): Bounding box [minx, miny, maxx, maxy].

    Returns:
        list: A list of smaller bounding boxes.
    """
    minx, miny, maxx, maxy = bounds
    x_step = (maxx - minx) / cols
    y_step = (maxy - miny) / rows

    bboxs = []
    for j in range(rows):
        for i in range(cols):
            x1, x2 = minx + i * x_step, minx + (i + 1) * x_step
            y1, y2 = miny + j * y_step, miny + (j + 1) * y_step
            bboxs.append([x1, y1, x2, y2])

    return bboxs


def merge_rasters(raster_files, out_dir):
    """
    Merges multiple raster files into a single raster.

    Parameters:
        raster_files (list): List of file paths to raster images.
        out_dir (str): Output file path for the merged raster.

    Returns:
        numpy.ndarray: Merged raster image array.
    """
    datasets = [rasterio.open(file) for file in raster_files]
    merged_raster, merged_transform = merge(datasets)
    count, height, width = merged_raster.shape
    profile = datasets[0].profile
    profile.update(height=height, width=width, transform=merged_transform, nodata=0)

    # Close all opened datasets
    for src in datasets:
        src.close()

    with rasterio.open(out_dir, 'w', **profile) as dst:
        dst.write(merged_raster)

    return merged_raster


def merge_images(images, rows, cols):
    """
    Merges multiple image tiles into a single image.

    Parameters:
        images (list): List of image arrays.
        rows (int): Number of rows in the merged image.
        cols (int): Number of columns in the merged image.

    Returns:
        numpy.ndarray: Merged image.
    """
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    resized_images = [cv2.resize(img, (max_width, max_height)) for img in images]

    row_images = [np.hstack(resized_images[i * cols:(i + 1) * cols]) for i in range(rows)]
    merged_image = np.vstack(row_images[::-1])
    return merged_image


def get_raster_image(time_interval, cloud_coverage, roi_bbox, image_size, config, satellite, raster_dir):
    """
    Retrieves raster images from Sentinel Hub.

    Parameters:
        time_interval (tuple): Date range for image retrieval (start_date, end_date).
        cloud_coverage (int): Maximum allowed cloud coverage percentage.
        roi_bbox (BBox): Bounding box for the region of interest.
        image_size (tuple): Dimensions of the requested image (width, height).
        config (SHConfig): Sentinel Hub API configuration.
        satellite (DataCollection): Sentinel satellite data collection.
        raster_dir (str): Directory to store the retrieved raster.

    Returns:
        SentinelHubRequest: Request object for image retrieval.
    """
    evalscript_all_bands = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B02","B03","B04","B05","B08","B09","B11","B12"],
                    units: "DN"
                }],
                output: {
                    bands: 8,
                    sampleType: "INT16"
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B02,
                    sample.B03,
                    sample.B04,
                    sample.B05,
                    sample.B08,
                    sample.B09,
                    sample.B11,
                    sample.B12];
        }
    """
    return SentinelHubRequest(
        data_folder=raster_dir,
        evalscript=evalscript_all_bands,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=satellite,
                time_interval=time_interval,
                maxcc=cloud_coverage / 100,
                mosaicking_order=MosaickingOrder.LEAST_CC
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=roi_bbox,
        size=image_size,
        config=config,
    )


def compute_indices(image, file=True):
    """
    Computes vegetation and water indices for a given raster image.

    Parameters:
        image (numpy.ndarray or str): Image array or file path to raster.
        file (bool): If True, modifies the raster file in place.

    Returns:
        numpy.ndarray: Image with computed indices.
    """

    def indices(image):
        red = image[2].astype('float32')
        green = image[1].astype('float32')
        nir = image[4].astype('float32')
        swir = image[6].astype('float32')

        ndvi = (nir - red) / (nir + red)
        gcvi = (nir / green) - 1
        ndwi = (green - nir) / (green + nir)
        savi = ((nir - red) / (nir + red + 0.5)) * 1.5
        ndbi = (swir - nir) / (swir + nir)

        return np.stack([ndvi, gcvi, ndwi, savi, ndbi])

    if file:
        with rasterio.open(image) as ras_src:
            ras_image = ras_src.read()
            profile = ras_src.profile

            indices_img = indices(ras_image)
            indices_image = np.concatenate([ras_image, indices_img])

        profile.update(count=indices_image.shape[0])
        with rasterio.open(image, 'w', **profile) as dst:
            dst.write(indices_image)
    else:
        ras_image = image
        indices_img = indices(ras_image)
        indices_image = np.concatenate([ras_image, indices_img])

    return indices_image


def images_check(config, collection, roi_bbox, time_interval, cloud_coverage):
    """
    Checks if satellite images are available for the given parameters.

    Parameters:
        config (SHConfig): Sentinel Hub API configuration.
        collection (DataCollection): Sentinel-2 data collection.
        roi_bbox (BBox): Bounding box for the region of interest.
        time_interval (tuple): Time range for satellite data retrieval.
        cloud_coverage (int): Maximum allowable cloud coverage percentage.

    Returns:
        bool: True if images are found, False otherwise.
    """
    catalog = SentinelHubCatalog(config=config)
    search_iterator = catalog.search(
        collection=collection,
        bbox=roi_bbox,
        time=time_interval,
        filter={"op": "<", "args": [{"property": "eo:cloud_cover"}, cloud_coverage]},
        filter_lang="cql2-json",
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []},
    )

    results = list(search_iterator)
    print(f"{len(results)} images found for the given time period and cloud cover")
    return bool(results)


def save_image(image, save_path, roi_bounds, polygon):
    """
    Saves a classified image to a GeoTIFF file.

    Parameters:
        image (numpy.ndarray): Image array.
        save_path (str): Directory to store the raster.
        roi_bounds (list): Bounding box [minx, miny, maxx, maxy].
        polygon (Polygon): shapely geometry polygon

    Returns:
        None
    """
    height, width = image.shape
    minx, miny, maxx, maxy = roi_bounds
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    mask = geometry_mask([polygon], transform=transform, invert=True, out_shape=(height, width))
    masked_image = np.where(mask, image, 0)

    meta_data = {
        "driver": "GTiff",
        "count": 1,
        "width": width,
        "height": height,
        "crs": 'EPSG:4326',
        "transform": transform,
        "dtype": "uint8",
        "nodata": 0,
        "compress": "lzw",
        "tiled": True
    }

    with rasterio.open(save_path, 'w', **meta_data) as dst :
        dst.write(masked_image,1)

    print("Image classified and saved")

    return None



def vector_dataset(raster_path, save_path):
    """
    Converts a classified raster image into a vector dataset.

    Parameters:
        raster_path (str): Directory of the raster.
        save_path (str): Directory to store the raster.

    Returns:
        None
    """
    with rasterio.open(raster_path) as src :
        transform = src.transform
        ras_arr = src.read(1)

    df = pd.DataFrame([(val, shape(shp)) for shp, val in shapes(ras_arr, transform=transform)])
    df.columns = ['Class_id', 'geometry']
    df = df[df['Class_id'] != 0]
    df['Class'] = df['Class_id'].map(class_dict)
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'], crs="EPSG:4326")
    gdf.to_file(save_path)

    return None
