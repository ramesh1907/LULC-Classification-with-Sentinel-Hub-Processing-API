# LULC-Classification-with-Sentinel-Hub-Processing-API
Mapping the extent of land use and land cover (LULC) categories over time is essential for better environmental monitoring, urban planning, and nature protection. This project focuses on training and fine-tuning a machine learning model to classify satellite images into 8 LULC categories. This project performs **Land Use Land Cover (LULC) classification** using Sentinel-2 imagery obtained via the **Sentinel Hub Processing API**. The classification is done using a **Random Forest Classifier**, and the final classified image is converted into vector format.

## Workflow
### 1. Data Acquisition
- Downloaded **Sentinel-2 imagery** for the **Region of Interest (ROI)** at **10m resolution**  with less than **10% cloud cover** using Sentinel Hub Processing API.
- Applied necessary pre-processing steps such as cloud masking and band selection.

### 2. Feature Extraction
- Computed spectral indices such as NDVI, NDWI, GCVI, SAVI, and NDBI using Sentinel-2 bands and incorporated them into the Sentinel-2 image.
- Extracted spectral features at ground truth points from the Sentinel-2 image and applied Standard Scaler to normalize them for training.

### 3. Model Training
- Trained a **Random Forest Classifier** using training data.
- Evaluated the model using:
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-score, Accuracy)

### 4. Image Classification
- Applied the trained model to classify the Sentinel-2 image.

### 5. Post-Processing
- Converted the classified raster image to **vector format** using `rasterio.features.shapes`.
- Exported the final vector data for GIS applications.

## Technologies Used
- **Python**
- **Sentinel Hub Processing API**
- **Scikit-learn** (for Random Forest classification)
- **Rasterio** (for raster processing)
- **Geopandas** (for vector processing)

## Installation
1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Update the Sentinel Hub API credentials in the lulc_training and lulc_classification files.
2. Run the script to create and train lulc model:
   ```bash
   python lulc_training.py
   ```
3. Classify image and Convert classified raster to vector:
   ```bash
   python lulc_classification.py
   ```

## Results
- The final classified map is stored as a classified_image.tif file.
- The vectorized land cover classes are saved as a shapefile.

## Acknowledgments
- **Sentinel Hub** for providing satellite imagery.
- **Open-source libraries** such as Scikit-learn, Rasterio, and Geopandas.

---
🚀 **Happy Mapping!**


