# WiFi CSI Sensing Project

This repository contains machine learning pipelines for processing synthetic WiFi Channel State Information (CSI) data to perform two distinct sensing tasks:
1. **Classification:** Detecting human occupancy and movement.
2. **Regression:** Estimating heart rate (BPM).

The project leverages amplitude and phase data from WiFi signals to build robust predictive models.

## Dataset
Both notebooks utilize the `synthetic_csi_data_1440min.parquet` dataset. The dataset includes:
- **15 Amplitude Features** (`amp_0` to `amp_14`)
- **15 Phase Features** (`phase_0` to `phase_14`)
- **Target Labels:** `occupancy` (binary), `movement` (binary), and `heart_rate_bpm` (continuous).

## Notebooks Overview

### 1. `classification.ipynb` (Occupancy and Movement Detection)
This notebook implements a multi-target classification pipeline to simultaneously predict binary occupancy and movement states from the CSI signals.

**Key Steps:**
- **Data Preprocessing:** Standardizes amplitude features and unwraps phase features to remove discontinuities.
- **Windowing & Feature Extraction:** Applies a sliding window approach (window size = 512, stride = 128) to extract summary statistics (mean, standard deviation, min, max) across the features, resulting in a 120-dimensional feature vector per window.
- **Modeling:** Trains a `MultiOutputClassifier` backed by a `RandomForestClassifier` (200 estimators).
- **Results:** Achieves near-perfect test accuracy (>99.8%) and F1-scores for both occupancy and movement detection. Includes confusion matrix visualizations for detailed evaluation.

### 2. `randforest4-49.ipynb` (Heart Rate Estimation)
This notebook focuses on predicting continuous heart rate (BPM) from the CSI data when a subject is present in the room.

**Key Steps:**
- **Data Filtering & Segmentation:** Filters the dataset for periods where the room is occupied (`occupancy == 1`) and the heart rate is within a valid human range (30-180 BPM). It groups continuous valid periods into individual segments.
- **Windowing & Feature Extraction:** Applies sliding windows (size = 512, stride = 128) only within valid, continuous segments to avoid mixing disjoint timeframes. Extracts 124 statistical features per window.
- **Modeling:** Compares two regression models:
  - `RandomForestRegressor`
  - `XGBRegressor` (GPU-accelerated using `tree_method="hist"` and `device="cuda"`)
- **Results:** The models successfully capture the continuous heart rate signal. On the test set, the Random Forest model achieves a Mean Absolute Error (MAE) of ~4.49 BPM, while XGBoost yields an MAE of ~4.78 BPM.

## Requirements
To run these notebooks, you will need the following libraries:
- `numpy`, `pandas`, `matplotlib`
- `scikit-learn`
- `xgboost` (for GPU-accelerated gradient boosting)
- `torch` (for device availability checks and random seed setting)

## Setup
1. Ensure your environment has the necessary dependencies installed (`pip install -r requirements.txt` if provided, or install the libraries above manually).
2. Place the `synthetic_csi_data_1440min.parquet` dataset in the expected directory (or update the `DATA_PATH` variable in the notebooks).
3. Run the cells sequentially to preprocess the data, train the models, and view the evaluation metrics.
