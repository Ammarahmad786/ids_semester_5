"""
Configuration settings for the Air Quality Analysis project.
Contains file paths, column names, and global constants.
"""

# File Paths
DATA_PATH = "data/global_air_quality_data_10000.csv"
RESULTS_DIR = "results"
PLOTS_DIR = f"{RESULTS_DIR}/plots"
MODELS_DIR = f"{RESULTS_DIR}/models"

# Column Definitions
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
METEOROLOGICAL = ["Temperature", "Humidity", "Wind Speed"]
CATEGORICAL = ["City", "Country"]
TEMPORAL = "Date"

# Target Variable
TARGET_COLUMN = "AQI"

# AQI Thresholds (Simplified for encoding)
AQI_CATEGORIES = {
    "Good": (0, 50),
    "Moderate": (51, 100),
    "Unhealthy": (101, 150),
    "Hazardous": (151, 500)
}

# Model settings
TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42
