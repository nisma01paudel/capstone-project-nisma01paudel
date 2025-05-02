# scripts/data_utils.py

import pandas as pd

def load_climate_data(file_path):
    """
    Load the climate data CSV
    """
    return pd.read_csv(file_path)

def load_flood_data(file_path):
    """
    Load flood data, for example, raster or other flood-related datasets
    """
    # You can use libraries like `rasterio` if working with .img files
    return file_path
