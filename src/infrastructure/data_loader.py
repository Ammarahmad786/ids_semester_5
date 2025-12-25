import pandas as pd
from src.shared.config import DATA_PATH

def load_air_quality_data(path=DATA_PATH):
    """
    Loads the air quality dataset from the specified CSV path.
    """
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return None
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        return None

def get_basic_info(df):
    """
    Returns basic information about the dataframe.
    """
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict()
    }
