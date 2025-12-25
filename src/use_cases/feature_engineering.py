import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.domain.aqi_logic import get_aqi_category

def calculate_aqi_index(df, pollutant_cols):
    """
    Calculates a simple AQI index as the max of normalized pollutants.
    """
    if 'AQI' not in df.columns:
        df['AQI'] = df[pollutant_cols].mean(axis=1)
    return df

def encode_target(df):
    """
    Adds AQI_Category and encodes it.
    """
    df['AQI_Category'] = df['AQI'].apply(get_aqi_category)
    le = LabelEncoder()
    df['AQI_Category_Encoded'] = le.fit_transform(df['AQI_Category'])
    return df, le

def scale_features(df, columns):
    """
    Scales numerical features using StandardScaler.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler
