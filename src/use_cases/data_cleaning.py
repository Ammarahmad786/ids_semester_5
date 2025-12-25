import pandas as pd
import numpy as np

def clean_missing_values(df):
    """
    Handles missing values using interpolation for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
    return df

def fill_remaining_na(df):
    """
    Fills any remaining NAs after interpolation with median.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

def drop_categorical_na(df):
    """
    Drops rows with missing categorical data.
    """
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    return df.dropna(subset=categorical_cols)

def handle_outliers_iqr(df, column):
    """
    Caps outliers using the IQR method for a specific column.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

def clean_all_outliers(df, columns):
    """
    Apply IQR capping to all specified columns.
    """
    for col in columns:
        df = handle_outliers_iqr(df, col)
    return df

def preprocess_pipeline(df, pollutant_cols):
    """
    Main pipeline for data cleaning.
    """
    df = clean_missing_values(df)
    df = fill_remaining_na(df)
    df = drop_categorical_na(df)
    df = clean_all_outliers(df, pollutant_cols)
    return df
