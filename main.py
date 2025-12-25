import pandas as pd
from sklearn.model_selection import train_test_split
from src.shared.config import POLLUTANTS, METEOROLOGICAL, TARGET_COLUMN
from src.infrastructure.data_loader import load_air_quality_data
from src.use_cases.data_cleaning import preprocess_pipeline
from src.use_cases.feature_engineering import calculate_aqi_index, encode_target
from src.use_cases.temporal_analysis import extract_temporal_features
from src.presentation.visualizer import (
    plot_univariate, plot_bivariate, 
    plot_correlation_matrix, plot_aqi_by_category,
    plot_temporal_trends, plot_country_comparison
)
from src.infrastructure.model_factory import run_all_models

def prepare_data():
    """
    Orchestrates data loading, cleaning, and preparation.
    """
    df = load_air_quality_data()
    df = calculate_aqi_index(df, POLLUTANTS)
    df = preprocess_pipeline(df, POLLUTANTS)
    df = extract_temporal_features(df)
    df, _ = encode_target(df)
    return df

def perform_eda(df):
    """
    Runs EDA and saves plots (Static and Interactive).
    """
    plot_aqi_by_category(df)
    plot_univariate(df, 'PM2.5')
    plot_bivariate(df, 'Temperature', 'AQI')
    plot_temporal_trends(df)
    plot_country_comparison(df)
    plot_correlation_matrix(df, POLLUTANTS + METEOROLOGICAL + ['AQI'])

def split_and_train(df):
    """
    Splits data and trains models.
    """
    features = POLLUTANTS + METEOROLOGICAL
    x = df[features]
    y = df['AQI']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    return run_all_models(x_train, x_test, y_train, y_test)

def main():
    """
    Main entry point for the analysis pipeline.
    """
    print("Starting Air Quality Analysis Pipeline...")
    df = prepare_data()
    print("Data prepared. Starting EDA...")
    perform_eda(df)
    print("EDA completed. Training models...")
    results = split_and_train(df)
    for name, data in results.items():
        print(f"Model: {name}, Metrics: {data['metrics']}")

if __name__ == "__main__":
    main()
