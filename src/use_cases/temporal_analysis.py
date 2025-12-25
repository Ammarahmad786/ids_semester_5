import pandas as pd

def extract_temporal_features(df, date_col='Date'):
    """
    Extracts month, season, and day of week from the date column.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df['Month'] = df[date_col].dt.month
    df['DayOfWeek'] = df[date_col].dt.dayofweek
    
    # Simple seasonal mapping
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        if month in [3, 4, 5]: return 'Spring'
        if month in [6, 7, 8]: return 'Summer'
        return 'Autumn'
    
    df['Season'] = df['Month'].apply(get_season)
    return df

def aggregate_by_month(df):
    """
    Aggregates pollution levels by month to identify cycles.
    """
    return df.groupby('Month')[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']].mean()

def compare_countries(df, top_n=10):
    """
    Compares the average AQI across different countries.
    """
    return df.groupby('Country')['AQI'].mean().sort_values(ascending=False).head(top_n)
