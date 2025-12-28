import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import pandas as pd
from src.shared import config

def save_plot(filename):
    """Saves the current matplotlib plot."""
    if not os.path.exists(config.PLOTS_DIR):
        os.makedirs(config.PLOTS_DIR)
    plt.savefig(f"{config.PLOTS_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()

def save_interactive_plot(fig, filename):
    """Saves a plotly figure as an interactive HTML file."""
    if not os.path.exists(config.PLOTS_DIR):
        os.makedirs(config.PLOTS_DIR)
    path = os.path.join(config.PLOTS_DIR, f"{filename}.html")
    fig.write_html(path)
    return path

def plot_univariate(df, column):
    """Generates distribution plots (Static & Interactive)."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, color='steelblue')
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    save_plot(f"univariate_{column}.png")
    
    if config.INCLUDE_INTERACTIVE_PLOTS:
        fig = px.histogram(df, x=column, marginal="box", title=f"Interactive Distribution of {column}")
        save_interactive_plot(fig, f"univariate_{column}_interactive")

def plot_all_univariate(df, columns):
    """Generates univariate plots for all specified columns."""
    for col in columns:
        plot_univariate(df, col)

def plot_boxplot(df, column):
    """Generates boxplot for outlier visualization."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column], color='coral')
    plt.title(f"Boxplot of {column} (Outlier Detection)")
    plt.xlabel(column)
    save_plot(f"boxplot_{column}.png")

def plot_all_boxplots(df, columns):
    """Generates boxplots for all specified columns."""
    for col in columns:
        plot_boxplot(df, col)

def plot_bivariate(df, x, y):
    """Generates scatter plots (Static & Interactive)."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.5)
    plt.title(f"{x} vs {y}")
    save_plot(f"bivariate_{x}_{y}.png")
    
    if config.INCLUDE_INTERACTIVE_PLOTS:
        fig = px.scatter(df, x=x, y=y, trendline="ols", title=f"Interactive Scatter: {x} vs {y}")
        save_interactive_plot(fig, f"bivariate_{x}_{y}_interactive")

def plot_all_bivariate_vs_aqi(df, columns):
    """Generates bivariate plots for all columns vs AQI."""
    for col in columns:
        plot_bivariate(df, col, 'AQI')

def plot_correlation_matrix(df, columns):
    """Generates a heatmap of the correlation matrix."""
    corr = df[columns].corr()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Matrix of Pollutants and Meteorological Variables")
    save_plot("correlation_matrix.png")
    
    if config.INCLUDE_INTERACTIVE_PLOTS:
        fig = px.imshow(corr, text_auto=True, title="Interactive Correlation Matrix")
        save_interactive_plot(fig, "correlation_matrix_interactive")

def plot_aqi_by_category(df):
    """Generates count plots for AQI categories."""
    plt.figure(figsize=(10, 6))
    order = ['Good', 'Moderate', 'Unhealthy', 'Hazardous']
    existing_order = [cat for cat in order if cat in df['AQI_Category'].unique()]
    sns.countplot(data=df, x='AQI_Category', palette='viridis', order=existing_order)
    plt.title("Distribution of AQI Categories")
    plt.xlabel("AQI Category")
    plt.ylabel("Count")
    save_plot("aqi_distribution.png")
    
    if config.INCLUDE_INTERACTIVE_PLOTS:
        fig = px.bar(df['AQI_Category'].value_counts().reset_index(), 
                     x='AQI_Category', y='count', color='AQI_Category',
                     title="Interactive AQI Category Distribution")
        save_interactive_plot(fig, "aqi_distribution_interactive")

def plot_temporal_trends(df):
    """Generates monthly trend plots to identify cycles."""
    monthly_avg = df.groupby('Month')[['AQI']].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_avg, x='Month', y='AQI', marker='o', linewidth=2, color='darkblue')
    plt.title("Monthly AQI Trend (Seasonal Cycle Identification)")
    plt.xlabel("Month")
    plt.ylabel("Average AQI")
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    save_plot("temporal_trends.png")
    
    if config.INCLUDE_INTERACTIVE_PLOTS:
        fig = px.line(monthly_avg, x='Month', y='AQI', title="Interactive Monthly AQI Trends", markers=True)
        save_interactive_plot(fig, "temporal_trends_interactive")

def plot_pollutant_temporal_trends(df, pollutants):
    """Generates monthly trend plots for each pollutant."""
    for pol in pollutants:
        monthly_avg = df.groupby('Month')[[pol]].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=monthly_avg, x='Month', y=pol, marker='o', linewidth=2)
        plt.title(f"Monthly {pol} Trend (Seasonal Cycle)")
        plt.xlabel("Month")
        plt.ylabel(f"Average {pol}")
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        save_plot(f"temporal_{pol}.png")

def plot_country_comparison(df):
    """Generates a comparison of AQI across countries."""
    country_avg = df.groupby('Country')['AQI'].mean().sort_values(ascending=False).head(15).reset_index()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=country_avg, x='AQI', y='Country', palette='rocket')
    plt.title("Top 15 Countries by Average AQI (Comparative Analysis)")
    plt.xlabel("Average AQI")
    plt.ylabel("Country")
    save_plot("country_comparison.png")
    
    if config.INCLUDE_INTERACTIVE_PLOTS:
        fig = px.bar(country_avg, x='AQI', y='Country', orientation='h', color='AQI',
                     title="Interactive Comparative Analysis: Top 15 Countries by AQI")
        save_interactive_plot(fig, "country_comparison_interactive")

def plot_pollutant_by_country(df, pollutant):
    """Generates comparison of a specific pollutant across countries."""
    country_avg = df.groupby('Country')[pollutant].mean().sort_values(ascending=False).head(15).reset_index()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=country_avg, x=pollutant, y='Country', palette='magma')
    plt.title(f"Top 15 Countries by Average {pollutant}")
    plt.xlabel(f"Average {pollutant}")
    plt.ylabel("Country")
    save_plot(f"country_{pollutant}.png")

def plot_aqi_vs_all_pollutants(df, pollutants):
    """Generates a combined subplot of all pollutants vs AQI."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, pol in enumerate(pollutants):
        sns.scatterplot(data=df, x=pol, y='AQI', ax=axes[i], alpha=0.5)
        axes[i].set_title(f"{pol} vs AQI")
    
    plt.tight_layout()
    save_plot("all_pollutants_vs_aqi.png")

def plot_pairplot(df, columns):
    """Generates a pairplot for selected columns."""
    subset = df[columns + ['AQI_Category']].sample(min(1000, len(df)))  # Sample for performance
    g = sns.pairplot(subset, hue='AQI_Category', palette='Set2', diag_kind='kde')
    g.fig.suptitle("Pairplot of Pollutants by AQI Category", y=1.02)
    save_plot("pairplot_pollutants.png")

def plot_model_comparison(results_path):
    """Generates model comparison bar charts from results CSV."""
    try:
        df = pd.read_csv(results_path)
        
        # R-Squared Score Comparison
        plt.figure(figsize=(14, 8))
        r2_col = 'R-Squared Score'
        colors = ['green' if x > 0.99 else 'blue' if x > 0.95 else 'orange' if x > 0.9 else 'red' for x in df[r2_col]]
        plt.barh(df['Model'], df[r2_col], color=colors)
        plt.xlabel('R-Squared Score')
        plt.title('Model Comparison: R-Squared Score (Higher is Better)')
        plt.xlim(0, 1.05)
        for i, v in enumerate(df[r2_col]):
            plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=8)
        save_plot("model_comparison_r2.png")
        
        # Mean Squared Error Comparison
        plt.figure(figsize=(14, 8))
        plt.barh(df['Model'], df['Mean Squared Error'], color='steelblue')
        plt.xlabel('Mean Squared Error')
        plt.title('Model Comparison: Mean Squared Error (Lower is Better)')
        plt.xscale('log')
        save_plot("model_comparison_mse.png")
        
        # Mean Absolute Error Comparison
        plt.figure(figsize=(14, 8))
        plt.barh(df['Model'], df['Mean Absolute Error'], color='coral')
        plt.xlabel('Mean Absolute Error')
        plt.title('Model Comparison: Mean Absolute Error (Lower is Better)')
        plt.xscale('log')
        save_plot("model_comparison_mae.png")
        
        # Root Mean Squared Error Comparison
        plt.figure(figsize=(14, 8))
        plt.barh(df['Model'], df['Root Mean Squared Error'], color='purple')
        plt.xlabel('Root Mean Squared Error')
        plt.title('Model Comparison: Root Mean Squared Error (Lower is Better)')
        plt.xscale('log')
        save_plot("model_comparison_rmse.png")
        
    except Exception as e:
        print(f"Could not generate model comparison charts: {e}")

def plot_heatmap_pollutants_by_month(df, pollutants):
    """Generates a heatmap of pollutants by month."""
    monthly_data = df.groupby('Month')[pollutants].mean()
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(monthly_data.T, annot=True, cmap='YlOrRd', fmt='.1f')
    plt.title("Average Pollutant Levels by Month")
    plt.xlabel("Month")
    plt.ylabel("Pollutant")
    save_plot("heatmap_pollutants_monthly.png")
