import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
from src.shared import config

def save_plot(filename):
    """Saves the current matplotlib plot."""
    if not os.path.exists(config.PLOTS_DIR):
        os.makedirs(config.PLOTS_DIR)
    plt.savefig(f"{config.PLOTS_DIR}/{filename}")
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
    # Static
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    save_plot(f"univariate_{column}.png")
    
    # Interactive
    if config.INCLUDE_INTERACTIVE_PLOTS:
        fig = px.histogram(df, x=column, marginal="box", title=f"Interactive Distribution of {column}")
        save_interactive_plot(fig, f"univariate_{column}_interactive")

def plot_bivariate(df, x, y):
    """Generates scatter plots (Static & Interactive)."""
    # Static
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f"{x} vs {y}")
    save_plot(f"bivariate_{x}_{y}.png")
    
    # Interactive
    if config.INCLUDE_INTERACTIVE_PLOTS:
        fig = px.scatter(df, x=x, y=y, trendline="ols", title=f"Interactive Scatter: {x} vs {y}")
        save_interactive_plot(fig, f"bivariate_{x}_{y}_interactive")

def plot_correlation_matrix(df, columns):
    """Generates a heatmap of the correlation matrix."""
    corr = df[columns].corr()
    
    # Static
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    save_plot("correlation_matrix.png")
    
    # Interactive
    if config.INCLUDE_INTERACTIVE_PLOTS:
        fig = px.imshow(corr, text_auto=True, title="Interactive Correlation Matrix")
        save_interactive_plot(fig, "correlation_matrix_interactive")

def plot_aqi_by_category(df):
    """Generates count plots for AQI categories."""
    # Static
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='AQI_Category', palette='viridis', hue='AQI_Category', legend=False)
    plt.title("Distribution of AQI Categories")
    save_plot("aqi_distribution.png")
    
    # Interactive
    if config.INCLUDE_INTERACTIVE_PLOTS:
        fig = px.bar(df['AQI_Category'].value_counts().reset_index(), 
                     x='AQI_Category', y='count', color='AQI_Category',
                     title="Interactive AQI Category Distribution")
        save_interactive_plot(fig, "aqi_distribution_interactive")

def plot_temporal_trends(df):
    """Generates monthly trend plots to identify cycles."""
    monthly_avg = df.groupby('Month')[['AQI']].mean().reset_index()
    
    # Static
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_avg, x='Month', y='AQI', marker='o')
    plt.title("Monthly AQI Trend (Cycle Identification)")
    plt.xticks(range(1, 13))
    save_plot("temporal_trends.png")
    
    # Interactive
    if config.INCLUDE_INTERACTIVE_PLOTS:
        fig = px.line(monthly_avg, x='Month', y='AQI', title="Interactive Monthly AQI Trends")
        save_interactive_plot(fig, "temporal_trends_interactive")

def plot_country_comparison(df):
    """Generates a comparison of AQI across countries."""
    country_avg = df.groupby('Country')['AQI'].mean().sort_values(ascending=False).head(15).reset_index()
    
    # Static
    plt.figure(figsize=(12, 8))
    sns.barplot(data=country_avg, x='AQI', y='Country', palette='rocket', hue='Country', legend=False)
    plt.title("Top 15 Countries by Average AQI (Comparative Analysis)")
    save_plot("country_comparison.png")
    
    # Interactive
    if config.INCLUDE_INTERACTIVE_PLOTS:
        fig = px.bar(country_avg, x='AQI', y='Country', orientation='h', color='AQI',
                     title="Interactive Comparative Analysis: Top 15 Countries by AQI")
        save_interactive_plot(fig, "country_comparison_interactive")
