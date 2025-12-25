import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
from src.shared.config import PLOTS_DIR

def save_plot(filename):
    """Saves the current matplotlib plot."""
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    plt.savefig(f"{PLOTS_DIR}/{filename}")
    plt.close()

def save_interactive_plot(fig, filename):
    """Saves a plotly figure as an interactive HTML file."""
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    path = os.path.join(PLOTS_DIR, f"{filename}.html")
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
    fig = px.bar(df['AQI_Category'].value_counts().reset_index(), 
                 x='AQI_Category', y='count', color='AQI_Category',
                 title="Interactive AQI Category Distribution")
    save_interactive_plot(fig, "aqi_distribution_interactive")
