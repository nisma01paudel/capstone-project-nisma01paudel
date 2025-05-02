# scripts/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_time_series(df, x_axis, y_axis, figsize=(10, 6)):
    """
    Plot a time series for the given data
    """
    plt.figure(figsize=figsize)
    plt.plot(df[x_axis], df[y_axis])
    plt.title(f"{y_axis} over time")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)
    # Return the plot object for Streamlit
    return plt.gcf()  # `gcf()` gets the current figure (plot)

def plot_time_series_double(df, col1, col2, x_axis, figsize=(10, 6)):
    """
    Plot two time series for comparison
    """
    plt.figure(figsize=figsize)
    plt.plot(df[x_axis], df[col1], label=col1)
    plt.plot(df[x_axis], df[col2], label=col2)
    plt.title(f"Comparison of {col1} and {col2}")
    plt.xlabel(x_axis)
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    # Return the plot object for Streamlit
    return plt.gcf()  # `gcf()` gets the current figure (plot)

def other_visualize(df, plot_type, x, y, group_by=None, cols=None, title=None):
    """
    Create various plots based on the selected plot type
    """
    plt.figure(figsize=(10, 6))
    
    if plot_type == "box":
        sns.boxplot(x=x, y=y, data=df)
    
    elif plot_type == "dist":
        sns.histplot(df[y], kde=True)
    
    elif plot_type == "scatter":
        sns.scatterplot(x=df[x], y=df[y], data=df)
    
    elif plot_type == "bar":
        if group_by:
            sns.barplot(x=x, y=y, data=df, hue=group_by)
        else:
            sns.barplot(x=x, y=y, data=df)
    
    elif plot_type == "pairplot":
        if cols:
            sns.pairplot(df[cols])
        else:
            sns.pairplot(df)
    
    plt.title(title)
    plt.tight_layout()
    # Return the plot object for Streamlit
    return plt.gcf()  # `gcf()` gets the current figure (plot)
