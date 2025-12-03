import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

sns.set_style("whitegrid")
sns.set_context("talk") 

def plot_results(y_test, predictions, save_dir='outputs'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    residuals = y_test - predictions

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions, color="#2c3e50", alpha=0.6, edgecolor=None)
    
    min_val = min(min(y_test), min(predictions))
    max_val = max(max(y_test), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linestyle='--', linewidth=2)
    
    plt.xlabel("Actual Price (₹)")
    plt.ylabel("Predicted Price (₹)")
    plt.title("Model Accuracy: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color="#3498db")
    plt.title("Error Distribution (Residuals)")
    plt.xlabel("Prediction Error (Rs.)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/residuals_dist.png", dpi=150)
    plt.close()

def plot_exploratory_data_analysis(df, save_dir='outputs'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if 'car_age' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='car_age', y='selling_price', data=df, color="#8e44ad", alpha=0.5)
        plt.title('Depreciation Curve: Car Age vs Price')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/car_age_vs_price.png", dpi=150)
        plt.close()

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='YlGnBu', fmt=".2f", linewidths=0.5)
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/correlation_heatmap.png", dpi=150)
        plt.close()