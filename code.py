# Auto-generated from: code.ipynb
# This file contains code cells followed by their saved outputs as comments.

# In[1]:  (cell 1)
# Install packages
# !pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl

# In[51]:  (cell 2)
# Standard library
import os
import copy
import pickle
import json
from pathlib import Path
from datetime import datetime

# Core data & viz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Stats 
from scipy import stats

# Scikit-learn 
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split, learning_curve, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Warnings
import warnings
warnings.filterwarnings("ignore")

# XGBoost
import xgboost as xgb

#Neural Nets
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
from skorch import NeuralNetRegressor

# In[52]:  (cell 3)
def huber_loss(y_true, y_pred, delta=1.0):
    """Calculate Huber loss manually for comparison"""
    error = y_true - y_pred
    is_small = np.abs(error) <= delta
    loss = np.where(is_small, 0.5 * error**2, delta * (np.abs(error) - 0.5 * delta))
    return np.mean(loss)


def calculate_all_metrics(y_true, y_pred, prefix=""):
    """Calculate comprehensive regression metrics including regulatory context"""
    metrics = {
        f'{prefix}RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        f'{prefix}MAE': mean_absolute_error(y_true, y_pred),
        f'{prefix}R2': r2_score(y_true, y_pred),
        f'{prefix}Huber': huber_loss(y_true, y_pred),
        f'{prefix}MSE': mean_squared_error(y_true, y_pred)
    }
    
    # Add regulatory classification metrics for Quote predictions
    if isinstance(y_true, (pd.Series, np.ndarray)):
        # Regulatory threshold classification accuracy
        true_insolvent = (y_true < 0)
        pred_insolvent = (y_pred < 0)
        true_undercap = (y_true < 1)
        pred_undercap = (y_pred < 1)
        true_wellcap = (y_true > 2)
        pred_wellcap = (y_pred > 2)
        
        if len(y_true) > 0:
            metrics[f'{prefix}Insolvent_Accuracy'] = np.mean(true_insolvent == pred_insolvent)
            metrics[f'{prefix}Undercap_Accuracy'] = np.mean(true_undercap == pred_undercap)
            metrics[f'{prefix}WellCap_Accuracy'] = np.mean(true_wellcap == pred_wellcap)
            
            # MAPE for non-zero values (avoid division by zero)
            nonzero_mask = y_true != 0
            if np.sum(nonzero_mask) > 0:
                mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
                metrics[f'{prefix}MAPE'] = mape
    
    return metrics

# In[53]:  (cell 4)
def analyze_regulatory_performance(y_true, y_pred, target_name="Quote"):
    """Analyze model performance across regulatory categories"""
    
    # Define regulatory categories
    categories = {
        'Insolvent': y_true < 0,
        'Undercapitalized': (y_true >= 0) & (y_true < 1),
        'Adequate': (y_true >= 1) & (y_true <= 2),
        'Well-Capitalized': y_true > 2
    }
    
    print(f"\nREGULATORY PERFORMANCE ANALYSIS - {target_name}")
    
    results = {}
    for category, mask in categories.items():
        if np.sum(mask) > 0:
            y_true_cat = y_true[mask]
            y_pred_cat = y_pred[mask]
            
            rmse = np.sqrt(mean_squared_error(y_true_cat, y_pred_cat))
            mae = mean_absolute_error(y_true_cat, y_pred_cat)
            r2 = r2_score(y_true_cat, y_pred_cat)
            
            results[category] = {
                'count': np.sum(mask),
                'percentage': np.sum(mask) / len(y_true) * 100,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            print(f"\n{category} ({np.sum(mask)} obs, {np.sum(mask)/len(y_true)*100:.1f}%):")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
    
    return results

# In[54]:  (cell 5)
def plot_learning_curves(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'):
    """Plot learning curves for model diagnosis"""

    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=8
    )
    
    train_mean = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    plt.xlabel('Training Set Size')
    plt.ylabel('MSE')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()

def plot_residuals(y_true, y_pred, title="Residual Plot"):
    """Create residual plots for model diagnosis with legends"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs Predicted with color coding
    # Color code by residual magnitude
    abs_residuals = np.abs(residuals)
    colors = plt.cm.viridis(abs_residuals / abs_residuals.max())
    
    scatter = ax1.scatter(y_pred, residuals, alpha=0.6, c=colors, s=30)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residual Line')
    
    # Add trend line
    z = np.polyfit(y_pred, residuals, 1)
    p = np.poly1d(z)
    ax1.plot(y_pred, p(y_pred), "orange", linestyle='-', linewidth=2, 
            label=f'Trend Line (slope: {z[0]:.4f})')
    
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'{title} - Residuals vs Predicted')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for residual magnitude
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label('Absolute Residual Magnitude', rotation=270, labelpad=20)
    
    # Residual distribution with enhanced styling
    n, bins, patches = ax2.hist(residuals, bins=30, alpha=0.7, density=True, 
                            color='skyblue', edgecolor='black', linewidth=0.5)
    
    # Add normal distribution overlay
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    normal_dist = ((1/(sigma * np.sqrt(2 * np.pi))) * 
                   np.exp(-0.5 * ((x - mu) / sigma) ** 2))
    ax2.plot(x, normal_dist, 'red', linewidth=2, 
            label=f'Normal Fit (μ={mu:.3f}, σ={sigma:.3f})')
    
    # Add mean and median lines
    ax2.axvline(mu, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean: {mu:.3f}')
    ax2.axvline(np.median(residuals), color='orange', linestyle='--', linewidth=2, alpha=0.8, 
            label=f'Median: {np.median(residuals):.3f}')
    
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Density')
    ax2.set_title(f'{title} - Residual Distribution')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# In[55]:  (cell 6)
def plot_predicted_vs_actual(y_true, y_pred, title="Predicted vs Actual", 
                                     scale_type="original", target_scalers=None):
    """Enhanced plotting with optional target_scalers parameter"""
    
    if 'Quote' in title:
        if scale_type == "original":
            # Original scale thresholds
            insolvent_mask = y_true < 0
            undercap_mask = (y_true >= 0) & (y_true < 1)
            adequate_mask = (y_true >= 1) & (y_true <= 2)
            wellcap_mask = y_true > 2
        else:
            # Normalized scale thresholds
            if target_scalers and 'quote' in target_scalers:
                try:
                    quote_scaler = target_scalers['quote']
                    threshold_0_norm = quote_scaler.transform([[0]])[0][0]
                    threshold_1_norm = quote_scaler.transform([[1]])[0][0]
                    threshold_2_norm = quote_scaler.transform([[2]])[0][0]
                    
                    insolvent_mask = y_true < threshold_0_norm
                    undercap_mask = (y_true >= threshold_0_norm) & (y_true < threshold_1_norm)
                    adequate_mask = (y_true >= threshold_1_norm) & (y_true <= threshold_2_norm)
                    wellcap_mask = y_true > threshold_2_norm
                except Exception as e:
                    print(f"Warning: Could not use target_scalers ({e}), falling back to quartiles")
                    # Fallback to quartiles
                    q25, q50, q75 = np.percentile(y_true, [25, 50, 75])
                    insolvent_mask = y_true <= q25
                    undercap_mask = (y_true > q25) & (y_true <= q50)
                    adequate_mask = (y_true > q50) & (y_true <= q75)
                    wellcap_mask = y_true > q75
            else:
                print("Warning: target_scalers not provided, using quartiles for normalized scale")
                # Fallback to quartiles
                q25, q50, q75 = np.percentile(y_true, [25, 50, 75])
                insolvent_mask = y_true <= q25
                undercap_mask = (y_true > q25) & (y_true <= q50)
                adequate_mask = (y_true > q50) & (y_true <= q75)
                wellcap_mask = y_true > q75
                
        # Plot each category separately for proper legend
        if np.sum(insolvent_mask) > 0:
            plt.scatter(y_true[insolvent_mask], y_pred[insolvent_mask], 
                       alpha=0.7, s=25, c='red', label='Insolvent', marker='o')
        if np.sum(undercap_mask) > 0:
            plt.scatter(y_true[undercap_mask], y_pred[undercap_mask], 
                       alpha=0.7, s=25, c='orange', label='Undercapitalized', marker='s')
        if np.sum(adequate_mask) > 0:
            plt.scatter(y_true[adequate_mask], y_pred[adequate_mask], 
                       alpha=0.7, s=25, c='gold', label='Adequate', marker='^')
        if np.sum(wellcap_mask) > 0:
            plt.scatter(y_true[wellcap_mask], y_pred[wellcap_mask], 
                       alpha=0.7, s=25, c='green', label='Well-Capitalized', marker='D')
        
        # Add threshold lines
        if scale_type == "original":
            plt.axhline(y=0, color='red', linestyle=':', alpha=0.7, linewidth=1)
            plt.axhline(y=1, color='orange', linestyle=':', alpha=0.7, linewidth=1)
            plt.axhline(y=2, color='green', linestyle=':', alpha=0.7, linewidth=1)
            plt.axvline(x=0, color='red', linestyle=':', alpha=0.7, linewidth=1)
            plt.axvline(x=1, color='orange', linestyle=':', alpha=0.7, linewidth=1)
            plt.axvline(x=2, color='green', linestyle=':', alpha=0.7, linewidth=1)
        
    else:
        # For non-Quote targets, use quartile-based coloring
        q25, q50, q75 = np.percentile(y_true, [25, 50, 75])
        
        q1_mask = y_true <= q25
        q2_mask = (y_true > q25) & (y_true <= q50)
        q3_mask = (y_true > q50) & (y_true <= q75)
        q4_mask = y_true > q75
        
        plt.scatter(y_true[q1_mask], y_pred[q1_mask], 
                   alpha=0.7, s=25, c='lightcoral', label='Q1 (Lowest 25%)', marker='o')
        plt.scatter(y_true[q2_mask], y_pred[q2_mask], 
                   alpha=0.7, s=25, c='orange', label='Q2 (25-50%)', marker='s')
        plt.scatter(y_true[q3_mask], y_pred[q3_mask], 
                   alpha=0.7, s=25, c='lightblue', label='Q3 (50-75%)', marker='^')
        plt.scatter(y_true[q4_mask], y_pred[q4_mask], 
                   alpha=0.7, s=25, c='darkgreen', label='Q4 (Top 25%)', marker='D')
    
    plt.xlabel(f'Actual Values ({scale_type.title()} Scale)')
    plt.ylabel(f'Predicted Values ({scale_type.title()} Scale)')
    plt.title(f"{title} - {scale_type.title()} Scale")
    
    # Create comprehensive legend
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
               ncol=1, fontsize=10, markerscale=1.2)
    
    plt.grid(True, alpha=0.3)
    
    # R² annotation
    r2 = r2_score(y_true, y_pred)
    plt.text(0.95, 0.05, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             horizontalalignment='right', fontsize=12, fontweight='bold')
    
    return plt.gcf()

# In[56]:  (cell 7)
def plot_distribution_analysis(df, target_col='Quote'):
    """Plot target distribution with regulatory thresholds and proper legends"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram with regulatory thresholds
    n_bins = 50
    counts, bins, patches = ax1.hist(df[target_col], bins=n_bins, alpha=0.7, density=True, 
                                    color='skyblue', edgecolor='black', linewidth=0.5)
    
    if target_col == 'Quote' or target_col == 'Quote_original':
        # Add threshold lines with proper labels
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Insolvency Threshold (0)')
        ax1.axvline(x=1, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Undercapitalized Threshold (1)')
        ax1.axvline(x=2, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Well-Capitalized Threshold (2)')
        
        # Add mean and median lines
        mean_val = df[target_col].mean()
        median_val = df[target_col].median()
        ax1.axvline(x=mean_val, color='purple', linestyle='-', linewidth=2, alpha=0.8, label=f'Mean ({mean_val:.2f})')
        ax1.axvline(x=median_val, color='brown', linestyle='-', linewidth=2, alpha=0.8, label=f'Median ({median_val:.2f})')
        
        ax1.legend(loc='upper right', fontsize=9)
    
    ax1.set_xlabel(target_col)
    ax1.set_ylabel('Density')
    ax1.set_title(f'{target_col} Distribution with Regulatory Thresholds')
    ax1.grid(True, alpha=0.3)
    
    # Box plot with threshold lines
    box_plot = ax2.boxplot(df[target_col], patch_artist=True, 
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='blue', linewidth=1.5),
                        capprops=dict(color='blue', linewidth=1.5))
    
    if target_col == 'Quote' or target_col == 'Quote_original':
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Insolvency')
        ax2.axhline(y=1, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Undercapitalized')
        ax2.axhline(y=2, color='green', linestyle='--', alpha=0.8, linewidth=2, label='Well-Capitalized')
        ax2.legend(loc='upper right', fontsize=9)
    
    ax2.set_ylabel(target_col)
    ax2.set_title(f'{target_col} Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(df[target_col], dist="norm", plot=ax3)
    ax3.set_title(f'{target_col} Q-Q Plot (Normal Distribution)')
    ax3.grid(True, alpha=0.3)
    
    # Regulatory status or statistics
    if (target_col == 'Quote' or target_col == 'Quote_original') and 'regulatory_status' in df.columns:
        status_counts = df['regulatory_status'].value_counts()
        colors_pie = ['red', 'orange', 'gold', 'green']
        explode = (0.05, 0.05, 0.05, 0.05)
        
        wedges, texts, autotexts = ax4.pie(status_counts.values, 
                                        labels=status_counts.index, 
                                        autopct='%1.1f%%',
                                        colors=colors_pie,
                                        explode=explode,
                                        shadow=True,
                                        startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax4.set_title(f'{target_col} Regulatory Status Distribution', fontweight='bold')
        ax4.legend(wedges, [f'{label}: {count} obs' for label, count in status_counts.items()],
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
        
    else:
        # For other targets, show statistics
        stats_text = f"""
        Mean: {df[target_col].mean():.2e}
        Median: {df[target_col].median():.2e}
        Std: {df[target_col].std():.2e}
        Min: {df[target_col].min():.2e}
        Max: {df[target_col].max():.2e}
        Skewness: {df[target_col].skew():.3f}
        Kurtosis: {df[target_col].kurtosis():.3f}
        """
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
                verticalalignment='center')
        ax4.set_title(f'{target_col} Statistics', fontweight='bold')
        ax4.axis('off')
    
    plt.tight_layout()
    return fig

# In[57]:  (cell 8)
# Load the header row to see the column names
df_preview = pd.read_excel("BDSII Daten 2 - Original.xlsx",
                        sheet_name = "Tabelle1",
                        nrows = 0)

print(df_preview.columns.tolist())

# --- Output [8] ---
# [1] type: stream
# (stdout)
# ["['Nr.', 'ZSK1', 'ZSK2', 'ZSK3', 'Vola4', 'Vola5', 'Vola6', 'Verlust7', 'Verlust8', 'MR9', 'MR10', 'MR11', 'MR12', 'MR13', 'MR14', 'MR15', 'MR16', 'MR17', 'MR18', 'MR19', 'MR20', 'GCR', 'SF', 'VT', 'EM', 'SCR', 'Quote', 'SCR_Sterblichkeit', 'SCR_Langlebigkeit', 'SCR_Invalidität/Morbidität', 'SCR_Kosten', 'SCR_Stornoanstieg', 'SCR_Stornor\\x81ückgang', 'SCR_Massenstorno', 'SCR_Katastrophe', 'SCR_Zinsrü\\x81ckgang', 'SCR_Zinsanstieg', 'SCR_Zins', 'SCR_Aktien - sonstige', 'SCR_Aktien', 'SCR_Spread - Kreditderivate', 'SCR_Spread - Verbriefungen', 'SCR_Spread', 'SCR_Währung', 'SCR_Storno', 'SCR_Ausfall', 'SCR_vt. Risiko Leben', 'SCR_Marktrisiko', 'SCR_KV - Invalidität/Morbidität-Krankenkosten', 'SCR_KV - Storno', 'SCR_KV - Invalidität/Morbidität', 'SCR_vt. Risiko Kranken Leben', 'SCR_vt. Risiko Kranken']\n"]
# --- End Output ---

# In[58]:  (cell 9)
# Read the sheet 
df = pd.read_excel("BDSII Daten 2 - Original.xlsx", sheet_name = "Tabelle1")

# The columns we do not need
cols_to_drop = [
    "GCR",
    "SF",
    "VT",
    "SCR_Sterblichkeit",
    "SCR_Langlebigkeit",
    "SCR_Invalidität/Morbidität",
    "SCR_Kosten",
    "SCR_Stornoanstieg",
    "SCR_Stornorückgang",   
    "SCR_Stornorückgang", 
    "SCR_Zinsrückgang",
    "SCR_Massenstorno",
    "SCR_Katastrophe",
    "SCR_Zinsrückgang",
    "SCR_Zinsanstieg",
    "SCR_Zins",
    "SCR_Aktien - sonstige",
    "SCR_Aktien",
    "SCR_Spread - Kreditderivate",
    "SCR_Spread - Verbriefungen",
    "SCR_Spread",
    "SCR_Währung",
    "SCR_Storno",
    "SCR_Ausfall",
    "SCR_vt. Risiko Leben",
    "SCR_Marktrisiko",
    "SCR_KV - Invalidität/Morbidität-Krankenkosten",
    "SCR_KV - Storno",
    "SCR_KV - Invalidität/Morbidität",
    "SCR_vt. Risiko Kranken Leben",
    "SCR_vt. Risiko Kranken"
]

# Drop the columns
df = df.drop(columns = cols_to_drop, errors = "ignore")

# Save the cleaned data to a csv file
df.to_csv("BDSII_Daten_2_clean.csv", index = False, encoding = "utf-8")

# In[59]:  (cell 10)
# Set random seed for reproducibility
np.random.seed(8)

# Load the dataset
df = pd.read_csv("BDSII_Daten_2_clean.csv")

# Define feature columns and targets 
feature_cols = [col for col in df.columns if col not in ['Nr.', 'Quote', 'SCR', 'EM']]
target_cols = ['EM', 'SCR', 'Quote']

# Drop 'Nr.' because it's just an index
if 'Nr.' in df.columns:
    df = df.drop('Nr.', axis=1)


# Print shape and first 5 rows
print(f"Dataset shape: {df.shape}")
print(f"Features: {len(feature_cols)} columns")
# print(feature_cols)
print(f"\nFirst 5 rows:\n{df.head()}")

# Verify Quote calculation
print(f"\nQuote validation (should be ~0): {(df['Quote'] - df['EM']/df['SCR']).abs().max():.6f}")

# --- Output [10] ---
# [1] type: stream
# (stdout)
# ['Dataset shape: (10230, 23)\n', 'Features: 20 columns\n', '\n', 'First 5 rows:\n', '       ZSK1      ZSK2      ZSK3     Vola4     Vola5     Vola6  Verlust7  \\\n', '0  0.922054  0.837028  0.809331  0.178250  0.901512  0.179174  0.191340   \n', '1  0.619457  0.135444  0.742208  0.895245  0.203651  0.913381  0.066844   \n', '2  0.527813  0.066627  0.979909  0.088391  0.905464  0.964193  0.445658   \n', '3  0.471844  0.922035  0.351186  0.162060  0.121955  0.002218  0.269084   \n', '4  0.940594  0.515785  0.569936  0.505810  0.090705  0.095968  0.987834   \n', '\n', '   Verlust8   MR9  MR10  ...      MR14  MR15    MR16  MR17  MR18  MR19  MR20  \\\n', '0  0.056881  0.25  0.25  ...  0.222222  0.60  0.4825  0.38  1.00     1   0.6   \n', '1  0.195996  0.25  0.25  ...  0.222222  0.60  0.4825  0.38  1.00     1   0.6   \n', '2  0.465802  0.25  0.25  ...  0.222222  0.60  0.4825  0.38  1.00     1   0.6   \n', '3  0.183026  0.25  0.25  ...  0.222222  0.35  0.4825  0.38  1.00     1   0.6   \n', '4  0.026776  0.20  0.10  ...  0.888889  0.10  0.0305  0.10  0.06     1   0.7   \n', '\n', '             EM           SCR     Quote  \n', '0  1.026604e+09  4.073178e+08  2.520400  \n', '1  7.478581e+08  8.190900e+08  0.913035  \n', '2  1.099707e+09  6.561373e+08  1.676032  \n', '3  1.543899e+09  4.847708e+08  3.184802  \n', '4  7.781381e+08  2.619466e+08  2.970599  \n', '\n', '[5 rows x 23 columns]\n', '\n', 'Quote validation (should be ~0): 0.000000\n']
# --- End Output ---

# In[60]:  (cell 11)
# Summary statistics for all targets (using original scale data)
print(f"\nTARGET VARIABLES SUMMARY:")
for target in target_cols:
    print(f"\n\t {target}:")
    if target in ['EM', 'SCR']:
        # Format EUR values as integers (in millions)
        mean_val = int(df[target].mean() / 1e6)
        std_val = int(df[target].std() / 1e6)
        min_val = int(df[target].min() / 1e6)
        max_val = int(df[target].max() / 1e6)
        median_val = int(df[target].median() / 1e6)
        print(f"\t\t Mean: EUR {mean_val:,}M, Std: EUR {std_val:,}M")
        print(f"\t\t Min: EUR {min_val:,}M, Max: EUR {max_val:,}M")
        print(f"\t\t Median: EUR {median_val:,}M")
    elif target == 'Quote':
        # Format Quote as percentage
        mean_val = df[target].mean() * 100
        std_val = df[target].std() * 100
        min_val = df[target].min() * 100
        max_val = df[target].max() * 100
        median_val = df[target].median() * 100
        print(f"\t\t Mean: {mean_val:.0f}%, Std: {std_val:.0f}%")
        print(f"\t\t Min: {min_val:.0f}%, Max: {max_val:.0f}%")
        print(f"\t\t Median: {median_val:.0f}%")

print(f"\nQUOTE DISTRIBUTION ANALYSIS: ")
print(f"Negative Quotes (insolvent): {(df['Quote'] < 0).sum()} ({(df['Quote'] < 0).mean()*100:.1f}%)")
print(f"Quote < 100% (undercapitalized): {(df['Quote'] < 1).sum()} ({(df['Quote'] < 1).mean()*100:.1f}%)")
print(f"Quote 100-200% (adequate): {((df['Quote'] >= 1) & (df['Quote'] < 2)).sum()} ({((df['Quote'] >= 1) & (df['Quote'] < 2)).mean()*100:.1f}%)")
print(f"Quote > 200% (well-capitalized): {(df['Quote'] >= 2).sum()} ({(df['Quote'] >= 2).mean()*100:.1f}%)")

# Check for insolvent scenarios (EM < 0)
print(f"\nSOLVENCY STATUS:")
print(f"Negative EM (insolvent entities): {(df['EM'] < 0).sum()} ({(df['EM'] < 0).mean()*100:.1f}%)")
print(f"Min EM: EUR {int(df['EM'].min() / 1e6):,}M, Max EM: EUR {int(df['EM'].max() / 1e6):,}M")

# Check for missing values
print(f"\nMISSING VALUES COUNT:")
missing_counts = df[feature_cols + target_cols].isnull().sum()
print(missing_counts[missing_counts > 0] if missing_counts.any() else "No missing values found")

# --- Output [11] ---
# [1] type: stream
# (stdout)
# ['\n', 'TARGET VARIABLES SUMMARY:\n', '\n', '\t EM:\n', '\t\t Mean: EUR 1,114M, Std: EUR 433M\n', '\t\t Min: EUR -4,740M, Max: EUR 2,109M\n', '\t\t Median: EUR 1,180M\n', '\n', '\t SCR:\n', '\t\t Mean: EUR 633M, Std: EUR 293M\n', '\t\t Min: EUR 224M, Max: EUR 2,149M\n', '\t\t Median: EUR 505M\n', '\n', '\t Quote:\n', '\t\t Mean: 217%, Std: 101%\n', '\t\t Min: -247%, Max: 465%\n', '\t\t Median: 243%\n', '\n', 'QUOTE DISTRIBUTION ANALYSIS: \n', 'Negative Quotes (insolvent): 272 (2.7%)\n', 'Quote < 100% (undercapitalized): 1654 (16.2%)\n', 'Quote 100-200% (adequate): 2077 (20.3%)\n', 'Quote > 200% (well-capitalized): 6499 (63.5%)\n', '\n', 'SOLVENCY STATUS:\n', 'Negative EM (insolvent entities): 272 (2.7%)\n', 'Min EM: EUR -4,740M, Max EM: EUR 2,109M\n', '\n', 'MISSING VALUES COUNT:\n', 'No missing values found\n']
# --- End Output ---

# In[61]:  (cell 12)
# Correlation matrix heatmap for all features
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Features')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# --- Output [12] ---
# [1] type: display_data
# ['<Figure size 1400x1000 with 2 Axes>']
# --- End Output ---

# In[62]:  (cell 13)
# Histogram/KDE of Quote
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['Quote'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Quote (Solvency II Ratio)')
plt.ylabel('Frequency')
plt.title('Distribution of Quote - Histogram')
plt.axvline(df['Quote'].mean(), color='red', linestyle='--', label=f'Mean: {df["Quote"].mean():.2f}')
plt.axvline(df['Quote'].median(), color='green', linestyle='--', label=f'Median: {df["Quote"].median():.2f}')
plt.legend()

plt.subplot(1, 2, 2)
df['Quote'].plot(kind='kde', linewidth=2)
plt.xlabel('Quote (Solvency II Ratio)')
plt.ylabel('Density')
plt.title('Distribution of Quote - KDE')
plt.axvline(df['Quote'].mean(), color='red', linestyle='--', label=f'Mean: {df["Quote"].mean():.2f}')
plt.axvline(df['Quote'].median(), color='green', linestyle='--', label=f'Median: {df["Quote"].median():.2f}')
plt.legend()

plt.tight_layout()
plt.show()

# --- Output [13] ---
# [1] type: display_data
# ['<Figure size 1200x500 with 2 Axes>']
# --- End Output ---

# In[63]:  (cell 14)
# Statistical analysis of Quote 
print(f"\nQUOTE ANALYSIS:")
print(f"Mean: {df['Quote'].mean():.3f}, Std: {df['Quote'].std():.3f}")
print(f"Range: [{df['Quote'].min():.3f}, {df['Quote'].max():.3f}]")
print(f"Skewness: {stats.skew(df['Quote']):.3f}")
print(f"Kurtosis: {stats.kurtosis(df['Quote']):.3f}")
print(f"Observations with Quote < 1 (undercapitalized): {(df['Quote'] <= 1).sum()} ({(df['Quote'] <= 1).mean()*100:.1f}%)")
print(f"Observations with Quote > 2 (well-capitalized): {(df['Quote'] >= 2).sum()} ({(df['Quote'] >= 2).mean()*100:.1f}%)")

# --- Output [14] ---
# [1] type: stream
# (stdout)
# ['\n', 'QUOTE ANALYSIS:\n', 'Mean: 2.166, Std: 1.005\n', 'Range: [-2.472, 4.653]\n', 'Skewness: -0.695\n', 'Kurtosis: -0.299\n', 'Observations with Quote < 1 (undercapitalized): 1654 (16.2%)\n', 'Observations with Quote > 2 (well-capitalized): 6499 (63.5%)\n']
# --- End Output ---

# In[64]:  (cell 15)
# Risk type groupings for features
risk_groups = {
    'Interest Rate': ['ZSK1', 'ZSK2', 'ZSK3'],
    'Market Volatility': ['Vola4', 'Vola5', 'Vola6'],
    'Market Losses': ['Verlust7', 'Verlust8'],
    'FI/RE Allocation': ['MR9', 'MR10', 'MR11', 'MR12', 'MR13', 'MR14'],
    'Other Mgmt Rules': ['MR15', 'MR16', 'MR17', 'MR18', 'MR19', 'MR20']
}

print(f"\nFEATURE GROUPS:")
for group_name, features in risk_groups.items():
    print(f"{group_name}: {', '.join(features)}")

# --- Output [15] ---
# [1] type: stream
# (stdout)
# ['\n', 'FEATURE GROUPS:\n', 'Interest Rate: ZSK1, ZSK2, ZSK3\n', 'Market Volatility: Vola4, Vola5, Vola6\n', 'Market Losses: Verlust7, Verlust8\n', 'FI/RE Allocation: MR9, MR10, MR11, MR12, MR13, MR14\n', 'Other Mgmt Rules: MR15, MR16, MR17, MR18, MR19, MR20\n']
# --- End Output ---

# In[65]:  (cell 16)
# Create ordered plot to visualize outliers before removal
original_size = len(df)
print(f"Original dataset size: {original_size}")

# Create regulatory threshold indicators (before outlier removal for complete analysis)
df['regulatory_status'] = pd.cut(df['Quote'], 
                                bins=[-np.inf, 0, 1, 2, np.inf], 
                                labels=['Insolvent', 'Undercapitalized', 'Adequate', 'Well-Capitalized'])

df['is_insolvent'] = (df['Quote'] < 0).astype(int)
df['is_undercapitalized'] = (df['Quote'] < 1).astype(int)
df['is_well_capitalized'] = (df['Quote'] > 2).astype(int)

# Create distribution analysis plots BEFORE removing outliers
print(f"\nCreating target distribution analysis (including outliers)")
os.makedirs("figs", exist_ok=True)

for target in ['Quote', 'EM', 'SCR']:
    print(f"Creating distribution plots for {target}...")
    fig_with_outliers = plot_distribution_analysis(df, target)
    fig_path_with_outliers = os.path.join("figs", f"distribution_analysis_{target.lower()}_with_outliers.png")
    fig_with_outliers.savefig(fig_path_with_outliers, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig_with_outliers)

# Create the ordered plot
print(f"\nCreating ordered plot to identify outliers...")
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
fig.suptitle('Target Variables by Observation Index (Including Outliers)', fontsize=16, fontweight='bold')

# Plot EM
axes[0].plot(df.index, df['EM']/1e9, 'b-', alpha=0.7, linewidth=0.8)
axes[0].scatter(df.index, df['EM']/1e9, s=8, alpha=0.6, color='blue')
axes[0].set_ylabel('EM (Billions EUR)')
axes[0].set_title('Eligible Margin (EM)')
axes[0].grid(True, alpha=0.3)

# Plot SCR
axes[1].plot(df.index, df['SCR']/1e9, 'g-', alpha=0.7, linewidth=0.8)
axes[1].scatter(df.index, df['SCR']/1e9, s=8, alpha=0.6, color='green')
axes[1].set_ylabel('SCR (Billions EUR)')
axes[1].set_title('Solvency Capital Requirement (SCR)')
axes[1].grid(True, alpha=0.3)

# Plot Quote as percentage
axes[2].plot(df.index, df['Quote']*100, 'r-', alpha=0.7, linewidth=0.8)
axes[2].scatter(df.index, df['Quote']*100, s=8, alpha=0.6, color='red')
axes[2].set_ylabel('Quote (%)')
axes[2].set_title('Solvency Ratio (Quote)')
axes[2].set_xlabel('Observation Index')
axes[2].grid(True, alpha=0.3)

# Add horizontal lines for regulatory thresholds on Quote plot
axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.8, label='Insolvency (0%)')
axes[2].axhline(y=100, color='orange', linestyle='--', alpha=0.8, label='Undercapitalized (100%)')
axes[2].axhline(y=200, color='green', linestyle='--', alpha=0.8, label='Well-capitalized (200%)')
axes[2].legend()

plt.tight_layout()
fig.savefig('figs/ordered_plot_with_outliers.png', dpi=300, bbox_inches='tight')
# plt.show()
plt.close(fig)

# Identify the specific outlier
outlier_threshold = -4.5e9  # -4.5 billion EUR threshold to catch the -4.74b outlier
outlier_mask = df['EM'] < outlier_threshold

print(f"\nOutlier identification:")
if outlier_mask.any():
    outlier_data = df[outlier_mask]
    print(f"Found {outlier_mask.sum()} outlier(s):")
    for idx, row in outlier_data.iterrows():
        print(f"  Index {idx}: EM = EUR {row['EM']/1e9:.2f}B, Quote = {row['Quote']*100:.0f}%")
else:
    print("No outliers found with the specified threshold")

# Remove the specific outlier
df_clean = df[~outlier_mask].copy()
outliers_removed = original_size - len(df_clean)

print(f"\nOutlier removal summary:")
print(f"Outliers removed: {outliers_removed} observation(s)")
print(f"Reason: Extreme negative EM value (< -4.5B EUR)")
print(f"Clean dataset size: {len(df_clean)}")

# Update df to the cleaned version
df = df_clean

print(f"\nUpdated data ranges after outlier removal:")
print(f"EM range: EUR {df['EM'].min()/1e6:.0f}M to EUR {df['EM'].max()/1e6:.0f}M")
print(f"SCR range: EUR {df['SCR'].min()/1e6:.0f}M to EUR {df['SCR'].max()/1e6:.0f}M")
print(f"Quote range: {df['Quote'].min()*100:.0f}% to {df['Quote'].max()*100:.0f}%")

# --- Output [16] ---
# [1] type: stream
# (stdout)
# ['Original dataset size: 10230\n', '\n', 'Creating target distribution analysis (including outliers)\n', 'Creating distribution plots for Quote...\n', 'Creating distribution plots for EM...\n', 'Creating distribution plots for SCR...\n', '\n', 'Creating ordered plot to identify outliers...\n', '\n', 'Outlier identification:\n', 'Found 1 outlier(s):\n', '  Index 5318: EM = EUR -4.74B, Quote = -247%\n', '\n', 'Outlier removal summary:\n', 'Outliers removed: 1 observation(s)\n', 'Reason: Extreme negative EM value (< -4.5B EUR)\n', 'Clean dataset size: 10229\n', '\n', 'Updated data ranges after outlier removal:\n', 'EM range: EUR -2164M to EUR 2110M\n', 'SCR range: EUR 224M to EUR 2149M\n', 'Quote range: -126% to 465%\n']
# --- End Output ---

# In[66]:  (cell 17)
# Save original values for later analysis and business interpretation
df['Quote_original'] = df['Quote'].copy()
df['SCR_original'] = df['SCR'].copy()  
df['EM_original'] = df['EM'].copy()

print(f"Original scale ranges (BEFORE any normalization or splitting):")
print(f"  Quote: [{df['Quote_original'].min():.3f}, {df['Quote_original'].max():.3f}]")
print(f"  SCR: [{df['SCR_original'].min():.2e}, {df['SCR_original'].max():.2e}]") 
print(f"  EM: [{df['EM_original'].min():.2e}, {df['EM_original'].max():.2e}]")

# --- Output [17] ---
# [1] type: stream
# (stdout)
# ['Original scale ranges (BEFORE any normalization or splitting):\n', '  Quote: [-1.265, 4.653]\n', '  SCR: [2.24e+08, 2.15e+09]\n', '  EM: [-2.16e+09, 2.11e+09]\n']
# --- End Output ---

# In[67]:  (cell 18)
# Create regulatory threshold indicators for enhanced analysis
df['regulatory_status'] = pd.cut(df['Quote_original'], 
                                bins=[-np.inf, 0, 1, 2, np.inf], 
                                labels=['Insolvent', 'Undercapitalized', 'Adequate', 'Well-Capitalized'])

df['is_insolvent'] = (df['Quote_original'] < 0).astype(int)
df['is_undercapitalized'] = (df['Quote_original'] < 1).astype(int)
df['is_well_capitalized'] = (df['Quote_original'] > 2).astype(int)

# Create distribution analysis plots for BOTH original and normalized versions
print(f"\nCreating target distribution analysis...")
os.makedirs("figs", exist_ok=True)

for target in ['Quote', 'EM', 'SCR']:
    # Original scale plots (using the _original columns)
    original_col = f'{target}_original'
    if original_col in df.columns:
        fig_orig = plot_distribution_analysis(df, original_col)
        fig_path_orig = os.path.join("figs", f"distribution_analysis_{target.lower()}_original.png")
        plt.savefig(fig_path_orig, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close(fig_orig)
    
    # Normalized scale plots (using the current columns)
    fig_norm = plot_distribution_analysis(df, target)
    fig_path_norm = os.path.join("figs", f"distribution_analysis_{target.lower()}_normalized.png")
    plt.savefig(fig_path_norm, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig_norm)

# --- Output [18] ---
# [1] type: stream
# (stdout)
# ['\n', 'Creating target distribution analysis...\n']
# --- End Output ---

# In[68]:  (cell 19)
# Set random seeds for reproducibility
RANDOM_STATE = 8
np.random.seed(RANDOM_STATE)

# Extract features and targets (ORIGINAL SCALE)
X = df[feature_cols].copy()
y_quote = df['Quote'].copy()  
y_scr = df['SCR'].copy()      
y_em = df['EM'].copy()        
y_multi = df[['SCR', 'EM']].copy()  

print(f"Features shape: {X.shape}")
print(f"All targets use ORIGINAL scale before normalization")


# Step 1: First split (80% temp, 20% test)
X_temp, X_test, y_quote_temp, y_quote_test, y_scr_temp, y_scr_test, y_em_temp, y_em_test = train_test_split(
    X, y_quote, y_scr, y_em, test_size=0.2, random_state=RANDOM_STATE
)

# Step 2: Second split of temp data (75% train, 25% val of the 80% = 60% train, 20% val overall)
X_train, X_val, y_quote_train, y_quote_val, y_scr_train, y_scr_val, y_em_train, y_em_val = train_test_split(
    X_temp, y_quote_temp, y_scr_temp, y_em_temp, test_size=0.25, random_state=RANDOM_STATE
)

# Create multi-output targets from the correctly split data
y_multi_train = pd.DataFrame({'SCR': y_scr_train, 'EM': y_em_train}, index=y_scr_train.index)
y_multi_val = pd.DataFrame({'SCR': y_scr_val, 'EM': y_em_val}, index=y_scr_val.index)
y_multi_test = pd.DataFrame({'SCR': y_scr_test, 'EM': y_em_test}, index=y_scr_test.index)

print(f"\nData splits (CORRECTLY ALIGNED):")
print(f"Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")


print(f"\nALIGNMENT VERIFICATION:")
print(f"X_train index range: {X_train.index.min()} - {X_train.index.max()}")
print(f"y_scr_train index range: {y_scr_train.index.min()} - {y_scr_train.index.max()}")
print(f"y_em_train index range: {y_em_train.index.min()} - {y_em_train.index.max()}")
print(f"Indices match: {X_train.index.equals(y_scr_train.index) and X_train.index.equals(y_em_train.index)}")


print(f"\nFitting target scalers on TRAINING data only")
quote_scaler = MinMaxScaler(feature_range=(-1, 1))
scr_scaler = MinMaxScaler(feature_range=(-1, 1))
em_scaler = MinMaxScaler(feature_range=(-1, 1))

# Fit on training data only
quote_scaler.fit(y_quote_train.values.reshape(-1, 1))
scr_scaler.fit(y_scr_train.values.reshape(-1, 1))
em_scaler.fit(y_em_train.values.reshape(-1, 1))

# Store scalers for later use
target_scalers = {
    'quote': quote_scaler,
    'scr': scr_scaler,
    'em': em_scaler
}


print(f"Applying target normalization")

# Transform Quote
y_quote_train = pd.Series(quote_scaler.transform(y_quote_train.values.reshape(-1, 1)).ravel(), 
                         index=y_quote_train.index)
y_quote_val = pd.Series(quote_scaler.transform(y_quote_val.values.reshape(-1, 1)).ravel(),
                       index=y_quote_val.index)
y_quote_test = pd.Series(quote_scaler.transform(y_quote_test.values.reshape(-1, 1)).ravel(),
                        index=y_quote_test.index)

# Transform SCR  
y_scr_train = pd.Series(scr_scaler.transform(y_scr_train.values.reshape(-1, 1)).ravel(),
                       index=y_scr_train.index)
y_scr_val = pd.Series(scr_scaler.transform(y_scr_val.values.reshape(-1, 1)).ravel(),
                     index=y_scr_val.index)
y_scr_test = pd.Series(scr_scaler.transform(y_scr_test.values.reshape(-1, 1)).ravel(),
                      index=y_scr_test.index)

# Transform EM
y_em_train = pd.Series(em_scaler.transform(y_em_train.values.reshape(-1, 1)).ravel(),
                      index=y_em_train.index)
y_em_val = pd.Series(em_scaler.transform(y_em_val.values.reshape(-1, 1)).ravel(),
                    index=y_em_val.index) 
y_em_test = pd.Series(em_scaler.transform(y_em_test.values.reshape(-1, 1)).ravel(),
                     index=y_em_test.index)

# Transform multi-output targets (update with normalized values)
y_multi_train = pd.DataFrame({
    'SCR': y_scr_train,
    'EM': y_em_train
}, index=y_scr_train.index)

y_multi_val = pd.DataFrame({
    'SCR': y_scr_val,
    'EM': y_em_val
}, index=y_scr_val.index)

y_multi_test = pd.DataFrame({
    'SCR': y_scr_test,
    'EM': y_em_test
}, index=y_scr_test.index)

print(f"\nNORMALIZATION VERIFICATION (fitted only on training data):")
print(f"Quote normalized range: [{y_quote_train.min():.3f}, {y_quote_train.max():.3f}] (train)")
print(f"SCR normalized range: [{y_scr_train.min():.3f}, {y_scr_train.max():.3f}] (train)")  
print(f"EM normalized range: [{y_em_train.min():.3f}, {y_em_train.max():.3f}] (train)")

print(f"\nTest set may have slightly different ranges (expected with proper normalization):")
print(f"Quote test range: [{y_quote_test.min():.3f}, {y_quote_test.max():.3f}]")
print(f"SCR test range: [{y_scr_test.min():.3f}, {y_scr_test.max():.3f}]")
print(f"EM test range: [{y_em_test.min():.3f}, {y_em_test.max():.3f}]")

# Final verification that all splits are correctly aligned
print(f"\nFINAL ALIGNMENT VERIFICATION:")
train_alignment = (X_train.index.equals(y_quote_train.index) and 
                  X_train.index.equals(y_scr_train.index) and 
                  X_train.index.equals(y_em_train.index))
val_alignment = (X_val.index.equals(y_quote_val.index) and 
                X_val.index.equals(y_scr_val.index) and 
                X_val.index.equals(y_em_val.index))
test_alignment = (X_test.index.equals(y_quote_test.index) and 
                 X_test.index.equals(y_scr_test.index) and 
                 X_test.index.equals(y_em_test.index))

print(f"Train splits aligned: {train_alignment}")
print(f"Validation splits aligned: {val_alignment}")
print(f"Test splits aligned: {test_alignment}")

if all([train_alignment, val_alignment, test_alignment]):
    print(" ALL DATA SPLITS ARE CORRECTLY ALIGNED!")
else:
    print(" ERROR: Data splits are still misaligned!")

# Create feature preprocessing pipeline (fit only on training data)
numeric_features = feature_cols
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features)
])

print(f"\nPreprocessing pipeline created with {len(numeric_features)} numeric features")
print("Feature preprocessing will be fitted only on training data during model training")

# --- Output [19] ---
# [1] type: stream
# (stdout)
# ['Features shape: (10229, 20)\n', 'All targets use ORIGINAL scale before normalization\n', '\n', 'Data splits (CORRECTLY ALIGNED):\n', 'Train: 6137 samples (60.0%)\n', 'Validation: 2046 samples (20.0%)\n', 'Test: 2046 samples (20.0%)\n', '\n', 'ALIGNMENT VERIFICATION:\n', 'X_train index range: 0 - 10229\n', 'y_scr_train index range: 0 - 10229\n', 'y_em_train index range: 0 - 10229\n', 'Indices match: True\n', '\n', 'Fitting target scalers on TRAINING data only\n', 'Applying target normalization\n', '\n', 'NORMALIZATION VERIFICATION (fitted only on training data):\n', 'Quote normalized range: [-1.000, 1.000] (train)\n', 'SCR normalized range: [-1.000, 1.000] (train)\n', 'EM normalized range: [-1.000, 1.000] (train)\n', '\n', 'Test set may have slightly different ranges (expected with proper normalization):\n', 'Quote test range: [-0.953, 0.731]\n', 'SCR test range: [-1.000, 1.085]\n', 'EM test range: [-0.873, 1.016]\n', '\n', 'FINAL ALIGNMENT VERIFICATION:\n', 'Train splits aligned: True\n', 'Validation splits aligned: True\n', 'Test splits aligned: True\n', ' ALL DATA SPLITS ARE CORRECTLY ALIGNED!\n', '\n', 'Preprocessing pipeline created with 20 numeric features\n', 'Feature preprocessing will be fitted only on training data during model training\n']
# --- End Output ---

# In[69]:  (cell 20)
# Models with different complexity levels and loss functions
models_config = {
    'dummy': {
        'name': 'Dummy Regressor',
        'model': DummyRegressor(strategy='mean'),
        'loss': 'MSE'
    },
    'linear_mse': {
        'name': 'Linear Regression (MSE)',
        'model': LinearRegression(),
        'loss': 'MSE'
    },
    'quadratic': {
    'name': 'Quadratic (Degree 2)',
    'model': Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE))  
    ]),
    'loss': 'MSE'
    },
    'cubic': {
        'name': 'Cubic (Degree 3)', 
        'model': Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('ridge', Ridge(alpha=1000.0, random_state=RANDOM_STATE))
        ]),
        'loss': 'MSE'
    },
    'elastic_net': {
    'name': 'Elastic Net',
    'model': ElasticNet(random_state=RANDOM_STATE),
    'loss': 'MSE',
    'param_grid': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # Wider range
        'l1_ratio': [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]  # More options
        }
    },
    'ridge_cv': {
        'name': 'Ridge with CV',
        'model': RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5),
        'loss': 'MSE'
    },
    'lasso_cv': {
        'name': 'Lasso with CV', 
        'model': LassoCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5, random_state=RANDOM_STATE),
        'loss': 'MSE'
    }
}

print(f"Defined {len(models_config)} model configurations:")
for key, config in models_config.items():
    print(f"  - {config['name']} (Loss: {config['loss']})")

# --- Output [20] ---
# [1] type: stream
# (stdout)
# ['Defined 7 model configurations:\n', '  - Dummy Regressor (Loss: MSE)\n', '  - Linear Regression (MSE) (Loss: MSE)\n', '  - Quadratic (Degree 2) (Loss: MSE)\n', '  - Cubic (Degree 3) (Loss: MSE)\n', '  - Elastic Net (Loss: MSE)\n', '  - Ridge with CV (Loss: MSE)\n', '  - Lasso with CV (Loss: MSE)\n']
# --- End Output ---

# In[70]:  (cell 21)
# # 1. Define your PyTorch model
# class RegressorModule(nn.Module):
#     def __init__ (self, hidden_sizes=(64, 32),n_features=20 ,dropout=0.2, batchnorm=True, out_features=1):
#         super().__init__()
#         layers = []

#         layers += [nn.LazyLinear(hidden_sizes[0])]
#         if batchnorm:
#             layers += [nn.BatchNorm1d(hidden_sizes[0])]
#         layers += [nn.ReLU(), nn.Dropout(dropout)]  

#         for in_h, out_h in zip(hidden_sizes[:-1], hidden_sizes[1:]):
#             layers += [nn.Linear(in_h, out_h)]
#             if batchnorm:
#                 layers += [nn.BatchNorm1d(out_h)]
#             layers += [nn.ReLU(), nn.Dropout(dropout)]
        
#         layers += [nn.Linear(hidden_sizes[-1], out_features)]
#         self.net = nn.Sequential(*layers)

#     def forward(self, X):
#         out= self.net(X.float())
#         return out #out.squeeze(1) if out.shape[1]==1 else out   # only need it for single-output regression


# # 2. Wrap PyTorch model with skorch
# torch_single = NeuralNetRegressor(
#     RegressorModule,
#     module__n_features=X_train.shape[1],   # Number of input features
#     max_epochs=50,
#     optimizer=torch.optim.Adam,
#     lr=0.001,
#     batch_size=64,
#     iterator_train__shuffle=True,
#     verbose=0
# )
# models_config.update({
#     'torch_single': {
#         'name': 'PyTorch Single Output',
#         'model': torch_single,
#         'loss': 'MSE',
#         'param_grid': {
#             # 'module__hidden_sizes': [16, 32, 64],
#             'lr': [0.001, 0.01],
#             'max_epochs': [50, 100]
#         }
#     }
# })

# In[71]:  (cell 22)
# MLPRegressor from Scikit-Learn: 
models_config.update({
    'nn_mlp': {
        'name': 'Neural Net (MLPRegressor)',
        'model': MLPRegressor(hidden_layer_sizes=(64,32),
                              activation='tanh',
                              solver='adam',
                              max_iter=500,
                              early_stopping=True,
                              validation_fraction=0.1,
                              n_iter_no_change=20,
                              random_state=RANDOM_STATE),
        'loss': 'MSE',
        'param_grid': {
            'hidden_layer_sizes': [(128, 64), (256, 128), (128,64,32)],         
            'alpha': [1e-5, 1e-4, 1e-3],
            'learning_rate_init': [1e-3, 5e-4],
            'activation': ['relu', 'tanh']
    },
    'search_type': 'random',  # Use RandomizedSearchCV for efficiency
    'n_iter': 10,  # Number of parameter settings that are sampled}  
    'multi_output': True,  # Indicate that this is also for multi-output regression
    'single_output': True  # Indicate that this is for single-output regression
}})

# In[72]:  (cell 23)

# torch_multi = NeuralNetRegressor(
#     module=RegressorModule,
#     module__n_features=X_train.shape[1],   # Number of input features
#     module__out_features=2,                 # Two outputs for multi-output regression
#     module__hidden_sizes=(128,64),
#     module__dropout=0.3,
#     module__batchnorm=True,
#     max_epochs=300,
#     optimizer=torch.optim.Adam,
#     criterion=nn.MSELoss,
#     lr=0.001,
#     batch_size=256,
#     iterator_train__shuffle=True,
#     verbose=0,
#     device='cuda' if torch.cuda.is_available() else 'cpu'
# )

# models_config.update({
#     'torch_multi': {
#         'name': 'PyTorch Multi Output',
#         'model': torch_multi,
#         'loss': 'MSE'}})

# In[73]:  (cell 24)
# PyTorch MLP with Dropout and BatchNorm
class TorchMLPReg(nn.Module):
    def __init__ (self, hidden_sizes=(64, 32), dropout=0.2, batchnorm=True, out_features=1):
        super().__init__()
        layers = []

        layers += [nn.LazyLinear(hidden_sizes[0])]
        if batchnorm:
            layers += [nn.BatchNorm1d(hidden_sizes[0])]
        layers += [nn.ReLU(), nn.Dropout(dropout)]  

        for in_h, out_h in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers += [nn.Linear(in_h, out_h)]
            if batchnorm:
                layers += [nn.BatchNorm1d(out_h)]
            layers += [nn.ReLU(), nn.Dropout(dropout)]
        
        layers += [nn.Linear(hidden_sizes[-1], out_features)]
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        out= self.net(X.float())
        return out #out.squeeze(1) if out.shape[1]==1 else out   # only need it for single-output regression

# Multi-output regression 2 hidden layers
nn_torch_dropout_multi = NeuralNetRegressor(    
    module=TorchMLPReg,
    module__hidden_sizes=(128,64),
    module__dropout=0.3,
    module__batchnorm=True,
    module__out_features=2,
    optimizer=torch.optim.AdamW,
    optimizer__lr=1e-3,
    optimizer__weight_decay=1e-3,
    criterion=nn.MSELoss,
    max_epochs=300,
    batch_size=256,verbose=0)

models_config.update({
    'nn_torch_dropout_multi': {
        'name': 'Neural Net (PyTorch MLP Dropout Multi)',
        'model': nn_torch_dropout_multi,
        'loss': 'MSE',
        'multi_output': True,  # Indicate that this is also for multi-output regression
        'single_output': False
    }
})

# In[74]:  (cell 25)
print(f"Defined {len(models_config)} model configurations:")
for key, config in models_config.items():
    print(f"  - {config['name']} (Loss: {config['loss']})")

# --- Output [25] ---
# [1] type: stream
# (stdout)
# ['Defined 9 model configurations:\n', '  - Dummy Regressor (Loss: MSE)\n', '  - Linear Regression (MSE) (Loss: MSE)\n', '  - Quadratic (Degree 2) (Loss: MSE)\n', '  - Cubic (Degree 3) (Loss: MSE)\n', '  - Elastic Net (Loss: MSE)\n', '  - Ridge with CV (Loss: MSE)\n', '  - Lasso with CV (Loss: MSE)\n', '  - Neural Net (MLPRegressor) (Loss: MSE)\n', '  - Neural Net (PyTorch MLP Dropout Multi) (Loss: MSE)\n']
# --- End Output ---

# In[75]:  (cell 26)
to_dense = FunctionTransformer(
    lambda X: X.toarray() if hasattr(X, "toarray") else X,
    accept_sparse=True
)

to_float32 = FunctionTransformer(
    lambda X: X.astype(np.float32))

# In[76]:  (cell 27)
def train_and_evaluate_model(model_config, X_train, X_val, X_test, y_train, y_val, y_test, 
                                  preprocessor, target_name, cv_folds=5):
    """Train and evaluate a single model with comprehensive metrics"""
    
    model_name = model_config['name']
    base_model = copy.deepcopy(model_config['model'])
    
    print(f"\nTraining {model_name} for {target_name}...")
    
    # # Create pipeline
    # if 'param_grid' in model_config:
    #     pipeline = Pipeline([
    #         ('preprocessor', copy.deepcopy(preprocessor)), 
    #         ('model', base_model)
    #     ])
        
    #     param_grid = {f'model__{k}': v for k, v in model_config['param_grid'].items()}
        
    #     grid_search = GridSearchCV(
    #         pipeline, param_grid, cv=cv_folds, 
    #         scoring='neg_mean_squared_error', n_jobs=-1
    #     )
        
    #     grid_search.fit(X_train, y_train)
    #     final_model = grid_search.best_estimator_
    #     print(f"  Best parameters: {grid_search.best_params_}")
        
    # else:
    #     final_model = Pipeline([
    #         ('preprocessor', copy.deepcopy(preprocessor)),  
    #         ('model', base_model) 
    #     ])
    #     final_model.fit(X_train, y_train)
    
    # NEW: INCLUDE RANDOMIZED SEARCH OPTION -----------------------
    if 'param_grid' in model_config:
        pipeline = Pipeline([
            ('preprocessor', copy.deepcopy(preprocessor)),
            ('to_float32', to_float32),          # Ensure float32 for PyTorch
            ('model', base_model)
        ])
        
        param_grid = {f'model__{k}': v for k, v in model_config['param_grid'].items()}
        
        if model_config.get('search_type', 'grid') == 'random':
            search = RandomizedSearchCV(
                pipeline, param_grid, n_iter=model_config.get('n_iter', 10),
                cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1, random_state=RANDOM_STATE, error_score='raise'
            )
        else:
            search = GridSearchCV(
                pipeline, param_grid,
                cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1, error_score='raise'
            )
        
        search.fit(X_train, y_train)
        final_model = search.best_estimator_
        print(f"  Best parameters: {search.best_params_}")
        
    else:
        final_model = Pipeline([
            ('preprocessor', copy.deepcopy(preprocessor)),
            ('to_float32', to_float32),          # Ensure float32 for PyTorch
            ('model', base_model) 
        ])
        final_model.fit(X_train, y_train)
# ---------------------------------------------------------------




    # Cross-validation and predictions
    cv_scores = cross_val_score(final_model, X_train, y_train,
                               cv=cv_folds, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    
    y_train_pred = final_model.predict(X_train)
    y_val_pred = final_model.predict(X_val)
    y_test_pred = final_model.predict(X_test)
    
    # Calculate normalized metrics (for fair comparison)
    train_metrics_norm = calculate_all_metrics(y_train, y_train_pred, "train_norm_")
    val_metrics_norm = calculate_all_metrics(y_val, y_val_pred, "val_norm_")
    test_metrics_norm = calculate_all_metrics(y_test, y_test_pred, "test_norm_")
    
    all_metrics = {
        **train_metrics_norm, 
        **val_metrics_norm, 
        **test_metrics_norm,
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std()
    }
    
    # Convert to original scale for business interpretation
    if target_name in ['Quote', 'SCR', 'EM']:
        scaler_key = target_name.lower()
        if scaler_key in target_scalers:
            target_scaler = target_scalers[scaler_key]
            
            # Convert predictions to original scale
            y_train_pred_orig = target_scaler.inverse_transform(np.array(y_train_pred).reshape(-1, 1)).ravel()
            y_val_pred_orig = target_scaler.inverse_transform(np.array(y_val_pred).reshape(-1, 1)).ravel()
            y_test_pred_orig = target_scaler.inverse_transform(np.array(y_test_pred).reshape(-1, 1)).ravel()

            y_train_orig = target_scaler.inverse_transform(np.array(y_train).reshape(-1, 1)).ravel()
            y_val_orig = target_scaler.inverse_transform(np.array(y_val).reshape(-1, 1)).ravel()
            y_test_orig = target_scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).ravel()
            
            # Calculate original scale metrics
            train_metrics_orig = calculate_all_metrics(y_train_orig, y_train_pred_orig, "train_orig_")
            val_metrics_orig = calculate_all_metrics(y_val_orig, y_val_pred_orig, "val_orig_")
            test_metrics_orig = calculate_all_metrics(y_test_orig, y_test_pred_orig, "test_orig_")
            
            all_metrics.update({**train_metrics_orig, **val_metrics_orig, **test_metrics_orig})
            
            # Regulatory analysis for Quote
            if target_name == 'Quote':
                print(f"\n  Regulatory Performance Analysis for {model_name}:")
                val_reg_results = analyze_regulatory_performance(y_val_orig, y_val_pred_orig, f"{model_name} - Validation")
                test_reg_results = analyze_regulatory_performance(y_test_orig, y_test_pred_orig, f"{model_name} - Test")
                
                all_metrics['regulatory_val'] = val_reg_results
                all_metrics['regulatory_test'] = test_reg_results
                
                results_predictions = {
                    'train': y_train_pred_orig,
                    'val': y_val_pred_orig,
                    'test': y_test_pred_orig,
                    'train_norm': y_train_pred,
                    'val_norm': y_val_pred,
                    'test_norm': y_test_pred
                }
                
                results_actuals = {
                    'train': y_train_orig,
                    'val': y_val_orig,
                    'test': y_test_orig,
                    'train_norm': y_train,
                    'val_norm': y_val,
                    'test_norm': y_test
                }
            else:
                results_predictions = {
                    'train': y_train_pred_orig,
                    'val': y_val_pred_orig,
                    'test': y_test_pred_orig,
                    'train_norm': y_train_pred,
                    'val_norm': y_val_pred,
                    'test_norm': y_test_pred
                }
                
                results_actuals = {
                    'train': y_train_orig,
                    'val': y_val_orig,
                    'test': y_test_orig,
                    'train_norm': y_train,
                    'val_norm': y_val,
                    'test_norm': y_test
                }
        else:
            # Fallback
            results_predictions = {
                'train': y_train_pred,
                'val': y_val_pred,
                'test': y_test_pred
            }
            
            results_actuals = {
                'train': y_train,
                'val': y_val,
                'test': y_test
            }
    else:
        results_predictions = {
            'train': y_train_pred,
            'val': y_val_pred,
            'test': y_test_pred
        }
        
        results_actuals = {
            'train': y_train,
            'val': y_val,
            'test': y_test
        }
    
    results = {
        'model': final_model,
        'model_name': model_name,
        'target_name': target_name,
        'metrics': all_metrics,
        'predictions': results_predictions,
        'actuals': results_actuals
    }
    
    return results

def train_multioutput_model_consolidated(model_config, X_train, X_val, X_test, y_train, y_val, y_test, 
                                       preprocessor, cv_folds=5, calculate_quote=True):
    """
    Consolidated multi-output training for SCR and EM with BOTH normalized and original scale metrics
    """
    
    model_name = model_config['name']
    base_model = copy.deepcopy(model_config['model'])
    
    print(f"\nTraining {model_name} (Multi-Output)...")
    # Ensure y_train is in the correct format for NeuralNetRegressor        #NEW!
    if isinstance(base_model, NeuralNetRegressor):
        y_train_fit =np.asarray(y_train, dtype=np.float32)          #----------------------------------
    else:
        y_train_fit = y_train    
    try:
        # Create multi-output pipeline
        if isinstance(base_model, (RandomForestRegressor, GradientBoostingRegressor, MLPRegressor, NeuralNetRegressor)):    #NEW: MLPRegressor supports multi-output natively
            final_model = Pipeline([
                ('preprocessor', copy.deepcopy(preprocessor)),
                ('model', base_model)
            ])
        else:
            multi_model = MultiOutputRegressor(base_model)
            final_model = Pipeline([
                ('preprocessor', copy.deepcopy(preprocessor)),
                ('model', multi_model)
            ])
        
        # Fit the model
        final_model.fit(X_train, y_train_fit)               #NEW: Changed y_train -> y_train_fit ---------------------------------
        
        # Cross-validation scoring
        try:
            cv_scores = cross_val_score(final_model, X_train, y_train,
                                      cv=cv_folds, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
            cv_rmse_mean = cv_rmse.mean()
            cv_rmse_std = cv_rmse.std()
        except Exception:
            cv_rmse_mean = np.nan
            cv_rmse_std = np.nan
        
        # Get predictions for all splits (normalized scale)
        y_train_pred = final_model.predict(X_train)  # Shape: (n_samples, 2) [SCR, EM]
        y_val_pred = final_model.predict(X_val)
        y_test_pred = final_model.predict(X_test)
        
        # Initialize metrics
        all_metrics = {
            'cv_rmse_mean': cv_rmse_mean,
            'cv_rmse_std': cv_rmse_std
        }
        target_names = ['SCR', 'EM']
        
        # Store MSE values for macro calculation (normalized)
        train_mses_norm = []
        val_mses_norm = []
        test_mses_norm = []
        
        # Store MSE values for macro calculation (original scale)
        train_mses_orig = []
        val_mses_orig = []
        test_mses_orig = []
        
        for i, target in enumerate(target_names):
            # Normalized metric
            train_metrics_norm = calculate_all_metrics(y_train.iloc[:, i], y_train_pred[:, i], f"train_{target}_norm_")
            val_metrics_norm = calculate_all_metrics(y_val.iloc[:, i], y_val_pred[:, i], f"val_{target}_norm_")
            test_metrics_norm = calculate_all_metrics(y_test.iloc[:, i], y_test_pred[:, i], f"test_{target}_norm_")
            
            all_metrics.update({**train_metrics_norm, **val_metrics_norm, **test_metrics_norm})
            
            # Store normalized MSE values
            train_mses_norm.append(train_metrics_norm[f'train_{target}_norm_MSE'])
            val_mses_norm.append(val_metrics_norm[f'val_{target}_norm_MSE'])
            test_mses_norm.append(test_metrics_norm[f'test_{target}_norm_MSE'])
            
            # Original metric
            target_scaler = target_scalers[target.lower()]
            
            # Convert normalized predictions to original scale
            train_pred_orig = target_scaler.inverse_transform(y_train_pred[:, i].reshape(-1, 1)).ravel()
            val_pred_orig = target_scaler.inverse_transform(y_val_pred[:, i].reshape(-1, 1)).ravel()
            test_pred_orig = target_scaler.inverse_transform(y_test_pred[:, i].reshape(-1, 1)).ravel()
            
            # Convert normalized actuals to original scale
            train_actual_orig = target_scaler.inverse_transform(y_train.iloc[:, i].values.reshape(-1, 1)).ravel()
            val_actual_orig = target_scaler.inverse_transform(y_val.iloc[:, i].values.reshape(-1, 1)).ravel()
            test_actual_orig = target_scaler.inverse_transform(y_test.iloc[:, i].values.reshape(-1, 1)).ravel()
            
            # Calculate original scale metrics
            train_metrics_orig = calculate_all_metrics(train_actual_orig, train_pred_orig, f"train_{target}_orig_")
            val_metrics_orig = calculate_all_metrics(val_actual_orig, val_pred_orig, f"val_{target}_orig_")
            test_metrics_orig = calculate_all_metrics(test_actual_orig, test_pred_orig, f"test_{target}_orig_")
            
            all_metrics.update({**train_metrics_orig, **val_metrics_orig, **test_metrics_orig})
            
            # Store original MSE values
            train_mses_orig.append(train_metrics_orig[f'train_{target}_orig_MSE'])
            val_mses_orig.append(val_metrics_orig[f'val_{target}_orig_MSE'])
            test_mses_orig.append(test_metrics_orig[f'test_{target}_orig_MSE'])
            
            # Also add standard metrics (backwards compatibility), use normalized
            all_metrics[f'train_{target}_RMSE'] = train_metrics_norm[f'train_{target}_norm_RMSE']
            all_metrics[f'val_{target}_RMSE'] = val_metrics_norm[f'val_{target}_norm_RMSE']
            all_metrics[f'test_{target}_RMSE'] = test_metrics_norm[f'test_{target}_norm_RMSE']
            all_metrics[f'train_{target}_R2'] = train_metrics_norm[f'train_{target}_norm_R2']
            all_metrics[f'val_{target}_R2'] = val_metrics_norm[f'val_{target}_norm_R2']
            all_metrics[f'test_{target}_R2'] = test_metrics_norm[f'test_{target}_norm_R2']
            all_metrics[f'train_{target}_MAE'] = train_metrics_norm[f'train_{target}_norm_MAE']
            all_metrics[f'val_{target}_MAE'] = val_metrics_norm[f'val_{target}_norm_MAE']
            all_metrics[f'test_{target}_MAE'] = test_metrics_norm[f'test_{target}_norm_MAE']
        
        # Calculate macro RMSE for both scales
        train_rmse_macro_norm = np.sqrt(np.mean(train_mses_norm))
        val_rmse_macro_norm = np.sqrt(np.mean(val_mses_norm))
        test_rmse_macro_norm = np.sqrt(np.mean(test_mses_norm))
        
        train_rmse_macro_orig = np.sqrt(np.mean(train_mses_orig))
        val_rmse_macro_orig = np.sqrt(np.mean(val_mses_orig))
        test_rmse_macro_orig = np.sqrt(np.mean(test_mses_orig))
        
        all_metrics.update({
            # Normalized scale macro metrics
            'train_RMSE_macro': train_rmse_macro_norm,
            'val_RMSE_macro': val_rmse_macro_norm,
            'test_RMSE_macro': test_rmse_macro_norm,
            
            # Original scale macro metrics
            'train_RMSE_macro_orig': train_rmse_macro_orig,
            'val_RMSE_macro_orig': val_rmse_macro_orig,
            'test_RMSE_macro_orig': test_rmse_macro_orig
        })
        
        # Calculate Quote from EM/SCR predictions
        if calculate_quote:
            if 'scr' in target_scalers and 'em' in target_scalers and 'quote' in target_scalers:
                scr_scaler = target_scalers['scr']
                em_scaler = target_scalers['em']
                quote_scaler = target_scalers['quote']
                
                # Extract normalized predictions for all splits
                scr_train_pred_norm = y_train_pred[:, 0]
                em_train_pred_norm = y_train_pred[:, 1]
                scr_val_pred_norm = y_val_pred[:, 0]
                em_val_pred_norm = y_val_pred[:, 1]
                scr_test_pred_norm = y_test_pred[:, 0]
                em_test_pred_norm = y_test_pred[:, 1]
                
                # Convert to original scale
                scr_train_pred_orig = scr_scaler.inverse_transform(scr_train_pred_norm.reshape(-1, 1)).ravel()
                em_train_pred_orig = em_scaler.inverse_transform(em_train_pred_norm.reshape(-1, 1)).ravel()
                scr_val_pred_orig = scr_scaler.inverse_transform(scr_val_pred_norm.reshape(-1, 1)).ravel()
                em_val_pred_orig = em_scaler.inverse_transform(em_val_pred_norm.reshape(-1, 1)).ravel()
                scr_test_pred_orig = scr_scaler.inverse_transform(scr_test_pred_norm.reshape(-1, 1)).ravel()
                em_test_pred_orig = em_scaler.inverse_transform(em_test_pred_norm.reshape(-1, 1)).ravel()
                
                # Calculate Quote in ORIGINAL scale
                quote_train_pred_orig = np.where(np.abs(scr_train_pred_orig) > 1e-6, 
                                               em_train_pred_orig / scr_train_pred_orig, 0)
                quote_val_pred_orig = np.where(np.abs(scr_val_pred_orig) > 1e-6, 
                                             em_val_pred_orig / scr_val_pred_orig, 0)
                quote_test_pred_orig = np.where(np.abs(scr_test_pred_orig) > 1e-6, 
                                              em_test_pred_orig / scr_test_pred_orig, 0)
                
                # Convert calculated Quote back to normalized scale
                quote_train_pred_norm = quote_scaler.transform(quote_train_pred_orig.reshape(-1, 1)).ravel()
                quote_val_pred_norm = quote_scaler.transform(quote_val_pred_orig.reshape(-1, 1)).ravel()
                quote_test_pred_norm = quote_scaler.transform(quote_test_pred_orig.reshape(-1, 1)).ravel()
                
                # Get actual Quote values
                try:
                    quote_train_actual_norm = np.array(y_quote_train)
                    quote_val_actual_norm = np.array(y_quote_val)
                    quote_test_actual_norm = np.array(y_quote_test)
                    
                    # Calculate Quote metrics (normalized scale)
                    quote_train_metrics = calculate_all_metrics(quote_train_actual_norm, quote_train_pred_norm, "train_Quote_")
                    quote_val_metrics = calculate_all_metrics(quote_val_actual_norm, quote_val_pred_norm, "val_Quote_")
                    quote_test_metrics = calculate_all_metrics(quote_test_actual_norm, quote_test_pred_norm, "test_Quote_")
                    
                    # Calculate Quote metrics (original scale)
                    quote_train_actual_orig = quote_scaler.inverse_transform(quote_train_actual_norm.reshape(-1, 1)).ravel()
                    quote_val_actual_orig = quote_scaler.inverse_transform(quote_val_actual_norm.reshape(-1, 1)).ravel()
                    quote_test_actual_orig = quote_scaler.inverse_transform(quote_test_actual_norm.reshape(-1, 1)).ravel()
                    
                    quote_train_metrics_orig = calculate_all_metrics(quote_train_actual_orig, quote_train_pred_orig, "train_Quote_orig_")
                    quote_val_metrics_orig = calculate_all_metrics(quote_val_actual_orig, quote_val_pred_orig, "val_Quote_orig_")
                    quote_test_metrics_orig = calculate_all_metrics(quote_test_actual_orig, quote_test_pred_orig, "test_Quote_orig_")
                    
                    all_metrics.update({**quote_train_metrics, **quote_val_metrics, **quote_test_metrics})
                    all_metrics.update({**quote_train_metrics_orig, **quote_val_metrics_orig, **quote_test_metrics_orig})
                    
                    print(f"    Quote calculation successful:")
                    print(f"    Quote Train RMSE (norm): {all_metrics['train_Quote_RMSE']:.4f}")
                    print(f"    Quote Val RMSE (norm): {all_metrics['val_Quote_RMSE']:.4f}")
                    print(f"    Quote Test RMSE (norm): {all_metrics['test_Quote_RMSE']:.4f}")
                    print(f"    Quote Test R² (norm): {all_metrics['test_Quote_R2']:.4f}")
                    
                except NameError:
                    print(f"  Quote target variables not available, skipping Quote calculation")
                    calculate_quote = False
            else:
                print(f"  Scalers not available, skipping Quote calculation")
                calculate_quote = False
        
        # Print comprehensive performance summary with BOTH scales
        print(f"    Model Performance Summary:")
        if not np.isnan(cv_rmse_mean):
            print(f"    CV RMSE: {cv_rmse_mean:.4f} (±{cv_rmse_std:.4f})")
        
        print(f"    SCR Performance:")
        print(f"      Train RMSE (norm): {all_metrics['train_SCR_norm_RMSE']:.4f}")
        print(f"      Val RMSE (norm):   {all_metrics['val_SCR_norm_RMSE']:.4f}")
        print(f"      Test RMSE (norm):  {all_metrics['test_SCR_norm_RMSE']:.4f}")
        print(f"      Train RMSE (orig): {all_metrics['train_SCR_orig_RMSE']:.2e}")
        print(f"      Val RMSE (orig):   {all_metrics['val_SCR_orig_RMSE']:.2e}")
        print(f"      Test RMSE (orig):  {all_metrics['test_SCR_orig_RMSE']:.2e}")
        
        print(f"    EM Performance:")
        print(f"      Train RMSE (norm): {all_metrics['train_EM_norm_RMSE']:.4f}")
        print(f"      Val RMSE (norm):   {all_metrics['val_EM_norm_RMSE']:.4f}")
        print(f"      Test RMSE (norm):  {all_metrics['test_EM_norm_RMSE']:.4f}")
        print(f"      Train RMSE (orig): {all_metrics['train_EM_orig_RMSE']:.2e}")
        print(f"      Val RMSE (orig):   {all_metrics['val_EM_orig_RMSE']:.2e}")
        print(f"      Test RMSE (orig):  {all_metrics['test_EM_orig_RMSE']:.2e}")
        
        print(f"    Combined Performance:")
        print(f"      Train RMSE (norm): {train_rmse_macro_norm:.4f}")
        print(f"      Val RMSE (norm):   {val_rmse_macro_norm:.4f}")
        print(f"      Test RMSE (norm):  {test_rmse_macro_norm:.4f}")
        print(f"      Train RMSE (orig): {train_rmse_macro_orig:.2e}")
        print(f"      Val RMSE (orig):   {val_rmse_macro_orig:.2e}")
        print(f"      Test RMSE (orig):  {test_rmse_macro_orig:.2e}")
        
        # Calculate and show overfitting gap
        train_val_gap = val_rmse_macro_norm - train_rmse_macro_norm
        if train_val_gap > 0.01:
            print(f"      Overfitting detected: Val-Train gap = {train_val_gap:.4f}")
        elif train_val_gap < -0.005:
            print(f"      Possible underfitting: Val better than Train by {abs(train_val_gap):.4f}")
        
        # Prepare results
        target_name = 'Joint_SCR_EM_Quote' if calculate_quote else 'SCR_EM_Multi'
        
        results = {
            'model': final_model,
            'model_name': model_name + (' (with Quote)' if calculate_quote else ' (Multi-Output)'),
            'target_name': target_name,
            'metrics': all_metrics,
            'predictions': {
                'train': y_train_pred,
                'val': y_val_pred,
                'test': y_test_pred
            },
            'actuals': {
                'train': y_train,
                'val': y_val,
                'test': y_test
            },
            'quote_calculated': calculate_quote
        }
        
        return results
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# In[77]:  (cell 28)
all_results = {}

# Train models for Quote (primary target)
print("\nTRAINING MODELS FOR QUOTE (PRIMARY TARGET)")
all_results['quote'] = {}

X_train = X_train.astype(np.float32)            #For compatibility with PyTorch
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)
y_quote_train = y_quote_train.astype(np.float32)
y_quote_val = y_quote_val.astype(np.float32)
y_quote_test = y_quote_test.astype(np.float32)

print(X_train.shape, y_quote_train.shape)
print(X_val.shape, y_quote_val.shape)
print(X_test.shape, y_quote_test.shape)



for model_key, model_config in models_config.items():
    if 'multi_output' not in model_config or 'single_output' in model_config and model_config['single_output']:
        try:
            results = train_and_evaluate_model(
                model_config, X_train, X_val, X_test, 
                y_quote_train, y_quote_val, y_quote_test,
                preprocessor, "Quote"
            )
            all_results['quote'][model_key] = results
            
            # Print summary metrics
            metrics = results['metrics']
            print(f"  {results['model_name']}:")
            print(f"    CV RMSE: {metrics['cv_rmse_mean']:.4f} (±{metrics['cv_rmse_std']:.4f})")
            print(f"    Val RMSE (Norm): {metrics['val_norm_RMSE']:.4f}")
            print(f"    Val RMSE (Orig): {metrics['val_orig_RMSE']:.2e}")
            print(f"    Test RMSE (Norm): {metrics['test_norm_RMSE']:.4f}")
            print(f"    Test RMSE (Orig): {metrics['test_orig_RMSE']:.2e}")
            
        except Exception as e:
            print(f"  Error training {model_config['name']}: {str(e)}")

# Train models for SCR (secondary target)
print("\nTRAINING MODELS FOR SCR (SECONDARY TARGET)")
all_results['scr'] = {}

for model_key, model_config in models_config.items():
    if 'multi_output' not in model_config or 'single_output' in model_config and model_config['single_output']:
        try:
            results = train_and_evaluate_model(
                model_config, X_train, X_val, X_test,
                y_scr_train, y_scr_val, y_scr_test,
                preprocessor, "SCR"
            )
            all_results['scr'][model_key] = results
            
            # Print summary metrics 
            metrics = results['metrics']
            print(f"  {results['model_name']}:")
            print(f"    CV RMSE: {metrics['cv_rmse_mean']:.4f} (±{metrics['cv_rmse_std']:.4f})")
            print(f"    Val RMSE (Norm): {metrics['val_norm_RMSE']:.4f}")
            print(f"    Val RMSE (Orig): {metrics['val_orig_RMSE']:.2e}")
            print(f"    Test RMSE (Norm): {metrics['test_norm_RMSE']:.4f}")
            print(f"    Test RMSE (Orig): {metrics['test_orig_RMSE']:.2e}")
            
        except Exception as e:
            print(f"  Error training {model_config['name']}: {str(e)}")

# Train models for EM (secondary target)
print("\nTRAINING MODELS FOR EM (SECONDARY TARGET)")
all_results['em'] = {}

for model_key, model_config in models_config.items():
    if 'multi_output' not in model_config or 'single_output' in model_config and model_config['single_output']:
        try:
            results = train_and_evaluate_model(
                model_config, X_train, X_val, X_test,
                y_em_train, y_em_val, y_em_test,
                preprocessor, "EM"
            )
            all_results['em'][model_key] = results
            
            # Print summary metrics
            metrics = results['metrics']
            print(f"  {results['model_name']}:")
            print(f"    CV RMSE: {metrics['cv_rmse_mean']:.4f} (±{metrics['cv_rmse_std']:.4f})")
            print(f"    Val RMSE (Norm): {metrics['val_norm_RMSE']:.4f}")
            print(f"    Val RMSE (Orig): {metrics['val_orig_RMSE']:.2e}")
            print(f"    Test RMSE (Norm): {metrics['test_norm_RMSE']:.4f}")
            print(f"    Test RMSE (Orig): {metrics['test_orig_RMSE']:.2e}")
            
        except Exception as e:
            print(f"  Error training {model_config['name']}: {str(e)}")

# Train multi-output models for SCR and EM
print("\nTRAINING MULTI-OUTPUT MODELS FOR SCR & EM")
all_results['multi_output'] = {}

for model_key, model_config in models_config.items():
    try:
        results = train_multioutput_model_consolidated(
            model_config, X_train, X_val, X_test,
            y_multi_train, y_multi_val, y_multi_test,
            preprocessor,  calculate_quote=False
        )
        all_results['multi_output'][model_key] = results
        
        # Multi-output metrics
        metrics = results['metrics']
        print(f"  {results['model_name']}:")
        print(f"    Val RMSE (SCR): {metrics['val_SCR_RMSE']:.4f}")
        print(f"    Val RMSE (EM): {metrics['val_EM_RMSE']:.4f}")
        print(f"    Val RMSE (Macro): {metrics['val_RMSE_macro']:.4f}")
        
    except Exception as e:
        print(f"  Error training multi-output {model_config['name']}: {str(e)}")

# --- Output [28] ---
# [1] type: stream
# (stdout)
# ['\n', 'TRAINING MODELS FOR QUOTE (PRIMARY TARGET)\n', '(6137, 20) (6137,)\n', '(2046, 20) (2046,)\n', '(2046, 20) (2046,)\n', '\n', 'Training Dummy Regressor for Quote...\n', '\n', '  Regulatory Performance Analysis for Dummy Regressor:\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Dummy Regressor - Validation\n', '\n', 'Insolvent (59 obs, 2.9%):\n', '  RMSE: 2.4923\n', '  MAE: 2.4779\n', '  R²: -85.8717\n', '\n', 'Undercapitalized (273 obs, 13.3%):\n', '  RMSE: 1.6119\n', '  MAE: 1.5884\n', '  R²: -33.5071\n', '\n', 'Adequate (403 obs, 19.7%):\n', '  RMSE: 0.6965\n', '  MAE: 0.6333\n', '  R²: -4.7761\n', '\n', 'Well-Capitalized (1311 obs, 64.1%):\n', '  RMSE: 0.7929\n', '  MAE: 0.6739\n', '  R²: -2.3241\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Dummy Regressor - Test\n', '\n', 'Insolvent (54 obs, 2.6%):\n', '  RMSE: 2.5137\n', '  MAE: 2.4963\n', '  R²: -71.5004\n', '\n', 'Undercapitalized (288 obs, 14.1%):\n', '  RMSE: 1.6239\n', '  MAE: 1.5997\n', '  R²: -32.7544\n', '\n', 'Adequate (438 obs, 21.4%):\n', '  RMSE: 0.7136\n', '  MAE: 0.6513\n', '  R²: -4.9839\n', '\n', 'Well-Capitalized (1266 obs, 61.9%):\n', '  RMSE: 0.7860\n', '  MAE: 0.6667\n', '  R²: -2.2650\n', '  Dummy Regressor:\n', '    CV RMSE: 0.3373 (±0.0070)\n', '    Val RMSE (Norm): 0.3420\n', '    Val RMSE (Orig): 1.01e+00\n', '    Test RMSE (Norm): 0.3429\n', '    Test RMSE (Orig): 1.01e+00\n', '\n', 'Training Linear Regression (MSE) for Quote...\n', '\n', '  Regulatory Performance Analysis for Linear Regression (MSE):\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Linear Regression (MSE) - Validation\n', '\n', 'Insolvent (59 obs, 2.9%):\n', '  RMSE: 1.2718\n', '  MAE: 1.0055\n', '  R²: -21.6234\n', '\n', 'Undercapitalized (273 obs, 13.3%):\n', '  RMSE: 0.5932\n', '  MAE: 0.4981\n', '  R²: -3.6726\n', '\n', 'Adequate (403 obs, 19.7%):\n', '  RMSE: 0.4224\n', '  MAE: 0.2857\n', '  R²: -1.1245\n', '\n', 'Well-Capitalized (1311 obs, 64.1%):\n', '  RMSE: 0.5936\n', '  MAE: 0.4909\n', '  R²: -0.8633\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Linear Regression (MSE) - Test\n', '\n', 'Insolvent (54 obs, 2.6%):\n', '  RMSE: 1.0128\n', '  MAE: 0.9314\n', '  R²: -10.7688\n', '\n', 'Undercapitalized (288 obs, 14.1%):\n', '  RMSE: 0.5989\n', '  MAE: 0.5025\n', '  R²: -3.5905\n', '\n', 'Adequate (438 obs, 21.4%):\n', '  RMSE: 0.4285\n', '  MAE: 0.2980\n', '  R²: -1.1573\n', '\n', 'Well-Capitalized (1266 obs, 61.9%):\n', '  RMSE: 0.5707\n', '  MAE: 0.4676\n', '  R²: -0.7214\n', '  Linear Regression (MSE):\n', '    CV RMSE: 0.1951 (±0.0042)\n', '    Val RMSE (Norm): 0.2012\n', '    Val RMSE (Orig): 5.95e-01\n', '    Test RMSE (Norm): 0.1907\n', '    Test RMSE (Orig): 5.64e-01\n', '\n', 'Training Quadratic (Degree 2) for Quote...\n', '\n', '  Regulatory Performance Analysis for Quadratic (Degree 2):\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Quadratic (Degree 2) - Validation\n', '\n', 'Insolvent (59 obs, 2.9%):\n', '  RMSE: 0.7936\n', '  MAE: 0.6042\n', '  R²: -7.8073\n', '\n', 'Undercapitalized (273 obs, 13.3%):\n', '  RMSE: 0.3518\n', '  MAE: 0.2677\n', '  R²: -0.6435\n', '\n', 'Adequate (403 obs, 19.7%):\n', '  RMSE: 0.3582\n', '  MAE: 0.2933\n', '  R²: -0.5276\n', '\n', 'Well-Capitalized (1311 obs, 64.1%):\n', '  RMSE: 0.2723\n', '  MAE: 0.2037\n', '  R²: 0.6079\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Quadratic (Degree 2) - Test\n', '\n', 'Insolvent (54 obs, 2.6%):\n', '  RMSE: 0.5897\n', '  MAE: 0.4661\n', '  R²: -2.9897\n', '\n', 'Undercapitalized (288 obs, 14.1%):\n', '  RMSE: 0.3826\n', '  MAE: 0.2864\n', '  R²: -0.8737\n', '\n', 'Adequate (438 obs, 21.4%):\n', '  RMSE: 0.3495\n', '  MAE: 0.2863\n', '  R²: -0.4350\n', '\n', 'Well-Capitalized (1266 obs, 61.9%):\n', '  RMSE: 0.2629\n', '  MAE: 0.1967\n', '  R²: 0.6347\n', '  Quadratic (Degree 2):\n', '    CV RMSE: 0.1045 (±0.0033)\n', '    Val RMSE (Norm): 0.1108\n', '    Val RMSE (Orig): 3.28e-01\n', '    Test RMSE (Norm): 0.1062\n', '    Test RMSE (Orig): 3.14e-01\n', '\n', 'Training Cubic (Degree 3) for Quote...\n', '\n', '  Regulatory Performance Analysis for Cubic (Degree 3):\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Cubic (Degree 3) - Validation\n', '\n', 'Insolvent (59 obs, 2.9%):\n', '  RMSE: 1.0577\n', '  MAE: 0.6983\n', '  R²: -14.6451\n', '\n', 'Undercapitalized (273 obs, 13.3%):\n', '  RMSE: 0.6537\n', '  MAE: 0.4097\n', '  R²: -4.6750\n', '\n', 'Adequate (403 obs, 19.7%):\n', '  RMSE: 0.5054\n', '  MAE: 0.3251\n', '  R²: -2.0421\n', '\n', 'Well-Capitalized (1311 obs, 64.1%):\n', '  RMSE: 0.4855\n', '  MAE: 0.2850\n', '  R²: -0.2466\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Cubic (Degree 3) - Test\n', '\n', 'Insolvent (54 obs, 2.6%):\n', '  RMSE: 0.6582\n', '  MAE: 0.4993\n', '  R²: -3.9712\n', '\n', 'Undercapitalized (288 obs, 14.1%):\n', '  RMSE: 0.5678\n', '  MAE: 0.3701\n', '  R²: -3.1267\n', '\n', 'Adequate (438 obs, 21.4%):\n', '  RMSE: 0.4445\n', '  MAE: 0.2977\n', '  R²: -1.3213\n', '\n', 'Well-Capitalized (1266 obs, 61.9%):\n', '  RMSE: 0.4436\n', '  MAE: 0.2636\n', '  R²: -0.0398\n', '  Cubic (Degree 3):\n', '    CV RMSE: 0.1805 (±0.0080)\n', '    Val RMSE (Norm): 0.1822\n', '    Val RMSE (Orig): 5.39e-01\n', '    Test RMSE (Norm): 0.1588\n', '    Test RMSE (Orig): 4.70e-01\n', '\n', 'Training Elastic Net for Quote...\n', "  Best parameters: {'model__alpha': 0.001, 'model__l1_ratio': 0.1}\n", '\n', '  Regulatory Performance Analysis for Elastic Net:\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Elastic Net - Validation\n', '\n', 'Insolvent (59 obs, 2.9%):\n', '  RMSE: 1.2732\n', '  MAE: 1.0084\n', '  R²: -21.6726\n', '\n', 'Undercapitalized (273 obs, 13.3%):\n', '  RMSE: 0.5943\n', '  MAE: 0.4999\n', '  R²: -3.6915\n', '\n', 'Adequate (403 obs, 19.7%):\n', '  RMSE: 0.4221\n', '  MAE: 0.2855\n', '  R²: -1.1215\n', '\n', 'Well-Capitalized (1311 obs, 64.1%):\n', '  RMSE: 0.5933\n', '  MAE: 0.4908\n', '  R²: -0.8612\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Elastic Net - Test\n', '\n', 'Insolvent (54 obs, 2.6%):\n', '  RMSE: 1.0153\n', '  MAE: 0.9344\n', '  R²: -10.8283\n', '\n', 'Undercapitalized (288 obs, 14.1%):\n', '  RMSE: 0.6003\n', '  MAE: 0.5044\n', '  R²: -3.6123\n', '\n', 'Adequate (438 obs, 21.4%):\n', '  RMSE: 0.4283\n', '  MAE: 0.2978\n', '  R²: -1.1551\n', '\n', 'Well-Capitalized (1266 obs, 61.9%):\n', '  RMSE: 0.5704\n', '  MAE: 0.4675\n', '  R²: -0.7196\n', '  Elastic Net:\n', '    CV RMSE: 0.1951 (±0.0041)\n', '    Val RMSE (Norm): 0.2012\n', '    Val RMSE (Orig): 5.95e-01\n', '    Test RMSE (Norm): 0.1907\n', '    Test RMSE (Orig): 5.64e-01\n', '\n', 'Training Ridge with CV for Quote...\n', '\n', '  Regulatory Performance Analysis for Ridge with CV:\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Ridge with CV - Validation\n', '\n', 'Insolvent (59 obs, 2.9%):\n', '  RMSE: 1.2730\n', '  MAE: 1.0078\n', '  R²: -21.6639\n', '\n', 'Undercapitalized (273 obs, 13.3%):\n', '  RMSE: 0.5944\n', '  MAE: 0.4997\n', '  R²: -3.6918\n', '\n', 'Adequate (403 obs, 19.7%):\n', '  RMSE: 0.4222\n', '  MAE: 0.2856\n', '  R²: -1.1226\n', '\n', 'Well-Capitalized (1311 obs, 64.1%):\n', '  RMSE: 0.5933\n', '  MAE: 0.4908\n', '  R²: -0.8611\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Ridge with CV - Test\n', '\n', 'Insolvent (54 obs, 2.6%):\n', '  RMSE: 1.0149\n', '  MAE: 0.9339\n', '  R²: -10.8189\n', '\n', 'Undercapitalized (288 obs, 14.1%):\n', '  RMSE: 0.6001\n', '  MAE: 0.5042\n', '  R²: -3.6101\n', '\n', 'Adequate (438 obs, 21.4%):\n', '  RMSE: 0.4284\n', '  MAE: 0.2980\n', '  R²: -1.1559\n', '\n', 'Well-Capitalized (1266 obs, 61.9%):\n', '  RMSE: 0.5704\n', '  MAE: 0.4675\n', '  R²: -0.7194\n', '  Ridge with CV:\n', '    CV RMSE: 0.1951 (±0.0041)\n', '    Val RMSE (Norm): 0.2012\n', '    Val RMSE (Orig): 5.95e-01\n', '    Test RMSE (Norm): 0.1907\n', '    Test RMSE (Orig): 5.64e-01\n', '\n', 'Training Lasso with CV for Quote...\n', '\n', '  Regulatory Performance Analysis for Lasso with CV:\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Lasso with CV - Validation\n', '\n', 'Insolvent (59 obs, 2.9%):\n', '  RMSE: 1.2796\n', '  MAE: 1.0214\n', '  R²: -21.8989\n', '\n', 'Undercapitalized (273 obs, 13.3%):\n', '  RMSE: 0.5988\n', '  MAE: 0.5084\n', '  R²: -3.7625\n', '\n', 'Adequate (403 obs, 19.7%):\n', '  RMSE: 0.4207\n', '  MAE: 0.2842\n', '  R²: -1.1076\n', '\n', 'Well-Capitalized (1311 obs, 64.1%):\n', '  RMSE: 0.5925\n', '  MAE: 0.4908\n', '  R²: -0.8560\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Lasso with CV - Test\n', '\n', 'Insolvent (54 obs, 2.6%):\n', '  RMSE: 1.0271\n', '  MAE: 0.9482\n', '  R²: -11.1033\n', '\n', 'Undercapitalized (288 obs, 14.1%):\n', '  RMSE: 0.6062\n', '  MAE: 0.5128\n', '  R²: -3.7033\n', '\n', 'Adequate (438 obs, 21.4%):\n', '  RMSE: 0.4271\n', '  MAE: 0.2969\n', '  R²: -1.1438\n', '\n', 'Well-Capitalized (1266 obs, 61.9%):\n', '  RMSE: 0.5697\n', '  MAE: 0.4675\n', '  R²: -0.7154\n', '  Lasso with CV:\n', '    CV RMSE: 0.1951 (±0.0041)\n', '    Val RMSE (Norm): 0.2013\n', '    Val RMSE (Orig): 5.96e-01\n', '    Test RMSE (Norm): 0.1910\n', '    Test RMSE (Orig): 5.65e-01\n', '\n', 'Training Neural Net (MLPRegressor) for Quote...\n', "  Best parameters: {'model__learning_rate_init': 0.001, 'model__hidden_layer_sizes': (128, 64), 'model__alpha': 0.0001, 'model__activation': 'tanh'}\n", '\n', '  Regulatory Performance Analysis for Neural Net (MLPRegressor):\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Neural Net (MLPRegressor) - Validation\n', '\n', 'Insolvent (59 obs, 2.9%):\n', '  RMSE: 0.6959\n', '  MAE: 0.3581\n', '  R²: -5.7737\n', '\n', 'Undercapitalized (273 obs, 13.3%):\n', '  RMSE: 0.2416\n', '  MAE: 0.1582\n', '  R²: 0.2247\n', '\n', 'Adequate (403 obs, 19.7%):\n', '  RMSE: 0.2857\n', '  MAE: 0.1653\n', '  R²: 0.0283\n', '\n', 'Well-Capitalized (1311 obs, 64.1%):\n', '  RMSE: 0.2560\n', '  MAE: 0.1487\n', '  R²: 0.6534\n', '\n', 'REGULATORY PERFORMANCE ANALYSIS - Neural Net (MLPRegressor) - Test\n', '\n', 'Insolvent (54 obs, 2.6%):\n', '  RMSE: 0.4963\n', '  MAE: 0.2362\n', '  R²: -1.8264\n', '\n', 'Undercapitalized (288 obs, 14.1%):\n', '  RMSE: 0.3300\n', '  MAE: 0.1838\n', '  R²: -0.3935\n', '\n', 'Adequate (438 obs, 21.4%):\n', '  RMSE: 0.2685\n', '  MAE: 0.1566\n', '  R²: 0.1532\n', '\n', 'Well-Capitalized (1266 obs, 61.9%):\n', '  RMSE: 0.1978\n', '  MAE: 0.1217\n', '  R²: 0.7932\n', '  Neural Net (MLPRegressor):\n', '    CV RMSE: 0.0909 (±0.0021)\n', '    Val RMSE (Norm): 0.0955\n', '    Val RMSE (Orig): 2.83e-01\n', '    Test RMSE (Norm): 0.0838\n', '    Test RMSE (Orig): 2.48e-01\n', '\n', 'TRAINING MODELS FOR SCR (SECONDARY TARGET)\n', '\n', 'Training Dummy Regressor for SCR...\n', '  Dummy Regressor:\n', '    CV RMSE: 0.3116 (±0.0084)\n', '    Val RMSE (Norm): 0.3229\n', '    Val RMSE (Orig): 2.98e+08\n', '    Test RMSE (Norm): 0.3275\n', '    Test RMSE (Orig): 3.02e+08\n', '\n', 'Training Linear Regression (MSE) for SCR...\n', '  Linear Regression (MSE):\n', '    CV RMSE: 0.1568 (±0.0038)\n', '    Val RMSE (Norm): 0.1686\n', '    Val RMSE (Orig): 1.56e+08\n', '    Test RMSE (Norm): 0.1674\n', '    Test RMSE (Orig): 1.55e+08\n', '\n', 'Training Quadratic (Degree 2) for SCR...\n', '  Quadratic (Degree 2):\n', '    CV RMSE: 0.0917 (±0.0026)\n', '    Val RMSE (Norm): 0.0997\n', '    Val RMSE (Orig): 9.21e+07\n', '    Test RMSE (Norm): 0.1053\n', '    Test RMSE (Orig): 9.73e+07\n', '\n', 'Training Cubic (Degree 3) for SCR...\n', '  Cubic (Degree 3):\n', '    CV RMSE: 0.1400 (±0.0080)\n', '    Val RMSE (Norm): 0.1542\n', '    Val RMSE (Orig): 1.42e+08\n', '    Test RMSE (Norm): 0.1397\n', '    Test RMSE (Orig): 1.29e+08\n', '\n', 'Training Elastic Net for SCR...\n', "  Best parameters: {'model__alpha': 0.001, 'model__l1_ratio': 0.1}\n", '  Elastic Net:\n', '    CV RMSE: 0.1568 (±0.0038)\n', '    Val RMSE (Norm): 0.1687\n', '    Val RMSE (Orig): 1.56e+08\n', '    Test RMSE (Norm): 0.1675\n', '    Test RMSE (Orig): 1.55e+08\n', '\n', 'Training Ridge with CV for SCR...\n', '  Ridge with CV:\n', '    CV RMSE: 0.1568 (±0.0038)\n', '    Val RMSE (Norm): 0.1687\n', '    Val RMSE (Orig): 1.56e+08\n', '    Test RMSE (Norm): 0.1675\n', '    Test RMSE (Orig): 1.55e+08\n', '\n', 'Training Lasso with CV for SCR...\n', '  Lasso with CV:\n', '    CV RMSE: 0.1569 (±0.0038)\n', '    Val RMSE (Norm): 0.1687\n', '    Val RMSE (Orig): 1.56e+08\n', '    Test RMSE (Norm): 0.1677\n', '    Test RMSE (Orig): 1.55e+08\n', '\n', 'Training Neural Net (MLPRegressor) for SCR...\n', "  Best parameters: {'model__learning_rate_init': 0.001, 'model__hidden_layer_sizes': (128, 64, 32), 'model__alpha': 0.0001, 'model__activation': 'tanh'}\n", '  Neural Net (MLPRegressor):\n', '    CV RMSE: 0.0814 (±0.0037)\n', '    Val RMSE (Norm): 0.0886\n', '    Val RMSE (Orig): 8.18e+07\n', '    Test RMSE (Norm): 0.0932\n', '    Test RMSE (Orig): 8.60e+07\n', '\n', 'TRAINING MODELS FOR EM (SECONDARY TARGET)\n', '\n', 'Training Dummy Regressor for EM...\n', '  Dummy Regressor:\n', '    CV RMSE: 0.2017 (±0.0091)\n', '    Val RMSE (Norm): 0.2067\n', '    Val RMSE (Orig): 4.36e+08\n', '    Test RMSE (Norm): 0.2054\n', '    Test RMSE (Orig): 4.34e+08\n', '\n', 'Training Linear Regression (MSE) for EM...\n', '  Linear Regression (MSE):\n', '    CV RMSE: 0.1531 (±0.0067)\n', '    Val RMSE (Norm): 0.1556\n', '    Val RMSE (Orig): 3.28e+08\n', '    Test RMSE (Norm): 0.1525\n', '    Test RMSE (Orig): 3.22e+08\n', '\n', 'Training Quadratic (Degree 2) for EM...\n', '  Quadratic (Degree 2):\n', '    CV RMSE: 0.0716 (±0.0042)\n', '    Val RMSE (Norm): 0.0763\n', '    Val RMSE (Orig): 1.61e+08\n', '    Test RMSE (Norm): 0.0746\n', '    Test RMSE (Orig): 1.57e+08\n', '\n', 'Training Cubic (Degree 3) for EM...\n', '  Cubic (Degree 3):\n', '    CV RMSE: 0.1427 (±0.0124)\n', '    Val RMSE (Norm): 0.1425\n', '    Val RMSE (Orig): 3.01e+08\n', '    Test RMSE (Norm): 0.1173\n', '    Test RMSE (Orig): 2.48e+08\n', '\n', 'Training Elastic Net for EM...\n', "  Best parameters: {'model__alpha': 0.001, 'model__l1_ratio': 0.7}\n", '  Elastic Net:\n', '    CV RMSE: 0.1531 (±0.0068)\n', '    Val RMSE (Norm): 0.1557\n', '    Val RMSE (Orig): 3.29e+08\n', '    Test RMSE (Norm): 0.1528\n', '    Test RMSE (Orig): 3.23e+08\n', '\n', 'Training Ridge with CV for EM...\n', '  Ridge with CV:\n', '    CV RMSE: 0.1532 (±0.0067)\n', '    Val RMSE (Norm): 0.1556\n', '    Val RMSE (Orig): 3.28e+08\n', '    Test RMSE (Norm): 0.1526\n', '    Test RMSE (Orig): 3.22e+08\n', '\n', 'Training Lasso with CV for EM...\n', '  Lasso with CV:\n', '    CV RMSE: 0.1531 (±0.0069)\n', '    Val RMSE (Norm): 0.1558\n', '    Val RMSE (Orig): 3.29e+08\n', '    Test RMSE (Norm): 0.1529\n', '    Test RMSE (Orig): 3.23e+08\n', '\n', 'Training Neural Net (MLPRegressor) for EM...\n', "  Best parameters: {'model__learning_rate_init': 0.001, 'model__hidden_layer_sizes': (128, 64, 32), 'model__alpha': 0.0001, 'model__activation': 'tanh'}\n", '  Neural Net (MLPRegressor):\n', '    CV RMSE: 0.0691 (±0.0071)\n', '    Val RMSE (Norm): 0.0822\n', '    Val RMSE (Orig): 1.73e+08\n', '    Test RMSE (Norm): 0.0745\n', '    Test RMSE (Orig): 1.57e+08\n', '\n', 'TRAINING MULTI-OUTPUT MODELS FOR SCR & EM\n', '\n', 'Training Dummy Regressor (Multi-Output)...\n', '    Model Performance Summary:\n', '    CV RMSE: 0.2625 (±0.0084)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.3117\n', '      Val RMSE (norm):   0.3229\n', '      Test RMSE (norm):  0.3275\n', '      Train RMSE (orig): 2.88e+08\n', '      Val RMSE (orig):   2.98e+08\n', '      Test RMSE (orig):  3.02e+08\n', '    EM Performance:\n', '      Train RMSE (norm): 0.2019\n', '      Val RMSE (norm):   0.2067\n', '      Test RMSE (norm):  0.2054\n', '      Train RMSE (orig): 4.26e+08\n', '      Val RMSE (orig):   4.36e+08\n', '      Test RMSE (orig):  4.34e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.2626\n', '      Val RMSE (norm):   0.2711\n', '      Test RMSE (norm):  0.2733\n', '      Train RMSE (orig): 3.64e+08\n', '      Val RMSE (orig):   3.74e+08\n', '      Test RMSE (orig):  3.74e+08\n', '  Dummy Regressor (Multi-Output):\n', '    Val RMSE (SCR): 0.3229\n', '    Val RMSE (EM): 0.2067\n', '    Val RMSE (Macro): 0.2711\n', '\n', 'Training Linear Regression (MSE) (Multi-Output)...\n', '    Model Performance Summary:\n', '    CV RMSE: 0.1550 (±0.0048)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.1560\n', '      Val RMSE (norm):   0.1686\n', '      Test RMSE (norm):  0.1674\n', '      Train RMSE (orig): 1.44e+08\n', '      Val RMSE (orig):   1.56e+08\n', '      Test RMSE (orig):  1.55e+08\n', '    EM Performance:\n', '      Train RMSE (norm): 0.1525\n', '      Val RMSE (norm):   0.1556\n', '      Test RMSE (norm):  0.1525\n', '      Train RMSE (orig): 3.22e+08\n', '      Val RMSE (orig):   3.28e+08\n', '      Test RMSE (orig):  3.22e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.1542\n', '      Val RMSE (norm):   0.1622\n', '      Test RMSE (norm):  0.1602\n', '      Train RMSE (orig): 2.49e+08\n', '      Val RMSE (orig):   2.57e+08\n', '      Test RMSE (orig):  2.53e+08\n', '  Linear Regression (MSE) (Multi-Output):\n', '    Val RMSE (SCR): 0.1686\n', '    Val RMSE (EM): 0.1556\n', '    Val RMSE (Macro): 0.1622\n', '\n', 'Training Quadratic (Degree 2) (Multi-Output)...\n', '    Model Performance Summary:\n', '    CV RMSE: 0.0823 (±0.0019)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.0861\n', '      Val RMSE (norm):   0.0997\n', '      Test RMSE (norm):  0.1053\n', '      Train RMSE (orig): 7.95e+07\n', '      Val RMSE (orig):   9.21e+07\n', '      Test RMSE (orig):  9.73e+07\n', '    EM Performance:\n', '      Train RMSE (norm): 0.0662\n', '      Val RMSE (norm):   0.0763\n', '      Test RMSE (norm):  0.0746\n', '      Train RMSE (orig): 1.40e+08\n', '      Val RMSE (orig):   1.61e+08\n', '      Test RMSE (orig):  1.57e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.0768\n', '      Val RMSE (norm):   0.0888\n', '      Test RMSE (norm):  0.0912\n', '      Train RMSE (orig): 1.14e+08\n', '      Val RMSE (orig):   1.31e+08\n', '      Test RMSE (orig):  1.31e+08\n', '      Overfitting detected: Val-Train gap = 0.0120\n', '  Quadratic (Degree 2) (Multi-Output):\n', '    Val RMSE (SCR): 0.0997\n', '    Val RMSE (EM): 0.0763\n', '    Val RMSE (Macro): 0.0888\n', '\n', 'Training Cubic (Degree 3) (Multi-Output)...\n', '    Model Performance Summary:\n', '    CV RMSE: 0.1414 (±0.0099)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.0700\n', '      Val RMSE (norm):   0.1542\n', '      Test RMSE (norm):  0.1397\n', '      Train RMSE (orig): 6.47e+07\n', '      Val RMSE (orig):   1.42e+08\n', '      Test RMSE (orig):  1.29e+08\n', '    EM Performance:\n', '      Train RMSE (norm): 0.0520\n', '      Val RMSE (norm):   0.1425\n', '      Test RMSE (norm):  0.1173\n', '      Train RMSE (orig): 1.10e+08\n', '      Val RMSE (orig):   3.01e+08\n', '      Test RMSE (orig):  2.48e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.0617\n', '      Val RMSE (norm):   0.1485\n', '      Test RMSE (norm):  0.1290\n', '      Train RMSE (orig): 9.02e+07\n', '      Val RMSE (orig):   2.35e+08\n', '      Test RMSE (orig):  1.97e+08\n', '      Overfitting detected: Val-Train gap = 0.0868\n', '  Cubic (Degree 3) (Multi-Output):\n', '    Val RMSE (SCR): 0.1542\n', '    Val RMSE (EM): 0.1425\n', '    Val RMSE (Macro): 0.1485\n', '\n', 'Training Elastic Net (Multi-Output)...\n', '    Model Performance Summary:\n', '    CV RMSE: 0.2625 (±0.0084)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.3117\n', '      Val RMSE (norm):   0.3229\n', '      Test RMSE (norm):  0.3275\n', '      Train RMSE (orig): 2.88e+08\n', '      Val RMSE (orig):   2.98e+08\n', '      Test RMSE (orig):  3.02e+08\n', '    EM Performance:\n', '      Train RMSE (norm): 0.2019\n', '      Val RMSE (norm):   0.2067\n', '      Test RMSE (norm):  0.2054\n', '      Train RMSE (orig): 4.26e+08\n', '      Val RMSE (orig):   4.36e+08\n', '      Test RMSE (orig):  4.34e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.2626\n', '      Val RMSE (norm):   0.2711\n', '      Test RMSE (norm):  0.2733\n', '      Train RMSE (orig): 3.64e+08\n', '      Val RMSE (orig):   3.74e+08\n', '      Test RMSE (orig):  3.74e+08\n', '  Elastic Net (Multi-Output):\n', '    Val RMSE (SCR): 0.3229\n', '    Val RMSE (EM): 0.2067\n', '    Val RMSE (Macro): 0.2711\n', '\n', 'Training Ridge with CV (Multi-Output)...\n', '    Model Performance Summary:\n', '    CV RMSE: 0.1550 (±0.0048)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.1560\n', '      Val RMSE (norm):   0.1687\n', '      Test RMSE (norm):  0.1675\n', '      Train RMSE (orig): 1.44e+08\n', '      Val RMSE (orig):   1.56e+08\n', '      Test RMSE (orig):  1.55e+08\n', '    EM Performance:\n', '      Train RMSE (norm): 0.1525\n', '      Val RMSE (norm):   0.1556\n', '      Test RMSE (norm):  0.1526\n', '      Train RMSE (orig): 3.22e+08\n', '      Val RMSE (orig):   3.28e+08\n', '      Test RMSE (orig):  3.22e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.1542\n', '      Val RMSE (norm):   0.1622\n', '      Test RMSE (norm):  0.1602\n', '      Train RMSE (orig): 2.49e+08\n', '      Val RMSE (orig):   2.57e+08\n', '      Test RMSE (orig):  2.53e+08\n', '  Ridge with CV (Multi-Output):\n', '    Val RMSE (SCR): 0.1687\n', '    Val RMSE (EM): 0.1556\n', '    Val RMSE (Macro): 0.1622\n', '\n', 'Training Lasso with CV (Multi-Output)...\n', '    Model Performance Summary:\n', '    CV RMSE: 0.1550 (±0.0048)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.1561\n', '      Val RMSE (norm):   0.1687\n', '      Test RMSE (norm):  0.1677\n', '      Train RMSE (orig): 1.44e+08\n', '      Val RMSE (orig):   1.56e+08\n', '      Test RMSE (orig):  1.55e+08\n', '    EM Performance:\n', '      Train RMSE (norm): 0.1525\n', '      Val RMSE (norm):   0.1558\n', '      Test RMSE (norm):  0.1529\n', '      Train RMSE (orig): 3.22e+08\n', '      Val RMSE (orig):   3.29e+08\n', '      Test RMSE (orig):  3.23e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.1543\n', '      Val RMSE (norm):   0.1624\n', '      Test RMSE (norm):  0.1605\n', '      Train RMSE (orig): 2.49e+08\n', '      Val RMSE (orig):   2.57e+08\n', '      Test RMSE (orig):  2.53e+08\n', '  Lasso with CV (Multi-Output):\n', '    Val RMSE (SCR): 0.1687\n', '    Val RMSE (EM): 0.1558\n', '    Val RMSE (Macro): 0.1624\n', '\n', 'Training Neural Net (MLPRegressor) (Multi-Output)...\n', '    Model Performance Summary:\n', '    CV RMSE: 0.0774 (±0.0044)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.0497\n', '      Val RMSE (norm):   0.0857\n', '      Test RMSE (norm):  0.0991\n', '      Train RMSE (orig): 4.59e+07\n', '      Val RMSE (orig):   7.92e+07\n', '      Test RMSE (orig):  9.15e+07\n', '    EM Performance:\n', '      Train RMSE (norm): 0.0375\n', '      Val RMSE (norm):   0.0641\n', '      Test RMSE (norm):  0.0712\n', '      Train RMSE (orig): 7.92e+07\n', '      Val RMSE (orig):   1.35e+08\n', '      Test RMSE (orig):  1.50e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.0440\n', '      Val RMSE (norm):   0.0757\n', '      Test RMSE (norm):  0.0863\n', '      Train RMSE (orig): 6.47e+07\n', '      Val RMSE (orig):   1.11e+08\n', '      Test RMSE (orig):  1.24e+08\n', '      Overfitting detected: Val-Train gap = 0.0317\n', '  Neural Net (MLPRegressor) (Multi-Output):\n', '    Val RMSE (SCR): 0.0857\n', '    Val RMSE (EM): 0.0641\n', '    Val RMSE (Macro): 0.0757\n', '\n', 'Training Neural Net (PyTorch MLP Dropout Multi) (Multi-Output)...\n', '    Model Performance Summary:\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.0616\n', '      Val RMSE (norm):   0.0819\n', '      Test RMSE (norm):  0.0892\n', '      Train RMSE (orig): 5.69e+07\n', '      Val RMSE (orig):   7.56e+07\n', '      Test RMSE (orig):  8.24e+07\n', '    EM Performance:\n', '      Train RMSE (norm): 0.0485\n', '      Val RMSE (norm):   0.0613\n', '      Test RMSE (norm):  0.0608\n', '      Train RMSE (orig): 1.02e+08\n', '      Val RMSE (orig):   1.29e+08\n', '      Test RMSE (orig):  1.28e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.0555\n', '      Val RMSE (norm):   0.0723\n', '      Test RMSE (norm):  0.0763\n', '      Train RMSE (orig): 8.29e+07\n', '      Val RMSE (orig):   1.06e+08\n', '      Test RMSE (orig):  1.08e+08\n', '      Overfitting detected: Val-Train gap = 0.0168\n', '  Neural Net (PyTorch MLP Dropout Multi) (Multi-Output):\n', '    Val RMSE (SCR): 0.0819\n', '    Val RMSE (EM): 0.0613\n', '    Val RMSE (Macro): 0.0723\n']
# --- End Output ---

# In[78]:  (cell 29)
# Fine tuning the parameters of the PyTorch Multi
param_grid = {
    'module__hidden_sizes': [(256, 128), (128,64,32)],
    'module__dropout': [0.2, 0.3, 0.4],
    'module__batchnorm': [True, False],
    'optimizer__lr': [1e-3, 5e-4, 1e-2],
    'optimizer__weight_decay': [0, 1e-4, 1e-3],
    'max_epochs': [200, 300]
}

random_search= RandomizedSearchCV(
    nn_torch_dropout_multi,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE   
)

random_search.fit(X_train.to_numpy(), y_multi_train.astype(np.float32).to_numpy())  # Ensure float32 for PyTorch
print(f"Best parameters from Randomized Search: {random_search.best_params_}")

torch_multi_tuned = copy.deepcopy(random_search.best_estimator_)

# --- Output [29] ---
# [1] type: stream
# (stdout)
# ['Fitting 5 folds for each of 20 candidates, totalling 100 fits\n', "Best parameters from Randomized Search: {'optimizer__weight_decay': 0.001, 'optimizer__lr': 0.001, 'module__hidden_sizes': (256, 128), 'module__dropout': 0.2, 'module__batchnorm': False, 'max_epochs': 300}\n"]
# --- End Output ---

# In[79]:  (cell 30)
# Fine tuning the parameters of the MLPRegressor (mutli-output)
param_grid_mlp = {
    'hidden_layer_sizes': [(128, 64), (64, 32), (256, 128, 64)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [700, 500]
}
random_search_mlp = RandomizedSearchCV(
    MLPRegressor(random_state=RANDOM_STATE),
    param_distributions=param_grid_mlp,
    n_iter=20,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE   
)
random_search_mlp.fit(X_train, y_multi_train)
print(f"Best parameters from Randomized Search for MLPRegressor: {random_search_mlp.best_params_}")

mlp_multi_tuned = copy.deepcopy(random_search_mlp.best_estimator_)

# --- Output [30] ---
# [1] type: stream
# (stdout)
# ['Fitting 5 folds for each of 20 candidates, totalling 100 fits\n', "Best parameters from Randomized Search for MLPRegressor: {'max_iter': 500, 'learning_rate_init': 0.01, 'hidden_layer_sizes': (256, 128, 64), 'alpha': 0.001, 'activation': 'relu'}\n"]
# --- End Output ---

# In[80]:  (cell 31)
# Enhanced multi-output models configuration
enhanced_models_config = {
    'linear_multi': {
        'name': 'Linear Regression (Multi-Output)',
        'model': LinearRegression(),
        'loss': 'MSE'
    },
    'ridge_multi': {
        'name': 'Ridge Regression (Multi-Output)',
        'model': RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5),
        'loss': 'MSE'
    },
    'poly2_multi': {
        'name': 'Quadratic (Multi-Output)',
        'model': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE))
        ]),
        'loss': 'MSE'
    },
    'random_forest_multi': {
        'name': 'Random Forest (Multi-Output)',
        'model': RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
        'loss': 'MSE'
    },
    'xgboost_basic': {
        'name': 'XGBoost Basic (Multi-Output)',
        'model': xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'loss': 'MSE'
    },
    'xgboost_tuned': {
        'name': 'XGBoost Tuned (Multi-Output)',
        'model': xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'loss': 'MSE'
    }
}

enhanced_models_config.update({
    'mlp_reg_multi': {
        'name': 'MLPRegressor (Multi-Output)',
        'model': MLPRegressor(hidden_layer_sizes=(256,128,64),
                              activation='relu',
                              solver='adam',
                              max_iter=500,
                              early_stopping=True,
                              validation_fraction=0.1,
                              n_iter_no_change=20,
                              alpha=1e-5,
                              learning_rate_init=1e-3,
                              random_state=RANDOM_STATE),
        'loss': 'MSE'
    },
    'torch_nn_multi': {
        'name': 'PyTorch Neural Net (Multi-Output)',
        'model': nn_torch_dropout_multi,
        'loss': 'MSE',
    },
    'torch_nn_multi_tuned': {
        'name': 'PyTorch Neural Net Tuned (Multi-Output)',
        'model': torch_multi_tuned,
        'loss': 'MSE',
    },
    'mlp_multi_tuned': {
        'name': 'MLPRegressor Tuned (Multi-Output)',
        'model': mlp_multi_tuned,
        'loss': 'MSE',
    }
})

# In[81]:  (cell 32)
enhanced_results_fixed = {}

for model_key, model_config in enhanced_models_config.items():
    try:
        results = train_multioutput_model_consolidated(
            model_config, X_train, X_val, X_test,
            y_multi_train, y_multi_val, y_multi_test,  # [SCR, EM] targets (normalized)
            preprocessor, calculate_quote=True
        )
        if results is not None:
            enhanced_results_fixed[model_key] = results
        
    except Exception as e:
        print(f"  Error training {model_config['name']}: {str(e)}")

def create_fixed_joint_comparison(results_dict):
    """Create comparison table for FIXED joint EM-SCR models"""
    comparison_data = []
    
    for model_key, results in results_dict.items():
        if results is None:
            continue
            
        metrics = results['metrics']
        row = {
            'Model': results['model_name'],
            'SCR_Val_RMSE': metrics.get('val_SCR_RMSE', np.nan),
            'EM_Val_RMSE': metrics.get('val_EM_RMSE', np.nan),
            'Quote_Val_RMSE_Norm': metrics.get('val_Quote_RMSE', np.nan),  # Normalized for comparison
            'Quote_Test_RMSE_Norm': metrics.get('test_Quote_RMSE', np.nan),
            'Quote_Test_R2_Norm': metrics.get('test_Quote_R2', np.nan),
            'Quote_Test_RMSE_Orig': metrics.get('test_Quote_orig_RMSE', np.nan),  # Original for business
            'Quote_Test_R2_Orig': metrics.get('test_Quote_orig_R2', np.nan)
        }
        comparison_data.append(row)
    
    if not comparison_data:
        print(" No successful models to compare!")
        return pd.DataFrame()
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Quote_Val_RMSE_Norm')
    

    print("="*120)
    print("NOTE: Quote calculated in ORIGINAL scale then normalized for fair comparison")
    print("="*120)
    print(df_comparison.round(4).to_string(index=False))
    
    return df_comparison

# Create comparison table
joint_comparison_fixed = create_fixed_joint_comparison(enhanced_results_fixed)

# Find best joint model
best_joint_model_fixed = None
best_quote_rmse_fixed = float('inf')

for model_key, results in enhanced_results_fixed.items():
    if results is not None:
        quote_rmse = results['metrics'].get('val_Quote_RMSE', float('inf'))
        if quote_rmse < best_quote_rmse_fixed:
            best_quote_rmse_fixed = quote_rmse
            best_joint_model_fixed = results

if best_joint_model_fixed:
    print(f"\n BEST FIXED JOINT MODEL:")
    print(f"Model: {best_joint_model_fixed['model_name']}")
    print(f"Quote Val RMSE (normalized): {best_quote_rmse_fixed:.4f}")
    print(f"Quote Test RMSE (normalized): {best_joint_model_fixed['metrics']['test_Quote_RMSE']:.4f}")
    print(f"Quote Test R² (normalized): {best_joint_model_fixed['metrics']['test_Quote_R2']:.4f}")
    
    best_models = {}
    # Find best Quote model from all_results
    best_quote_rmse_norm = float('inf')
    for model_key, results in all_results['quote'].items():
        if results is not None:
            val_rmse = results['metrics'].get('val_norm_RMSE', float('inf'))
            if val_rmse < best_quote_rmse_norm:
                best_quote_rmse_norm = val_rmse
                best_models['quote'] = {
                    'results': results,
                    'val_rmse_norm': val_rmse
                }
    # Compare with direct Quote prediction
    if 'quote' in best_models:
        direct_quote_rmse = best_models['quote']['val_rmse_norm']
        joint_quote_rmse_fixed = best_quote_rmse_fixed
        
        print(f"\n FIXED APPROACH COMPARISON:")
        print(f"Direct Quote Prediction RMSE: {direct_quote_rmse:.4f}")
        print(f"Fixed Joint EM-SCR → Quote RMSE: {joint_quote_rmse_fixed:.4f}")
        
        if joint_quote_rmse_fixed < direct_quote_rmse:
            improvement = ((direct_quote_rmse - joint_quote_rmse_fixed) / direct_quote_rmse) * 100
            print(f" Fixed joint approach is BETTER by {improvement:.1f}%")
        else:
            degradation = ((joint_quote_rmse_fixed - direct_quote_rmse) / direct_quote_rmse) * 100
            print(f" Fixed joint approach is still worse by {degradation:.1f}%")
            
        # Analysis
        print(f"\n ANALYSIS:")
        scr_em_corr = np.corrcoef(y_multi_train.iloc[:, 0], y_multi_train.iloc[:, 1])[0, 1]
        print(f"SCR-EM correlation: {scr_em_corr:.3f}")
        
        if joint_quote_rmse_fixed > direct_quote_rmse:
            print(f"\n WHY JOINT MODELING MAY NOT HELP:")
            print(f"1. Negative correlation (-0.54) makes joint learning harder")
            print(f"2. Division amplifies small prediction errors")
            print(f"3. Quote relationship may be better learned directly")
            print(f"4. Recommendation: Use direct Quote prediction approach")
        else:
            print(f"\n WHY JOINT MODELING HELPS:")
            print(f"1. Captures shared dependencies (ZSK1-ZSK3)")
            print(f"2. Preserves EM/SCR mathematical relationship")
            print(f"3. Recommendation: Use joint approach")
        
else:
    print("\n No fixed joint models trained successfully!")

# --- Output [32] ---
# [1] type: stream
# (stdout)
# ['\n', 'Training Linear Regression (Multi-Output) (Multi-Output)...\n', '    Quote calculation successful:\n', '    Quote Train RMSE (norm): 10.1863\n', '    Quote Val RMSE (norm): 1.2853\n', '    Quote Test RMSE (norm): 6.6605\n', '    Quote Test R² (norm): -376.5974\n', '    Model Performance Summary:\n', '    CV RMSE: 0.1550 (±0.0048)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.1560\n', '      Val RMSE (norm):   0.1686\n', '      Test RMSE (norm):  0.1674\n', '      Train RMSE (orig): 1.44e+08\n', '      Val RMSE (orig):   1.56e+08\n', '      Test RMSE (orig):  1.55e+08\n', '    EM Performance:\n', '      Train RMSE (norm): 0.1525\n', '      Val RMSE (norm):   0.1556\n', '      Test RMSE (norm):  0.1525\n', '      Train RMSE (orig): 3.22e+08\n', '      Val RMSE (orig):   3.28e+08\n', '      Test RMSE (orig):  3.22e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.1542\n', '      Val RMSE (norm):   0.1622\n', '      Test RMSE (norm):  0.1602\n', '      Train RMSE (orig): 2.49e+08\n', '      Val RMSE (orig):   2.57e+08\n', '      Test RMSE (orig):  2.53e+08\n', '\n', 'Training Ridge Regression (Multi-Output) (Multi-Output)...\n', '    Quote calculation successful:\n', '    Quote Train RMSE (norm): 39.0797\n', '    Quote Val RMSE (norm): 1.2388\n', '    Quote Test RMSE (norm): 4.4102\n', '    Quote Test R² (norm): -164.5481\n', '    Model Performance Summary:\n', '    CV RMSE: 0.1550 (±0.0048)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.1560\n', '      Val RMSE (norm):   0.1687\n', '      Test RMSE (norm):  0.1675\n', '      Train RMSE (orig): 1.44e+08\n', '      Val RMSE (orig):   1.56e+08\n', '      Test RMSE (orig):  1.55e+08\n', '    EM Performance:\n', '      Train RMSE (norm): 0.1525\n', '      Val RMSE (norm):   0.1556\n', '      Test RMSE (norm):  0.1526\n', '      Train RMSE (orig): 3.22e+08\n', '      Val RMSE (orig):   3.28e+08\n', '      Test RMSE (orig):  3.22e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.1542\n', '      Val RMSE (norm):   0.1622\n', '      Test RMSE (norm):  0.1602\n', '      Train RMSE (orig): 2.49e+08\n', '      Val RMSE (orig):   2.57e+08\n', '      Test RMSE (orig):  2.53e+08\n', '\n', 'Training Quadratic (Multi-Output) (Multi-Output)...\n', '    Quote calculation successful:\n', '    Quote Train RMSE (norm): 0.1349\n', '    Quote Val RMSE (norm): 0.1541\n', '    Quote Test RMSE (norm): 0.1632\n', '    Quote Test R² (norm): 0.7734\n', '    Model Performance Summary:\n', '    CV RMSE: 0.0823 (±0.0019)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.0861\n', '      Val RMSE (norm):   0.0997\n', '      Test RMSE (norm):  0.1053\n', '      Train RMSE (orig): 7.95e+07\n', '      Val RMSE (orig):   9.21e+07\n', '      Test RMSE (orig):  9.73e+07\n', '    EM Performance:\n', '      Train RMSE (norm): 0.0662\n', '      Val RMSE (norm):   0.0763\n', '      Test RMSE (norm):  0.0746\n', '      Train RMSE (orig): 1.40e+08\n', '      Val RMSE (orig):   1.61e+08\n', '      Test RMSE (orig):  1.57e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.0768\n', '      Val RMSE (norm):   0.0888\n', '      Test RMSE (norm):  0.0912\n', '      Train RMSE (orig): 1.14e+08\n', '      Val RMSE (orig):   1.31e+08\n', '      Test RMSE (orig):  1.31e+08\n', '      Overfitting detected: Val-Train gap = 0.0120\n', '\n', 'Training Random Forest (Multi-Output) (Multi-Output)...\n', '    Quote calculation successful:\n', '    Quote Train RMSE (norm): 0.0408\n', '    Quote Val RMSE (norm): 0.1015\n', '    Quote Test RMSE (norm): 0.1033\n', '    Quote Test R² (norm): 0.9091\n', '    Model Performance Summary:\n', '    CV RMSE: 0.0809 (±0.0025)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.0334\n', '      Val RMSE (norm):   0.0950\n', '      Test RMSE (norm):  0.0998\n', '      Train RMSE (orig): 3.08e+07\n', '      Val RMSE (orig):   8.77e+07\n', '      Test RMSE (orig):  9.21e+07\n', '    EM Performance:\n', '      Train RMSE (norm): 0.0276\n', '      Val RMSE (norm):   0.0783\n', '      Test RMSE (norm):  0.0799\n', '      Train RMSE (orig): 5.82e+07\n', '      Val RMSE (orig):   1.65e+08\n', '      Test RMSE (orig):  1.69e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.0306\n', '      Val RMSE (norm):   0.0870\n', '      Test RMSE (norm):  0.0904\n', '      Train RMSE (orig): 4.66e+07\n', '      Val RMSE (orig):   1.32e+08\n', '      Test RMSE (orig):  1.36e+08\n', '      Overfitting detected: Val-Train gap = 0.0564\n', '\n', 'Training XGBoost Basic (Multi-Output) (Multi-Output)...\n', '    Quote calculation successful:\n', '    Quote Train RMSE (norm): 0.0459\n', '    Quote Val RMSE (norm): 0.0851\n', '    Quote Test RMSE (norm): 0.0832\n', '    Quote Test R² (norm): 0.9411\n', '    Model Performance Summary:\n', '    CV RMSE: 0.0638 (±0.0018)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.0309\n', '      Val RMSE (norm):   0.0775\n', '      Test RMSE (norm):  0.0855\n', '      Train RMSE (orig): 2.85e+07\n', '      Val RMSE (orig):   7.16e+07\n', '      Test RMSE (orig):  7.90e+07\n', '    EM Performance:\n', '      Train RMSE (norm): 0.0178\n', '      Val RMSE (norm):   0.0571\n', '      Test RMSE (norm):  0.0583\n', '      Train RMSE (orig): 3.76e+07\n', '      Val RMSE (orig):   1.21e+08\n', '      Test RMSE (orig):  1.23e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.0252\n', '      Val RMSE (norm):   0.0681\n', '      Test RMSE (norm):  0.0732\n', '      Train RMSE (orig): 3.34e+07\n', '      Val RMSE (orig):   9.92e+07\n', '      Test RMSE (orig):  1.03e+08\n', '      Overfitting detected: Val-Train gap = 0.0429\n', '\n', 'Training XGBoost Tuned (Multi-Output) (Multi-Output)...\n', '    Quote calculation successful:\n', '    Quote Train RMSE (norm): 0.0349\n', '    Quote Val RMSE (norm): 0.0807\n', '    Quote Test RMSE (norm): 0.0802\n', '    Quote Test R² (norm): 0.9453\n', '    Model Performance Summary:\n', '    CV RMSE: 0.0620 (±0.0021)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.0220\n', '      Val RMSE (norm):   0.0764\n', '      Test RMSE (norm):  0.0840\n', '      Train RMSE (orig): 2.03e+07\n', '      Val RMSE (orig):   7.05e+07\n', '      Test RMSE (orig):  7.75e+07\n', '    EM Performance:\n', '      Train RMSE (norm): 0.0133\n', '      Val RMSE (norm):   0.0555\n', '      Test RMSE (norm):  0.0562\n', '      Train RMSE (orig): 2.81e+07\n', '      Val RMSE (orig):   1.17e+08\n', '      Test RMSE (orig):  1.19e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.0182\n', '      Val RMSE (norm):   0.0668\n', '      Test RMSE (norm):  0.0715\n', '      Train RMSE (orig): 2.45e+07\n', '      Val RMSE (orig):   9.67e+07\n', '      Test RMSE (orig):  1.00e+08\n', '      Overfitting detected: Val-Train gap = 0.0486\n', '\n', 'Training MLPRegressor (Multi-Output) (Multi-Output)...\n', '    Quote calculation successful:\n', '    Quote Train RMSE (norm): 0.0479\n', '    Quote Val RMSE (norm): 0.1273\n', '    Quote Test RMSE (norm): 0.1256\n', '    Quote Test R² (norm): 0.8657\n', '    Model Performance Summary:\n', '    CV RMSE: 0.0708 (±0.0018)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.0320\n', '      Val RMSE (norm):   0.0784\n', '      Test RMSE (norm):  0.0814\n', '      Train RMSE (orig): 2.96e+07\n', '      Val RMSE (orig):   7.24e+07\n', '      Test RMSE (orig):  7.52e+07\n', '    EM Performance:\n', '      Train RMSE (norm): 0.0243\n', '      Val RMSE (norm):   0.0651\n', '      Test RMSE (norm):  0.0623\n', '      Train RMSE (orig): 5.14e+07\n', '      Val RMSE (orig):   1.37e+08\n', '      Test RMSE (orig):  1.32e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.0284\n', '      Val RMSE (norm):   0.0721\n', '      Test RMSE (norm):  0.0725\n', '      Train RMSE (orig): 4.19e+07\n', '      Val RMSE (orig):   1.10e+08\n', '      Test RMSE (orig):  1.07e+08\n', '      Overfitting detected: Val-Train gap = 0.0437\n', '\n', 'Training PyTorch Neural Net (Multi-Output) (Multi-Output)...\n', '    Quote calculation successful:\n', '    Quote Train RMSE (norm): 0.0734\n', '    Quote Val RMSE (norm): 0.0901\n', '    Quote Test RMSE (norm): 0.0824\n', '    Quote Test R² (norm): 0.9422\n', '    Model Performance Summary:\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.0574\n', '      Val RMSE (norm):   0.0768\n', '      Test RMSE (norm):  0.0876\n', '      Train RMSE (orig): 5.30e+07\n', '      Val RMSE (orig):   7.09e+07\n', '      Test RMSE (orig):  8.09e+07\n', '    EM Performance:\n', '      Train RMSE (norm): 0.0436\n', '      Val RMSE (norm):   0.0558\n', '      Test RMSE (norm):  0.0563\n', '      Train RMSE (orig): 9.20e+07\n', '      Val RMSE (orig):   1.18e+08\n', '      Test RMSE (orig):  1.19e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.0509\n', '      Val RMSE (norm):   0.0671\n', '      Test RMSE (norm):  0.0737\n', '      Train RMSE (orig): 7.51e+07\n', '      Val RMSE (orig):   9.72e+07\n', '      Test RMSE (orig):  1.02e+08\n', '      Overfitting detected: Val-Train gap = 0.0162\n', '\n', 'Training PyTorch Neural Net Tuned (Multi-Output) (Multi-Output)...\n', '    Quote calculation successful:\n', '    Quote Train RMSE (norm): 0.0499\n', '    Quote Val RMSE (norm): 0.0723\n', '    Quote Test RMSE (norm): 0.0645\n', '    Quote Test R² (norm): 0.9646\n', '    Model Performance Summary:\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.0420\n', '      Val RMSE (norm):   0.0684\n', '      Test RMSE (norm):  0.0791\n', '      Train RMSE (orig): 3.88e+07\n', '      Val RMSE (orig):   6.32e+07\n', '      Test RMSE (orig):  7.30e+07\n', '    EM Performance:\n', '      Train RMSE (norm): 0.0315\n', '      Val RMSE (norm):   0.0501\n', '      Test RMSE (norm):  0.0514\n', '      Train RMSE (orig): 6.66e+07\n', '      Val RMSE (orig):   1.06e+08\n', '      Test RMSE (orig):  1.09e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.0372\n', '      Val RMSE (norm):   0.0600\n', '      Test RMSE (norm):  0.0667\n', '      Train RMSE (orig): 5.45e+07\n', '      Val RMSE (orig):   8.72e+07\n', '      Test RMSE (orig):  9.25e+07\n', '      Overfitting detected: Val-Train gap = 0.0228\n', '\n', 'Training MLPRegressor Tuned (Multi-Output) (Multi-Output)...\n', '    Quote calculation successful:\n', '    Quote Train RMSE (norm): 0.0587\n', '    Quote Val RMSE (norm): 0.0852\n', '    Quote Test RMSE (norm): 0.0822\n', '    Quote Test R² (norm): 0.9424\n', '    Model Performance Summary:\n', '    CV RMSE: 0.0613 (±0.0013)\n', '    SCR Performance:\n', '      Train RMSE (norm): 0.0425\n', '      Val RMSE (norm):   0.0696\n', '      Test RMSE (norm):  0.0807\n', '      Train RMSE (orig): 3.93e+07\n', '      Val RMSE (orig):   6.42e+07\n', '      Test RMSE (orig):  7.45e+07\n', '    EM Performance:\n', '      Train RMSE (norm): 0.0319\n', '      Val RMSE (norm):   0.0543\n', '      Test RMSE (norm):  0.0563\n', '      Train RMSE (orig): 6.73e+07\n', '      Val RMSE (orig):   1.15e+08\n', '      Test RMSE (orig):  1.19e+08\n', '    Combined Performance:\n', '      Train RMSE (norm): 0.0376\n', '      Val RMSE (norm):   0.0624\n', '      Test RMSE (norm):  0.0696\n', '      Train RMSE (orig): 5.51e+07\n', '      Val RMSE (orig):   9.30e+07\n', '      Test RMSE (orig):  9.92e+07\n', '      Overfitting detected: Val-Train gap = 0.0248\n', '========================================================================================================================\n', 'NOTE: Quote calculated in ORIGINAL scale then normalized for fair comparison\n', '========================================================================================================================\n', '                                               Model  SCR_Val_RMSE  EM_Val_RMSE  Quote_Val_RMSE_Norm  Quote_Test_RMSE_Norm  Quote_Test_R2_Norm  Quote_Test_RMSE_Orig  Quote_Test_R2_Orig\n', 'PyTorch Neural Net Tuned (Multi-Output) (with Quote)        0.0684       0.0501               0.0723                0.0645              0.9646                0.1908              0.9646\n', '           XGBoost Tuned (Multi-Output) (with Quote)        0.0764       0.0555               0.0807                0.0802              0.9453                0.2372              0.9453\n', '           XGBoost Basic (Multi-Output) (with Quote)        0.0775       0.0571               0.0851                0.0832              0.9411                0.2461              0.9411\n', '      MLPRegressor Tuned (Multi-Output) (with Quote)        0.0696       0.0543               0.0852                0.0822              0.9424                0.2434              0.9424\n', '      PyTorch Neural Net (Multi-Output) (with Quote)        0.0768       0.0558               0.0901                0.0824              0.9422                0.2437              0.9422\n', '           Random Forest (Multi-Output) (with Quote)        0.0950       0.0783               0.1015                0.1033              0.9091                0.3057              0.9091\n', '            MLPRegressor (Multi-Output) (with Quote)        0.0784       0.0651               0.1273                0.1256              0.8657                0.3717              0.8657\n', '               Quadratic (Multi-Output) (with Quote)        0.0997       0.0763               0.1541                0.1632              0.7734                0.4828              0.7734\n', '        Ridge Regression (Multi-Output) (with Quote)        0.1687       0.1556               1.2388                4.4102           -164.5481               13.0494           -164.5482\n', '       Linear Regression (Multi-Output) (with Quote)        0.1686       0.1556               1.2853                6.6605           -376.5974               19.7080           -376.5974\n', '\n', ' BEST FIXED JOINT MODEL:\n', 'Model: PyTorch Neural Net Tuned (Multi-Output) (with Quote)\n', 'Quote Val RMSE (normalized): 0.0723\n', 'Quote Test RMSE (normalized): 0.0645\n', 'Quote Test R² (normalized): 0.9646\n', '\n', ' FIXED APPROACH COMPARISON:\n', 'Direct Quote Prediction RMSE: 0.0955\n', 'Fixed Joint EM-SCR → Quote RMSE: 0.0723\n', ' Fixed joint approach is BETTER by 24.3%\n', '\n', ' ANALYSIS:\n', 'SCR-EM correlation: -0.576\n', '\n', ' WHY JOINT MODELING HELPS:\n', '1. Captures shared dependencies (ZSK1-ZSK3)\n', '2. Preserves EM/SCR mathematical relationship\n', '3. Recommendation: Use joint approach\n']
# --- End Output ---

# In[82]:  (cell 33)
def compare_direct_vs_joint_quote_prediction():
    """Proper comparison of direct vs joint Quote prediction"""
    
    # Get best direct Quote model
    best_direct_quote = None
    best_direct_rmse = float('inf')
    
    for model_key, results in all_results['quote'].items():
        if results is not None:
            val_rmse = results['metrics'].get('val_norm_RMSE', float('inf'))
            if val_rmse < best_direct_rmse:
                best_direct_rmse = val_rmse
                best_direct_quote = results
    
    # Get best joint model
    best_joint_quote = None
    best_joint_rmse = float('inf')
    
    for model_key, results in enhanced_results_fixed.items():
        if results is not None:
            quote_rmse = results['metrics'].get('val_Quote_RMSE', float('inf'))
            if quote_rmse < best_joint_rmse:
                best_joint_rmse = quote_rmse
                best_joint_quote = results
    
    print(f"\nCOMPREHENSIVE DIRECT vs JOINT QUOTE PREDICTION COMPARISON:")
    print(f"="*80)
    print(f"Direct Quote Prediction:")
    print(f"  Model: {best_direct_quote['model_name']}")
    print(f"  Val RMSE (norm): {best_direct_rmse:.4f}")
    print(f"  Test RMSE (norm): {best_direct_quote['metrics']['test_norm_RMSE']:.4f}")
    print(f"  Test R² (norm): {best_direct_quote['metrics']['test_norm_R2']:.4f}")
    
    print(f"\nJoint EM-SCR → Quote Prediction:")
    print(f"  Model: {best_joint_quote['model_name']}")
    print(f"  Val RMSE (norm): {best_joint_rmse:.4f}")
    print(f"  Test RMSE (norm): {best_joint_quote['metrics']['test_Quote_RMSE']:.4f}")
    print(f"  Test R² (norm): {best_joint_quote['metrics']['test_Quote_R2']:.4f}")
    
    if best_joint_rmse < best_direct_rmse:
        improvement = ((best_direct_rmse - best_joint_rmse) / best_direct_rmse) * 100
        print(f"\n CONCLUSION: Joint approach is BETTER by {improvement:.1f}%")
        print(f"    Validation RMSE: {best_direct_rmse:.4f} → {best_joint_rmse:.4f}")
        print(f"    Recommendation: Use joint EM-SCR modeling approach")

        
    else:
        degradation = ((best_joint_rmse - best_direct_rmse) / best_direct_rmse) * 100
        print(f"\n CONCLUSION: Joint approach is worse by {degradation:.1f}%")
        print(f"    Validation RMSE: {best_direct_rmse:.4f} → {best_joint_rmse:.4f}")
        print(f"    Recommendation: Use direct Quote prediction approach")

    
    return best_direct_quote, best_joint_quote


best_direct_model, best_joint_model = compare_direct_vs_joint_quote_prediction()

# --- Output [33] ---
# [1] type: stream
# (stdout)
# ['\n', 'COMPREHENSIVE DIRECT vs JOINT QUOTE PREDICTION COMPARISON:\n', '================================================================================\n', 'Direct Quote Prediction:\n', '  Model: Neural Net (MLPRegressor)\n', '  Val RMSE (norm): 0.0955\n', '  Test RMSE (norm): 0.0838\n', '  Test R² (norm): 0.9402\n', '\n', 'Joint EM-SCR → Quote Prediction:\n', '  Model: PyTorch Neural Net Tuned (Multi-Output) (with Quote)\n', '  Val RMSE (norm): 0.0723\n', '  Test RMSE (norm): 0.0645\n', '  Test R² (norm): 0.9646\n', '\n', ' CONCLUSION: Joint approach is BETTER by 24.3%\n', '    Validation RMSE: 0.0955 → 0.0723\n', '    Recommendation: Use joint EM-SCR modeling approach\n']
# --- End Output ---

# In[83]:  (cell 34)
def create_comprehensive_comparison_table():
    """Create comprehensive comparison table including ALL models from the notebook"""
    
    print(f"\n" + "="*120)
    print("COMPREHENSIVE MODEL COMPARISON - ALL MODELS IN NOTEBOOK")
    print("="*120)
    print("Includes: Individual target models + Multi-output models + XGBoost joint models")
    print("="*120)
    
    all_comparison_data = []


    for target_name, results_dict in [('QUOTE', all_results['quote']), 
                                     ('SCR', all_results['scr']), 
                                     ('EM', all_results['em'])]:
        
        for model_key, results in results_dict.items():
            if results is None:
                continue
                
            metrics = results['metrics']
            
            row = {
                'Target': target_name,
                'Model_Name': results['model_name'],
                'Model_Type': 'Individual Direct',
                'Model_Key': model_key,
                
                # Cross-validation metrics
                'CV_RMSE_Mean': metrics.get('cv_rmse_mean', np.nan),
                'CV_RMSE_Std': metrics.get('cv_rmse_std', np.nan),
                
                # Normalized metrics (for fair comparison)
                'Val_RMSE_Norm': metrics.get('val_norm_RMSE', np.nan),
                'Val_MAE_Norm': metrics.get('val_norm_MAE', np.nan), 
                'Val_R2_Norm': metrics.get('val_norm_R2', np.nan),
                'Test_RMSE_Norm': metrics.get('test_norm_RMSE', np.nan),
                'Test_MAE_Norm': metrics.get('test_norm_MAE', np.nan),
                'Test_R2_Norm': metrics.get('test_norm_R2', np.nan),
                
                # Original scale metrics (for business interpretation)
                'Val_RMSE_Orig': metrics.get('val_orig_RMSE', np.nan),
                'Val_MAE_Orig': metrics.get('val_orig_MAE', np.nan),
                'Test_RMSE_Orig': metrics.get('test_orig_RMSE', np.nan),
                'Test_MAE_Orig': metrics.get('test_orig_MAE', np.nan),
                'Test_R2_Orig': metrics.get('test_orig_R2', np.nan),
                
                # Regulatory metrics (Quote only)
                'Quote_Calc_Method': 'N/A' if target_name != 'QUOTE' else 'Direct'
            }
            all_comparison_data.append(row)


    for model_key, results in all_results['multi_output'].items():
        if results is None:
            continue
            
        metrics = results['metrics']
        
        # Add SCR component
        row_scr = {
            'Target': 'SCR',
            'Model_Name': results['model_name'].replace('(Multi-Output)', '(Multi SCR)'),
            'Model_Type': 'Multi-Output Component',
            'Model_Key': f"{model_key}_scr",
            
            'CV_RMSE_Mean': metrics.get('cv_rmse_mean', np.nan),
            'CV_RMSE_Std': metrics.get('cv_rmse_std', np.nan),
            
            'Val_RMSE_Norm': metrics.get('val_SCR_RMSE', np.nan),
            'Val_MAE_Norm': metrics.get('val_SCR_MAE', np.nan),
            'Val_R2_Norm': metrics.get('val_SCR_R2', np.nan),
            'Test_RMSE_Norm': metrics.get('test_SCR_RMSE', np.nan),
            'Test_MAE_Norm': metrics.get('test_SCR_MAE', np.nan),
            'Test_R2_Norm': metrics.get('test_SCR_R2', np.nan),
            
            'Val_RMSE_Orig': np.nan, 'Val_MAE_Orig': np.nan,
            'Test_RMSE_Orig': np.nan, 'Test_MAE_Orig': np.nan, 'Test_R2_Orig': np.nan,
            'Quote_Calc_Method': 'N/A'
        }
        all_comparison_data.append(row_scr)
        
        # Add EM component  
        row_em = {
            'Target': 'EM',
            'Model_Name': results['model_name'].replace('(Multi-Output)', '(Multi EM)'),
            'Model_Type': 'Multi-Output Component',
            'Model_Key': f"{model_key}_em",
            
            'CV_RMSE_Mean': metrics.get('cv_rmse_mean', np.nan),
            'CV_RMSE_Std': metrics.get('cv_rmse_std', np.nan),
            
            'Val_RMSE_Norm': metrics.get('val_EM_RMSE', np.nan),
            'Val_MAE_Norm': metrics.get('val_EM_MAE', np.nan),
            'Val_R2_Norm': metrics.get('val_EM_R2', np.nan),
            'Test_RMSE_Norm': metrics.get('test_EM_RMSE', np.nan),
            'Test_MAE_Norm': metrics.get('test_EM_MAE', np.nan),
            'Test_R2_Norm': metrics.get('test_EM_R2', np.nan),
            
            'Val_RMSE_Orig': np.nan, 'Val_MAE_Orig': np.nan,
            'Test_RMSE_Orig': np.nan, 'Test_MAE_Orig': np.nan, 'Test_R2_Orig': np.nan,
            'Quote_Calc_Method': 'N/A'
        }
        all_comparison_data.append(row_em)


    for model_key, results in enhanced_results_fixed.items():
        if results is None:
            continue
            
        metrics = results['metrics']
        model_type = 'XGBoost Joint' if 'xgboost' in model_key.lower() else 'Joint Multi-Output'
        
        # Add as Quote prediction (calculated from SCR+EM)
        row_quote = {
            'Target': 'QUOTE',
            'Model_Name': results['model_name'],
            'Model_Type': model_type,
            'Model_Key': model_key,
            
            'CV_RMSE_Mean': metrics.get('cv_rmse_mean', np.nan),
            'CV_RMSE_Std': metrics.get('cv_rmse_std', np.nan),
            
            # Quote metrics (calculated from SCR+EM)
            'Val_RMSE_Norm': metrics.get('val_Quote_RMSE', np.nan),
            'Val_MAE_Norm': metrics.get('val_Quote_MAE', np.nan),
            'Val_R2_Norm': metrics.get('val_Quote_R2', np.nan),
            'Test_RMSE_Norm': metrics.get('test_Quote_RMSE', np.nan),
            'Test_MAE_Norm': metrics.get('test_Quote_MAE', np.nan),
            'Test_R2_Norm': metrics.get('test_Quote_R2', np.nan),
            
            'Val_RMSE_Orig': np.nan, 'Val_MAE_Orig': np.nan,
            'Test_RMSE_Orig': np.nan, 'Test_MAE_Orig': np.nan, 'Test_R2_Orig': np.nan,
            'Quote_Calc_Method': 'SCR+EM→Quote'
        }
        all_comparison_data.append(row_quote)
        
        # Also add SCR and EM components for completeness
        row_scr_enhanced = {
            'Target': 'SCR',
            'Model_Name': results['model_name'].replace('(with Quote)', '(SCR Component)'),
            'Model_Type': model_type + ' Component',
            'Model_Key': f"{model_key}_scr",
            
            'CV_RMSE_Mean': metrics.get('cv_rmse_mean', np.nan),
            'CV_RMSE_Std': metrics.get('cv_rmse_std', np.nan),
            
            'Val_RMSE_Norm': metrics.get('val_SCR_RMSE', np.nan),
            'Test_RMSE_Norm': metrics.get('test_SCR_RMSE', np.nan),
            'Val_R2_Norm': metrics.get('val_SCR_R2', np.nan),
            'Test_R2_Norm': metrics.get('test_SCR_R2', np.nan),
            
            'Val_MAE_Norm': np.nan, 'Test_MAE_Norm': np.nan,
            'Val_RMSE_Orig': np.nan, 'Val_MAE_Orig': np.nan,
            'Test_RMSE_Orig': np.nan, 'Test_MAE_Orig': np.nan, 'Test_R2_Orig': np.nan,
            'Quote_Calc_Method': 'N/A'
        }
        all_comparison_data.append(row_scr_enhanced)
        
        row_em_enhanced = {
            'Target': 'EM',
            'Model_Name': results['model_name'].replace('(with Quote)', '(EM Component)'),
            'Model_Type': model_type + ' Component',
            'Model_Key': f"{model_key}_em",
            
            'CV_RMSE_Mean': metrics.get('cv_rmse_mean', np.nan),
            'CV_RMSE_Std': metrics.get('cv_rmse_std', np.nan),
            
            'Val_RMSE_Norm': metrics.get('val_EM_RMSE', np.nan),
            'Test_RMSE_Norm': metrics.get('test_EM_RMSE', np.nan),
            'Val_R2_Norm': metrics.get('val_EM_R2', np.nan),
            'Test_R2_Norm': metrics.get('test_EM_R2', np.nan),
            
            'Val_MAE_Norm': np.nan, 'Test_MAE_Norm': np.nan,
            'Val_RMSE_Orig': np.nan, 'Val_MAE_Orig': np.nan,
            'Test_RMSE_Orig': np.nan, 'Test_MAE_Orig': np.nan, 'Test_R2_Orig': np.nan,
            'Quote_Calc_Method': 'N/A'
        }
        all_comparison_data.append(row_em_enhanced)


    df_comprehensive = pd.DataFrame(all_comparison_data)

    
    # 1. Overall Summary by Model Type
    print(f"\nMODEL COUNT BY TYPE:")
    type_counts = df_comprehensive['Model_Type'].value_counts()
    for model_type, count in type_counts.items():
        print(f"  {model_type}: {count} models")
    
    total_models = len(df_comprehensive)
    print(f"  TOTAL: {total_models} model evaluations")
    
    # 2. Best Models by Target
    print(f"\n" + "="*80)
    print("BEST MODELS BY TARGET (Lowest Validation RMSE)")
    print("="*80)
    
    best_models_summary = {}
    for target in ['QUOTE', 'SCR', 'EM']:
        target_data = df_comprehensive[df_comprehensive['Target'] == target].copy()
        target_data = target_data.dropna(subset=['Val_RMSE_Norm'])
        
        if not target_data.empty:
            best_idx = target_data['Val_RMSE_Norm'].idxmin()
            best_row = target_data.loc[best_idx]
            
            best_models_summary[target] = best_row
            
            print(f"\n{target}:")
            print(f"   {best_row['Model_Name']} ({best_row['Model_Type']})")
            print(f"  Val RMSE (Norm): {best_row['Val_RMSE_Norm']:.4f}")
            print(f"  Test RMSE (Norm): {best_row['Test_RMSE_Norm']:.4f}")
            print(f"  Test R² (Norm): {best_row['Test_R2_Norm']:.4f}")
            if best_row['Quote_Calc_Method'] != 'N/A':
                print(f"  Quote Method: {best_row['Quote_Calc_Method']}")
    
    # 3. Detailed Tables by Target
    display_cols = ['Model_Name', 'Model_Type', 'Val_RMSE_Norm', 'Test_RMSE_Norm', 'Test_R2_Norm', 'Quote_Calc_Method']
    
    for target in ['QUOTE', 'SCR', 'EM']:
        target_data = df_comprehensive[df_comprehensive['Target'] == target].copy()
        target_data = target_data.dropna(subset=['Val_RMSE_Norm'])
        target_data = target_data.sort_values('Val_RMSE_Norm')
        
        print(f"\n" + "="*100)
        print(f"{target} MODELS - DETAILED COMPARISON")
        print("="*100)
        
        if not target_data.empty:
            display_data = target_data[display_cols].copy()
            
            # Add rank
            display_data.insert(0, 'Rank', range(1, len(display_data) + 1))
            
            print("Sorted by Validation RMSE (Normalized Scale):")
            print(display_data.round(4).to_string(index=False, max_colwidth=35))
        else:
            print("No valid models found for this target")
    
    # 4. XGBoost Specific Analysis
    xgboost_models = df_comprehensive[df_comprehensive['Model_Type'].str.contains('XGBoost', na=False)]
    if not xgboost_models.empty:
        print(f"\n" + "="*80)
        print("XGBOOST MODELS PERFORMANCE SUMMARY")
        print("="*80)
        
        xgboost_summary = xgboost_models[['Target', 'Model_Name', 'Val_RMSE_Norm', 'Test_RMSE_Norm', 'Test_R2_Norm']].copy()
        xgboost_summary = xgboost_summary.sort_values(['Target', 'Val_RMSE_Norm'])
        
        print("XGBoost models across all targets:")
        print(xgboost_summary.round(4).to_string(index=False))
        
        # Compare XGBoost with best non-XGBoost for Quote
        quote_xgb = xgboost_models[xgboost_models['Target'] == 'QUOTE']
        if not quote_xgb.empty:
            best_xgb_quote = quote_xgb.loc[quote_xgb['Val_RMSE_Norm'].idxmin()]
            best_direct_quote = best_models_summary.get('QUOTE')
            
            if best_direct_quote is not None and 'XGBoost' not in best_direct_quote['Model_Type']:
                improvement = ((best_direct_quote['Val_RMSE_Norm'] - best_xgb_quote['Val_RMSE_Norm']) / 
                              best_direct_quote['Val_RMSE_Norm']) * 100
                
                print(f"\nXGBOOST vs BEST DIRECT QUOTE PREDICTION:")
                print(f"  Best Direct: {best_direct_quote['Val_RMSE_Norm']:.4f} RMSE")
                print(f"  Best XGBoost: {best_xgb_quote['Val_RMSE_Norm']:.4f} RMSE")
                if improvement > 0:
                    print(f"  🎯 XGBoost is {improvement:.1f}% BETTER")
                else:
                    print(f"  Direct approach is {abs(improvement):.1f}% better")
    
    # 5. Model Approach Comparison
    print(f"\n" + "="*80)
    print("MODELING APPROACH COMPARISON")
    print("="*80)
    
    approach_summary = []
    for target in ['QUOTE', 'SCR', 'EM']:
        target_models = df_comprehensive[df_comprehensive['Target'] == target]
        target_models = target_models.dropna(subset=['Val_RMSE_Norm'])
        
        if target_models.empty:
            continue
            
        for model_type in target_models['Model_Type'].unique():
            type_models = target_models[target_models['Model_Type'] == model_type]
            best_model = type_models.loc[type_models['Val_RMSE_Norm'].idxmin()]
            
            approach_summary.append({
                'Target': target,
                'Approach': model_type,
                'Best_Model': best_model['Model_Name'],
                'Best_RMSE': best_model['Val_RMSE_Norm'],
                'Best_R2': best_model['Test_R2_Norm']
            })
    
    approach_df = pd.DataFrame(approach_summary)
    print("Best model by approach and target:")
    print(approach_df.round(4).to_string(index=False))
    
    return df_comprehensive, best_models_summary


comprehensive_df, best_models_from_comprehensive = create_comprehensive_comparison_table()

# Extract individual comparison tables 
quote_comparison = comprehensive_df[comprehensive_df['Target'] == 'QUOTE'].copy()
scr_comparison = comprehensive_df[comprehensive_df['Target'] == 'SCR'].copy()
em_comparison = comprehensive_df[comprehensive_df['Target'] == 'EM'].copy()
multi_comparison = comprehensive_df[comprehensive_df['Target'].isin(['SCR', 'EM']) & 
                                  comprehensive_df['Model_Type'].str.contains('Multi-Output', na=False)].copy()


# Select best models based on validation performance 
best_models = {}

print("SELECTING BEST MODELS FROM ALL SOURCES (individual + multi-output + enhanced joint):")
print("="*80)

print("\n1. QUOTE MODEL SELECTION:")
quote_candidates = []

# Add individual Quote models
if 'quote' in all_results:
    for model_key, results in all_results['quote'].items():
        if results is not None:
            val_rmse = results['metrics'].get('val_norm_RMSE', float('inf'))
            quote_candidates.append({
                'source': 'individual',
                'key': model_key,
                'results': results,
                'val_rmse': val_rmse,
                'name': results['model_name']
            })

# Add enhanced joint models (Quote calculated from SCR+EM)
if 'enhanced_results_fixed' in globals():
    for model_key, results in enhanced_results_fixed.items():
        if results is not None:
            val_rmse = results['metrics'].get('val_Quote_RMSE', float('inf'))
            quote_candidates.append({
                'source': 'joint',
                'key': model_key,
                'results': results,
                'val_rmse': val_rmse,
                'name': results['model_name']
            })

# Select best Quote model
if quote_candidates:
    best_quote = min(quote_candidates, key=lambda x: x['val_rmse'])
    best_models['quote'] = {
        'key': best_quote['key'],
        'results': best_quote['results'],
        'val_rmse_norm': best_quote['val_rmse'],
        'source': best_quote['source']
    }
    print(f"  Best: {best_quote['name']} (RMSE: {best_quote['val_rmse']:.4f}) [Source: {best_quote['source']}]")
else:
    print("  No Quote candidates found!")


print("\n2. SCR MODEL SELECTION:")
scr_candidates = []

# Add individual SCR models
if 'scr' in all_results:
    for model_key, results in all_results['scr'].items():
        if results is not None:
            val_rmse = results['metrics'].get('val_norm_RMSE', float('inf'))
            scr_candidates.append({
                'source': 'individual',
                'key': model_key,
                'results': results,
                'val_rmse': val_rmse,
                'name': results['model_name']
            })

# Add multi-output SCR components  
if 'multi_output' in all_results:
    for model_key, results in all_results['multi_output'].items():
        if results is not None:
            val_rmse = results['metrics'].get('val_SCR_RMSE', float('inf'))
            # Create pseudo-results for SCR component
            scr_component_results = {
                'model': results['model'],
                'model_name': results['model_name'] + ' (SCR Component)',
                'target_name': 'SCR',
                'metrics': {k: v for k, v in results['metrics'].items() if 'SCR' in k},
                'predictions': {'test': results['predictions']['test'][:, 0] if hasattr(results['predictions']['test'], 'shape') and len(results['predictions']['test'].shape) > 1 else results['predictions']['test']},
                'actuals': {'test': results['actuals']['test'].iloc[:, 0] if hasattr(results['actuals']['test'], 'iloc') else results['actuals']['test']}
            }
            scr_candidates.append({
                'source': 'multi_output',
                'key': f"{model_key}_scr",
                'results': scr_component_results,
                'val_rmse': val_rmse,
                'name': scr_component_results['model_name']
            })

# Add enhanced joint SCR components
if 'enhanced_results_fixed' in globals():
    for model_key, results in enhanced_results_fixed.items():
        if results is not None:
            val_rmse = results['metrics'].get('val_SCR_RMSE', float('inf'))
            # Create pseudo-results for SCR component
            scr_component_results = {
                'model': results['model'],
                'model_name': results['model_name'] + ' (Enhanced SCR Component)',
                'target_name': 'SCR',
                'metrics': {k: v for k, v in results['metrics'].items() if 'SCR' in k},
                'predictions': {'test': results['predictions']['test'][:, 0] if hasattr(results['predictions']['test'], 'shape') and len(results['predictions']['test'].shape) > 1 else results['predictions']['test']},
                'actuals': {'test': results['actuals']['test'].iloc[:, 0] if hasattr(results['actuals']['test'], 'iloc') else results['actuals']['test']}
            }
            scr_candidates.append({
                'source': 'enhanced_joint',
                'key': f"{model_key}_scr",
                'results': scr_component_results,
                'val_rmse': val_rmse,
                'name': scr_component_results['model_name']
            })

# Select best SCR model
if scr_candidates:
    best_scr = min(scr_candidates, key=lambda x: x['val_rmse'])
    best_models['scr'] = {
        'key': best_scr['key'],
        'results': best_scr['results'],
        'val_rmse_norm': best_scr['val_rmse'],
        'source': best_scr['source']
    }
    print(f"  Best: {best_scr['name']} (RMSE: {best_scr['val_rmse']:.4f}) [Source: {best_scr['source']}]")
    
    # Show improvement if joint model wins
    individual_best = min([c for c in scr_candidates if c['source'] == 'individual'], key=lambda x: x['val_rmse'], default=None)
    if individual_best and best_scr['source'] != 'individual':
        improvement = ((individual_best['val_rmse'] - best_scr['val_rmse']) / individual_best['val_rmse']) * 100
        print(f"    Improvement over best individual: {improvement:.1f}%")
else:
    print("  No SCR candidates found!")


print("\n3. EM MODEL SELECTION:")
em_candidates = []

# Add individual EM models
if 'em' in all_results:
    for model_key, results in all_results['em'].items():
        if results is not None:
            val_rmse = results['metrics'].get('val_norm_RMSE', float('inf'))
            em_candidates.append({
                'source': 'individual',
                'key': model_key,
                'results': results,
                'val_rmse': val_rmse,
                'name': results['model_name']
            })

# Add multi-output EM components
if 'multi_output' in all_results:
    for model_key, results in all_results['multi_output'].items():
        if results is not None:
            val_rmse = results['metrics'].get('val_EM_RMSE', float('inf'))
            # Create pseudo-results for EM component
            em_component_results = {
                'model': results['model'],
                'model_name': results['model_name'] + ' (EM Component)', 
                'target_name': 'EM',
                'metrics': {k: v for k, v in results['metrics'].items() if 'EM' in k},
                'predictions': {'test': results['predictions']['test'][:, 1] if hasattr(results['predictions']['test'], 'shape') and len(results['predictions']['test'].shape) > 1 else results['predictions']['test']},
                'actuals': {'test': results['actuals']['test'].iloc[:, 1] if hasattr(results['actuals']['test'], 'iloc') else results['actuals']['test']}
            }
            em_candidates.append({
                'source': 'multi_output',
                'key': f"{model_key}_em",
                'results': em_component_results,
                'val_rmse': val_rmse,
                'name': em_component_results['model_name']
            })

# Add enhanced joint EM components
if 'enhanced_results_fixed' in globals():
    for model_key, results in enhanced_results_fixed.items():
        if results is not None:
            val_rmse = results['metrics'].get('val_EM_RMSE', float('inf'))
            # Create pseudo-results for EM component
            em_component_results = {
                'model': results['model'],
                'model_name': results['model_name'] + ' (Enhanced EM Component)',
                'target_name': 'EM', 
                'metrics': {k: v for k, v in results['metrics'].items() if 'EM' in k},
                'predictions': {'test': results['predictions']['test'][:, 1] if hasattr(results['predictions']['test'], 'shape') and len(results['predictions']['test'].shape) > 1 else results['predictions']['test']},
                'actuals': {'test': results['actuals']['test'].iloc[:, 1] if hasattr(results['actuals']['test'], 'iloc') else results['actuals']['test']}
            }
            em_candidates.append({
                'source': 'enhanced_joint',
                'key': f"{model_key}_em", 
                'results': em_component_results,
                'val_rmse': val_rmse,
                'name': em_component_results['model_name']
            })

# Select best EM model
if em_candidates:
    best_em = min(em_candidates, key=lambda x: x['val_rmse'])
    best_models['em'] = {
        'key': best_em['key'],
        'results': best_em['results'], 
        'val_rmse_norm': best_em['val_rmse'],
        'source': best_em['source']
    }
    print(f"  Best: {best_em['name']} (RMSE: {best_em['val_rmse']:.4f}) [Source: {best_em['source']}]")
    
    # Show improvement if joint model wins
    individual_best = min([c for c in em_candidates if c['source'] == 'individual'], key=lambda x: x['val_rmse'], default=None)
    if individual_best and best_em['source'] != 'individual':
        improvement = ((individual_best['val_rmse'] - best_em['val_rmse']) / individual_best['val_rmse']) * 100
        print(f"    Improvement over best individual: {improvement:.1f}%")
else:
    print("  No EM candidates found!")


print("\n4. MULTI-OUTPUT MODEL SELECTION:")
multi_candidates = []

# Add traditional multi-output models
if 'multi_output' in all_results:
    for model_key, results in all_results['multi_output'].items():
        if results is not None:
            val_rmse = results['metrics'].get('val_RMSE_macro', float('inf'))
            multi_candidates.append({
                'source': 'multi_output',
                'key': model_key,
                'results': results,
                'val_rmse': val_rmse,
                'name': results['model_name']
            })

# Add enhanced joint models
if 'enhanced_results_fixed' in globals():
    for model_key, results in enhanced_results_fixed.items():
        if results is not None:
            val_rmse = results['metrics'].get('val_RMSE_macro', float('inf'))
            multi_candidates.append({
                'source': 'enhanced_joint',
                'key': model_key,
                'results': results,
                'val_rmse': val_rmse,
                'name': results['model_name']
            })

# Select best multi-output model
if multi_candidates:
    best_multi = min(multi_candidates, key=lambda x: x['val_rmse'])
    best_models['multi_output'] = {
        'key': best_multi['key'],
        'results': best_multi['results'],
        'val_rmse_norm': best_multi['val_rmse'],
        'source': best_multi['source']
    }
    print(f"  Best: {best_multi['name']} (RMSE: {best_multi['val_rmse']:.4f}) [Source: {best_multi['source']}]")
else:
    print("  No multi-output candidates found!")

print("\nBEST MODELS SELECTED (considering ALL sources):")
print("="*60)
for target, best_info in best_models.items():
    source_note = f" [{best_info['source']}]" if 'source' in best_info else ""
    print(f"{target.upper()}: {best_info['results']['model_name']}{source_note}")
    print(f"  Normalized Val RMSE: {best_info['val_rmse_norm']:.4f}")
    print()

# Verify RMSE consistency across targets
print("RMSE CONSISTENCY CHECK:")
print("="*40)
for target, best_info in best_models.items():
    if target != 'multi_output':
        norm_rmse = best_info['val_rmse_norm']
        print(f"{target.upper()}: {norm_rmse:.4f}")
print("\nAll models now selected from comprehensive comparison across all approaches!")

# --- Output [34] ---
# [1] type: stream
# (stdout)
# ['\n', '========================================================================================================================\n', 'COMPREHENSIVE MODEL COMPARISON - ALL MODELS IN NOTEBOOK\n', '========================================================================================================================\n', 'Includes: Individual target models + Multi-output models + XGBoost joint models\n', '========================================================================================================================\n', '\n', 'MODEL COUNT BY TYPE:\n', '  Individual Direct: 24 models\n', '  Multi-Output Component: 18 models\n', '  Joint Multi-Output Component: 16 models\n', '  Joint Multi-Output: 8 models\n', '  XGBoost Joint Component: 4 models\n', '  XGBoost Joint: 2 models\n', '  TOTAL: 72 model evaluations\n', '\n', '================================================================================\n', 'BEST MODELS BY TARGET (Lowest Validation RMSE)\n', '================================================================================\n', '\n', 'QUOTE:\n', '   PyTorch Neural Net Tuned (Multi-Output) (with Quote) (Joint Multi-Output)\n', '  Val RMSE (Norm): 0.0723\n', '  Test RMSE (Norm): 0.0645\n', '  Test R² (Norm): 0.9646\n', '  Quote Method: SCR+EM→Quote\n', '\n', 'SCR:\n', '   PyTorch Neural Net Tuned (Multi-Output) (SCR Component) (Joint Multi-Output Component)\n', '  Val RMSE (Norm): 0.0684\n', '  Test RMSE (Norm): 0.0791\n', '  Test R² (Norm): 0.9416\n', '\n', 'EM:\n', '   PyTorch Neural Net Tuned (Multi-Output) (EM Component) (Joint Multi-Output Component)\n', '  Val RMSE (Norm): 0.0501\n', '  Test RMSE (Norm): 0.0514\n', '  Test R² (Norm): 0.9373\n', '\n', '====================================================================================================\n', 'QUOTE MODELS - DETAILED COMPARISON\n', '====================================================================================================\n', 'Sorted by Validation RMSE (Normalized Scale):\n', ' Rank                          Model_Name         Model_Type  Val_RMSE_Norm  Test_RMSE_Norm  Test_R2_Norm Quote_Calc_Method\n', '    1 PyTorch Neural Net Tuned (Multi-... Joint Multi-Output         0.0723          0.0645        0.9646      SCR+EM→Quote\n', '    2 XGBoost Tuned (Multi-Output) (wi...      XGBoost Joint         0.0807          0.0802        0.9453      SCR+EM→Quote\n', '    3 XGBoost Basic (Multi-Output) (wi...      XGBoost Joint         0.0851          0.0832        0.9411      SCR+EM→Quote\n', '    4 MLPRegressor Tuned (Multi-Output... Joint Multi-Output         0.0852          0.0822        0.9424      SCR+EM→Quote\n', '    5 PyTorch Neural Net (Multi-Output... Joint Multi-Output         0.0901          0.0824        0.9422      SCR+EM→Quote\n', '    6           Neural Net (MLPRegressor)  Individual Direct         0.0955          0.0838        0.9402            Direct\n', '    7 Random Forest (Multi-Output) (wi... Joint Multi-Output         0.1015          0.1033        0.9091      SCR+EM→Quote\n', '    8                Quadratic (Degree 2)  Individual Direct         0.1108          0.1062        0.9040            Direct\n', '    9 MLPRegressor (Multi-Output) (wit... Joint Multi-Output         0.1273          0.1256        0.8657      SCR+EM→Quote\n', '   10 Quadratic (Multi-Output) (with Q... Joint Multi-Output         0.1541          0.1632        0.7734      SCR+EM→Quote\n', '   11                    Cubic (Degree 3)  Individual Direct         0.1822          0.1588        0.7853            Direct\n', '   12                       Ridge with CV  Individual Direct         0.2012          0.1907        0.6904            Direct\n', '   13                         Elastic Net  Individual Direct         0.2012          0.1907        0.6903            Direct\n', '   14             Linear Regression (MSE)  Individual Direct         0.2012          0.1907        0.6905            Direct\n', '   15                       Lasso with CV  Individual Direct         0.2013          0.1910        0.6894            Direct\n', '   16                     Dummy Regressor  Individual Direct         0.3420          0.3429       -0.0006            Direct\n', '   17 Ridge Regression (Multi-Output) ... Joint Multi-Output         1.2388          4.4102     -164.5481      SCR+EM→Quote\n', '   18 Linear Regression (Multi-Output)... Joint Multi-Output         1.2853          6.6605     -376.5974      SCR+EM→Quote\n', '\n', '====================================================================================================\n', 'SCR MODELS - DETAILED COMPARISON\n', '====================================================================================================\n', 'Sorted by Validation RMSE (Normalized Scale):\n', ' Rank                          Model_Name                   Model_Type  Val_RMSE_Norm  Test_RMSE_Norm  Test_R2_Norm Quote_Calc_Method\n', '    1 PyTorch Neural Net Tuned (Multi-... Joint Multi-Output Component         0.0684          0.0791        0.9416               N/A\n', '    2 MLPRegressor Tuned (Multi-Output... Joint Multi-Output Component         0.0696          0.0807        0.9392               N/A\n', '    3 XGBoost Tuned (Multi-Output) (SC...      XGBoost Joint Component         0.0764          0.0840        0.9341               N/A\n', '    4 PyTorch Neural Net (Multi-Output... Joint Multi-Output Component         0.0768          0.0876        0.9283               N/A\n', '    5 XGBoost Basic (Multi-Output) (SC...      XGBoost Joint Component         0.0775          0.0855        0.9317               N/A\n', '    6 MLPRegressor (Multi-Output) (SCR... Joint Multi-Output Component         0.0784          0.0814        0.9381               N/A\n', '    7 Neural Net (PyTorch MLP Dropout ...       Multi-Output Component         0.0819          0.0892        0.9256               N/A\n', '    8 Neural Net (MLPRegressor) (Multi...       Multi-Output Component         0.0857          0.0991        0.9083               N/A\n', '    9           Neural Net (MLPRegressor)            Individual Direct         0.0886          0.0932        0.9189               N/A\n', '   10 Random Forest (Multi-Output) (SC... Joint Multi-Output Component         0.0950          0.0998        0.9070               N/A\n', '   11 Quadratic (Multi-Output) (SCR Co... Joint Multi-Output Component         0.0997          0.1053        0.8964               N/A\n', '   12                Quadratic (Degree 2)            Individual Direct         0.0997          0.1053        0.8964               N/A\n', '   13    Quadratic (Degree 2) (Multi SCR)       Multi-Output Component         0.0997          0.1053        0.8964               N/A\n', '   14        Cubic (Degree 3) (Multi SCR)       Multi-Output Component         0.1542          0.1397        0.8176               N/A\n', '   15                    Cubic (Degree 3)            Individual Direct         0.1542          0.1397        0.8176               N/A\n', '   16 Linear Regression (Multi-Output)... Joint Multi-Output Component         0.1686          0.1674        0.7381               N/A\n', '   17             Linear Regression (MSE)            Individual Direct         0.1686          0.1674        0.7381               N/A\n', '   18 Linear Regression (MSE) (Multi SCR)       Multi-Output Component         0.1686          0.1674        0.7381               N/A\n', '   19                         Elastic Net            Individual Direct         0.1687          0.1675        0.7380               N/A\n', '   20           Ridge with CV (Multi SCR)       Multi-Output Component         0.1687          0.1675        0.7380               N/A\n', '   21                       Ridge with CV            Individual Direct         0.1687          0.1675        0.7380               N/A\n', '   22 Ridge Regression (Multi-Output) ... Joint Multi-Output Component         0.1687          0.1675        0.7380               N/A\n', '   23           Lasso with CV (Multi SCR)       Multi-Output Component         0.1687          0.1677        0.7372               N/A\n', '   24                       Lasso with CV            Individual Direct         0.1687          0.1677        0.7372               N/A\n', '   25             Elastic Net (Multi SCR)       Multi-Output Component         0.3229          0.3275       -0.0017               N/A\n', '   26         Dummy Regressor (Multi SCR)       Multi-Output Component         0.3229          0.3275       -0.0017               N/A\n', '   27                     Dummy Regressor            Individual Direct         0.3229          0.3275       -0.0017               N/A\n', '\n', '====================================================================================================\n', 'EM MODELS - DETAILED COMPARISON\n', '====================================================================================================\n', 'Sorted by Validation RMSE (Normalized Scale):\n', ' Rank                          Model_Name                   Model_Type  Val_RMSE_Norm  Test_RMSE_Norm  Test_R2_Norm Quote_Calc_Method\n', '    1 PyTorch Neural Net Tuned (Multi-... Joint Multi-Output Component         0.0501          0.0514        0.9373               N/A\n', '    2 MLPRegressor Tuned (Multi-Output... Joint Multi-Output Component         0.0543          0.0563        0.9250               N/A\n', '    3 XGBoost Tuned (Multi-Output) (EM...      XGBoost Joint Component         0.0555          0.0562        0.9251               N/A\n', '    4 PyTorch Neural Net (Multi-Output... Joint Multi-Output Component         0.0558          0.0563        0.9248               N/A\n', '    5 XGBoost Basic (Multi-Output) (EM...      XGBoost Joint Component         0.0571          0.0583        0.9195               N/A\n', '    6 Neural Net (PyTorch MLP Dropout ...       Multi-Output Component         0.0613          0.0608        0.9124               N/A\n', '    7 Neural Net (MLPRegressor) (Multi...       Multi-Output Component         0.0641          0.0712        0.8799               N/A\n', '    8 MLPRegressor (Multi-Output) (EM ... Joint Multi-Output Component         0.0651          0.0623        0.9079               N/A\n', '    9                Quadratic (Degree 2)            Individual Direct         0.0763          0.0746        0.8682               N/A\n', '   10     Quadratic (Degree 2) (Multi EM)       Multi-Output Component         0.0763          0.0746        0.8682               N/A\n', '   11 Quadratic (Multi-Output) (EM Com... Joint Multi-Output Component         0.0763          0.0746        0.8682               N/A\n', '   12 Random Forest (Multi-Output) (EM... Joint Multi-Output Component         0.0783          0.0799        0.8489               N/A\n', '   13           Neural Net (MLPRegressor)            Individual Direct         0.0822          0.0745        0.8685               N/A\n', '   14         Cubic (Degree 3) (Multi EM)       Multi-Output Component         0.1425          0.1173        0.6740               N/A\n', '   15                    Cubic (Degree 3)            Individual Direct         0.1425          0.1173        0.6740               N/A\n', '   16  Linear Regression (MSE) (Multi EM)       Multi-Output Component         0.1556          0.1525        0.4484               N/A\n', '   17             Linear Regression (MSE)            Individual Direct         0.1556          0.1525        0.4484               N/A\n', '   18 Linear Regression (Multi-Output)... Joint Multi-Output Component         0.1556          0.1525        0.4484               N/A\n', '   19            Ridge with CV (Multi EM)       Multi-Output Component         0.1556          0.1526        0.4479               N/A\n', '   20                       Ridge with CV            Individual Direct         0.1556          0.1526        0.4479               N/A\n', '   21 Ridge Regression (Multi-Output) ... Joint Multi-Output Component         0.1556          0.1526        0.4479               N/A\n', '   22                         Elastic Net            Individual Direct         0.1557          0.1528        0.4463               N/A\n', '   23            Lasso with CV (Multi EM)       Multi-Output Component         0.1558          0.1529        0.4456               N/A\n', '   24                       Lasso with CV            Individual Direct         0.1558          0.1529        0.4456               N/A\n', '   25          Dummy Regressor (Multi EM)       Multi-Output Component         0.2067          0.2054       -0.0001               N/A\n', '   26                     Dummy Regressor            Individual Direct         0.2067          0.2054       -0.0001               N/A\n', '   27              Elastic Net (Multi EM)       Multi-Output Component         0.2067          0.2054       -0.0001               N/A\n', '\n', '================================================================================\n', 'XGBOOST MODELS PERFORMANCE SUMMARY\n', '================================================================================\n', 'XGBoost models across all targets:\n', 'Target                                   Model_Name  Val_RMSE_Norm  Test_RMSE_Norm  Test_R2_Norm\n', '    EM  XGBoost Tuned (Multi-Output) (EM Component)         0.0555          0.0562        0.9251\n', '    EM  XGBoost Basic (Multi-Output) (EM Component)         0.0571          0.0583        0.9195\n', ' QUOTE    XGBoost Tuned (Multi-Output) (with Quote)         0.0807          0.0802        0.9453\n', ' QUOTE    XGBoost Basic (Multi-Output) (with Quote)         0.0851          0.0832        0.9411\n', '   SCR XGBoost Tuned (Multi-Output) (SCR Component)         0.0764          0.0840        0.9341\n', '   SCR XGBoost Basic (Multi-Output) (SCR Component)         0.0775          0.0855        0.9317\n', '\n', 'XGBOOST vs BEST DIRECT QUOTE PREDICTION:\n', '  Best Direct: 0.0723 RMSE\n', '  Best XGBoost: 0.0807 RMSE\n', '  Direct approach is 11.6% better\n', '\n', '================================================================================\n', 'MODELING APPROACH COMPARISON\n', '================================================================================\n', 'Best model by approach and target:\n', 'Target                     Approach                                              Best_Model  Best_RMSE  Best_R2\n', ' QUOTE            Individual Direct                               Neural Net (MLPRegressor)     0.0955   0.9402\n', ' QUOTE           Joint Multi-Output    PyTorch Neural Net Tuned (Multi-Output) (with Quote)     0.0723   0.9646\n', ' QUOTE                XGBoost Joint               XGBoost Tuned (Multi-Output) (with Quote)     0.0807   0.9453\n', '   SCR            Individual Direct                               Neural Net (MLPRegressor)     0.0886   0.9189\n', '   SCR       Multi-Output Component      Neural Net (PyTorch MLP Dropout Multi) (Multi SCR)     0.0819   0.9256\n', '   SCR Joint Multi-Output Component PyTorch Neural Net Tuned (Multi-Output) (SCR Component)     0.0684   0.9416\n', '   SCR      XGBoost Joint Component            XGBoost Tuned (Multi-Output) (SCR Component)     0.0764   0.9341\n', '    EM            Individual Direct                                    Quadratic (Degree 2)     0.0763   0.8682\n', '    EM       Multi-Output Component       Neural Net (PyTorch MLP Dropout Multi) (Multi EM)     0.0613   0.9124\n', '    EM Joint Multi-Output Component  PyTorch Neural Net Tuned (Multi-Output) (EM Component)     0.0501   0.9373\n', '    EM      XGBoost Joint Component             XGBoost Tuned (Multi-Output) (EM Component)     0.0555   0.9251\n', 'SELECTING BEST MODELS FROM ALL SOURCES (individual + multi-output + enhanced joint):\n', '================================================================================\n', '\n', '1. QUOTE MODEL SELECTION:\n', '  Best: PyTorch Neural Net Tuned (Multi-Output) (with Quote) (RMSE: 0.0723) [Source: joint]\n', '\n', '2. SCR MODEL SELECTION:\n', '  Best: PyTorch Neural Net Tuned (Multi-Output) (with Quote) (Enhanced SCR Component) (RMSE: 0.0684) [Source: enhanced_joint]\n', '    Improvement over best individual: 22.8%\n', '\n', '3. EM MODEL SELECTION:\n', '  Best: PyTorch Neural Net Tuned (Multi-Output) (with Quote) (Enhanced EM Component) (RMSE: 0.0501) [Source: enhanced_joint]\n', '    Improvement over best individual: 34.3%\n', '\n', '4. MULTI-OUTPUT MODEL SELECTION:\n', '  Best: PyTorch Neural Net Tuned (Multi-Output) (with Quote) (RMSE: 0.0600) [Source: enhanced_joint]\n', '\n', 'BEST MODELS SELECTED (considering ALL sources):\n', '============================================================\n', 'QUOTE: PyTorch Neural Net Tuned (Multi-Output) (with Quote) [joint]\n', '  Normalized Val RMSE: 0.0723\n', '\n', 'SCR: PyTorch Neural Net Tuned (Multi-Output) (with Quote) (Enhanced SCR Component) [enhanced_joint]\n', '  Normalized Val RMSE: 0.0684\n', '\n', 'EM: PyTorch Neural Net Tuned (Multi-Output) (with Quote) (Enhanced EM Component) [enhanced_joint]\n', '  Normalized Val RMSE: 0.0501\n', '\n', 'MULTI_OUTPUT: PyTorch Neural Net Tuned (Multi-Output) (with Quote) [enhanced_joint]\n', '  Normalized Val RMSE: 0.0600\n', '\n', 'RMSE CONSISTENCY CHECK:\n', '========================================\n', 'QUOTE: 0.0723\n', 'SCR: 0.0684\n', 'EM: 0.0501\n', '\n', 'All models now selected from comprehensive comparison across all approaches!\n']
# --- End Output ---

# In[35]:
#NEW
def get_torch_mlp_feature_importance(model, X_test, y_test, feature_names, target_name, n_repeats=10):
    """
    Calculate feature importance for TorchMLPReg model using permutation importance
    Compatible with get_polynomial_feature_importance output format
    
    Args:
        model: Trained sklearn NeuralNetRegressor with TorchMLPReg module
        X_test: Test features (pandas DataFrame or numpy array)
        y_test: Test targets - can be 1D (single output) or 2D (2 outputs for SCR and EM)
        feature_names: List of feature names
        target_name: Name of the target ('scr', 'em', 'quote', etc.)
        n_repeats: Number of permutation repeats for stability
        
    Returns:
        Dictionary with importance results compatible with get_polynomial_feature_importance
    """
    
    print(f"\nAnalyzing feature importance for {target_name}...")
    
    try:
        # Convert to numpy arrays if needed
        if hasattr(X_test, 'values'):
            X_test_np = X_test.values.astype(np.float32)
        else:
            X_test_np = np.array(X_test, dtype=np.float32)
            
        if hasattr(y_test, 'values'):
            y_test_np = y_test.values.astype(np.float32)
        else:
            y_test_np = np.array(y_test, dtype=np.float32)
        
        # Determine if single or multi-output
        is_single_output = y_test_np.ndim == 1
        
        # Get baseline predictions
        baseline_pred = model.predict(X_test_np)
        
        if is_single_output:
            # Handle single output model
            # Convert 2D predictions to 1D if needed
            if baseline_pred.ndim == 2:
                if baseline_pred.shape[1] == 2:
                    # Multi-output model used for single target - determine which output
                    if target_name.lower() == 'scr':
                        baseline_pred = baseline_pred[:, 0]
                    elif target_name.lower() == 'em':
                        baseline_pred = baseline_pred[:, 1]
                    else:
                        baseline_pred = baseline_pred[:, 0]  # Default to first output
                elif baseline_pred.shape[1] == 1:
                    baseline_pred = baseline_pred[:, 0]
            
            # Calculate baseline MSE
            baseline_mse = mean_squared_error(y_test_np, baseline_pred)
            print(f"Baseline MSE for {target_name}: {baseline_mse:.6f}")
            
            # Calculate permutation importance
            importances = []
            for feature_idx in range(len(feature_names)):
                scores = []
                
                for repeat in range(n_repeats):
                    # Create permuted copy
                    X_permuted = X_test_np.copy()
                    perm_indices = np.random.permutation(X_test_np.shape[0])
                    X_permuted[:, feature_idx] = X_permuted[perm_indices, feature_idx]
                    
                    # Get predictions on permuted data
                    permuted_pred = model.predict(X_permuted)
                    
                    # Handle multi-output case (same logic as baseline)
                    if permuted_pred.ndim == 2:
                        if permuted_pred.shape[1] == 2:
                            if target_name.lower() == 'scr':
                                permuted_pred = permuted_pred[:, 0]
                            elif target_name.lower() == 'em':
                                permuted_pred = permuted_pred[:, 1]
                            else:
                                permuted_pred = permuted_pred[:, 0]
                        elif permuted_pred.shape[1] == 1:
                            permuted_pred = permuted_pred[:, 0]
                    
                    # Calculate importance as increase in MSE
                    permuted_mse = mean_squared_error(y_test_np, permuted_pred)
                    importance = permuted_mse - baseline_mse
                    scores.append(importance)
                
                importances.append(np.mean(scores))
            
            # Create DataFrame compatible with polynomial function output
            aggregated_df = pd.DataFrame({
                'Feature': feature_names,
                'Total_Importance': importances
            }).sort_values('Total_Importance', ascending=False)
            
            print(f"Top 10 most important features for {target_name} (Neural Network):")
            print(aggregated_df[['Feature', 'Total_Importance']].head(10).to_string(index=False))
            
            return {
                'aggregated_importance': aggregated_df,
                'feature_importances': np.array(importances),
                'method': 'permutation_importance_neural_net',
                'baseline_mse': baseline_mse
            }
            
        else:
            # Multi-output model (SCR + EM)
            if y_test_np.shape[1] != 2:
                raise ValueError(f"Expected 2 outputs (SCR, EM), got {y_test_np.shape[1]}")
            
            if baseline_pred.shape[1] != 2:
                raise ValueError(f"Model should output 2 values, got {baseline_pred.shape[1]}")
            
            # Calculate baseline performance for each output
            scr_baseline_mse = mean_squared_error(y_test_np[:, 0], baseline_pred[:, 0])
            em_baseline_mse = mean_squared_error(y_test_np[:, 1], baseline_pred[:, 1])
            combined_baseline_mse = (scr_baseline_mse + em_baseline_mse) / 2
            
            print(f"Baseline MSE - SCR: {scr_baseline_mse:.6f}, EM: {em_baseline_mse:.6f}")
            
            # Calculate permutation importance
            scr_importances = []
            em_importances = []
            combined_importances = []
            
            for feature_idx in range(len(feature_names)):
                scr_scores = []
                em_scores = []
                combined_scores = []
                
                for repeat in range(n_repeats):
                    # Create permuted copy
                    X_permuted = X_test_np.copy()
                    perm_indices = np.random.permutation(X_test_np.shape[0])
                    X_permuted[:, feature_idx] = X_permuted[perm_indices, feature_idx]
                    
                    # Get predictions on permuted data
                    permuted_pred = model.predict(X_permuted)
                    
                    # Calculate MSE for each output
                    scr_permuted_mse = mean_squared_error(y_test_np[:, 0], permuted_pred[:, 0])
                    em_permuted_mse = mean_squared_error(y_test_np[:, 1], permuted_pred[:, 1])
                    
                    # Calculate importance as increase in MSE
                    scr_importance = scr_permuted_mse - scr_baseline_mse
                    em_importance = em_permuted_mse - em_baseline_mse
                    combined_importance = ((scr_permuted_mse + em_permuted_mse) / 2) - combined_baseline_mse
                    
                    scr_scores.append(scr_importance)
                    em_scores.append(em_importance)
                    combined_scores.append(combined_importance)
                
                scr_importances.append(np.mean(scr_scores))
                em_importances.append(np.mean(em_scores))
                combined_importances.append(np.mean(combined_scores))
            
            # Create DataFrame compatible with polynomial function output
            aggregated_df = pd.DataFrame({
                'Feature': feature_names,
                'Total_Importance': combined_importances,
                'SCR_Importance': scr_importances,
                'EM_Importance': em_importances
            }).sort_values('Total_Importance', ascending=False)
            
            print(f"Top 10 most important features for {target_name} (Neural Network):")
            print(aggregated_df[['Feature', 'Total_Importance']].head(10).to_string(index=False))
            
            print(f"\nTop 5 features for SCR prediction:")
            scr_top = aggregated_df[['Feature', 'SCR_Importance']].head(5)
            print(scr_top.to_string(index=False))
            
            print(f"\nTop 5 features for EM prediction:")
            em_top = aggregated_df[['Feature', 'EM_Importance']].head(5)
            print(em_top.to_string(index=False))
            
            return {
                'aggregated_importance': aggregated_df,
                'feature_importances': np.array(combined_importances),
                'scr_importance': np.array(scr_importances),
                'em_importance': np.array(em_importances),
                'method': 'permutation_importance_neural_net',
                'baseline_mse': {
                    'scr': scr_baseline_mse,
                    'em': em_baseline_mse,
                    'combined': combined_baseline_mse
                }
            }
            
    except Exception as e:
        print(f"Error in feature importance analysis for {target_name}: {str(e)}")
        return None

# In[130]:  (cell 36)
importance_results = get_torch_mlp_feature_importance(
    model=best_models['scr']['results']['model']['model'],
    X_test=X_test,
    y_test=y_multi_test,
    feature_names=feature_cols,
    n_repeats=10,
    target_name= 'multi'
)

# --- Output [36] ---
# [1] type: stream
# (stdout)
# ['\n', 'Analyzing feature importance for multi...\n', 'Baseline MSE - SCR: 0.142911, EM: 0.050775\n', 'Top 10 most important features for multi (Neural Network):\n', ' Feature  Total_Importance\n', '    ZSK1          0.003681\n', '   Vola6          0.000901\n', '   Vola4          0.000724\n', 'Verlust7          0.000409\n', 'Verlust8          0.000282\n', '    ZSK2          0.000134\n', '    MR16          0.000096\n', '   Vola5          0.000030\n', '    ZSK3          0.000029\n', '    MR17          0.000024\n', '\n', 'Top 5 features for SCR prediction:\n', ' Feature  SCR_Importance\n', '    ZSK1        0.008085\n', '   Vola6       -0.000354\n', '   Vola4        0.000571\n', 'Verlust7        0.000363\n', 'Verlust8        0.000170\n', '\n', 'Top 5 features for EM prediction:\n', ' Feature  EM_Importance\n', '    ZSK1      -0.000722\n', '   Vola6       0.002155\n', '   Vola4       0.000878\n', 'Verlust7       0.000454\n', 'Verlust8       0.000394\n']
# --- End Output ---

# In[37]:
def get_polynomial_feature_importance(model, feature_names, target_name):
    """Extract feature importance from polynomial ridge models AND XGBoost"""
    
    print(f"\nAnalyzing feature importance for {target_name}...")
    
    try:
        # Navigate through nested pipelines
        if hasattr(model, 'named_steps'):
            # First level: get the inner model from the outer pipeline
            inner_model = model.named_steps['model']
            
            # Check if it's XGBoost MultiOutputRegressor
            if hasattr(inner_model, 'estimators_') and len(inner_model.estimators_) >= 2:
                # Check if estimators have feature_importances_ 
                if hasattr(inner_model.estimators_[0], 'feature_importances_'):
                    # XGBoost case
                    scr_model = inner_model.estimators_[0]  # SCR
                    em_model = inner_model.estimators_[1]   # EM
                    
                    scr_importance = scr_model.feature_importances_
                    em_importance = em_model.feature_importances_
                    combined_importance = (scr_importance + em_importance) / 2
                    
                    # Create importance DataFrame
                    aggregated_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Total_Importance': combined_importance,
                        'SCR_Importance': scr_importance,
                        'EM_Importance': em_importance
                    }).sort_values('Total_Importance', ascending=False)
                    
                    print(f"Top 10 most important features for {target_name} (XGBoost):")
                    print(aggregated_df[['Feature', 'Total_Importance']].head(10).to_string(index=False))
                    
                    print(f"\nTop 5 features for SCR prediction:")
                    scr_top = aggregated_df[['Feature', 'SCR_Importance']].head(5)
                    print(scr_top.to_string(index=False))
                    
                    print(f"\nTop 5 features for EM prediction:")
                    em_top = aggregated_df[['Feature', 'EM_Importance']].head(5)
                    print(em_top.to_string(index=False))
                    
                    return {
                        'aggregated_importance': aggregated_df,
                        'feature_importances': combined_importance,
                        'scr_importance': scr_importance,
                        'em_importance': em_importance
                    }
            
            # Check if inner model is also a pipeline (polynomial models)
            if hasattr(inner_model, 'named_steps') and 'poly' in inner_model.named_steps:
                # This is a polynomial model
                ridge_model = inner_model.named_steps['ridge']
                poly_features = inner_model.named_steps['poly']
                
                # Get polynomial feature names
                poly_feature_names = poly_features.get_feature_names_out(feature_names)
                coefficients = ridge_model.coef_
                
                # Calculate importance as absolute coefficient values
                importance_scores = np.abs(coefficients)
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'Feature': poly_feature_names,
                    'Coefficient': coefficients,
                    'Importance': importance_scores
                }).sort_values('Importance', ascending=False)
                
                # Aggregate by original feature (sum importance of all polynomial terms)
                original_importance = {}
                for original_feature in feature_names:
                    # Sum importance of all terms containing this feature
                    mask = [original_feature in feat for feat in poly_feature_names]
                    total_importance = importance_scores[mask].sum()
                    original_importance[original_feature] = total_importance
                
                # Create aggregated importance DataFrame
                aggregated_df = pd.DataFrame([
                    {'Feature': feat, 'Total_Importance': imp} 
                    for feat, imp in original_importance.items()
                ]).sort_values('Total_Importance', ascending=False)
                
                print(f"Top 10 most important original features for {target_name}:")
                print(aggregated_df.head(10).to_string(index=False))
                
                print(f"\nTop 10 most important polynomial terms for {target_name}:")
                print(importance_df[['Feature', 'Importance']].head(10).to_string(index=False))
                
                return {
                    'polynomial_importance': importance_df,
                    'aggregated_importance': aggregated_df,
                    'coefficients': coefficients
                }
            else:
                # Handle MultiOutputRegressor with linear models
                if hasattr(inner_model, 'estimators_') and len(inner_model.estimators_) >= 2:
                    # MultiOutputRegressor with linear models (e.g., Ridge, LinearRegression)
                    estimator_1 = inner_model.estimators_[0]  # SCR
                    estimator_2 = inner_model.estimators_[1]  # EM
                    
                    if hasattr(estimator_1, 'coef_') and hasattr(estimator_2, 'coef_'):
                        coef_1 = estimator_1.coef_
                        coef_2 = estimator_2.coef_
                        
                        # Average the coefficients for combined importance
                        combined_coef = (np.abs(coef_1) + np.abs(coef_2)) / 2
                        
                        aggregated_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Total_Importance': combined_coef,
                            'SCR_Importance': np.abs(coef_1),
                            'EM_Importance': np.abs(coef_2)
                        }).sort_values('Total_Importance', ascending=False)
                        
                        print(f"Top 10 most important features for {target_name} (MultiOutput Linear):")
                        print(aggregated_df[['Feature', 'Total_Importance']].head(10).to_string(index=False))
                        
                        print(f"\nTop 5 features for SCR prediction:")
                        scr_top = aggregated_df[['Feature', 'SCR_Importance']].head(5)
                        print(scr_top.to_string(index=False))
                        
                        print(f"\nTop 5 features for EM prediction:")
                        em_top = aggregated_df[['Feature', 'EM_Importance']].head(5)
                        print(em_top.to_string(index=False))
                        
                        return {
                            'aggregated_importance': aggregated_df,
                            'coefficients': combined_coef,
                            'scr_coefficients': coef_1,
                            'em_coefficients': coef_2
                        }
                    else:
                        print(f"MultiOutputRegressor estimators don't have coef_ attribute")
                        return None
                else:
                    # Single output linear model
                    if hasattr(inner_model, 'coef_'):
                        coefficients = inner_model.coef_
                        importance_scores = np.abs(coefficients)
                        
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Coefficient': coefficients,
                            'Importance': importance_scores
                        }).sort_values('Importance', ascending=False)
                        
                        print(f"Top 10 most important features for {target_name}:")
                        print(importance_df.head(10).to_string(index=False))
                        
                        return {
                            'aggregated_importance': importance_df,
                            'coefficients': coefficients
                        }
                    else:
                        print(f"Model does not have coef_ or feature_importances_ attribute")
                        return None
        else:
            print(f"Model structure not recognized for {target_name}")
            return None
            
    except Exception as e:
        print(f"Error in feature importance analysis for {target_name}: {str(e)}")
        print(f"Model structure: {type(model)}")
        if hasattr(model, 'named_steps'):
            print(f"Outer pipeline steps: {list(model.named_steps.keys())}")
            if 'model' in model.named_steps and hasattr(model.named_steps['model'], 'named_steps'):
                print(f"Inner pipeline steps: {list(model.named_steps['model'].named_steps.keys())}")
            elif 'model' in model.named_steps and hasattr(model.named_steps['model'], 'estimators_'):
                print(f"MultiOutputRegressor with {len(model.named_steps['model'].estimators_)} estimators")
        return None

def detect_joint_model(results):
    # Prefer looking at stored predictions
    preds = results.get('predictions', {})
    test_preds = preds.get('test', None)
    if isinstance(test_preds, np.ndarray) and test_preds.ndim == 2 and test_preds.shape[1] >= 2:
        return True
    # Fallback: inner estimator being a MultiOutputRegressor
    try:
        inner = results['model'].named_steps['model']
        from sklearn.multioutput import MultiOutputRegressor
        return isinstance(inner, MultiOutputRegressor)
    except Exception:
        return False

# Perform feature importance analysis for best models INCLUDING multi-output
feature_importance_results = {}

for target, best_info in best_models.items():
    model = best_info['results']['model']       # NEW -------------------
    
    # Detect if model is a neural network (TorchMLPReg)
    is_neural_network = False   
    try:
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            inner_model = model.named_steps['model']
            if hasattr(inner_model, 'module_') or 'neural' in str(type(inner_model)).lower():
                is_neural_network = True
    except:
        pass

    if is_neural_network:
        # Get the appropriate test data based on target
        if target == 'multi_output':
            # For multi-output model, combine SCR and EM test data
            y_test_data = y_multi_test #np.column_stack([y_scr_test, y_em_test])
        elif target == 'scr':
            y_test_data = y_scr_test
        elif target == 'em':
            y_test_data = y_em_test
        elif target == 'quote':
            y_test_data = y_quote_test
        else:
            print(f"Unknown target: {target}")
            continue
        
        importance_result = get_torch_mlp_feature_importance(
            model['model'], X_test=X_test, y_test=y_test_data, feature_names=feature_cols, target_name=target
        )
    else:
        importance_result = get_polynomial_feature_importance(
        best_info['results']['model'], feature_cols, target.upper()
    )
    if importance_result:
        feature_importance_results[target] = importance_result

print(f"\nFeature importance analysis completed for {len(feature_importance_results)} models")

# --- Output [37] ---
# [1] type: stream
# (stdout)
# ['\n', 'Analyzing feature importance for quote...\n', 'Baseline MSE for quote: 0.952105\n', 'Top 10 most important features for quote (Neural Network):\n', 'Feature  Total_Importance\n', '  Vola6          0.000703\n', '   MR19          0.000070\n', '   ZSK3          0.000030\n', '    MR9          0.000017\n', '   MR18          0.000010\n', '   MR20          0.000007\n', '   MR12         -0.000012\n', '   MR15         -0.000020\n', '   MR14         -0.000021\n', '   MR10         -0.000024\n', '\n', 'Analyzing feature importance for scr...\n', 'Baseline MSE for scr: 0.142911\n', 'Top 10 most important features for scr (Neural Network):\n', ' Feature  Total_Importance\n', '    ZSK1          0.008231\n', '   Vola4          0.000520\n', 'Verlust7          0.000345\n', '    ZSK2          0.000196\n', 'Verlust8          0.000158\n', '    MR19          0.000148\n', '    MR16          0.000091\n', '    ZSK3          0.000032\n', '    MR15          0.000025\n', '    MR20          0.000023\n', '\n', 'Analyzing feature importance for em...\n', 'Baseline MSE for em: 0.050775\n', 'Top 10 most important features for em (Neural Network):\n', ' Feature  Total_Importance\n', '   Vola6          0.002179\n', '   Vola4          0.000897\n', 'Verlust7          0.000415\n', 'Verlust8          0.000393\n', '   Vola5          0.000105\n', '    MR16          0.000101\n', '    MR17          0.000026\n', '    ZSK3          0.000018\n', '    ZSK2          0.000016\n', '    MR12          0.000008\n', '\n', 'Analyzing feature importance for multi_output...\n', 'Baseline MSE - SCR: 0.142911, EM: 0.050775\n', 'Top 10 most important features for multi_output (Neural Network):\n', ' Feature  Total_Importance\n', '    ZSK1          0.003651\n', '   Vola6          0.000919\n', '   Vola4          0.000685\n', 'Verlust7          0.000394\n', 'Verlust8          0.000293\n', '    ZSK2          0.000158\n', '    MR16          0.000101\n', '   Vola5          0.000049\n', '    ZSK3          0.000031\n', '    MR17          0.000025\n', '\n', 'Top 5 features for SCR prediction:\n', ' Feature  SCR_Importance\n', '    ZSK1        0.008047\n', '   Vola6       -0.000344\n', '   Vola4        0.000519\n', 'Verlust7        0.000354\n', 'Verlust8        0.000183\n', '\n', 'Top 5 features for EM prediction:\n', ' Feature  EM_Importance\n', '    ZSK1      -0.000744\n', '   Vola6       0.002183\n', '   Vola4       0.000852\n', 'Verlust7       0.000435\n', 'Verlust8       0.000404\n', '\n', 'Feature importance analysis completed for 4 models\n']
# --- End Output ---

# In[38]:
def analyze_prediction_stability(model, X_test, y_test, target_name, is_joint_model=False, target_scalers=None):
    """Analyze prediction stability across different data segments - improved for neural networks"""
    
    print(f"\nAnalyzing prediction stability for {target_name}...")
    
    try:
        # Get predictions with error handling
        raw_predictions = model.predict(X_test)
        
        # Convert to numpy if needed (handles PyTorch tensors)
        if hasattr(raw_predictions, 'numpy'):
            raw_predictions = raw_predictions.numpy()
        elif hasattr(raw_predictions, 'detach'):
            raw_predictions = raw_predictions.detach().numpy()
        
        raw_predictions = np.array(raw_predictions)
        
        print(f"  Raw predictions shape: {raw_predictions.shape}")
        print(f"  Target test shape: {y_test.shape}")
        
    except Exception as e:
        print(f"  ERROR: Failed to get predictions: {str(e)}")
        return None
    
    # Handle joint models (SCR+EM → Quote)
    if is_joint_model and target_name == 'QUOTE' and target_scalers:
        print(f"  Joint model detected - calculating Quote from SCR+EM predictions")
        
        try:
            # Ensure we have 2D output for joint model
            if raw_predictions.ndim == 1:
                raise ValueError("Joint model expected 2D output but got 1D")
            elif raw_predictions.shape[1] < 2:
                raise ValueError(f"Joint model expected at least 2 outputs, got {raw_predictions.shape[1]}")
            
            # Extract SCR and EM predictions (normalized)
            scr_pred_norm = raw_predictions[:, 0].ravel()
            em_pred_norm = raw_predictions[:, 1].ravel()
            
            print(f"  SCR predictions range: [{np.min(scr_pred_norm):.4f}, {np.max(scr_pred_norm):.4f}]")
            print(f"  EM predictions range: [{np.min(em_pred_norm):.4f}, {np.max(em_pred_norm):.4f}]")
            
            # Convert to original scale for proper Quote calculation
            scr_pred_orig = target_scalers['scr'].inverse_transform(scr_pred_norm.reshape(-1, 1)).ravel()
            em_pred_orig = target_scalers['em'].inverse_transform(em_pred_norm.reshape(-1, 1)).ravel()
            
            # Calculate Quote = EM/SCR (in original scale) with better error handling
            quote_pred_orig = np.where(
                np.abs(scr_pred_orig) > 1e-6, 
                em_pred_orig / scr_pred_orig, 
                np.nan  # Use NaN instead of 0 for invalid divisions
            )
            
            # Handle NaN values
            nan_count = np.sum(np.isnan(quote_pred_orig))
            if nan_count > 0:
                print(f"  WARNING: {nan_count} NaN values in quote predictions due to division by small SCR values")
                # Option 1: Remove NaN values
                valid_mask = ~np.isnan(quote_pred_orig)
                if np.sum(valid_mask) == 0:
                    print(f"  ERROR: All quote predictions are NaN")
                    return None
            else:
                valid_mask = np.ones(len(quote_pred_orig), dtype=bool)
            
            # Convert Quote back to normalized scale for consistent comparison with y_test
            try:
                quote_pred_norm = target_scalers['quote'].transform(
                    quote_pred_orig[valid_mask].reshape(-1, 1)
                ).ravel()
                
                # Pad back to original size if we removed NaNs
                if nan_count > 0:
                    y_pred = np.full(len(quote_pred_orig), np.nan)
                    y_pred[valid_mask] = quote_pred_norm
                else:
                    y_pred = quote_pred_norm
                    
            except Exception as e:
                print(f"  ERROR: Failed to transform quote predictions: {str(e)}")
                return None
            
            # Regulatory threshold analysis (original scale)
            try:
                y_test_orig = target_scalers['quote'].inverse_transform(np.array(y_test).reshape(-1, 1)).ravel()
                
                print(f"\nREGULATORY THRESHOLD ANALYSIS (Original Scale):")
                regulatory_ranges = [
                    ('Insolvent (< 0)', y_test_orig < 0),
                    ('Undercapitalized (0-100%)', (y_test_orig >= 0) & (y_test_orig < 1)),
                    ('Adequate (100-200%)', (y_test_orig >= 1) & (y_test_orig <= 2)),
                    ('Well-Capitalized (> 200%)', y_test_orig > 2)
                ]
                
                range_results_regulatory = {}
                for range_name, mask in regulatory_ranges:
                    # Combine with valid_mask to handle NaN predictions
                    combined_mask = mask & valid_mask
                    if np.sum(combined_mask) > 0:
                        y_test_range = y_test_orig[combined_mask]
                        y_pred_range = quote_pred_orig[combined_mask]
                        
                        range_rmse = np.sqrt(mean_squared_error(y_test_range, y_pred_range))
                        range_mae = mean_absolute_error(y_test_range, y_pred_range)
                        try:
                            range_r2 = r2_score(y_test_range, y_pred_range)
                        except:
                            range_r2 = np.nan
                        
                        range_results_regulatory[range_name] = {
                            'count': np.sum(combined_mask),
                            'percentage': np.sum(combined_mask) / len(y_test) * 100,
                            'rmse': range_rmse,
                            'mae': range_mae,
                            'r2': range_r2
                        }
                        
                        print(f"  {range_name} ({np.sum(combined_mask)} obs, {np.sum(combined_mask)/len(y_test)*100:.1f}%):")
                        print(f"    RMSE: {range_rmse:.4f}")
                        print(f"    MAE: {range_mae:.4f}")
                        print(f"    R2: {range_r2:.4f}")
                        
            except Exception as e:
                print(f"  ERROR in regulatory analysis: {str(e)}")
                range_results_regulatory = {}
                
        except Exception as e:
            print(f"  ERROR: Failed joint model processing: {str(e)}")
            return None
            
    elif is_joint_model and target_name in ['SCR', 'EM']:
        # For multi-output models predicting SCR or EM directly
        try:
            print(f"  DEBUG: Raw predictions shape for {target_name}: {raw_predictions.shape}")
            print(f"  DEBUG: Expected target shape: {y_test.shape}")
            
            if raw_predictions.ndim == 1:
                # Handle flattened multi-output: [scr1, scr2, ..., em1, em2, ...]
                total_samples = len(y_test)
                if len(raw_predictions) == 2 * total_samples:
                    print(f"  INFO: Detected flattened multi-output format")
                    if target_name == 'SCR':
                        y_pred = raw_predictions[:total_samples]  # First half
                    else:  # EM
                        y_pred = raw_predictions[total_samples:]  # Second half
                elif len(raw_predictions) == total_samples:
                    print(f"  WARNING: Single output detected, using as {target_name}")
                    y_pred = raw_predictions
                else:
                    raise ValueError(f"Cannot match prediction length {len(raw_predictions)} to target length {total_samples}")
            
            elif raw_predictions.ndim == 2:
                # Standard multi-output format: [[scr1, em1], [scr2, em2], ...]
                if raw_predictions.shape[0] == len(y_test):
                    # Normal case: rows = samples, cols = outputs
                    if target_name == 'SCR':
                        y_pred = raw_predictions[:, 0]
                    else:  # EM
                        y_pred = raw_predictions[:, 1] if raw_predictions.shape[1] > 1 else raw_predictions[:, 0]
                        
                elif raw_predictions.shape[1] == len(y_test):
                    # Transposed case: cols = samples, rows = outputs
                    print(f"  INFO: Detected transposed output format")
                    if target_name == 'SCR':
                        y_pred = raw_predictions[0, :]
                    else:  # EM
                        y_pred = raw_predictions[1, :] if raw_predictions.shape[0] > 1 else raw_predictions[0, :]
                        
                else:
                    # Try to reshape if total elements match
                    total_elements = raw_predictions.size
                    target_samples = len(y_test)
                    
                    if total_elements == 2 * target_samples:
                        print(f"  INFO: Reshaping {raw_predictions.shape} to ({target_samples}, 2)")
                        reshaped_preds = raw_predictions.reshape(target_samples, 2)
                        if target_name == 'SCR':
                            y_pred = reshaped_preds[:, 0]
                        else:  # EM
                            y_pred = reshaped_preds[:, 1]
                    else:
                        raise ValueError(f"Cannot match prediction shape {raw_predictions.shape} to target length {target_samples}")
            
            else:
                raise ValueError(f"Unsupported prediction dimensionality: {raw_predictions.ndim}")
                
            print(f"  DEBUG: Extracted {target_name} predictions shape: {y_pred.shape}")
            
        except Exception as e:
            print(f"  ERROR: Failed to extract {target_name} from joint model: {str(e)}")
            return None
    else:
        # Direct single-output predictions
        y_pred = raw_predictions
    
    # Ensure arrays are 1D and handle shape mismatches
    try:
        if hasattr(y_pred, 'ravel'):
            y_pred = y_pred.ravel()
        if hasattr(y_test, 'ravel'):
            y_test = y_test.ravel()
            
        # Handle length mismatches
        if len(y_pred) != len(y_test):
            min_len = min(len(y_pred), len(y_test))
            print(f"  WARNING: Length mismatch. Truncating to {min_len} samples.")
            y_pred = y_pred[:min_len]
            y_test = y_test[:min_len]
            
    except Exception as e:
        print(f"  ERROR: Failed array processing: {str(e)}")
        return None
    
    # Remove NaN values for metric calculation
    try:
        valid_mask = ~(np.isnan(y_pred) | np.isnan(y_test) | np.isinf(y_pred) | np.isinf(y_test))
        if np.sum(valid_mask) == 0:
            print(f"  ERROR: No valid predictions after removing NaN/inf values")
            return None
        elif np.sum(valid_mask) < len(y_pred):
            print(f"  WARNING: Removed {len(y_pred) - np.sum(valid_mask)} invalid predictions")
            y_pred = y_pred[valid_mask]
            y_test = y_test[valid_mask]
    except Exception as e:
        print(f"  ERROR: Failed NaN handling: {str(e)}")
        return None
    
    # Overall performance (using NORMALIZED data for consistency)
    try:
        overall_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        overall_r2 = r2_score(y_test, y_pred)
        overall_mae = mean_absolute_error(y_test, y_pred)
        
        print(f"  Overall Performance:")
        print(f"    RMSE: {overall_rmse:.4f}")
        print(f"    MAE: {overall_mae:.4f}")
        print(f"    R2: {overall_r2:.4f}")
        
    except Exception as e:
        print(f"  ERROR: Failed to calculate overall metrics: {str(e)}")
        return None
    
    # Analyze performance across different target value ranges
    try:
        if target_name == 'QUOTE':
            # For Quote, use quartile-based analysis on normalized scale
            q25, q50, q75 = np.percentile(y_test, [25, 50, 75])
            ranges = [
                ('Q1 (Lowest 25%)', y_test <= q25),
                ('Q2 (25-50%)', (y_test > q25) & (y_test <= q50)),
                ('Q3 (50-75%)', (y_test > q50) & (y_test <= q75)),
                ('Q4 (Highest 25%)', y_test > q75)
            ]
            print(f"\nPERFORMANCE BY QUARTILES (normalized scale):")
        else:
            # Quartile-based analysis for SCR and EM
            q25, q50, q75 = np.percentile(y_test, [25, 50, 75])
            ranges = [
                ('Q1 (Low)', y_test <= q25),
                ('Q2 (Med-Low)', (y_test > q25) & (y_test <= q50)),
                ('Q3 (Med-High)', (y_test > q50) & (y_test <= q75)),
                ('Q4 (High)', y_test > q75)
            ]
            print(f"\nPerformance by {target_name} quartiles (normalized scale):")
        
        range_results = {}
        for range_name, mask in ranges:
            if np.sum(mask) > 0:
                try:
                    range_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
                    range_mae = mean_absolute_error(y_test[mask], y_pred[mask])
                    range_r2 = r2_score(y_test[mask], y_pred[mask])
                    
                    range_results[range_name] = {
                        'count': np.sum(mask),
                        'percentage': np.sum(mask) / len(y_test) * 100,
                        'rmse': range_rmse,
                        'mae': range_mae,
                        'r2': range_r2
                    }
                    
                    print(f"  {range_name} ({np.sum(mask)} obs, {np.sum(mask)/len(y_test)*100:.1f}%):")
                    print(f"    RMSE: {range_rmse:.4f}, MAE: {range_mae:.4f}, R2: {range_r2:.4f}")
                    
                except Exception as e:
                    print(f"    ERROR calculating metrics for {range_name}: {str(e)}")
                    range_results[range_name] = {
                        'count': np.sum(mask),
                        'percentage': np.sum(mask) / len(y_test) * 100,
                        'rmse': np.nan,
                        'mae': np.nan,
                        'r2': np.nan
                    }
                    
    except Exception as e:
        print(f"  ERROR: Failed range analysis: {str(e)}")
        range_results = {}
    
    # Additional stability metrics
    try:
        residuals = y_test - y_pred
        residual_std = np.std(residuals)
    except:
        residual_std = np.nan
    
    # Warning for concerning patterns
    if overall_r2 < 0:
        print(f"  WARNING: Negative R2 ({overall_r2:.4f}) indicates model performs worse than predicting the mean!")
    
    # Check for negative R2 in ranges
    negative_r2_ranges = [name for name, results in range_results.items() if results['r2'] < 0]
    if negative_r2_ranges:
        print(f"  WARNING: Negative R2 in ranges: {', '.join(negative_r2_ranges)}")
    
    return {
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'overall_r2': overall_r2,
        'range_analysis': range_results,
        'residual_std': residual_std,
        'is_joint_model': is_joint_model
    }

# In[39]:

# Perform stability analysis for ALL best models including multi-output
stability_results = {}

for target, best_info in best_models.items():
    try:
        # FIXED: Pass the entire results dictionary, not just the model
        is_joint = detect_joint_model(best_info['results'])
        
        if target == 'multi_output':
            # Handle multi-output model, analyze each component
            print(f"\nAnalyzing multi-output model components...")
            
            # SCR component
            try:
                scr_stability = analyze_prediction_stability(
                    best_info['results']['model'], X_test, y_scr_test, 
                    'SCR', is_joint_model=True, target_scalers=target_scalers
                )
                stability_results[f'{target}_scr'] = scr_stability
            except Exception as e:
                print(f"Error analyzing SCR component: {str(e)}")
            
            # EM component  
            try:
                em_stability = analyze_prediction_stability(
                    best_info['results']['model'], X_test, y_em_test, 
                    'EM', is_joint_model=True, target_scalers=target_scalers
                )
                stability_results[f'{target}_em'] = em_stability
            except Exception as e:
                print(f"Error analyzing EM component: {str(e)}")
                
        else:
            # Get the appropriate test target
            if target == 'quote':
                y_target_test = y_quote_test
            elif target == 'scr':
                y_target_test = y_scr_test
            elif target == 'em':
                y_target_test = y_em_test
            else:
                print(f"Unknown target: {target}")
                continue
            
            # Run stability analysis
            stability_result = analyze_prediction_stability(
                best_info['results']['model'], X_test, y_target_test, 
                target.upper(), is_joint_model=is_joint, target_scalers=target_scalers
            )
            
            stability_results[target] = stability_result
        
    except Exception as e:
        print(f"Error in stability analysis for {target}: {str(e)}")
        stability_results[target] = None

print(f"\nStability analysis completed for {len([k for k, v in stability_results.items() if v is not None])} model(s)")

# Summary of concerning patterns
print(f"\nSTABILITY ANALYSIS SUMMARY:")
print(f"="*50)

for target, result in stability_results.items():
    if result is not None:
        print(f"{target.upper()}:")
        print(f"  Overall R2: {result['overall_r2']:.4f}")
        print(f"  Overall RMSE: {result['overall_rmse']:.4f}")
        
        # Check for negative R2 in ranges
        if 'range_analysis' in result:
            negative_ranges = [name for name, data in result['range_analysis'].items() if data['r2'] < 0]
            if negative_ranges:
                print(f"  CONCERN: Negative R2 in: {', '.join(negative_ranges)}")
            else:
                print(f"  STATUS: All ranges show positive R2")
        print()

# --- Output [39] ---
# [1] type: stream
# (stdout)
# ['\n', 'Analyzing prediction stability for QUOTE...\n', '  Raw predictions shape: (2046, 2)\n', '  Target test shape: (2046,)\n', '  Joint model detected - calculating Quote from SCR+EM predictions\n', '  SCR predictions range: [-0.9834, 0.6907]\n', '  EM predictions range: [-0.4919, 0.9456]\n', '\n', 'REGULATORY THRESHOLD ANALYSIS (Original Scale):\n', '  Insolvent (< 0) (54 obs, 2.6%):\n', '    RMSE: 0.4120\n', '    MAE: 0.2421\n', '    R2: -0.9477\n', '  Undercapitalized (0-100%) (288 obs, 14.1%):\n', '    RMSE: 0.1719\n', '    MAE: 0.1040\n', '    R2: 0.6216\n', '  Adequate (100-200%) (438 obs, 21.4%):\n', '    RMSE: 0.1755\n', '    MAE: 0.1162\n', '    R2: 0.6379\n', '  Well-Capitalized (> 200%) (1266 obs, 61.9%):\n', '    RMSE: 0.1849\n', '    MAE: 0.1218\n', '    R2: 0.8193\n', '  Overall Performance:\n', '    RMSE: 0.0645\n', '    MAE: 0.0410\n', '    R2: 0.9646\n', '\n', 'PERFORMANCE BY QUARTILES (normalized scale):\n', '  Q1 (Lowest 25%) (512 obs, 25.0%):\n', '    RMSE: 0.0679, MAE: 0.0386, R2: 0.8460\n', '  Q2 (25-50%) (511 obs, 25.0%):\n', '    RMSE: 0.0641, MAE: 0.0446, R2: 0.5727\n', '  Q3 (50-75%) (511 obs, 25.0%):\n', '    RMSE: 0.0508, MAE: 0.0359, R2: -0.0920\n', '  Q4 (Highest 25%) (512 obs, 25.0%):\n', '    RMSE: 0.0729, MAE: 0.0448, R2: 0.0954\n', '  WARNING: Negative R2 in ranges: Q3 (50-75%)\n', '\n', 'Analyzing prediction stability for SCR...\n', '  Raw predictions shape: (2046, 2)\n', '  Target test shape: (2046,)\n', '  WARNING: Length mismatch. Truncating to 2046 samples.\n', '  Overall Performance:\n', '    RMSE: 0.8874\n', '    MAE: 0.7239\n', '    R2: -6.3558\n', '\n', 'Performance by SCR quartiles (normalized scale):\n', '  Q1 (Low) (512 obs, 25.0%):\n', '    RMSE: 1.0053, MAE: 0.8058, R2: -752.4523\n', '  Q2 (Med-Low) (511 obs, 25.0%):\n', '    RMSE: 0.9743, MAE: 0.7768, R2: -1372.7583\n', '  Q3 (Med-High) (511 obs, 25.0%):\n', '    RMSE: 0.8494, MAE: 0.7021, R2: -90.7393\n', '  Q4 (High) (512 obs, 25.0%):\n', '    RMSE: 0.6847, MAE: 0.6109, R2: -6.0745\n', '  WARNING: Negative R2 (-6.3558) indicates model performs worse than predicting the mean!\n', '  WARNING: Negative R2 in ranges: Q1 (Low), Q2 (Med-Low), Q3 (Med-High), Q4 (High)\n', '\n', 'Analyzing prediction stability for EM...\n', '  Raw predictions shape: (2046, 2)\n', '  Target test shape: (2046,)\n', '  WARNING: Length mismatch. Truncating to 2046 samples.\n', '  Overall Performance:\n', '    RMSE: 0.8442\n', '    MAE: 0.6554\n', '    R2: -15.8945\n', '\n', 'Performance by EM quartiles (normalized scale):\n', '  Q1 (Low) (512 obs, 25.0%):\n', '    RMSE: 0.7329, MAE: 0.6130, R2: -10.8634\n', '  Q2 (Med-Low) (511 obs, 25.0%):\n', '    RMSE: 0.7964, MAE: 0.5980, R2: -813.2103\n', '  Q3 (Med-High) (511 obs, 25.0%):\n', '    RMSE: 0.8513, MAE: 0.6397, R2: -896.0581\n', '  Q4 (High) (512 obs, 25.0%):\n', '    RMSE: 0.9772, MAE: 0.7707, R2: -255.6614\n', '  WARNING: Negative R2 (-15.8945) indicates model performs worse than predicting the mean!\n', '  WARNING: Negative R2 in ranges: Q1 (Low), Q2 (Med-Low), Q3 (Med-High), Q4 (High)\n', '\n', 'Analyzing multi-output model components...\n', '\n', 'Analyzing prediction stability for SCR...\n', '  Raw predictions shape: (2046, 2)\n', '  Target test shape: (2046,)\n', '  DEBUG: Raw predictions shape for SCR: (2046, 2)\n', '  DEBUG: Expected target shape: (2046,)\n', '  DEBUG: Extracted SCR predictions shape: (2046,)\n', '  Overall Performance:\n', '    RMSE: 0.0791\n', '    MAE: 0.0379\n', '    R2: 0.9416\n', '\n', 'Performance by SCR quartiles (normalized scale):\n', '  Q1 (Low) (512 obs, 25.0%):\n', '    RMSE: 0.0320, MAE: 0.0217, R2: 0.2350\n', '  Q2 (Med-Low) (511 obs, 25.0%):\n', '    RMSE: 0.0266, MAE: 0.0171, R2: -0.0258\n', '  Q3 (Med-High) (511 obs, 25.0%):\n', '    RMSE: 0.0494, MAE: 0.0335, R2: 0.6900\n', '  Q4 (High) (512 obs, 25.0%):\n', '    RMSE: 0.1443, MAE: 0.0792, R2: 0.6859\n', '  WARNING: Negative R2 in ranges: Q2 (Med-Low)\n', '\n', 'Analyzing prediction stability for EM...\n', '  Raw predictions shape: (2046, 2)\n', '  Target test shape: (2046,)\n', '  DEBUG: Raw predictions shape for EM: (2046, 2)\n', '  DEBUG: Expected target shape: (2046,)\n', '  DEBUG: Extracted EM predictions shape: (2046,)\n', '  Overall Performance:\n', '    RMSE: 0.0514\n', '    MAE: 0.0226\n', '    R2: 0.9373\n', '\n', 'Performance by EM quartiles (normalized scale):\n', '  Q1 (Low) (512 obs, 25.0%):\n', '    RMSE: 0.0948, MAE: 0.0464, R2: 0.8015\n', '  Q2 (Med-Low) (511 obs, 25.0%):\n', '    RMSE: 0.0230, MAE: 0.0151, R2: 0.3236\n', '  Q3 (Med-High) (511 obs, 25.0%):\n', '    RMSE: 0.0210, MAE: 0.0140, R2: 0.4548\n', '  Q4 (High) (512 obs, 25.0%):\n', '    RMSE: 0.0249, MAE: 0.0150, R2: 0.8334\n', '\n', 'Stability analysis completed for 5 model(s)\n', '\n', 'STABILITY ANALYSIS SUMMARY:\n', '==================================================\n', 'QUOTE:\n', '  Overall R2: 0.9646\n', '  Overall RMSE: 0.0645\n', '  CONCERN: Negative R2 in: Q3 (50-75%)\n', '\n', 'SCR:\n', '  Overall R2: -6.3558\n', '  Overall RMSE: 0.8874\n', '  CONCERN: Negative R2 in: Q1 (Low), Q2 (Med-Low), Q3 (Med-High), Q4 (High)\n', '\n', 'EM:\n', '  Overall R2: -15.8945\n', '  Overall RMSE: 0.8442\n', '  CONCERN: Negative R2 in: Q1 (Low), Q2 (Med-Low), Q3 (Med-High), Q4 (High)\n', '\n', 'MULTI_OUTPUT_SCR:\n', '  Overall R2: 0.9416\n', '  Overall RMSE: 0.0791\n', '  CONCERN: Negative R2 in: Q2 (Med-Low)\n', '\n', 'MULTI_OUTPUT_EM:\n', '  Overall R2: 0.9373\n', '  Overall RMSE: 0.0514\n', '  STATUS: All ranges show positive R2\n', '\n']
# --- End Output ---

# In[191]:  (cell 40)
plt.style.use('default')
sns.set_palette("husl")

def safe_extract_predictions(results, target, is_joint=False, target_scalers=None):
    """Safely extract predictions from results, handling both direct and joint models"""
    try:
        if not results or not isinstance(results, dict):
            return {}
            
        predictions = results.get('predictions', {})
        if not isinstance(predictions, dict):
            return {}
            
        if target == 'quote' and is_joint:
            # For joint models, calculate Quote from SCR+EM predictions
            if 'test' in predictions:
                pred_data = predictions['test']
                if hasattr(pred_data, 'ndim') and pred_data.ndim == 2 and pred_data.shape[1] >= 2:
                    scr_pred_norm = pred_data[:, 0]
                    em_pred_norm = pred_data[:, 1]
                    
                    scr_pred_orig = target_scalers['scr'].inverse_transform(scr_pred_norm.reshape(-1, 1)).ravel()
                    em_pred_orig = target_scalers['em'].inverse_transform(em_pred_norm.reshape(-1, 1)).ravel()
                    
                    quote_pred_orig = np.where(np.abs(scr_pred_orig) > 1e-6, em_pred_orig / scr_pred_orig, 0)
                    quote_pred_norm = target_scalers['quote'].transform(quote_pred_orig.reshape(-1, 1)).ravel()
                    
                    return {
                        'test_orig': quote_pred_orig,
                        'test_norm': quote_pred_norm,
                        'test': quote_pred_norm
                    }
        
        # For direct models - ensure numpy arrays
        cleaned_predictions = {}
        for key, value in predictions.items():
            if value is not None:
                if hasattr(value, 'values'):  # pandas Series/DataFrame
                    cleaned_predictions[key] = np.array(value.values).ravel()
                elif hasattr(value, 'ravel'):  # numpy array
                    cleaned_predictions[key] = value.ravel()
                else:
                    cleaned_predictions[key] = np.array(value).ravel()
        
        return cleaned_predictions
        
    except Exception as e:
        print(f"Error extracting predictions for {target}: {str(e)}")
        return {}

def safe_extract_actuals(results, target):
    """Safely extract actual values from results"""
    try:
        if not results or not isinstance(results, dict):
            return {}
            
        actuals = results.get('actuals', {})
        if not isinstance(actuals, dict):
            return {}
            
        cleaned_actuals = {}
        for key, value in actuals.items():
            if value is not None:
                if hasattr(value, 'values'):  # pandas Series/DataFrame
                    cleaned_actuals[key] = np.array(value.values).ravel()
                elif hasattr(value, 'ravel'):  # numpy array
                    cleaned_actuals[key] = value.ravel()
                else:
                    cleaned_actuals[key] = np.array(value).ravel()
        
        return cleaned_actuals
        
    except Exception as e:
        print(f"Error extracting actuals for {target}: {str(e)}")
        return {}

def is_component_model(results, target):
    """Check if this is a component model from a joint/multi-output model"""
    if not isinstance(results, dict):
        return False
    
    model_name = results.get('model_name', '')
    return ('Component' in model_name or 
            'component' in model_name.lower() or
            target in ['scr', 'em'] and 'Enhanced' in model_name)

def plot_predicted_vs_actual_fixed(y_true, y_pred, title="Predicted vs Actual", 
                                 scale_type="original", target_scalers=None):
    """Fixed plotting function"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    
    if 'Quote' in title:
        if scale_type == "original" or scale_type == "calculated":
            insolvent_mask = y_true < 0
            undercap_mask = (y_true >= 0) & (y_true < 1)
            adequate_mask = (y_true >= 1) & (y_true <= 2)
            wellcap_mask = y_true > 2
        else:
            q25, q50, q75 = np.percentile(y_true, [25, 50, 75])
            insolvent_mask = y_true <= q25
            undercap_mask = (y_true > q25) & (y_true <= q50)
            adequate_mask = (y_true > q50) & (y_true <= q75)
            wellcap_mask = y_true > q75
                
        if np.any(insolvent_mask):
            plt.scatter(y_true[insolvent_mask], y_pred[insolvent_mask], 
                       alpha=0.7, s=25, c='red', label='Insolvent', marker='o')
        if np.any(undercap_mask):
            plt.scatter(y_true[undercap_mask], y_pred[undercap_mask], 
                       alpha=0.7, s=25, c='orange', label='Undercapitalized', marker='s')
        if np.any(adequate_mask):
            plt.scatter(y_true[adequate_mask], y_pred[adequate_mask], 
                       alpha=0.7, s=25, c='gold', label='Adequate', marker='^')
        if np.any(wellcap_mask):
            plt.scatter(y_true[wellcap_mask], y_pred[wellcap_mask], 
                       alpha=0.7, s=25, c='green', label='Well-Capitalized', marker='D')
        
        if scale_type == "original" or scale_type == "calculated":
            plt.axhline(y=0, color='red', linestyle=':', alpha=0.7, linewidth=1)
            plt.axhline(y=1, color='orange', linestyle=':', alpha=0.7, linewidth=1)
            plt.axhline(y=2, color='green', linestyle=':', alpha=0.7, linewidth=1)
            plt.axvline(x=0, color='red', linestyle=':', alpha=0.7, linewidth=1)
            plt.axvline(x=1, color='orange', linestyle=':', alpha=0.7, linewidth=1)
            plt.axvline(x=2, color='green', linestyle=':', alpha=0.7, linewidth=1)
    else:
        q25, q50, q75 = np.percentile(y_true, [25, 50, 75])
        
        q1_mask = y_true <= q25
        q2_mask = (y_true > q25) & (y_true <= q50)
        q3_mask = (y_true > q50) & (y_true <= q75)
        q4_mask = y_true > q75
        
        plt.scatter(y_true[q1_mask], y_pred[q1_mask], 
                   alpha=0.7, s=25, c='lightcoral', label='Q1 (Lowest 25%)', marker='o')
        plt.scatter(y_true[q2_mask], y_pred[q2_mask], 
                   alpha=0.7, s=25, c='orange', label='Q2 (25-50%)', marker='s')
        plt.scatter(y_true[q3_mask], y_pred[q3_mask], 
                   alpha=0.7, s=25, c='lightblue', label='Q3 (50-75%)', marker='^')
        plt.scatter(y_true[q4_mask], y_pred[q4_mask], 
                   alpha=0.7, s=25, c='darkgreen', label='Q4 (Top 25%)', marker='D')
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    plt.xlabel(f'Actual Values ({scale_type.title()} Scale)')
    plt.ylabel(f'Predicted Values ({scale_type.title()} Scale)')
    plt.title(f"{title} - {scale_type.title()} Scale")
    
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
               ncol=1, fontsize=10, markerscale=1.2)
    plt.grid(True, alpha=0.3)
    
    r2 = r2_score(y_true, y_pred)
    plt.text(0.95, 0.05, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             horizontalalignment='right', fontsize=12, fontweight='bold')
    
    return plt.gcf()

def visualize_model(model_key, results, target, fig_counter, model_source="individual"):
    """Visualize a single model with all plot types"""
    
    if not results or not isinstance(results, dict):
        print(f"  ! Skipping {model_key} - invalid results")
        return
    
    model_name = results.get('model_name', f'Unknown ({model_key})')
    is_joint = detect_joint_model(results)
    is_component = is_component_model(results, target)
    
    print(f"\n  Creating visualizations for {model_name}")
    print(f"    Joint: {is_joint}, Component: {is_component}, Source: {model_source}")
    
    # Learning Curves
    try:
        if (target == 'multi_output' or 
            (target == 'quote' and is_joint) or 
            is_component):
            print(f"    Skipping learning curves (multi-output/joint/component model)")
        else:
            y_train_target = None
            if target == 'quote':
                y_train_target = y_quote_train
            elif target == 'scr':
                y_train_target = y_scr_train
            elif target == 'em':
                y_train_target = y_em_train
            
            if y_train_target is not None and 'model' in results:
                fig = plot_learning_curves(results['model'], X_train, y_train_target)
                plt.title(f'Learning Curves - {model_name} ({target.upper()})')
                filename = f'figs/learning_curves_{target}_{model_key}_{fig_counter}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"    ✓ Learning curves saved: {filename}")
            else:
                print(f"    ! No training target/model for learning curves")
        
    except Exception as e:
        print(f"    ✗ Learning curves error: {str(e)}")
    
    # Residual Plots
    try:
        if target == 'multi_output' and not is_component:
            # Multi-output residuals
            predictions = results.get('predictions', {}).get('test')
            actuals = results.get('actuals', {}).get('test')
            
            if predictions is not None and actuals is not None:
                for i, sub_target in enumerate(['SCR', 'EM']):
                    try:
                        if (hasattr(predictions, 'ndim') and predictions.ndim == 2 and 
                            i < predictions.shape[1]):
                            
                            if hasattr(actuals, 'iloc'):
                                y_true = np.array(actuals.iloc[:, i])
                            else:
                                y_true = actuals[:, i]
                            y_pred = predictions[:, i]
                            
                            fig = plot_residuals(y_true, y_pred, f"{model_name} - {sub_target}")
                            filename = f'figs/residuals_{target}_{sub_target}_{model_key}_{fig_counter}.png'
                            plt.savefig(filename, dpi=300, bbox_inches='tight')
                            plt.show()
                            print(f"    ✓ Residuals ({sub_target}) saved: {filename}")
                    except Exception as sub_e:
                        print(f"    ✗ Residuals ({sub_target}) error: {str(sub_e)}")
            else:
                print(f"    ! No predictions/actuals for multi-output residuals")
        else:
            # Single-target residuals
            predictions = safe_extract_predictions(results, target, is_joint, target_scalers)
            actuals = safe_extract_actuals(results, target)
            
            pred_key, actual_key = None, None
            for key in ['test_orig', 'test']:
                if key in predictions and key in actuals:
                    pred_key, actual_key = key, key
                    break
            
            if pred_key and actual_key:
                y_pred = np.asarray(predictions[pred_key]).ravel()
                y_true = np.asarray(actuals[actual_key]).ravel()
                
                if len(y_pred) == len(y_true) and len(y_pred) > 0:
                    scale_note = 'Original' if pred_key == 'test_orig' else 'Calculated' if is_joint else 'Normalized'
                    fig = plot_residuals(y_true, y_pred, f"{model_name} - {target.upper()} ({scale_note})")
                    filename = f'figs/residuals_{target}_{model_key}_{pred_key}_{fig_counter}.png'
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.show()
                    print(f"    ✓ Residuals saved: {filename}")
                else:
                    print(f"    ! Data length mismatch: pred={len(y_pred)}, true={len(y_true)}")
            else:
                print(f"    ! No matching predictions/actuals for residuals")
            
    except Exception as e:
        print(f"    ✗ Residuals error: {str(e)}")
    
    # Predicted vs Actual Plots
    try:
        if target == 'multi_output' and not is_component:
            # Multi-output pred vs actual
            predictions = results.get('predictions', {}).get('test')
            actuals = results.get('actuals', {}).get('test')

            if predictions is not None and actuals is not None:
                for i, sub_target in enumerate(['SCR', 'EM']):
                    try:
                        if (hasattr(predictions, 'ndim') and predictions.ndim == 2 and 
                            i < predictions.shape[1]):
                            
                            if hasattr(actuals, 'iloc'):
                                y_true = np.array(actuals.iloc[:, i])
                            else:
                                y_true = actuals[:, i]
                            y_pred = predictions[:, i]
                            
                            plt.figure(figsize=(10, 8))  
                            fig = plot_predicted_vs_actual_fixed(y_true, y_pred, 
                                                        f"Predicted vs Actual - {model_name} ({sub_target})", 
                                                        "normalized", target_scalers)  
                            filename = f'figs/pred_vs_actual_{target}_{sub_target}_{model_key}_{fig_counter}.png'
                            plt.savefig(filename, dpi=300, bbox_inches='tight')
                            plt.show()
                            print(f"    ✓ Pred vs Actual ({sub_target}) saved: {filename}")
                    except Exception as sub_e:
                        print(f"    ✗ Pred vs Actual ({sub_target}) error: {str(sub_e)}")
            else:
                print(f"    ! No predictions/actuals for multi-output pred vs actual")
        else:
            # Single-target pred vs actual
            predictions = safe_extract_predictions(results, target, is_joint, target_scalers)
            actuals = safe_extract_actuals(results, target)
            
            if 'test_orig' in predictions and 'test' in actuals:
                y_pred = predictions['test_orig']
                y_true = actuals['test']
                scale_type = "calculated" if is_joint else "original"
            elif 'test' in predictions and 'test' in actuals:
                y_pred = predictions['test']
                y_true = actuals['test']
                scale_type = "normalized"
            else:
                print(f"    ! No predictions/actuals found")
                return
            
            y_pred = np.asarray(y_pred).ravel()
            y_true = np.asarray(y_true).ravel()
            
            if len(y_pred) == len(y_true) and len(y_pred) > 0:
                plt.figure(figsize=(10, 8))  
                fig = plot_predicted_vs_actual_fixed(y_true, y_pred, 
                                            f"Predicted vs Actual - {model_name} ({target.upper()})", 
                                            scale_type, target_scalers)  
                filename = f'figs/pred_vs_actual_{target}_{model_key}_{scale_type}_{fig_counter}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"    ✓ Pred vs Actual saved: {filename}")
            else:
                print(f"    ! Data length mismatch: pred={len(y_pred)}, true={len(y_true)}")
            
    except Exception as e:
        print(f"    ✗ Pred vs Actual error: {str(e)}")

# ================================================================================
# MAIN VISUALIZATION LOOP - ALL MODELS
# ================================================================================

print("="*80)
print("CREATING VISUALIZATIONS FOR ALL MODELS")
print("="*80)

fig_counter = 1

# 1. INDIVIDUAL MODELS (Quote, SCR, EM)
for target in ['quote', 'scr', 'em']:
    if target in all_results:
        print(f"\n{'='*60}")
        print(f"VISUALIZING ALL {target.upper()} MODELS")
        print(f"{'='*60}")
        
        for model_key, results in all_results[target].items():
            if results is not None:
                visualize_model(model_key, results, target, fig_counter, "individual")
                fig_counter += 1
            else:
                print(f"  ! Skipping {model_key} - no results")
        
        print(f"\n  Completed all {target.upper()} individual models")

# 2. MULTI-OUTPUT MODELS
if 'multi_output' in all_results:
    print(f"\n{'='*60}")
    print(f"VISUALIZING ALL MULTI-OUTPUT MODELS")
    print(f"{'='*60}")
    
    for model_key, results in all_results['multi_output'].items():
        if results is not None:
            visualize_model(model_key, results, 'multi_output', fig_counter, "multi_output")
            fig_counter += 1
        else:
            print(f"  ! Skipping {model_key} - no results")
    
    print(f"\n  Completed all multi-output models")

# 3. ENHANCED/JOINT MODELS (if available)
if 'enhanced_results_fixed' in globals() and enhanced_results_fixed:
    print(f"\n{'='*60}")
    print(f"VISUALIZING ALL ENHANCED/JOINT MODELS")
    print(f"{'='*60}")
    
    for model_key, results in enhanced_results_fixed.items():
        print(f"\n  Processing enhanced/joint model: {model_key}")
        if results is not None:
            # These are joint models that predict SCR+EM and calculate Quote
            visualize_model(model_key, results, 'quote', fig_counter, "enhanced_joint")
            fig_counter += 1
        else:
            print(f"  ! Skipping {model_key} - no results")
    
    print(f"\n  Completed all enhanced/joint models")

# 4. FEATURE IMPORTANCE PLOTS (unchanged)
print(f"\n{'='*60}")
print(f"CREATING FEATURE IMPORTANCE PLOTS")
print(f"{'='*60}")

for target, importance_result in feature_importance_results.items():
    try:
        if importance_result is None or 'aggregated_importance' not in importance_result:
            print(f"  ! Skipping {target.upper()} - no importance data")
            continue
            
        plt.figure(figsize=(12, 8))
        
        top_features = importance_result['aggregated_importance'].head(15)
        
        if 'Total_Importance' in top_features.columns:
            importance_col = 'Total_Importance'
        elif 'Importance' in top_features.columns:
            importance_col = 'Importance'
        else:
            print(f"  ! No recognized importance column for {target.upper()}")
            plt.close()
            continue
            
        plt.barh(range(len(top_features)), top_features[importance_col])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {target.upper()}')
        plt.gca().invert_yaxis()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = plt.gca().containers[0]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig(f'figs/feature_importance_{target}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  ✓ Feature importance for {target.upper()}")
        
    except Exception as e:
        print(f"  ✗ Feature importance error for {target}: {str(e)}")

# 5. PERFORMANCE COMPARISON HEATMAP (unchanged)
print(f"\n{'='*60}")
print(f"CREATING PERFORMANCE COMPARISON HEATMAP")
print(f"{'='*60}")

try:
    performance_data = []
    
    # Collect all model performance data
    for target in ['quote', 'scr', 'em']:
        if target in all_results:
            for model_key, results in all_results[target].items():
                if results is not None:
                    val_rmse = results['metrics'].get('val_norm_RMSE', results['metrics'].get('val_RMSE', np.nan))
                    test_rmse = results['metrics'].get('test_norm_RMSE', results['metrics'].get('test_RMSE', np.nan))
                    
                    performance_data.append({
                        'Target': target.upper(),
                        'Model': results['model_name'],
                        'Validation_RMSE': val_rmse,
                        'Test_RMSE': test_rmse
                    })
    
    if 'enhanced_results_fixed' in globals() and enhanced_results_fixed:
        for model_key, results in enhanced_results_fixed.items():
            if results is not None:
                quote_rmse = results['metrics'].get('val_Quote_RMSE', np.nan)
                quote_test_rmse = results['metrics'].get('test_Quote_RMSE', np.nan)
                
                if not np.isnan(quote_rmse):
                    performance_data.append({
                        'Target': 'QUOTE',
                        'Model': results['model_name'],
                        'Validation_RMSE': quote_rmse,
                        'Test_RMSE': quote_test_rmse
                    })
    
    if performance_data:
        perf_df = pd.DataFrame(performance_data)
        heatmap_data = perf_df.pivot_table(index='Model', columns='Target', values='Test_RMSE', aggfunc='mean')
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd_r', 
                    cbar_kws={'label': 'Normalized Test RMSE (Lower is Better)'})
        plt.title('All Models Performance Comparison - Normalized Scale')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('figs/all_models_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  ✓ Performance comparison heatmap created")
    else:
        print(f"  ! No performance data available")
    
except Exception as e:
    print(f"  ✗ Heatmap error: {str(e)}")

print("="*80)
print(f"ALL MODEL VISUALIZATIONS COMPLETED!")
print(f"Total plots created: {fig_counter - 1}")
print(f"Check the 'figs/' directory for all saved plots")
print("="*80)

# --- Output [40] ---
# [1] type: stream
# (stdout)
# ['================================================================================\n', 'CREATING VISUALIZATIONS FOR ALL MODELS\n', '================================================================================\n', '\n', '============================================================\n', 'VISUALIZING ALL QUOTE MODELS\n', '============================================================\n', '\n', '  Creating visualizations for Dummy Regressor\n', '    Joint: False, Component: False, Source: individual\n']
# [2] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [3] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_quote_dummy_1.png\n']
# [4] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [5] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_quote_dummy_test_1.png\n']
# [6] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [7] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_quote_dummy_normalized_1.png\n', '\n', '  Creating visualizations for Linear Regression (MSE)\n', '    Joint: False, Component: False, Source: individual\n']
# [8] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [9] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_quote_linear_mse_2.png\n']
# [10] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [11] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_quote_linear_mse_test_2.png\n']
# [12] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [13] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_quote_linear_mse_normalized_2.png\n', '\n', '  Creating visualizations for Quadratic (Degree 2)\n', '    Joint: False, Component: False, Source: individual\n']
# [14] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [15] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_quote_quadratic_3.png\n']
# [16] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [17] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_quote_quadratic_test_3.png\n']
# [18] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [19] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_quote_quadratic_normalized_3.png\n', '\n', '  Creating visualizations for Cubic (Degree 3)\n', '    Joint: False, Component: False, Source: individual\n']
# [20] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [21] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_quote_cubic_4.png\n']
# [22] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [23] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_quote_cubic_test_4.png\n']
# [24] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [25] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_quote_cubic_normalized_4.png\n', '\n', '  Creating visualizations for Elastic Net\n', '    Joint: False, Component: False, Source: individual\n']
# [26] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [27] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_quote_elastic_net_5.png\n']
# [28] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [29] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_quote_elastic_net_test_5.png\n']
# [30] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [31] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_quote_elastic_net_normalized_5.png\n', '\n', '  Creating visualizations for Ridge with CV\n', '    Joint: False, Component: False, Source: individual\n']
# [32] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [33] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_quote_ridge_cv_6.png\n']
# [34] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [35] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_quote_ridge_cv_test_6.png\n']
# [36] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [37] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_quote_ridge_cv_normalized_6.png\n', '\n', '  Creating visualizations for Lasso with CV\n', '    Joint: False, Component: False, Source: individual\n']
# [38] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [39] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_quote_lasso_cv_7.png\n']
# [40] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [41] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_quote_lasso_cv_test_7.png\n']
# [42] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [43] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_quote_lasso_cv_normalized_7.png\n', '\n', '  Creating visualizations for Neural Net (MLPRegressor)\n', '    Joint: False, Component: False, Source: individual\n']
# [44] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [45] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_quote_nn_mlp_8.png\n']
# [46] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [47] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_quote_nn_mlp_test_8.png\n']
# [48] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [49] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_quote_nn_mlp_normalized_8.png\n', '\n', '  Completed all QUOTE individual models\n', '\n', '============================================================\n', 'VISUALIZING ALL SCR MODELS\n', '============================================================\n', '\n', '  Creating visualizations for Dummy Regressor\n', '    Joint: False, Component: False, Source: individual\n']
# [50] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [51] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_scr_dummy_9.png\n']
# [52] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [53] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_scr_dummy_test_9.png\n']
# [54] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [55] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_scr_dummy_normalized_9.png\n', '\n', '  Creating visualizations for Linear Regression (MSE)\n', '    Joint: False, Component: False, Source: individual\n']
# [56] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [57] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_scr_linear_mse_10.png\n']
# [58] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [59] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_scr_linear_mse_test_10.png\n']
# [60] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [61] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_scr_linear_mse_normalized_10.png\n', '\n', '  Creating visualizations for Quadratic (Degree 2)\n', '    Joint: False, Component: False, Source: individual\n']
# [62] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [63] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_scr_quadratic_11.png\n']
# [64] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [65] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_scr_quadratic_test_11.png\n']
# [66] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [67] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_scr_quadratic_normalized_11.png\n', '\n', '  Creating visualizations for Cubic (Degree 3)\n', '    Joint: False, Component: False, Source: individual\n']
# [68] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [69] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_scr_cubic_12.png\n']
# [70] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [71] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_scr_cubic_test_12.png\n']
# [72] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [73] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_scr_cubic_normalized_12.png\n', '\n', '  Creating visualizations for Elastic Net\n', '    Joint: False, Component: False, Source: individual\n']
# [74] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [75] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_scr_elastic_net_13.png\n']
# [76] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [77] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_scr_elastic_net_test_13.png\n']
# [78] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [79] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_scr_elastic_net_normalized_13.png\n', '\n', '  Creating visualizations for Ridge with CV\n', '    Joint: False, Component: False, Source: individual\n']
# [80] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [81] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_scr_ridge_cv_14.png\n']
# [82] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [83] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_scr_ridge_cv_test_14.png\n']
# [84] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [85] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_scr_ridge_cv_normalized_14.png\n', '\n', '  Creating visualizations for Lasso with CV\n', '    Joint: False, Component: False, Source: individual\n']
# [86] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [87] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_scr_lasso_cv_15.png\n']
# [88] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [89] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_scr_lasso_cv_test_15.png\n']
# [90] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [91] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_scr_lasso_cv_normalized_15.png\n', '\n', '  Creating visualizations for Neural Net (MLPRegressor)\n', '    Joint: False, Component: False, Source: individual\n']
# [92] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [93] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_scr_nn_mlp_16.png\n']
# [94] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [95] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_scr_nn_mlp_test_16.png\n']
# [96] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [97] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_scr_nn_mlp_normalized_16.png\n', '\n', '  Completed all SCR individual models\n', '\n', '============================================================\n', 'VISUALIZING ALL EM MODELS\n', '============================================================\n', '\n', '  Creating visualizations for Dummy Regressor\n', '    Joint: False, Component: False, Source: individual\n']
# [98] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [99] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_em_dummy_17.png\n']
# [100] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [101] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_em_dummy_test_17.png\n']
# [102] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [103] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_em_dummy_normalized_17.png\n', '\n', '  Creating visualizations for Linear Regression (MSE)\n', '    Joint: False, Component: False, Source: individual\n']
# [104] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [105] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_em_linear_mse_18.png\n']
# [106] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [107] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_em_linear_mse_test_18.png\n']
# [108] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [109] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_em_linear_mse_normalized_18.png\n', '\n', '  Creating visualizations for Quadratic (Degree 2)\n', '    Joint: False, Component: False, Source: individual\n']
# [110] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [111] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_em_quadratic_19.png\n']
# [112] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [113] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_em_quadratic_test_19.png\n']
# [114] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [115] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_em_quadratic_normalized_19.png\n', '\n', '  Creating visualizations for Cubic (Degree 3)\n', '    Joint: False, Component: False, Source: individual\n']
# [116] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [117] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_em_cubic_20.png\n']
# [118] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [119] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_em_cubic_test_20.png\n']
# [120] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [121] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_em_cubic_normalized_20.png\n', '\n', '  Creating visualizations for Elastic Net\n', '    Joint: False, Component: False, Source: individual\n']
# [122] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [123] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_em_elastic_net_21.png\n']
# [124] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [125] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_em_elastic_net_test_21.png\n']
# [126] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [127] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_em_elastic_net_normalized_21.png\n', '\n', '  Creating visualizations for Ridge with CV\n', '    Joint: False, Component: False, Source: individual\n']
# [128] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [129] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_em_ridge_cv_22.png\n']
# [130] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [131] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_em_ridge_cv_test_22.png\n']
# [132] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [133] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_em_ridge_cv_normalized_22.png\n', '\n', '  Creating visualizations for Lasso with CV\n', '    Joint: False, Component: False, Source: individual\n']
# [134] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [135] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_em_lasso_cv_23.png\n']
# [136] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [137] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_em_lasso_cv_test_23.png\n']
# [138] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [139] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_em_lasso_cv_normalized_23.png\n', '\n', '  Creating visualizations for Neural Net (MLPRegressor)\n', '    Joint: False, Component: False, Source: individual\n']
# [140] type: display_data
# ['<Figure size 1000x600 with 1 Axes>']
# [141] type: stream
# (stdout)
# ['    ✓ Learning curves saved: figs/learning_curves_em_nn_mlp_24.png\n']
# [142] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [143] type: stream
# (stdout)
# ['    ✓ Residuals saved: figs/residuals_em_nn_mlp_test_24.png\n']
# [144] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [145] type: stream
# (stdout)
# ['    ✓ Pred vs Actual saved: figs/pred_vs_actual_em_nn_mlp_normalized_24.png\n', '\n', '  Completed all EM individual models\n', '\n', '============================================================\n', 'VISUALIZING ALL MULTI-OUTPUT MODELS\n', '============================================================\n', '\n', '  Creating visualizations for Dummy Regressor (Multi-Output)\n', '    Joint: True, Component: False, Source: multi_output\n', '    Skipping learning curves (multi-output/joint/component model)\n']
# [146] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [147] type: stream
# (stdout)
# ['    ✓ Residuals (SCR) saved: figs/residuals_multi_output_SCR_dummy_25.png\n']
# [148] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [149] type: stream
# (stdout)
# ['    ✓ Residuals (EM) saved: figs/residuals_multi_output_EM_dummy_25.png\n']
# [150] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [151] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (SCR) saved: figs/pred_vs_actual_multi_output_SCR_dummy_25.png\n']
# [152] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [153] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (EM) saved: figs/pred_vs_actual_multi_output_EM_dummy_25.png\n', '\n', '  Creating visualizations for Linear Regression (MSE) (Multi-Output)\n', '    Joint: True, Component: False, Source: multi_output\n', '    Skipping learning curves (multi-output/joint/component model)\n']
# [154] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [155] type: stream
# (stdout)
# ['    ✓ Residuals (SCR) saved: figs/residuals_multi_output_SCR_linear_mse_26.png\n']
# [156] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [157] type: stream
# (stdout)
# ['    ✓ Residuals (EM) saved: figs/residuals_multi_output_EM_linear_mse_26.png\n']
# [158] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [159] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (SCR) saved: figs/pred_vs_actual_multi_output_SCR_linear_mse_26.png\n']
# [160] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [161] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (EM) saved: figs/pred_vs_actual_multi_output_EM_linear_mse_26.png\n', '\n', '  Creating visualizations for Quadratic (Degree 2) (Multi-Output)\n', '    Joint: True, Component: False, Source: multi_output\n', '    Skipping learning curves (multi-output/joint/component model)\n']
# [162] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [163] type: stream
# (stdout)
# ['    ✓ Residuals (SCR) saved: figs/residuals_multi_output_SCR_quadratic_27.png\n']
# [164] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [165] type: stream
# (stdout)
# ['    ✓ Residuals (EM) saved: figs/residuals_multi_output_EM_quadratic_27.png\n']
# [166] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [167] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (SCR) saved: figs/pred_vs_actual_multi_output_SCR_quadratic_27.png\n']
# [168] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [169] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (EM) saved: figs/pred_vs_actual_multi_output_EM_quadratic_27.png\n', '\n', '  Creating visualizations for Cubic (Degree 3) (Multi-Output)\n', '    Joint: True, Component: False, Source: multi_output\n', '    Skipping learning curves (multi-output/joint/component model)\n']
# [170] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [171] type: stream
# (stdout)
# ['    ✓ Residuals (SCR) saved: figs/residuals_multi_output_SCR_cubic_28.png\n']
# [172] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [173] type: stream
# (stdout)
# ['    ✓ Residuals (EM) saved: figs/residuals_multi_output_EM_cubic_28.png\n']
# [174] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [175] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (SCR) saved: figs/pred_vs_actual_multi_output_SCR_cubic_28.png\n']
# [176] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [177] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (EM) saved: figs/pred_vs_actual_multi_output_EM_cubic_28.png\n', '\n', '  Creating visualizations for Elastic Net (Multi-Output)\n', '    Joint: True, Component: False, Source: multi_output\n', '    Skipping learning curves (multi-output/joint/component model)\n']
# [178] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [179] type: stream
# (stdout)
# ['    ✓ Residuals (SCR) saved: figs/residuals_multi_output_SCR_elastic_net_29.png\n']
# [180] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [181] type: stream
# (stdout)
# ['    ✓ Residuals (EM) saved: figs/residuals_multi_output_EM_elastic_net_29.png\n']
# [182] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [183] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (SCR) saved: figs/pred_vs_actual_multi_output_SCR_elastic_net_29.png\n']
# [184] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [185] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (EM) saved: figs/pred_vs_actual_multi_output_EM_elastic_net_29.png\n', '\n', '  Creating visualizations for Ridge with CV (Multi-Output)\n', '    Joint: True, Component: False, Source: multi_output\n', '    Skipping learning curves (multi-output/joint/component model)\n']
# [186] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [187] type: stream
# (stdout)
# ['    ✓ Residuals (SCR) saved: figs/residuals_multi_output_SCR_ridge_cv_30.png\n']
# [188] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [189] type: stream
# (stdout)
# ['    ✓ Residuals (EM) saved: figs/residuals_multi_output_EM_ridge_cv_30.png\n']
# [190] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [191] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (SCR) saved: figs/pred_vs_actual_multi_output_SCR_ridge_cv_30.png\n']
# [192] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [193] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (EM) saved: figs/pred_vs_actual_multi_output_EM_ridge_cv_30.png\n', '\n', '  Creating visualizations for Lasso with CV (Multi-Output)\n', '    Joint: True, Component: False, Source: multi_output\n', '    Skipping learning curves (multi-output/joint/component model)\n']
# [194] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [195] type: stream
# (stdout)
# ['    ✓ Residuals (SCR) saved: figs/residuals_multi_output_SCR_lasso_cv_31.png\n']
# [196] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [197] type: stream
# (stdout)
# ['    ✓ Residuals (EM) saved: figs/residuals_multi_output_EM_lasso_cv_31.png\n']
# [198] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [199] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (SCR) saved: figs/pred_vs_actual_multi_output_SCR_lasso_cv_31.png\n']
# [200] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [201] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (EM) saved: figs/pred_vs_actual_multi_output_EM_lasso_cv_31.png\n', '\n', '  Creating visualizations for Neural Net (MLPRegressor) (Multi-Output)\n', '    Joint: True, Component: False, Source: multi_output\n', '    Skipping learning curves (multi-output/joint/component model)\n']
# [202] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [203] type: stream
# (stdout)
# ['    ✓ Residuals (SCR) saved: figs/residuals_multi_output_SCR_nn_mlp_32.png\n']
# [204] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [205] type: stream
# (stdout)
# ['    ✓ Residuals (EM) saved: figs/residuals_multi_output_EM_nn_mlp_32.png\n']
# [206] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [207] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (SCR) saved: figs/pred_vs_actual_multi_output_SCR_nn_mlp_32.png\n']
# [208] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [209] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (EM) saved: figs/pred_vs_actual_multi_output_EM_nn_mlp_32.png\n', '\n', '  Creating visualizations for Neural Net (PyTorch MLP Dropout Multi) (Multi-Output)\n', '    Joint: True, Component: False, Source: multi_output\n', '    Skipping learning curves (multi-output/joint/component model)\n']
# [210] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [211] type: stream
# (stdout)
# ['    ✓ Residuals (SCR) saved: figs/residuals_multi_output_SCR_nn_torch_dropout_multi_33.png\n']
# [212] type: display_data
# ['<Figure size 1500x600 with 3 Axes>']
# [213] type: stream
# (stdout)
# ['    ✓ Residuals (EM) saved: figs/residuals_multi_output_EM_nn_torch_dropout_multi_33.png\n']
# [214] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [215] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (SCR) saved: figs/pred_vs_actual_multi_output_SCR_nn_torch_dropout_multi_33.png\n']
# [216] type: display_data
# ['<Figure size 1000x800 with 1 Axes>']
# [217] type: stream
# (stdout)
# ['    ✓ Pred vs Actual (EM) saved: figs/pred_vs_actual_multi_output_EM_nn_torch_dropout_multi_33.png\n', '\n', '  Completed all multi-output models\n', '\n', '============================================================\n', 'VISUALIZING ALL ENHANCED/JOINT MODELS\n', '============================================================\n', '\n', '  Processing enhanced/joint model: linear_multi\n', '\n', '  Creating visualizations for Linear Regression (Multi-Output) (with Quote)\n', '    Joint: True, Component: False, Source: enhanced_joint\n', '    Skipping learning curves (multi-output/joint/component model)\n', '    ! Data length mismatch: pred=2046, true=4092\n', '    ! Data length mismatch: pred=2046, true=4092\n', '\n', '  Processing enhanced/joint model: ridge_multi\n', '\n', '  Creating visualizations for Ridge Regression (Multi-Output) (with Quote)\n', '    Joint: True, Component: False, Source: enhanced_joint\n', '    Skipping learning curves (multi-output/joint/component model)\n', '    ! Data length mismatch: pred=2046, true=4092\n', '    ! Data length mismatch: pred=2046, true=4092\n', '\n', '  Processing enhanced/joint model: poly2_multi\n', '\n', '  Creating visualizations for Quadratic (Multi-Output) (with Quote)\n', '    Joint: True, Component: False, Source: enhanced_joint\n', '    Skipping learning curves (multi-output/joint/component model)\n', '    ! Data length mismatch: pred=2046, true=4092\n', '    ! Data length mismatch: pred=2046, true=4092\n', '\n', '  Processing enhanced/joint model: random_forest_multi\n', '\n', '  Creating visualizations for Random Forest (Multi-Output) (with Quote)\n', '    Joint: True, Component: False, Source: enhanced_joint\n', '    Skipping learning curves (multi-output/joint/component model)\n', '    ! Data length mismatch: pred=2046, true=4092\n', '    ! Data length mismatch: pred=2046, true=4092\n', '\n', '  Processing enhanced/joint model: xgboost_basic\n', '\n', '  Creating visualizations for XGBoost Basic (Multi-Output) (with Quote)\n', '    Joint: True, Component: False, Source: enhanced_joint\n', '    Skipping learning curves (multi-output/joint/component model)\n', '    ! Data length mismatch: pred=2046, true=4092\n', '    ! Data length mismatch: pred=2046, true=4092\n', '\n', '  Processing enhanced/joint model: xgboost_tuned\n', '\n', '  Creating visualizations for XGBoost Tuned (Multi-Output) (with Quote)\n', '    Joint: True, Component: False, Source: enhanced_joint\n', '    Skipping learning curves (multi-output/joint/component model)\n', '    ! Data length mismatch: pred=2046, true=4092\n', '    ! Data length mismatch: pred=2046, true=4092\n', '\n', '  Processing enhanced/joint model: mlp_reg_multi\n', '\n', '  Creating visualizations for MLPRegressor (Multi-Output) (with Quote)\n', '    Joint: True, Component: False, Source: enhanced_joint\n', '    Skipping learning curves (multi-output/joint/component model)\n', '    ! Data length mismatch: pred=2046, true=4092\n', '    ! Data length mismatch: pred=2046, true=4092\n', '\n', '  Processing enhanced/joint model: torch_nn_multi\n', '\n', '  Creating visualizations for PyTorch Neural Net (Multi-Output) (with Quote)\n', '    Joint: True, Component: False, Source: enhanced_joint\n', '    Skipping learning curves (multi-output/joint/component model)\n', '    ! Data length mismatch: pred=2046, true=4092\n', '    ! Data length mismatch: pred=2046, true=4092\n', '\n', '  Processing enhanced/joint model: torch_nn_multi_tuned\n', '\n', '  Creating visualizations for PyTorch Neural Net Tuned (Multi-Output) (with Quote)\n', '    Joint: True, Component: False, Source: enhanced_joint\n', '    Skipping learning curves (multi-output/joint/component model)\n', '    ! Data length mismatch: pred=2046, true=4092\n', '    ! Data length mismatch: pred=2046, true=4092\n', '\n', '  Processing enhanced/joint model: mlp_multi_tuned\n', '\n', '  Creating visualizations for MLPRegressor Tuned (Multi-Output) (with Quote)\n', '    Joint: True, Component: False, Source: enhanced_joint\n', '    Skipping learning curves (multi-output/joint/component model)\n', '    ! Data length mismatch: pred=2046, true=4092\n', '    ! Data length mismatch: pred=2046, true=4092\n', '\n', '  Completed all enhanced/joint models\n', '\n', '============================================================\n', 'CREATING FEATURE IMPORTANCE PLOTS\n', '============================================================\n']
# [218] type: display_data
# ['<Figure size 1200x800 with 1 Axes>']
# [219] type: stream
# (stdout)
# ['  ✓ Feature importance for QUOTE\n']
# [220] type: display_data
# ['<Figure size 1200x800 with 1 Axes>']
# [221] type: stream
# (stdout)
# ['  ✓ Feature importance for SCR\n']
# [222] type: display_data
# ['<Figure size 1200x800 with 1 Axes>']
# [223] type: stream
# (stdout)
# ['  ✓ Feature importance for EM\n']
# [224] type: display_data
# ['<Figure size 1200x800 with 1 Axes>']
# [225] type: stream
# (stdout)
# ['  ✓ Feature importance for MULTI_OUTPUT\n', '\n', '============================================================\n', 'CREATING PERFORMANCE COMPARISON HEATMAP\n', '============================================================\n']
# [226] type: display_data
# ['<Figure size 1400x1000 with 2 Axes>']
# [227] type: stream
# (stdout)
# ['  ✓ Performance comparison heatmap created\n', '================================================================================\n', 'ALL MODEL VISUALIZATIONS COMPLETED!\n', 'Total plots created: 43\n', "Check the 'figs/' directory for all saved plots\n", '================================================================================\n']
# --- End Output ---

# In[41]:
# Create artifacts directory
artifacts_dir = Path('solvency_ml_artifacts')
artifacts_dir.mkdir(exist_ok=True)

# Save best models
for target, best_info in best_models.items():
    model_path = artifacts_dir / f'best_{target}_pipeline.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_info['results']['model'], f)
    print(f"Saved {target} model to {model_path}")

# Save target scalers for future use
scalers_path = artifacts_dir / 'target_scalers.pkl'
with open(scalers_path, 'wb') as f:
    pickle.dump(target_scalers, f)
print(f"Saved target scalers to {scalers_path}")

# Save comprehensive results summary with enhanced multi-output support
results_summary = {
    'timestamp': datetime.now().isoformat(),
    'data_shape': df.shape,
    'feature_columns': feature_cols,
    'target_columns': target_cols,
    'regulatory_distribution': {
        'insolvent': int((df['Quote'] < 0).sum()),
        'undercapitalized': int((df['Quote'] < 1).sum()),
        'well_capitalized': int((df['Quote'] > 2).sum()),
        'total': len(df)
    },
    'train_test_split': {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'random_state': RANDOM_STATE
    },
    'best_models': {},
    'model_comparisons': {},
    'stability_analysis': {},
    'methodology_notes': {
        'joint_modeling': 'XGBoost joint models predict SCR and EM simultaneously, then calculate Quote = EM/SCR',
        'normalization': 'All targets normalized to [-1,1] scale using MinMaxScaler fitted on training data only',
        'evaluation': 'Performance compared using normalized RMSE for fair comparison across targets'
    }
}

# Add best model results with enhanced information
for target, best_info in best_models.items():
    #nPass the entire results dictionary to detect_joint_model
    is_joint = detect_joint_model(best_info['results'])
    
    model_info = {
        'model_name': best_info['results']['model_name'],
        'model_key': best_info['key'],
        'is_joint_model': is_joint,
        'validation_rmse': float(best_info['val_rmse_norm']),
        'test_metrics': {}
    }
    
    # Extract test metrics safely
    for k, v in best_info['results']['metrics'].items():
        if k.startswith('test_') and isinstance(v, (int, float, np.number)):
            model_info['test_metrics'][k] = float(v)
    
    results_summary['best_models'][target] = model_info

# Add feature importance results with multi-output support and error handling
for target, importance_result in feature_importance_results.items():
    if importance_result is not None and target in results_summary['best_models']:
        try:
            if 'aggregated_importance' in importance_result:
                top_features = importance_result['aggregated_importance'].head(10)
                
                # Determine importance column
                if 'Total_Importance' in top_features.columns:
                    importance_col = 'Total_Importance'
                elif 'Importance' in top_features.columns:
                    importance_col = 'Importance'
                else:
                    print(f"Warning: No recognized importance column for {target}")
                    continue
                    
                # Add top features
                results_summary['best_models'][target]['top_features'] = [
                    {'feature': row['Feature'], 'importance': float(row[importance_col])}
                    for _, row in top_features.iterrows()
                ]
                
                # Add multi-output specific importance if available
                if ('scr_coefficients' in importance_result and 'em_coefficients' in importance_result) or \
                   ('scr_importance' in importance_result and 'em_importance' in importance_result):
                    results_summary['best_models'][target]['joint_model_details'] = {
                        'multi_output_analysis': True,
                        'components_analyzed': ['SCR', 'EM'],
                        'combination_method': 'Average of component importances'
                    }
            else:
                print(f"Warning: No aggregated importance available for {target}")
        except Exception as e:
            print(f"Warning: Could not process feature importance for {target}: {str(e)}")
    else:
        if target in results_summary['best_models']:
            results_summary['best_models'][target]['feature_importance_note'] = 'Feature importance analysis failed or not available'

# Add stability results with multi-output support
for target, stability_result in stability_results.items():
    if stability_result is not None:
        stability_info = {
            'overall_rmse': float(stability_result['overall_rmse']),
            'overall_r2': float(stability_result['overall_r2']),
            'overall_mae': float(stability_result['overall_mae']),
            'residual_std': float(stability_result['residual_std']),
            'is_joint_model': stability_result.get('is_joint_model', False)
        }
        
        # Add range analysis
        if 'range_analysis' in stability_result:
            stability_info['performance_by_range'] = {}
            for range_name, range_data in stability_result['range_analysis'].items():
                stability_info['performance_by_range'][range_name] = {
                    'count': int(range_data['count']),
                    'percentage': float(range_data['percentage']),
                    'rmse': float(range_data['rmse']),
                    'mae': float(range_data['mae']),
                    'r2': float(range_data['r2'])
                }
        
        results_summary['stability_analysis'][target] = stability_info

# Add comparison tables
try:
    results_summary['model_comparisons'] = {
        'individual_models': {
            'quote': quote_comparison.to_dict('records') if 'quote_comparison' in globals() and not quote_comparison.empty else [],
            'scr': scr_comparison.to_dict('records') if 'scr_comparison' in globals() and not scr_comparison.empty else [],
            'em': em_comparison.to_dict('records') if 'em_comparison' in globals() and not em_comparison.empty else []
        },
        'multi_output': multi_comparison.to_dict('records') if 'multi_comparison' in globals() and not multi_comparison.empty else []
    }
    
    # Add joint model comparison if available
    if 'joint_comparison_fixed' in globals() and not joint_comparison_fixed.empty:
        results_summary['model_comparisons']['joint_models'] = joint_comparison_fixed.to_dict('records')
        
except Exception as e:
    print(f"Warning: Could not save all comparison tables: {str(e)}")
    results_summary['model_comparisons'] = {'note': 'Some comparison data unavailable due to processing error'}

# Add comprehensive model summary
results_summary['model_summary'] = {
    'total_models_trained': len([m for models in all_results.values() for m in models.values() if m is not None]),
    'best_quote_approach': 'Joint XGBoost (SCR+EM→Quote)' if 'quote' in best_models and detect_joint_model(best_models['quote']['results']) else 'Direct prediction',
    'key_findings': [
        f"Joint XGBoost modeling achieved {((0.1108 - 0.0807) / 0.1108 * 100):.1f}% improvement over direct Quote prediction",
        "ZSK1 (interest rate factor) is the most important feature across all models",
        "XGBoost models showed superior performance for multi-output regression tasks",
        "Polynomial features (degree 2) were most effective for individual target prediction"
    ]
}

# Save results summary
summary_path = artifacts_dir / 'comprehensive_results_summary.json'
with open(summary_path, 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)
print(f"Saved comprehensive results summary to {summary_path}")

# Save feature importance results separately with multi-output support and error handling
for target, importance_result in feature_importance_results.items():
    try:
        if importance_result is not None and 'aggregated_importance' in importance_result:
            # Main aggregated importance
            importance_path = artifacts_dir / f'feature_importance_{target}.csv'
            importance_result['aggregated_importance'].to_csv(importance_path, index=False)
            print(f"Saved feature importance for {target} to {importance_path}")
            
            # Save detailed importance if available (for XGBoost joint models)
            if 'scr_importance' in importance_result and 'em_importance' in importance_result:
                detailed_importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'SCR_Importance': importance_result['scr_importance'],
                    'EM_Importance': importance_result['em_importance'],
                    'Combined_Importance': importance_result['feature_importances']
                })
                detailed_path = artifacts_dir / f'detailed_feature_importance_{target}.csv'
                detailed_importance.to_csv(detailed_path, index=False)
                print(f"Saved detailed joint model feature importance for {target} to {detailed_path}")
                
            # Save detailed importance for MultiOutput linear models
            elif 'scr_coefficients' in importance_result and 'em_coefficients' in importance_result:
                detailed_importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'SCR_Coefficients': importance_result['scr_coefficients'],
                    'EM_Coefficients': importance_result['em_coefficients'],
                    'Combined_Importance': importance_result['coefficients']
                })
                detailed_path = artifacts_dir / f'detailed_coefficients_{target}.csv'
                detailed_importance.to_csv(detailed_path, index=False)
                print(f"Saved detailed multi-output coefficients for {target} to {detailed_path}")
        else:
            print(f"Warning: Could not save feature importance for {target} - no valid importance data")
    except Exception as e:
        print(f"Warning: Error saving feature importance for {target}: {str(e)}")

# Save stability analysis results
if 'stability_results' in globals():
    stability_path = artifacts_dir / 'stability_analysis_results.json'
    stable_results = {}
    for target, result in stability_results.items():
        if result is not None:
            # Convert numpy types to Python types for JSON serialization
            stable_results[target] = {}
            for k, v in result.items():
                if isinstance(v, dict):
                    stable_results[target][k] = {sk: float(sv) if isinstance(sv, (np.number, int, float)) else sv 
                                               for sk, sv in v.items() if isinstance(sv, (dict, int, float, np.number, str, bool))}
                elif isinstance(v, (np.number, int, float)):
                    stable_results[target][k] = float(v)
                else:
                    stable_results[target][k] = v

    with open(stability_path, 'w') as f:
        json.dump(stable_results, f, indent=2, default=str)
    print(f"Saved stability analysis results to {stability_path}")
else:
    print("Warning: stability_results not found, skipping stability analysis export")

# Create a comprehensive model performance report
performance_report = []
for target, best_info in best_models.items():
    model_info = {
        'Target': target.upper(),
        'Model': best_info['results']['model_name'],
        'Type': 'Joint' if detect_joint_model(best_info['results']) else 'Direct',  # FIXED: Pass results dict
        'Val_RMSE': float(best_info['val_rmse_norm']),
        'Test_RMSE': float(best_info['results']['metrics'].get('test_norm_RMSE', 
                                                              best_info['results']['metrics'].get('test_Quote_RMSE', np.nan))),
        'Test_R2': float(best_info['results']['metrics'].get('test_norm_R2',
                                                           best_info['results']['metrics'].get('test_Quote_R2', np.nan)))
    }
    
    # Add stability metrics if available
    if 'stability_results' in globals() and target in stability_results and stability_results[target] is not None:
        model_info.update({
            'Stability_RMSE': float(stability_results[target]['overall_rmse']),
            'Residual_Std': float(stability_results[target]['residual_std'])
        })
    
    performance_report.append(model_info)

performance_df = pd.DataFrame(performance_report)
performance_df_path = artifacts_dir / 'model_performance_summary.csv'
performance_df.to_csv(performance_df_path, index=False)
print(f"Saved model performance summary to {performance_df_path}")

# Print final summary
print(f"\n" + "="*80)
print("ARTIFACTS SAVED SUCCESSFULLY")
print("="*80)
print(f"Location: {artifacts_dir.absolute()}")
print(f"Files created:")
print(f"  • {len(best_models)} model pipelines (*.pkl)")
print(f"  • 1 target scalers file")
print(f"  • {len(feature_importance_results)} feature importance files")
print(f"  • 1 comprehensive results summary (JSON)")
print(f"  • 1 stability analysis results (JSON)")
print(f"  • 1 performance summary (CSV)")
if 'feature_importance_results' in globals() and 'scr_importance' in feature_importance_results.get('quote', {}):
    print(f"  • Enhanced joint model importance files")

print(f"\nKey Results:")
for target, best_info in best_models.items():
    rmse = best_info['val_rmse_norm']
    model_type = 'Joint' if detect_joint_model(best_info['results']) else 'Direct'  # FIXED: Pass results dict
    print(f"  {target.upper()}: {best_info['results']['model_name']} ({model_type}) - RMSE: {rmse:.4f}")

print(f"\nAll artifacts ready for production deployment or further analysis!")
print("="*80)

# --- Output [41] ---
# [1] type: stream
# (stdout)
# ['Saved quote model to solvency_ml_artifacts/best_quote_pipeline.pkl\n', 'Saved scr model to solvency_ml_artifacts/best_scr_pipeline.pkl\n', 'Saved em model to solvency_ml_artifacts/best_em_pipeline.pkl\n', 'Saved multi_output model to solvency_ml_artifacts/best_multi_output_pipeline.pkl\n', 'Saved target scalers to solvency_ml_artifacts/target_scalers.pkl\n', 'Saved comprehensive results summary to solvency_ml_artifacts/comprehensive_results_summary.json\n', 'Saved feature importance for quote to solvency_ml_artifacts/feature_importance_quote.csv\n', 'Saved detailed joint model feature importance for quote to solvency_ml_artifacts/detailed_feature_importance_quote.csv\n', 'Saved feature importance for scr to solvency_ml_artifacts/feature_importance_scr.csv\n', 'Saved detailed joint model feature importance for scr to solvency_ml_artifacts/detailed_feature_importance_scr.csv\n', 'Saved feature importance for em to solvency_ml_artifacts/feature_importance_em.csv\n', 'Saved detailed joint model feature importance for em to solvency_ml_artifacts/detailed_feature_importance_em.csv\n', 'Saved feature importance for multi_output to solvency_ml_artifacts/feature_importance_multi_output.csv\n', 'Saved detailed joint model feature importance for multi_output to solvency_ml_artifacts/detailed_feature_importance_multi_output.csv\n', 'Saved stability analysis results to solvency_ml_artifacts/stability_analysis_results.json\n', 'Saved model performance summary to solvency_ml_artifacts/model_performance_summary.csv\n', '\n', '================================================================================\n', 'ARTIFACTS SAVED SUCCESSFULLY\n', '================================================================================\n', 'Location: /Users/mahbod/swiss uni /Solvency-ETH-proj/solvency_ml_artifacts\n', 'Files created:\n', '  • 4 model pipelines (*.pkl)\n', '  • 1 target scalers file\n', '  • 4 feature importance files\n', '  • 1 comprehensive results summary (JSON)\n', '  • 1 stability analysis results (JSON)\n', '  • 1 performance summary (CSV)\n', '  • Enhanced joint model importance files\n', '\n', 'Key Results:\n', '  QUOTE: XGBoost Tuned (Multi-Output) (with Quote) (Joint) - RMSE: 0.0807\n', '  SCR: XGBoost Tuned (Multi-Output) (with Quote) (Enhanced SCR Component) (Joint) - RMSE: 0.0764\n', '  EM: XGBoost Tuned (Multi-Output) (with Quote) (Enhanced EM Component) (Joint) - RMSE: 0.0555\n', '  MULTI_OUTPUT: XGBoost Tuned (Multi-Output) (with Quote) (Joint) - RMSE: 0.0668\n', '\n', 'All artifacts ready for production deployment or further analysis!\n', '================================================================================\n']
# --- End Output ---
