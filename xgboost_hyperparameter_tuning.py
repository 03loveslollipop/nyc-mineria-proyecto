"""
XGBoost Hyperparameter Tuning with Grid Search and Cross-Validation
Performs exhaustive grid search with 5-fold CV on ~400 XGBoost configurations
Uses CUDA acceleration for GPU-accelerated training
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("XGBOOST HYPERPARAMETER TUNING WITH GRID SEARCH AND 5-FOLD CV")
print("="*80)

# ============================================================================
# CHECK CUDA AVAILABILITY
# ============================================================================

# Check if CUDA is available
import torch
if torch.cuda.is_available():
    device = 'cuda'
    print(f"\n[OK] CUDA available - GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("\n[WARNING] CUDA not available - using CPU")

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("\nLoading training data (70%)...")
train_df = pd.read_parquet('train_70.parquet')
y_train_full = train_df['trip_duration'].values
X_train_full = train_df.drop('trip_duration', axis=1)

print(f"Training data shape: {X_train_full.shape}")
print(f"Target shape: {y_train_full.shape}")

# Categorical columns
categorical_cols = ['payment_type', 'dropoff_borough', 'pickup_borough']

# ============================================================================
# DATA PREPROCESSING FOR XGBOOST
# ============================================================================

print("\nPreparing data for XGBoost...")
X_train_prep = X_train_full.copy()

# Convert from pyarrow backend to numpy backend
numeric_cols = [col for col in X_train_prep.columns if col not in categorical_cols]

# Convert numeric columns to float64 (numpy dtype)
for col in numeric_cols:
    X_train_prep[col] = X_train_prep[col].astype('float64')

# Convert categorical columns from pyarrow string to python str wrapped in category
for col in categorical_cols:
    X_train_prep[col] = X_train_prep[col].astype(str).astype('category')

print(f"XGBoost data shape: {X_train_prep.shape}")
print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {categorical_cols}")

# ============================================================================
# GRID SEARCH SETUP
# ============================================================================

print("\n" + "="*80)
print("GRID SEARCH CONFIGURATION")
print("="*80)

# Define parameter grid (~400 configurations)
param_grid = {
    'n_estimators': [50, 100, 200, 300],  # 4 options
    'max_depth': [3, 5, 7, 9],  # 4 options
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 4 options
    'subsample': [0.7, 0.8, 0.9, 1.0],  # 4 options
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  # 4 options
    'min_child_weight': [1, 3, 5],  # 3 options
    'gamma': [0, 0.1, 0.2],  # 3 options
    'reg_alpha': [0, 0.01, 0.1],  # 3 options (L1 regularization)
    'reg_lambda': [1, 1.5, 2],  # 3 options (L2 regularization)
    'tree_method': ['hist'],  # GPU-compatible
    'device': [device],  # Use CUDA or CPU
    'enable_categorical': [True],
    'random_state': [42]
}

# Calculate total combinations
# 4*4*4*4*4 = 1024 base combinations
# But we'll use a smaller grid to get ~400
# Let's reduce some dimensions
param_grid_reduced = {
    'n_estimators': [50, 100, 200],  # 3 options
    'max_depth': [3, 5, 7, 9],  # 4 options
    'learning_rate': [0.01, 0.05, 0.1],  # 3 options
    'subsample': [0.7, 0.8, 1.0],  # 3 options
    'colsample_bytree': [0.7, 0.8, 1.0],  # 3 options
    'min_child_weight': [1, 3, 5],  # 3 options
    'gamma': [0, 0.1, 0.2],  # 3 options
    'reg_alpha': [0, 0.1],  # 2 options
    'reg_lambda': [1, 2],  # 2 options
    'tree_method': ['hist'],
    'device': [device],
    'enable_categorical': [True],
    'random_state': [42]
}

total_configs = (3 * 4 * 3 * 3 * 3 * 3 * 3 * 2 * 2)  # = 11,664 - too many!

# Let's create a more balanced grid (~400 configs)
param_grid_final = {
    'n_estimators': [50, 100, 200, 300],  # 4 options
    'max_depth': [3, 5, 7, 9],  # 4 options
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 4 options
    'subsample': [0.8, 1.0],  # 2 options
    'colsample_bytree': [0.8, 1.0],  # 2 options
    'min_child_weight': [1, 3],  # 2 options
    'gamma': [0, 0.1],  # 2 options
    'reg_alpha': [0, 0.1],  # 2 options
    'reg_lambda': [1],  # 1 option
    'tree_method': ['hist'],
    'device': [device],
    'enable_categorical': [True],
    'random_state': [42]
}

total_configs = 4 * 4 * 4 * 2 * 2 * 2 * 2 * 2  # = 2048

# To get closer to 400, let's use a different strategy - sample parameters
# Actually, let's stick with a grid that gives us a reasonable number
param_grid_balanced = {
    'n_estimators': [50, 100, 200],  # 3 options
    'max_depth': [4, 6, 8],  # 3 options
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 4 options
    'subsample': [0.7, 0.85, 1.0],  # 3 options
    'colsample_bytree': [0.7, 0.85, 1.0],  # 3 options
    'min_child_weight': [1, 3],  # 2 options
    'gamma': [0, 0.1],  # 2 options
    'reg_alpha': [0, 0.1],  # 2 options
    'reg_lambda': [1],  # 1 option
    'tree_method': ['hist'],
    'device': [device],
    'enable_categorical': [True],
    'random_state': [42]
}

total_configs = 3 * 3 * 4 * 3 * 3 * 2 * 2 * 2  # = 1296

# Let's go with this one that gives us 432 configs
param_grid_optimal = {
    'n_estimators': [50, 100, 200],  # 3 options
    'max_depth': [3, 5, 7, 9],  # 4 options
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 4 options
    'subsample': [0.8, 1.0],  # 2 options
    'colsample_bytree': [0.8, 1.0],  # 2 options
    'min_child_weight': [1, 3],  # 2 options
    'gamma': [0, 0.1],  # 2 options
    'reg_alpha': [0, 0.1],  # 2 options
    'reg_lambda': [1],  # 1 option
    'tree_method': ['hist'],
    'device': [device],
    'enable_categorical': [True],
    'random_state': [42]
}

total_configs = 3 * 4 * 4 * 2 * 2 * 2 * 2 * 2  # = 384 configurations

print(f"\nParameter grid:")
print(f"  n_estimators: {param_grid_optimal['n_estimators']}")
print(f"  max_depth: {param_grid_optimal['max_depth']}")
print(f"  learning_rate: {param_grid_optimal['learning_rate']}")
print(f"  subsample: {param_grid_optimal['subsample']}")
print(f"  colsample_bytree: {param_grid_optimal['colsample_bytree']}")
print(f"  min_child_weight: {param_grid_optimal['min_child_weight']}")
print(f"  gamma: {param_grid_optimal['gamma']}")
print(f"  reg_alpha: {param_grid_optimal['reg_alpha']}")
print(f"  reg_lambda: {param_grid_optimal['reg_lambda']}")
print(f"  tree_method: {param_grid_optimal['tree_method']}")
print(f"  device: {param_grid_optimal['device']}")
print(f"  enable_categorical: {param_grid_optimal['enable_categorical']}")
print(f"\nTotal configurations: {total_configs}")
print(f"With 5-fold CV: {total_configs * 5} total fits")
print("="*80)

# ============================================================================
# RUN GRID SEARCH
# ============================================================================

print("\nInitializing GridSearchCV with 5-fold cross-validation...")

# Custom scoring
scoring = {
    'neg_mse': 'neg_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error',
    'r2': 'r2'
}

# Initialize XGBoost regressor
xgb_regressor = xgb.XGBRegressor(
    objective='reg:squarederror',
    verbosity=0  # Reduce output noise
)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_regressor,
    param_grid=param_grid_optimal,
    scoring=scoring,
    refit='neg_mse',  # Refit with best MSE
    cv=5,
    n_jobs=-1,  # Use all CPU cores for parallel CV folds
    verbose=2,
    return_train_score=True,
    error_score='raise'
)

print("\nStarting grid search with GPU acceleration...")
print("This may take a while depending on dataset size...")
print("="*80)

start_time = time.time()

try:
    # Note: do not pass `categorical_feature` to sklearn's GridSearchCV.fit because
    # XGBModel.fit (sklearn API) does not accept that keyword. Instead, rely on
    # pandas categorical dtypes and XGBoost's `enable_categorical` parameter
    # (set in the param grid) so the estimator handles categorical features.
    grid_search.fit(
        X_train_prep, y_train_full
    )
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(f"[OK] Grid search completed in {total_time/60:.2f} minutes")
    print(f"Average time per configuration: {total_time/total_configs:.2f} seconds")
    print("="*80)
    
except Exception as e:
    print(f"\n[ERROR] Grid search failed: {str(e)}")
    raise

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

print("\n" + "="*80)
print("GRID SEARCH RESULTS")
print("="*80)

# Get results
results_df = pd.DataFrame(grid_search.cv_results_)

# Calculate RMSE from MSE
results_df['mean_test_rmse'] = np.sqrt(-results_df['mean_test_neg_mse'])
results_df['mean_test_mae'] = -results_df['mean_test_neg_mae']
results_df['std_test_rmse'] = results_df['std_test_neg_mse'] / (2 * results_df['mean_test_rmse'])

# Sort by RMSE
results_df = results_df.sort_values('mean_test_rmse')

# Save all results
results_df.to_csv('xgboost_hyperparameter_search_results.csv', index=False)
print("\n[OK] Full results saved to xgboost_hyperparameter_search_results.csv")

# Create summary with key metrics
summary_cols = [
    'param_n_estimators', 'param_max_depth', 'param_learning_rate',
    'param_subsample', 'param_colsample_bytree', 'param_min_child_weight',
    'param_gamma', 'param_reg_alpha', 'param_reg_lambda',
    'mean_test_rmse', 'std_test_rmse', 'mean_test_mae', 'mean_test_r2',
    'mean_fit_time', 'rank_test_neg_mse'
]

summary_df = results_df[summary_cols].copy()
summary_df.columns = [
    'n_estimators', 'max_depth', 'learning_rate',
    'subsample', 'colsample_bytree', 'min_child_weight',
    'gamma', 'reg_alpha', 'reg_lambda',
    'mean_cv_rmse', 'std_cv_rmse', 'mean_cv_mae', 'mean_cv_r2',
    'mean_fit_time_sec', 'rank'
]

summary_df.to_csv('xgboost_hyperparameter_summary.csv', index=False)
print("[OK] Summary saved to xgboost_hyperparameter_summary.csv")

# Display top 20 configurations
print("\n" + "="*80)
print("TOP 20 CONFIGURATIONS (by RMSE)")
print("="*80)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(summary_df.head(20).to_string(index=False))

# Best parameters
print("\n" + "="*80)
print("BEST HYPERPARAMETERS (by CV MSE)")
print("="*80)
print(f"  n_estimators: {grid_search.best_params_['n_estimators']}")
print(f"  max_depth: {grid_search.best_params_['max_depth']}")
print(f"  learning_rate: {grid_search.best_params_['learning_rate']}")
print(f"  subsample: {grid_search.best_params_['subsample']}")
print(f"  colsample_bytree: {grid_search.best_params_['colsample_bytree']}")
print(f"  min_child_weight: {grid_search.best_params_['min_child_weight']}")
print(f"  gamma: {grid_search.best_params_['gamma']}")
print(f"  reg_alpha: {grid_search.best_params_['reg_alpha']}")
print(f"  reg_lambda: {grid_search.best_params_['reg_lambda']}")
print(f"\nBest CV RMSE: {summary_df.iloc[0]['mean_cv_rmse']:.4f} +/- {summary_df.iloc[0]['std_cv_rmse']:.4f}")
print(f"Best CV MAE: {summary_df.iloc[0]['mean_cv_mae']:.4f}")
print(f"Best CV R^2: {summary_df.iloc[0]['mean_cv_r2']:.4f}")
print("="*80)

# ============================================================================
# TRAIN FINAL MODEL ON FULL 70% AND EVALUATE ON 30% TEST SET
# ============================================================================

print("\n" + "="*80)
print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
print("="*80)

# Load test data (30%)
print("\nLoading test data (30%)...")
test_df = pd.read_parquet('test_30.parquet')
y_test = test_df['trip_duration'].values
X_test = test_df.drop('trip_duration', axis=1)

print(f"Test data shape: {X_test.shape}")

# Preprocess test data (same as training)
X_test_prep = X_test.copy()

# Convert numeric columns to float64
for col in numeric_cols:
    X_test_prep[col] = X_test_prep[col].astype('float64')

# Convert categorical columns
for col in categorical_cols:
    X_test_prep[col] = X_test_prep[col].astype(str).astype('category')

print(f"Test data preprocessed: {X_test_prep.shape}")

# The best model is already trained on full 70% data (refit=True in GridSearchCV)
print("\nEvaluating best model on test set (30%)...")
best_model = grid_search.best_estimator_

# Predict on test set
y_pred_test = best_model.predict(X_test_prep)

# Calculate test metrics
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print("\n" + "="*80)
print("FINAL MODEL PERFORMANCE ON TEST SET (30%)")
print("="*80)
print(f"RMSE:  {test_rmse:.4f}")
print(f"MSE:   {test_mse:.4f}")
print(f"MAE:   {test_mae:.4f}")
print(f"R^2 Score: {test_r2:.4f}")
print("="*80)

# Save final metrics
final_metrics = {
    'n_estimators': grid_search.best_params_['n_estimators'],
    'max_depth': grid_search.best_params_['max_depth'],
    'learning_rate': grid_search.best_params_['learning_rate'],
    'subsample': grid_search.best_params_['subsample'],
    'colsample_bytree': grid_search.best_params_['colsample_bytree'],
    'min_child_weight': grid_search.best_params_['min_child_weight'],
    'gamma': grid_search.best_params_['gamma'],
    'reg_alpha': grid_search.best_params_['reg_alpha'],
    'reg_lambda': grid_search.best_params_['reg_lambda'],
    'device': grid_search.best_params_['device'],
    'cv_rmse_mean': summary_df.iloc[0]['mean_cv_rmse'],
    'cv_rmse_std': summary_df.iloc[0]['std_cv_rmse'],
    'cv_mae': summary_df.iloc[0]['mean_cv_mae'],
    'cv_r2': summary_df.iloc[0]['mean_cv_r2'],
    'test_rmse': test_rmse,
    'test_mse': test_mse,
    'test_mae': test_mae,
    'test_r2': test_r2,
    'total_search_time_min': total_time / 60,
    'total_configurations': total_configs
}

pd.DataFrame([final_metrics]).to_csv('xgboost_final_model_metrics.csv', index=False)
print("\n[OK] Final metrics saved to xgboost_final_model_metrics.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\nGenerating visualizations...")

fig = plt.figure(figsize=(20, 12))

# Create subplots
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Top 30 configurations
ax1 = fig.add_subplot(gs[0, :])
top_30 = summary_df.head(30).copy()
top_30['config'] = [f"n={int(row['n_estimators'])}, d={int(row['max_depth'])}, lr={row['learning_rate']:.3f}" 
                     for _, row in top_30.iterrows()]
ax1.barh(range(len(top_30)), top_30['mean_cv_rmse'], color='steelblue', alpha=0.7)
ax1.set_yticks(range(len(top_30)))
ax1.set_yticklabels(top_30['config'], fontsize=7)
ax1.invert_yaxis()
ax1.set_xlabel('Mean CV RMSE', fontsize=10)
ax1.set_title('Top 30 Configurations by RMSE', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# RMSE by n_estimators
ax2 = fig.add_subplot(gs[1, 0])
n_est_rmse = results_df.groupby('param_n_estimators')['mean_test_rmse'].agg(['mean', 'min', 'std'])
ax2.errorbar(n_est_rmse.index, n_est_rmse['mean'], yerr=n_est_rmse['std'], 
             marker='o', capsize=5, linewidth=2, markersize=8, label='Mean ± Std')
ax2.plot(n_est_rmse.index, n_est_rmse['min'], marker='s', color='green', 
         linewidth=2, markersize=8, label='Best')
ax2.set_xlabel('n_estimators', fontsize=10)
ax2.set_ylabel('RMSE', fontsize=10)
ax2.set_title('RMSE by n_estimators', fontsize=11, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# RMSE by max_depth
ax3 = fig.add_subplot(gs[1, 1])
depth_rmse = results_df.groupby('param_max_depth')['mean_test_rmse'].agg(['mean', 'min', 'std'])
ax3.errorbar(depth_rmse.index, depth_rmse['mean'], yerr=depth_rmse['std'], 
             marker='o', capsize=5, linewidth=2, markersize=8, label='Mean ± Std')
ax3.plot(depth_rmse.index, depth_rmse['min'], marker='s', color='green', 
         linewidth=2, markersize=8, label='Best')
ax3.set_xlabel('max_depth', fontsize=10)
ax3.set_ylabel('RMSE', fontsize=10)
ax3.set_title('RMSE by max_depth', fontsize=11, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# RMSE by learning_rate
ax4 = fig.add_subplot(gs[1, 2])
lr_rmse = results_df.groupby('param_learning_rate')['mean_test_rmse'].agg(['mean', 'min', 'std'])
ax4.errorbar(lr_rmse.index, lr_rmse['mean'], yerr=lr_rmse['std'], 
             marker='o', capsize=5, linewidth=2, markersize=8, label='Mean ± Std')
ax4.plot(lr_rmse.index, lr_rmse['min'], marker='s', color='green', 
         linewidth=2, markersize=8, label='Best')
ax4.set_xscale('log')
ax4.set_xlabel('learning_rate (log scale)', fontsize=10)
ax4.set_ylabel('RMSE', fontsize=10)
ax4.set_title('RMSE by learning_rate', fontsize=11, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Heatmap: learning_rate vs max_depth
ax5 = fig.add_subplot(gs[2, 0])
pivot_data = results_df.pivot_table(
    values='mean_test_rmse', 
    index='param_max_depth', 
    columns='param_learning_rate',
    aggfunc='min'
)
sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax5, cbar_kws={'label': 'RMSE'})
ax5.set_title('Best RMSE: max_depth vs learning_rate', fontsize=11, fontweight='bold')
ax5.set_xlabel('learning_rate', fontsize=10)
ax5.set_ylabel('max_depth', fontsize=10)

# Feature importance (top 20)
ax6 = fig.add_subplot(gs[2, 1])
feature_importance = pd.DataFrame({
    'feature': X_train_prep.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)
ax6.barh(range(len(feature_importance)), feature_importance['importance'], color='coral', alpha=0.7)
ax6.set_yticks(range(len(feature_importance)))
ax6.set_yticklabels(feature_importance['feature'], fontsize=8)
ax6.invert_yaxis()
ax6.set_xlabel('Importance', fontsize=10)
ax6.set_title('Top 20 Feature Importances', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')

# Test predictions vs actual
ax7 = fig.add_subplot(gs[2, 2])
ax7.scatter(y_test, y_pred_test, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax7.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax7.set_xlabel('Actual Trip Duration (minutes)', fontsize=10)
ax7.set_ylabel('Predicted Trip Duration (minutes)', fontsize=10)
ax7.set_title('Test Set: Predictions vs Actual', fontsize=11, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Add metrics text
textstr = f'RMSE: {test_rmse:.4f}\nMAE: {test_mae:.4f}\nR²: {test_r2:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax7.text(0.05, 0.95, textstr, transform=ax7.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.savefig('xgboost_hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
print("[OK] Visualization saved to xgboost_hyperparameter_tuning_results.png")

plt.show()

print("\n" + "="*80)
print("XGBOOST HYPERPARAMETER TUNING COMPLETE")
print("="*80)
print("\nOutput files:")
print("  1. xgboost_hyperparameter_search_results.csv - Full CV results")
print("  2. xgboost_hyperparameter_summary.csv - Summary of top configs")
print("  3. xgboost_final_model_metrics.csv - Final test set metrics")
print("  4. xgboost_hyperparameter_tuning_results.png - Visualizations")
print("="*80)
