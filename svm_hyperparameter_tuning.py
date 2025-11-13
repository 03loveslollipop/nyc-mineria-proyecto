"""
SVM Hyperparameter Tuning with Grid Search and Cross-Validation
Performs exhaustive grid search with 5-fold CV on multiple SVM configurations
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import ParameterGrid, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("SVM HYPERPARAMETER TUNING WITH GRID SEARCH AND 5-FOLD CV")
print("="*80)

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
# DATA PREPROCESSING
# ============================================================================

print("\nPreparing data for SVM...")
X_train_prep = X_train_full.copy()

# Identify numeric columns
numeric_cols_all = [col for col in X_train_prep.columns if col not in categorical_cols]

# Separate columns for different scaling
std_scaler_cols = ['mta_tax', 'extra']
power_transform_cols = [col for col in numeric_cols_all if col not in std_scaler_cols]

# Apply StandardScaler to mta_tax and extra
scaler_std = StandardScaler()
X_train_prep[std_scaler_cols] = scaler_std.fit_transform(X_train_prep[std_scaler_cols])

# Apply PowerTransformer to other numeric columns
scaler_power = PowerTransformer()
X_train_prep[power_transform_cols] = scaler_power.fit_transform(X_train_prep[power_transform_cols])

print(f"StandardScaler applied to: {std_scaler_cols}")
print(f"PowerTransformer applied to: {power_transform_cols}")

# Convert categorical to dummies
for col in categorical_cols:
    X_train_prep[col] = X_train_prep[col].astype('category')

X_train_prep = pd.get_dummies(X_train_prep, columns=categorical_cols, drop_first=True)

# Ensure all columns are numeric
X_train_prep = X_train_prep.astype('float64')

# Convert to numpy arrays
X_train_np = X_train_prep.values

print(f"Preprocessed data shape: {X_train_np.shape}")
print(f"Features: {X_train_prep.shape[1]}")

# ============================================================================
# GRID SEARCH SETUP
# ============================================================================

print("\n" + "="*80)
print("GRID SEARCH CONFIGURATION")
print("="*80)

# Build a memory-friendly parameter grid: degree only included for 'poly'
param_grid_common = {
    'C': [1, 5, 10, 50, 100, 500],
    'gamma': ['auto', 'scale', 0.01, 0.05, 0.1, 0.5, 1],
    'epsilon': [0.1],  # Keep fixed for regression
    'cache_size': [256],  # MB - reduced to fit in 12GB RAM
    'max_iter': [1000]
}

# Separate grids so we don't multiply by degree for non-poly kernels
grid_rbfsig = {
    **{k: v for k, v in param_grid_common.items()},
    'kernel': ['rbf', 'sigmoid']
}

grid_poly = {
    **{k: v for k, v in param_grid_common.items()},
    'kernel': ['poly'],
    'degree': [2, 4]
}

# Approximate total configurations (sequential evaluation will be used)
total_configs = (len(grid_rbfsig['C']) * len(grid_rbfsig['gamma']) * len(grid_rbfsig['kernel'])
                 + len(grid_poly['C']) * len(grid_poly['gamma']) * len(grid_poly['degree']))

print(f"\nParameter grid (memory-optimized):")
print(f"  C: {grid_rbfsig['C']}")
print(f"  gamma: {grid_rbfsig['gamma']}")
print(f"  kernel (rbf/sigmoid): {grid_rbfsig['kernel']}")
print(f"  kernel (poly) degrees: {grid_poly['degree']}")
print(f"  epsilon: {param_grid_common['epsilon']}")
print(f"  cache_size: {param_grid_common['cache_size']} MB (reduced)")
print(f"  max_iter: {param_grid_common['max_iter']}")
print(f"\nApproximate total configurations: {total_configs}")
print(f"With 5-fold CV (sequential runs, n_jobs=1): {total_configs * 5} total fits")
print("="*80)

# ============================================================================
# RUN GRID SEARCH
# ============================================================================

print("\nStarting sequential parameter search using ParameterGrid + cross_validate")

# Scoring mapping for cross_validate
scoring = {
    'neg_mean_squared_error': 'neg_mean_squared_error',
    'neg_mean_absolute_error': 'neg_mean_absolute_error',
    'r2': 'r2'
}

results = []
start_time = time.time()

# Helper to evaluate a single parameter dict using 5-fold CV sequentially (low memory)
def evaluate_params(params, X, y, cv=5):
    # Ensure cache_size is small to limit memory
    params_local = params.copy()
    params_local['cache_size'] = int(params_local.get('cache_size', 256))

    model = SVR(**{k: v for k, v in params_local.items() if k in SVR().get_params()})

    cv_res = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=8,
        error_score='raise'
    )

    mean_neg_mse = np.mean(cv_res['test_neg_mean_squared_error'])
    mean_neg_mae = np.mean(cv_res['test_neg_mean_absolute_error'])
    mean_r2 = np.mean(cv_res['test_r2'])

    return {
        'mean_neg_mse': mean_neg_mse,
        'mean_neg_mae': mean_neg_mae,
        'mean_r2': mean_r2,
        'fit_times': np.mean(cv_res.get('fit_time', [np.nan]))
    }


# Iterate grids sequentially (degree only applied for poly)
grids = [grid_rbfsig, grid_poly]
config_counter = 0
for g in grids:
    for params in ParameterGrid(g):
        config_counter += 1
        print(f"\n[{config_counter}/{total_configs}] Evaluating: {params}")
        t0 = time.time()
        try:
            metrics = evaluate_params(params, X_train_np, y_train_full, cv=5)
            elapsed = time.time() - t0
            results.append({
                **params,
                'mean_fit_time': metrics['fit_times'],
                'mean_test_neg_mse': metrics['mean_neg_mse'],
                'mean_test_neg_mae': metrics['mean_neg_mae'],
                'mean_test_r2': metrics['mean_r2'],
                'eval_time_sec': elapsed,
                'status': 'success'
            })
            print(f"  -> mean_test_neg_mse: {metrics['mean_neg_mse']:.6f}, time: {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [ERROR] Params failed: {e} (time: {elapsed:.2f}s)")
            results.append({**params, 'status': 'failed', 'error': str(e), 'eval_time_sec': elapsed})

total_time = time.time() - start_time
print("\n" + "="*80)
print(f"Sequential search completed in {total_time/60:.2f} minutes")
print("="*80)

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SEARCH RESULTS")
print("="*80)

# Build DataFrame from sequential results
results_df = pd.DataFrame(results)

# Convert stored negative metrics to interpretable metrics
if 'mean_test_neg_mse' in results_df.columns:
    results_df['mean_test_rmse'] = np.sqrt(-results_df['mean_test_neg_mse'])
else:
    results_df['mean_test_rmse'] = np.nan

if 'mean_test_neg_mae' in results_df.columns:
    results_df['mean_test_mae'] = -results_df['mean_test_neg_mae']
else:
    results_df['mean_test_mae'] = np.nan

results_df = results_df.sort_values('mean_test_rmse', na_position='last')

# Save full results
results_df.to_csv('svm_hyperparameter_search_results.csv', index=False)
print("\n[OK] Full results saved to svm_hyperparameter_search_results.csv")

# Create a concise summary
summary_cols = [
    'C', 'gamma', 'kernel', 'degree',
    'mean_test_rmse', 'mean_test_mae', 'mean_test_r2',
    'mean_fit_time', 'eval_time_sec', 'status'
]

summary_df = results_df.copy()
for c in ['C', 'gamma', 'kernel', 'degree']:
    if c not in summary_df.columns:
        summary_df[c] = np.nan

summary_df = summary_df[summary_cols]
summary_df.columns = [
    'C', 'gamma', 'kernel', 'degree',
    'mean_cv_rmse', 'mean_cv_mae', 'mean_cv_r2',
    'mean_fit_time_sec', 'eval_time_sec', 'status'
]

summary_df.to_csv('svm_hyperparameter_summary.csv', index=False)
print("[OK] Summary saved to svm_hyperparameter_summary.csv")

# Display top 20 configurations
print("\n" + "="*80)
print("TOP 20 CONFIGURATIONS (by RMSE)")
print("="*80)
print(summary_df[summary_df['status'] == 'success'].head(20).to_string(index=False))

# Best parameters (from sequential results)
best_row = results_df[results_df['status'] == 'success'].sort_values('mean_test_rmse').head(1)
if best_row.shape[0] == 0:
    raise RuntimeError('No successful parameter evaluations — nothing to refit')

best_row = best_row.iloc[0]
best_params = best_row.to_dict()

print("\n" + "="*80)
print("BEST HYPERPARAMETERS (from sequential CV)")
print("="*80)
print(f"  C: {best_params.get('C')}")
print(f"  gamma: {best_params.get('gamma')}")
print(f"  kernel: {best_params.get('kernel')}")
print(f"  degree: {best_params.get('degree')}")
print(f"  epsilon: {best_params.get('epsilon')}")
print(f"\nBest CV RMSE: {best_row.get('mean_test_rmse'):.4f}")
print(f"Best CV MAE: {best_row.get('mean_test_mae'):.4f}")
print(f"Best CV R^2: {best_row.get('mean_test_r2'):.4f}")
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

# Apply same scaling
X_test_prep[std_scaler_cols] = scaler_std.transform(X_test_prep[std_scaler_cols])
X_test_prep[power_transform_cols] = scaler_power.transform(X_test_prep[power_transform_cols])

# Convert categorical to dummies
for col in categorical_cols:
    X_test_prep[col] = X_test_prep[col].astype('category')

X_test_prep = pd.get_dummies(X_test_prep, columns=categorical_cols, drop_first=True)

# Align columns with training data
X_test_prep = X_test_prep.reindex(columns=X_train_prep.columns, fill_value=0)

# Ensure all columns are numeric
X_test_prep = X_test_prep.astype('float64')

# Convert to numpy
X_test_np = X_test_prep.values

print(f"Test data preprocessed: {X_test_np.shape}")

print("\nRefitting best model on full 70% training data and evaluating on test set (30%)...")

# Build sklearn-friendly params for re-fitting
svr_params = {}
for k in ['C', 'gamma', 'kernel', 'degree', 'epsilon', 'cache_size', 'max_iter']:
    if k in best_params and not pd.isna(best_params[k]):
        svr_params[k] = best_params[k]

best_model = SVR(**svr_params)
best_model.fit(X_train_np, y_train_full)

# Predict on test set
y_pred_test = best_model.predict(X_test_np)

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
    'C': best_params.get('C'),
    'gamma': best_params.get('gamma'),
    'kernel': best_params.get('kernel'),
    'degree': best_params.get('degree'),
    'epsilon': best_params.get('epsilon'),
    'cv_rmse_mean': best_row.get('mean_test_rmse'),
    'cv_rmse_std': np.nan,
    'cv_mae': best_row.get('mean_test_mae'),
    'cv_r2': best_row.get('mean_test_r2'),
    'test_rmse': test_rmse,
    'test_mse': test_mse,
    'test_mae': test_mae,
    'test_r2': test_r2,
    'total_search_time_min': total_time / 60
}

pd.DataFrame([final_metrics]).to_csv('svm_final_model_metrics.csv', index=False)
print("\n[OK] Final metrics saved to svm_final_model_metrics.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\nGenerating visualizations...")

# Figure 1: Top configurations comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top 20 RMSE
top_20 = summary_df.head(20).copy()
top_20['config'] = [f"C={row['C']}, g={row['gamma']}, k={row['kernel']}" 
                     for _, row in top_20.iterrows()]

axes[0, 0].barh(range(len(top_20)), top_20['mean_cv_rmse'], color='steelblue', alpha=0.7)
axes[0, 0].set_yticks(range(len(top_20)))
axes[0, 0].set_yticklabels(top_20['config'], fontsize=8)
axes[0, 0].invert_yaxis()
axes[0, 0].set_xlabel('Mean CV RMSE')
axes[0, 0].set_title('Top 20 Configurations by RMSE')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# RMSE by kernel
kernel_rmse = results_df.groupby('kernel')['mean_test_rmse'].agg(['mean', 'min', 'max'])
axes[0, 1].bar(kernel_rmse.index, kernel_rmse['mean'], alpha=0.7, label='Mean')
axes[0, 1].scatter(kernel_rmse.index, kernel_rmse['min'], color='green', s=100, 
                   marker='v', label='Best', zorder=3)
axes[0, 1].set_xlabel('Kernel')
axes[0, 1].set_ylabel('RMSE')
axes[0, 1].set_title('RMSE by Kernel Type')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# RMSE by C (for best kernel)
best_kernel = best_params.get('kernel')
kernel_data = results_df[results_df['kernel'] == best_kernel]
c_rmse = kernel_data.groupby('C')['mean_test_rmse'].agg(['mean', 'min'])
axes[1, 0].plot(c_rmse.index, c_rmse['mean'], marker='o', label='Mean', linewidth=2)
axes[1, 0].plot(c_rmse.index, c_rmse['min'], marker='s', label='Best', linewidth=2)
axes[1, 0].set_xscale('log')
axes[1, 0].set_xlabel('C (log scale)')
axes[1, 0].set_ylabel('RMSE')
axes[1, 0].set_title(f'RMSE by C (kernel={best_kernel})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Test predictions vs actual
axes[1, 1].scatter(y_test, y_pred_test, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual Trip Duration (minutes)')
axes[1, 1].set_ylabel('Predicted Trip Duration (minutes)')
axes[1, 1].set_title('Test Set: Predictions vs Actual')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Add metrics text
textstr = f'RMSE: {test_rmse:.4f}\nMAE: {test_mae:.4f}\nR^2: {test_r2:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
axes[1, 1].text(0.05, 0.95, textstr, transform=axes[1, 1].transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('svm_hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
print("[OK] Visualization saved to svm_hyperparameter_tuning_results.png")

plt.show()

print("\n" + "="*80)
print("SVM HYPERPARAMETER TUNING COMPLETE")
print("="*80)
print("\nOutput files:")
print("  1. svm_hyperparameter_search_results.csv - Full CV results")
print("  2. svm_hyperparameter_summary.csv - Summary of top configs")
print("  3. svm_final_model_metrics.csv - Final test set metrics")
print("  4. svm_hyperparameter_tuning_results.png - Visualizations")
print("="*80)
