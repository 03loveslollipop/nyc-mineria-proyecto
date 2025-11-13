"""
Bayesian Optimization for XGBoost Hyperparameters
Uses scikit-optimize to search within ±25% of best numerical parameters with 400 iterations and 5-fold CV
"""

import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("BAYESIAN OPTIMIZATION FOR XGBOOST - 400 ITERATIONS WITH 5-FOLD CV")
print("="*80)

# ============================================================================
# DETERMINE DEVICE
# ============================================================================

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
# EXTRACT BEST PARAMETERS FROM PREVIOUS GRID SEARCH
# ============================================================================

print("\n" + "="*80)
print("LOADING BEST PARAMETERS FROM PREVIOUS GRID SEARCH")
print("="*80)

# Load previous grid search results
prev_results = pd.read_csv('xgboost_hyperparameter_search_results.csv')

# Get best configuration (lowest mean_test_neg_mse = highest rank)
best_config = prev_results.sort_values('mean_test_neg_mse', ascending=False).iloc[0]

# Extract numerical parameters
base_n_estimators = int(best_config['param_n_estimators'])
base_max_depth = int(best_config['param_max_depth'])
base_learning_rate = float(best_config['param_learning_rate'])
base_subsample = float(best_config['param_subsample'])
base_colsample_bytree = float(best_config['param_colsample_bytree'])
base_min_child_weight = int(best_config['param_min_child_weight'])
base_gamma = float(best_config['param_gamma'])
base_reg_alpha = float(best_config['param_reg_alpha'])
base_reg_lambda = float(best_config['param_reg_lambda'])

print(f"\nBest previous configuration:")
print(f"  n_estimators: {base_n_estimators}")
print(f"  max_depth: {base_max_depth}")
print(f"  learning_rate: {base_learning_rate}")
print(f"  subsample: {base_subsample}")
print(f"  colsample_bytree: {base_colsample_bytree}")
print(f"  min_child_weight: {base_min_child_weight}")
print(f"  gamma: {base_gamma}")
print(f"  reg_alpha: {base_reg_alpha}")
print(f"  reg_lambda: {base_reg_lambda}")
print(f"  Best CV RMSE: {np.sqrt(-best_config['mean_test_neg_mse']):.4f}")

# ============================================================================
# DEFINE SEARCH SPACE (±25% for numerical parameters)
# ============================================================================

print("\n" + "="*80)
print("BAYESIAN OPTIMIZATION SEARCH SPACE (±25% BOUNDS)")
print("="*80)

# Calculate ±25% bounds for each parameter
def get_bounds(value, lower_clip=None, upper_clip=None, min_range=None):
    """Get ±25% bounds, with optional clipping and minimum range"""
    lower = value * 0.75
    upper = value * 1.25
    
    # If value is very small or zero, use a minimum range
    if min_range is not None and (upper - lower) < min_range:
        mid = (lower + upper) / 2
        lower = mid - min_range / 2
        upper = mid + min_range / 2
    
    if lower_clip is not None:
        lower = max(lower, lower_clip)
    if upper_clip is not None:
        upper = min(upper, upper_clip)
    
    # Ensure lower < upper
    if lower >= upper:
        if lower_clip is not None and upper_clip is not None:
            # Use the full clip range
            lower = lower_clip
            upper = upper_clip
        else:
            # Add small epsilon
            upper = lower + 0.01
    
    return lower, upper

# Define search space
n_est_low, n_est_high = get_bounds(base_n_estimators, lower_clip=10, upper_clip=500)
max_depth_low, max_depth_high = get_bounds(base_max_depth, lower_clip=3, upper_clip=15)
lr_low, lr_high = get_bounds(base_learning_rate, lower_clip=0.001, upper_clip=0.5)
subsample_low, subsample_high = get_bounds(base_subsample, lower_clip=0.5, upper_clip=1.0)
colsample_low, colsample_high = get_bounds(base_colsample_bytree, lower_clip=0.5, upper_clip=1.0)
min_child_low, min_child_high = get_bounds(base_min_child_weight, lower_clip=1, upper_clip=10)
gamma_low, gamma_high = get_bounds(base_gamma, lower_clip=0.0, upper_clip=1.0, min_range=0.2)
reg_alpha_low, reg_alpha_high = get_bounds(base_reg_alpha, lower_clip=0.0, upper_clip=1.0, min_range=0.2)
reg_lambda_low, reg_lambda_high = get_bounds(base_reg_lambda, lower_clip=0.1, upper_clip=5.0, min_range=0.5)

space = [
    Integer(int(n_est_low), int(n_est_high), name='n_estimators'),
    Integer(int(max_depth_low), int(max_depth_high), name='max_depth'),
    Real(lr_low, lr_high, prior='log-uniform', name='learning_rate'),
    Real(subsample_low, subsample_high, prior='uniform', name='subsample'),
    Real(colsample_low, colsample_high, prior='uniform', name='colsample_bytree'),
    Integer(int(min_child_low), int(min_child_high), name='min_child_weight'),
    Real(gamma_low, gamma_high, prior='uniform', name='gamma'),
    Real(reg_alpha_low, reg_alpha_high, prior='uniform', name='reg_alpha'),
    Real(reg_lambda_low, reg_lambda_high, prior='uniform', name='reg_lambda'),
]

print("\nSearch space (±25% from best):")
print(f"  n_estimators: [{int(n_est_low)}, {int(n_est_high)}]")
print(f"  max_depth: [{int(max_depth_low)}, {int(max_depth_high)}]")
print(f"  learning_rate: [{lr_low:.4f}, {lr_high:.4f}] (log-uniform)")
print(f"  subsample: [{subsample_low:.4f}, {subsample_high:.4f}]")
print(f"  colsample_bytree: [{colsample_low:.4f}, {colsample_high:.4f}]")
print(f"  min_child_weight: [{int(min_child_low)}, {int(min_child_high)}]")
print(f"  gamma: [{gamma_low:.4f}, {gamma_high:.4f}]")
print(f"  reg_alpha: [{reg_alpha_low:.4f}, {reg_alpha_high:.4f}]")
print(f"  reg_lambda: [{reg_lambda_low:.4f}, {reg_lambda_high:.4f}]")
print("\nTotal iterations: 400")
print("Cross-validation: 5-fold")
print("="*80)

# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

# Keep track of iteration history for detailed logging
iteration_history = []
iteration_counter = [0]  # Use list to allow modification in nested function

@use_named_args(space)
def objective(n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
              min_child_weight, gamma, reg_alpha, reg_lambda):
    """Objective function for Bayesian optimization"""
    iteration_counter[0] += 1
    
    # Build XGBoost model parameters
    xgb_params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'learning_rate': float(learning_rate),
        'subsample': float(subsample),
        'colsample_bytree': float(colsample_bytree),
        'min_child_weight': int(min_child_weight),
        'gamma': float(gamma),
        'reg_alpha': float(reg_alpha),
        'reg_lambda': float(reg_lambda),
        'tree_method': 'hist',
        'device': device,
        'enable_categorical': True,
        'random_state': 42
    }
    
    print(f"\n[Iteration {iteration_counter[0]}/400]")
    print(f"  n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate:.4f}")
    print(f"  subsample={subsample:.3f}, colsample={colsample_bytree:.3f}")
    print(f"  min_child_weight={min_child_weight}, gamma={gamma:.3f}")
    print(f"  reg_alpha={reg_alpha:.3f}, reg_lambda={reg_lambda:.3f}")
    
    try:
        # Create model
        model = xgb.XGBRegressor(**xgb_params)
        
        # Perform 5-fold cross-validation
        scoring = {
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }
        
        start_time = time.time()
        cv_results = cross_validate(
            model,
            X_train_prep,
            y_train_full,
            cv=5,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,  # XGBoost handles parallelism internally with GPU
            error_score='raise'
        )
        elapsed = time.time() - start_time
        
        # Extract metrics
        mean_neg_mse = np.mean(cv_results['test_neg_mse'])
        mean_rmse = np.sqrt(-mean_neg_mse)
        mean_mae = -np.mean(cv_results['test_neg_mae'])
        mean_r2 = np.mean(cv_results['test_r2'])
        
        print(f"  -> CV RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}, R²: {mean_r2:.4f} ({elapsed:.1f}s)")
        
        # Log to history
        iteration_history.append({
            'iteration': iteration_counter[0],
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
            'learning_rate': float(learning_rate),
            'subsample': float(subsample),
            'colsample_bytree': float(colsample_bytree),
            'min_child_weight': int(min_child_weight),
            'gamma': float(gamma),
            'reg_alpha': float(reg_alpha),
            'reg_lambda': float(reg_lambda),
            'cv_rmse': mean_rmse,
            'cv_mae': mean_mae,
            'cv_r2': mean_r2,
            'cv_neg_mse': mean_neg_mse,
            'elapsed_sec': elapsed,
            'status': 'success'
        })
        
        # Return RMSE (to minimize)
        return mean_rmse
        
    except Exception as e:
        print(f"  [ERROR] {str(e)}")
        
        # Log failed iteration
        iteration_history.append({
            'iteration': iteration_counter[0],
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
            'learning_rate': float(learning_rate),
            'subsample': float(subsample),
            'colsample_bytree': float(colsample_bytree),
            'min_child_weight': int(min_child_weight),
            'gamma': float(gamma),
            'reg_alpha': float(reg_alpha),
            'reg_lambda': float(reg_lambda),
            'cv_rmse': np.nan,
            'cv_mae': np.nan,
            'cv_r2': np.nan,
            'cv_neg_mse': np.nan,
            'elapsed_sec': 0.0,
            'status': 'failed',
            'error': str(e)
        })
        
        # Return a large penalty value
        return 1e6

# ============================================================================
# RUN BAYESIAN OPTIMIZATION
# ============================================================================

print("\n" + "="*80)
print("STARTING BAYESIAN OPTIMIZATION")
print("="*80)

start_time = time.time()

# Run optimization
result = gp_minimize(
    objective,
    dimensions=space,
    n_calls=600,  # Total iterations
    n_initial_points=30,  # Random exploration before Gaussian Process
    acq_func='EI',  # Expected Improvement
    random_state=42,
    verbose=False
)

total_time = time.time() - start_time

print("\n" + "="*80)
print(f"OPTIMIZATION COMPLETED IN {total_time/60:.2f} MINUTES ({total_time/3600:.2f} HOURS)")
print("="*80)

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

print("\n" + "="*80)
print("BEST HYPERPARAMETERS FOUND")
print("="*80)

best_n_estimators = result.x[0]
best_max_depth = result.x[1]
best_learning_rate = result.x[2]
best_subsample = result.x[3]
best_colsample_bytree = result.x[4]
best_min_child_weight = result.x[5]
best_gamma = result.x[6]
best_reg_alpha = result.x[7]
best_reg_lambda = result.x[8]

print(f"  n_estimators: {best_n_estimators}")
print(f"  max_depth: {best_max_depth}")
print(f"  learning_rate: {best_learning_rate:.6f}")
print(f"  subsample: {best_subsample:.6f}")
print(f"  colsample_bytree: {best_colsample_bytree:.6f}")
print(f"  min_child_weight: {best_min_child_weight}")
print(f"  gamma: {best_gamma:.6f}")
print(f"  reg_alpha: {best_reg_alpha:.6f}")
print(f"  reg_lambda: {best_reg_lambda:.6f}")
print(f"\nBest CV RMSE: {result.fun:.4f}")

# Save best parameters
best_params = {
    'n_estimators': int(best_n_estimators),
    'max_depth': int(best_max_depth),
    'learning_rate': float(best_learning_rate),
    'subsample': float(best_subsample),
    'colsample_bytree': float(best_colsample_bytree),
    'min_child_weight': int(best_min_child_weight),
    'gamma': float(best_gamma),
    'reg_alpha': float(best_reg_alpha),
    'reg_lambda': float(best_reg_lambda),
    'best_cv_rmse': float(result.fun)
}

with open('best_bayes_xgboost_hyperparameters.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print("\n[OK] Best parameters saved to best_bayes_xgboost_hyperparameters.json")

# Save iteration history
history_df = pd.DataFrame(iteration_history)
history_df = history_df.sort_values('cv_rmse', na_position='last')
history_df.to_csv('bayes_xgboost_optimization_results.csv', index=False)
print("[OK] Full optimization history saved to bayes_xgboost_optimization_results.csv")

# Display top 10 configurations
print("\n" + "="*80)
print("TOP 10 CONFIGURATIONS")
print("="*80)
successful_df = history_df[history_df['status'] == 'success']
if len(successful_df) > 0:
    display_cols = ['iteration', 'n_estimators', 'max_depth', 'learning_rate', 'cv_rmse', 'cv_mae', 'cv_r2']
    print(successful_df[display_cols].head(10).to_string(index=False))
else:
    print("No successful iterations found!")

# ============================================================================
# TRAIN FINAL MODEL AND EVALUATE ON TEST SET
# ============================================================================

print("\n" + "="*80)
print("TRAINING FINAL MODEL ON FULL 70% TRAINING SET")
print("="*80)

# Load test data
print("\nLoading test data (30%)...")
test_df = pd.read_parquet('test_30.parquet')
y_test = test_df['trip_duration'].values
X_test = test_df.drop('trip_duration', axis=1)

print(f"Test data shape: {X_test.shape}")

# Preprocess test data (same transformations)
X_test_prep = X_test.copy()

# Convert numeric columns to float64
for col in numeric_cols:
    X_test_prep[col] = X_test_prep[col].astype('float64')

# Convert categorical columns
for col in categorical_cols:
    X_test_prep[col] = X_test_prep[col].astype(str).astype('category')

print(f"Preprocessed test data shape: {X_test_prep.shape}")

# Build final model with best parameters
final_model_params = {
    'n_estimators': int(best_n_estimators),
    'max_depth': int(best_max_depth),
    'learning_rate': float(best_learning_rate),
    'subsample': float(best_subsample),
    'colsample_bytree': float(best_colsample_bytree),
    'min_child_weight': int(best_min_child_weight),
    'gamma': float(best_gamma),
    'reg_alpha': float(best_reg_alpha),
    'reg_lambda': float(best_reg_lambda),
    'tree_method': 'hist',
    'device': device,
    'enable_categorical': True,
    'random_state': 42
}

print("\nTraining final model with best parameters...")
final_model = xgb.XGBRegressor(**final_model_params)
final_model.fit(X_train_prep, y_train_full)

# Evaluate on test set
print("\nEvaluating on test set...")
y_pred_test = final_model.predict(X_test_prep)

# Calculate metrics
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
print(f"R² Score: {test_r2:.4f}")
print("="*80)

# Save test metrics
test_metrics = {
    'n_estimators': int(best_n_estimators),
    'max_depth': int(best_max_depth),
    'learning_rate': float(best_learning_rate),
    'subsample': float(best_subsample),
    'colsample_bytree': float(best_colsample_bytree),
    'min_child_weight': int(best_min_child_weight),
    'gamma': float(best_gamma),
    'reg_alpha': float(best_reg_alpha),
    'reg_lambda': float(best_reg_lambda),
    'cv_rmse': float(result.fun),
    'test_rmse': float(test_rmse),
    'test_mse': float(test_mse),
    'test_mae': float(test_mae),
    'test_r2': float(test_r2)
}

pd.DataFrame([test_metrics]).to_csv('bayes_xgboost_final_model_metrics.csv', index=False)
print("\n[OK] Test metrics saved to bayes_xgboost_final_model_metrics.csv")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating visualizations...")

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Convergence plot
ax1 = axes[0, 0]
successful_history = history_df[history_df['status'] == 'success'].copy()
if len(successful_history) > 0:
    ax1.plot(successful_history['iteration'], successful_history['cv_rmse'], 
             marker='o', alpha=0.6, linestyle='-', linewidth=1, markersize=3)
    ax1.axhline(y=result.fun, color='r', linestyle='--', linewidth=2, label=f'Best: {result.fun:.4f}')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('CV RMSE')
    ax1.set_title('Bayesian Optimization Convergence (400 iterations)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# 2. Predictions vs Actual
ax2 = axes[0, 1]
# Subsample for clearer visualization
sample_size = min(5000, len(y_test))
sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
y_test_sample = y_test[sample_idx]
y_pred_sample = y_pred_test[sample_idx]

ax2.scatter(y_test_sample, y_pred_sample, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Trip Duration (minutes)')
ax2.set_ylabel('Predicted Trip Duration (minutes)')
ax2.set_title('Test Set: Predictions vs Actual')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add metrics text
textstr = f'RMSE: {test_rmse:.4f}\nMAE: {test_mae:.4f}\nR²: {test_r2:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# 3. Learning rate vs RMSE
ax3 = axes[1, 0]
if len(successful_history) > 0:
    scatter = ax3.scatter(successful_history['learning_rate'], successful_history['cv_rmse'],
                         c=successful_history['iteration'], cmap='viridis',
                         s=50, alpha=0.7, edgecolors='k', linewidths=0.5)
    ax3.set_xlabel('Learning Rate (log scale)')
    ax3.set_ylabel('CV RMSE')
    ax3.set_xscale('log')
    ax3.set_title('Parameter Exploration: Learning Rate vs RMSE')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Iteration')

# 4. Max depth distribution
ax4 = axes[1, 1]
if len(successful_history) > 0:
    depth_stats = successful_history.groupby('max_depth')['cv_rmse'].agg(['mean', 'min', 'count'])
    if len(depth_stats) > 0:
        depth_names = depth_stats.index.tolist()
        means = depth_stats['mean'].values
        mins = depth_stats['min'].values
        counts = depth_stats['count'].values
        
        x_pos = np.arange(len(depth_names))
        bars = ax4.bar(x_pos, means, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.scatter(x_pos, mins, color='red', s=100, zorder=5, label='Best in depth')
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'n={int(count)}', ha='center', va='bottom', fontsize=9)
        
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(depth_names)
        ax4.set_xlabel('Max Depth')
        ax4.set_ylabel('CV RMSE')
        ax4.set_title('Performance by Max Depth')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('bayes_xgboost_optimization_results.png', dpi=300, bbox_inches='tight')
print("[OK] Visualization saved to bayes_xgboost_optimization_results.png")

print("\n" + "="*80)
print("BAYESIAN OPTIMIZATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - bayes_xgboost_optimization_results.csv (full iteration history)")
print("  - best_bayes_xgboost_hyperparameters.json (best parameters)")
print("  - bayes_xgboost_final_model_metrics.csv (test set performance)")
print("  - bayes_xgboost_optimization_results.png (visualizations)")
print("="*80)
