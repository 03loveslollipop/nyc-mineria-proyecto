"""
Bayesian Optimization for SVM Hyperparameters
Uses scikit-optimize to search over all SVM parameters with 100 iterations and 5-fold CV
"""

import json
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("BAYESIAN OPTIMIZATION FOR SVM - 100 ITERATIONS WITH 5-FOLD CV")
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
# DEFINE SEARCH SPACE
# ============================================================================

print("\n" + "="*80)
print("BAYESIAN OPTIMIZATION SEARCH SPACE")
print("="*80)

# Define search space for all SVM hyperparameters
space = [
    Real(0.1, 1000.0, prior='log-uniform', name='C'),
    Categorical(['scale', 'auto'], name='gamma_type'),  # Will map to gamma value
    Real(0.001, 10.0, prior='log-uniform', name='gamma_value'),  # Used when gamma_type='custom'
    Categorical(['rbf', 'poly', 'sigmoid'], name='kernel'),
    Integer(2, 5, name='degree'),  # Only relevant for poly kernel
    Real(0.01, 0.5, prior='uniform', name='epsilon'),
]

print("Search space:")
print("  C: [0.1, 1000.0] (log-uniform)")
print("  gamma: ['scale', 'auto', or custom value in [0.001, 10.0] (log-uniform)]")
print("  kernel: ['rbf', 'poly', 'sigmoid']")
print("  degree: [2, 5] (integer, only for poly)")
print("  epsilon: [0.01, 0.5] (uniform)")
print("\nTotal iterations: 100")
print("Cross-validation: 5-fold")
print("="*80)

# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

# Keep track of iteration history for detailed logging
iteration_history = []
iteration_counter = [0]  # Use list to allow modification in nested function

@use_named_args(space)
def objective(C, gamma_type, gamma_value, kernel, degree, epsilon):
    """Objective function for Bayesian optimization"""
    iteration_counter[0] += 1
    
    # Determine gamma value
    if gamma_type in ['scale', 'auto']:
        gamma = gamma_type
    else:
        gamma = float(gamma_value)
    
    # Build SVM parameters
    svm_params = {
        'C': float(C),
        'gamma': gamma,
        'kernel': kernel,
        'epsilon': float(epsilon),
        'cache_size': 512,  # MB - moderate cache for speed
        'max_iter': 2000
    }
    
    # Add degree only if kernel is poly
    if kernel == 'poly':
        svm_params['degree'] = int(degree)
    
    print(f"\n[Iteration {iteration_counter[0]}/100]")
    print(f"  Testing: C={C:.4f}, gamma={gamma}, kernel={kernel}, degree={degree if kernel=='poly' else 'N/A'}, epsilon={epsilon:.4f}")
    
    try:
        # Create model
        model = SVR(**svm_params)
        
        # Perform 5-fold cross-validation
        scoring = {
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }
        
        start_time = time.time()
        cv_results = cross_validate(
            model,
            X_train_np,
            y_train_full,
            cv=5,
            scoring=scoring,
            return_train_score=False,
            n_jobs=8,  # Parallelize CV folds
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
            'C': float(C),
            'gamma_type': gamma_type,
            'gamma_value': float(gamma_value) if gamma_type not in ['scale', 'auto'] else None,
            'gamma_used': gamma if isinstance(gamma, str) else float(gamma),
            'kernel': kernel,
            'degree': int(degree) if kernel == 'poly' else None,
            'epsilon': float(epsilon),
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
            'C': float(C),
            'gamma_type': gamma_type,
            'gamma_value': float(gamma_value) if gamma_type not in ['scale', 'auto'] else None,
            'gamma_used': gamma if isinstance(gamma, str) else float(gamma),
            'kernel': kernel,
            'degree': int(degree) if kernel == 'poly' else None,
            'epsilon': float(epsilon),
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
    n_calls=100,  # Total iterations
    n_initial_points=20,  # Random exploration before Gaussian Process
    acq_func='EI',  # Expected Improvement
    random_state=42,
    verbose=False
)

total_time = time.time() - start_time

print("\n" + "="*80)
print(f"OPTIMIZATION COMPLETED IN {total_time/60:.2f} MINUTES")
print("="*80)

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

print("\n" + "="*80)
print("BEST HYPERPARAMETERS FOUND")
print("="*80)

best_C = result.x[0]
best_gamma_type = result.x[1]
best_gamma_value = result.x[2]
best_kernel = result.x[3]
best_degree = result.x[4]
best_epsilon = result.x[5]

# Determine actual gamma
if best_gamma_type in ['scale', 'auto']:
    best_gamma = best_gamma_type
else:
    best_gamma = best_gamma_value

print(f"  C: {best_C:.6f}")
print(f"  gamma: {best_gamma}")
print(f"  kernel: {best_kernel}")
print(f"  degree: {best_degree} (only used if kernel='poly')")
print(f"  epsilon: {best_epsilon:.6f}")
print(f"\nBest CV RMSE: {result.fun:.4f}")

# Save best parameters
best_params = {
    'C': float(best_C),
    'gamma': best_gamma if isinstance(best_gamma, str) else float(best_gamma),
    'kernel': best_kernel,
    'degree': int(best_degree) if best_kernel == 'poly' else None,
    'epsilon': float(best_epsilon),
    'best_cv_rmse': float(result.fun)
}

with open('best_bayes_svm_hyperparameters.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print("\n[OK] Best parameters saved to best_bayes_svm_hyperparameters.json")

# Save iteration history
history_df = pd.DataFrame(iteration_history)
history_df = history_df.sort_values('cv_rmse', na_position='last')
history_df.to_csv('bayes_svm_optimization_results.csv', index=False)
print("[OK] Full optimization history saved to bayes_svm_optimization_results.csv")

# Display top 10 configurations
print("\n" + "="*80)
print("TOP 10 CONFIGURATIONS")
print("="*80)
successful_df = history_df[history_df['status'] == 'success']
if len(successful_df) > 0:
    display_cols = ['iteration', 'C', 'gamma_used', 'kernel', 'degree', 'epsilon', 'cv_rmse', 'cv_mae', 'cv_r2']
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

# Apply same scaling (already fitted on training data)
X_test_prep[std_scaler_cols] = scaler_std.transform(X_test_prep[std_scaler_cols])
X_test_prep[power_transform_cols] = scaler_power.transform(X_test_prep[power_transform_cols])

# Convert categorical to dummies
for col in categorical_cols:
    X_test_prep[col] = X_test_prep[col].astype('category')

X_test_prep = pd.get_dummies(X_test_prep, columns=categorical_cols, drop_first=True)

# Align columns with training data
X_test_prep = X_test_prep.reindex(columns=X_train_prep.columns, fill_value=0)

# Convert to numpy
X_test_np = X_test_prep.values

print(f"Preprocessed test data shape: {X_test_np.shape}")

# Build final model with best parameters
final_model_params = {
    'C': float(best_C),
    'gamma': best_gamma,
    'kernel': best_kernel,
    'epsilon': float(best_epsilon),
    'cache_size': 512,
    'max_iter': 2000
}

if best_kernel == 'poly':
    final_model_params['degree'] = int(best_degree)

print("\nTraining final model with best parameters...")
final_model = SVR(**final_model_params)
final_model.fit(X_train_np, y_train_full)

# Evaluate on test set
print("\nEvaluating on test set...")
y_pred_test = final_model.predict(X_test_np)

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
    'C': float(best_C),
    'gamma': best_gamma if isinstance(best_gamma, str) else float(best_gamma),
    'kernel': best_kernel,
    'degree': int(best_degree) if best_kernel == 'poly' else None,
    'epsilon': float(best_epsilon),
    'cv_rmse': float(result.fun),
    'test_rmse': float(test_rmse),
    'test_mse': float(test_mse),
    'test_mae': float(test_mae),
    'test_r2': float(test_r2)
}

pd.DataFrame([test_metrics]).to_csv('bayes_svm_final_model_metrics.csv', index=False)
print("\n[OK] Test metrics saved to bayes_svm_final_model_metrics.csv")

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
             marker='o', alpha=0.6, linestyle='-', linewidth=1, markersize=4)
    ax1.axhline(y=result.fun, color='r', linestyle='--', linewidth=2, label=f'Best: {result.fun:.4f}')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('CV RMSE')
    ax1.set_title('Bayesian Optimization Convergence')
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

# 3. Parameter exploration - C vs RMSE
ax3 = axes[1, 0]
if len(successful_history) > 0:
    scatter = ax3.scatter(successful_history['C'], successful_history['cv_rmse'],
                         c=successful_history['iteration'], cmap='viridis',
                         s=50, alpha=0.7, edgecolors='k', linewidths=0.5)
    ax3.set_xlabel('C (log scale)')
    ax3.set_ylabel('CV RMSE')
    ax3.set_xscale('log')
    ax3.set_title('Parameter Exploration: C vs RMSE')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Iteration')

# 4. Kernel performance comparison
ax4 = axes[1, 1]
if len(successful_history) > 0:
    kernel_stats = successful_history.groupby('kernel')['cv_rmse'].agg(['mean', 'min', 'count'])
    if len(kernel_stats) > 0:
        kernel_names = kernel_stats.index.tolist()
        means = kernel_stats['mean'].values
        mins = kernel_stats['min'].values
        counts = kernel_stats['count'].values
        
        x_pos = np.arange(len(kernel_names))
        bars = ax4.bar(x_pos, means, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.scatter(x_pos, mins, color='red', s=100, zorder=5, label='Best in kernel')
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'n={int(count)}', ha='center', va='bottom', fontsize=9)
        
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(kernel_names)
        ax4.set_ylabel('CV RMSE')
        ax4.set_title('Performance by Kernel Type')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('bayes_svm_optimization_results.png', dpi=300, bbox_inches='tight')
print("[OK] Visualization saved to bayes_svm_optimization_results.png")

print("\n" + "="*80)
print("BAYESIAN OPTIMIZATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - bayes_svm_optimization_results.csv (full iteration history)")
print("  - best_bayes_svm_hyperparameters.json (best parameters)")
print("  - bayes_svm_final_model_metrics.csv (test set performance)")
print("  - bayes_svm_optimization_results.png (visualizations)")
print("="*80)
