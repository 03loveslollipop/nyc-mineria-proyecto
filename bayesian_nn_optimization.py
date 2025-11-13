"""
Bayesian Optimization for Neural Network Hyperparameters (learning rate, optimizer, activation)
Uses scikit-optimize to search within +/-30% of best previous learning rate.
Architecture is kept fixed to the best-performing one from prior grid search.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args

# Reuse architectures from hyperparameter_tuning
from hyperparameter_tuning import (
    Architecture1_Large,
    Architecture2_Medium,
    Architecture3_Small,
    Architecture4_Wide,
    prepare_data,
    train_with_gpu_batching,
)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------------
# Sklearn-style PyTorch regressor (fixed architecture, variable lr/optimizer/activation)
# ----------------------------------------------------------------------------
class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim:int, architecture_name:str, learning_rate:float=1e-3,
                 optimizer_name:str='AdamW', activation_name:str='ReLU',
                 epochs:int=150, batch_size:int=65536, patience:int=15, huber_delta:float=1.0,
                 seed:int=42):
        self.input_dim = input_dim
        self.architecture_name = architecture_name
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.activation_name = activation_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.huber_delta = huber_delta
        self.seed = seed
        self._model = None

    def _activation(self):
        return {'ReLU': nn.ReLU, 'GELU': nn.GELU, 'Sigmoid': nn.Sigmoid}[self.activation_name]

    def _architecture_cls(self):
        return {
            'Large': Architecture1_Large,
            'Medium': Architecture2_Medium,
            'Small': Architecture3_Small,
            'Wide': Architecture4_Wide
        }[self.architecture_name]

    def _build_model(self):
        torch.manual_seed(self.seed)
        model_cls = self._architecture_cls()
        self._model = model_cls(self.input_dim, self._activation()).to(DEVICE)
        return self._model

    def _build_optimizer(self):
        if self.optimizer_name == 'AdamW':
            return optim.AdamW(self._model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        if self.optimizer_name == 'Adam':
            return optim.Adam(self._model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        if self.optimizer_name == 'SGD':
            return optim.SGD(self._model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
        raise ValueError(f"Unsupported optimizer {self.optimizer_name}")

    def fit(self, X, y):
        X_t = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=DEVICE).view(-1, 1)
        self._build_model()
        optimizer = self._build_optimizer()
        criterion = nn.HuberLoss(delta=self.huber_delta)

        n = X_t.shape[0]
        best_val = math.inf
        epochs_no_improve = 0

        # Simple train/val split
        val_size = int(0.15 * n)
        idx = torch.randperm(n, device=DEVICE)
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]
        X_train = X_t[train_idx]
        y_train = y_t[train_idx]
        X_val = X_t[val_idx]
        y_val = y_t[val_idx]

        for epoch in range(self.epochs):
            # Shuffle training indices each epoch
            perm = torch.randperm(X_train.shape[0], device=DEVICE)
            X_train_epoch = X_train[perm]
            y_train_epoch = y_train[perm]

            # Manual batching
            for start in range(0, X_train_epoch.shape[0], self.batch_size):
                end = start + self.batch_size
                batch_X = X_train_epoch[start:end]
                batch_y = y_train_epoch[start:end]
                optimizer.zero_grad(set_to_none=True)
                out = self._model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_out = self._model(X_val)
                val_loss = criterion(val_out, y_val).item()
            self._model.train()

            if val_loss < best_val:
                best_val = val_loss
                epochs_no_improve = 0
                # Keep best weights
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        # Restore best state
        self._model.load_state_dict(best_state)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("Model not fitted")
        X_t = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(X_t).squeeze().cpu().numpy()
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        mse = mean_squared_error(y, preds)
        return -mse  # For minimization

    def get_params(self, deep=True):
        return {
            'input_dim': self.input_dim,
            'architecture_name': self.architecture_name,
            'learning_rate': self.learning_rate,
            'optimizer_name': self.optimizer_name,
            'activation_name': self.activation_name,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'patience': self.patience,
            'huber_delta': self.huber_delta,
            'seed': self.seed,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

# ----------------------------------------------------------------------------
# Helper to load data and compute input dimension
# ----------------------------------------------------------------------------

def load_dataset():
    train_df = pd.read_parquet('train_70.parquet')
    y = train_df['trip_duration'].values
    X = train_df.drop('trip_duration', axis=1)
    categorical_cols = ['payment_type', 'dropoff_borough', 'pickup_borough']
    X_prep, _ = prepare_data(X, X.iloc[:100], y, y[:100], categorical_cols)
    return X_prep, y, X_prep.shape[1]

# ----------------------------------------------------------------------------
# Main Bayesian optimization routine
# ----------------------------------------------------------------------------

def main():
    print('='*80)
    print('Bayesian Optimization - NN Hyperparameters (lr, optimizer, activation)')
    print('='*80)

    # Load previous grid search results to get best architecture & learning rate
    prev = pd.read_csv('hyperparameter_search_results.csv')
    prev_success = prev[prev['status'] == 'success'].sort_values('mse_approx')
    best_row = prev_success.iloc[0]
    base_lr = float(best_row['learning_rate'])
    architecture_name = best_row['architecture']
    print(f"Best previous config -> Architecture: {architecture_name} | LR: {base_lr}")

    # Define search bounds (+/-30%)
    lr_low = base_lr * 0.7
    lr_high = base_lr * 1.3
    # Ensure bounds remain positive and not degenerate
    lr_low = max(lr_low, 1e-6)

    print(f"Learning rate search space: [{lr_low:.6g}, {lr_high:.6g}] (log-uniform)")

    # Load dataset & input dimension
    X, y, input_dim = load_dataset()
    print(f"Dataset shape: {X.shape}, Input dim: {input_dim}")

    # Cache ALL data on GPU once, and create 3-fold boolean masks
    X_all_gpu = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    y_all_gpu = torch.as_tensor(y, dtype=torch.float32, device=DEVICE).view(-1, 1)
    n_folds = 3
    n_samples = X_all_gpu.shape[0]
    indices = torch.randperm(n_samples, device=DEVICE)
    fold_size = n_samples // n_folds
    fold_indices = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
        val_mask = torch.zeros(n_samples, dtype=torch.bool, device=DEVICE)
        val_mask[indices[val_start:val_end]] = True
        train_mask = ~val_mask
        fold_indices.append({'train': train_mask, 'val': val_mask})

    # Define search space
    space = [
        Real(lr_low, lr_high, prior='log-uniform', name='learning_rate'),
        Categorical(['AdamW', 'Adam', 'SGD'], name='optimizer_name'),
        Categorical(['ReLU', 'GELU', 'Sigmoid'], name='activation_name')
    ]

    # Objective function using GPU K-fold CV and manual batching
    act_map = {'ReLU': nn.ReLU, 'GELU': nn.GELU, 'Sigmoid': nn.Sigmoid}

    @use_named_args(space)
    def objective(**params):
        lr = float(params['learning_rate'])
        opt_name = params['optimizer_name']
        act_name = params['activation_name']

        # CV loop
        fold_losses = []
        batch_size = 65536
        for masks in fold_indices:
            X_tr = X_all_gpu[masks['train']]
            y_tr = y_all_gpu[masks['train']]
            X_va = X_all_gpu[masks['val']]
            y_va = y_all_gpu[masks['val']]

            # Build model per fold
            model = {
                'Large': Architecture1_Large,
                'Medium': Architecture2_Medium,
                'Small': Architecture3_Small,
                'Wide': Architecture4_Wide,
            }[architecture_name](input_dim, act_map[act_name]).to(DEVICE)

            # Optimizer
            if opt_name == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            elif opt_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            elif opt_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            else:
                raise ValueError(f'Unsupported optimizer {opt_name}')

            criterion = nn.HuberLoss(delta=1.0)
            val_loss = train_with_gpu_batching(
                model, X_tr, y_tr, X_va, y_va,
                criterion, optimizer,
                batch_size=batch_size, epochs=100, patience=10, device=DEVICE
            )
            fold_losses.append(val_loss)

            # Cleanup
            del model, optimizer

        mse = float(np.mean(fold_losses))
        print(f"Tested params {{'learning_rate': {lr:.6g}, 'optimizer': {opt_name}, 'activation': {act_name}}} -> CV MSE {mse:.5f}")
        return mse

    # Run Bayesian optimization
    start_time = time.time()
    res = gp_minimize(objective, dimensions=space, n_calls=100, random_state=42,
                      n_initial_points=10, acq_func='EI')
    total_time = time.time() - start_time

    print(f"Optimization completed in {total_time/60:.2f} minutes")
    print(f"Best MSE: {res.fun:.6f}")

    best_params = {
        'learning_rate': res.x[0],
        'optimizer_name': res.x[1],
        'activation_name': res.x[2],
        'architecture_name': architecture_name,
        'best_mse': res.fun
    }
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Persist results
    # Persist detailed optimization history (params + score)
    history_rows = []
    for i, (x, f) in enumerate(zip(res.x_iters, res.func_vals)):
        history_rows.append({
            'iteration': i,
            'learning_rate': x[0],
            'optimizer_name': x[1],
            'activation_name': x[2],
            'mse': f,
        })
    pd.DataFrame(history_rows).to_csv('bayes_optimization_results.csv', index=False)

    with open('best_bayes_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    # Train final model on full data with best params
    final_regressor = PyTorchRegressor(
        input_dim=input_dim,
        architecture_name=architecture_name,
        learning_rate=best_params['learning_rate'],
        optimizer_name=best_params['optimizer_name'],
        activation_name=best_params['activation_name'],
        epochs=300,
        patience=30,
        batch_size=65536
    )
    final_regressor.fit(X, y)

    # Save model weights
    torch.save(final_regressor._model.state_dict(), 'best_bayes_model.pth')
    print('[OK] Saved best Bayesian optimized model weights to best_bayes_model.pth')

    # Simple convergence plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.plot(res.func_vals, marker='o', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.title('Bayesian Optimization Convergence')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('bayes_opt_history.png', dpi=300)
        print('[OK] Saved convergence plot to bayes_opt_history.png')
    except Exception as e:
        print(f'[WARN] Could not generate plot: {e}')

    # Optional: Train a fresh final model using consistent train/test preprocessing and evaluate on test set
    try:
        from torch.utils.data import TensorDataset, DataLoader
        import matplotlib.pyplot as plt

        print('\nTraining and evaluating final model on held-out test set (30%)...')
        train_df = pd.read_parquet('train_70.parquet')
        test_df = pd.read_parquet('test_30.parquet')
        y_train_full = train_df['trip_duration'].values
        X_train_full = train_df.drop('trip_duration', axis=1)
        y_test = test_df['trip_duration'].values
        X_test = test_df.drop('trip_duration', axis=1)
        categorical_cols = ['payment_type', 'dropoff_borough', 'pickup_borough']

        # Preprocess with train-fitted transformers applied to test
        X_tr_prep, X_te_prep = prepare_data(X_train_full, X_test, y_train_full, y_test, categorical_cols)

        # Train/val split
        val_size = int(0.15 * len(X_tr_prep))
        idx = np.random.permutation(len(X_tr_prep))
        tr_idx, va_idx = idx[val_size:], idx[:val_size]

        X_tr = torch.as_tensor(X_tr_prep[tr_idx], dtype=torch.float32, device=DEVICE)
        y_tr = torch.as_tensor(y_train_full[tr_idx], dtype=torch.float32, device=DEVICE).view(-1,1)
        X_va = torch.as_tensor(X_tr_prep[va_idx], dtype=torch.float32, device=DEVICE)
        y_va = torch.as_tensor(y_train_full[va_idx], dtype=torch.float32, device=DEVICE).view(-1,1)
        X_te = torch.as_tensor(X_te_prep, dtype=torch.float32, device=DEVICE)
        y_te = torch.as_tensor(y_test, dtype=torch.float32, device=DEVICE).view(-1,1)

        input_dim_final = X_tr.shape[1]
        act_map = {'ReLU': nn.ReLU, 'GELU': nn.GELU, 'Sigmoid': nn.Sigmoid}
        model = {
            'Large': Architecture1_Large,
            'Medium': Architecture2_Medium,
            'Small': Architecture3_Small,
            'Wide': Architecture4_Wide,
        }[architecture_name](input_dim_final, act_map[best_params['activation_name']]).to(DEVICE)

        # Optimizer
        lr = float(best_params['learning_rate'])
        opt_name = best_params['optimizer_name']
        if opt_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        elif opt_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        criterion = nn.HuberLoss(delta=1.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

        # Dataloaders
        train_ds = TensorDataset(X_tr, y_tr)
        val_ds = TensorDataset(X_va, y_va)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

        best_val = math.inf
        epochs_no_improve = 0
        train_losses, val_losses = [], []
        for epoch in range(500):
            model.train()
            ep_loss = 0.0
            for bx, by in train_loader:
                optimizer.zero_grad(set_to_none=True)
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ep_loss += loss.detach()
            train_losses.append((ep_loss/len(train_loader)).item())

            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for bx, by in val_loader:
                    val_loss += criterion(model(bx), by).detach()
            val_loss = (val_loss/len(val_loader)).item()
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), 'best_bayes_model.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= 30:
                    break

        # Load best and evaluate on test
        model.load_state_dict(torch.load('best_bayes_model.pth', map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_te).squeeze().cpu().numpy()

        mse = mean_squared_error(y_test, y_pred_test)
        rmse = float(np.sqrt(mse))
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = float(mean_absolute_error(y_test, y_pred_test))
        r2 = float(r2_score(y_test, y_pred_test))

        pd.DataFrame([{
            'architecture': architecture_name,
            'learning_rate': best_params['learning_rate'],
            'optimizer': best_params['optimizer_name'],
            'activation': best_params['activation_name'],
            'test_rmse': rmse,
            'test_mse': mse,
            'test_mae': mae,
            'test_r2': r2,
        }]).to_csv('bayes_final_model_metrics.csv', index=False)
        print('[OK] Saved test metrics to bayes_final_model_metrics.csv')

        # Plots
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend(); plt.title('Training History (Huber)'); plt.grid(alpha=0.3)

        plt.subplot(1,2,2)
        plt.scatter(y_test, y_pred_test, s=12, alpha=0.5, edgecolors='k', linewidths=0.2)
        mn, mx = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
        plt.plot([mn, mx], [mn, mx], 'r--', lw=1)
        plt.title(f'Pred vs Actual\nRMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('bayes_final_model_results.png', dpi=300)
        print('[OK] Saved test plots to bayes_final_model_results.png')
    except Exception as e:
        print(f"[WARN] Skipped final test evaluation due to: {e}")

if __name__ == '__main__':
    main()
