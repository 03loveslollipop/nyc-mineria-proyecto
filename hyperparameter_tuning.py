"""
Hyperparameter Tuning for Neural Network Regression Model
Performs grid search with cross-validation on 100+ configurations
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class Architecture1_Large(nn.Module):
    """Large architecture - Current most complex"""
    def __init__(self, input_dim, activation_fn):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, input_dim),
            nn.Softmax(dim=1)
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, 352),
            nn.BatchNorm1d(352),
            activation_fn(),
            nn.Dropout(0.3),
            nn.Linear(352, 288),
            nn.BatchNorm1d(288),
            activation_fn(),
            nn.Dropout(0.25),
            nn.Linear(288, 224),
            nn.BatchNorm1d(224),
            activation_fn(),
            nn.Dropout(0.2),
            nn.Linear(224, 128),
            nn.BatchNorm1d(128),
            activation_fn(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            activation_fn(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        x_attended = x * attention_weights
        return self.network(x_attended)


class Architecture2_Medium(nn.Module):
    """Medium architecture - Reduced depth"""
    def __init__(self, input_dim, activation_fn):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            activation_fn(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            activation_fn(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            activation_fn(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class Architecture3_Small(nn.Module):
    """Small architecture - Fewer layers"""
    def __init__(self, input_dim, activation_fn):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            activation_fn(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            activation_fn(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class Architecture4_Wide(nn.Module):
    """Wide architecture - Fewer layers but wider"""
    def __init__(self, input_dim, activation_fn):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            activation_fn(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            activation_fn(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(X_train, X_test, y_train, y_test, categorical_cols):
    """Prepare data with scaling and encoding"""
    X_train_prep = X_train.copy()
    X_test_prep = X_test.copy()
    
    # Identify numeric columns
    numeric_cols_all = [col for col in X_train_prep.columns if col not in categorical_cols]
    
    # Separate columns for different scaling
    std_scaler_cols = ['mta_tax', 'extra']
    power_transform_cols = [col for col in numeric_cols_all if col not in std_scaler_cols]
    
    # Apply StandardScaler
    scaler_std = StandardScaler()
    X_train_prep[std_scaler_cols] = scaler_std.fit_transform(X_train_prep[std_scaler_cols])
    X_test_prep[std_scaler_cols] = scaler_std.transform(X_test_prep[std_scaler_cols])
    
    # Apply PowerTransformer
    scaler_power = PowerTransformer()
    X_train_prep[power_transform_cols] = scaler_power.fit_transform(X_train_prep[power_transform_cols])
    X_test_prep[power_transform_cols] = scaler_power.transform(X_test_prep[power_transform_cols])
    
    # Convert categorical to dummies
    for col in categorical_cols:
        X_train_prep[col] = X_train_prep[col].astype('category')
        X_test_prep[col] = X_test_prep[col].astype('category')
    
    X_train_prep = pd.get_dummies(X_train_prep, columns=categorical_cols, drop_first=True)
    X_test_prep = pd.get_dummies(X_test_prep, columns=categorical_cols, drop_first=True)
    
    # Align columns
    X_train_prep, X_test_prep = X_train_prep.align(X_test_prep, join='left', axis=1, fill_value=0)
    
    # Convert to float
    X_train_prep = X_train_prep.astype('float32')
    X_test_prep = X_test_prep.astype('float32')
    
    return X_train_prep.values, X_test_prep.values


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, 
                       epochs=100, patience=10, device='cpu'):
    """Train model with early stopping and return best validation loss"""
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training - accumulate loss on GPU
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Data already on GPU from cached tensors
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # Accumulate as tensor on GPU (no sync)
            epoch_loss += loss.detach()
        
        # Only sync once per epoch
        train_loss = (epoch_loss / len(train_loader)).item()
        train_losses.append(train_loss)
        
        # Validation - accumulate loss on GPU
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Data already on GPU from cached tensors
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                # Accumulate as tensor on GPU (no sync)
                epoch_val_loss += loss.detach()
        
        # Only sync once per epoch
        val_loss = (epoch_val_loss / len(val_loader)).item()
        val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    
    return best_val_loss, train_losses, val_losses


def cross_validate_config(X, y, config, categorical_cols, cached_gpu_data=None, device='cpu'):
    """Perform K-fold cross-validation keeping all data on GPU"""
    
    # Use cached GPU data
    X_gpu, y_gpu, fold_indices, input_dim = cached_gpu_data
    
    # Perform K-fold CV using pre-computed fold indices
    fold_scores = []
    n_folds = len(fold_indices)
    
    # Larger batch size for tabular data on GPU
    batch_size = 65536
    
    for fold_idx in range(n_folds):
        # Get train and val indices for this fold
        train_mask = fold_indices[fold_idx]['train']
        val_mask = fold_indices[fold_idx]['val']
        
        # Use boolean indexing to create views (no data copying!)
        X_train_fold = X_gpu[train_mask]
        y_train_fold = y_gpu[train_mask]
        X_val_fold = X_gpu[val_mask]
        y_val_fold = y_gpu[val_mask]
        
        # Manual batching on GPU (no DataLoader overhead)
        n_train = X_train_fold.shape[0]
        n_val = X_val_fold.shape[0]
        
        # Initialize model
        model = config['architecture'](input_dim, config['activation']).to(device)
        
        # Initialize optimizer
        if config['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)
        elif config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        
        criterion = nn.HuberLoss(delta=1.0)
        
        # Train model with manual GPU batching
        best_val_loss = train_with_gpu_batching(
            model, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
            criterion, optimizer, batch_size=batch_size,
            epochs=100, patience=10, device=device
        )
        
        fold_scores.append(best_val_loss)
        
        # Clean up model to free GPU memory (no empty_cache - let PyTorch manage)
        del model, optimizer
    
    return np.mean(fold_scores), np.std(fold_scores)


def train_with_gpu_batching(model, X_train, y_train, X_val, y_val, criterion, optimizer,
                           batch_size=2048, epochs=100, patience=10, device='cpu'):
    """Train model with manual GPU batching (no DataLoader overhead)"""
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_batches = (n_train + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        # Training - manual batching on GPU
        model.train()
        epoch_loss = 0.0
        
        # Shuffle indices on GPU
        train_indices = torch.randperm(n_train, device=device)
        
        for i in range(0, n_train, batch_size):
            # Get batch indices (all on GPU)
            batch_idx = train_indices[i:min(i + batch_size, n_train)]
            batch_X = X_train[batch_idx]
            batch_y = y_train[batch_idx]
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate as tensor on GPU (no sync)
            epoch_loss += loss.detach()
        
        # Validation - no shuffling needed
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                batch_X = X_val[i:min(i + batch_size, n_val)]
                batch_y = y_val[i:min(i + batch_size, n_val)]
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Accumulate as tensor on GPU (no sync)
                epoch_val_loss += loss.detach()
        
        # Only sync once per epoch
        val_loss = (epoch_val_loss * batch_size / n_val).item()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    
    return best_val_loss


# ============================================================================
# MAIN GRID SEARCH
# ============================================================================

def main():
    print("="*80)
    print("NEURAL NETWORK HYPERPARAMETER TUNING WITH GRID SEARCH")
    print("="*80)
    
    # Load training data (70%)
    print("\nLoading training data...")
    train_df = pd.read_parquet('train_70.parquet')
    y_train_full = train_df['trip_duration'].values
    X_train_full = train_df.drop('trip_duration', axis=1)
    
    print(f"Training data shape: {X_train_full.shape}")
    print(f"Target shape: {y_train_full.shape}")
    
    # Categorical columns
    categorical_cols = ['payment_type', 'dropoff_borough', 'pickup_borough']
    
    # Define hyperparameter grid
    architectures = {
        'Large': Architecture1_Large,
        'Medium': Architecture2_Medium,
        'Small': Architecture3_Small,
        'Wide': Architecture4_Wide
    }
    
    learning_rates = [0.001, 0.0001, 0.01]
    optimizers = ['AdamW', 'SGD', 'Adam']
    activations = {
        'ReLU': nn.ReLU,
        'GELU': nn.GELU,
        'Sigmoid': nn.Sigmoid
    }
    
    # Generate all combinations
    configs = []
    config_id = 0
    for arch_name, arch_class in architectures.items():
        for lr in learning_rates:
            for opt in optimizers:
                for act_name, act_fn in activations.items():
                    configs.append({
                        'config_id': config_id,
                        'architecture_name': arch_name,
                        'architecture': arch_class,
                        'lr': lr,
                        'optimizer': opt,
                        'activation_name': act_name,
                        'activation': act_fn
                    })
                    config_id += 1
    
    print(f"\nTotal configurations to test: {len(configs)}")
    print("="*80)
    
    # Prepare and cache ALL data on GPU once (to avoid repeated preprocessing)
    print("\nPreparing and caching data on GPU...")
    
    # Prepare ALL data using dummy test set to maintain consistency
    # Use the same data for both train and test to get proper preprocessing
    X_all_prep, _ = prepare_data(
        X_train_full, X_train_full.iloc[:100], y_train_full, y_train_full[:100], categorical_cols
    )
    
    input_dim = X_all_prep.shape[1]
    print(f"Preprocessed feature dimension: {input_dim}")
    
    # Move ALL data to GPU as single tensor
    X_all_gpu = torch.FloatTensor(X_all_prep).to(device)
    y_all_gpu = torch.FloatTensor(y_train_full).unsqueeze(1).to(device)
    
    print(f"All data on GPU: {X_all_gpu.shape}")
    
    # Create K-fold indices (3 folds)
    n_folds = 3
    n_samples = len(X_all_gpu)
    indices = torch.randperm(n_samples, device=device)  # Shuffle on GPU
    fold_size = n_samples // n_folds
    
    fold_indices = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
        
        # Create boolean masks on GPU
        val_mask = torch.zeros(n_samples, dtype=torch.bool, device=device)
        val_mask[indices[val_start:val_end]] = True
        train_mask = ~val_mask
        
        fold_indices.append({
            'train': train_mask,
            'val': val_mask
        })
        
        print(f"Fold {fold+1}: Train={train_mask.sum().item()}, Val={val_mask.sum().item()}")
    
    cached_gpu_data = (X_all_gpu, y_all_gpu, fold_indices, input_dim)
    print("="*80)
    
    # Run grid search with cross-validation
    results = []
    start_time = time.time()
    
    for i, config in enumerate(configs):
        config_start = time.time()
        print(f"\n[{i+1}/{len(configs)}] Testing configuration:")
        print(f"  Architecture: {config['architecture_name']}")
        print(f"  Learning Rate: {config['lr']}")
        print(f"  Optimizer: {config['optimizer']}")
        print(f"  Activation: {config['activation_name']}")
        
        try:
            mean_val_loss, std_val_loss = cross_validate_config(
                X_train_full, y_train_full, config, categorical_cols,
                cached_gpu_data=cached_gpu_data, device=device
            )
            
            # Calculate approximate metrics from validation loss (Huber loss)
            # These are approximations for ranking purposes
            mse_approx = mean_val_loss  # Huber loss approximates MSE
            rmse_approx = np.sqrt(mse_approx)
            
            results.append({
                'config_id': config['config_id'],
                'architecture': config['architecture_name'],
                'learning_rate': config['lr'],
                'optimizer': config['optimizer'],
                'activation': config['activation_name'],
                'mean_val_loss': mean_val_loss,
                'std_val_loss': std_val_loss,
                'mse_approx': mse_approx,
                'rmse_approx': rmse_approx,
                'status': 'success'
            })
            
            config_time = time.time() - config_start
            print(f"  [OK] Mean Validation Loss: {mean_val_loss:.4f} +/- {std_val_loss:.4f}")
            print(f"  Time: {config_time:.2f}s")
            
        except Exception as e:
            print(f"  [ERROR] Error: {str(e)}")
            results.append({
                'config_id': config['config_id'],
                'architecture': config['architecture_name'],
                'learning_rate': config['lr'],
                'optimizer': config['optimizer'],
                'activation': config['activation_name'],
                'mean_val_loss': np.nan,
                'std_val_loss': np.nan,
                'mse_approx': np.nan,
                'rmse_approx': np.nan,
                'status': 'failed'
            })
    
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"Grid search completed in {total_time/60:.2f} minutes")
    print("="*80)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mse_approx')
    results_df.to_csv('hyperparameter_search_results.csv', index=False)
    print("\n[OK] Results saved to hyperparameter_search_results.csv")
    
    # Display top 10 configurations
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS (by MSE)")
    print("="*80)
    print(results_df[results_df['status'] == 'success'].head(10).to_string(index=False))
    
    # Train final model with best hyperparameters
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("="*80)
    
    best_config_row = results_df[results_df['status'] == 'success'].iloc[0]
    print("\nBest configuration:")
    print(f"  Architecture: {best_config_row['architecture']}")
    print(f"  Learning Rate: {best_config_row['learning_rate']}")
    print(f"  Optimizer: {best_config_row['optimizer']}")
    print(f"  Activation: {best_config_row['activation']}")
    print(f"  CV MSE: {best_config_row['mse_approx']:.4f}")
    
    # Train final model on full 70% training set
    print("\nTraining on full 70% training set...")
    train_final_model(best_config_row, X_train_full, y_train_full, categorical_cols, device)


def train_final_model(best_config, X_train_full, y_train_full, categorical_cols, device):
    """Train final model with best hyperparameters on full training set"""
    
    # Load test data (30%)
    test_df = pd.read_parquet('test_30.parquet')
    y_test = test_df['trip_duration'].values
    X_test = test_df.drop('trip_duration', axis=1)
    
    # Prepare data
    X_train_prep, X_test_prep = prepare_data(
        X_train_full, X_test, y_train_full, y_test, categorical_cols
    )
    
    # Split training data for train/val
    val_size = int(0.15 * len(X_train_prep))
    indices = np.random.permutation(len(X_train_prep))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    X_tr = X_train_prep[train_indices]
    y_tr = y_train_full[train_indices]
    X_val = X_train_prep[val_indices]
    y_val = y_train_full[val_indices]
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_tr).to(device)
    y_train_tensor = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test_prep).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Initialize model with best config
    input_dim = X_train_prep.shape[1]
    
    # Map architecture name to class
    arch_map = {
        'Large': Architecture1_Large,
        'Medium': Architecture2_Medium,
        'Small': Architecture3_Small,
        'Wide': Architecture4_Wide
    }
    
    act_map = {
        'ReLU': nn.ReLU,
        'GELU': nn.GELU,
        'Sigmoid': nn.Sigmoid
    }
    
    model = arch_map[best_config['architecture']](
        input_dim, 
        act_map[best_config['activation']]
    ).to(device)
    
    # Initialize optimizer
    lr = best_config['learning_rate']
    if best_config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif best_config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif best_config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Train model
    epochs = 500
    patience = 30
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    print(f"\nTraining for up to {epochs} epochs with patience {patience}...")
    
    for epoch in range(epochs):
        # Training - accumulate loss on GPU
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # Accumulate as tensor on GPU (no sync)
            epoch_loss += loss.detach()
        
        # Only sync once per epoch
        train_loss = (epoch_loss / len(train_loader)).item()
        train_losses.append(train_loss)
        
        # Validation - accumulate loss on GPU
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                # Accumulate as tensor on GPU (no sync)
                epoch_val_loss += loss.detach()
        
        # Only sync once per epoch
        val_loss = (epoch_val_loss / len(val_loader)).item()
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate on test set
    print("\nEvaluating on test set (30%)...")
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor).squeeze().cpu().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    print("\n" + "="*80)
    print("FINAL MODEL PERFORMANCE ON TEST SET (30%)")
    print("="*80)
    print(f"RMSE:  {rmse:.4f}")
    print(f"MSE:   {mse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print("="*80)
    
    # Save final metrics
    final_metrics = {
        'architecture': best_config['architecture'],
        'learning_rate': best_config['learning_rate'],
        'optimizer': best_config['optimizer'],
        'activation': best_config['activation'],
        'test_rmse': rmse,
        'test_mse': mse,
        'test_mae': mae,
        'test_r2': r2,
        'epochs_trained': len(train_losses),
        'best_epoch': len(train_losses) - patience
    }
    
    pd.DataFrame([final_metrics]).to_csv('final_model_metrics.csv', index=False)
    print("\n[OK] Final metrics saved to final_model_metrics.csv")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Huber)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Trip Duration (minutes)')
    plt.ylabel('Predicted Trip Duration (minutes)')
    plt.title('Predictions vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add metrics text
    textstr = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR^2: {r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('final_model_results.png', dpi=300, bbox_inches='tight')
    print("[OK] Training plots saved to final_model_results.png")
    
    plt.show()


if __name__ == "__main__":
    main()
