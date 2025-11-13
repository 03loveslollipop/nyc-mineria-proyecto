"""
Sklearn-compatible Neural Network Wrapper for NYC Taxi Trip Duration Prediction
Wraps the winning Medium architecture (AdamW, ReLU, lr=0.001) as an sklearn estimator
"""

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin


class MediumArchitecture(nn.Module):
    """Medium architecture - Best performing model"""
    def __init__(self, input_dim, activation_fn=nn.ReLU):
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


class NeuralNetworkRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for PyTorch Neural Network
    
    Best configuration from hyperparameter search:
    - Architecture: Medium
    - Learning rate: 0.001
    - Optimizer: AdamW
    - Activation: ReLU
    """
    
    def __init__(self, input_dim=None, learning_rate=0.001, epochs=500, 
                 patience=30, batch_size=128, random_state=42, verbose=True):
        """
        Initialize Neural Network Regressor
        
        Parameters:
        -----------
        input_dim : int
            Number of input features (will be set during fit if None)
        learning_rate : float
            Learning rate for AdamW optimizer
        epochs : int
            Maximum number of training epochs
        patience : int
            Early stopping patience
        batch_size : int
            Batch size for training
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Whether to print training progress
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        
        # Model components (initialized during fit)
        self.model_ = None
        self.device_ = None
        self.best_loss_ = None
        self.training_history_ = None
    
    def _initialize_model(self):
        """Initialize the neural network model"""
        torch.manual_seed(self.random_state)
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_ = MediumArchitecture(self.input_dim, activation_fn=nn.ReLU).to(self.device_)
        return self.model_
    
    def fit(self, X, y):
        """
        Fit the neural network to the training data
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : object
            Fitted estimator
        """
        # Convert to numpy arrays if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Set input dimension
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        # Initialize model
        self._initialize_model()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device_)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device_)
        
        # Split into train/validation
        val_size = int(0.15 * len(X_tensor))
        indices = torch.randperm(len(X_tensor))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train = X_tensor[train_indices]
        y_train = y_tensor[train_indices]
        X_val = X_tensor[val_indices]
        y_val = y_tensor[val_indices]
        
        # Create data loaders
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize optimizer and criterion
        optimizer = optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=1.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        
        # Training loop
        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training phase
            self.model_.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.detach()
            
            train_loss = (epoch_loss / len(train_loader)).item()
            train_losses.append(train_loss)
            
            # Validation phase
            self.model_.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model_(batch_X)
                    loss = criterion(outputs, batch_y)
                    epoch_val_loss += loss.detach()
            
            val_loss = (epoch_val_loss / len(val_loader)).item()
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model state
                self.best_state_dict_ = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if self.verbose and (epoch + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Load best model
        self.model_.load_state_dict(self.best_state_dict_)
        self.best_loss_ = best_val_loss
        self.training_history_ = {'train_loss': train_losses, 'val_loss': val_losses}
        self.fitted_ = True
        
        if self.verbose:
            print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        
        return self

    def load_pretrained(self, state_dict_path, input_dim=None, device='cpu'):
        """
        Load pretrained weights from disk without running fit.
        """
        if input_dim is not None:
            self.input_dim = input_dim
        if self.input_dim is None:
            raise ValueError("input_dim must be provided to load pretrained weights.")
        if isinstance(device, str):
            device = torch.device(device)
        self.device_ = device
        torch.manual_seed(self.random_state)
        self.model_ = MediumArchitecture(self.input_dim, activation_fn=nn.ReLU).to(self.device_)
        state_dict = torch.load(state_dict_path, map_location=self.device_)
        self.model_.load_state_dict(state_dict)
        self.best_state_dict_ = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
        self.training_history_ = None
        self.best_loss_ = None
        self.fitted_ = True
        self.model_.eval()
        if self.verbose:
            print(f"Loaded pretrained weights from {state_dict_path}")
        return self
    
    def predict(self, X):
        """
        Predict using the neural network
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
        
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device_)
        
        # Predict
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor).cpu().numpy()
        
        return outputs.reshape(-1)
    
    def score(self, X, y):
        """
        Return the R² score of the prediction
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values
        
        Returns:
        --------
        score : float
            R² score
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'input_dim': self.input_dim,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'patience': self.patience,
            'batch_size': self.batch_size,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


sys.modules.setdefault('deployment.nn_sklearn_wrapper', sys.modules[__name__])
