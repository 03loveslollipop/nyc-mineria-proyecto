"""
Pipeline Preparation Script
Builds sklearn pipeline with preprocessing and neural network, then exports it for deployment.
Supports loading pretrained PyTorch weights to avoid retraining.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from pipeline_components import DummyEncoder, MixedScaler
from nn_sklearn_wrapper import NeuralNetworkRegressor

# Configuration ----------------------------------------------------------------
USE_PRETRAINED = True
PRETRAINED_MODEL_PATH = (BASE_DIR / 'best_model.pth').resolve()
PRETRAINED_DEVICE = 'cpu'


def main():
    print("=" * 80)
    print("BUILDING NEURAL NETWORK PIPELINE FOR DEPLOYMENT")
    print("=" * 80)

    # ============================================================================
    # LOAD DATA
    # ============================================================================

    print("\nLoading training data (70%)...")
    train_df = pd.read_parquet(BASE_DIR.parent / 'train_70.parquet')
    y_train = train_df['trip_duration'].values
    X_train = train_df.drop('trip_duration', axis=1)

    print(f"Training data shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")

    # Categorical columns (treated as categories for encoding)
    categorical_cols = ['payment_type', 'dropoff_borough', 'pickup_borough']
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {categorical_cols}")

    # ============================================================================
    # BUILD PREPROCESSING PIPELINE
    # ============================================================================

    print("\n" + "=" * 80)
    print("BUILDING PREPROCESSING PIPELINE")
    print("=" * 80)

    # Columns for different scaling strategies
    std_scaler_cols = ['mta_tax', 'extra']
    power_transform_cols = [col for col in numeric_cols if col not in std_scaler_cols]

    print(f"\nStandardScaler columns: {std_scaler_cols}")
    print(f"PowerTransformer columns: {len(power_transform_cols)} columns")

    def build_preprocessing_pipeline():
        return Pipeline([
            ('scaler', MixedScaler(std_scaler_cols, power_transform_cols)),
            ('encoder', DummyEncoder(categorical_cols))
        ])

    # ============================================================================
    # BUILD / LOAD FULL PIPELINE WITH NEURAL NETWORK
    # ============================================================================

    print("\n" + "=" * 80)
    print("CONFIGURING FULL PIPELINE")
    print("=" * 80)

    if USE_PRETRAINED:
        if not PRETRAINED_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Pretrained model file not found: {PRETRAINED_MODEL_PATH}"
            )
        print(f"\n[INFO] Loading pretrained weights from {PRETRAINED_MODEL_PATH}")
        preprocessing_pipeline = build_preprocessing_pipeline()
        preprocessing_pipeline.fit(X_train, y_train)
        encoder = preprocessing_pipeline.named_steps['encoder']
        input_dim = len(encoder.columns_)

        nn_estimator = NeuralNetworkRegressor(
            learning_rate=0.001,
            epochs=1,           # Unused when loading pretrained weights
            patience=1,
            batch_size=128,
            random_state=42,
            verbose=True
        )
        nn_estimator.load_pretrained(
            PRETRAINED_MODEL_PATH,
            input_dim=input_dim,
            device=PRETRAINED_DEVICE
        )

        full_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('model', nn_estimator)
        ])
    else:
        preprocessing_pipeline = build_preprocessing_pipeline()
        nn_estimator = NeuralNetworkRegressor(
            learning_rate=0.001,
            epochs=500,
            patience=30,
            batch_size=128,
            random_state=42,
            verbose=True
        )
        full_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('model', nn_estimator)
        ])

        print("\nTraining pipeline on 70% training data...")
        full_pipeline.fit(X_train, y_train)
        print("[OK] Pipeline training completed!")

    # ============================================================================
    # EVALUATE ON TEST SET
    # ============================================================================

    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET (30%)")
    print("=" * 80)

    # Load test data
    test_df = pd.read_parquet(BASE_DIR.parent / 'test_30.parquet')
    y_test = test_df['trip_duration'].values
    X_test = test_df.drop('trip_duration', axis=1)

    print(f"Test data shape: {X_test.shape}")

    # Make predictions
    y_pred = full_pipeline.predict(X_test)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"\nTest Set Performance:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")

    # ============================================================================
    # EXPORT PIPELINE
    # ============================================================================

    print("\n" + "=" * 80)
    print("EXPORTING PIPELINE")
    print("=" * 80)

    deployment_dir = BASE_DIR

    # Save pipeline
    pipeline_path = deployment_dir / 'nn_pipeline.pkl'
    with open(pipeline_path, 'wb') as f:
        pickle.dump(full_pipeline, f)

    print(f"\n[OK] Pipeline saved to: {pipeline_path}")
    print(f"Pipeline file size: {pipeline_path.stat().st_size / (1024 * 1024):.2f} MB")

    fitted_preprocessing = full_pipeline.named_steps['preprocessing']
    fitted_model = full_pipeline.named_steps['model']
    encoder_columns = list(fitted_preprocessing.named_steps['encoder'].columns_)

    feature_info = {
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'std_scaler_columns': std_scaler_cols,
        'power_transform_columns': power_transform_cols,
        'input_features': X_train.columns.tolist(),
        'encoded_feature_count': len(encoder_columns),
        'encoded_feature_names': encoder_columns,
        'model_architecture': 'Medium',
        'learning_rate': 0.001,
        'optimizer': 'AdamW',
        'activation': 'ReLU',
        'uses_pretrained_weights': USE_PRETRAINED,
        'pretrained_model_path': str(PRETRAINED_MODEL_PATH) if USE_PRETRAINED else None,
        'pretrained_device': PRETRAINED_DEVICE if USE_PRETRAINED else None,
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'input_dimension': int(fitted_model.input_dim),
        'fitting_strategy': 'pretrained_weights' if USE_PRETRAINED else 'trained_from_scratch'
    }

    import json
    feature_info_path = deployment_dir / 'model_info.json'
    with open(feature_info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)

    print(f"[OK] Model info saved to: {feature_info_path}")

    print("\n" + "=" * 80)
    print("PIPELINE PREPARATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files in deployment/ folder:")
    print("  - nn_pipeline.pkl (full trained pipeline)")
    print("  - model_info.json (model configuration and performance)")
    print("=" * 80)


if __name__ == "__main__":
    main()
