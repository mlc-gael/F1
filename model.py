# /f1_predictor/model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
import xgboost as xgb
import lightgbm as lgb
import joblib
import config
import utils
import os

logger = utils.get_logger(__name__)

def get_model():
    """Initializes the selected ML model based on config."""
    model_type = config.MODEL_TYPE.lower()
    if model_type == 'randomforest':
        logger.info(f"Initializing RandomForestRegressor with params: {config.RF_PARAMS}")
        model = RandomForestRegressor(**config.RF_PARAMS)
    elif model_type == 'xgboost':
        logger.info(f"Initializing XGBoost Regressor with params: {config.XGB_PARAMS}")
        model = xgb.XGBRegressor(**config.XGB_PARAMS)
    elif model_type == 'lightgbm':
        logger.info(f"Initializing LightGBM Regressor with params: {config.LGBM_PARAMS}")
        model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    else:
        logger.error(f"Unsupported model type: {config.MODEL_TYPE}. Defaulting to RandomForest.")
        model = RandomForestRegressor(**config.RF_PARAMS)
    return model

def train_model(df_features, feature_cols, target_col):
    """Trains the F1 prediction model using specified features and saves it."""
    logger.info("Starting model training process...")

    if df_features is None or df_features.empty: logger.error("Cannot train: Features DataFrame is empty."); return None
    if not feature_cols: logger.error("Cannot train: No feature columns provided."); return None
    if target_col not in df_features.columns: logger.error(f"Cannot train: Target '{target_col}' not found."); return None

    df_features = df_features.sort_values(by=['Year', 'RoundNumber'])
    logger.info(f"Training data shape: {df_features.shape}")

    X = df_features[feature_cols]
    y = df_features[target_col]

    # --- Data Validation ---
    if X.isnull().values.any():
        logger.warning("NaNs found in features (X) before training. Applying fill value.")
        X = X.fillna(config.FILL_NA_VALUE)
    if y.isnull().values.any():
        logger.warning(f"NaNs found in target ('{target_col}'). Removing rows with invalid targets.")
        valid_indices = y.notna()
        X = X.loc[valid_indices, :]
        y = y.loc[valid_indices]
        logger.info(f"Removed {sum(~valid_indices)} rows. New training shape: {X.shape}")
        if X.empty: logger.error("No valid training data remaining."); return None

    # --- Model Initialization ---
    model = get_model()
    model_type = config.MODEL_TYPE.lower()

    # --- Cross-Validation (Time Series Split) ---
    try:
        logger.info("Performing Time Series Cross-Validation...")
        tscv = TimeSeriesSplit(n_splits=5)
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

        # Prepare fit parameters ONLY if needed (e.g., for LightGBM)
        fit_params = {}
        cv_kwargs = {} # Arguments specifically for cross_val_score
        if model_type == 'lightgbm':
             categorical_feature_names = [col for col in feature_cols if X[col].dtype.name == 'category']
             if categorical_feature_names:
                  # LightGBM needs fit_params during cross_val_score
                  fit_params['categorical_feature'] = categorical_feature_names
                  cv_kwargs['fit_params'] = fit_params # Pass fit_params to cross_val_score
                  logger.info(f"Using categorical features for LGBM CV: {categorical_feature_names}")

        # <<< --- CORRECTED cross_val_score call --- >>>
        # Pass fit_params dict using the 'fit_params' argument if not empty
        scores = cross_val_score(model, X, y, cv=tscv, scoring=mae_scorer, n_jobs=-1, **cv_kwargs)
        # <<< --- END CORRECTION --- >>>

        avg_mae = -np.mean(scores)
        std_mae = np.std(scores)
        logger.info(f"Time Series CV MAE: {avg_mae:.3f} (+/- {std_mae:.3f})")

    except Exception as e:
        logger.error(f"Error during cross-validation: {e}", exc_info=True)
        logger.warning("Proceeding with training on full data despite CV error.")


    # --- Final Model Training on ALL provided data ---
    try:
        logger.info(f"Training final model on {len(X)} samples...")
        # Use same fit_params logic for final fit if needed
        if model_type == 'lightgbm' and 'categorical_feature' in fit_params:
             model.fit(X, y, **fit_params)
        else:
             model.fit(X, y)
        logger.info("Final model training complete.")

        # --- Save Model ---
        try:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            joblib.dump(model, config.MODEL_PATH)
            logger.info(f"Model saved successfully to: {config.MODEL_PATH}")
        except Exception as e:
             logger.error(f"Error saving model to {config.MODEL_PATH}: {e}", exc_info=True)

        # --- Feature Importances ---
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                # Ensure feature_cols matches the number of importances
                if len(feature_cols) == len(importances):
                    feature_importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
                    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
                    logger.info("--- Feature Importances (Top 15) ---")
                    logger.info(f"\n{feature_importance_df.head(15).to_string(index=False)}")
                else:
                    logger.warning(f"Mismatch between feature count ({len(feature_cols)}) and importance count ({len(importances)}). Skipping importance display.")
            elif hasattr(model, 'coef_'): # For linear models
                 logger.info("Model has coefficients (linear model?), not feature importances.")
                 # Potentially display coefficients if useful
            else:
                 logger.info("Model type does not support feature_importances_ or coef_ attribute.")
        except Exception as e:
            logger.warning(f"Could not display feature importances/coefficients: {e}")

        return model # Return the trained model object

    except Exception as e:
        logger.error(f"CRITICAL error during final model training: {e}", exc_info=True)
        return None


def load_model():
    """Loads the trained ML model from the path specified in config."""
    # (Keep this function the same as the previous working version)
    model_path = config.MODEL_PATH
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from: {model_path}")
            if not hasattr(model, 'predict'):
                 logger.error(f"Loaded object from {model_path} lacks predict method."); return None
            # Optional: Log model type or parameters upon loading
            logger.info(f"Loaded model type: {type(model).__name__}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}", exc_info=True); return None
    else:
        logger.error(f"Model file not found at: {model_path}."); return None