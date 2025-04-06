# /f1_predictor/model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    TimeSeriesSplit,
    cross_val_score,
    RandomizedSearchCV # <-- Added
)
from sklearn.metrics import mean_absolute_error, make_scorer
import xgboost as xgb
import lightgbm as lgb
import joblib
import config
import utils
import os
import json # <-- Added
# --- NEW: Import distributions for tuning ---
from scipy.stats import randint, uniform # Keep if using distributions from config

logger = utils.get_logger(__name__)

def get_model(params=None):
    """
    Initializes the selected ML model based on config.
    Accepts optional params, typically the best found during tuning.
    """
    model_type = config.MODEL_TYPE.lower()
    final_params = {} # Start with empty dict

    # Load default fixed params first
    if model_type == 'randomforest':
        final_params = config.RF_PARAMS.copy()
        ModelClass = RandomForestRegressor
    elif model_type == 'xgboost':
        final_params = config.XGB_PARAMS.copy()
        ModelClass = xgb.XGBRegressor
    elif model_type == 'lightgbm':
        final_params = config.LGBM_PARAMS.copy()
        ModelClass = lgb.LGBMRegressor
    else:
        logger.error(f"Unsupported model type: {config.MODEL_TYPE}. Defaulting to RandomForest.")
        final_params = config.RF_PARAMS.copy()
        ModelClass = RandomForestRegressor
        model_type = 'randomforest' # Ensure consistent type

    # If tuning results (params) are provided, update the defaults
    if params:
        logger.info(f"Updating model defaults with provided params: {params}")
        final_params.update(params)
        # Ensure required params like random_state/n_jobs are still set if not tuned
        if 'random_state' not in final_params: final_params['random_state'] = 42
        if 'n_jobs' not in final_params: final_params['n_jobs'] = -1
        # Specific objective for XGBoost if needed and not in tuned params
        if model_type == 'xgboost' and 'objective' not in final_params:
            final_params['objective'] = 'reg:squarederror'
        # Verbose setting for LightGBM
        if model_type == 'lightgbm' and 'verbose' not in final_params:
             final_params['verbose'] = -1


    logger.info(f"Initializing {ModelClass.__name__} with final params: {final_params}")
    model = ModelClass(**final_params)
    return model

def train_model(df_features, feature_cols, target_col):
    """
    Trains the F1 prediction model. Includes hyperparameter tuning
    using RandomizedSearchCV if enabled in config. Saves the best model.
    """
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

    # --- Model Initialization & Tuning ---
    model_type = config.MODEL_TYPE.lower()
    best_params = None
    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # --- Hyperparameter Tuning (RandomizedSearchCV) ---
    if config.ENABLE_TUNING:
        logger.info(f"--- Starting Hyperparameter Tuning (RandomizedSearchCV, {config.TUNING_N_ITER} iterations) ---")
        if model_type == 'randomforest':
            base_estimator = RandomForestRegressor(random_state=42, n_jobs=1) # n_jobs=1 for base estimator inside CV
            param_distributions = config.RF_PARAM_DIST
        elif model_type == 'xgboost':
            base_estimator = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=1)
            param_distributions = config.XGB_PARAM_DIST
        elif model_type == 'lightgbm':
            base_estimator = lgb.LGBMRegressor(objective='regression_l1', random_state=42, n_jobs=1, verbose=-1)
            param_distributions = config.LGBM_PARAM_DIST
        else:
            logger.error("Tuning not supported for this model type. Using fixed params.")
            base_estimator = get_model() # Get model with fixed params
            param_distributions = None

        if param_distributions:
            search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=param_distributions,
                n_iter=config.TUNING_N_ITER,
                cv=tscv,
                scoring=mae_scorer,
                n_jobs=-1, # Parallelize CV folds
                random_state=42,
                verbose=1 # Log search progress
            )

            # Handle fit_params for LightGBM categorical features during search
            fit_params_for_search = {}
            if model_type == 'lightgbm':
                categorical_feature_names = [col for col in feature_cols if X[col].dtype.name == 'category']
                if categorical_feature_names:
                    fit_params_for_search['categorical_feature'] = categorical_feature_names
                    logger.info(f"Passing categorical features to RandomizedSearchCV: {categorical_feature_names}")

            try:
                logger.info(f"Fitting RandomizedSearchCV for {model_type}...")
                search.fit(X, y, **fit_params_for_search)
                best_params = search.best_params_
                best_score = -search.best_score_ # Mae scorer is negative
                logger.info(f"--- Tuning Complete ---")
                logger.info(f"Best Score (MAE): {best_score:.4f}")
                logger.info(f"Best Parameters: {best_params}")

            except Exception as e:
                logger.error(f"Error during RandomizedSearchCV: {e}", exc_info=True)
                logger.warning("Tuning failed. Proceeding with default fixed parameters.")
                best_params = None # Reset best_params so fixed ones are used

    # --- Final Model Initialization ---
    # Use best_params found during tuning (if tuning was enabled and successful)
    # Otherwise, get_model() will use the fixed params from config
    model = get_model(params=best_params)

    # --- Cross-Validation (using final params) ---
    # Optional: Re-run CV with the final chosen parameters for confirmation
    try:
        logger.info("Performing Time Series Cross-Validation with final parameters...")
        fit_params_for_cv = {}
        cv_kwargs = {}
        if model_type == 'lightgbm' and isinstance(model, lgb.LGBMRegressor):
             categorical_feature_names = [col for col in feature_cols if X[col].dtype.name == 'category']
             if categorical_feature_names:
                  fit_params_for_cv['categorical_feature'] = categorical_feature_names
                  cv_kwargs['fit_params'] = fit_params_for_cv
                  logger.info(f"Using categorical features for final LGBM CV: {categorical_feature_names}")

        scores = cross_val_score(model, X, y, cv=tscv, scoring=mae_scorer, n_jobs=-1, **cv_kwargs)
        avg_mae = -np.mean(scores)
        std_mae = np.std(scores)
        logger.info(f"Final Params Time Series CV MAE: {avg_mae:.3f} (+/- {std_mae:.3f})")

    except Exception as e:
        logger.error(f"Error during final cross-validation: {e}", exc_info=True)
        logger.warning("Proceeding with training on full data despite final CV error.")


    # --- Final Model Training on ALL provided data ---
    try:
        logger.info(f"Training final model on {len(X)} samples...")
        # Handle fit_params for final fit if using LightGBM
        fit_params_for_final_fit = {}
        if model_type == 'lightgbm' and isinstance(model, lgb.LGBMRegressor):
             categorical_feature_names = [col for col in feature_cols if X[col].dtype.name == 'category']
             if categorical_feature_names:
                  fit_params_for_final_fit['categorical_feature'] = categorical_feature_names

        if fit_params_for_final_fit:
            model.fit(X, y, **fit_params_for_final_fit)
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

        # --- Save Feature List ---
        try:
             model_feature_path = os.path.join(config.MODEL_DIR, 'model_features.json')
             with open(model_feature_path, 'w') as f:
                  json.dump(feature_cols, f)
             logger.info(f"Feature list saved successfully to: {model_feature_path}")
        except Exception as e:
             logger.error(f"Error saving feature list to {model_feature_path}: {e}", exc_info=True)

        # --- Feature Importances ---
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                if len(feature_cols) == len(importances):
                    feature_importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
                    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
                    logger.info("--- Feature Importances (Top 15) ---")
                    logger.info(f"\n{feature_importance_df.head(15).to_string(index=False)}")
                else:
                    logger.warning(f"Mismatch feature count ({len(feature_cols)}) vs importance count ({len(importances)}). Skipping importance display.")
            elif hasattr(model, 'coef_'):
                 logger.info("Model has coefficients (linear model?), not feature importances.")
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
    # (Keep this function as is)
    model_path = config.MODEL_PATH
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from: {model_path}")
            if not hasattr(model, 'predict'):
                 logger.error(f"Loaded object from {model_path} lacks predict method."); return None
            logger.info(f"Loaded model type: {type(model).__name__}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}", exc_info=True); return None
    else:
        logger.error(f"Model file not found at: {model_path}."); return None