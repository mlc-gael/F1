# /f1_predictor/model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    TimeSeriesSplit, cross_val_score, RandomizedSearchCV, cross_val_predict
)
from sklearn.metrics import mean_absolute_error, make_scorer
import xgboost as xgb
import lightgbm as lgb
import joblib
import config
import utils
import os
import json
from scipy.stats import randint, uniform # For tuning distributions
from scipy.stats import spearmanr, kendalltau # For rank correlation
import logging # For using logging constants if needed

# Import predict module ONLY for accessing the fill value helper if needed
# Reduces tight coupling if the helper is moved to utils or config later
try:
    import predict
    GET_FILL_VALUE = predict.get_feature_fill_value
except ImportError:
    # Fallback if predict isn't available (e.g., during isolated testing)
    # Use config directly, less flexible if pattern logic is complex
    print("Warning: Could not import 'predict' module. Using basic config fill values.")
    def get_config_fill_value(col_name):
        return config.FEATURE_FILL_DEFAULTS.get(col_name, config.FEATURE_FILL_DEFAULTS.get('default'))
    GET_FILL_VALUE = get_config_fill_value


logger = utils.get_logger(__name__)

def get_model(params=None):
    """
    Initializes the selected ML model based on config.
    Accepts optional params (typically best found during tuning).
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
        logger.info(f"Updating model defaults with provided tuning params: {params}")
        # Ensure numerical types from tuning (like np.float64) are converted if needed
        cleaned_params = {}
        for k, v in params.items():
             if isinstance(v, np.floating): cleaned_params[k] = float(v)
             elif isinstance(v, np.integer): cleaned_params[k] = int(v)
             else: cleaned_params[k] = v
        final_params.update(cleaned_params)

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
    try:
        model = ModelClass(**final_params)
        return model
    except Exception as e:
        logger.error(f"Failed to initialize {ModelClass.__name__} with params {final_params}: {e}", exc_info=True)
        return None


def calculate_rank_correlation(df_with_preds, true_col='TruePosition', pred_col='PredictedPosition_Raw'):
    """Calculates Spearman and Kendall correlation on a DataFrame with true/predicted values."""
    if df_with_preds is None or df_with_preds.empty or true_col not in df_with_preds or pred_col not in df_with_preds:
        logger.warning("Invalid input for rank correlation calculation.")
        return np.nan, np.nan # Return NaNs if data is invalid

    # Ensure ranks are calculated per race
    df_with_preds['TrueRank'] = df_with_preds.groupby(['Year', 'RoundNumber'])[true_col].rank(method='dense')
    df_with_preds['PredictedRank'] = df_with_preds.groupby(['Year', 'RoundNumber'])[pred_col].rank(method='dense')

    # Drop rows where ranks couldn't be calculated (e.g., single-entry races) or predictions are NaN
    df_ranked = df_with_preds.dropna(subset=['TrueRank', 'PredictedRank'])
    if len(df_ranked) < 2: # Need at least 2 pairs to correlate
        logger.warning(f"Not enough valid rank pairs ({len(df_ranked)}) to calculate correlation.")
        return np.nan, np.nan

    try:
        spearman_corr, _ = spearmanr(df_ranked['TrueRank'], df_ranked['PredictedRank'])
        kendall_corr, _ = kendalltau(df_ranked['TrueRank'], df_ranked['PredictedRank'])
        # Handle potential NaN results from correlation functions if input is strange
        spearman_corr = np.nan if pd.isna(spearman_corr) else spearman_corr
        kendall_corr = np.nan if pd.isna(kendall_corr) else kendall_corr
    except Exception as e:
         logger.error(f"Error calculating rank correlation: {e}", exc_info=True)
         return np.nan, np.nan

    return spearman_corr, kendall_corr


def train_model(df_dev, feature_cols, target_col):
    """
    Trains the F1 prediction model using the development dataset.
    Includes tuning, walk-forward evaluation on dev set, final training, and saving.
    """
    logger.info("Starting model training process on Development Set...")

    # --- Data Prep & Validation ---
    if df_dev is None or df_dev.empty: logger.error("Cannot train: Development DataFrame is empty."); return None
    if not feature_cols: logger.error("Cannot train: No feature columns provided."); return None
    if target_col not in df_dev.columns: logger.error(f"Cannot train: Target '{target_col}' not found in Dev set."); return None

    df_dev = df_dev.sort_values(by=['Year', 'RoundNumber']).reset_index(drop=True) # Reset index after sort
    logger.info(f"Development data shape: {df_dev.shape}")

    X_dev = df_dev[feature_cols].copy() # Explicit copy
    y_dev = df_dev[target_col].copy()

    # Validate and clean Dev features (robustness check)
    numeric_cols_dev = X_dev.select_dtypes(include=np.number).columns
    if not numeric_cols_dev.empty:
        nan_mask = X_dev[numeric_cols_dev].isnull()
        if nan_mask.values.any():
            nan_cols = numeric_cols_dev[nan_mask.any()].tolist()
            logger.warning(f"NaNs found in Dev features before training: {nan_cols}. Filling with defaults.")
            for col in nan_cols:
                X_dev[col].fillna(GET_FILL_VALUE(col), inplace=True)
        inf_mask = np.isinf(X_dev[numeric_cols_dev])
        if inf_mask.values.any():
             inf_cols = numeric_cols_dev[inf_mask.any()].tolist()
             logger.warning(f"Infs found in Dev features before training: {inf_cols}. Replacing.")
             X_dev[numeric_cols_dev] = X_dev[numeric_cols_dev].replace([np.inf, -np.inf], config.FEATURE_FILL_DEFAULTS['default'] * 100)

    if y_dev.isnull().any():
        logger.warning(f"NaNs found in Dev target ('{target_col}'). Removing these rows for training.")
        valid_indices = y_dev.notna()
        X_dev = X_dev.loc[valid_indices, :]
        y_dev = y_dev.loc[valid_indices]
        df_dev = df_dev.loc[valid_indices, :].copy() # Keep original df aligned for OOF eval
        logger.info(f"Removed {sum(~valid_indices)} rows from Dev set due to NaN target. New training shape: {X_dev.shape}")
        if X_dev.empty: logger.error("No valid Dev data remaining after target NaN removal."); return None

    # Capture dtypes *after* potential cleaning
    feature_dtypes = X_dev.dtypes.astype(str).to_dict()

    # --- Hyperparameter Tuning (on Dev set using TimeSeriesSplit) ---
    model_type = config.MODEL_TYPE.lower()
    best_params = None # Will store the best found parameters
    tscv_tune = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    tuning_scorer = config.TUNING_SCORER
    logger.info(f"Using scorer for tuning: {tuning_scorer}")

    if config.ENABLE_TUNING:
        logger.info(f"--- Starting Hyperparameter Tuning on Dev Set ({config.TUNING_N_ITER} iterations) ---")
        param_distributions = None
        if model_type == 'randomforest':
            base_estimator = RandomForestRegressor(random_state=42, n_jobs=1); param_distributions = config.RF_PARAM_DIST
        elif model_type == 'xgboost':
            base_estimator = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=1); param_distributions = config.XGB_PARAM_DIST
        elif model_type == 'lightgbm':
            base_estimator = lgb.LGBMRegressor(objective='regression_l1', random_state=42, n_jobs=1, verbose=-1); param_distributions = config.LGBM_PARAM_DIST
        else: logger.error("Tuning not supported for this model type. Using fixed params.")

        if base_estimator is not None and param_distributions is not None:
            search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=param_distributions,
                n_iter=config.TUNING_N_ITER,
                cv=tscv_tune, # Use the time series splitter
                scoring=tuning_scorer,
                n_jobs=-1, # Use all available cores for CV folds
                random_state=42,
                verbose=1 # Log search progress
            )
            fit_params_for_search = {}
            if model_type == 'lightgbm':
                categorical_feature_names = [col for col in feature_cols if X_dev[col].dtype.name == 'category']
                if categorical_feature_names:
                    fit_params_for_search['categorical_feature'] = categorical_feature_names
                    logger.info(f"Passing categorical features to RandomizedSearchCV: {categorical_feature_names}")
            try:
                logger.info(f"Fitting RandomizedSearchCV for {model_type} on Dev set...")
                search.fit(X_dev, y_dev, **fit_params_for_search)
                best_params = search.best_params_ # Get the best parameters dictionary
                best_score = search.best_score_ # Get the best score achieved during tuning
                logger.info(f"--- Tuning Complete ---")
                # Adjust score sign for display if using negative scorer
                display_score = -best_score if tuning_scorer.startswith('neg_') else best_score
                score_name = tuning_scorer.replace('neg_', '') if tuning_scorer.startswith('neg_') else tuning_scorer
                logger.info(f"Best Tuning Score ({score_name}): {display_score:.4f}")
                logger.info(f"Best Parameters Found: {best_params}")
            except Exception as e:
                logger.error(f"Error during RandomizedSearchCV: {e}", exc_info=True)
                logger.warning("Proceeding with default fixed parameters due to tuning error.")
                best_params = None # Fallback to defaults
        else:
            logger.warning("Could not set up tuning (invalid model type or missing distributions). Using fixed params.")
            best_params = None
    else:
        logger.info("Hyperparameter tuning disabled. Using fixed parameters.")
        best_params = None # Ensure fixed params are used below

    # --- Initialize Model with Best/Fixed Params ---
    model_for_eval = get_model(params=best_params)
    if model_for_eval is None:
         logger.error("Failed to initialize model even with default parameters. Aborting training.")
         return None

    # --- Walk-Forward Evaluation on Development Set ---
    logger.info(f"--- Performing Walk-Forward Evaluation on Dev Set ({config.CV_SPLITS} splits) ---")
    tscv_eval = TimeSeriesSplit(n_splits=config.CV_SPLITS) # Use same splitter for consistency
    oof_predictions = pd.Series(index=df_dev.index, dtype=float) # Use original df_dev index

    try:
        fold_indices = list(tscv_eval.split(X_dev, y_dev))
        fit_params_for_cv = {} # Determine fit params if needed (e.g., for LightGBM)
        cv_kwargs = {}
        if model_type == 'lightgbm' and isinstance(model_for_eval, lgb.LGBMRegressor):
             categorical_feature_names = [col for col in feature_cols if X_dev[col].dtype.name == 'category']
             if categorical_feature_names:
                  fit_params_for_cv['categorical_feature'] = categorical_feature_names
                  cv_kwargs['fit_params'] = fit_params_for_cv
                  logger.info(f"Using categorical features for LGBM walk-forward CV: {categorical_feature_names}")

        for i, (train_index, val_index) in enumerate(fold_indices):
            logger.debug(f"Evaluating Fold {i+1}/{config.CV_SPLITS} (Train size: {len(train_index)}, Val size: {len(val_index)})")
            # Ensure indices are valid after potential row removal due to NaN targets
            if not np.all(np.isin(train_index, X_dev.index)) or not np.all(np.isin(val_index, X_dev.index)):
                 logger.error(f"Index mismatch detected in Fold {i+1}. Check NaN target removal logic.")
                 continue # Skip fold if indices are broken

            X_train_fold, X_val_fold = X_dev.loc[train_index], X_dev.loc[val_index]
            y_train_fold = y_dev.loc[train_index]

            # Clone model to ensure fresh fit for each fold
            model_fold = get_model(params=best_params)
            if model_fold is None:
                 logger.error(f"Failed to initialize model for Fold {i+1}. Skipping fold.")
                 continue

            if cv_kwargs.get('fit_params'): model_fold.fit(X_train_fold, y_train_fold, **cv_kwargs['fit_params'])
            else: model_fold.fit(X_train_fold, y_train_fold)

            fold_preds = model_fold.predict(X_val_fold)
            # Assign predictions using the original DataFrame's index slice
            oof_predictions.loc[val_index] = fold_preds

        # Filter out NaNs (indices where predictions weren't made - should be minimal with TimeSeriesSplit > 1)
        valid_oof_indices = oof_predictions.notna()
        if not valid_oof_indices.any():
             logger.error("Walk-forward CV failed to generate any predictions.")
        else:
            # Create DataFrame for metric calculation using original df_dev for identifiers
            df_oof = df_dev.loc[valid_oof_indices, ['Year', 'RoundNumber', 'Abbreviation']].copy()
            df_oof['TruePosition'] = y_dev.loc[valid_oof_indices]
            df_oof['PredictedPosition_Raw'] = oof_predictions.loc[valid_oof_indices]

            # Calculate Metrics on OOF predictions
            oof_mae = mean_absolute_error(df_oof['TruePosition'], df_oof['PredictedPosition_Raw'])
            oof_spearman, oof_kendall = calculate_rank_correlation(df_oof, 'TruePosition', 'PredictedPosition_Raw')

            logger.info(f"Dev Set Walk-Forward MAE: {oof_mae:.4f}")
            logger.info(f"Dev Set Walk-Forward Spearman's Rho: {oof_spearman:.4f}")
            logger.info(f"Dev Set Walk-Forward Kendall's Tau: {oof_kendall:.4f}")

    except Exception as e:
        logger.error(f"Error during Walk-Forward Evaluation on Dev Set: {e}", exc_info=True)
        logger.warning("Proceeding to final training despite CV evaluation error.")

    # --- Final Model Training on ALL Development data ---
    logger.info(f"Training final model on {len(X_dev)} Development samples using best parameters...")
    final_model = get_model(params=best_params) # Use best params found
    if final_model is None:
         logger.error("Failed to initialize final model. Aborting.")
         return None

    try:
        fit_params_for_final_fit = {} # Determine fit params again if needed
        if model_type == 'lightgbm' and isinstance(final_model, lgb.LGBMRegressor):
             categorical_feature_names = [col for col in feature_cols if X_dev[col].dtype.name == 'category']
             if categorical_feature_names: fit_params_for_final_fit['categorical_feature'] = categorical_feature_names

        if fit_params_for_final_fit: final_model.fit(X_dev, y_dev, **fit_params_for_final_fit)
        else: final_model.fit(X_dev, y_dev)
        logger.info("Final model training complete.")

        # --- Save Model, Features, and Dtypes ---
        try:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            joblib.dump(final_model, config.MODEL_PATH)
            logger.info(f"Model saved successfully to: {config.MODEL_PATH}")
            meta_data = {'features': feature_cols, 'dtypes': feature_dtypes}
            with open(config.MODEL_FEATURES_META_PATH, 'w') as f: json.dump(meta_data, f, indent=4)
            logger.info(f"Feature list and dtypes saved successfully to: {config.MODEL_FEATURES_META_PATH}")
        except Exception as e: logger.error(f"Error saving model/metadata: {e}", exc_info=True) # Log error but continue

        # --- Feature Importances (of the final model) ---
        try:
            if hasattr(final_model, 'feature_importances_'):
                importances = final_model.feature_importances_
                feature_importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
                feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
                logger.info("--- Feature Importances (Top 15) ---")
                logger.info(f"\n{feature_importance_df.head(15).to_string(index=False)}")
            elif hasattr(final_model, 'coef_'): logger.info("Model has coefficients.")
            else: logger.info("Model type does not support feature_importances_ or coef_.")
        except Exception as e: logger.warning(f"Could not display feature importances: {e}")

        # Return the single, final model trained on all dev data
        return final_model

    except Exception as e:
        logger.error(f"CRITICAL error during final model training: {e}", exc_info=True)
        return None


def load_model_and_meta():
    """Loads the model, feature list, and dtypes from config paths."""
    model_path = config.MODEL_PATH
    meta_path = config.MODEL_FEATURES_META_PATH
    model, features, dtypes = None, None, None

    if os.path.exists(model_path) and os.path.exists(meta_path):
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from: {model_path}")
            if not hasattr(model, 'predict'):
                 logger.error(f"Loaded object from {model_path} lacks predict method."); return None, None, None
            logger.info(f"Loaded model type: {type(model).__name__}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}", exc_info=True); return None, None, None

        try:
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
            features = meta_data.get('features')
            dtypes = meta_data.get('dtypes')
            if features and dtypes:
                 logger.info(f"Features ({len(features)}) and dtypes loaded from: {meta_path}")
            else:
                 logger.error(f"Metadata file {meta_path} missing 'features' or 'dtypes' key."); return model, None, None
        except Exception as e:
            logger.error(f"Error loading metadata from {meta_path}: {e}", exc_info=True); return model, None, None
    else:
        if not os.path.exists(model_path): logger.error(f"Model file not found at: {model_path}.")
        if not os.path.exists(meta_path): logger.error(f"Features/Dtypes meta file not found at: {meta_path}.")
        return None, None, None

    return model, features, dtypes


def load_model():
    """Loads only the trained ML model object (simple version)."""
    model_path = config.MODEL_PATH
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from: {model_path}")
            if not hasattr(model, 'predict'): logger.error(f"Loaded object from {model_path} lacks predict method."); return None
            return model
        except Exception as e: logger.error(f"Error loading model from {model_path}: {e}", exc_info=True); return None
    else: logger.error(f"Model file not found at: {model_path}."); return None