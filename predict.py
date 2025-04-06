# /f1_predictor/predict.py

import pandas as pd
import numpy as np
import warnings
import os

# Import project modules
import config
import database
import utils
import model as model_loader
import feature_engineering

# Setup logger
logger = utils.get_logger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


def get_target_race_info(target_year, target_race_identifier):
    """Finds the round number, event name, and location for the target race."""
    # (Same as previous version - No changes needed here)
    logger.info(f"Looking up info for target race: Year {target_year}, Identifier '{target_race_identifier}'")
    query = "SELECT RoundNumber, EventName, Location FROM events WHERE Year = :year"
    params = {"year": target_year}
    schedule_df = database.load_data(query, params=params)
    if schedule_df is None or schedule_df.empty: logger.error(f"No schedule found for year {target_year}."); return None, None, None

    target_round, target_location, target_name = None, None, None
    try: # Match by RoundNumber first
        target_round_input = int(target_race_identifier)
        race_match = schedule_df[schedule_df['RoundNumber'] == target_round_input]
        if not race_match.empty:
            target_round, target_location, target_name = target_round_input, race_match.iloc[0]['Location'], race_match.iloc[0]['EventName']
            logger.info(f"Found target race by RoundNumber: R{target_round} - {target_name} ({target_location})")
            return target_round, target_location, target_name
    except ValueError: logger.debug("Attempting name match..."); pass
    except KeyError as e: logger.error(f"Schedule columns error: {e}"); return None, None, None

    identifier_lower = str(target_race_identifier).lower()
    try: # Match by Name/Location
        race_match = schedule_df[schedule_df['EventName'].str.lower().str.contains(identifier_lower, na=False)]
        if race_match.empty: race_match = schedule_df[schedule_df['Location'].str.lower().str.contains(identifier_lower, na=False)]
        if not race_match.empty:
            if len(race_match) > 1: logger.warning(f"Multiple matches for '{target_race_identifier}'. Using first: {race_match.iloc[0]['EventName']}")
            target_round, target_location, target_name = int(race_match.iloc[0]['RoundNumber']), race_match.iloc[0]['Location'], race_match.iloc[0]['EventName']
            logger.info(f"Found target race by Name/Location: R{target_round} - {target_name} ({target_location})")
            return target_round, target_location, target_name
        else: logger.error(f"Could not find race matching '{target_race_identifier}' for {target_year}."); return None, None, None
    except KeyError as e: logger.error(f"Schedule columns error: {e}"); return None, None, None
    except Exception as e: logger.error(f"Race lookup error: {e}", exc_info=True); return None, None, None


def prepare_prediction_data(target_year, target_round, target_location, feature_cols):
    """Prepares the feature set (X_predict) including historical and forecast features."""
    # ... (Keep section 1: Get Qualifying Data) ...
    # ... (Keep section 2: Get Latest Historical Features) ...
    # ... (Keep section 3: Merge Historical Features) ...
    # ... (Keep section 4: Handle Missing Historical Data) ...

    # --- 5. Add Weather Forecast Features ---
    logger.info("Attempting to get weather forecast...")
    expected_forecast_cols = ['ForecastTemp', 'ForecastRainProb', 'ForecastWindSpeed']
    forecast_features = {}
    coords = utils.TRACK_COORDINATES.get(target_location)
    if coords:
         latitude, longitude = coords
         forecast_data = utils.get_weather_forecast(latitude, longitude) # Call the LIVE utils function
         if forecast_data: forecast_features = forecast_data; logger.info(f"Obtained forecast data: {forecast_features}")
         else: logger.warning("Failed to get weather forecast data.")
    else: logger.warning(f"Coordinates not found for track: {target_location}.")

    # Add forecast features to the DataFrame
    for fc_col in expected_forecast_cols:
         predict_df[fc_col] = forecast_features.get(fc_col, np.nan) # Get value or NaN

         # --- Revised NaN Handling for Forecast ---
         # Fill missing forecast features using the general FILL_NA_VALUE from config
         if predict_df[fc_col].isnull().any():
             fill_value = config.FILL_NA_VALUE # Use the default numeric fill value
             logger.warning(f"Filling missing '{fc_col}' with default value: {fill_value}")
             predict_df[fc_col].fillna(fill_value, inplace=True)
         # --- End Revised NaN Handling ---

    # --- 6. Handle Categorical Track Features ---
    # (Keep this section the same as previous version)
    try:
        track_type = config.TRACK_CHARACTERISTICS.get(target_location, 'Unknown'); logger.info(f"Mapping target location '{target_location}' to TrackType: '{track_type}'")
        is_categorical_model = config.MODEL_TYPE.lower() in ['lightgbm']
        is_ohe_model = any(f.startswith('Track_') for f in feature_cols)
        if is_categorical_model and 'TrackType' in feature_cols: predict_df['TrackType'] = track_type; predict_df['TrackType'] = predict_df['TrackType'].astype('category')
        elif is_ohe_model:
            predict_df['TrackType'] = track_type; predict_df = pd.get_dummies(predict_df, columns=['TrackType'], prefix='Track', dummy_na=False, dtype=int)
            missing_ohe_cols = []
            for expected_col in feature_cols:
                 if expected_col.startswith('Track_') and expected_col not in predict_df.columns: predict_df[expected_col] = 0; missing_ohe_cols.append(expected_col)
            if missing_ohe_cols: logger.info(f"Added missing OHE columns: {missing_ohe_cols}")
    except Exception as e: logger.error(f"Error handling categorical features: {e}", exc_info=True); return None, None

    # --- 7. Final Feature Selection and Validation ---
    # (Keep this section the same as previous version, including the numeric-only NaN/Inf check)
    final_missing_cols = [col for col in feature_cols if col not in predict_df.columns]
    if final_missing_cols: logger.error(f"CRITICAL: Final features missing before selection: {final_missing_cols}"); return None, None
    try: X_predict = predict_df[feature_cols].copy(); logger.info(f"Final feature set selected. Shape: {X_predict.shape}")
    except KeyError as e: logger.error(f"KeyError selecting final features: {e}. Expected: {feature_cols}. Available: {predict_df.columns.tolist()}"); return None, None

    try: # Corrected Final Validation
        numeric_cols_in_X = X_predict.select_dtypes(include=np.number).columns
        if not numeric_cols_in_X.empty:
            logger.debug(f"Final NaN/Inf check on numeric columns: {numeric_cols_in_X.tolist()}")
            nan_mask = X_predict[numeric_cols_in_X].isnull()
            if nan_mask.values.any(): logger.warning("NaNs detected in final numeric features! Filling."); nan_cols_final = numeric_cols_in_X[nan_mask.any()].tolist(); logger.warning(f"NaNs in: {nan_cols_final}"); X_predict.fillna(config.FILL_NA_VALUE, inplace=True) # Fill ALL NaNs just in case
            inf_mask = np.isinf(X_predict[numeric_cols_in_X].values)
            if inf_mask.any(): logger.warning("Infs detected in final numeric features! Replacing."); inf_cols_final = numeric_cols_in_X[np.isinf(X_predict[numeric_cols_in_X]).any()].tolist(); logger.warning(f"Infs in: {inf_cols_final}"); X_predict.replace([np.inf, -np.inf], config.FILL_NA_VALUE * 1000, inplace=True)
        else: logger.debug("No numeric columns for final check.")
    except Exception as e: logger.error(f"Error during final NaN/Inf check: {e}", exc_info=True)

    logger.info(f"Prediction feature set prepared successfully.")
    return X_predict, predict_info_df

# --- make_predictions function ---

def make_predictions(target_year, target_race_identifier):
    """Orchestrates the prediction process."""
    logger.info(f"--- Starting Prediction: {target_year} Race: {target_race_identifier} ---")
    model = model_loader.load_model()
    if model is None: logger.error("Prediction failed: Model not loaded."); return None
    target_round, target_location, target_name = get_target_race_info(target_year, target_race_identifier)
    if target_round is None: logger.error(f"Prediction failed: Target race '{target_race_identifier}' not identified for {target_year}."); return None

    feature_cols = None
    if hasattr(model, 'feature_names_in_'):
        try: feature_cols = model.feature_names_in_.tolist(); logger.info(f"Inferred {len(feature_cols)} features from model.")
        except Exception as e: logger.warning(f"Could not get features from model: {e}. Falling back."); feature_cols = None
    if feature_cols is None: # Fallback
        logger.warning("Falling back to feature_engineering for feature names."); _, feature_cols_regen, _ = feature_engineering.create_features()
        if not feature_cols_regen: logger.error("Fallback feature_engineering failed."); return None
        feature_cols = feature_cols_regen; logger.info(f"Using {len(feature_cols)} features from feature_engineering.")
    logger.debug(f"Feature columns expected by model (or fallback): {feature_cols}")

    X_predict, predict_info_df = prepare_prediction_data(target_year, target_round, target_location, feature_cols)
    if X_predict is None or predict_info_df is None: logger.error("Prediction failed: Data preparation failed."); return None

    try: # Ensure columns match
        if list(X_predict.columns) != feature_cols: logger.warning(f"Columns mismatch/reordering needed."); X_predict = X_predict[feature_cols]
    except KeyError as e: logger.error(f"CRITICAL Column mismatch KeyError: {e}."); return None
    except Exception as e: logger.error(f"Error aligning columns: {e}"); return None

    logger.info(f"Making predictions for {len(X_predict)} drivers using {len(feature_cols)} features...")
    try: predicted_positions_raw = model.predict(X_predict); logger.info("Raw predictions generated.")
    except Exception as e: logger.error(f"Error during model.predict(): {e}", exc_info=True); return None

    try: # Format Results
        predict_info_df['PredictedPosition_Raw'] = predicted_positions_raw
        predict_info_df.sort_values(by='PredictedPosition_Raw', inplace=True, ascending=True, kind='stable')
        predict_info_df['PredictedRank'] = range(1, len(predict_info_df) + 1)
        logger.info("Prediction results processed and ranked.")
        display_cols = ['PredictedRank', 'Abbreviation', 'TeamName', 'GridPosition', 'PredictedPosition_Raw']
        final_predictions = predict_info_df[[col for col in display_cols if col in predict_info_df.columns]].copy()
        return final_predictions
    except Exception as e: logger.error(f"Error formatting results: {e}", exc_info=True); return None