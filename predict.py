# /f1_predictor/predict.py

import pandas as pd
import numpy as np
import warnings
import os
import datetime
import json # Added import

# Import project modules
import config
import database
import utils
import model as model_loader
import feature_engineering # Keep this import

# Setup logger
utils.setup_logging()
logger = utils.get_logger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


def get_target_race_info(target_year, target_race_identifier):
    """Finds the round number, event name, and location for the target race."""
    # (Keep this function as is)
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
        race_match_exact = schedule_df[schedule_df['EventName'].str.lower() == identifier_lower]
        if race_match_exact.empty:
             race_match_exact = schedule_df[schedule_df['Location'].str.lower() == identifier_lower]

        if not race_match_exact.empty:
             race_match = race_match_exact
        else:
             race_match_contains = schedule_df[schedule_df['EventName'].str.lower().str.contains(identifier_lower, na=False)]
             if race_match_contains.empty:
                  race_match_contains = schedule_df[schedule_df['Location'].str.lower().str.contains(identifier_lower, na=False)]
             race_match = race_match_contains

        if not race_match.empty:
            if len(race_match) > 1: logger.warning(f"Multiple matches for '{target_race_identifier}'. Using first: {race_match.iloc[0]['EventName']}")
            target_round, target_location, target_name = int(race_match.iloc[0]['RoundNumber']), race_match.iloc[0]['Location'], race_match.iloc[0]['EventName']
            logger.info(f"Found target race by Name/Location: R{target_round} - {target_name} ({target_location})")
            return target_round, target_location, target_name
        else: logger.error(f"Could not find race matching '{target_race_identifier}' for {target_year}."); return None, None, None
    except KeyError as e: logger.error(f"Schedule columns error: {e}"); return None, None, None
    except Exception as e: logger.error(f"Race lookup error: {e}", exc_info=True); return None, None, None


def get_historical_weather_averages(track_location):
    """Gets historical average weather for a specific track from the DB."""
    # (Keep this function as is)
    logger.info(f"Querying historical weather averages for track: {track_location}")
    query = """
        SELECT
            AVG(w.AirTemp) as AvgTemp,
            AVG(CASE WHEN w.Rainfall = 1 THEN 1.0 ELSE 0.0 END) as AvgRainProb, -- Avg probability
            AVG(w.WindSpeed) as AvgWindSpeed
        FROM weather w
        JOIN events e ON w.Year = e.Year AND w.RoundNumber = e.RoundNumber
        WHERE e.Location = :location AND w.SessionName = 'R' -- Use Race session weather
    """
    params = {"location": track_location}
    hist_weather = database.load_data(query, params=params)

    if hist_weather is not None and not hist_weather.empty and not hist_weather.isnull().all().all():
         averages = hist_weather.iloc[0].to_dict()
         mapped_averages = {
             'ForecastTemp': averages.get('AvgTemp'),
             'ForecastRainProb': averages.get('AvgRainProb'),
             'ForecastWindSpeed': averages.get('AvgWindSpeed')
         }
         final_averages = {k: v for k, v in mapped_averages.items() if v is not None and not pd.isna(v)}
         if final_averages:
             logger.info(f"Found historical weather averages: {final_averages}")
             return final_averages
         else:
              logger.warning("Historical weather query succeeded but returned no valid averages.")
              return None
    else:
        logger.warning(f"Could not find historical weather averages for track: {track_location}")
        return None


def prepare_prediction_data(target_year, target_round, target_location, feature_cols):
    """
    Prepares the feature set (X_predict).
    Handles missing qualifying data and missing weather forecasts.
    """
    # (Keep Sections 1-4 as they were in the previous corrected version)
    logger.info(f"Preparing prediction data for {target_year} R{target_round} ({target_location}).")
    use_quali_fallback = False

    # --- 1. Get Qualifying Data (Optional) ---
    logger.info(f"Attempting to fetch Qualifying data for {target_year} R{target_round}")
    quali_query = "SELECT Abbreviation, TeamName, Position as QualiPositionRaw FROM results WHERE Year = :year AND RoundNumber = :round AND SessionName = 'Q'"
    quali_params = {"year": target_year, "round": target_round}
    df_quali_target = database.load_data(quali_query, params=quali_params)
    predict_info_df = pd.DataFrame()

    if df_quali_target.empty:
        logger.warning(f"No Qualifying data found for target race {target_year} R{target_round}. GridPosition will use fallbacks.")
        use_quali_fallback = True
    else:
        df_quali_target['QualiPositionRaw'] = utils.safe_to_numeric(df_quali_target['QualiPositionRaw'], fallback=np.nan)
        min_quali_indices = df_quali_target.groupby('Abbreviation')['QualiPositionRaw'].idxmin().dropna()
        if min_quali_indices.empty:
            logger.warning(f"Quali data found, but no valid times set. GridPosition will use fallbacks.")
            use_quali_fallback = True
            predict_info_df = df_quali_target[['Abbreviation', 'TeamName']].drop_duplicates().copy()
            predict_info_df['GridPosition'] = np.nan
        else:
            df_quali_final = df_quali_target.loc[min_quali_indices].copy()
            df_quali_final.rename(columns={'QualiPositionRaw': 'GridPosition'}, inplace=True)
            all_drivers_in_quali = df_quali_target['Abbreviation'].unique()
            drivers_with_valid_time = df_quali_final['Abbreviation'].unique()
            drivers_without_valid_time = np.setdiff1d(all_drivers_in_quali, drivers_with_valid_time)
            if len(drivers_without_valid_time) > 0:
                logger.warning(f"Drivers with no valid quali time found: {drivers_without_valid_time}. Assigning worst grid position.")
                missing_drivers_df = pd.DataFrame({
                    'Abbreviation': drivers_without_valid_time,
                    'TeamName': [df_quali_target[df_quali_target['Abbreviation'] == abbr]['TeamName'].iloc[0] if not df_quali_target[df_quali_target['Abbreviation'] == abbr].empty else 'Unknown' for abbr in drivers_without_valid_time],
                    'GridPosition': config.WORST_EXPECTED_POS
                })
                df_quali_final = pd.concat([df_quali_final, missing_drivers_df], ignore_index=True)
            df_quali_final['GridPosition'] = utils.safe_to_numeric(df_quali_final['GridPosition'], fallback=config.WORST_EXPECTED_POS).astype(int)
            logger.info(f"Processed Qualifying data. Grid positions determined for {len(df_quali_final)} drivers.")
            predict_info_df = df_quali_final[['Abbreviation', 'TeamName', 'GridPosition']].copy()

    # --- 2. Get Latest Historical Features ---
    logger.info("Generating historical features to find latest values...")
    df_features_all, _, _ = feature_engineering.create_features()
    if df_features_all is None or df_features_all.empty:
        logger.error("CRITICAL: Feature generation for historical data failed. Cannot proceed.")
        return None, None
    df_features_all.sort_values(by=['Year', 'RoundNumber'], inplace=True)
    previous_race_entries = df_features_all[
        (df_features_all['Year'] < target_year) |
        ((df_features_all['Year'] == target_year) & (df_features_all['RoundNumber'] < target_round))
    ]
    if previous_race_entries.empty:
        logger.warning(f"No historical race data found prior to target {target_year} R{target_round}. Using defaults/NaNs for historical features.")
        latest_historical_features = pd.DataFrame(columns=['Abbreviation'] + feature_cols)
        if use_quali_fallback:
             logger.error("CRITICAL: No qualifying data AND no previous race data found. Cannot determine drivers for prediction.")
             return None, None
    else:
        last_hist_year = previous_race_entries['Year'].iloc[-1]
        last_hist_round = previous_race_entries['RoundNumber'].iloc[-1]
        logger.info(f"Latest historical data point found: {last_hist_year} R{last_hist_round}")
        latest_historical_features = df_features_all[
            (df_features_all['Year'] == last_hist_year) &
            (df_features_all['RoundNumber'] == last_hist_round)
        ].copy()
        if use_quali_fallback:
             logger.info("Using drivers from last historical race as participants.")
             predict_info_df = latest_historical_features[['Abbreviation', 'TeamName']].drop_duplicates().copy()
             predict_info_df['GridPosition'] = np.nan

    # --- 3. Merge Historical Features ---
    logger.info("Merging latest historical features...")
    cols_to_merge = ['Abbreviation'] + [col for col in feature_cols if col != 'GridPosition' and col in latest_historical_features.columns]
    features_to_merge = latest_historical_features[cols_to_merge].drop_duplicates(subset=['Abbreviation'], keep='last')
    if predict_info_df.empty:
         logger.error("CRITICAL: No drivers identified for prediction.")
         return None, None
    predict_df = pd.merge(predict_info_df, features_to_merge, on='Abbreviation', how='left')
    logger.info(f"Merged features. Shape: {predict_df.shape}. Columns: {predict_df.columns.tolist()}")

    # --- 4. Handle Missing Data ---
    if use_quali_fallback:
        logger.warning("Applying fallback logic for GridPosition.")
        if 'RollingAvgPosLastN' in predict_df.columns:
             predict_df['GridPosition'] = predict_df['GridPosition'].fillna(predict_df['RollingAvgPosLastN'].round()).fillna(11.0)
             logger.info("Used RollingAvgPosLastN for GridPosition fallback.")
        else:
             logger.warning("RollingAvgPosLastN not available, using default mid-pack (11) for GridPosition.")
             predict_df['GridPosition'].fillna(11.0, inplace=True)
        predict_df['GridPosition'] = predict_df['GridPosition'].astype(int)
    missing_feature_mask = predict_df[[col for col in feature_cols if col != 'GridPosition']].isnull()
    cols_with_miss = missing_feature_mask.any()
    missing_cols = cols_with_miss[cols_with_miss].index.tolist()
    if missing_cols:
        logger.warning(f"Missing historical feature data detected for some drivers in columns: {missing_cols}. Filling with defaults.")
        for col in missing_cols:
            if col in predict_df.columns:
                fill_val = config.FILL_NA_VALUE
                if 'Pts' in col or 'Points' in col: fill_val = 0.0
                elif 'Pos' in col or 'Position' in col: fill_val = float(config.WORST_EXPECTED_POS)
                predict_df[col].fillna(fill_val, inplace=True)
            else: logger.error(f"Logic error: Column '{col}' expected but not found during NaN fill.")

    # --- 5. Add Weather Forecast Features (with Fallback) ---
    logger.info("Attempting to get weather forecast...")
    expected_forecast_cols = ['ForecastTemp', 'ForecastRainProb', 'ForecastWindSpeed']
    forecast_features = {}
    target_event_date = None
    coords = utils.TRACK_COORDINATES.get(target_location)
    if coords:
        latitude, longitude = coords
        forecast_data = utils.get_weather_forecast(latitude, longitude, target_date=target_event_date)
        if forecast_data:
            forecast_features = forecast_data
            logger.info(f"Obtained live forecast data: {forecast_features}")
        else:
            logger.warning("Failed to get live weather forecast. Attempting historical fallback.")
            hist_weather_avg = get_historical_weather_averages(target_location)
            if hist_weather_avg:
                 logger.warning(f"Using historical weather averages for fallback: {hist_weather_avg}")
                 forecast_features = hist_weather_avg
            else:
                 logger.error("Historical weather fallback failed. Using generic defaults.")
                 forecast_features = {'ForecastTemp': 20.0, 'ForecastRainProb': 0.1, 'ForecastWindSpeed': 10.0}
    else:
        logger.warning(f"Coordinates not found for track: {target_location}. Using generic weather defaults.")
        forecast_features = { 'ForecastTemp': 20.0, 'ForecastRainProb': 0.1, 'ForecastWindSpeed': 10.0 }
    for fc_col in expected_forecast_cols:
        value = forecast_features.get(fc_col)
        if value is None or pd.isna(value):
             default_val = config.FILL_NA_VALUE
             if fc_col == 'ForecastTemp': default_val = 20.0
             elif fc_col == 'ForecastRainProb': default_val = 0.1
             elif fc_col == 'ForecastWindSpeed': default_val = 10.0
             logger.warning(f"Assigning default value {default_val} for missing weather feature '{fc_col}'")
             predict_df.loc[:, fc_col] = default_val
        else:
             predict_df.loc[:, fc_col] = value
        predict_df[fc_col] = predict_df[fc_col].astype(float)

    # --- 6. Handle Categorical Track Features ---
    # --- FIX START: Refined OHE Handling ---
    try:
        track_type = config.TRACK_CHARACTERISTICS.get(target_location, 'Unknown')
        logger.info(f"Mapping target location '{target_location}' to TrackType: '{track_type}'")
        is_categorical_model = config.MODEL_TYPE.lower() in ['lightgbm']
        # Get list of expected OHE columns from feature_cols
        expected_ohe_cols = [f for f in feature_cols if f.startswith('Track_')]
        is_ohe_model = bool(expected_ohe_cols) # True if any Track_ cols are expected

        if is_categorical_model and 'TrackType' in feature_cols:
            predict_df['TrackType'] = track_type
            predict_df['TrackType'] = predict_df['TrackType'].astype('category')
            logger.info("Added TrackType as category for LightGBM.")
            # Ensure 'TrackType' is the only track-related feature expected
            if any(f.startswith('Track_') for f in feature_cols):
                logger.warning("Model expects both 'TrackType' and OHE 'Track_*' columns? Check feature_cols.")
        elif is_ohe_model:
            # Add all expected OHE columns first, initialized to 0
            logger.debug(f"Initializing expected OHE columns: {expected_ohe_cols}")
            for ohe_col in expected_ohe_cols:
                if ohe_col not in predict_df.columns:
                     predict_df[ohe_col] = 0
                else:
                     # If it somehow already exists (e.g., from historical merge), ensure it's numeric 0/1
                     predict_df[ohe_col] = utils.safe_to_numeric(predict_df[ohe_col], fallback=0).astype(int)


            # Determine the specific OHE column for the current track
            current_track_ohe_col = f"Track_{track_type.replace(' ', '_')}" # Construct expected column name
            logger.info(f"Applying OHE for current track. Column: '{current_track_ohe_col}'")

            if current_track_ohe_col in predict_df.columns:
                # Set the specific column for this track to 1
                predict_df[current_track_ohe_col] = 1
            else:
                # This should only happen if the track type is new and wasn't in feature_cols
                logger.warning(f"OHE column '{current_track_ohe_col}' for current track not found in expected features. Track might be new or config/features mismatch.")
                # Ensure Track_Unknown exists and set it if the specific column is missing
                if 'Track_Unknown' in predict_df.columns:
                    logger.warning("Setting 'Track_Unknown' to 1 as fallback.")
                    predict_df['Track_Unknown'] = 1
                else:
                     logger.error("Cannot set OHE column for current track, and 'Track_Unknown' is also missing!")

            # Remove the original 'TrackType' if it exists and we used OHE
            if 'TrackType' in predict_df.columns:
                predict_df.drop(columns=['TrackType'], inplace=True, errors='ignore')

        else:
            if 'TrackType' in feature_cols or any(f.startswith('Track_') for f in feature_cols):
                 logger.warning("Track feature handling mismatch: Model expects track features but couldn't apply them.")
            else:
                 logger.info("TrackType feature not used by this model or not found in feature_cols.")

    except Exception as e:
        logger.error(f"Error handling categorical features: {e}", exc_info=True)
        return None, None
    # --- FIX END ---

    # --- 7. Final Feature Selection and Validation ---
    final_missing_cols = [col for col in feature_cols if col not in predict_df.columns]
    if final_missing_cols:
        logger.error(f"CRITICAL: Final features missing before selection: {final_missing_cols}. Available: {predict_df.columns.tolist()}")
        for col in final_missing_cols: predict_df[col] = config.FILL_NA_VALUE
        logger.warning(f"Added missing columns with value {config.FILL_NA_VALUE}.")

    try:
        # Ensure no duplicate columns exist *before* selection
        if predict_df.columns.has_duplicates:
             logger.error(f"Duplicate columns found in predict_df BEFORE final selection: {predict_df.columns[predict_df.columns.duplicated()].tolist()}")
             # Attempt to remove duplicates, keeping the first occurrence
             predict_df = predict_df.loc[:, ~predict_df.columns.duplicated()]
             logger.warning("Attempted to remove duplicate columns.")

        # Now select the final features
        X_predict = predict_df[feature_cols].copy()
        logger.info(f"Final feature set selected. Shape: {X_predict.shape}")
    except KeyError as e:
        logger.error(f"KeyError selecting final features: {e}. Expected: {feature_cols}. Available: {predict_df.columns.tolist()}"); return None, None
    except Exception as e: # Catch other potential errors like duplicate labels during selection
         logger.error(f"Error during final feature selection: {e}", exc_info=True); return None, None


    try: # Final NaN/Inf check
        numeric_cols_in_X = X_predict.select_dtypes(include=np.number).columns
        if not numeric_cols_in_X.empty:
            logger.debug(f"Final NaN/Inf check on numeric columns: {numeric_cols_in_X.tolist()}")
            nan_mask = X_predict[numeric_cols_in_X].isnull()
            if nan_mask.values.any():
                logger.warning("NaNs detected in final numeric features! Filling.")
                nan_cols_final = numeric_cols_in_X[nan_mask.any()].tolist(); logger.warning(f"NaNs in: {nan_cols_final}")
                for col in nan_cols_final: X_predict[col].fillna(config.FILL_NA_VALUE, inplace=True)
            inf_mask = np.isinf(X_predict[numeric_cols_in_X]).any().any()
            if inf_mask:
                logger.warning("Infs detected in final numeric features! Replacing.")
                inf_cols_final = numeric_cols_in_X[np.isinf(X_predict[numeric_cols_in_X]).any()].tolist(); logger.warning(f"Infs in: {inf_cols_final}")
                X_predict[numeric_cols_in_X] = X_predict[numeric_cols_in_X].replace([np.inf, -np.inf], config.FILL_NA_VALUE * 1000)
        else: logger.debug("No numeric columns for final check.")
    except Exception as e: logger.error(f"Error during final NaN/Inf check: {e}", exc_info=True)

    logger.info(f"Prediction feature set prepared successfully.")
    return X_predict, predict_info_df


# --- make_predictions function ---
def make_predictions(target_year, target_race_identifier):
    """Orchestrates the prediction process."""
    # (Keep this function as is - alignment logic is needed)
    logger.info(f"--- Starting Prediction: {target_year} Race: {target_race_identifier} ---")
    model = model_loader.load_model()
    if model is None: logger.error("Prediction failed: Model not loaded."); return None
    target_round, target_location, target_name = get_target_race_info(target_year, target_race_identifier)
    if target_round is None: logger.error(f"Prediction failed: Target race '{target_race_identifier}' not identified for {target_year}."); return None

    feature_cols = None
    model_feature_path = os.path.join(config.MODEL_DIR, 'model_features.json')
    if os.path.exists(model_feature_path):
        try:
            with open(model_feature_path, 'r') as f:
                feature_cols = json.load(f)
            logger.info(f"Loaded {len(feature_cols)} features from {model_feature_path}.")
        except Exception as e:
            logger.warning(f"Could not load features from {model_feature_path}: {e}. Will try inferring from model.")
            feature_cols = None

    if feature_cols is None and hasattr(model, 'feature_names_in_'):
        try:
             feature_cols = model.feature_names_in_.tolist(); logger.info(f"Inferred {len(feature_cols)} features from model.")
        except Exception as e: logger.warning(f"Could not get features from model: {e}. Falling back."); feature_cols = None

    if feature_cols is None:
        logger.warning("Falling back to feature_engineering for feature names.");
        _, feature_cols_regen, _ = feature_engineering.create_features()
        if not feature_cols_regen:
            logger.error("Fallback feature_engineering failed. Cannot determine features.")
            return None
        feature_cols = feature_cols_regen
        logger.info(f"Using {len(feature_cols)} features from feature_engineering fallback.")

    if not feature_cols:
         logger.error("CRITICAL: Could not determine feature columns required for prediction.")
         return None

    logger.debug(f"Feature columns expected by model (or fallback): {feature_cols}")

    X_predict, predict_info_df = prepare_prediction_data(target_year, target_round, target_location, feature_cols)
    if X_predict is None or predict_info_df is None: logger.error("Prediction failed: Data preparation failed."); return None

    # Ensure columns match EXACTLY (order matters)
    try:
        if list(X_predict.columns) != feature_cols:
             logger.warning(f"Columns mismatch/reordering needed. Aligning prediction data to model's expected features: {feature_cols}")
             # Check for duplicates *before* reindexing
             if X_predict.columns.has_duplicates:
                  logger.error(f"Duplicate columns found in X_predict BEFORE reindexing: {X_predict.columns[X_predict.columns.duplicated()].tolist()}")
                  # Attempt removal again - this indicates an issue in prepare_prediction_data
                  X_predict = X_predict.loc[:, ~X_predict.columns.duplicated()]
                  logger.warning("Attempted duplicate column removal before reindexing.")

             X_predict = X_predict.reindex(columns=feature_cols) # Reindex based on the definitive list
             nan_check = X_predict.isnull().sum()
             nan_cols = nan_check[nan_check > 0].index.tolist()
             if nan_cols:
                 logger.warning(f"NaNs introduced after reindexing columns: {nan_cols}. Filling with {config.FILL_NA_VALUE}.")
                 X_predict.fillna(config.FILL_NA_VALUE, inplace=True)
    except ValueError as e: # Catch duplicate label error during reindex
         if "cannot reindex on an axis with duplicate labels" in str(e):
              logger.error(f"Duplicate label error during reindex. X_predict columns: {list(X_predict.columns)}. Expected features: {feature_cols}", exc_info=True)
         else:
              logger.error(f"ValueError aligning columns: {e}", exc_info=True)
         return None
    except KeyError as e: logger.error(f"CRITICAL Column mismatch KeyError during reindex: {e}. Model needs {feature_cols}, Data has {list(X_predict.columns)}"); return None
    except Exception as e: logger.error(f"Error aligning columns: {e}", exc_info=True); return None


    logger.info(f"Making predictions for {len(X_predict)} drivers using {len(feature_cols)} features...")
    try:
        predicted_positions_raw = model.predict(X_predict); logger.info("Raw predictions generated.")
    except Exception as e: logger.error(f"Error during model.predict(): {e}", exc_info=True); return None

    try: # Format Results
        predict_info_df['PredictedPosition_Raw'] = predicted_positions_raw
        predict_info_df.sort_values(by='PredictedPosition_Raw', inplace=True, ascending=True, kind='stable')
        predict_info_df['PredictedRank'] = range(1, len(predict_info_df) + 1)
        logger.info("Prediction results processed and ranked.")
        display_cols = ['PredictedRank', 'Abbreviation', 'TeamName', 'GridPosition', 'PredictedPosition_Raw']
        final_display_cols = [col for col in display_cols if col in predict_info_df.columns]
        final_predictions = predict_info_df[final_display_cols].copy()
        if 'GridPosition' in final_predictions.columns:
             final_predictions['GridPosition'].fillna('N/A', inplace=True)
        return final_predictions
    except Exception as e: logger.error(f"Error formatting results: {e}", exc_info=True); return None