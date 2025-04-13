# /f1_predictor/predict.py

import pandas as pd
import numpy as np
import warnings
import os
import datetime
import json
try:
    import pyarrow
except ImportError:
    print("Warning: 'pyarrow' not installed. Saving/loading features cache will fail if using Feather format.")

# Import project modules
import config
import database
import utils
import model as model_loader

# Setup logger
# Assuming utils.setup_logging() is called elsewhere (e.g., main.py)
# utils.setup_logging()
logger = utils.get_logger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


def get_target_race_info(target_year, target_race_identifier):
    """Finds the round number, event name, location, and date for the target race."""
    logger.info(f"Looking up info for target race: Year {target_year}, Identifier '{target_race_identifier}'")
    query = "SELECT RoundNumber, EventName, Location, EventDate FROM events WHERE Year = :year"
    params = {"year": target_year}
    schedule_df = database.load_data(query, params=params)
    if schedule_df is None or schedule_df.empty:
        logger.error(f"No schedule found for year {target_year}.")
        return None, None, None, None

    target_round, target_location, target_name, target_date_str = None, None, None, None
    try:
        target_round_input = int(target_race_identifier)
        race_match = schedule_df[schedule_df['RoundNumber'] == target_round_input]
        if not race_match.empty:
            target_round = target_round_input
            target_location = race_match.iloc[0]['Location']
            target_name = race_match.iloc[0]['EventName']
            target_date_str = race_match.iloc[0]['EventDate']
            logger.info(f"Found target race by RoundNumber: R{target_round} - {target_name} ({target_location}) Date: {target_date_str}")
    except ValueError: logger.debug("Attempting name match..."); pass
    except KeyError as e: logger.error(f"Schedule columns error: {e}"); return None, None, None, None

    if target_round is None:
        identifier_lower = str(target_race_identifier).lower()
        try:
            # Prioritize exact matches before contains
            race_match = schedule_df[schedule_df['EventName'].str.lower() == identifier_lower]
            if race_match.empty: race_match = schedule_df[schedule_df['Location'].str.lower() == identifier_lower]
            # Fallback to contains if no exact match
            if race_match.empty: race_match = schedule_df[schedule_df['EventName'].str.lower().str.contains(identifier_lower, na=False)]
            if race_match.empty: race_match = schedule_df[schedule_df['Location'].str.lower().str.contains(identifier_lower, na=False)]

            if not race_match.empty:
                if len(race_match) > 1: logger.warning(f"Multiple matches for '{target_race_identifier}'. Using first: {race_match.iloc[0]['EventName']}")
                target_round = int(race_match.iloc[0]['RoundNumber'])
                target_location = race_match.iloc[0]['Location']
                target_name = race_match.iloc[0]['EventName']
                target_date_str = race_match.iloc[0]['EventDate']
                logger.info(f"Found target race by Name/Location: R{target_round} - {target_name} ({target_location}) Date: {target_date_str}")
            else:
                logger.error(f"Could not find race matching '{target_race_identifier}' for {target_year}.")
                return None, None, None, None
        except KeyError as e: logger.error(f"Schedule columns error: {e}"); return None, None, None, None
        except Exception as e: logger.error(f"Race lookup error: {e}", exc_info=True); return None, None, None, None

    target_date = None
    if target_date_str:
        try:
            # Handle potential 'NaT' strings explicitly before parsing
            if isinstance(target_date_str, str) and target_date_str.lower() != 'nat':
                 target_date = pd.to_datetime(target_date_str).date()
                 logger.info(f"Parsed target date: {target_date}")
            elif pd.isna(target_date_str):
                 logger.warning("EventDate string is NaT/NaN. Cannot parse.")
            else:
                 logger.warning(f"EventDate string '{target_date_str}' is not standard NaT but failed other checks.")

        except Exception as e:
             logger.warning(f"Could not parse EventDate string '{target_date_str}': {e}. Forecast will use defaults.")

    return target_round, target_location, target_name, target_date


def get_historical_weather_averages(track_location):
    logger.info(f"Querying historical weather averages for track: {track_location}")
    # Assume the original query was correct, otherwise adjust here
    query = """
        SELECT
            AVG(AvgAirTemp) as AvgTemp,
            AVG(AvgWindSpeed) as AvgWindSpeed,
            SUM(CASE WHEN MaxRainfall > 0 THEN 1.0 ELSE 0.0 END) * 1.0 / COUNT(*) as AvgRainProb
        FROM (
            SELECT
                e.Location,
                w.Year,
                w.RoundNumber,
                AVG(w.AirTemp) as AvgAirTemp,
                AVG(w.WindSpeed) as AvgWindSpeed,
                MAX(w.Rainfall) as MaxRainfall
            FROM weather w
            JOIN events e ON w.Year = e.Year AND w.RoundNumber = e.RoundNumber
            WHERE w.SessionName = 'R' AND e.Location = :location
            GROUP BY e.Location, w.Year, w.RoundNumber
        ) as RaceWeatherStats
        GROUP BY Location;
    """
    params = {"location": track_location}
    hist_weather = database.load_data(query, params=params)
    if hist_weather is not None and not hist_weather.empty and not hist_weather.isnull().all().all():
         averages = hist_weather.iloc[0].to_dict()
         # Ensure keys match expected forecast column names
         mapped_averages = {
              'ForecastTemp': averages.get('AvgTemp'),
              'ForecastRainProb': averages.get('AvgRainProb'), # Already calculated as avg probability
              'ForecastWindSpeed': averages.get('AvgWindSpeed')
              }
         final_averages = {k: v for k, v in mapped_averages.items() if v is not None and not pd.isna(v)}
         if final_averages: logger.info(f"Found historical weather averages: {final_averages}"); return final_averages
         else: logger.warning("Historical weather query succeeded but returned no valid averages.")
    else: logger.warning(f"Could not find historical weather averages for track: {track_location}")
    return None


def get_feature_fill_value(column_name):
    """Gets the appropriate fill value for a feature column from config."""
    # Use the actual 'default' value from the config dictionary
    fill_val = config.FEATURE_FILL_DEFAULTS.get('default', -999.0) # Use -999.0 or whatever is in config
    # Check for specific patterns
    for pattern, value in config.FEATURE_FILL_DEFAULTS.items():
        if pattern != 'default' and pattern.lower() in column_name.lower():
            fill_val = value
            break # Use first matching pattern
    # Ensure float type if appropriate for the column based on config value type
    if isinstance(fill_val, (int, float)):
        return float(fill_val)
    return fill_val


def prepare_prediction_data(target_year, target_round, target_location, target_date, feature_cols, feature_dtypes):
    """
    Prepares the feature set (X_predict) using cached features and applying necessary updates.
    Handles missing qualifying data, weather forecasts, and ensures dtype consistency.
    Returns:
        tuple: (X_predict, final_info_df, used_grid_fallback_flag)
               Returns (None, None, False) on critical failure.
    """
    logger.info(f"Preparing prediction data for {target_year} R{target_round} ({target_location}).")
    used_grid_fallback = False # --- CHANGE: Flag to track fallback usage ---
    predict_info_df = pd.DataFrame() # Holds driver, team, grid for the target race

    # --- 1. Load Cached Engineered Features ---
    logger.info(f"Attempting to load cached features from: {config.FEATURES_CACHE_PATH}")
    if os.path.exists(config.FEATURES_CACHE_PATH):
        try:
            df_features_all = pd.read_feather(config.FEATURES_CACHE_PATH)
            logger.info(f"Loaded {len(df_features_all)} rows of cached features.")
            # Ensure categorical columns are set correctly after loading
            if 'TrackType' in df_features_all.columns and 'TrackType' in feature_dtypes and feature_dtypes['TrackType'] == 'category':
                logger.debug("Casting loaded TrackType to category.")
                df_features_all['TrackType'] = df_features_all['TrackType'].astype('category')
        except Exception as e:
            logger.error(f"Failed to load cached features: {e}. Prediction cannot proceed without historical data.", exc_info=True)
            return None, None, False
    else:
        logger.error(f"Cached features file not found at {config.FEATURES_CACHE_PATH}. Run training first to generate it. Prediction cannot proceed.")
        return None, None, False


    # --- 2. Get Qualifying Data (for GridPosition and Driver List) ---
    logger.info(f"Attempting to fetch Qualifying data for {target_year} R{target_round}")
    quali_query = "SELECT Abbreviation, TeamName, Position as QualiPositionRaw, GridPosition as OfficialGrid FROM results WHERE Year = :year AND RoundNumber = :round AND SessionName = 'Q'"
    quali_params = {"year": target_year, "round": target_round}
    df_quali_target = database.load_data(quali_query, params=quali_params)

    if df_quali_target is None or df_quali_target.empty: # Check for None from DB load failure too
        logger.warning(f"No Qualifying session data found in DB for target race {target_year} R{target_round}. Triggering fallback.")
        used_grid_fallback = True # --- CHANGE: Set fallback flag ---
    else:
        # Get unique participants from this session
        predict_info_df = df_quali_target[['Abbreviation', 'TeamName']].drop_duplicates().copy()
        predict_info_df['Abbreviation'] = predict_info_df['Abbreviation'].astype(str)
        predict_info_df['TeamName'] = predict_info_df['TeamName'].astype(str)
        logger.info(f"Identified {len(predict_info_df)} participants from Qualifying session.")

        # Try to get official grid first
        df_quali_target['OfficialGrid'] = utils.safe_to_numeric(df_quali_target.get('OfficialGrid'), fallback=np.nan)
        grid_map = df_quali_target.dropna(subset=['OfficialGrid']).drop_duplicates(subset=['Abbreviation'], keep='first').set_index('Abbreviation')['OfficialGrid']

        if not grid_map.empty:
            logger.info("Using official GridPosition from Qualifying results.")
            predict_info_df['GridPosition'] = predict_info_df['Abbreviation'].map(grid_map)
        else:
            logger.warning("Official GridPosition not found in Quali data. Using Quali Position Rank as fallback.")
            df_quali_target['QualiPositionRaw'] = utils.safe_to_numeric(df_quali_target['QualiPositionRaw'], fallback=np.nan)
            min_quali_pos = df_quali_target.groupby('Abbreviation')['QualiPositionRaw'].min()
            df_quali_target['MinQualiPos'] = df_quali_target['Abbreviation'].map(min_quali_pos)
            df_quali_target['QualiRank'] = df_quali_target['MinQualiPos'].rank(method='min', na_option='bottom')
            rank_map = df_quali_target.dropna(subset=['QualiRank']).drop_duplicates(subset=['Abbreviation'], keep='first').set_index('Abbreviation')['QualiRank']
            predict_info_df['GridPosition'] = predict_info_df['Abbreviation'].map(rank_map)

        missing_grid_count = predict_info_df['GridPosition'].isnull().sum()
        if missing_grid_count > 0:
             logger.warning(f"{missing_grid_count} drivers have missing GridPosition after primary Quali methods. Triggering fallback.")
             used_grid_fallback = True # --- CHANGE: Set fallback flag ---
        else:
             logger.info("Successfully assigned GridPosition using primary Quali methods.")


    # --- 3. Get Latest Historical Features for Participating Drivers ---
    logger.info("Extracting latest historical features...")
    previous_race_entries = df_features_all[
        (df_features_all['Year'] < target_year) |
        ((df_features_all['Year'] == target_year) & (df_features_all['RoundNumber'] < target_round))
    ].copy()

    if previous_race_entries.empty:
        logger.warning(f"No historical race data found prior to target {target_year} R{target_round}. Using defaults for historical features.")
        cols_to_extract = ['Abbreviation'] + [f for f in feature_cols if f != 'GridPosition']
        latest_historical_features = pd.DataFrame(columns=cols_to_extract)
        if used_grid_fallback: # If Quali ALSO failed
            logger.error("CRITICAL: No qualifying data AND no previous race data found. Cannot determine drivers for prediction.")
            return None, None, False
    else:
        latest_historical_features = previous_race_entries.loc[previous_race_entries.groupby('Abbreviation').tail(1).index].copy()
        cols_to_merge = ['Abbreviation'] + [col for col in feature_cols if col != 'GridPosition' and col in latest_historical_features.columns]
        latest_historical_features = latest_historical_features[cols_to_merge]

        if used_grid_fallback: # If Quali failed, use drivers from history
             logger.info("Using drivers from last historical race as participants (Quali fallback).")
             predict_info_df = latest_historical_features[['Abbreviation']].copy()
             team_map = previous_race_entries.loc[previous_race_entries.groupby('Abbreviation').tail(1).index].set_index('Abbreviation')['TeamName']
             predict_info_df['TeamName'] = predict_info_df['Abbreviation'].map(team_map).fillna('Unknown')
             predict_info_df['GridPosition'] = np.nan # Grid needs fallback method applied below
             logger.info(f"Identified {len(predict_info_df)} participants from last race (Quali fallback).")


    # --- 4. Merge Historical Features with Target Race Info ---
    if predict_info_df.empty:
         logger.error("CRITICAL: No drivers identified for prediction.")
         return None, None, False

    logger.info(f"Merging historical features for {len(predict_info_df)} drivers...")
    predict_info_df['Abbreviation'] = predict_info_df['Abbreviation'].astype(str)
    latest_historical_features['Abbreviation'] = latest_historical_features['Abbreviation'].astype(str)
    predict_df = pd.merge(predict_info_df, latest_historical_features, on='Abbreviation', how='left')
    logger.info(f"Merged features. Shape: {predict_df.shape}.")


    # --- 5. Apply Grid Position Fallback (if Quali data was missing/incomplete) ---
    # Apply fallback if the flag is set OR if there are still NaNs after primary attempt
    if used_grid_fallback or predict_df['GridPosition'].isnull().any():
        missing_grid_mask = predict_df['GridPosition'].isnull()
        num_missing = missing_grid_mask.sum()
        if num_missing > 0: # Apply only if there are actual NaNs to fill
            logger.warning(f"Applying GridPosition fallback for {num_missing} drivers using method: '{config.GRID_POS_FALLBACK_METHOD}'.")
            method = config.GRID_POS_FALLBACK_METHOD
            fallback_value = config.WORST_EXPECTED_POS # Default

            if method == 'RollingAvg' and 'RollingAvgPosLastN' in predict_df.columns:
                logger.info("Using RollingAvgPosLastN for GridPosition fallback.")
                fallback_values = predict_df.loc[missing_grid_mask, 'RollingAvgPosLastN'].round()
                predict_df.loc[missing_grid_mask, 'GridPosition'] = fallback_values
            elif method == 'MidPack':
                logger.info(f"Using MidPack ({config.GRID_POS_FALLBACK_MIDPACK_VALUE}) for GridPosition fallback.")
                fallback_value = config.GRID_POS_FALLBACK_MIDPACK_VALUE
                predict_df.loc[missing_grid_mask, 'GridPosition'] = fallback_value
            else: # 'Worst' or unspecified
                logger.info(f"Using WorstExpected ({config.WORST_EXPECTED_POS}) for GridPosition fallback.")
                predict_df.loc[missing_grid_mask, 'GridPosition'] = fallback_value

            # Final catch-all fill if the chosen fallback method still resulted in NaNs
            predict_df['GridPosition'].fillna(config.WORST_EXPECTED_POS, inplace=True)
            used_grid_fallback = True # Ensure flag is true if any fallback was applied here
        else:
             logger.info("Grid fallback was triggered, but no NaNs needed filling.")

    # --- Steps 6, 7, 8, 9 remain the same as the previous corrected version ---
    # ... (Assume steps 6-9 are correct as per the last revision) ...
    # --- 6. Handle Missing Historical Features ---
    logger.info("Handling missing historical features...")
    hist_feature_cols = [
        col for col in feature_cols
        if col != 'GridPosition' and not col.startswith('Forecast') and not col.startswith('Track_') and col != 'TrackType'
    ]
    missing_hist_data_mask = predict_df[hist_feature_cols].isnull()
    if missing_hist_data_mask.values.any():
        cols_with_miss = predict_df[hist_feature_cols].columns[missing_hist_data_mask.any()].tolist()
        logger.warning(f"Missing historical feature data detected in: {cols_with_miss}. Filling with defaults.")
        for col in cols_with_miss:
            if col in predict_df.columns:
                fill_val = get_feature_fill_value(col)
                logger.debug(f"Filling NaN in '{col}' with {fill_val}")
                predict_df[col].fillna(fill_val, inplace=True)
            else:
                 logger.error(f"Logic error: Historical column '{col}' expected but not found in predict_df during NaN fill.")

    # --- 7. Add Weather Forecast Features ---
    logger.info("Adding weather forecast features...")
    expected_forecast_cols = ['ForecastTemp', 'ForecastRainProb', 'ForecastWindSpeed']
    forecast_features = {}
    coords = utils.TRACK_COORDINATES.get(target_location)
    if coords and not any(pd.isna(c) for c in coords):
        latitude, longitude = coords
        live_forecast = utils.get_weather_forecast(latitude, longitude, target_date=target_date)
        if live_forecast: forecast_features = live_forecast; logger.info(f"Obtained live forecast data: {forecast_features}")
        else: logger.warning("Failed to get live weather forecast. Attempting historical fallback.")
    else: logger.warning(f"Coordinates not found or invalid for track: {target_location}. Attempting historical fallback for weather.")
    if not forecast_features:
        hist_weather_avg = get_historical_weather_averages(target_location)
        if hist_weather_avg: forecast_features = hist_weather_avg; logger.info(f"Using historical weather averages: {forecast_features}")
        else: logger.warning("Historical weather fallback failed. Using generic defaults.")
    for fc_col in expected_forecast_cols:
        value = forecast_features.get(fc_col)
        if value is None or pd.isna(value): fill_val = get_feature_fill_value(fc_col); logger.warning(f"Assigning default value {fill_val} for missing weather feature '{fc_col}'"); predict_df[fc_col] = fill_val
        else: predict_df[fc_col] = value
        predict_df[fc_col] = predict_df[fc_col].astype(float)

    # --- 8. Add Track Characteristics ---
    logger.info("Adding track characteristic features...")
    try:
        track_type = config.TRACK_CHARACTERISTICS.get(target_location, 'Unknown')
        logger.info(f"TrackType for {target_location}: '{track_type}'")
        is_categorical_model = feature_dtypes.get('TrackType') == 'category'
        expected_ohe_cols = [f for f in feature_cols if f.startswith('Track_')]
        is_ohe_model = bool(expected_ohe_cols)
        if is_categorical_model:
             logger.info("Handling TrackType as categorical feature.")
             if 'TrackType' not in predict_df.columns: predict_df['TrackType'] = 'Unknown'
             if not pd.api.types.is_categorical_dtype(predict_df['TrackType']): predict_df['TrackType'] = predict_df['TrackType'].astype('category')
             if track_type not in predict_df['TrackType'].cat.categories: logger.warning(f"Track type '{track_type}' not in known categories during prediction. Adding it."); predict_df['TrackType'] = predict_df['TrackType'].cat.add_categories([track_type])
             predict_df['TrackType'] = predict_df['TrackType'].fillna('Unknown'); predict_df.loc[:, 'TrackType'] = track_type; predict_df['TrackType'] = predict_df['TrackType'].astype('category')
             logger.info("Set TrackType as category.")
        elif is_ohe_model:
             logger.info("Handling TrackType using One-Hot Encoding.")
             for ohe_col in expected_ohe_cols:
                 if ohe_col not in predict_df.columns: predict_df[ohe_col] = 0
                 else: predict_df[ohe_col] = pd.to_numeric(predict_df[ohe_col], errors='coerce').fillna(0)
                 predict_df[ohe_col] = predict_df[ohe_col].astype(int)
             potential_ohe_col = f"Track_{track_type}"
             if potential_ohe_col in expected_ohe_cols: logger.info(f"Setting OHE column '{potential_ohe_col}' to 1."); predict_df[potential_ohe_col] = 1
             else:
                 logger.warning(f"Constructed OHE column '{potential_ohe_col}' (for track '{track_type}') not found in expected features: {expected_ohe_cols}. Trying 'Track_Unknown'.")
                 unknown_col = 'Track_Unknown';
                 if unknown_col in expected_ohe_cols: logger.info(f"Setting OHE column '{unknown_col}' to 1."); predict_df[unknown_col] = 1
                 else: logger.error(f"Cannot set OHE for current track '{track_type}', and '{unknown_col}' is also missing from expected features! Prediction may be inaccurate.")
             if 'TrackType' in predict_df.columns: predict_df.drop(columns=['TrackType'], inplace=True, errors='ignore')
        else: logger.info("TrackType feature (categorical or OHE) not found in model's expected features. Skipping.");
        if 'TrackType' in predict_df.columns: predict_df.drop(columns=['TrackType'], inplace=True, errors='ignore')
    except Exception as e: logger.error(f"Error handling track characteristic features: {e}", exc_info=True); return None, None, False

    # --- 9. Final Validation and Feature Selection ---
    logger.info("Performing final validation and column alignment...")
    final_missing_cols = [col for col in feature_cols if col not in predict_df.columns]
    if final_missing_cols:
        logger.error(f"CRITICAL: Features missing before final selection: {final_missing_cols}.")
        # Attempt to add defaults
        for col in final_missing_cols: fill_val = get_feature_fill_value(col); logger.warning(f"Adding missing column '{col}' with fill value: {fill_val}"); predict_df[col] = fill_val
        final_missing_cols = [col for col in feature_cols if col not in predict_df.columns] # Re-check
        if final_missing_cols: logger.error(f"CRITICAL: Still missing features after attempting to add defaults: {final_missing_cols}. Cannot proceed."); return None, None, False
    logger.info("Ensuring correct data types based on training dtypes...")
    for col, expected_dtype_str in feature_dtypes.items():
        if col in predict_df.columns:
            try:
                current_dtype = predict_df[col].dtype; logger.debug(f"Checking column '{col}'. Expected: {expected_dtype_str}, Current: {current_dtype}")
                if expected_dtype_str == 'category':
                    if not pd.api.types.is_categorical_dtype(current_dtype): logger.debug(f"Casting column '{col}' to category."); predict_df[col] = predict_df[col].fillna('Unknown').astype('category')
                elif expected_dtype_str.startswith('int') or expected_dtype_str.startswith('float'):
                    target_numpy_type = np.int64 if expected_dtype_str.startswith('int') else np.float64
                    if not np.issubdtype(current_dtype, np.number) or current_dtype != target_numpy_type:
                        fill_val = get_feature_fill_value(col); logger.debug(f"Casting column '{col}' to {target_numpy_type} using fill: {fill_val}")
                        numeric_series = pd.to_numeric(predict_df[col], errors='coerce')
                        if numeric_series.isnull().any(): logger.warning(f"NaNs found in numeric column '{col}' before final cast. Filling with {fill_val}."); numeric_series.fillna(fill_val, inplace=True)
                        predict_df[col] = numeric_series.astype(target_numpy_type)
            except Exception as e: logger.error(f"Failed to cast column '{col}' to expected type '{expected_dtype_str}'. Error: {e}", exc_info=True); return None, None, False
        else: logger.error(f"Column '{col}' required by model but not found during dtype check."); return None, None, False
    if predict_df.columns.has_duplicates: logger.warning(f"Duplicate columns found before final selection: {predict_df.columns[predict_df.columns.duplicated()].tolist()}. Dropping duplicates."); predict_df = predict_df.loc[:, ~predict_df.columns.duplicated(keep='first')]
    try:
        missing_after_dedup = [col for col in feature_cols if col not in predict_df.columns]
        if missing_after_dedup: logger.error(f"Features missing after duplicate removal: {missing_after_dedup}. Cannot proceed."); return None, None, False
        X_predict = predict_df[feature_cols].copy()
        logger.info(f"Final feature set selected and ordered. Shape: {X_predict.shape}")
    except KeyError as e: logger.error(f"KeyError selecting final features: {e}. Model needs: {feature_cols}. Available: {predict_df.columns.tolist()}"); return None, None, False
    except Exception as e: logger.error(f"Error during final feature selection/reordering: {e}", exc_info=True); return None, None, False
    numeric_cols_in_X = X_predict.select_dtypes(include=np.number).columns
    if not numeric_cols_in_X.empty:
        nan_mask = X_predict[numeric_cols_in_X].isnull(); inf_mask = X_predict[numeric_cols_in_X].isin([np.inf, -np.inf])
        if nan_mask.values.any(): nan_cols_final = numeric_cols_in_X[nan_mask.any()].tolist(); logger.warning(f"NaNs still detected in final numeric features: {nan_cols_final}. Filling."); [X_predict[col].fillna(get_feature_fill_value(col), inplace=True) for col in nan_cols_final]
        if inf_mask.values.any(): inf_cols_final = numeric_cols_in_X[inf_mask.any()].tolist(); logger.warning(f"Infs detected in final numeric features: {inf_cols_final}. Replacing."); default_fill = get_feature_fill_value('default'); X_predict.replace([np.inf, -np.inf], default_fill * 100 if default_fill != 0 else -99999, inplace=True)

    logger.info(f"Prediction feature set prepared successfully.")
    # Return the prepared features (X), the info DF, and the fallback flag
    # Cast GridPosition to int *here* after all filling is done
    predict_df['GridPosition'] = predict_df['GridPosition'].astype(int)
    final_info_df = predict_df[['Abbreviation', 'TeamName', 'GridPosition']].copy()

    return X_predict, final_info_df, used_grid_fallback


# --- make_predictions function ---
def make_predictions(target_year, target_race_identifier):
    """Orchestrates the prediction process using cached features."""
    logger.info(f"--- Starting Prediction: {target_year} Race: {target_race_identifier} ---")

    model, feature_cols, feature_dtypes = model_loader.load_model_and_meta()
    if model is None or feature_cols is None or feature_dtypes is None:
        logger.error("Prediction failed: Model, features, or dtypes not loaded.")
        return None

    target_round, target_location, target_name, target_date = get_target_race_info(target_year, target_race_identifier)
    if target_round is None:
        logger.error(f"Prediction failed: Target race '{target_race_identifier}' not identified for {target_year}.")
        return None

    # --- CHANGE: Capture the fallback flag ---
    X_predict, predict_info_df, used_grid_fallback = prepare_prediction_data(
        target_year, target_round, target_location, target_date, feature_cols, feature_dtypes
    )

    # Check if prepare_prediction_data failed
    if X_predict is None or predict_info_df is None:
        logger.error("Prediction failed: Data preparation step failed.")
        return None
    if X_predict.empty or predict_info_df.empty:
        logger.error("Prediction failed: Data preparation resulted in empty dataframes.")
        return None


    logger.info(f"Making predictions for {len(X_predict)} drivers using {len(feature_cols)} features...")
    try:
        if hasattr(model, 'predict_proba'): logger.warning("Loaded model has predict_proba, might be a classifier. Attempting predict anyway.")
        predicted_positions_raw = model.predict(X_predict)
        logger.info("Raw predictions generated.")
        if not pd.api.types.is_numeric_dtype(predicted_positions_raw): logger.warning("Model predictions are not numeric. Attempting conversion."); predicted_positions_raw = pd.to_numeric(predicted_positions_raw, errors='coerce')
        if np.isnan(predicted_positions_raw).any(): nan_count = np.isnan(predicted_positions_raw).sum(); logger.error(f"Model produced {nan_count} NaN predictions!"); predicted_positions_raw = np.nan_to_num(predicted_positions_raw, nan=999.0) # Rank NaNs last

    except Exception as e:
        logger.error(f"Error during model.predict(): {e}", exc_info=True)
        logger.error(f"Prediction data (X_predict) info:\n{X_predict.info()}")
        problematic_cols = X_predict.columns[X_predict.isnull().any()].tolist()
        logger.error(f"X_predict head (check for NaNs/Infs, especially in {problematic_cols}):\n{X_predict.head()}")
        return None

    # --- Process and Format Results ---
    try:
        results_df = predict_info_df.copy()
        if len(results_df) != len(predicted_positions_raw): logger.error(f"Mismatch between info rows ({len(results_df)}) and predictions ({len(predicted_positions_raw)}). Cannot create results."); return None
        results_df['PredictedPosition_Raw'] = predicted_positions_raw
        results_df.sort_values(by='PredictedPosition_Raw', inplace=True, ascending=True, kind='stable', na_position='last')
        results_df['PredictedRank'] = range(1, len(results_df) + 1)
        logger.info("Prediction results processed and ranked.")

        # --- CHANGE: Add warning if fallback grid was used ---
        if used_grid_fallback:
            logger.warning("NOTE: Predictions below are based on FALLBACK Grid Positions (Qualifying data was missing). Accuracy may be reduced.")
            print("\n *** WARNING: Using fallback Grid Positions (Qualifying data missing) ***\n")
        # --- CHANGE END ---

        display_cols = ['PredictedRank', 'Abbreviation', 'TeamName', 'GridPosition', 'PredictedPosition_Raw']
        final_display_cols = [col for col in display_cols if col in results_df.columns]
        final_predictions = results_df[final_display_cols].copy()

        # Final formatting for display
        if 'GridPosition' in final_predictions.columns: final_predictions['GridPosition'] = final_predictions['GridPosition'].astype(str)
        if 'PredictedPosition_Raw' in final_predictions.columns: final_predictions['PredictedPosition_Raw'] = final_predictions['PredictedPosition_Raw'].round(2)
        if 'PredictedRank' in final_predictions.columns: final_predictions['PredictedRank'] = final_predictions['PredictedRank'].astype(int)

        return final_predictions
    except Exception as e:
        logger.error(f"Error formatting results: {e}", exc_info=True)
        return None