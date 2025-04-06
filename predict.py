# /f1_predictor/predict.py

import pandas as pd
import numpy as np
import warnings
import os # For checking model file existence

# Import project modules
import config
import database
import utils
import model as model_loader # Use alias to avoid name clash with model object
import feature_engineering # Need access to feature creation logic/columns

# Setup logger for this module
logger = utils.get_logger(__name__)

# Suppress warnings if needed
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None # Suppress SettingWithCopyWarning


def get_target_race_info(target_year, target_race_identifier):
    """
    Finds the round number, event name, and location for the target race.

    Args:
        target_year (int): The year of the target race.
        target_race_identifier (str or int): The name (partial match) or round number.

    Returns:
        tuple: (target_round, target_location, target_name) or (None, None, None) if not found.
    """
    logger.info(f"Looking up info for target race: Year {target_year}, Identifier '{target_race_identifier}'")
    query = "SELECT RoundNumber, EventName, Location FROM events WHERE Year = :year"
    params = {"year": target_year}
    schedule_df = database.load_data(query, params=params)

    if schedule_df is None or schedule_df.empty:
        logger.error(f"No schedule found for year {target_year} in database.")
        return None, None, None

    target_round = None
    target_location = None
    target_name = None

    # Try matching identifier as integer (RoundNumber) first
    try:
        target_round_input = int(target_race_identifier)
        race_match = schedule_df[schedule_df['RoundNumber'] == target_round_input]
        if not race_match.empty:
            target_round = target_round_input
            target_location = race_match.iloc[0]['Location']
            target_name = race_match.iloc[0]['EventName']
            logger.info(f"Found target race by RoundNumber: R{target_round} - {target_name} ({target_location})")
            return target_round, target_location, target_name
    except ValueError:
        # Identifier is likely a name, proceed to name matching
        logger.debug(f"'{target_race_identifier}' is not a round number, attempting name match.")
        pass
    except KeyError as e:
         logger.error(f"Error accessing schedule columns during round number lookup: {e}")
         return None, None, None # Cannot proceed if schedule columns missing


    # Match by EventName or Location (case-insensitive partial match)
    identifier_lower = str(target_race_identifier).lower()
    logger.debug(f"Attempting name match for: '{identifier_lower}'")

    try:
        # Prioritize EventName match
        race_match = schedule_df[schedule_df['EventName'].str.lower().str.contains(identifier_lower, na=False)]
        if race_match.empty:
            # Fallback to Location match
            logger.debug("No match on EventName, trying Location.")
            race_match = schedule_df[schedule_df['Location'].str.lower().str.contains(identifier_lower, na=False)]

        if not race_match.empty:
            if len(race_match) > 1:
                logger.warning(f"Multiple potential matches found for '{target_race_identifier}'. Using the first one: {race_match.iloc[0]['EventName']}")
            target_round = int(race_match.iloc[0]['RoundNumber'])
            target_location = race_match.iloc[0]['Location']
            target_name = race_match.iloc[0]['EventName']
            logger.info(f"Found target race by Name/Location: R{target_round} - {target_name} ({target_location})")
            return target_round, target_location, target_name
        else:
            logger.error(f"Could not find race matching identifier '{target_race_identifier}' in schedule for {target_year}.")
            return None, None, None
    except KeyError as e:
         logger.error(f"Error accessing schedule columns during name/location lookup: {e}")
         return None, None, None
    except Exception as e:
         logger.error(f"Unexpected error during race info lookup: {e}", exc_info=True)
         return None, None, None


def prepare_prediction_data(target_year, target_round, target_location, feature_cols):
    """
    Prepares the feature set (X_predict) for the drivers in the target race.

    Args:
        target_year (int): Year of the race.
        target_round (int): Round number of the race.
        target_location (str): Location (track identifier) of the race.
        feature_cols (list): List of feature column names expected by the model.

    Returns:
        tuple: (X_predict, predict_info_df)
               X_predict (pd.DataFrame): Features ready for model.predict().
               predict_info_df (pd.DataFrame): DataFrame with driver info and features.
               Returns (None, None) on critical failure.
    """
    if target_round is None or target_location is None:
        logger.error("Invalid target race info (round/location). Cannot prepare prediction data.")
        return None, None
    if not feature_cols:
        logger.error("No feature columns provided. Cannot prepare prediction data.")
        return None, None

    logger.info(f"Preparing prediction data for {target_year} Round {target_round} ({target_location})")

    # --- 1. Get Qualifying Data ---
    quali_query = """ SELECT Abbreviation, TeamName, Position as QualiPosition FROM results WHERE Year = :year AND RoundNumber = :round AND SessionName = 'Q' ORDER BY Position """
    params = {"year": target_year, "round": target_round}
    target_quali_results = database.load_data(quali_query, params=params)

    if target_quali_results is None or target_quali_results.empty:
        logger.error(f"Missing Qualifying data for target race ({target_year} R{target_round}) in database. Cannot predict.")
        return None, None

    # Create base prediction DataFrame from qualifying results
    predict_df = target_quali_results[['Abbreviation', 'TeamName', 'QualiPosition']].copy()
    predict_df.rename(columns={'QualiPosition': 'GridPosition'}, inplace=True)
    predict_df['GridPosition'] = utils.safe_to_numeric(predict_df['GridPosition'], fallback=config.WORST_EXPECTED_POS).astype(int)
    logger.info(f"Found {len(predict_df)} drivers in qualifying for target race.")

    # --- 2. Get Latest Historical Features ---
    logger.info("Regenerating historical features to get latest driver state...")
    all_features_df, _, _ = feature_engineering.create_features() # Use the consistent feature engineering logic

    if all_features_df is None or all_features_df.empty:
        logger.error("Failed to generate historical features for prediction context. Cannot proceed.")
        return None, None

    # Filter features to be strictly before the target race/round
    latest_features_df = all_features_df[
        (all_features_df['Year'] < target_year) |
        ((all_features_df['Year'] == target_year) & (all_features_df['RoundNumber'] < target_round))
    ].copy()

    if latest_features_df.empty:
         logger.warning(f"No historical feature data found prior to {target_year} R{target_round}. Predictions may be based only on GridPosition/Defaults.")
         # Create an empty df with expected columns to allow merge/fillna later
         latest_driver_features = pd.DataFrame(columns=['Abbreviation'] + feature_cols)
         # Ensure essential columns used for merge exist
         if 'Abbreviation' not in latest_driver_features.columns: latest_driver_features['Abbreviation'] = None
    else:
        # Get the absolute most recent record for each driver *before* the target race
        latest_features_df.sort_values(by=['Year', 'RoundNumber'], ascending=False, inplace=True)
        latest_driver_features = latest_features_df.drop_duplicates(subset=['Abbreviation'], keep='first')
        logger.info(f"Found latest historical features for {len(latest_driver_features)} unique drivers.")

    # --- 3. Merge Historical Features ---
    # Define columns to merge from historical data (exclude target, GridPos, OHE Tracks)
    historical_cols_to_merge = [
        col for col in feature_cols
        if col in latest_driver_features.columns and col != 'GridPosition' and not col.startswith('Track_')
    ]
    logger.debug(f"Merging historical features: {historical_cols_to_merge}")

    # Merge historical features onto the prediction set (drivers from qualifying)
    # Ensure Abbreviation exists in latest_driver_features before merge
    if 'Abbreviation' not in latest_driver_features.columns:
        logger.error("Abbreviation column missing in latest historical features. Cannot merge.")
        return None, None

    predict_df = pd.merge(
        predict_df,
        latest_driver_features[['Abbreviation'] + historical_cols_to_merge],
        on='Abbreviation',
        how='left' # Keep all drivers from qualifying, fill missing history with NaN
    )
    logger.info(f"Merged historical features. predict_df shape after merge: {predict_df.shape}")
    logger.debug(f"Columns after merge: {predict_df.columns.tolist()}")


    # --- 4. Handle Missing Data (Imputation) ---
    # Define default/fallback values for prediction context
    fill_values_predict = {
        'RollingAvgPosLastN': float(config.WORST_EXPECTED_POS),
        'RollingAvgPtsLastN': 0.0,
        'ExpandingAvgPosThisTrack': float(config.WORST_EXPECTED_POS),
        'ExpandingAvgPtsThisTrack': 0.0,
        'RollingAvgTeamPtsLastN': 0.0,
        'RaceCount': 0, # Assume 0 races if no history
        'SeasonPointsBeforeRace': 0.0,
        # Other numerical features default to config.FILL_NA_VALUE
    }
    cols_filled = []
    for col in feature_cols:
        # Only fill columns that are actually features (excluding GridPosition already handled)
        if col != 'GridPosition' and col in predict_df.columns and predict_df[col].isnull().any():
            # Use specific fill value if defined, otherwise use the general numeric fill value
            fill_val = fill_values_predict.get(col, config.FILL_NA_VALUE)
            # Ensure fill value has correct type (e.g., float for numeric columns)
            try:
                 if pd.api.types.is_numeric_dtype(predict_df[col]):
                      fill_val = float(fill_val)
            except: pass # Ignore type conversion errors if fill_val isn't numeric

            predict_df[col].fillna(fill_val, inplace=True)
            cols_filled.append(f"{col} (filled with {fill_val})")

    if cols_filled:
        logger.info(f"Filled NaNs in prediction features: {', '.join(cols_filled)}")


    # --- 5. Handle Categorical Features (TrackType) ---
    try:
        track_type = config.TRACK_CHARACTERISTICS.get(target_location, 'Unknown')
        logger.info(f"Target track location '{target_location}' mapped to TrackType: '{track_type}'")

        # Check if TrackType or Track_ OHE columns are expected features
        is_categorical_model = config.MODEL_TYPE.lower() in ['lightgbm'] # Add other models if needed
        is_ohe_model = any(f.startswith('Track_') for f in feature_cols)

        if is_categorical_model and 'TrackType' in feature_cols:
            predict_df['TrackType'] = track_type
            predict_df['TrackType'] = predict_df['TrackType'].astype('category')
            # TODO: Ensure categories match training categories if possible
            logger.debug("Set TrackType as category for prediction.")
        elif is_ohe_model:
            predict_df['TrackType'] = track_type # Add the column first
            # Use pandas.get_dummies to create the OHE column for the current track
            predict_df = pd.get_dummies(predict_df, columns=['TrackType'], prefix='Track', dummy_na=False)
            logger.debug(f"Applied OHE for TrackType '{track_type}'. Columns now: {predict_df.columns.tolist()}")

            # Add any missing OHE track columns expected by the model, fill with 0
            missing_ohe_cols = []
            for expected_col in feature_cols:
                 if expected_col.startswith('Track_') and expected_col not in predict_df.columns:
                     predict_df[expected_col] = 0 # OHE flags are 0 if not the current track type
                     missing_ohe_cols.append(expected_col)
            if missing_ohe_cols:
                logger.info(f"Added missing OHE columns expected by model: {missing_ohe_cols}")
        else:
             logger.debug("TrackType feature not used or not OHE for this model type.")
    except Exception as e:
        logger.error(f"Error handling categorical features for prediction: {e}", exc_info=True)
        return None, None


    # --- 6. Final Feature Selection and Validation ---
    # Ensure all required feature columns exist *before* selecting
    final_missing_cols = [col for col in feature_cols if col not in predict_df.columns]
    if final_missing_cols:
        logger.error(f"CRITICAL: Final required feature columns missing before selection: {final_missing_cols}. Cannot predict.")
        logger.debug(f"Columns available in predict_df: {predict_df.columns.tolist()}")
        return None, None

    # Select columns in the exact order required by the model
    try:
        X_predict = predict_df[feature_cols].copy()
        logger.info(f"Final feature set selected for prediction. Shape: {X_predict.shape}")
    except KeyError as e:
        logger.error(f"KeyError selecting final feature columns: {e}. Model expects: {feature_cols}. Available: {predict_df.columns.tolist()}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error selecting final features: {e}", exc_info=True)
        return None, None

    # --- Corrected Final Validation (on numeric cols only) ---
    try:
        numeric_cols_in_X = X_predict.select_dtypes(include=np.number).columns
        if not numeric_cols_in_X.empty:
            logger.debug(f"Performing final NaN/Inf check on numeric columns: {numeric_cols_in_X.tolist()}")

            # Check and fill NaNs
            nan_mask_num = X_predict[numeric_cols_in_X].isnull()
            if nan_mask_num.values.any():
                logger.warning("NaN values detected in final numeric prediction features! Applying final fill.")
                nan_cols_final = numeric_cols_in_X[nan_mask_num.any()].tolist()
                logger.warning(f"NaNs found in numeric columns: {nan_cols_final}")
                for col in nan_cols_final:
                    X_predict[col].fillna(config.FILL_NA_VALUE, inplace=True)

            # Check and replace Infs
            inf_mask_num = np.isinf(X_predict[numeric_cols_in_X].values)
            if inf_mask_num.any():
                logger.warning("Infinity values detected in final numeric prediction features! Replacing.")
                # Identify columns with inf
                inf_cols_final = numeric_cols_in_X[np.isinf(X_predict[numeric_cols_in_X]).any()].tolist()
                logger.warning(f"Infs found in numeric columns: {inf_cols_final}")
                for col in inf_cols_final:
                     X_predict[col].replace([np.inf, -np.inf], config.FILL_NA_VALUE * 1000, inplace=True) # Use large number
        else:
             logger.debug("No numeric columns found in X_predict for final NaN/Inf check.")

    except Exception as e:
        logger.error(f"Error during final NaN/Inf check: {e}", exc_info=True)
        # Decide whether to proceed with potential issues or return None
        # return None, None # Option: Be strict and fail

    logger.info(f"Prediction feature set prepared successfully.")
    return X_predict, predict_df # Return features and the dataframe with driver info


def make_predictions(target_year, target_race_identifier):
    """
    Orchestrates the prediction process: loads model, prepares data, predicts, formats.

    Args:
        target_year (int): Year of the race.
        target_race_identifier (str or int): Race name or round number.

    Returns:
        pd.DataFrame or None: DataFrame with predictions or None on failure.
    """
    logger.info(f"--- Starting Prediction Process for {target_year} Race: {target_race_identifier} ---")

    # --- 1. Load Model ---
    model = model_loader.load_model()
    if model is None:
        logger.error("Prediction failed: Could not load the trained model.")
        return None

    # --- 2. Get Target Race Info ---
    target_round, target_location, target_name = get_target_race_info(target_year, target_race_identifier)
    if target_round is None:
        logger.error(f"Prediction failed: Could not identify target race '{target_race_identifier}' for year {target_year}.")
        return None

    # --- 3. Infer Feature Columns from Model ---
    # Use feature_names_in_ attribute if available (preferred)
    feature_cols = None
    if hasattr(model, 'feature_names_in_'):
        try:
            feature_cols = model.feature_names_in_.tolist()
            logger.info(f"Inferred {len(feature_cols)} feature columns from loaded model.")
            logger.debug(f"Model features: {feature_cols}")
        except Exception as e:
            logger.warning(f"Could not get feature names from model attribute: {e}. Falling back to feature_engineering.")
            feature_cols = None # Reset feature_cols

    if feature_cols is None:
        # Fallback: Regenerate features to get the column list (less reliable)
        logger.warning("Falling back to feature_engineering module to get feature column names.")
        try:
            _, feature_cols_regen, _ = feature_engineering.create_features()
            if not feature_cols_regen:
                 logger.error("Prediction failed: Fallback feature_engineering failed to return feature columns.")
                 return None
            feature_cols = feature_cols_regen
            logger.info(f"Using {len(feature_cols)} feature columns from feature_engineering module.")
        except Exception as e:
             logger.error(f"Prediction failed: Error running feature_engineering for fallback: {e}", exc_info=True)
             return None

    # --- 4. Prepare Prediction Data ---
    X_predict, predict_info_df = prepare_prediction_data(target_year, target_round, target_location, feature_cols)

    if X_predict is None or predict_info_df is None:
        logger.error("Prediction failed: Could not prepare prediction data.")
        return None

    # Final check: Ensure X_predict columns match feature_cols exactly
    if list(X_predict.columns) != feature_cols:
        logger.error("CRITICAL: Column mismatch or order difference between X_predict and expected features just before prediction.")
        logger.error(f"X_predict columns: {list(X_predict.columns)}")
        logger.error(f"Expected features: {feature_cols}")
        # Attempt to reorder X_predict if all columns are present but maybe in wrong order
        try:
            logger.warning("Attempting to reorder X_predict columns...")
            X_predict = X_predict[feature_cols]
        except Exception as e:
            logger.error(f"Failed to reorder columns: {e}. Aborting prediction.")
            return None

    # --- 5. Predict ---
    logger.info(f"Making predictions for {len(X_predict)} drivers...")
    try:
        # Ensure input is purely numeric if required by model (should be after validation)
        # X_predict_numeric = X_predict.select_dtypes(include=np.number) # If model strictly needs numeric only
        predicted_positions_raw = model.predict(X_predict)
        logger.info("Raw predictions generated successfully.")
    except Exception as e:
        logger.error(f"Error during model.predict(): {e}", exc_info=True)
        logger.error(f"Input data shape: {X_predict.shape}")
        logger.error(f"Input data types:\n{X_predict.dtypes.to_string()}")
        logger.error(f"Input data head:\n{X_predict.head().to_string()}")
        return None

    # --- 6. Format Results ---
    try:
        predict_info_df['PredictedPosition_Raw'] = predicted_positions_raw
        # Sort by the predicted float value (lower is better for position)
        predict_info_df.sort_values(by='PredictedPosition_Raw', inplace=True, ascending=True, kind='stable') # Use stable sort
        # Assign integer ranks based on the sorted order
        predict_info_df['PredictedRank'] = range(1, len(predict_info_df) + 1)
        logger.info("Prediction results processed and ranked.")

        # Select and order columns for the final output DataFrame
        display_cols = ['PredictedRank', 'Abbreviation', 'TeamName', 'GridPosition', 'PredictedPosition_Raw']
        # Optionally add some key input features for context
        # context_features = ['RollingAvgPosLastN', 'RollingAvgPtsLastN', 'ExpandingAvgPosThisTrack']
        # display_cols.extend([f for f in context_features if f in predict_info_df.columns])

        # Select only columns that actually exist in the dataframe
        final_predictions = predict_info_df[[col for col in display_cols if col in predict_info_df.columns]].copy()

        return final_predictions
    except Exception as e:
        logger.error(f"Error formatting prediction results: {e}", exc_info=True)
        # Return raw predictions if formatting fails? Or None?
        # Returning raw predictions attached to predict_info_df might be useful
        if 'PredictedPosition_Raw' in predict_info_df.columns:
            logger.warning("Returning prediction info with raw predictions due to formatting error.")
            return predict_info_df[['Abbreviation', 'TeamName', 'GridPosition', 'PredictedPosition_Raw']]
        else:
            return None