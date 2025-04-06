# /f1_predictor/feature_engineering.py

import pandas as pd
import numpy as np
import config
import database
import utils

logger = utils.get_logger(__name__)

def add_rolling_features(df, group_col, target_col, window, name_suffix, fill_value):
    """Adds rolling average features after shifting, handles fill value."""
    shifted_col = f'Shifted_{target_col}'
    rolling_col = f'RollingAvg{name_suffix}'
    df[shifted_col] = df.groupby(group_col)[target_col].shift(1)
    # Calculate rolling mean on the shifted column
    df[rolling_col] = df.groupby(group_col)[shifted_col].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
    # Fill NaNs created by shift/rolling ONLY - NaNs already present in target_col should remain if not handled earlier
    df[rolling_col].fillna(fill_value, inplace=True)
    df.drop(columns=[shifted_col], inplace=True) # Drop intermediate shifted column
    logger.debug(f"Added rolling feature: {rolling_col}")
    return df

def add_expanding_features(df, group_cols, target_col, name_suffix, fill_value):
    """Adds expanding average features after shifting, handles fill value."""
    shifted_col = f'Shifted_{target_col}'
    expanding_col = f'ExpandingAvg{name_suffix}'
    df[shifted_col] = df.groupby(group_cols)[target_col].shift(1)
    # Calculate expanding mean on the shifted column
    # Note: reset_index for expanding needs careful handling with multiple group_cols
    if isinstance(group_cols, list) and len(group_cols) > 1:
        df[expanding_col] = df.groupby(group_cols)[shifted_col].expanding(min_periods=1).mean().reset_index(level=list(range(len(group_cols))), drop=True)
    else: # Single group column or already indexed correctly
         df[expanding_col] = df.groupby(group_cols)[shifted_col].expanding(min_periods=1).mean().reset_index(level=0, drop=True)

    df[expanding_col].fillna(fill_value, inplace=True)
    df.drop(columns=[shifted_col], inplace=True)
    logger.debug(f"Added expanding feature: {expanding_col}")
    return df

def create_features():
    """Loads data from DB and engineers features for modeling."""
    logger.info("Starting feature engineering...")

    # --- Load Base Data ---
    # Load results joined with event location for track features
    query = f"""
        SELECT r.*, e.Location, e.EventName
        FROM results r
        JOIN events e ON r.Year = e.Year AND r.RoundNumber = e.RoundNumber
        WHERE r.SessionName IN ('R', 'Q') -- Load Race and relevant Quali results
        ORDER BY r.Year, r.RoundNumber -- Ensure chronological order
    """
    df_raw = database.load_data(query)

    if df_raw.empty:
        logger.error("No data loaded from database for feature engineering. Exiting.")
        return pd.DataFrame(), [], None # Return empty structure

    logger.info(f"Loaded {len(df_raw)} Qualifying/Race results from DB.")

    # --- Basic Cleaning & Preparation ---
    # Pivot Qualifying data to get QualiPosition per driver per race
    df_quali = df_raw[df_raw['SessionName'] == 'Q'].pivot_table(
        index=['Year', 'RoundNumber', 'Abbreviation'],
        values='Position',
        aggfunc='min' # Take best quali position if multiple entries (unlikely)
    ).rename(columns={'Position': 'QualiPosition'}).reset_index()

    # Filter for Race results
    df_race = df_raw[df_raw['SessionName'] == 'R'].copy()
    logger.info(f"Processing {len(df_race)} Race results.")

    # Merge QualiPosition onto Race results
    df = pd.merge(df_race, df_quali, on=['Year', 'RoundNumber', 'Abbreviation'], how='left')

    # Handle missing QualiPosition & ensure numeric types, using config fallbacks
    df['GridPosition'] = utils.safe_to_numeric(df['GridPosition'], fallback=None) # Try race grid first
    df['GridPosition'].fillna(df['QualiPosition'], inplace=True) # Fallback to Quali
    df['GridPosition'] = utils.safe_to_numeric(df['GridPosition'], fallback=config.WORST_EXPECTED_POS).astype(int) # Final fallback

    # Handle Position/Points - treat non-classified as worst position
    df['Position'] = utils.safe_to_numeric(df['Position'], fallback=config.WORST_EXPECTED_POS).astype(int)
    df['Points'] = utils.safe_to_numeric(df['Points'], fallback=0.0).astype(float)

    # Use Location for track characteristics (often more stable than EventName)
    df['TrackLocation'] = df['Location'].fillna('Unknown') # Use Location column from events table

    # Sort for time-based calculations
    df.sort_values(by=['Year', 'RoundNumber', 'Abbreviation'], inplace=True)
    df.reset_index(drop=True, inplace=True) # Reset index after sort

    logger.info("Base data prepared. Starting feature calculation...")

    # --- Feature Creation ---

    # 1. Rolling Driver Performance (Last N Races) - uses prior races
    df = add_rolling_features(df, 'Abbreviation', 'Position', config.N_LAST_RACES_FEATURES, 'PosLastN', config.WORST_EXPECTED_POS)
    df = add_rolling_features(df, 'Abbreviation', 'Points', config.N_LAST_RACES_FEATURES, 'PtsLastN', 0.0)

    # 2. Track Specific Performance (Expanding average over previous races at this track location)
    df = add_expanding_features(df, ['Abbreviation', 'TrackLocation'], 'Position', 'PosThisTrack', config.WORST_EXPECTED_POS)
    df = add_expanding_features(df, ['Abbreviation', 'TrackLocation'], 'Points', 'PtsThisTrack', 0.0)

    # 3. Rolling Team Performance
    # Calculate avg points per team per race first
    team_points = df.groupby(['Year', 'RoundNumber', 'TeamName'])['Points'].mean().reset_index()
    team_points.sort_values(by=['Year', 'RoundNumber', 'TeamName'], inplace=True)
    # Apply rolling feature calculation to the team averages
    team_points = add_rolling_features(team_points, 'TeamName', 'Points', config.N_LAST_RACES_TEAM, 'TeamPtsLastN', 0.0)
    # Merge back team features (use Year, RoundNumber, TeamName as keys)
    df = pd.merge(df, team_points[['Year', 'RoundNumber', 'TeamName', 'RollingAvgTeamPtsLastN']],
                  on=['Year', 'RoundNumber', 'TeamName'], how='left')
    # Fill NaNs that result from the merge (e.g., first race for a team)
    df['RollingAvgTeamPtsLastN'].fillna(0.0, inplace=True)
    logger.debug("Added rolling team points feature.")

    # 4. Driver Experience (Simple count of previous races in the dataset)
    df['RaceCount'] = df.groupby('Abbreviation').cumcount() # Starts from 0
    logger.debug("Added race count feature.")

    # 5. Championship Standings (Points accumulated *before* this race in the season)
    # Shift points within each Year/Abbreviation group, fill first race NaN with 0, then compute cumulative sum
    df['SeasonPointsBeforeRace'] = df.groupby(['Year', 'Abbreviation'])['Points'].shift(1).fillna(0)
    df['SeasonPointsBeforeRace'] = df.groupby(['Year', 'Abbreviation'])['SeasonPointsBeforeRace'].cumsum()
    logger.debug("Added season points before race feature.")

    # 6. Track Characteristics (Categorical)
    # Map Location to defined characteristics, fill unknowns
    df['TrackType'] = df['TrackLocation'].map(config.TRACK_CHARACTERISTICS).fillna('Unknown')
    logger.debug("Added track type feature.")

    # --- Final Data Cleaning for Features ---
    # Replace any remaining infinite values if they somehow occur
    df.replace([np.inf, -np.inf], config.FILL_NA_VALUE, inplace=True)
    # Fill any remaining NaNs in numerical columns with the defined fill value
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(config.FILL_NA_VALUE)


    # --- Define Feature Columns ---
    feature_cols = [
        'GridPosition',
        'RollingAvgPosLastN',
        'RollingAvgPtsLastN',
        'ExpandingAvgPosThisTrack',
        'ExpandingAvgPtsThisTrack',
        'RollingAvgTeamPtsLastN',
        'RaceCount',
        'SeasonPointsBeforeRace',
        # TrackType will be handled based on model type below
    ]

    # Handle Categorical TrackType based on Model Choice
    categorical_features = ['TrackType']
    if config.MODEL_TYPE in ['LightGBM']: # Models that handle categoricals directly
        for col in categorical_features:
            df[col] = df[col].astype('category') # Ensure correct dtype
        feature_cols.extend(categorical_features)
        logger.info(f"Using categorical features directly for {config.MODEL_TYPE}: {categorical_features}")
    else: # Models requiring One-Hot Encoding (RandomForest, XGBoost)
        logger.info(f"One-hot encoding features for {config.MODEL_TYPE}: {categorical_features}")
        # Use pandas.get_dummies, handle potential unknown categories if needed later
        df = pd.get_dummies(df, columns=categorical_features, prefix='Track', dummy_na=False)
        # Add the generated dummy columns to the feature list
        ohe_cols = [col for col in df.columns if col.startswith('Track_')]
        feature_cols.extend(ohe_cols)
        logger.debug(f"Added OHE columns: {ohe_cols}")


    # --- Target Variable ---
    target_col = 'Position' # Predicting finishing position

    # --- Final Validation ---
    # Check for NaN/inf values in the final feature set + target
    check_cols = feature_cols + [target_col]
    nan_check = df[check_cols].isnull().sum()
    inf_check = df[check_cols].isin([np.inf, -np.inf]).sum()
    nan_cols = nan_check[nan_check > 0]
    inf_cols = inf_check[inf_check > 0]

    if not nan_cols.empty:
        logger.warning(f"NaN values remain AFTER cleaning in columns:\n{nan_cols}")
        # Optionally drop rows with NaNs in features/target if they persist
        # df.dropna(subset=check_cols, inplace=True)
    if not inf_cols.empty:
        logger.warning(f"Infinity values remain AFTER cleaning in columns:\n{inf_cols}")
        # Replace again just in case
        df.replace([np.inf, -np.inf], config.FILL_NA_VALUE, inplace=True)


    logger.info(f"Feature engineering complete. DataFrame shape: {df.shape}")
    logger.info(f"Target column: {target_col}")
    logger.info(f"Feature columns selected ({len(feature_cols)}): {feature_cols}")

    # Return only necessary columns for modeling
    # Ensure target and all feature columns are present
    final_cols_to_keep = list(set(['Year', 'RoundNumber', 'Abbreviation', 'TeamName', target_col] + feature_cols))
    # Make sure all feature_cols are actually in the dataframe
    final_cols_to_keep = [col for col in final_cols_to_keep if col in df.columns]
    df_final = df[final_cols_to_keep].copy()

    return df_final, feature_cols, target_col