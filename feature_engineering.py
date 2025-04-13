# /f1_predictor/feature_engineering.py

import pandas as pd
import numpy as np
import config
import database
import utils

logger = utils.get_logger(__name__)

# --- Helper Functions ---
def calculate_stint_stats(group):
    # (Keep this function as is)
    group = group.copy()
    group['LapTime'] = pd.to_numeric(group['LapTime'], errors='coerce')
    group['IsAccurate'] = pd.to_numeric(group['IsAccurate'], errors='coerce').fillna(0).astype(int)
    accurate_laps = group[(group['IsAccurate'] == 1) & (~group['LapTime'].isna())]
    if len(accurate_laps) < 2: return pd.Series({'AvgLapTime': np.nan, 'StdDevLapTime': np.nan, 'LapCount': len(accurate_laps), 'Degradation': np.nan, 'Compound': group['Compound'].iloc[0] if not group.empty else 'UNKNOWN'})
    avg_time = accurate_laps['LapTime'].mean()
    std_dev_time = accurate_laps['LapTime'].std()
    lap_count = len(accurate_laps)
    degradation = np.nan
    n_deg = config.DEGRADATION_LAP_COUNT
    if lap_count >= max(2, (2 * n_deg)):
        try:
            first_laps, last_laps = accurate_laps.iloc[:n_deg]['LapTime'], accurate_laps.iloc[-n_deg:]['LapTime']
            if not first_laps.isna().any() and not last_laps.isna().any(): degradation = last_laps.mean() - first_laps.mean()
            else: logger.debug(f"NaNs found in degradation slice for stint."); degradation = np.nan
        except IndexError: logger.debug(f"IndexError during degradation calc (LapCount={lap_count}, n_deg={n_deg})"); degradation = np.nan
        except Exception as e: logger.warning(f"Unexpected error calculating degradation: {e}"); degradation = np.nan
    return pd.Series({'AvgLapTime': avg_time, 'StdDevLapTime': std_dev_time, 'LapCount': lap_count, 'Degradation': degradation, 'Compound': group['Compound'].iloc[0]})


def _load_data_for_features():
    """Loads all necessary data tables from the database."""
    logger.info("Loading data for feature engineering...")
    query_results = "SELECT r.*, e.Location, e.EventName, e.EventDate FROM results r JOIN events e ON r.Year = e.Year AND r.RoundNumber = e.RoundNumber WHERE r.SessionName IN ('R', 'Q') ORDER BY r.Year, r.RoundNumber"
    df_raw_results = database.load_data(query_results)
    if df_raw_results.empty:
        logger.error("CRITICAL: No Race/Quali results found.")
        return None, None, None, None

    df_laps = pd.DataFrame(); df_weather = pd.DataFrame(); df_pits = pd.DataFrame()
    if config.LOAD_LAPS:
        df_laps = database.load_data("SELECT Year, RoundNumber, SessionName, DriverNumber, LapNumber, LapTime, Stint, Compound, IsAccurate, IsPitOutLap, IsPitInLap FROM laps")
        if not df_laps.empty: logger.info(f"Loaded {len(df_laps)} laps.")
        else: logger.warning("No lap data loaded.")

        if database.table_exists('pit_stops'):
            df_pits = database.load_data("SELECT Year, RoundNumber, SessionName, DriverNumber, StopNumber FROM pit_stops WHERE SessionName='R'")
            if not df_pits.empty: logger.info(f"Loaded {len(df_pits)} race pit stops.")
            else: logger.warning("Pit stop table exists but returned no Race pit stop data.")
        else: logger.warning("Pit stop table ('pit_stops') does not exist.")

    if config.LOAD_WEATHER:
        df_weather = database.load_data("SELECT Year, RoundNumber, SessionName, AirTemp, TrackTemp, Humidity, Pressure, WindSpeed, WindDirection, Rainfall FROM weather WHERE SessionName='R'")
        if not df_weather.empty: logger.info(f"Loaded {len(df_weather)} race weather records.")
        else: logger.warning("No Race weather data loaded.")

    return df_raw_results, df_laps, df_weather, df_pits


def _prepare_base_results(df_raw_results):
    """Prepares the base DataFrame with Race results merged with Quali position."""
    logger.info("Preparing base results data...")
    df_quali_raw = df_raw_results[df_raw_results['SessionName'] == 'Q'].copy()
    df_quali_raw['QualiPositionRaw'] = utils.safe_to_numeric(df_quali_raw['Position'], fallback=np.nan)
    min_quali_indices = df_quali_raw.groupby(['Year', 'RoundNumber', 'Abbreviation'])['QualiPositionRaw'].idxmin().dropna()
    if min_quali_indices.empty:
        logger.warning("No valid Qualifying positions found. QualiPosition feature will be NaN.")
        df_quali = pd.DataFrame(columns=['Year', 'RoundNumber', 'Abbreviation', 'QualiPosition'])
    else:
        df_quali = df_quali_raw.loc[min_quali_indices].rename(columns={'QualiPositionRaw': 'QualiPosition'})[['Year', 'RoundNumber', 'Abbreviation', 'QualiPosition']].reset_index(drop=True)
        df_quali['QualiPosition'] = utils.safe_to_numeric(df_quali['QualiPosition'], fallback=np.nan).astype(float)

    df_race = df_raw_results[df_raw_results['SessionName'] == 'R'].copy()
    if df_race.empty:
        logger.error("CRITICAL: No Race results found.")
        return None

    df = pd.merge(df_race, df_quali, on=['Year', 'RoundNumber', 'Abbreviation'], how='left')

    # Clean key columns
    df['DriverNumber'] = df['DriverNumber'].astype(str).fillna('UNK')
    df['QualiPosition'] = utils.safe_to_numeric(df.get('QualiPosition'), fallback=np.nan).astype(float) # Ensure exists and is float
    df['GridPosition'] = utils.safe_to_numeric(df['GridPosition'], fallback=np.nan)
    df['GridPosition'].fillna(df['QualiPosition'], inplace=True)
    df['GridPosition'] = utils.safe_to_numeric(df['GridPosition'], fallback=config.WORST_EXPECTED_POS).astype(int)

    df['Position'] = utils.safe_to_numeric(df['Position'], fallback=config.WORST_EXPECTED_POS).astype(int) # Target variable
    df['Points'] = utils.safe_to_numeric(df['Points'], fallback=0.0).astype(float)
    df['TrackLocation'] = df['Location'].fillna('Unknown').astype(str)
    df['TeamName'] = df['TeamName'].astype(str).fillna('Unknown')

    # Add EventDate for weather forecast logic later
    df['EventDate'] = pd.to_datetime(df['EventDate'], errors='coerce').dt.date

    return df


def _add_practice_features(df, df_laps):
    """Adds features derived from Free Practice lap data."""
    new_cols = []
    if not config.LOAD_LAPS or df_laps.empty:
        logger.info("Skipping practice lap features (LOAD_LAPS=False or no data).")
        return df, new_cols

    logger.info("Calculating features from practice lap data...");
    laps_prac = df_laps[df_laps['SessionName'].isin(['FP1', 'FP2', 'FP3'])].copy()
    if laps_prac.empty:
        logger.warning("No practice lap data found.")
        return df, new_cols

    laps_prac['LapTime'] = pd.to_numeric(laps_prac['LapTime'], errors='coerce')
    laps_prac['Stint'] = pd.to_numeric(laps_prac['Stint'], errors='coerce').fillna(-1).astype(int)
    laps_prac['IsAccurate'] = pd.to_numeric(laps_prac['IsAccurate'], errors='coerce').fillna(0).astype(int)
    laps_prac['DriverNumber'] = laps_prac['DriverNumber'].astype(str).fillna('UNK')
    stint_stats = laps_prac.groupby(['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'Stint']).apply(calculate_stint_stats).reset_index()
    relevant_stints = stint_stats[stint_stats['LapCount'] >= config.MIN_STINT_LAPS].copy()

    if relevant_stints.empty:
        logger.warning("No relevant practice stints found for pace/deg features.")
        return df, new_cols

    practice_features = relevant_stints.groupby(['Year', 'RoundNumber', 'DriverNumber']).agg(
        AvgPaceMediumFP=('AvgLapTime', lambda x: x[relevant_stints['Compound'] == 'MEDIUM'].mean()),
        AvgPaceHardFP=('AvgLapTime', lambda x: x[relevant_stints['Compound'] == 'HARD'].mean()),
        AvgPaceSoftFP=('AvgLapTime', lambda x: x[relevant_stints['Compound'] == 'SOFT'].mean()),
        AvgDegMediumFP=('Degradation', lambda x: x[relevant_stints['Compound'] == 'MEDIUM'].mean()),
        AvgDegHardFP=('Degradation', lambda x: x[relevant_stints['Compound'] == 'HARD'].mean()),
        AvgDegSoftFP=('Degradation', lambda x: x[relevant_stints['Compound'] == 'SOFT'].mean())
    ).reset_index()
    practice_features['DriverNumber'] = practice_features['DriverNumber'].astype(str).fillna('UNK')
    df = pd.merge(df, practice_features, on=['Year', 'RoundNumber', 'DriverNumber'], how='left')
    new_cols = practice_features.columns.drop(['Year', 'RoundNumber', 'DriverNumber']).tolist()
    logger.info(f"Added practice pace/deg features: {new_cols}")
    return df, new_cols


def _add_weather_features(df, df_weather):
    """Adds features based on historical race weather data."""
    new_cols = []
    if not config.LOAD_WEATHER or df_weather.empty:
        logger.info("Skipping historical weather features (LOAD_WEATHER=False or no data).")
        return df, new_cols

    logger.info("Calculating historical race weather features...")
    weather_agg = df_weather.groupby(['Year', 'RoundNumber']).agg(
         HistAvgTrackTemp=('TrackTemp', 'mean'), HistMaxTrackTemp=('TrackTemp', 'max'),
         HistAvgAirTemp=('AirTemp', 'mean'), HistAvgHumidity=('Humidity', 'mean'),
         HistAvgWindSpeed=('WindSpeed', 'mean'), HistWasRainy=('Rainfall', lambda x: (x > 0).any().astype(int))
    ).reset_index()
    df = pd.merge(df, weather_agg, on=['Year', 'RoundNumber'], how='left')
    new_cols = weather_agg.columns.drop(['Year', 'RoundNumber']).tolist()
    logger.info(f"Added historical race weather features: {new_cols}")
    return df, new_cols

def _add_pit_stop_features(df, df_pits):
    """Adds the number of pit stops feature."""
    new_cols = []
    if df_pits.empty:
        logger.info("Skipping pit count feature (no pit data loaded).")
        df['NumPits'] = 0.0 # Add default even if no data
        return df, new_cols

    logger.info("Calculating pit stop count feature...");
    pit_counts = df_pits.groupby(['Year', 'RoundNumber', 'DriverNumber'])['StopNumber'].max().reset_index().rename(columns={'StopNumber': 'NumPits'})
    pit_counts['DriverNumber'] = pit_counts['DriverNumber'].astype(str).fillna('UNK')
    df['DriverNumber'] = df['DriverNumber'].astype(str).fillna('UNK') # Ensure before merge
    df = pd.merge(df, pit_counts, on=['Year', 'RoundNumber', 'DriverNumber'], how='left')
    df['NumPits'].fillna(0, inplace=True)
    new_cols.append('NumPits')
    logger.info("Added pit count feature (NumPits).")
    return df, new_cols


def _add_rolling_expanding_features(df):
    """Adds rolling and expanding window features for driver and team."""
    new_cols = []
    logger.info("Calculating rolling/expanding driver features...");
    df = utils.add_rolling_features(df, 'Abbreviation', 'Position', config.N_LAST_RACES_FEATURES, 'PosLastN', config.FEATURE_FILL_DEFAULTS['Pos'])
    df = utils.add_rolling_features(df, 'Abbreviation', 'Points', config.N_LAST_RACES_FEATURES, 'PtsLastN', config.FEATURE_FILL_DEFAULTS['Pts'])
    if 'RollingAvgPosLastN' in df.columns: new_cols.append('RollingAvgPosLastN')
    if 'RollingAvgPtsLastN' in df.columns: new_cols.append('RollingAvgPtsLastN')

    df = utils.add_expanding_features(df, ['Abbreviation', 'TrackLocation'], 'Position', 'PosThisTrack', config.FEATURE_FILL_DEFAULTS['Pos'])
    df = utils.add_expanding_features(df, ['Abbreviation', 'TrackLocation'], 'Points', 'PtsThisTrack', config.FEATURE_FILL_DEFAULTS['Pts'])
    if 'ExpandingAvgPosThisTrack' in df.columns: new_cols.append('ExpandingAvgPosThisTrack')
    if 'ExpandingAvgPtsThisTrack' in df.columns: new_cols.append('ExpandingAvgPtsThisTrack')

    logger.info("Calculating rolling/expanding team features...");
    team_points_per_race = df.groupby(['Year', 'RoundNumber', 'TeamName'])['Points'].mean().reset_index()
    team_points_per_race.sort_values(by=['Year', 'RoundNumber', 'TeamName'], inplace=True)
    team_points_per_race = utils.add_rolling_features(team_points_per_race, 'TeamName', 'Points', config.N_LAST_RACES_TEAM, 'TeamPtsLastN', config.FEATURE_FILL_DEFAULTS['Pts'])
    df = pd.merge(df, team_points_per_race[['Year', 'RoundNumber', 'TeamName', 'RollingAvgTeamPtsLastN']], on=['Year', 'RoundNumber', 'TeamName'], how='left')
    if 'RollingAvgTeamPtsLastN' in df.columns:
        df['RollingAvgTeamPtsLastN'].fillna(config.FEATURE_FILL_DEFAULTS['Pts'], inplace=True)
        new_cols.append('RollingAvgTeamPtsLastN')

    return df, new_cols


def _add_cumulative_features(df):
    """Adds simple cumulative features like race count and season points."""
    logger.info("Calculating cumulative features (RaceCount, SeasonPoints)...")
    df['RaceCount'] = df.groupby('Abbreviation').cumcount() # 0 for first race
    df['SeasonPointsBeforeRace'] = df.groupby(['Year', 'Abbreviation'])['Points'].shift(1).fillna(0)
    df['SeasonPointsBeforeRace'] = df.groupby(['Year', 'Abbreviation'])['SeasonPointsBeforeRace'].cumsum()
    new_cols = ['RaceCount', 'SeasonPointsBeforeRace']
    logger.info("Added cumulative features.")
    return df, new_cols


def _add_standings_features(df):
    """Adds championship standings features before the current race."""
    logger.info("Calculating championship standings features...")
    new_cols = []
    # Use unique Race identifier (Year-Round)
    df['RaceKey'] = df['Year'].astype(str) + '-' + df['RoundNumber'].astype(str).str.zfill(2)

    # Driver Standing
    # Use SeasonPointsBeforeRace already calculated
    standings_snapshot = df[['RaceKey', 'Year', 'RoundNumber', 'Abbreviation', 'SeasonPointsBeforeRace']].drop_duplicates()
    standings_snapshot['DriverStandingBeforeRace'] = standings_snapshot.groupby(['RaceKey'])['SeasonPointsBeforeRace'].rank(method='dense', ascending=False) # dense rank better
    df = pd.merge(df, standings_snapshot[['Year', 'RoundNumber', 'Abbreviation', 'DriverStandingBeforeRace']], on=['Year', 'RoundNumber', 'Abbreviation'], how='left')
    df['DriverStandingBeforeRace'].fillna(config.FEATURE_FILL_DEFAULTS['Standing'], inplace=True)
    new_cols.append('DriverStandingBeforeRace')

    # Constructor Standing
    constructor_points_race = df.groupby(['Year', 'RoundNumber', 'TeamName'])['Points'].sum().reset_index().rename(columns={'Points':'ConstructorPointsRace'})
    constructor_points_race.sort_values(['Year', 'TeamName', 'RoundNumber'], inplace=True)
    constructor_points_race['ConstructorPointsBeforeRace'] = constructor_points_race.groupby(['Year', 'TeamName'])['ConstructorPointsRace'].shift(1).fillna(0)
    constructor_points_race['ConstructorPointsBeforeRace'] = constructor_points_race.groupby(['Year', 'TeamName'])['ConstructorPointsBeforeRace'].cumsum()
    constructor_points_race['RaceKey'] = constructor_points_race['Year'].astype(str) + '-' + constructor_points_race['RoundNumber'].astype(str).str.zfill(2)
    constructor_points_race['ConstructorStandingBeforeRace'] = constructor_points_race.groupby('RaceKey')['ConstructorPointsBeforeRace'].rank(method='dense', ascending=False) # dense rank
    df = pd.merge(df, constructor_points_race[['Year', 'RoundNumber', 'TeamName', 'ConstructorPointsBeforeRace', 'ConstructorStandingBeforeRace']], on=['Year', 'RoundNumber', 'TeamName'], how='left')
    df['ConstructorPointsBeforeRace'].fillna(config.FEATURE_FILL_DEFAULTS['Pts'], inplace=True)
    df['ConstructorStandingBeforeRace'].fillna(config.FEATURE_FILL_DEFAULTS['Standing'], inplace=True) # Use config fallback
    new_cols.extend(['ConstructorPointsBeforeRace', 'ConstructorStandingBeforeRace'])

    df.drop(columns=['RaceKey'], inplace=True, errors='ignore')
    logger.info("Added standings features.")
    return df, new_cols


def _add_teammate_comparison(df):
    """Adds features comparing driver performance to team averages."""
    logger.info("Calculating teammate comparison features...")
    new_cols = []
    # Calculate team averages per race for key metrics
    team_avg_metrics = df.groupby(['Year', 'RoundNumber', 'TeamName']).agg(
        AvgTeamQualiPos=('QualiPosition', 'mean'),
        AvgTeamGridPos=('GridPosition', 'mean'),
        AvgTeamFinishPos=('Position', 'mean'),
        AvgTeamPoints=('Points', 'mean')
    ).reset_index()

    df = pd.merge(df, team_avg_metrics, on=['Year', 'RoundNumber', 'TeamName'], how='left')

    # Calculate difference features
    df['QualiDiffToTeamAvg'] = df['QualiPosition'] - df['AvgTeamQualiPos']
    df['GridDiffToTeamAvg'] = df['GridPosition'] - df['AvgTeamGridPos']
    df['FinishDiffToTeamAvg'] = df['Position'] - df['AvgTeamFinishPos']
    df['PointsDiffToTeamAvg'] = df['Points'] - df['AvgTeamPoints']

    diff_cols = ['QualiDiffToTeamAvg', 'GridDiffToTeamAvg', 'FinishDiffToTeamAvg', 'PointsDiffToTeamAvg']
    for col in diff_cols: df[col].fillna(0, inplace=True) # Fill NaNs with 0 (no difference)

    # Rolling averages of differences
    df = utils.add_rolling_features(df, 'Abbreviation', 'QualiDiffToTeamAvg', config.N_LAST_RACES_TEAMMATE_COMPARISON, 'QualiDiffAvg', 0.0)
    df = utils.add_rolling_features(df, 'Abbreviation', 'GridDiffToTeamAvg', config.N_LAST_RACES_TEAMMATE_COMPARISON, 'GridDiffAvg', 0.0)
    df = utils.add_rolling_features(df, 'Abbreviation', 'FinishDiffToTeamAvg', config.N_LAST_RACES_TEAMMATE_COMPARISON, 'FinishDiffAvg', 0.0)
    df = utils.add_rolling_features(df, 'Abbreviation', 'PointsDiffToTeamAvg', config.N_LAST_RACES_TEAMMATE_COMPARISON, 'PointsDiffAvg', 0.0)

    rolling_teammate_cols = ['RollingAvgQualiDiffAvg', 'RollingAvgGridDiffAvg', 'RollingAvgFinishDiffAvg', 'RollingAvgPointsDiffAvg']
    for col in rolling_teammate_cols:
        if col in df.columns: new_cols.append(col)

    # Clean up intermediate columns
    df.drop(columns=['AvgTeamQualiPos', 'AvgTeamGridPos', 'AvgTeamFinishPos', 'AvgTeamPoints'] + diff_cols, inplace=True, errors='ignore')
    logger.info("Added teammate comparison features.")
    return df, new_cols

def _add_forecast_placeholders(df):
    """Adds placeholder columns for forecast features, filled with historicals."""
    logger.info("Creating historical equivalents for forecast features...")
    new_cols = []
    # Use Hist values if available, otherwise fill with NaNs for now (final fill later)
    df['ForecastTemp'] = df.get('HistAvgAirTemp', pd.Series(np.nan, index=df.index))
    df['ForecastRainProb'] = df.get('HistWasRainy', pd.Series(np.nan, index=df.index)).astype(float) # Use HistWasRainy if exists
    df['ForecastWindSpeed'] = df.get('HistAvgWindSpeed', pd.Series(np.nan, index=df.index))
    forecast_equiv_cols = ['ForecastTemp', 'ForecastRainProb', 'ForecastWindSpeed']
    new_cols.extend([col for col in forecast_equiv_cols if col in df.columns])
    return df, new_cols


def _handle_track_type(df):
    """Adds TrackType feature and handles encoding based on model type."""
    logger.info("Handling TrackType feature...")
    new_ohe_cols = []
    df['TrackType'] = df['TrackLocation'].map(config.TRACK_CHARACTERISTICS).fillna('Unknown').astype(str)

    if config.MODEL_TYPE.lower() in ['lightgbm']:
        logger.info("Using LightGBM: Converting TrackType to 'category' dtype.")
        df['TrackType'] = df['TrackType'].astype('category')
        # Feature list will just contain 'TrackType'
    else: # One-Hot Encode
        logger.info(f"Using {config.MODEL_TYPE}: One-Hot Encoding TrackType.")
        df = pd.get_dummies(df, columns=['TrackType'], prefix='Track', dummy_na=False, dtype=int)
        new_ohe_cols = [col for col in df.columns if col.startswith('Track_')]
        logger.info(f"Added OHE track features: {new_ohe_cols}")

    return df, new_ohe_cols


def _finalize_cleaning_and_features(df, all_generated_feature_cols):
    """Performs final cleaning, NaN filling, and determines final feature list."""
    logger.info("Performing final cleaning and feature selection...")
    target_col = 'Position'

    # Define initial feature set (including GridPosition)
    feature_cols = sorted(list(dict.fromkeys(['GridPosition'] + all_generated_feature_cols)))

    # Explicitly remove identifiers or intermediate columns
    # Keep QualiPosition if needed for analysis later, but remove from final features
    cols_to_remove = ['Location', 'DriverNumber', 'QualiPosition', 'EventDate', 'TrackLocation']
    feature_cols = [col for col in feature_cols if col not in cols_to_remove]
    # Remove original TrackType if OHE was performed
    if any(f.startswith('Track_') for f in feature_cols):
        feature_cols = [f for f in feature_cols if f != 'TrackType']
    # Add TrackType if LightGBM and it wasn't already added somehow
    elif config.MODEL_TYPE.lower() == 'lightgbm' and 'TrackType' in df.columns and 'TrackType' not in feature_cols:
         feature_cols.append('TrackType')


    # Ensure all expected features exist in the dataframe
    for col in feature_cols:
        if col not in df.columns:
            logger.warning(f"Expected feature '{col}' not found in DataFrame. Adding as NaN.")
            df[col] = np.nan

    # Select final columns for model input + identifiers/target
    cols_to_keep_final = list(set(['Year', 'RoundNumber', 'Abbreviation', 'TeamName', target_col] + feature_cols))
    df_model_input = df[cols_to_keep_final].copy()

    
    # --- Consolidated NaN/Inf Filling for numeric features ---
    numeric_feature_cols = [col for col in feature_cols if col in df_model_input.columns and pd.api.types.is_numeric_dtype(df_model_input[col])]
    logger.debug(f"Final NaN/Inf filling for numeric features: {numeric_feature_cols}")
    for col in numeric_feature_cols:
        df_model_input[col] = df_model_input[col].replace([np.inf, -np.inf], np.nan)
                # Use specific fill defaults from config if available
        # Use the 'default' value from the dictionary as the fallback
        fill_val = config.FEATURE_FILL_DEFAULTS.get('default', -999.0) # Use the 'default' value
        for pattern, value in config.FEATURE_FILL_DEFAULTS.items():
             if pattern != 'default' and pattern.lower() in col.lower():
                  fill_val = value
                  break # Use first matching pattern
        # Apply fill value
        df_model_input[col].fillna(fill_val, inplace=True)


    # Final check for target NaNs
    if df_model_input[target_col].isnull().any():
        logger.warning(f"NaNs found in target column '{target_col}'. These rows might be dropped during training.")

    # Final unique sort of feature list
    final_feature_cols = sorted(list(set(feature_cols)))

    logger.info(f"Final feature set determined ({len(final_feature_cols)}): {final_feature_cols}")

    return df_model_input, final_feature_cols, target_col


# --- Main Feature Creation Function ---
def create_features():
    """Loads data from DB and engineers features for modeling."""
    logger.info("--- Starting Feature Engineering ---")

    df_raw_results, df_laps, df_weather, df_pits = _load_data_for_features()
    if df_raw_results is None: return None, [], None

    df_base = _prepare_base_results(df_raw_results)
    if df_base is None: return None, [], None

    # --- Add features step-by-step ---
    all_feature_cols = [] # Accumulate generated feature names

    df, practice_cols = _add_practice_features(df_base, df_laps); all_feature_cols.extend(practice_cols)
    df, weather_cols = _add_weather_features(df, df_weather); all_feature_cols.extend(weather_cols)
    df, pit_cols = _add_pit_stop_features(df, df_pits); all_feature_cols.extend(pit_cols)

    # --- Sort before time-based features ---
    df.sort_values(by=['Year', 'RoundNumber', 'Abbreviation'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df, rolling_cols = _add_rolling_expanding_features(df); all_feature_cols.extend(rolling_cols)
    df, cumulative_cols = _add_cumulative_features(df); all_feature_cols.extend(cumulative_cols)
    df, standings_cols = _add_standings_features(df); all_feature_cols.extend(standings_cols)
    df, teammate_cols = _add_teammate_comparison(df); all_feature_cols.extend(teammate_cols)
    df, forecast_cols = _add_forecast_placeholders(df); all_feature_cols.extend(forecast_cols)
    df, track_ohe_cols = _handle_track_type(df); all_feature_cols.extend(track_ohe_cols)


    # --- Finalize ---
    df_final, final_feature_list, target_col = _finalize_cleaning_and_features(df, all_feature_cols)

    if df_final is None or not final_feature_list or target_col is None:
        logger.error("Feature engineering failed during finalization.")
        return None, [], None

    logger.info(f"Feature engineering complete. Final DataFrame shape: {df_final.shape}");
    return df_final, final_feature_list, target_col