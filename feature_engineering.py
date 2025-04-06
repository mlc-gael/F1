# /f1_predictor/feature_engineering.py

import pandas as pd
import numpy as np
import config
import database
import utils

logger = utils.get_logger(__name__)

# --- Helper Functions (calculate_stint_stats - Keep as is) ---
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

# --- Main Feature Creation Function ---
def create_features():
    """Loads data from DB and engineers features for modeling."""
    logger.info("--- Starting Feature Engineering ---")

    # --- Load Base Data ---
    query_results = "SELECT r.*, e.Location, e.EventName FROM results r JOIN events e ON r.Year = e.Year AND r.RoundNumber = e.RoundNumber WHERE r.SessionName IN ('R', 'Q') ORDER BY r.Year, r.RoundNumber"
    df_raw_results = database.load_data(query_results)
    if df_raw_results.empty: logger.error("CRITICAL: No Race/Quali results found."); return pd.DataFrame(), [], None

    df_laps = pd.DataFrame(); df_weather = pd.DataFrame(); df_pits = pd.DataFrame()
    if config.LOAD_LAPS:
        logger.info("Attempting to load lap data...")
        df_laps = database.load_data("SELECT Year, RoundNumber, SessionName, DriverNumber, LapNumber, LapTime, Stint, Compound, IsAccurate, IsPitOutLap, IsPitInLap FROM laps")
        if df_laps.empty: logger.warning("No lap data loaded.")
        else: logger.info(f"Loaded {len(df_laps)} laps.")
        logger.info("Attempting to load pit stop data...")
        if database.table_exists('pit_stops'):
             df_pits = database.load_data("SELECT Year, RoundNumber, SessionName, DriverNumber, StopNumber FROM pit_stops WHERE SessionName='R'")
             if df_pits.empty: logger.warning("Pit stop table exists but returned no Race pit stop data.")
             else: logger.info(f"Loaded {len(df_pits)} race pit stops.")
        else: logger.warning("Pit stop table ('pit_stops') does not exist.")

    if config.LOAD_WEATHER:
        logger.info("Attempting to load weather data...")
        df_weather = database.load_data("SELECT Year, RoundNumber, SessionName, AirTemp, TrackTemp, Humidity, Pressure, WindSpeed, WindDirection, Rainfall FROM weather WHERE SessionName='R'")
        if df_weather.empty: logger.warning("No Race weather data loaded.")
        else: logger.info(f"Loaded {len(df_weather)} race weather records.")

    # --- Prepare Base Results DataFrame ---
    logger.info("Preparing base results data...")
    df_quali_raw = df_raw_results[df_raw_results['SessionName'] == 'Q'].copy()
    df_quali_raw['QualiPositionRaw'] = utils.safe_to_numeric(df_quali_raw['Position'], fallback=np.nan)
    min_quali_indices = df_quali_raw.groupby(['Year', 'RoundNumber', 'Abbreviation'])['QualiPositionRaw'].idxmin().dropna()
    if min_quali_indices.empty:
         logger.warning("No valid Qualifying positions found. QualiPosition feature will be NaN.")
         df_quali = pd.DataFrame(columns=['Year', 'RoundNumber', 'Abbreviation', 'QualiPosition'])
    else:
         df_quali = df_quali_raw.loc[min_quali_indices].rename(columns={'QualiPositionRaw': 'QualiPosition'})[['Year', 'RoundNumber', 'Abbreviation', 'QualiPosition']].reset_index(drop=True)
         # --- Ensure QualiPosition is float for calculations ---
         df_quali['QualiPosition'] = utils.safe_to_numeric(df_quali['QualiPosition'], fallback=np.nan).astype(float)


    df_race = df_raw_results[df_raw_results['SessionName'] == 'R'].copy()
    if df_race.empty: logger.error("CRITICAL: No Race results found."); return pd.DataFrame(), [], None

    # Merge race results with the best quali position
    df = pd.merge(df_race, df_quali, on=['Year', 'RoundNumber', 'Abbreviation'], how='left')

    # Ensure DriverNumber exists and is string early
    if 'DriverNumber' not in df.columns:
         logger.error("CRITICAL: 'DriverNumber' missing. Check data_loader.py:process_session_results.")
         df['DriverNumber'] = 'UNK'
    df['DriverNumber'] = df['DriverNumber'].astype(str).fillna('UNK')

    # Clean key columns: GridPosition, QualiPosition, Position, Points
    df['GridPosition'] = utils.safe_to_numeric(df['GridPosition'], fallback=np.nan)
    if 'QualiPosition' in df.columns:
        # Ensure QualiPosition is float before filling GridPosition
        df['QualiPosition'] = utils.safe_to_numeric(df['QualiPosition'], fallback=np.nan).astype(float)
        df['GridPosition'].fillna(df['QualiPosition'], inplace=True)
    df['GridPosition'] = utils.safe_to_numeric(df['GridPosition'], fallback=config.WORST_EXPECTED_POS).astype(int)

    # Ensure QualiPosition exists after merge, convert to float for calculations
    if 'QualiPosition' not in df.columns: df['QualiPosition'] = np.nan
    df['QualiPosition'] = utils.safe_to_numeric(df['QualiPosition'], fallback=np.nan).astype(float)

    df['Position'] = utils.safe_to_numeric(df['Position'], fallback=config.WORST_EXPECTED_POS).astype(int) # Target variable
    df['Points'] = utils.safe_to_numeric(df['Points'], fallback=0.0).astype(float)
    df['TrackLocation'] = df['Location'].fillna('Unknown').astype(str)
    df['TeamName'] = df['TeamName'].astype(str).fillna('Unknown') # Ensure TeamName is str

    # --- Feature Engineering ---
    base_feature_cols = []

    # 1. Lap Features (Practice)
    # (Keep this section as is)
    if config.LOAD_LAPS and not df_laps.empty:
        logger.info("Calculating features from practice lap data...");
        laps_prac = df_laps[df_laps['SessionName'].isin(['FP1', 'FP2', 'FP3'])].copy()
        if not laps_prac.empty:
            laps_prac['LapTime'] = pd.to_numeric(laps_prac['LapTime'], errors='coerce')
            laps_prac['Stint'] = pd.to_numeric(laps_prac['Stint'], errors='coerce').fillna(-1).astype(int)
            laps_prac['IsAccurate'] = pd.to_numeric(laps_prac['IsAccurate'], errors='coerce').fillna(0).astype(int)
            laps_prac['DriverNumber'] = laps_prac['DriverNumber'].astype(str).fillna('UNK')
            stint_stats = laps_prac.groupby(['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'Stint']).apply(calculate_stint_stats).reset_index()
            relevant_stints = stint_stats[stint_stats['LapCount'] >= config.MIN_STINT_LAPS].copy()
            if not relevant_stints.empty:
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
                base_feature_cols.extend(new_cols)
                logger.info(f"Added practice pace/deg features: {new_cols}")
            else: logger.warning(f"No relevant practice stints found for pace/deg features.")
        else: logger.warning("No practice lap data found.")
    else: logger.info("Skipping practice lap features.")

    # 2. Weather Features (Historical)
    # (Keep this section as is)
    if config.LOAD_WEATHER and not df_weather.empty:
        logger.info("Calculating historical race weather features...")
        weather_agg = df_weather.groupby(['Year', 'RoundNumber']).agg(
             HistAvgTrackTemp=('TrackTemp', 'mean'), HistMaxTrackTemp=('TrackTemp', 'max'),
             HistAvgAirTemp=('AirTemp', 'mean'), HistAvgHumidity=('Humidity', 'mean'),
             HistAvgWindSpeed=('WindSpeed', 'mean'), HistWasRainy=('Rainfall', lambda x: (x > 0).any().astype(int))
        ).reset_index()
        df = pd.merge(df, weather_agg, on=['Year', 'RoundNumber'], how='left')
        new_cols = weather_agg.columns.drop(['Year', 'RoundNumber']).tolist(); base_feature_cols.extend(new_cols);
        logger.info(f"Added historical race weather features: {new_cols}")
    else: logger.info("Skipping historical weather features.")

    # 3. Pit Stop Features
    # (Keep this section as is)
    if not df_pits.empty:
         logger.info("Calculating pit stop count feature...");
         pit_counts = df_pits.groupby(['Year', 'RoundNumber', 'DriverNumber'])['StopNumber'].max().reset_index().rename(columns={'StopNumber': 'NumPits'})
         pit_counts['DriverNumber'] = pit_counts['DriverNumber'].astype(str).fillna('UNK')
         df['DriverNumber'] = df['DriverNumber'].astype(str).fillna('UNK') # Ensure before merge
         df = pd.merge(df, pit_counts, on=['Year', 'RoundNumber', 'DriverNumber'], how='left')
         df['NumPits'].fillna(0, inplace=True)
         base_feature_cols.append('NumPits'); logger.info("Added pit count feature (NumPits).")
    else:
        logger.info("Skipping pit count feature (no pit data loaded).")
        df['NumPits'] = 0.0

    # --- Ensure base features exist ---
    expected_base_cols = [
        'AvgPaceMediumFP', 'AvgPaceHardFP', 'AvgPaceSoftFP', 'AvgDegMediumFP', 'AvgDegHardFP', 'AvgDegSoftFP',
        'HistAvgTrackTemp', 'HistMaxTrackTemp', 'HistAvgAirTemp', 'HistAvgHumidity', 'HistAvgWindSpeed', 'HistWasRainy',
        'NumPits' ]
    for col in expected_base_cols:
         if col not in df.columns: df[col] = np.nan
         if col not in base_feature_cols and col in df.columns: base_feature_cols.append(col)

    # --- Sort before time-based features ---
    df.sort_values(by=['Year', 'RoundNumber', 'Abbreviation'], inplace=True)
    df.reset_index(drop=True, inplace=True)


    # --- 4. Rolling/Expanding Driver Features ---
    logger.info("Calculating rolling/expanding driver features...");
    df = utils.add_rolling_features(df, 'Abbreviation', 'Position', config.N_LAST_RACES_FEATURES, 'PosLastN', config.WORST_EXPECTED_POS)
    df = utils.add_rolling_features(df, 'Abbreviation', 'Points', config.N_LAST_RACES_FEATURES, 'PtsLastN', 0.0)
    if 'RollingAvgPosLastN' in df.columns: base_feature_cols.append('RollingAvgPosLastN')
    if 'RollingAvgPtsLastN' in df.columns: base_feature_cols.append('RollingAvgPtsLastN')
    df = utils.add_expanding_features(df, ['Abbreviation', 'TrackLocation'], 'Position', 'PosThisTrack', config.WORST_EXPECTED_POS)
    df = utils.add_expanding_features(df, ['Abbreviation', 'TrackLocation'], 'Points', 'PtsThisTrack', 0.0)
    if 'ExpandingAvgPosThisTrack' in df.columns: base_feature_cols.append('ExpandingAvgPosThisTrack')
    if 'ExpandingAvgPtsThisTrack' in df.columns: base_feature_cols.append('ExpandingAvgPtsThisTrack')

    # --- 5. Rolling/Expanding Team Features ---
    logger.info("Calculating rolling/expanding team features...");
    team_points_per_race = df.groupby(['Year', 'RoundNumber', 'TeamName'])['Points'].mean().reset_index()
    team_points_per_race.sort_values(by=['Year', 'RoundNumber', 'TeamName'], inplace=True)
    team_points_per_race = utils.add_rolling_features(team_points_per_race, 'TeamName', 'Points', config.N_LAST_RACES_TEAM, 'TeamPtsLastN', 0.0)
    df = pd.merge(df, team_points_per_race[['Year', 'RoundNumber', 'TeamName', 'RollingAvgTeamPtsLastN']], on=['Year', 'RoundNumber', 'TeamName'], how='left')
    if 'RollingAvgTeamPtsLastN' in df.columns:
        df['RollingAvgTeamPtsLastN'].fillna(0.0, inplace=True)
        base_feature_cols.append('RollingAvgTeamPtsLastN')

    # --- 6. Simple Cumulative Features ---
    df['RaceCount'] = df.groupby('Abbreviation').cumcount() # 0 for first race
    df['SeasonPointsBeforeRace'] = df.groupby(['Year', 'Abbreviation'])['Points'].shift(1).fillna(0)
    df['SeasonPointsBeforeRace'] = df.groupby(['Year', 'Abbreviation'])['SeasonPointsBeforeRace'].cumsum()
    base_feature_cols.extend(['RaceCount', 'SeasonPointsBeforeRace'])


    # --- 7. NEW: Championship Standings Features ---
    logger.info("Calculating championship standings features...")
    # Driver Standing Before Race
    # Create a temporary key for merging standings across rounds
    df['RaceKey'] = df['Year'].astype(str) + '-' + df['RoundNumber'].astype(str).str.zfill(2)
    standings_snapshot = df[['RaceKey', 'Year', 'RoundNumber', 'Abbreviation', 'SeasonPointsBeforeRace']].drop_duplicates()
    # Calculate rank within each race group based on points *before* that race
    standings_snapshot['DriverStandingBeforeRace'] = standings_snapshot.groupby(['RaceKey'])['SeasonPointsBeforeRace'].rank(method='min', ascending=False)
    # Merge back - use Year, RoundNumber, Abbreviation as keys
    df = pd.merge(df, standings_snapshot[['Year', 'RoundNumber', 'Abbreviation', 'DriverStandingBeforeRace']], on=['Year', 'RoundNumber', 'Abbreviation'], how='left')
    df['DriverStandingBeforeRace'].fillna(config.WORST_EXPECTED_POS + 5, inplace=True) # Fallback for first race / missing data
    base_feature_cols.append('DriverStandingBeforeRace')

    # Constructor Points & Standing Before Race
    constructor_points_race = df.groupby(['Year', 'RoundNumber', 'TeamName'])['Points'].sum().reset_index().rename(columns={'Points':'ConstructorPointsRace'})
    constructor_points_race.sort_values(['Year', 'TeamName', 'RoundNumber'], inplace=True)
    constructor_points_race['ConstructorPointsBeforeRace'] = constructor_points_race.groupby(['Year', 'TeamName'])['ConstructorPointsRace'].shift(1).fillna(0)
    constructor_points_race['ConstructorPointsBeforeRace'] = constructor_points_race.groupby(['Year', 'TeamName'])['ConstructorPointsBeforeRace'].cumsum()
    # Rank constructors within each race group
    constructor_points_race['RaceKey'] = constructor_points_race['Year'].astype(str) + '-' + constructor_points_race['RoundNumber'].astype(str).str.zfill(2)
    constructor_points_race['ConstructorStandingBeforeRace'] = constructor_points_race.groupby('RaceKey')['ConstructorPointsBeforeRace'].rank(method='min', ascending=False)
    # Merge back
    df = pd.merge(df, constructor_points_race[['Year', 'RoundNumber', 'TeamName', 'ConstructorPointsBeforeRace', 'ConstructorStandingBeforeRace']], on=['Year', 'RoundNumber', 'TeamName'], how='left')
    df['ConstructorPointsBeforeRace'].fillna(0, inplace=True)
    df['ConstructorStandingBeforeRace'].fillna(11, inplace=True) # Fallback rank (assuming 10 teams)
    base_feature_cols.extend(['ConstructorPointsBeforeRace', 'ConstructorStandingBeforeRace'])
    df.drop(columns=['RaceKey'], inplace=True, errors='ignore') # Clean up temporary key


    # --- 8. NEW: Teammate Comparison Features ---
    logger.info("Calculating teammate comparison features...")
    # Calculate team averages per race for key metrics
    team_avg_metrics = df.groupby(['Year', 'RoundNumber', 'TeamName']).agg(
        AvgTeamQualiPos=('QualiPosition', 'mean'), # Use QualiPosition merged earlier
        AvgTeamGridPos=('GridPosition', 'mean'),
        AvgTeamFinishPos=('Position', 'mean'),
        AvgTeamPoints=('Points', 'mean')
    ).reset_index()

    # Merge team averages back to the main dataframe
    df = pd.merge(df, team_avg_metrics, on=['Year', 'RoundNumber', 'TeamName'], how='left')

    # Calculate difference features (Driver Metric - Team Average Metric)
    # Handle potential division by zero or NaNs if team average is NaN or driver is alone
    df['QualiDiffToTeamAvg'] = df['QualiPosition'] - df['AvgTeamQualiPos']
    df['GridDiffToTeamAvg'] = df['GridPosition'] - df['AvgTeamGridPos']
    df['FinishDiffToTeamAvg'] = df['Position'] - df['AvgTeamFinishPos']
    df['PointsDiffToTeamAvg'] = df['Points'] - df['AvgTeamPoints']

    # Fill NaNs in difference features (e.g., if driver was alone or avg was NaN) with 0
    diff_cols = ['QualiDiffToTeamAvg', 'GridDiffToTeamAvg', 'FinishDiffToTeamAvg', 'PointsDiffToTeamAvg']
    for col in diff_cols:
        df[col].fillna(0, inplace=True)

    # Calculate rolling averages of these difference features
    df = utils.add_rolling_features(df, 'Abbreviation', 'QualiDiffToTeamAvg', config.N_LAST_RACES_TEAMMATE_COMPARISON, 'QualiDiffAvg', 0.0)
    df = utils.add_rolling_features(df, 'Abbreviation', 'GridDiffToTeamAvg', config.N_LAST_RACES_TEAMMATE_COMPARISON, 'GridDiffAvg', 0.0)
    df = utils.add_rolling_features(df, 'Abbreviation', 'FinishDiffToTeamAvg', config.N_LAST_RACES_TEAMMATE_COMPARISON, 'FinishDiffAvg', 0.0)
    df = utils.add_rolling_features(df, 'Abbreviation', 'PointsDiffToTeamAvg', config.N_LAST_RACES_TEAMMATE_COMPARISON, 'PointsDiffAvg', 0.0)

    # Add new rolling features to base list
    new_teammate_features = ['RollingAvgQualiDiffAvg', 'RollingAvgGridDiffAvg', 'RollingAvgFinishDiffAvg', 'RollingAvgPointsDiffAvg']
    for col in new_teammate_features:
         if col in df.columns: base_feature_cols.append(col)

    # Clean up intermediate columns
    df.drop(columns=['AvgTeamQualiPos', 'AvgTeamGridPos', 'AvgTeamFinishPos', 'AvgTeamPoints'] + diff_cols, inplace=True, errors='ignore')


    # --- 9. Historical Equivalents for Forecast Features ---
    # (Keep this section as is)
    logger.info("Creating historical equivalents for forecast features...")
    df['ForecastTemp'] = df.get('HistAvgAirTemp', pd.Series(np.nan, index=df.index))
    df['ForecastRainProb'] = df.get('HistWasRainy', pd.Series(0, index=df.index)).astype(float)
    df['ForecastWindSpeed'] = df.get('HistAvgWindSpeed', pd.Series(np.nan, index=df.index))
    forecast_equiv_cols = ['ForecastTemp', 'ForecastRainProb', 'ForecastWindSpeed']
    base_feature_cols.extend([col for col in forecast_equiv_cols if col in df.columns])


    # --- 10. Track Characteristics (Categorical) ---
    # (Keep this section as is)
    df['TrackType'] = df['TrackLocation'].map(config.TRACK_CHARACTERISTICS).fillna('Unknown').astype(str)


    # --- Finalize Feature List ---
    # Include GridPosition, ensure QualiPosition is available for features but maybe not final model input
    initial_feature_cols = ['GridPosition'] + base_feature_cols
    feature_cols = sorted(list(dict.fromkeys(initial_feature_cols))) # Unique and sorted

    # Explicitly remove identifiers or intermediate columns if accidentally included
    cols_to_remove = ['Location', 'DriverNumber', 'QualiPosition'] # QualiPos used for GridPos fill and teammate diff, but maybe not needed as direct input
    feature_cols = [col for col in feature_cols if col not in cols_to_remove]


    # --- Final Data Cleaning & Type Handling ---
    logger.info("Performing final cleaning on feature set...")
    cols_to_keep_intermediate = list(set(['Year', 'RoundNumber', 'Abbreviation', 'TeamName', 'Position'] + feature_cols + ['TrackType']))
    missing_cols_in_df = [col for col in cols_to_keep_intermediate if col not in df.columns]
    if missing_cols_in_df:
        logger.error(f"CRITICAL: Columns missing from DataFrame before final selection: {missing_cols_in_df}. Adding as NaN.")
        for col in missing_cols_in_df: df[col] = np.nan
        # Re-check which columns actually exist now
        cols_to_keep_intermediate = [col for col in cols_to_keep_intermediate if col in df.columns]

    df_model_input = df[cols_to_keep_intermediate].copy()

    # Fill NaNs/Infs in numeric FEATURE columns
    numeric_feature_cols = [col for col in feature_cols if col in df_model_input.columns and pd.api.types.is_numeric_dtype(df_model_input[col])]
    logger.debug(f"Filling NaNs/Infs in numeric feature columns: {numeric_feature_cols}")
    for col in numeric_feature_cols:
         df_model_input[col] = df_model_input[col].replace([np.inf, -np.inf], np.nan)
         df_model_input[col] = df_model_input[col].fillna(config.FILL_NA_VALUE)

    # --- Handle Categorical TrackType ---
    final_feature_cols = feature_cols.copy()
    if config.MODEL_TYPE.lower() in ['lightgbm']:
        logger.info("Using LightGBM: Converting TrackType to 'category' dtype.")
        if 'TrackType' in df_model_input.columns:
            df_model_input['TrackType'] = df_model_input['TrackType'].astype('category')
            if 'TrackType' not in final_feature_cols: final_feature_cols.append('TrackType')
            # Remove any OHE cols if they accidentally got into final_feature_cols
            final_feature_cols = [f for f in final_feature_cols if not f.startswith('Track_')]
        else: logger.warning("TrackType column not found for LightGBM.")
    else: # One-Hot Encode
        logger.info(f"Using {config.MODEL_TYPE}: One-Hot Encoding TrackType.")
        if 'TrackType' in df_model_input.columns:
             df_model_input = pd.get_dummies(df_model_input, columns=['TrackType'], prefix='Track', dummy_na=False, dtype=int)
             ohe_cols = [col for col in df_model_input.columns if col.startswith('Track_')];
             # Add only NEW ohe_cols to the list, remove original TrackType
             final_feature_cols = [f for f in final_feature_cols if f != 'TrackType']
             final_feature_cols.extend([c for c in ohe_cols if c not in final_feature_cols])
        else: logger.warning("TrackType column not found for OHE.")

    # Final unique sort
    final_feature_cols = sorted(list(set(final_feature_cols)))

    # --- Target Variable ---
    target_col = 'Position'

    # --- Final Validation ---
    logger.info("Performing final validation checks on features and target...")
    # Check all final features and target exist
    check_cols_exist = final_feature_cols + [target_col]
    missing_in_df_final = [c for c in check_cols_exist if c not in df_model_input.columns]
    if missing_in_df_final:
        logger.error(f"CRITICAL: Columns missing from DataFrame just before returning: {missing_in_df_final}")
        if target_col in missing_in_df_final: logger.error(f"FATAL: Target '{target_col}' missing."); return pd.DataFrame(), [], None
        else:
            logger.warning(f"Adding missing FINAL features as NaN: {missing_in_df_final}")
            for col in missing_in_df_final: df_model_input[col] = config.FILL_NA_VALUE
            final_feature_cols = [col for col in final_feature_cols if col in df_model_input.columns] # Re-verify

    # Re-check NaNs/Infs in the FINAL selected numeric features
    final_numeric_features = [f for f in final_feature_cols if f in df_model_input.columns and pd.api.types.is_numeric_dtype(df_model_input[f])]
    if final_numeric_features:
        nan_check = df_model_input[final_numeric_features].isnull().sum(); nan_cols = nan_check[nan_check > 0]
        if not nan_cols.empty:
            logger.warning(f"NaNs remain in FINAL numeric features AFTER cleaning:\n{nan_cols}")
            df_model_input[final_numeric_features] = df_model_input[final_numeric_features].fillna(config.FILL_NA_VALUE)
        inf_check = df_model_input[final_numeric_features].isin([np.inf, -np.inf]).sum(); inf_cols = inf_check[inf_check > 0]
        if not inf_cols.empty:
            logger.warning(f"Infs remain in FINAL numeric features AFTER cleaning:\n{inf_cols}")
            df_model_input[final_numeric_features] = df_model_input[final_numeric_features].replace([np.inf, -np.inf], config.FILL_NA_VALUE * 1000)
    else: logger.warning("No numeric features found in the final list to check NaNs/Infs.")

    if df_model_input[target_col].isnull().any(): logger.warning(f"NaNs found in target column '{target_col}'.")

    logger.info(f"Feature engineering complete. Final DataFrame shape: {df_model_input.shape}");
    logger.info(f"Target column: {target_col}");
    logger.info(f"Final features ({len(final_feature_cols)}): {final_feature_cols}")

    final_cols_to_return = list(set(['Year', 'RoundNumber', 'Abbreviation', 'TeamName', target_col] + final_feature_cols))
    final_cols_available = [col for col in final_cols_to_return if col in df_model_input.columns]
    df_final_return = df_model_input[final_cols_available].copy()

    return df_final_return, final_feature_cols, target_col