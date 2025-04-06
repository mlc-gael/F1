# /f1_predictor/feature_engineering.py

import pandas as pd
import numpy as np
import config
import database
import utils
# Removed LabelEncoder as we handle categoricals differently now

logger = utils.get_logger(__name__)

# --- Helper Functions (calculate_stint_stats - Keep as is) ---
def calculate_stint_stats(group):
    """Calculates stats for a given stint (group of laps)."""
    group = group.copy()
    group['LapTime'] = pd.to_numeric(group['LapTime'], errors='coerce')
    accurate_laps = group[group['IsAccurate'] == 1].dropna(subset=['LapTime'])
    if len(accurate_laps) < 2:
        return pd.Series({'AvgLapTime': np.nan, 'StdDevLapTime': np.nan, 'LapCount': len(accurate_laps), 'Degradation': np.nan, 'Compound': group['Compound'].iloc[0] if not group.empty else 'UNKNOWN'})
    avg_time = accurate_laps['LapTime'].mean()
    std_dev_time = accurate_laps['LapTime'].std(); lap_count = len(accurate_laps); degradation = np.nan
    n_deg = config.DEGRADATION_LAP_COUNT
    if lap_count >= (2 * n_deg):
        try: first = accurate_laps.iloc[:n_deg]['LapTime'].mean(); last = accurate_laps.iloc[-n_deg:]['LapTime'].mean(); degradation = last - first
        except IndexError: pass
    return pd.Series({'AvgLapTime': avg_time, 'StdDevLapTime': std_dev_time, 'LapCount': lap_count, 'Degradation': degradation, 'Compound': group['Compound'].iloc[0]})


# --- Main Feature Creation Function ---
def create_features():
    """Loads data from DB and engineers features for modeling."""
    logger.info("--- Starting Feature Engineering ---")

    # --- Load Base Data ---
    query_results = "SELECT r.*, e.Location, e.EventName FROM results r JOIN events e ON r.Year = e.Year AND r.RoundNumber = e.RoundNumber WHERE r.SessionName IN ('R', 'Q') ORDER BY r.Year, r.RoundNumber"
    df_raw_results = database.load_data(query_results);
    if df_raw_results.empty: logger.error("No Race/Quali results found."); return pd.DataFrame(), [], None
    df_laps = pd.DataFrame(); df_weather = pd.DataFrame(); df_pits = pd.DataFrame()
    if config.LOAD_LAPS:
        df_laps = database.load_data("SELECT Year, RoundNumber, SessionName, DriverNumber, LapNumber, LapTime, Stint, Compound, IsAccurate, IsPitOutLap, IsPitInLap FROM laps")
        if df_laps.empty: logger.warning("No lap data found."); 
        else: logger.info(f"Loaded {len(df_laps)} laps.")
        if database.table_exists('pit_stops'):
             df_pits = database.load_data("SELECT Year, RoundNumber, SessionName, DriverNumber, StopNumber FROM pit_stops")
             if df_pits.empty: logger.warning("Pit stop table empty."); 
             else: logger.info(f"Loaded {len(df_pits)} pits.")
    if config.LOAD_WEATHER:
        df_weather = database.load_data("SELECT Year, RoundNumber, SessionName, AirTemp, TrackTemp, Humidity, Pressure, WindSpeed, WindDirection, Rainfall FROM weather")
        if df_weather.empty: logger.warning("No weather data found."); 
        else: logger.info(f"Loaded {len(df_weather)} weather.")

    # --- Prepare Base Results DataFrame ---
    logger.info("Preparing base results data...")
    df_quali = df_raw_results[df_raw_results['SessionName'] == 'Q'].pivot_table(index=['Year', 'RoundNumber', 'Abbreviation'], values='Position', aggfunc='min').rename(columns={'Position': 'QualiPosition'}).reset_index()
    df_race = df_raw_results[df_raw_results['SessionName'] == 'R'].copy()
    if df_race.empty: logger.error("No Race results found."); return pd.DataFrame(), [], None
    df = pd.merge(df_race, df_quali, on=['Year', 'RoundNumber', 'Abbreviation'], how='left')
    df['GridPosition'] = utils.safe_to_numeric(df['GridPosition'], fallback=None); df['GridPosition'].fillna(df['QualiPosition'], inplace=True); df['GridPosition'] = utils.safe_to_numeric(df['GridPosition'], fallback=config.WORST_EXPECTED_POS).astype(int)
    df['Position'] = utils.safe_to_numeric(df['Position'], fallback=config.WORST_EXPECTED_POS).astype(int)
    df['Points'] = utils.safe_to_numeric(df['Points'], fallback=0.0).astype(float)
    df['TrackLocation'] = df['Location'].fillna('Unknown')


    # --- Feature Engineering ---
    base_feature_cols = []

    # 1. Lap Features
    if config.LOAD_LAPS and not df_laps.empty:
        logger.info("Calculating features from lap data..."); laps_prac = df_laps[df_laps['SessionName'].isin(['FP1', 'FP2', 'FP3'])].copy()
        if not laps_prac.empty:
            laps_prac['LapTime'] = pd.to_numeric(laps_prac['LapTime'], errors='coerce'); laps_prac['Stint'] = pd.to_numeric(laps_prac['Stint'], errors='coerce'); laps_prac['IsAccurate'] = pd.to_numeric(laps_prac['IsAccurate'], errors='coerce')
            stint_stats = laps_prac.groupby(['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'Stint']).apply(calculate_stint_stats).reset_index()
            relevant_stints = stint_stats[stint_stats['LapCount'] >= config.MIN_STINT_LAPS]
            if not relevant_stints.empty:
                pace_medium = relevant_stints[relevant_stints['Compound'] == 'MEDIUM'].groupby(['Year', 'RoundNumber', 'DriverNumber'])['AvgLapTime'].mean().reset_index().rename(columns={'AvgLapTime': 'AvgPaceMediumFP'})
                pace_hard = relevant_stints[relevant_stints['Compound'] == 'HARD'].groupby(['Year', 'RoundNumber', 'DriverNumber'])['AvgLapTime'].mean().reset_index().rename(columns={'AvgLapTime': 'AvgPaceHardFP'})
                deg_medium = relevant_stints[relevant_stints['Compound'] == 'MEDIUM'].groupby(['Year', 'RoundNumber', 'DriverNumber'])['Degradation'].mean().reset_index().rename(columns={'Degradation': 'AvgDegMediumFP'})
                practice_pace_features = pd.merge(pace_medium, pace_hard, on=['Year', 'RoundNumber', 'DriverNumber'], how='outer')
                practice_pace_features = pd.merge(practice_pace_features, deg_medium, on=['Year', 'RoundNumber', 'DriverNumber'], how='outer')
                df = pd.merge(df, practice_pace_features, on=['Year', 'RoundNumber', 'DriverNumber'], how='left'); new_cols = practice_pace_features.columns.drop(['Year', 'RoundNumber', 'DriverNumber']).tolist(); base_feature_cols.extend(new_cols); logger.info(f"Added practice pace/deg features: {new_cols}")
            else: logger.warning("No relevant practice stints found.")
        else: logger.warning("No practice lap data found.")

    # 2. Weather Features (Historical Averages)
    if config.LOAD_WEATHER and not df_weather.empty:
        logger.info("Calculating historical weather features...")
        weather_race = df_weather[df_weather['SessionName'] == 'R'].copy()
        if not weather_race.empty:
             weather_agg = weather_race.groupby(['Year', 'RoundNumber']).agg(
                 HistAvgTrackTemp=('TrackTemp', 'mean'), # Use distinct names for historical
                 HistMaxTrackTemp=('TrackTemp', 'max'),
                 HistAvgAirTemp=('AirTemp', 'mean'),
                 HistAvgHumidity=('Humidity', 'mean'),
                 HistAvgWindSpeed=('WindSpeed', 'mean'),
                 HistWasRainy=('Rainfall', lambda x: (x > 0).any().astype(int))
             ).reset_index()
             df = pd.merge(df, weather_agg, on=['Year', 'RoundNumber'], how='left'); new_cols = weather_agg.columns.drop(['Year', 'RoundNumber']).tolist(); base_feature_cols.extend(new_cols); logger.info(f"Added historical race weather features: {new_cols}")
        else: logger.warning("No Race weather data found for historical aggregation.")

    # 3. Pit Stop Features
    if config.LOAD_LAPS and not df_pits.empty:
         logger.info("Calculating pit stop features..."); pits_race = df_pits[df_pits['SessionName'] == 'R']
         if not pits_race.empty:
              pit_counts = pits_race.groupby(['Year', 'RoundNumber', 'DriverNumber'])['StopNumber'].max().reset_index().rename(columns={'StopNumber': 'NumPits'})
              df = pd.merge(df, pit_counts, on=['Year', 'RoundNumber', 'DriverNumber'], how='left')
              df['NumPits'].fillna(0, inplace=True); base_feature_cols.append('NumPits'); logger.info("Added pit count feature.")
         else: df['NumPits'] = 0.0 # Ensure float if added manually
    else: df['NumPits'] = 0.0 # Ensure float if no pit data at all

    # --- Ensure base feature columns exist even if data was missing ---
    expected_lap_pit_weather_cols = ['AvgPaceMediumFP', 'AvgPaceHardFP', 'AvgDegMediumFP', 'HistAvgTrackTemp', 'HistMaxTrackTemp', 'HistAvgAirTemp', 'HistAvgHumidity', 'HistAvgWindSpeed', 'HistWasRainy', 'NumPits']
    for col in expected_lap_pit_weather_cols:
         if col not in df.columns: df[col] = np.nan; # Add as NaN
         if col not in base_feature_cols: base_feature_cols.append(col)


    # --- Calculate Rolling/Expanding Features ---
    logger.info("Calculating rolling/expanding features..."); df.sort_values(by=['Year', 'RoundNumber', 'Abbreviation'], inplace=True); df.reset_index(drop=True, inplace=True)
    df = add_rolling_features(df, 'Abbreviation', 'Position', config.N_LAST_RACES_FEATURES, 'PosLastN', config.WORST_EXPECTED_POS); df = add_rolling_features(df, 'Abbreviation', 'Points', config.N_LAST_RACES_FEATURES, 'PtsLastN', 0.0); base_feature_cols.extend(['RollingAvgPosLastN', 'RollingAvgPtsLastN'])
    df = add_expanding_features(df, ['Abbreviation', 'TrackLocation'], 'Position', 'PosThisTrack', config.WORST_EXPECTED_POS); df = add_expanding_features(df, ['Abbreviation', 'TrackLocation'], 'Points', 'PtsThisTrack', 0.0); base_feature_cols.extend(['ExpandingAvgPosThisTrack', 'ExpandingAvgPtsThisTrack'])
    team_points = df.groupby(['Year', 'RoundNumber', 'TeamName'])['Points'].mean().reset_index(); team_points.sort_values(by=['Year', 'RoundNumber', 'TeamName'], inplace=True); team_points = add_rolling_features(team_points, 'TeamName', 'Points', config.N_LAST_RACES_TEAM, 'TeamPtsLastN', 0.0)
    df = pd.merge(df, team_points[['Year', 'RoundNumber', 'TeamName', 'RollingAvgTeamPtsLastN']], on=['Year', 'RoundNumber', 'TeamName'], how='left'); df['RollingAvgTeamPtsLastN'].fillna(0.0, inplace=True); base_feature_cols.append('RollingAvgTeamPtsLastN')
    df['RaceCount'] = df.groupby('Abbreviation').cumcount()
    df['SeasonPointsBeforeRace'] = df.groupby(['Year', 'Abbreviation'])['Points'].shift(1).fillna(0); df['SeasonPointsBeforeRace'] = df.groupby(['Year', 'Abbreviation'])['SeasonPointsBeforeRace'].cumsum(); base_feature_cols.extend(['RaceCount', 'SeasonPointsBeforeRace'])


    # --- <<< CREATE HISTORICAL EQUIVALENTS FOR FORECAST FEATURES >>> ---
    # Use historical calculated averages/flags to create columns with "Forecast" names
    # This allows the model to learn a relationship for these named features
    logger.info("Creating historical equivalents for forecast features...")
    df['ForecastTemp'] = df.get('HistAvgAirTemp', pd.Series(np.nan, index=df.index)) # Use Avg Air Temp as proxy
    df['ForecastRainProb'] = df.get('HistWasRainy', pd.Series(0, index=df.index)).astype(float) # Use 0.0 or 1.0 based on HistWasRainy
    df['ForecastWindSpeed'] = df.get('HistAvgWindSpeed', pd.Series(np.nan, index=df.index))
    # Add these to the base feature list so they are included in final selection
    forecast_equiv_cols = ['ForecastTemp', 'ForecastRainProb', 'ForecastWindSpeed']
    base_feature_cols.extend(forecast_equiv_cols)
    logger.info(f"Added historical forecast equivalents: {forecast_equiv_cols}")
    # --- <<< END FORECAST EQUIVALENTS >>> ---


    # Track Characteristics (Categorical)
    df['TrackType'] = df['TrackLocation'].map(config.TRACK_CHARACTERISTICS).fillna('Unknown')


    # --- Finalize Feature List ---
    feature_cols = ['GridPosition'] + list(dict.fromkeys(base_feature_cols)) # Unique, keeps order somewhat

    # --- Final Data Cleaning for Features ---
    logger.info("Performing final cleaning on feature set...")
    df[feature_cols] = df[feature_cols].fillna(config.FILL_NA_VALUE) # Fill any remaining NaNs
    df.replace([np.inf, -np.inf], config.FILL_NA_VALUE, inplace=True)

    # --- Handle Categorical TrackType ---
    categorical_features = ['TrackType']
    if config.MODEL_TYPE.lower() in ['lightgbm']:
        if 'TrackType' in df.columns: df[col] = df[col].astype('category'); feature_cols.append('TrackType')
    else: # OHE
        if 'TrackType' in df.columns:
             df = pd.get_dummies(df, columns=categorical_features, prefix='Track', dummy_na=False, dtype=int)
             ohe_cols = [col for col in df.columns if col.startswith('Track_')]; feature_cols.extend(ohe_cols)
             if 'TrackType' in feature_cols: feature_cols.remove('TrackType')

    # Final pass to ensure unique and sorted columns for consistency
    feature_cols = sorted(list(set(feature_cols)))

    # --- Target Variable ---
    target_col = 'Position'

    # --- Final Validation ---
    check_cols = feature_cols + [target_col]; missing_in_df = [c for c in check_cols if c not in df.columns]
    if missing_in_df: logger.error(f"CRITICAL: Columns missing from DataFrame: {missing_in_df}"); return pd.DataFrame(), [], None
    nan_check = df[check_cols].isnull().sum(); nan_cols = nan_check[nan_check > 0]
    if not nan_cols.empty: logger.warning(f"NaNs remain AFTER final cleaning:\n{nan_cols}"); df.fillna(config.FILL_NA_VALUE, inplace=True)
    inf_check = df[check_cols].select_dtypes(include=np.number).isin([np.inf, -np.inf]).sum(); inf_cols = inf_check[inf_check > 0]
    if not inf_cols.empty: logger.warning(f"Infs remain AFTER final cleaning:\n{inf_cols}"); df.replace([np.inf, -np.inf], config.FILL_NA_VALUE * 1000, inplace=True)

    logger.info(f"Feature engineering complete. Final DataFrame shape: {df.shape}"); logger.info(f"Target: {target_col}"); logger.info(f"Final features ({len(feature_cols)}): {feature_cols}")
    final_cols_to_keep = list(set(['Year', 'RoundNumber', 'Abbreviation', 'TeamName', target_col] + feature_cols)); final_cols_to_keep = [col for col in final_cols_to_keep if col in df.columns]
    df_final = df[final_cols_to_keep].copy()

    return df_final, feature_cols, target_col