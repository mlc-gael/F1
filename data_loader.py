# /f1_predictor/data_loader.py

import fastf1 as ff1
import pandas as pd
import numpy as np
import time
import warnings
import os
import sys
from tqdm import tqdm
import datetime

# Import project modules
import config
import database
import utils

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# Setup logger
logger = utils.get_logger(__name__)

# --- Constants ---
DEFAULT_API_SLEEP_S = 6.0 # Increased sleep
RATE_LIMIT_SLEEP_S = 60.0 * 5 # 5 minutes
MAX_RATE_LIMIT_RETRIES = 2
SESSION_IDENTIFIERS_TO_FETCH = ['Q', 'S', 'SQ', 'SS', 'FP1', 'FP2', 'FP3','R']
# Define a specific return status for persistent rate limit failure
RATE_LIMIT_FAILURE_STATUS = "RATE_LIMIT_FAILURE"

# Configure FastF1 Cache
try:
    if config.CACHE_DIR and not os.path.exists(config.CACHE_DIR): os.makedirs(config.CACHE_DIR, exist_ok=True)
    ff1.Cache.enable_cache(config.CACHE_DIR); logger.info(f"FastF1 cache enabled at: {config.CACHE_DIR}")
except Exception as e: logger.error(f"CRITICAL: Failed to enable FastF1 cache: {e}", exc_info=True)


# --- Helper Functions ---
def parse_lap_time(time_obj):
    return utils.parse_timedelta_to_seconds(time_obj)


# --- Data Processing Functions ---

def process_session_results(session, year, round_number, session_name):
    # --- MODIFICATION START: Add DataNotLoadedError check ---
    try:
        # --- CHANGE START: Improved Logging for Missing Results ---
        # Check if results data was loaded before accessing session.results
        if session is None or not hasattr(session, 'results') or session.results is None or session.results.empty:
            logger.warning(f"No session.results data found or loaded for {year} R{round_number} Session {session_name}. Returning empty results DataFrame.")
            # Log specific reason if possible
            if session is None: logger.debug("Reason: Session object itself is None.")
            elif not hasattr(session, 'results'): logger.debug("Reason: Session object lacks 'results' attribute.")
            elif session.results is None: logger.debug("Reason: session.results attribute is None.")
            elif session.results.empty: logger.debug("Reason: session.results DataFrame is empty.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Return empty
        # --- CHANGE END ---

        results = session.results.copy(); logger.debug(f"Processing results for {year} R{round_number} {session_name}. Input columns: {results.columns.tolist()}")
        results['Year'] = int(year); results['RoundNumber'] = int(round_number); results['SessionName'] = str(session_name)
        # Ensure Abbreviation exists and is string - critical for joins later
        if 'Abbreviation' not in results.columns:
             logger.warning(f"Missing 'Abbreviation' in raw results for {year} R{round_number} {session_name}. Cannot process results reliably.")
             return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Return empty
        results['Abbreviation'] = results['Abbreviation'].astype(str)

        # Map source columns, handle missing ones gracefully
        source_column_map = { 'DriverNumber': 'DriverNumber', 'Abbreviation': 'Abbreviation', 'TeamName': 'TeamName', 'GridPosition': 'GridPosition', 'Position': 'Position', 'Points': 'Points', 'Status': 'Status', 'Laps': 'Laps', 'FastestLapTime': 'FastestLapTime', 'Q1': 'Q1', 'Q2': 'Q2', 'Q3': 'Q3', 'FullName': 'FullName' }
        results_df = pd.DataFrame() # Initialize empty DataFrame
        cols_to_process = {k: v for k, v in source_column_map.items() if k in results.columns}
        results_df = results[list(cols_to_process.keys())].rename(columns=cols_to_process)

        # Define schema expected in DB
        db_results_schema = ['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Points', 'Status', 'Laps', 'FastestLapTime', 'Q1', 'Q2', 'Q3']

        # Ensure basic IDs are present and string type
        for col in ['DriverNumber', 'Abbreviation', 'TeamName']:
            if col not in results_df.columns: results_df[col] = 'UNK' # Assign default if missing
            results_df[col] = results_df[col].astype(str).fillna('UNK')
        results_df['Year'] = int(year); results_df['RoundNumber'] = int(round_number); results_df['SessionName'] = str(session_name)

        # Process numeric/time columns
        results_df['GridPosition'] = utils.safe_to_numeric(results_df.get('GridPosition'), fallback=float(config.WORST_EXPECTED_POS))
        results_df['Position'] = utils.safe_to_numeric(results_df.get('Position'), fallback=None) # Keep NaN here initially
        results_df['Points'] = utils.safe_to_numeric(results_df.get('Points'), fallback=0.0)
        results_df['Laps'] = utils.safe_to_numeric(results_df.get('Laps'), fallback=0.0)
        for time_col in ['FastestLapTime', 'Q1', 'Q2', 'Q3']:
            if time_col in results_df.columns: results_df[time_col] = results_df[time_col].apply(parse_lap_time).astype(float)
            else: results_df[time_col] = np.nan # Ensure column exists if expected

        # Process string columns
        if 'Status' in results_df.columns: results_df['Status'] = results_df['Status'].astype(str).fillna('Unknown')
        else: results_df['Status'] = 'Unknown'
        if 'FullName' in results_df.columns: results_df['FullName'] = results_df['FullName'].astype(str).fillna('')
        else: results_df['FullName'] = '' # Ensure exists for driver info

        # --- Create Drivers/Teams DataFrames ---
        driver_info_cols = ['DriverNumber', 'Abbreviation', 'TeamName', 'FullName']
        # Use Abbreviation as the key for uniqueness
        drivers_df = results_df.dropna(subset=['Abbreviation']).loc[:, [c for c in driver_info_cols if c in results_df.columns]].drop_duplicates(subset=['Abbreviation']).copy()
        if not drivers_df.empty: drivers_df['Nationality'] = None # Add placeholder

        teams_df = results_df.dropna(subset=['TeamName']).loc[:, ['TeamName']].drop_duplicates().copy()
        if not teams_df.empty: teams_df['Nationality'] = None # Add placeholder

        # Ensure all DB schema columns exist in results_df before selection
        for col in db_results_schema:
             if col not in results_df.columns:
                  # Assign appropriate defaults based on expected type
                  if col in ['GridPosition', 'Position', 'Points', 'Laps', 'FastestLapTime', 'Q1', 'Q2', 'Q3']: results_df[col] = np.nan
                  elif col in ['DriverNumber', 'TeamName', 'Status']: results_df[col] = 'UNK'
                  elif col == 'Abbreviation': results_df[col] = 'UNK' # Should exist from check above
                  # Year, RoundNumber, SessionName already set
                  else: results_df[col] = None # Fallback for any unexpected missing cols

        # Select final columns in DB order
        results_df = results_df[db_results_schema]

        # Check for NaNs in essential identifiers AFTER processing and BEFORE dropna
        essential_ids = ['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'Abbreviation']
        nan_check_before_drop = results_df[essential_ids].isnull().sum()
        if nan_check_before_drop.sum() > 0:
            logger.warning(f"NaNs found in essential identifiers BEFORE dropna for {year} R{round_number} {session_name}: \n{nan_check_before_drop[nan_check_before_drop > 0]}")

        initial_rows = len(results_df)
        # Drop rows where essential identifiers couldn't be determined (should be rare now)
        results_df = results_df.dropna(subset=essential_ids)
        rows_dropped = initial_rows - len(results_df)
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} rows from results due to missing essential identifiers for {year} R{round_number} {session_name}.")
        logger.debug(f"Shape *after* dropna essential_ids: {results_df.shape}")

        # Ensure remaining required schemas for drivers/teams
        db_drivers_schema = ['Abbreviation', 'DriverNumber', 'FullName', 'Nationality']
        drivers_df = drivers_df[[col for col in db_drivers_schema if col in drivers_df.columns]]

        db_teams_schema = ['TeamName', 'Nationality']
        teams_df = teams_df[[col for col in db_teams_schema if col in teams_df.columns]]

        logger.debug(f"Finished processing results for {year} R{round_number} {session_name}. Final shape: {results_df.shape}")
        if results_df.empty:
             logger.warning(f"Results processing resulted in an empty DataFrame for {year} R{round_number} {session_name}")
        return results_df, drivers_df, teams_df

    except ff1.core.DataNotLoadedError:
        logger.warning(f"Session results data not loaded (likely API issue or future event) for {year} R{round_number} {session_name}. Skipping results processing.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Return empty
    except Exception as e:
        logger.error(f"CRITICAL ERROR during process_session_results for {year} R{round_number} {session_name}: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # --- MODIFICATION END ---


def process_laps_and_pits(session, year, round_number, session_name):
    """Processes lap times and pit stops into DataFrames."""
    laps_df_final = pd.DataFrame()
    pits_df_final = pd.DataFrame()

    try:
        # Check if lap data was loaded before accessing session.laps
        if session is None or not hasattr(session, 'laps') or session.laps is None or session.laps.empty:
            logger.warning(f"No lap data found or loaded for {year} R{round_number} {session_name}")
            return laps_df_final, pits_df_final

        laps = session.laps.copy()
        logger.debug(f"Processing {len(laps)} laps for {year} R{round_number} {session_name}")
        laps['Year'] = int(year); laps['RoundNumber'] = int(round_number); laps['SessionName'] = str(session_name)

        lap_cols_to_keep = ['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'LapNumber', 'LapTime', 'Stint', 'TyreLife', 'Compound', 'IsAccurate', 'IsPitOutLap', 'IsPitInLap', 'Sector1Time', 'Sector2Time', 'Sector3Time']
        laps_df_final = laps[[col for col in lap_cols_to_keep if col in laps.columns]].copy()

        # Ensure all expected DB columns exist and have correct types
        lap_cols_db = ['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'LapNumber', 'LapTime', 'Stint', 'TyreLife', 'Compound', 'IsAccurate', 'IsPitOutLap', 'IsPitInLap', 'Sector1Time', 'Sector2Time', 'Sector3Time']
        for col in lap_cols_db:
             if col not in laps_df_final.columns:
                 # Assign defaults based on expected type
                 if col in ['LapTime', 'TyreLife', 'Sector1Time', 'Sector2Time', 'Sector3Time']: laps_df_final[col] = np.nan
                 elif col in ['IsAccurate', 'IsPitOutLap', 'IsPitInLap']: laps_df_final[col] = 0 # Default to False/0
                 elif col in ['Stint', 'LapNumber']: laps_df_final[col] = -1
                 elif col == 'Compound': laps_df_final[col] = 'UNKNOWN'
                 elif col == 'DriverNumber': laps_df_final[col] = 'UNK'
                 # Year, RoundNumber, SessionName are added initially
                 else: laps_df_final[col] = None

        # Ensure DriverNumber is string
        laps_df_final['DriverNumber'] = laps_df_final['DriverNumber'].astype(str).fillna('UNK')

        # Convert types
        for time_col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
             laps_df_final[time_col] = laps_df_final[time_col].apply(parse_lap_time).astype(float)
        for bool_col in ['IsAccurate', 'IsPitOutLap', 'IsPitInLap']:
             laps_df_final[bool_col] = laps_df_final[bool_col].fillna(0).astype(bool).astype(int)
        for int_col in ['Stint', 'LapNumber']:
            laps_df_final[int_col] = utils.safe_to_numeric(laps_df_final[int_col], fallback=-1).astype(int)
        laps_df_final['TyreLife'] = utils.safe_to_numeric(laps_df_final['TyreLife'], fallback=np.nan).astype(float)
        laps_df_final['Compound'] = laps_df_final['Compound'].astype(str).fillna('UNKNOWN')

        # --- Extract Pit Stops ---
        pits_df_final = pd.DataFrame()
        pit_cols_db = ['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'StopNumber', 'LapNumber', 'PitDuration']

        # Use get_pit_log if available (preferable) - wrapped in try/except
        try:
            if hasattr(session, 'get_pit_log') and callable(session.get_pit_log):
                 pit_log = session.get_pit_log()
                 if not pit_log.empty:
                     pit_log_cols_needed = ['DriverNumber', 'LapNumber', 'StopDuration']
                     if all(col in pit_log.columns for col in pit_log_cols_needed):
                        pits_df_tmp = pit_log[pit_log_cols_needed].copy()
                        pits_df_tmp.rename(columns={'StopDuration': 'PitDuration'}, inplace=True)
                        pits_df_tmp['Year'] = int(year); pits_df_tmp['RoundNumber'] = int(round_number); pits_df_tmp['SessionName'] = str(session_name)
                        pits_df_tmp['DriverNumber'] = pits_df_tmp['DriverNumber'].astype(str).fillna('UNK') # Ensure string
                        pits_df_tmp.sort_values(by=['DriverNumber', 'LapNumber'], inplace=True)
                        pits_df_tmp['StopNumber'] = pits_df_tmp.groupby('DriverNumber').cumcount() + 1
                        pits_df_tmp['PitDuration'] = pits_df_tmp['PitDuration'].apply(parse_lap_time).astype(float)
                        pits_df_tmp['LapNumber'] = utils.safe_to_numeric(pits_df_tmp['LapNumber'], fallback=-1).astype(int)
                        # Ensure all DB columns exist
                        for col in pit_cols_db:
                            if col not in pits_df_tmp.columns: pits_df_tmp[col] = np.nan if col == 'PitDuration' else -1 if col in ['StopNumber', 'LapNumber'] else 'UNK'
                        pits_df_final = pits_df_tmp[pit_cols_db]
                        logger.debug(f"Extracted {len(pits_df_final)} pits using get_pit_log.")
                     else:
                          logger.warning(f"Pit log available but missing expected columns for {year} R{round_number} {session_name}. Columns: {pit_log.columns}. Skipping pit log.")
                 else:
                      logger.debug(f"get_pit_log() returned empty DataFrame for {year} R{round_number} {session_name}")
            else:
                 logger.debug(f"get_pit_log() not available for {year} R{round_number} {session_name}")

        except ff1.core.DataNotLoadedError:
            logger.warning(f"Could not run get_pit_log() - data likely not loaded for {year} R{round_number} {session_name}.")
        except Exception as pit_log_err:
            logger.error(f"Error processing get_pit_log() for {year} R{round_number} {session_name}: {pit_log_err}", exc_info=True)


        # Fallback only if PitInLap exists and pit_log failed/unavailable
        # --- CHANGE START: Add diagnostic logging ---
        logger.debug(f"Pit fallback check: pits_df_final empty? {pits_df_final.empty}. Laps df empty? {laps_df_final.empty}")
        if not laps_df_final.empty:
            logger.debug(f"Laps df columns: {laps_df_final.columns.tolist()}")
            if 'IsPitInLap' in laps_df_final.columns:
                pit_laps_found = (laps_df_final['IsPitInLap'] == 1).any()
                logger.debug(f"'IsPitInLap' column exists. Any pit laps found? {pit_laps_found}")
            else:
                logger.debug("'IsPitInLap' column NOT found in laps_df_final.")
        # --- CHANGE END ---
        if pits_df_final.empty and 'IsPitInLap' in laps_df_final.columns:
            logger.debug(f"Falling back to IsPitInLap for pit stops for {year} R{round_number} {session_name}")
            pits_in = laps_df_final[(laps_df_final['IsPitInLap'] == 1) & (laps_df_final['DriverNumber'] != 'UNK')].copy() # Filter out UNK drivers
            if not pits_in.empty:
                pits_df_tmp = pits_in[['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'LapNumber']].copy()
                pits_df_tmp['DriverNumber'] = pits_df_tmp['DriverNumber'].astype(str) # Already string, but ensure
                pits_df_tmp.sort_values(by=['DriverNumber', 'LapNumber'], inplace=True)
                pits_df_tmp['StopNumber'] = pits_df_tmp.groupby('DriverNumber').cumcount() + 1
                pits_df_tmp['PitDuration'] = np.nan # Duration unknown
                # Ensure all DB columns exist
                for col in pit_cols_db:
                     if col not in pits_df_tmp.columns: pits_df_tmp[col] = np.nan if col == 'PitDuration' else -1 if col in ['StopNumber', 'LapNumber'] else 'UNK'
                pits_df_final = pits_df_tmp[pit_cols_db]
                logger.debug(f"Extracted {len(pits_df_final)} potential pits using IsPitInLap (duration unavailable).")
            else:
                logger.debug(f"No laps found with IsPitInLap == 1 for fallback in {year} R{round_number} {session_name}.")

        # Log final warning if still no pits found
        if pits_df_final.empty:
             logger.warning(f"Could not determine pit stops for {year} R{round_number} {session_name} using either method.")

        logger.debug(f"Processed laps ({len(laps_df_final)}) and pits ({len(pits_df_final)}) for {year} R{round_number} {session_name}")

    except ff1.core.DataNotLoadedError:
        logger.warning(f"Lap data not loaded (likely API issue or future event) for {year} R{round_number} {session_name}. Skipping lap/pit processing.")
        laps_df_final, pits_df_final = pd.DataFrame(), pd.DataFrame() # Return empty
    except Exception as e:
        logger.error(f"Error processing laps/pits for {year} R{round_number} {session_name}: {e}", exc_info=True)
        laps_df_final, pits_df_final = pd.DataFrame(), pd.DataFrame() # Return empty on error


    # Ensure consistent dtypes before returning
    if not laps_df_final.empty:
         laps_df_final = laps_df_final.astype({
             'Year': 'int64', 'RoundNumber': 'int64', 'SessionName': 'str',
             'DriverNumber': 'str', 'LapNumber': 'int64', 'LapTime': 'float64',
             'Stint': 'int64', 'TyreLife': 'float64', 'Compound': 'str',
             'IsAccurate': 'int64', 'IsPitOutLap': 'int64', 'IsPitInLap': 'int64',
             'Sector1Time': 'float64', 'Sector2Time': 'float64', 'Sector3Time': 'float64'
         }, errors='ignore') # Ignore errors if cast fails

    if not pits_df_final.empty:
        pits_df_final = pits_df_final.astype({
            'Year': 'int64', 'RoundNumber': 'int64', 'SessionName': 'str',
            'DriverNumber': 'str', 'StopNumber': 'int64', 'LapNumber': 'int64',
            'PitDuration': 'float64'
        }, errors='ignore') # Ignore errors if cast fails

    return laps_df_final, pits_df_final


def process_weather_data(session, year, round_number, session_name):
    weather_df_final = pd.DataFrame()
    try:
        # Check if weather data was loaded
        if session is None or not hasattr(session, 'weather_data') or session.weather_data is None or session.weather_data.empty:
            logger.warning(f"No weather data found or loaded for {year} R{round_number} {session_name}")
            return weather_df_final

        weather = session.weather_data.copy(); logger.debug(f"Processing {len(weather)} weather entries for {year} R{round_number} {session_name}")
        weather['Year'] = int(year); weather['RoundNumber'] = int(round_number); weather['SessionName'] = str(session_name)

        if 'Time' in weather.columns:
             weather['TimeSeconds'] = weather['Time'].apply(parse_lap_time).astype(float)
        else:
             logger.warning("Weather data missing 'Time' column. Cannot create TimeSeconds."); weather['TimeSeconds'] = np.nan

        weather_cols_to_keep = ['Year', 'RoundNumber', 'SessionName', 'TimeSeconds', 'AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'WindDirection', 'Rainfall']
        weather_df_final = weather[[col for col in weather_cols_to_keep if col in weather.columns]].copy()
        # Rename TimeSeconds back to Time for DB schema consistency
        weather_df_final.rename(columns={'TimeSeconds': 'Time'}, inplace=True)

        # Ensure all DB schema columns exist and have correct types
        weather_cols_db = ['Year', 'RoundNumber', 'SessionName', 'Time', 'AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'WindDirection', 'Rainfall']
        for col in weather_cols_db:
            if col not in weather_df_final.columns:
                # Assign defaults
                if col in ['Time', 'AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'WindDirection']: weather_df_final[col] = np.nan
                elif col == 'Rainfall': weather_df_final[col] = 0
                else: weather_df_final[col] = None # Should only be ID cols if missing

            # Convert types AFTER ensuring column exists
            if col == 'Rainfall':
                # Convert boolean/object/numeric to 0 or 1 integer
                 weather_df_final[col] = pd.to_numeric(weather_df_final[col], errors='coerce').fillna(0).astype(bool).astype(int)
            elif col == 'WindDirection':
                weather_df_final[col] = utils.safe_to_numeric(weather_df_final[col], fallback=np.nan).astype(float) # Keep as float
            elif col not in ['Year', 'RoundNumber', 'SessionName']: # Other numeric float columns
                  weather_df_final[col] = utils.safe_to_numeric(weather_df_final[col], fallback=np.nan).astype(float)

        # Drop rows where Time is NaN (essential key)
        initial_rows = len(weather_df_final)
        weather_df_final.dropna(subset=['Time'], inplace=True)
        rows_dropped = initial_rows - len(weather_df_final)
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} weather rows due to missing Time for {year} R{round_number} {session_name}")

        logger.debug(f"Processed weather data for {year} R{round_number} {session_name}. Final shape: {weather_df_final.shape}")

    except ff1.core.DataNotLoadedError:
        logger.warning(f"Weather data not loaded (likely API issue or future event) for {year} R{round_number} {session_name}. Skipping weather processing.")
        weather_df_final = pd.DataFrame() # Return empty
    except Exception as e:
        logger.error(f"Error processing weather for {year} R{round_number} {session_name}: {e}", exc_info=True); weather_df_final = pd.DataFrame()

    return weather_df_final


def fetch_single_session_with_retry(year, round_number, session_code):
    """
    Attempts to fetch a single session, handling rate limits and specific errors.
    Returns the session object on success, None on non-rate-limit failure,
    or RATE_LIMIT_FAILURE_STATUS on persistent rate limit failure.
    Reduces log level for expected "session does not exist" errors.
    """
    session = None
    retries = 0
    while retries <= MAX_RATE_LIMIT_RETRIES:
        try:
            session = ff1.get_session(year, round_number, session_code)
            # Define load options - disable messages unless needed for specific debugging
            load_kwargs = {'laps': config.LOAD_LAPS, 'weather': config.LOAD_WEATHER, 'messages': False, 'telemetry': False}
            logger.debug(f"Attempting load: {year} R{round_number} {session_code} with {load_kwargs}")

            try:
                 # Attempt to load data - this might fail for future/incomplete events
                 session.load(**load_kwargs)
                 logger.debug(f"Session loaded successfully: {year} R{round_number} {session_code}")
                 # Check if essential data (like results) might be missing even after load attempt
                 if not hasattr(session, 'results') or session.results is None:
                      logger.warning(f"Session loaded, but 'session.results' is missing or None for {year} R{round_number} {session_code}. Data might be incomplete.")
                      # Decide whether to return None or the session object based on needs
                      # Returning the object allows processing laps/weather if they exist
                 return session # Return session even if some parts might be missing

            except ff1.core.DataNotLoadedError as load_err:
                 # This can happen for future events or API issues where core data is missing
                 logger.warning(f"DataNotLoadedError during session.load() for {year} R{round_number} {session_code}: {load_err}. Treating as 'session not fully available'.")
                 # Log the specific error message if informative
                 logger.debug(f"Specific DataNotLoadedError: {str(load_err)}")
                 return None # Return None, indicating data isn't usable for results/laps etc.
            except ff1.exceptions.FastF1Error as ff1_err:
                # Catch specific FastF1 errors during load
                logger.error(f"FastF1Error during session.load() for {year} R{round_number} {session_code}: {ff1_err}")
                return None # Treat as failure for this session
            except Exception as load_err:
                 # Catch other potential errors during load
                 logger.error(f"Unexpected error during session.load() for {year} R{round_number} {session_code}: {load_err}", exc_info=True)
                 return None # Treat as failure for this session

        except ff1.RateLimitExceededError as e:
            retries += 1
            if retries > MAX_RATE_LIMIT_RETRIES:
                logger.error(f"RATE LIMIT EXCEEDED after {MAX_RATE_LIMIT_RETRIES} retries for {year} R{round_number} {session_code}: {e}. Giving up on this session.")
                return RATE_LIMIT_FAILURE_STATUS
            wait = RATE_LIMIT_SLEEP_S * retries
            logger.warning(f"Rate limit hit ({retries}/{MAX_RATE_LIMIT_RETRIES}) for {year} R{round_number} {session_code}. Sleeping {wait}s...")
            time.sleep(wait)
            logger.info(f"Retrying fetch {retries}...")

        except ValueError as e:
             # Handle errors related to invalid session identifiers or missing schedule data
             error_msg_lower = str(e).lower()
             session_not_exist_patterns = [ "does not exist for this event", "no event found", "session identifier", "session type", "not found for season" ]
             is_non_existent_error = any(p in error_msg_lower for p in session_not_exist_patterns)
             if is_non_existent_error:
                  logger.debug(f"Session '{session_code}' not available or invalid for {year} R{round_number}. Skipping. (Reason: {e})")
                  return None # Session doesn't exist, not an error
             elif "failed to load any schedule data" in error_msg_lower:
                  logger.error(f"Failed to load schedule data during get_session for {year} R{round_number} {session_code} (likely due to earlier rate limit/network issue). Skipping session.")
                  return None # Cannot proceed without schedule
             else:
                  logger.error(f"Unexpected ValueError loading {year} R{round_number} {session_code}: {e}", exc_info=True)
                  return None # Other ValueError

        except ConnectionError as e:
            # Handle network-related issues
            logger.error(f"Connection error fetching {year} R{round_number} {session_code}: {e}. Skipping session.")
            time.sleep(5) # Wait a bit before potentially trying the next session/event
            return None
        except ff1.exceptions.FastF1Error as e:
            # Catch broader FastF1 API or internal errors
            logger.error(f"FastF1 API/Internal error loading {year} R{round_number} {session_code}: {e}", exc_info=False) # Don't need full traceback usually
            return None
        except Exception as e:
            # Catch any other unexpected exceptions during get_session
            logger.error(f"Generic unexpected error during get_session for {year} R{round_number} {session_code}: {e}", exc_info=True)
            return None

    # Should only be reached if loop finishes due to retries > MAX_RATE_LIMIT_RETRIES
    logger.error(f"Exceeded max rate limit retries ({MAX_RATE_LIMIT_RETRIES}) for {year} R{round_number} {session_code}.")
    return RATE_LIMIT_FAILURE_STATUS


def fetch_and_store_event(year, event_row):
    """
    Fetches, processes, and stores data for relevant sessions of a single event.
    Returns True on success (all sessions attempted),
    Returns False if a persistent rate limit failure occurred for any session.
    """
    try:
        round_number = int(event_row['RoundNumber'])
        event_location = event_row.get('Location', f'UNK R{round_number}')
        event_name = event_row.get('EventName', f'UNK R{round_number}')
        event_date_str = event_row.get('EventDate')
        logger.info(f"--- Processing Event: {year} R{round_number} - {event_name} ({event_location}) ---")
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Invalid event data: {e}. Row: {event_row}")
        return True # Continue processing other events, treat this one as failed

    session_iterator = SESSION_IDENTIFIERS_TO_FETCH
    persist_rate_limit_hit = False

    for session_code in session_iterator:
        # --- CHANGE START: Add enhanced logging for data existence check ---
        logger.debug(f"Checking data existence for: {year} R{round_number} {session_code}")
        data_already_exists = False
        try:
            # Check specifically for 'results' table data as a proxy for session processing
            data_already_exists = database.check_data_exists(year, round_number, session_code)
            if data_already_exists:
                logger.info(f"Data check returned TRUE: Results data exists for {year} R{round_number} {session_code}. Skipping fetch.")
                continue # Skip to next session code
            else:
                 logger.debug(f"Data check returned FALSE for {year} R{round_number} {session_code}. Proceeding with fetch.")
        except Exception as e:
            logger.error(f"DB check failed for {year} R{round_number} {session_code}: {e}. Attempting fetch anyway.")
        # --- CHANGE END ---

        logger.info(f"Fetching: {year} R{round_number} ({event_name}) Session '{session_code}'...")
        session_or_status = fetch_single_session_with_retry(year, round_number, session_code)

        if session_or_status == RATE_LIMIT_FAILURE_STATUS:
            logger.error(f"Persistent rate limit hit for {year} R{round_number} {session_code}. Aborting event processing.")
            persist_rate_limit_hit = True
            break # Stop processing sessions for this event
        elif session_or_status is None:
             # Indicates session doesn't exist, is invalid, or load failed (e.g., future event, DataNotLoadedError)
             logger.info(f"Fetch/load failed or session not available for {year} R{round_number} {session_code}. Skipping save for this session.")
             time.sleep(1.0) # Small pause
             continue # Move to the next session code for this event
        else:
            # We have a valid session object (though some data might be missing, e.g. results)
            session = session_or_status
            logger.debug(f"Successfully fetched/loaded {year} R{round_number} {session_code}. Sleeping {DEFAULT_API_SLEEP_S}s...")
            time.sleep(DEFAULT_API_SLEEP_S) # Be nice to the API

            # --- Process all data types ---
            # Functions now handle internal errors and return empty DataFrames if needed
            results_df, drivers_df, teams_df = process_session_results(session, year, round_number, session_code)
            laps_df, pits_df = process_laps_and_pits(session, year, round_number, session_code)
            weather_df = process_weather_data(session, year, round_number, session_code)

            # --- Save all data types ---
            # Wrap saving in a try block to catch DB errors per session
            try:
                # Save Results (only if not empty)
                if not results_df.empty:
                    database.save_data(results_df, 'results')
                else:
                    logger.debug(f"Skipping save for 'results' as DataFrame is empty ({year} R{round_number} {session_code}).")

                # Save New Drivers (check against existing)
                if not drivers_df.empty:
                    existing_drivers_q = "SELECT Abbreviation FROM drivers"
                    existing_drivers_df = database.load_data(existing_drivers_q)
                    existing_list = [] if existing_drivers_df.empty else existing_drivers_df['Abbreviation'].tolist()
                    new_drivers = drivers_df[~drivers_df['Abbreviation'].isin(existing_list)].copy()
                    # Ensure schema before saving
                    db_schema = ['Abbreviation', 'DriverNumber', 'FullName', 'Nationality']; [new_drivers.setdefault(c, None) for c in db_schema if c not in new_drivers.columns]
                    if not new_drivers.empty:
                         logger.info(f"Saving {len(new_drivers)} new drivers from {year} R{round_number} {session_code}.")
                         database.save_data(new_drivers[db_schema], 'drivers')

                # Save New Teams (check against existing)
                if not teams_df.empty:
                    existing_teams_q = "SELECT TeamName FROM teams"
                    existing_teams_df = database.load_data(existing_teams_q)
                    existing_list = [] if existing_teams_df.empty else existing_teams_df['TeamName'].tolist()
                    new_teams = teams_df[~teams_df['TeamName'].isin(existing_list)].copy()
                    # Ensure schema before saving
                    db_schema = ['TeamName', 'Nationality']; [new_teams.setdefault(c, None) for c in db_schema if c not in new_teams.columns]
                    if not new_teams.empty:
                        logger.info(f"Saving {len(new_teams)} new teams from {year} R{round_number} {session_code}.")
                        database.save_data(new_teams[db_schema], 'teams')

                # Save Laps (only if not empty)
                if not laps_df.empty:
                    database.save_data(laps_df, 'laps')
                else:
                     logger.debug(f"Skipping save for 'laps' as DataFrame is empty ({year} R{round_number} {session_code}).")

                # Save Weather (only if not empty)
                if not weather_df.empty:
                    database.save_data(weather_df, 'weather')
                else:
                     logger.debug(f"Skipping save for 'weather' as DataFrame is empty ({year} R{round_number} {session_code}).")

                # Save Pit Stops (only if not empty)
                if not pits_df.empty:
                    database.save_data(pits_df, 'pit_stops')
                else:
                    logger.debug(f"Skipping save for 'pit_stops' as DataFrame is empty ({year} R{round_number} {session_code}).")

            except Exception as e:
                # Log DB errors, but don't crash the whole process unless critical
                logger.error(f"DB save error during processing for {year} R{round_number} {session_code}: {e}", exc_info=True)
                # Optionally, depending on severity, could set persist_rate_limit_hit = True here

    # Return False if rate limit was hit, True otherwise (even if some sessions failed individually)
    return not persist_rate_limit_hit


# --- Main Function to Update Database ---
def update_database(years_list):
    """Fetches, processes, and stores F1 data for the specified years."""
    logger.info(f"Starting database update for years: {years_list}")
    year_iterator = sorted(years_list)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    logger.info(f"Current UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    overall_stop_processing = False # Flag to stop processing subsequent years

    for year in year_iterator:
        if overall_stop_processing:
            logger.warning(f"Skipping year {year} due to persistent rate limit in a previous year.")
            continue

        logger.info(f"===== Processing Year: {year} =====")
        schedule = None; schedule_load_attempts = 0
        while schedule_load_attempts <= MAX_RATE_LIMIT_RETRIES:
            try:
                schedule = ff1.get_event_schedule(year, include_testing=False);
                if schedule is not None and not schedule.empty:
                    # Convert types early and handle NaNs
                    schedule['EventDate'] = pd.to_datetime(schedule['EventDate'], utc=True, errors='coerce')
                    schedule['RoundNumber'] = pd.to_numeric(schedule['RoundNumber'], errors='coerce')
                    schedule.dropna(subset=['RoundNumber'], inplace=True) # Drop events without a round number
                    schedule['RoundNumber'] = schedule['RoundNumber'].astype(int)
                    logger.info(f"Schedule loaded and pre-processed for {year} ({len(schedule)} events).")
                    break # Success
                else:
                    logger.warning(f"FastF1 returned empty or invalid schedule for {year}. Assuming no events.")
                    schedule = pd.DataFrame() # Ensure empty DataFrame for checks below
                    break
            except ff1.RateLimitExceededError as e:
                 schedule_load_attempts += 1
                 if schedule_load_attempts > MAX_RATE_LIMIT_RETRIES:
                      logger.error(f"RATE LIMIT EXCEEDED fetching schedule for {year} after {MAX_RATE_LIMIT_RETRIES} retries: {e}. Skipping year.")
                      schedule = None; overall_stop_processing = True; break # Stop processing this and subsequent years
                 wait = RATE_LIMIT_SLEEP_S * schedule_load_attempts
                 logger.warning(f"Rate limit on schedule ({schedule_load_attempts}/{MAX_RATE_LIMIT_RETRIES}). Sleeping {wait}s...")
                 time.sleep(wait)
                 logger.info("Retrying schedule fetch...")
            except ConnectionError as e:
                 logger.error(f"Connection error fetching schedule for {year}: {e}. Skipping year.")
                 schedule = None; break # Skip this year, but maybe next year works
            except Exception as e:
                 logger.error(f"Failed schedule fetch for {year}: {e}", exc_info=True)
                 schedule = None; break # Skip this year

        if schedule is None: continue # Skip year if fetch failed permanently
        if schedule.empty: logger.warning(f"No schedule events found for {year} after loading/cleaning. Skipping year."); continue

        # Determine which events have already passed for current/future years
        schedule_for_sessions = pd.DataFrame() # Initialize empty
        if year >= now_utc.year:
            initial_count = len(schedule)
            # Allow processing events that finished very recently or are today/tomorrow
            buffer_days = 1
            cutoff_date = now_utc + datetime.timedelta(days=buffer_days)
            # Filter based on EventDate (already datetime objects)
            schedule_filtered = schedule[(schedule['EventDate'] <= cutoff_date) | (schedule['EventDate'].isna())].copy()
            logger.info(f"Filtering {year} schedule: {initial_count} -> {len(schedule_filtered)} past or near-future/undated events relative to {cutoff_date.date()}.")
            if schedule_filtered.empty:
                 logger.info(f"No past/near-future/undated events found to process for {year} yet. Skipping year's session data fetch.")
            else:
                 schedule_for_sessions = schedule_filtered
        else:
            # For historical years, process all events
            schedule_for_sessions = schedule.copy()
            logger.info(f"Processing all {len(schedule_for_sessions)} events for historical year {year}.")


        # --- Save/Update Event Schedule Info into DB ---
        try:
            event_cols_db = ['Year', 'RoundNumber', 'EventName', 'Country', 'Location', 'OfficialEventName', 'EventDate']
            # Select available columns from loaded schedule
            avail_cols = [c for c in event_cols_db if c in schedule.columns and c != 'Year'] # Exclude Year temporarily
            schedule_to_save = schedule[avail_cols].copy()
            schedule_to_save['Year'] = int(year) # Add Year back

            # Ensure all DB columns exist, assign None if missing
            for col in event_cols_db:
                 if col not in schedule_to_save.columns: schedule_to_save[col] = None

            # Convert EventDate back to string for DB compatibility, handle NaT
            schedule_to_save['EventDate'] = schedule_to_save['EventDate'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').replace({pd.NaT: None})

            # Ensure correct types before DB interaction
            schedule_to_save['RoundNumber'] = pd.to_numeric(schedule_to_save['RoundNumber'], errors='coerce').fillna(-1).astype(int)
            schedule_to_save['Year'] = schedule_to_save['Year'].astype(int)
            # Ensure all other text columns are strings
            for col in ['EventName', 'Country', 'Location', 'OfficialEventName']:
                 if col in schedule_to_save.columns: schedule_to_save[col] = schedule_to_save[col].astype(str).fillna('')

            # Select columns in DB order
            schedule_to_save = schedule_to_save[event_cols_db]

            # Check existing events to avoid duplicates
            existing_q = "SELECT Year, RoundNumber FROM events WHERE Year = ?"
            existing_df = database.load_data(existing_q, params=(year,))

            if not existing_df.empty:
                existing_pairs = set(zip(existing_df['Year'].astype(int), existing_df['RoundNumber'].astype(int)))
                # Filter out rows that already exist in the DB based on Year/RoundNumber primary key
                filtered_new = schedule_to_save[~schedule_to_save.apply(lambda r: (r['Year'], r['RoundNumber']) in existing_pairs, axis=1)].copy()
            else:
                filtered_new = schedule_to_save.copy()

            # Ensure no invalid RoundNumbers (-1) are saved
            filtered_new = filtered_new[filtered_new['RoundNumber'] != -1]

            if not filtered_new.empty:
                logger.info(f"Adding {len(filtered_new)} new events to DB for {year}.")
                database.save_data(filtered_new, 'events')
            else:
                logger.info(f"No new events to add to DB for {year}.")
        except Exception as e:
            logger.error(f"Error saving schedule to DB for {year}: {e}", exc_info=True)


        # --- Fetch Session Data for Each Relevant Event ---
        if schedule_for_sessions.empty:
            logger.info(f"Skipping session data fetching for {year} as no relevant events were found.")
            continue

        logger.info(f"Fetching session data for {len(schedule_for_sessions)} relevant events in {year}...")
        # Sort events by round number before iterating
        schedule_for_sessions.sort_values(by='RoundNumber', inplace=True)
        event_iterator = tqdm(schedule_for_sessions.iterrows(), total=len(schedule_for_sessions), desc=f"Events {year}", unit="event", leave=False)

        for index, event_row in event_iterator:
              # Wrap fetch_and_store_event in try/except to catch unexpected errors within event processing
              try:
                   success = fetch_and_store_event(year, event_row)
                   if not success:
                       logger.error(f"Stopping processing for year {year} due to persistent rate limit reported by fetch_and_store_event.")
                       overall_stop_processing = True # Signal to stop subsequent years
                       break # Stop processing events for this year
              except Exception as inner_e:
                   # Log the error for the specific event, but *continue* to the next event
                   # This prevents one bad event from stopping the whole year (unless it's a rate limit)
                   rn = event_row.get('RoundNumber', 'N/A')
                   logger.error(f"Unhandled error during event processing loop for {year} R{rn}: {inner_e}", exc_info=True)
                   logger.warning(f"Continuing to next event despite error in R{rn}.")

        if overall_stop_processing:
            logger.warning(f"Aborted processing remaining events for year {year} due to rate limit.")
            # No need to break again, outer loop check handles this

    logger.info("===== Database update process finished. =====")