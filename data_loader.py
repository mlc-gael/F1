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
SESSION_IDENTIFIERS_TO_FETCH = ['R', 'Q', 'S', 'SQ', 'SS', 'FP1', 'FP2', 'FP3']
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
    # (Keep this function exactly as it was in the previous corrected version)
    if session is None or not hasattr(session, 'results') or session.results is None or session.results.empty:
        logger.warning(f"No session.results data found for {year} R{round_number} Session {session_name}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        results = session.results.copy(); logger.debug(f"Processing results for {year} R{round_number} {session_name}. Input columns: {results.columns.tolist()}")
        results['Year'] = int(year); results['RoundNumber'] = int(round_number); results['SessionName'] = str(session_name)
        source_column_map = { 'DriverNumber': 'DriverNumber', 'Abbreviation': 'Abbreviation', 'TeamName': 'TeamName', 'GridPosition': 'GridPosition', 'Position': 'Position', 'Points': 'Points', 'Status': 'Status', 'Laps': 'Laps', 'FastestLapTime': 'FastestLapTime', 'Q1': 'Q1', 'Q2': 'Q2', 'Q3': 'Q3', 'FullName': 'FullName' }
        results_df = results[[col for col in source_column_map if col in results.columns]].rename(columns=source_column_map)
        for col in ['DriverNumber', 'Abbreviation', 'TeamName']:
            if col not in results_df.columns: results_df[col] = None
            results_df[col] = results_df[col].astype(str).fillna('UNK')
        results_df['GridPosition'] = utils.safe_to_numeric(results_df.get('GridPosition'), fallback=float(config.WORST_EXPECTED_POS))
        results_df['Position'] = utils.safe_to_numeric(results_df.get('Position'), fallback=None) # Keep NaN here initially
        results_df['Points'] = utils.safe_to_numeric(results_df.get('Points'), fallback=0.0)
        results_df['Laps'] = utils.safe_to_numeric(results_df.get('Laps'), fallback=0.0)
        for time_col in ['FastestLapTime', 'Q1', 'Q2', 'Q3']:
            if time_col in results_df.columns: results_df[time_col] = results_df[time_col].apply(parse_lap_time).astype(float)
            else: results_df[time_col] = np.nan # Ensure column exists if expected
        if 'Status' in results_df.columns: results_df['Status'] = results_df['Status'].astype(str).fillna('Unknown')
        else: results_df['Status'] = 'Unknown'
        if 'FullName' in results_df.columns: results_df['FullName'] = results_df['FullName'].astype(str).fillna('')
        else: results_df['FullName'] = ''

        driver_info_cols = ['DriverNumber', 'Abbreviation', 'TeamName', 'FullName']
        drivers_df = results_df.dropna(subset=['Abbreviation']).loc[:, [c for c in driver_info_cols if c in results_df.columns]].drop_duplicates(subset=['Abbreviation']).copy()
        if not drivers_df.empty: drivers_df['Nationality'] = None
        teams_df = results_df.dropna(subset=['TeamName']).loc[:, ['TeamName']].drop_duplicates().copy()
        if not teams_df.empty: teams_df['Nationality'] = None

        # Ensure all DB schema columns exist before selection
        db_results_schema = ['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Points', 'Status', 'Laps', 'FastestLapTime', 'Q1', 'Q2', 'Q3']
        for col in db_results_schema:
             if col not in results_df.columns:
                  # Default based on expected type
                  if col in ['GridPosition', 'Position', 'Points', 'Laps', 'FastestLapTime', 'Q1', 'Q2', 'Q3']: results_df[col] = np.nan
                  elif col in ['DriverNumber', 'Abbreviation', 'TeamName', 'Status', 'SessionName']: results_df[col] = 'UNK' # Or None/np.nan if preferred
                  elif col in ['Year', 'RoundNumber']: results_df[col] = -1 # Should not happen due to earlier assignment
                  else: results_df[col] = None
        results_df = results_df[db_results_schema] # Select only desired columns

        # --- Corrected NaN Handling for Essential IDs ---
        essential_ids = ['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'Abbreviation']
        # Check for NaNs *before* dropping, log warning, then drop. No filling here.
        nan_check_before_drop = results_df[essential_ids].isnull().sum()
        if nan_check_before_drop.sum() > 0:
            logger.warning(f"NaNs found in essential identifiers BEFORE dropna for {year} R{round_number} {session_name}: \n{nan_check_before_drop[nan_check_before_drop > 0]}")

        initial_rows = len(results_df)
        results_df = results_df.dropna(subset=essential_ids)
        rows_dropped = initial_rows - len(results_df)
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} rows from results due to missing essential identifiers for {year} R{round_number} {session_name}.")
        # --- End Correction ---

        # Ensure remaining required schemas for drivers/teams
        db_drivers_schema = ['Abbreviation', 'DriverNumber', 'FullName', 'Nationality']; drivers_df = drivers_df[[col for col in db_drivers_schema if col in drivers_df.columns]]
        db_teams_schema = ['TeamName', 'Nationality']; teams_df = teams_df[[col for col in db_teams_schema if col in teams_df.columns]]

        logger.debug(f"Finished processing results for {year} R{round_number} {session_name}. Final shape: {results_df.shape}")
        return results_df, drivers_df, teams_df

    except Exception as e:
        logger.error(f"CRITICAL ERROR during process_session_results for {year} R{round_number} {session_name}: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def process_laps_and_pits(session, year, round_number, session_name):
    """Processes lap times and pit stops into DataFrames."""
    # (Keep this function exactly as it was in the previous corrected version)
    laps_df_final = pd.DataFrame()
    pits_df_final = pd.DataFrame()

    if session is None or not hasattr(session, 'laps') or session.laps is None or session.laps.empty:
        logger.warning(f"No lap data found for {year} R{round_number} {session_name}")
        return laps_df_final, pits_df_final

    try:
        laps = session.laps.copy()
        logger.debug(f"Processing {len(laps)} laps for {year} R{round_number} {session_name}")
        laps['Year'] = int(year); laps['RoundNumber'] = int(round_number); laps['SessionName'] = str(session_name)

        lap_cols_to_keep = ['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'LapNumber', 'LapTime', 'Stint', 'TyreLife', 'Compound', 'IsAccurate', 'IsPitOutLap', 'IsPitInLap', 'Sector1Time', 'Sector2Time', 'Sector3Time']
        laps_df_final = laps[[col for col in lap_cols_to_keep if col in laps.columns]].copy()

        # Ensure all expected DB columns exist, adding NaN/defaults if missing
        lap_cols_db = ['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'LapNumber', 'LapTime', 'Stint', 'TyreLife', 'Compound', 'IsAccurate', 'IsPitOutLap', 'IsPitInLap', 'Sector1Time', 'Sector2Time', 'Sector3Time']
        for col in lap_cols_db:
             if col not in laps_df_final.columns:
                 if col in ['LapTime', 'TyreLife', 'Sector1Time', 'Sector2Time', 'Sector3Time']: laps_df_final[col] = np.nan
                 elif col in ['IsAccurate', 'IsPitOutLap', 'IsPitInLap']: laps_df_final[col] = 0 # Default bools to 0
                 elif col in ['Stint', 'LapNumber']: laps_df_final[col] = -1 # Default ints to -1
                 elif col == 'Compound': laps_df_final[col] = 'UNKNOWN'
                 # Year, RoundNumber, SessionName, DriverNumber should always exist from above

        # Convert types
        for time_col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
             laps_df_final[time_col] = laps_df_final[time_col].apply(parse_lap_time).astype(float)
        for bool_col in ['IsAccurate', 'IsPitOutLap', 'IsPitInLap']:
             laps_df_final[bool_col] = laps_df_final[bool_col].fillna(0).astype(bool).astype(int) # Fill potential NaNs before converting
        for int_col in ['Stint', 'LapNumber']:
            # Ensure conversion happens *after* potential addition if column was missing
            laps_df_final[int_col] = utils.safe_to_numeric(laps_df_final[int_col], fallback=-1).astype(int)
        laps_df_final['TyreLife'] = utils.safe_to_numeric(laps_df_final['TyreLife'], fallback=np.nan).astype(float)
        laps_df_final['Compound'] = laps_df_final['Compound'].astype(str).fillna('UNKNOWN')

        # --- Extract Pit Stops ---
        pits_df_final = pd.DataFrame() # Ensure it's initialized
        pit_cols_db = ['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'StopNumber', 'LapNumber', 'PitDuration']

        # Use get_pit_log if available (preferable)
        if hasattr(session, 'get_pit_log') and callable(session.get_pit_log):
             try:
                 pit_log = session.get_pit_log()
                 if not pit_log.empty:
                     pit_log_cols_needed = ['DriverNumber', 'LapNumber', 'StopDuration']
                     if all(col in pit_log.columns for col in pit_log_cols_needed):
                        pits_df_tmp = pit_log[pit_log_cols_needed].copy()
                        pits_df_tmp.rename(columns={'StopDuration': 'PitDuration'}, inplace=True)
                        pits_df_tmp['Year'] = int(year); pits_df_tmp['RoundNumber'] = int(round_number); pits_df_tmp['SessionName'] = str(session_name)
                        pits_df_tmp['DriverNumber'] = pits_df_tmp['DriverNumber'].astype(str) # Ensure consistent type
                        pits_df_tmp.sort_values(by=['DriverNumber', 'LapNumber'], inplace=True) # Sort before cumcount
                        pits_df_tmp['StopNumber'] = pits_df_tmp.groupby('DriverNumber').cumcount() + 1
                        pits_df_tmp['PitDuration'] = pits_df_tmp['PitDuration'].apply(parse_lap_time).astype(float)
                        pits_df_tmp['LapNumber'] = utils.safe_to_numeric(pits_df_tmp['LapNumber'], fallback=-1).astype(int)
                        # Ensure all columns exist before final selection
                        for col in pit_cols_db:
                            if col not in pits_df_tmp.columns: pits_df_tmp[col] = np.nan # Add missing cols as NaN
                        pits_df_final = pits_df_tmp[pit_cols_db] # Select final schema
                        logger.debug(f"Extracted {len(pits_df_final)} pits using get_pit_log.")
                     else:
                          logger.warning(f"Pit log available but missing expected columns for {year} R{round_number} {session_name}. Columns: {pit_log.columns}. Skipping pit log.")
                 else:
                      logger.debug(f"get_pit_log() returned empty DataFrame for {year} R{round_number} {session_name}")
             except Exception as pit_log_err:
                  logger.error(f"Error processing get_pit_log() for {year} R{round_number} {session_name}: {pit_log_err}", exc_info=True)

        # Fallback (less reliable duration) only if PitInLap exists and pit_log failed/unavailable
        if pits_df_final.empty and 'IsPitInLap' in laps_df_final.columns:
            logger.debug(f"Falling back to IsPitInLap for pit stops for {year} R{round_number} {session_name}")
            pits_in = laps_df_final[laps_df_final['IsPitInLap'] == 1]
            if not pits_in.empty:
                pits_df_tmp = pits_in[['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'LapNumber']].copy()
                pits_df_tmp['DriverNumber'] = pits_df_tmp['DriverNumber'].astype(str)
                pits_df_tmp.sort_values(by=['DriverNumber', 'LapNumber'], inplace=True)
                pits_df_tmp['StopNumber'] = pits_df_tmp.groupby('DriverNumber').cumcount() + 1
                pits_df_tmp['PitDuration'] = np.nan # Duration unknown
                # Ensure all columns exist
                for col in pit_cols_db:
                     if col not in pits_df_tmp.columns: pits_df_tmp[col] = np.nan # Add missing cols
                pits_df_final = pits_df_tmp[pit_cols_db]
                logger.debug(f"Extracted {len(pits_df_final)} potential pits using IsPitInLap (duration unavailable).")

        if pits_df_final.empty:
             logger.warning(f"Could not determine pit stops for {year} R{round_number} {session_name}")


        logger.debug(f"Processed laps ({len(laps_df_final)}) and pits ({len(pits_df_final)}) for {year} R{round_number} {session_name}")

    except Exception as e:
        logger.error(f"Error processing laps/pits for {year} R{round_number} {session_name}: {e}", exc_info=True)
        laps_df_final, pits_df_final = pd.DataFrame(), pd.DataFrame() # Return empty on error

    # Ensure consistent dtypes before returning
    if not laps_df_final.empty:
        # Re-apply dtypes after potential column additions/fills
         for col in ['Year', 'RoundNumber']: laps_df_final[col]=laps_df_final[col].astype(int)
         for col in ['SessionName','DriverNumber','Compound']: laps_df_final[col]=laps_df_final[col].astype(str)
         for col in ['LapNumber','Stint','IsAccurate','IsPitOutLap','IsPitInLap']: laps_df_final[col]=laps_df_final[col].astype(int)
         for col in ['LapTime','TyreLife','Sector1Time','Sector2Time','Sector3Time']: laps_df_final[col]=laps_df_final[col].astype(float)

    if not pits_df_final.empty:
         for col in ['Year', 'RoundNumber', 'StopNumber', 'LapNumber']: pits_df_final[col]=utils.safe_to_numeric(pits_df_final[col], fallback=-1).astype(int)
         for col in ['SessionName','DriverNumber']: pits_df_final[col]=pits_df_final[col].astype(str)
         pits_df_final['PitDuration']=pits_df_final['PitDuration'].astype(float)


    return laps_df_final, pits_df_final


def process_weather_data(session, year, round_number, session_name):
    # (Keep this function exactly as it was in the previous corrected version)
    weather_df_final = pd.DataFrame()
    if session is None or not hasattr(session, 'weather_data') or session.weather_data is None or session.weather_data.empty:
        logger.warning(f"No weather data found for {year} R{round_number} {session_name}")
        return weather_df_final
    try:
        weather = session.weather_data.copy(); logger.debug(f"Processing {len(weather)} weather entries for {year} R{round_number} {session_name}")
        weather['Year'] = int(year); weather['RoundNumber'] = int(round_number); weather['SessionName'] = str(session_name)

        # Handle Time column - Convert Timedelta to seconds
        if 'Time' in weather.columns:
             weather['TimeSeconds'] = weather['Time'].apply(parse_lap_time).astype(float)
        else:
             logger.warning("Weather data missing 'Time' column. Cannot create TimeSeconds."); weather['TimeSeconds'] = np.nan

        weather_cols_to_keep = ['Year', 'RoundNumber', 'SessionName', 'TimeSeconds', 'AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'WindDirection', 'Rainfall']
        weather_df_final = weather[[col for col in weather_cols_to_keep if col in weather.columns]].copy()
        weather_df_final.rename(columns={'TimeSeconds': 'Time'}, inplace=True) # Rename to match DB schema 'Time'

        # Ensure all DB schema columns exist and have correct types
        weather_cols_db = ['Year', 'RoundNumber', 'SessionName', 'Time', 'AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'WindDirection', 'Rainfall']
        for col in weather_cols_db:
            if col not in weather_df_final.columns:
                if col in ['Time', 'AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'WindDirection']: weather_df_final[col] = np.nan
                elif col == 'Rainfall': weather_df_final[col] = 0 # Default bool to 0
                # Year, RoundNumber, SessionName should exist
            # Convert types AFTER ensuring column exists
            if col == 'Rainfall': weather_df_final[col] = weather_df_final[col].fillna(0).astype(bool).astype(int)
            elif col == 'WindDirection': weather_df_final[col] = utils.safe_to_numeric(weather_df_final[col], fallback=np.nan).astype(float) # Allow float for direction
            elif col not in ['Year', 'RoundNumber', 'SessionName']: # Numeric float columns
                  weather_df_final[col] = utils.safe_to_numeric(weather_df_final[col], fallback=np.nan).astype(float)

        # Drop rows where Time is NaN as it's part of the primary key
        initial_rows = len(weather_df_final)
        weather_df_final.dropna(subset=['Time'], inplace=True)
        rows_dropped = initial_rows - len(weather_df_final)
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} weather rows due to missing Time for {year} R{round_number} {session_name}")

        logger.debug(f"Processed weather data for {year} R{round_number} {session_name}. Final shape: {weather_df_final.shape}")
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
            load_kwargs = {'laps': config.LOAD_LAPS, 'weather': config.LOAD_WEATHER, 'messages': False, 'telemetry': False}
            logger.debug(f"Attempting load: {year} R{round_number} {session_code} with {load_kwargs}")
            session.load(**load_kwargs)
            logger.debug(f"Session loaded: {year} R{round_number} {session_code}")
            return session # Success

        except ff1.RateLimitExceededError as e: # Specific catch
            retries += 1
            if retries > MAX_RATE_LIMIT_RETRIES:
                logger.error(f"RATE LIMIT EXCEEDED after {MAX_RATE_LIMIT_RETRIES} retries for {year} R{round_number} {session_code}: {e}. Giving up on this session.")
                return RATE_LIMIT_FAILURE_STATUS
            wait = RATE_LIMIT_SLEEP_S * retries
            logger.warning(f"Rate limit hit ({retries}/{MAX_RATE_LIMIT_RETRIES}) for {year} R{round_number} {session_code}. Sleeping {wait}s...")
            time.sleep(wait)
            logger.info(f"Retrying fetch {retries}...")

        except ValueError as e: # Catch broader ValueError
             error_msg_lower = str(e).lower()
             # Check for specific non-existence errors
             session_not_exist_patterns = [
                 "does not exist for this event",
                 "no event found", # Less likely here, more in get_event_schedule
                 "session identifier", # Can indicate wrong code like 'FP4'
                 "session type" # Explicitly mentioned in some logs
             ]
             # --- MODIFICATION START: Log expected non-existence at DEBUG ---
             is_non_existent_error = any(p in error_msg_lower for p in session_not_exist_patterns)

             if is_non_existent_error:
                  # Log at DEBUG level instead of WARNING for less noise
                  logger.debug(f"Session '{session_code}' not available or invalid for {year} R{round_number}. Skipping. (Reason: {e})")
                  return None # Session genuinely doesn't exist or invalid request
             # --- MODIFICATION END ---
             elif "failed to load any schedule data" in error_msg_lower:
                  logger.error(f"Failed to load schedule data during get_session for {year} R{round_number} {session_code} (likely due to earlier rate limit/network issue). Skipping session.")
                  return None
             else: # Other unexpected ValueErrors
                  logger.error(f"Unexpected ValueError loading {year} R{round_number} {session_code}: {e}", exc_info=True)
                  return None # Treat as failure for this session

        except ConnectionError as e:
            logger.error(f"Connection error fetching {year} R{round_number} {session_code}: {e}. Skipping session.")
            time.sleep(5) # Brief pause before potentially trying next session
            return None
        except ff1.exceptions.FastF1Error as e: # Catch generic FastF1 errors not caught above
            logger.error(f"FastF1 API error loading {year} R{round_number} {session_code}: {e}", exc_info=False) # exc_info=False to reduce noise for known API issues
            return None # Treat as failure for this session
        except Exception as e:
            logger.error(f"Generic unexpected error loading {year} R{round_number} {session_code}: {e}", exc_info=True)
            return None # Treat as failure for this session

    # This point should only be reached if the loop finishes due to retries > MAX_RATE_LIMIT_RETRIES
    logger.error(f"Exceeded max rate limit retries ({MAX_RATE_LIMIT_RETRIES}) for {year} R{round_number} {session_code}.")
    return RATE_LIMIT_FAILURE_STATUS # Should have been returned inside loop, but as safety


def fetch_and_store_event(year, event_row):
    """
    Fetches, processes, and stores data for relevant sessions of a single event.
    Returns True on success (all sessions attempted),
    Returns False if a persistent rate limit failure occurred for any session.
    """
    # (Keep this function exactly as it was in the previous corrected version,
    # including the fix for saving pit_stops_final)
    try:
        round_number = int(event_row['RoundNumber'])
        event_location = event_row.get('Location', f'UNK R{round_number}')
        event_name = event_row.get('EventName', f'UNK R{round_number}')
        event_date_str = event_row.get('EventDate') # Keep as string for now
        logger.info(f"--- Processing Event: {year} R{round_number} - {event_name} ({event_location}) ---")
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid event data: {e}. Row: {event_row}")
        return True # Continue processing other events

    session_iterator = SESSION_IDENTIFIERS_TO_FETCH
    persist_rate_limit_hit = False # Flag for this event

    for session_code in session_iterator:
        logger.debug(f"Checking data: {year} R{round_number} {session_code}")
        try:
            if database.check_data_exists(year, round_number, session_code):
                logger.info(f"Data exists: {year} R{round_number} {session_code}. Skipping fetch.")
                continue
        except Exception as e:
            logger.error(f"DB check failed for {year} R{round_number} {session_code}: {e}. Attempting fetch anyway.")

        logger.info(f"Fetching: {year} R{round_number} ({event_name}) Session '{session_code}'...")
        session_or_status = fetch_single_session_with_retry(year, round_number, session_code)

        if session_or_status == RATE_LIMIT_FAILURE_STATUS:
            logger.error(f"Persistent rate limit hit for {year} R{round_number} {session_code}. Aborting event processing.")
            persist_rate_limit_hit = True
            break # Stop processing sessions for THIS event
        elif session_or_status is None:
             # This message will now appear less often due to the change in fetch_single_session_with_retry
             logger.debug(f"Failed fetch/load or session not found for {year} R{round_number} {session_code}. Skipping save for this session.")
             time.sleep(1.0) # Still sleep briefly
             continue # Move to the next session for this event
        else:
            # We have a valid session object
            session = session_or_status
            logger.debug(f"Successfully fetched/loaded {year} R{round_number} {session_code}. Sleeping {DEFAULT_API_SLEEP_S}s...")
            time.sleep(DEFAULT_API_SLEEP_S)

            # Process all data types
            results_df, drivers_df, teams_df = process_session_results(session, year, round_number, session_code)
            laps_df, pits_df_final = process_laps_and_pits(session, year, round_number, session_code)
            weather_df = process_weather_data(session, year, round_number, session_code)

            # Save all data types
            try:
                if not results_df.empty: database.save_data(results_df, 'results')
                if not drivers_df.empty:
                    # Check existing drivers more efficiently
                    existing_drivers_q = "SELECT Abbreviation FROM drivers"
                    existing_drivers_df = database.load_data(existing_drivers_q)
                    existing_list = [] if existing_drivers_df.empty else existing_drivers_df['Abbreviation'].tolist()
                    new_drivers = drivers_df[~drivers_df['Abbreviation'].isin(existing_list)].copy()
                    db_schema = ['Abbreviation', 'DriverNumber', 'FullName', 'Nationality']; [new_drivers.setdefault(c, None) for c in db_schema if c not in new_drivers.columns]
                    if not new_drivers.empty:
                         logger.info(f"Saving {len(new_drivers)} new drivers from {year} R{round_number} {session_code}.")
                         database.save_data(new_drivers[db_schema], 'drivers')

                if not teams_df.empty:
                    existing_teams_q = "SELECT TeamName FROM teams"
                    existing_teams_df = database.load_data(existing_teams_q)
                    existing_list = [] if existing_teams_df.empty else existing_teams_df['TeamName'].tolist()
                    new_teams = teams_df[~teams_df['TeamName'].isin(existing_list)].copy()
                    db_schema = ['TeamName', 'Nationality']; [new_teams.setdefault(c, None) for c in db_schema if c not in new_teams.columns]
                    if not new_teams.empty:
                        logger.info(f"Saving {len(new_teams)} new teams from {year} R{round_number} {session_code}.")
                        database.save_data(new_teams[db_schema], 'teams')

                if not laps_df.empty: database.save_data(laps_df, 'laps')
                if not weather_df.empty: database.save_data(weather_df, 'weather')
                # Corrected variable name check here
                if not pits_df_final.empty:
                     database.save_data(pits_df_final, 'pit_stops')
            except Exception as e:
                logger.error(f"DB save error for {year} R{round_number} {session_code}: {e}", exc_info=True)

    # Return False if rate limit was hit, True otherwise
    return not persist_rate_limit_hit


# --- Main Function to Update Database ---
def update_database(years_list):
    """Fetches, processes, and stores F1 data for the specified years."""
    # (Keep this function exactly as it was in the previous corrected version,
    # including the fix for saving schedule info)
    logger.info(f"Starting database update for years: {years_list}")
    year_iterator = sorted(years_list)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    logger.info(f"Current UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    for year in year_iterator:
        logger.info(f"===== Processing Year: {year} =====")
        schedule = None; schedule_load_attempts = 0
        while schedule_load_attempts <= MAX_RATE_LIMIT_RETRIES:
            try:
                schedule = ff1.get_event_schedule(year, include_testing=False);
                if schedule is not None and not schedule.empty:
                    logger.debug(f"Loaded schedule for {year} with {len(schedule)} events.")
                    schedule['EventDate_dt'] = pd.to_datetime(schedule['EventDate'], utc=True, errors='coerce')
                    schedule['RoundNumber'] = pd.to_numeric(schedule['RoundNumber'], errors='coerce')
                    schedule.dropna(subset=['RoundNumber'], inplace=True)
                    schedule['RoundNumber'] = schedule['RoundNumber'].astype(int)
                    logger.info(f"Schedule loaded and pre-processed for {year}.")
                    break
                else:
                    logger.warning(f"FastF1 returned empty schedule for {year}. Assuming no events.")
                    schedule = pd.DataFrame()
                    break
            except ff1.RateLimitExceededError as e:
                 schedule_load_attempts += 1
                 if schedule_load_attempts > MAX_RATE_LIMIT_RETRIES:
                      logger.error(f"RATE LIMIT EXCEEDED fetching schedule for {year} after {MAX_RATE_LIMIT_RETRIES} retries: {e}. Skipping year.")
                      schedule = None; break
                 wait = RATE_LIMIT_SLEEP_S * schedule_load_attempts
                 logger.warning(f"Rate limit on schedule ({schedule_load_attempts}/{MAX_RATE_LIMIT_RETRIES}). Sleeping {wait}s...")
                 time.sleep(wait)
                 logger.info("Retrying schedule fetch...")
            except ConnectionError as e:
                 logger.error(f"Connection error fetching schedule for {year}: {e}. Skipping year.")
                 schedule = None; break
            except Exception as e:
                 logger.error(f"Failed schedule fetch for {year}: {e}", exc_info=True)
                 schedule = None; break

        if schedule is None: continue
        if schedule.empty: logger.warning(f"No schedule events found for {year} after loading/cleaning. Skipping year."); continue

        # Determine which events have already passed for current/future years
        if year >= now_utc.year:
            initial_count = len(schedule)
            schedule_filtered = schedule[(schedule['EventDate_dt'] <= now_utc) | (schedule['EventDate_dt'].isna())].copy()
            logger.info(f"Filtering {year} schedule: {initial_count} -> {len(schedule_filtered)} past or undated events relative to {now_utc.date()}.")
            if schedule_filtered.empty:
                 logger.info(f"No past/undated events found to process for {year} yet. Skipping year's session data fetch.")
                 schedule_for_sessions = pd.DataFrame()
            else:
                 schedule_for_sessions = schedule_filtered
        else:
            schedule_for_sessions = schedule.copy()
            logger.info(f"Processing all {len(schedule_for_sessions)} events for historical year {year}.")


        # Save/Update Event Schedule Info (using the original full schedule)
        try:
            event_cols = ['RoundNumber', 'EventName', 'Country', 'Location', 'OfficialEventName', 'EventDate']
            avail_cols = [c for c in event_cols if c in schedule.columns]
            schedule_to_save = schedule[avail_cols].copy()
            schedule_to_save['Year'] = int(year)

            db_cols = ['Year', 'RoundNumber', 'EventName', 'Country', 'Location', 'OfficialEventName', 'EventDate']
            for col in db_cols:
                 if col not in schedule_to_save.columns:
                     schedule_to_save[col] = None

            schedule_to_save['EventDate'] = schedule_to_save['EventDate'].astype(str).replace('NaT', None)
            schedule_to_save['RoundNumber'] = pd.to_numeric(schedule_to_save['RoundNumber'], errors='coerce').fillna(-1).astype(int)
            schedule_to_save['Year'] = schedule_to_save['Year'].astype(int)
            schedule_to_save = schedule_to_save[db_cols]

            existing_q = "SELECT Year, RoundNumber FROM events WHERE Year = ?"
            existing = database.load_data(existing_q, params=(year,))

            if not existing.empty:
                existing_pairs = set(zip(existing['Year'].astype(int), existing['RoundNumber'].astype(int)))
                filtered_new = schedule_to_save[~schedule_to_save.apply(lambda r: (r['Year'], r['RoundNumber']) in existing_pairs, axis=1)].copy()
            else:
                filtered_new = schedule_to_save.copy()

            filtered_new = filtered_new[filtered_new['RoundNumber'] != -1]

            if not filtered_new.empty:
                logger.info(f"Adding {len(filtered_new)} new events to DB for {year}.")
                database.save_data(filtered_new, 'events')
            else:
                logger.info(f"No new events to add to DB for {year}.")
        except Exception as e:
            logger.error(f"Error saving schedule for {year}: {e}", exc_info=True)


        # Fetch Session Data for Each Event (using schedule_for_sessions)
        if schedule_for_sessions.empty:
            logger.info(f"Skipping session data fetching for {year} as no relevant events were found.")
            continue

        logger.info(f"Fetching session data for {len(schedule_for_sessions)} relevant events in {year}...")
        schedule_for_sessions.sort_values(by='RoundNumber', inplace=True)
        event_iterator = tqdm(schedule_for_sessions.iterrows(), total=len(schedule_for_sessions), desc=f"Events {year}", unit="event", leave=False)
        stop_year_processing = False

        for index, event_row in event_iterator:
              try:
                   success_or_rate_limit_stop = fetch_and_store_event(year, event_row)
                   if not success_or_rate_limit_stop:
                       logger.error(f"Stopping processing for year {year} due to persistent rate limit reported by fetch_and_store_event.")
                       stop_year_processing = True
                       break
              except Exception as inner_e:
                   logger.error(f"Unhandled error during event processing loop for R{event_row.get('RoundNumber', 'N/A')}: {inner_e}", exc_info=True)

        if stop_year_processing:
            logger.warning(f"Aborted processing remaining events for year {year}.")
            logger.error(f"Stopping all further processing due to persistent rate limit in year {year}.")
            break

    logger.info("===== Database update process finished. =====")