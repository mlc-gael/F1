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
DEFAULT_API_SLEEP_S = 4.0 # Increased sleep time
RATE_LIMIT_SLEEP_S = 60.0
MAX_RATE_LIMIT_RETRIES = 2
SESSION_IDENTIFIERS_TO_FETCH = ['R', 'Q', 'S', 'SQ', 'SS', 'FP1', 'FP2', 'FP3']


# Configure FastF1 Cache
try:
    if config.CACHE_DIR and not os.path.exists(config.CACHE_DIR): os.makedirs(config.CACHE_DIR, exist_ok=True)
    ff1.Cache.enable_cache(config.CACHE_DIR)
    logger.info(f"FastF1 cache enabled at: {config.CACHE_DIR}")
except Exception as e:
    logger.error(f"CRITICAL: Failed to enable FastF1 cache: {e}", exc_info=True)


# --- Helper Functions ---
def parse_lap_time(time_obj): # Renamed from previous
    """Converts Pandas Timedelta or numeric to seconds (float) or returns np.nan."""
    return utils.parse_timedelta_to_seconds(time_obj) # Use function from utils


# --- Data Processing Functions ---

def process_session_results(session, year, round_number, session_name):
    """Processes results into DataFrames matching DB schema (drivers, teams, results)."""
    if session is None or not hasattr(session, 'results') or session.results is None or session.results.empty:
        logger.warning(f"No session.results data found for {year} R{round_number} {session_name}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        results = session.results.copy()
        logger.debug(f"Processing results for {year} R{round_number} {session_name}. Input columns: {results.columns.tolist()}")

        results['Year'] = int(year)
        results['RoundNumber'] = int(round_number)
        results['SessionName'] = str(session_name)

        source_column_map = { 'DriverNumber': 'DriverNumber', 'Abbreviation': 'Abbreviation', 'TeamName': 'TeamName', 'GridPosition': 'GridPosition', 'Position': 'Position', 'Points': 'Points', 'Status': 'Status', 'Laps': 'Laps', 'FastestLapTime': 'FastestLapTime', 'Q1': 'Q1', 'Q2': 'Q2', 'Q3': 'Q3', 'FullName': 'FullName' }
        results_df = results[[col for col in source_column_map if col in results.columns]].rename(columns=source_column_map)

        for col in ['DriverNumber', 'Abbreviation', 'TeamName']:
            if col not in results_df.columns: results_df[col] = None
            results_df[col] = results_df[col].astype(str).fillna('UNK')

        results_df['GridPosition'] = utils.safe_to_numeric(results_df.get('GridPosition'), fallback=float(config.WORST_EXPECTED_POS))
        results_df['Position'] = utils.safe_to_numeric(results_df.get('Position'), fallback=None)
        results_df['Points'] = utils.safe_to_numeric(results_df.get('Points'), fallback=0.0)
        results_df['Laps'] = utils.safe_to_numeric(results_df.get('Laps'), fallback=0.0)

        for time_col in ['FastestLapTime', 'Q1', 'Q2', 'Q3']:
            if time_col in results_df.columns:
                results_df[time_col] = results_df[time_col].apply(parse_lap_time).astype(float)

        if 'Status' in results_df.columns: results_df['Status'] = results_df['Status'].astype(str).fillna('Unknown')
        else: results_df['Status'] = 'Unknown'
        if 'FullName' in results_df.columns: results_df['FullName'] = results_df['FullName'].astype(str).fillna('')
        else: results_df['FullName'] = ''

        driver_info_cols = ['DriverNumber', 'Abbreviation', 'TeamName', 'FullName']
        drivers_df = results_df.dropna(subset=['Abbreviation']).loc[:, [c for c in driver_info_cols if c in results_df.columns]].drop_duplicates(subset=['Abbreviation']).copy()
        if not drivers_df.empty: drivers_df['Nationality'] = None

        teams_df = results_df.dropna(subset=['TeamName']).loc[:, ['TeamName']].drop_duplicates().copy()
        if not teams_df.empty: teams_df['Nationality'] = None

        results_df['Year'] = int(year)
        results_df['RoundNumber'] = int(round_number)
        results_df['SessionName'] = str(session_name)

        db_results_schema = ['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Points', 'Status', 'Laps', 'FastestLapTime', 'Q1', 'Q2', 'Q3']
        for col in db_results_schema:
             if col not in results_df.columns:
                  if col in ['GridPosition', 'Position', 'Points', 'Laps', 'FastestLapTime', 'Q1', 'Q2', 'Q3']: results_df[col] = np.nan
                  else: results_df[col] = None
        results_df = results_df[db_results_schema]

        db_drivers_schema = ['Abbreviation', 'DriverNumber', 'FullName', 'Nationality']
        drivers_df = drivers_df[[col for col in db_drivers_schema if col in drivers_df.columns]]
        db_teams_schema = ['TeamName', 'Nationality']
        teams_df = teams_df[[col for col in db_teams_schema if col in teams_df.columns]]

        essential_ids = ['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'Abbreviation']
        for eid in essential_ids: # Ensure no NaNs in essentials before dropna
             if results_df[eid].isnull().any():
                  if pd.api.types.is_string_dtype(results_df[eid]) or results_df[eid].dtype == object: results_df[eid].fillna('UNK_ID', inplace=True)
                  elif pd.api.types.is_numeric_dtype(results_df[eid]): results_df[eid].fillna(-1, inplace=True)

        initial_rows = len(results_df)
        results_df = results_df.dropna(subset=essential_ids)
        rows_dropped = initial_rows - len(results_df)
        if rows_dropped > 0: logger.warning(f"Dropped {rows_dropped} rows from results due to missing essential identifiers for {year} R{round_number} {session_name}.")

        logger.debug(f"Finished processing results for {year} R{round_number} {session_name}. Final shape: {results_df.shape}")
        return results_df, drivers_df, teams_df

    except Exception as e:
        logger.error(f"CRITICAL ERROR during process_session_results for {year} R{round_number} {session_name}: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def process_laps_and_pits(session, year, round_number, session_name):
    """Processes lap times and pit stops into DataFrames."""
    laps_df_final = pd.DataFrame()
    pits_df_final = pd.DataFrame()

    if session is None or not hasattr(session, 'laps') or session.laps is None or session.laps.empty:
        logger.warning(f"No lap data found for {year} R{round_number} {session_name}")
        return laps_df_final, pits_df_final

    try:
        laps = session.laps.copy()
        logger.debug(f"Processing {len(laps)} laps for {year} R{round_number} {session_name}")

        # Add identifiers
        laps['Year'] = int(year)
        laps['RoundNumber'] = int(round_number)
        laps['SessionName'] = str(session_name)

        # Define columns to keep for laps table
        lap_cols_db = [
            'Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'LapNumber', 'LapTime',
            'Stint', 'TyreLife', 'Compound', 'IsAccurate', 'IsPitOutLap', 'IsPitInLap',
            'Sector1Time', 'Sector2Time', 'Sector3Time'
        ]
        # Select available columns
        laps_df_final = laps[[col for col in lap_cols_db if col in laps.columns]].copy()

        # Convert time columns to seconds
        for time_col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
             if time_col in laps_df_final.columns:
                  laps_df_final[time_col] = laps_df_final[time_col].apply(parse_lap_time).astype(float)

        # Convert boolean flags to integer
        for bool_col in ['IsAccurate', 'IsPitOutLap', 'IsPitInLap']:
             if bool_col in laps_df_final.columns:
                  laps_df_final[bool_col] = laps_df_final[bool_col].astype(bool).astype(int) # Ensure boolean then int

        # Ensure other types
        for int_col in ['Stint', 'LapNumber']:
            if int_col in laps_df_final.columns: laps_df_final[int_col] = utils.safe_to_numeric(laps_df_final[int_col], fallback=-1).astype(int)
        if 'TyreLife' in laps_df_final.columns: laps_df_final['TyreLife'] = utils.safe_to_numeric(laps_df_final['TyreLife'], fallback=np.nan).astype(float)
        if 'Compound' in laps_df_final.columns: laps_df_final['Compound'] = laps_df_final['Compound'].astype(str).fillna('UNKNOWN')

        # --- Extract Pit Stops ---
        # Use get_pit_log if available (newer FastF1 versions)
        if hasattr(laps, 'get_pit_log') and callable(laps.get_pit_log):
             pit_log = laps.get_pit_log()
             if not pit_log.empty:
                pits_df_final = pit_log[['DriverNumber', 'LapNumber', 'StopDuration']].copy()
                pits_df_final.rename(columns={'StopDuration': 'PitDuration'}, inplace=True)
                # Add identifiers and StopNumber
                pits_df_final['Year'] = int(year)
                pits_df_final['RoundNumber'] = int(round_number)
                pits_df_final['SessionName'] = str(session_name)
                # Generate StopNumber per driver within the session
                pits_df_final['StopNumber'] = pits_df_final.groupby('DriverNumber').cumcount() + 1
                # Convert duration
                pits_df_final['PitDuration'] = pits_df_final['PitDuration'].apply(parse_lap_time).astype(float)
                # Reorder
                pit_cols_db = ['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'StopNumber', 'LapNumber', 'PitDuration']
                pits_df_final = pits_df_final[[col for col in pit_cols_db if col in pits_df_final.columns]]

        else: # Fallback: Infer pits from IsPitInLap/IsPitOutLap (less accurate duration)
            pits_in = laps_df_final[laps_df_final['IsPitInLap'] == 1]
            if not pits_in.empty:
                # Basic pit info - duration requires more complex calculation
                pits_df_final = pits_in[['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'LapNumber']].copy()
                pits_df_final['StopNumber'] = pits_df_final.groupby('DriverNumber').cumcount() + 1
                pits_df_final['PitDuration'] = np.nan # Duration not easily available here
                pit_cols_db = ['Year', 'RoundNumber', 'SessionName', 'DriverNumber', 'StopNumber', 'LapNumber', 'PitDuration']
                pits_df_final = pits_df_final[[col for col in pit_cols_db if col in pits_df_final.columns]]

        logger.debug(f"Processed laps ({len(laps_df_final)}) and pits ({len(pits_df_final)}) for {year} R{round_number} {session_name}")

    except Exception as e:
        logger.error(f"Error processing laps/pits for {year} R{round_number} {session_name}: {e}", exc_info=True)
        laps_df_final = pd.DataFrame() # Return empty on error
        pits_df_final = pd.DataFrame()

    return laps_df_final, pits_df_final


def process_weather_data(session, year, round_number, session_name):
    """Processes weather data into a DataFrame."""
    weather_df_final = pd.DataFrame()

    if session is None or not hasattr(session, 'weather_data') or session.weather_data is None or session.weather_data.empty:
        logger.warning(f"No weather data found for {year} R{round_number} {session_name}")
        return weather_df_final

    try:
        weather = session.weather_data.copy()
        logger.debug(f"Processing {len(weather)} weather entries for {year} R{round_number} {session_name}")

        # Add identifiers
        weather['Year'] = int(year)
        weather['RoundNumber'] = int(round_number)
        weather['SessionName'] = str(session_name)

        # Convert Time column (timedelta from session start) to seconds
        if 'Time' in weather.columns:
            weather['Time'] = weather['Time'].apply(parse_lap_time).astype(float)
        else:
            logger.warning("Weather data missing 'Time' column.")
            weather['Time'] = np.nan # Add time as NaN if missing

        # Define columns to keep and clean
        weather_cols_db = [
            'Year', 'RoundNumber', 'SessionName', 'Time', 'AirTemp', 'TrackTemp',
            'Humidity', 'Pressure', 'WindSpeed', 'WindDirection', 'Rainfall'
        ]
        weather_df_final = weather[[col for col in weather_cols_db if col in weather.columns]].copy()

        # Convert boolean Rainfall to integer
        if 'Rainfall' in weather_df_final.columns:
            weather_df_final['Rainfall'] = weather_df_final['Rainfall'].astype(bool).astype(int)

        # Ensure numeric types
        for col in ['AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'WindDirection']:
             if col in weather_df_final.columns:
                  weather_df_final[col] = utils.safe_to_numeric(weather_df_final[col], fallback=np.nan).astype(float)

        logger.debug(f"Processed weather data for {year} R{round_number} {session_name}")

    except Exception as e:
        logger.error(f"Error processing weather for {year} R{round_number} {session_name}: {e}", exc_info=True)
        weather_df_final = pd.DataFrame()

    return weather_df_final


def fetch_single_session_with_retry(year, round_number, session_code):
    """Attempts to fetch a single session, handling rate limits with retries."""
    session = None
    retries = 0
    while retries <= MAX_RATE_LIMIT_RETRIES:
        try:
            session = ff1.get_session(year, round_number, session_code)
            # Load necessary data based on flags
            load_kwargs = {'laps': config.LOAD_LAPS, 'weather': config.LOAD_WEATHER,
                           'messages': False, 'telemetry': False} # Keep telemetry False
            logger.debug(f"Loading session with args: {load_kwargs}")
            session.load(**load_kwargs)
            logger.debug(f"Session loaded successfully for {year} R{round_number} {session_code}.")
            return session # Success

        except ff1.RateLimitExceededError as e:
            retries += 1
            if retries > MAX_RATE_LIMIT_RETRIES:
                logger.error(f"RATE LIMIT EXCEEDED after retries fetching {year} R{round_number} {session_code}: {e}. Giving up.")
                return None
            else:
                wait = RATE_LIMIT_SLEEP_S * (retries) # Increase wait time on subsequent retries
                logger.warning(f"Rate limit hit ({retries}/{MAX_RATE_LIMIT_RETRIES}) fetching {year} R{round_number} {session_code}. Sleeping {wait}s...")
                time.sleep(wait)
                logger.info(f"Retrying fetch {retries} for {year} R{round_number} {session_code}...")

        except ValueError as e:
             if "does not exist for this event" in str(e).lower() or "invalid literal" in str(e).lower():
                  logger.warning(f"Session '{session_code}' not available for {year} R{round_number}. Skipping. (ValueError: {e})")
                  return None
             else:
                  logger.error(f"Unexpected ValueError loading {year} R{round_number} {session_code}: {e}", exc_info=True)
                  return None

        except ConnectionError as e:
             logger.error(f"Connection error fetching {year} R{round_number} {session_code}: {e}. Skipping session.")
             time.sleep(5)
             return None
        except Exception as e:
            logger.error(f"Failed to get/load session {year} R{round_number} {session_code}: {e}", exc_info=True)
            return None

    logger.error(f"Exceeded max retries for {year} R{round_number} {session_code} due to persistent errors.")
    return None


def fetch_and_store_event(year, event_row):
    """Fetches, processes, and stores data for relevant sessions of a single event."""
    try:
        round_number = int(event_row['RoundNumber'])
        event_location = event_row.get('Location', f'Unknown Location R{round_number}')
        event_name = event_row.get('EventName', f'Unknown Event R{round_number}')
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid event data: {e}. Row data: {event_row}")
        return True # Allow processing next event

    session_iterator = SESSION_IDENTIFIERS_TO_FETCH

    for session_code in session_iterator:
        logger.debug(f"Checking data for {year} R{round_number} ({event_name}) Session {session_code}")

        # 1. Check Database First (Check 'results' table as proxy)
        try:
            if database.check_data_exists(year, round_number, session_code):
                logger.info(f"Data exists for {year} R{round_number} ({event_name}) {session_code}. Skipping fetch.")
                continue
        except Exception as e:
            logger.error(f"DB check failed for {year} R{round_number} {session_code}: {e}.", exc_info=False)

        logger.info(f"Attempting fetch for {year} R{round_number} ({event_name}) Session {session_code}...")

        # 2. Fetch and Load Data with Retry Logic
        session = fetch_single_session_with_retry(year, round_number, session_code)

        if session is None:
             logger.warning(f"Failed to fetch/load session {session_code} for {year} R{round_number} after handling. Skipping save.")
             continue # Move to next session_code

        # Add the default sleep AFTER successful load
        logger.debug(f"Sleeping for {DEFAULT_API_SLEEP_S}s...")
        time.sleep(DEFAULT_API_SLEEP_S)

        # 3. Process Data (Results, Laps, Weather, Pits)
        results_df, drivers_df, teams_df = process_session_results(session, year, round_number, session_code)
        laps_df, pits_df = process_laps_and_pits(session, year, round_number, session_code)
        weather_df = process_weather_data(session, year, round_number, session_code)

        # 4. Save ALL processed data to Database
        try:
            # Save Results, Drivers, Teams
            if not results_df.empty: database.save_data(results_df, 'results')
            else: logger.info(f"No valid results processed for {year} R{round_number} {session_code}.")

            if not drivers_df.empty:
                existing_drivers_df = database.load_data("SELECT Abbreviation FROM drivers")
                existing_drivers_list = []
                if not existing_drivers_df.empty and 'Abbreviation' in existing_drivers_df.columns:
                    existing_drivers_list = existing_drivers_df['Abbreviation'].tolist()
                new_drivers = drivers_df[~drivers_df['Abbreviation'].isin(existing_drivers_list)].copy()
                db_drivers_schema = ['Abbreviation', 'DriverNumber', 'FullName', 'Nationality']
                for col in db_drivers_schema:
                    if col not in new_drivers.columns: new_drivers[col] = None
                if not new_drivers.empty: database.save_data(new_drivers[db_drivers_schema], 'drivers')

            if not teams_df.empty:
                existing_teams_df = database.load_data("SELECT TeamName FROM teams")
                existing_teams_list = []
                if not existing_teams_df.empty and 'TeamName' in existing_teams_df.columns:
                     existing_teams_list = existing_teams_df['TeamName'].tolist()
                new_teams = teams_df[~teams_df['TeamName'].isin(existing_teams_list)].copy()
                db_teams_schema = ['TeamName', 'Nationality']
                for col in db_teams_schema:
                    if col not in new_teams.columns: new_teams[col] = None
                if not new_teams.empty: database.save_data(new_teams[db_teams_schema], 'teams')

            # Save Laps, Weather, Pits
            if not laps_df.empty: database.save_data(laps_df, 'laps')
            if not weather_df.empty: database.save_data(weather_df, 'weather')
            if not pits_df.empty: database.save_data(pits_df, 'pit_stops')

        except Exception as e:
            logger.error(f"Database save error for {year} R{round_number} {session_code}: {e}", exc_info=True)

    return True # Signal success for this event


# --- Main Function to Update Database ---
def update_database(years_list):
    """Fetches, processes, and stores F1 data for the specified years into the database."""
    logger.info(f"Starting database update process for years: {years_list}")
    year_iterator = sorted(years_list)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    logger.info(f"Current UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    for year in year_iterator:
        logger.info(f"===== Processing Year: {year} =====")
        schedule = None
        schedule_load_attempts = 0
        while schedule_load_attempts <= MAX_RATE_LIMIT_RETRIES:
            try:
                # 1. Get Schedule
                schedule = ff1.get_event_schedule(year, include_testing=False)
                logger.debug(f"Successfully loaded schedule for {year}")
                break
            except ff1.RateLimitExceededError as e:
                 schedule_load_attempts += 1
                 if schedule_load_attempts > MAX_RATE_LIMIT_RETRIES:
                      logger.error(f"RATE LIMIT EXCEEDED after retries fetching schedule for {year}: {e}. Skipping year.")
                      schedule = None; break
                 else:
                      wait = RATE_LIMIT_SLEEP_S * (schedule_load_attempts)
                      logger.warning(f"Rate limit hit ({schedule_load_attempts}/{MAX_RATE_LIMIT_RETRIES}) fetching schedule for {year}. Sleeping {wait}s...")
                      time.sleep(wait); logger.info(f"Retrying schedule fetch for {year}...")
            except Exception as e: logger.error(f"Failed to fetch schedule for {year}: {e}", exc_info=True); schedule = None; break

        if schedule is None or schedule.empty: logger.warning(f"No schedule found/loaded for {year}. Skipping year processing."); continue
        logger.info(f"Found {len(schedule)} events initially for {year}.")

        # Filter schedule for current year
        if year == now_utc.year:
            logger.info(f"Filtering schedule for current year ({year}) to past events...")
            try:
                schedule['EventDate_dt'] = pd.to_datetime(schedule['EventDate'], utc=True, errors='coerce')
                original_count = len(schedule)
                schedule = schedule[(schedule['EventDate_dt'] <= now_utc) | (schedule['EventDate_dt'].isnull())]
                filtered_count = len(schedule)
                if filtered_count < original_count: logger.info(f"Filtered schedule: Kept {filtered_count}/{original_count} events on/before {now_utc.date()}.")
            except Exception as e: logger.error(f"Error during date filtering for {year}: {e}", exc_info=True)
        if schedule.empty: logger.warning(f"No past events found for {year}. Skipping year processing."); continue

        # 2. Save/Update Event Schedule Info
        try:
            event_cols_from_schedule = ['RoundNumber', 'EventName', 'Country', 'Location', 'OfficialEventName', 'EventDate']
            available_cols = [col for col in event_cols_from_schedule if col in schedule.columns]
            schedule_to_save = schedule[available_cols].copy(); schedule_to_save['Year'] = year
            final_event_cols_for_db = ['Year', 'RoundNumber', 'EventName', 'Country', 'Location', 'OfficialEventName', 'EventDate']
            for col in final_event_cols_for_db:
                if col not in schedule_to_save.columns: schedule_to_save[col] = None
            schedule_to_save = schedule_to_save[final_event_cols_for_db]
            if 'EventDate' in schedule_to_save.columns: schedule_to_save['EventDate'] = pd.to_datetime(schedule_to_save['EventDate'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
            existing_events_df = database.load_data(f"SELECT Year, RoundNumber FROM events WHERE Year = ?", params=(year,))
            if not existing_events_df.empty:
                existing_pairs = set(zip(existing_events_df['Year'].astype(int), existing_events_df['RoundNumber'].astype(int)))
                schedule_to_save['RoundNumber'] = pd.to_numeric(schedule_to_save['RoundNumber'], errors='coerce').fillna(-1).astype(int)
                schedule_to_save_filtered = schedule_to_save[~schedule_to_save.apply(lambda row: (int(row['Year']), row['RoundNumber']) in existing_pairs, axis=1)].copy()
            else: schedule_to_save_filtered = schedule_to_save.copy()
            if not schedule_to_save_filtered.empty:
                logger.info(f"Adding {len(schedule_to_save_filtered)} new events to schedule table for {year}.")
                database.save_data(schedule_to_save_filtered, 'events')
            else: logger.info(f"No new events to add to schedule table for {year}.")
        except Exception as e: logger.error(f"Non-critical error processing/saving schedule data for {year}: {e}", exc_info=True)

        # 3. Fetch Session Data for Each Event
        logger.info(f"Fetching session data for {len(schedule)} relevant events in {year}...")
        event_iterator = tqdm(schedule.iterrows(), total=len(schedule), desc=f"Events {year}", unit="event", leave=True)
        stop_year_processing = False
        for index, event_row in event_iterator:
              try:
                   success = fetch_and_store_event(year, event_row)
                   if not success: # False means rate limit hit after retries
                       logger.error(f"Stopping processing for year {year} due to persistent rate limit.")
                       stop_year_processing = True
                       break
              except Exception as inner_e:
                   logger.error(f"Unhandled error processing event R{event_row.get('RoundNumber', 'N/A')}: {inner_e}", exc_info=True)
        if stop_year_processing: break # Stop processing more years if rate limit hit

    logger.info("===== Database update process finished. =====")