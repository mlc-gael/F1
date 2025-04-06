# /f1_predictor/data_loader.py

import fastf1 as ff1
# Exceptions are directly under ff1
# import fastf1.exceptions

import pandas as pd
import numpy as np
import time
import warnings
import os
import sys
import datetime 
from tqdm import tqdm

# Import project modules
import config
import database
import utils

# Suppress specific warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# Setup logger
logger = utils.get_logger(__name__)

# --- Constants ---
# Increase base sleep time significantly
DEFAULT_API_SLEEP_S = 1.0
# Longer sleep when rate limit is hit
RATE_LIMIT_SLEEP_S = 60.0
MAX_RATE_LIMIT_RETRIES = 1
# Known session identifiers (add more if needed)
SESSION_IDENTIFIERS_TO_FETCH = ['R', 'Q', 'S', 'SQ', 'SS', 'FP1', 'FP2', 'FP3']


# Configure FastF1 Cache
try:
    if config.CACHE_DIR and not os.path.exists(config.CACHE_DIR):
         os.makedirs(config.CACHE_DIR, exist_ok=True)
         logger.info(f"Created FastF1 cache directory: {config.CACHE_DIR}")
    ff1.Cache.enable_cache(config.CACHE_DIR)
    logger.info(f"FastF1 cache enabled at: {config.CACHE_DIR}")
except Exception as e:
    logger.error(f"CRITICAL: Failed to enable FastF1 cache at {config.CACHE_DIR}: {e}", exc_info=True)


# --- Helper Functions ---
def parse_lap_time(time_obj):
    """Converts Pandas Timedelta or numeric to seconds (float) or returns np.nan."""
    if pd.isna(time_obj): return np.nan
    try:
        if isinstance(time_obj, (int, float)): return float(time_obj)
        if isinstance(time_obj, pd.Timedelta): return time_obj.total_seconds()
        return np.nan
    except (AttributeError, ValueError): return np.nan


def process_session_results(session, year, round_number, session_name):
    """Processes results from a loaded FastF1 session object into DataFrames matching DB schema."""
    if session is None or not hasattr(session, 'results') or session.results is None or session.results.empty:
        logger.warning(f"No session.results data found for {year} R{round_number} Session {session_name}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        results = session.results.copy()
        logger.debug(f"Processing results for {year} R{round_number} {session_name}. Input columns: {results.columns.tolist()}")

        results['Year'] = int(year)
        results['RoundNumber'] = int(round_number)
        results['SessionName'] = str(session_name)

        source_column_map = { 'DriverNumber': 'DriverNumber', 'Abbreviation': 'Abbreviation', 'TeamName': 'TeamName', 'GridPosition': 'GridPosition', 'Position': 'Position', 'Points': 'Points', 'Status': 'Status', 'Laps': 'Laps', 'FastestLapTime': 'FastestLapTime', 'Q1': 'Q1', 'Q2': 'Q2', 'Q3': 'Q3', 'FullName': 'FullName' }
        results_df = results[[col for col in source_column_map if col in results.columns]].rename(columns=source_column_map)

        # --- Ensure Essential Columns & Clean Data Types ---
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

        # --- Extract Driver/Team Info ---
        driver_info_cols = ['DriverNumber', 'Abbreviation', 'TeamName', 'FullName']
        drivers_df = results_df.dropna(subset=['Abbreviation']).loc[:, [c for c in driver_info_cols if c in results_df.columns]].drop_duplicates(subset=['Abbreviation']).copy()
        if not drivers_df.empty: drivers_df['Nationality'] = None

        teams_df = results_df.dropna(subset=['TeamName']).loc[:, ['TeamName']].drop_duplicates().copy()
        if not teams_df.empty: teams_df['Nationality'] = None

        # --- Final Schema Alignment & Integrity Check ---
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
        # Explicitly fill NaNs in essentials before dropna, just in case
        for eid in essential_ids:
            if results_df[eid].isnull().any():
                 if results_df[eid].dtype == object or pd.api.types.is_string_dtype(results_df[eid]):
                      results_df[eid].fillna('UNK_ID', inplace=True)
                 elif pd.api.types.is_numeric_dtype(results_df[eid]):
                      results_df[eid].fillna(-1, inplace=True) # Or appropriate numeric placeholder

        nan_check = results_df[essential_ids].isnull().sum()
        if nan_check.sum() > 0: logger.warning(f"NaNs remain in essential identifiers BEFORE dropna for {year} R{round_number} {session_name}: \n{nan_check[nan_check > 0]}")

        initial_rows = len(results_df)
        results_df = results_df.dropna(subset=essential_ids) # Assign back
        rows_dropped = initial_rows - len(results_df)
        if rows_dropped > 0: logger.warning(f"Dropped {rows_dropped} rows from results due to missing essential identifiers for {year} R{round_number} {session_name}.")

        logger.debug(f"Finished processing results for {year} R{round_number} {session_name}. Final shape: {results_df.shape}")
        return results_df, drivers_df, teams_df

    except Exception as e:
        logger.error(f"CRITICAL ERROR during process_session_results for {year} R{round_number} {session_name}: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Return empty dfs


def fetch_single_session_with_retry(year, round_number, session_code):
    """Attempts to fetch a single session, handling rate limits with retries."""
    session = None
    retries = 0
    while retries <= MAX_RATE_LIMIT_RETRIES:
        try:
            session = ff1.get_session(year, round_number, session_code)
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            logger.debug(f"Session loaded successfully for {year} R{round_number} {session_code}.")
            return session # Success

        except ff1.RateLimitExceededError as e:
            retries += 1
            if retries > MAX_RATE_LIMIT_RETRIES:
                logger.error(f"RATE LIMIT EXCEEDED after {retries-1} retries fetching {year} R{round_number} {session_code}: {e}. Giving up on this session.")
                return None # Give up after max retries
            else:
                logger.warning(f"Rate limit hit ({retries}/{MAX_RATE_LIMIT_RETRIES}) fetching {year} R{round_number} {session_code}. Sleeping for {RATE_LIMIT_SLEEP_S}s...")
                time.sleep(RATE_LIMIT_SLEEP_S)
                logger.info(f"Retrying fetch for {year} R{round_number} {session_code}...")

        except ValueError as e:
             if "does not exist for this event" in str(e).lower() or "invalid literal for int() with base 10" in str(e).lower():
                  logger.warning(f"Session '{session_code}' not available for {year} R{round_number}. Skipping. (ValueError: {e})")
                  return None # Session genuinely doesn't exist
             else:
                  logger.error(f"Unexpected ValueError loading {year} R{round_number} {session_code}: {e}", exc_info=True)
                  return None # Skip on other ValueErrors

        except ConnectionError as e:
             logger.error(f"Connection error fetching {year} R{round_number} {session_code}: {e}. Skipping session.")
             time.sleep(5) # Wait a bit after connection error
             return None # Skip this session attempt
        except Exception as e:
            logger.error(f"Failed to get/load session {year} R{round_number} {session_code}: {e}", exc_info=True)
            return None # Skip on other load errors

    return None # Should only be reached if max retries exceeded


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

        # 1. Check Database First
        try:
            if database.check_data_exists(year, round_number, session_code):
                logger.info(f"Data exists for {year} R{round_number} ({event_name}) {session_code}. Skipping.")
                continue
        except Exception as e:
            logger.error(f"DB check failed for {year} R{round_number} {session_code}: {e}.", exc_info=False)

        logger.info(f"Attempting fetch for {year} R{round_number} ({event_name}) Session {session_code}...")

        # 2. Fetch and Load Data with Retry Logic
        session = fetch_single_session_with_retry(year, round_number, session_code)

        if session is None:
            # Error occurred (rate limit exceeded after retries, session not available, or other load error)
            # Check if it was specifically a rate limit error that caused failure after retries
            # (The function would have logged the final error)
            # For simplicity, we just continue to the next session_code here.
            # If RateLimit caused the *final* failure, the outer loop needs to detect it.
            # Let's refine the return signal later if needed.
             logger.warning(f"Failed to fetch/load session {session_code} after handling known issues/retries. Skipping save.")
             continue # Move to next session

        # Add the default sleep AFTER successful load, before next request
        logger.debug(f"Sleeping for {DEFAULT_API_SLEEP_S}s...")
        time.sleep(DEFAULT_API_SLEEP_S)

        # 3. Process Session Results
        results_df, drivers_df, teams_df = process_session_results(session, year, round_number, session_code)

        # 4. Save to Database
        try:
            if not results_df.empty:
                database.save_data(results_df, 'results')
            else:
                logger.info(f"No valid results processed/saved for {year} R{round_number} {session_code}.")

            # Save new drivers/teams
            if not drivers_df.empty:
                existing_drivers_df = database.load_data("SELECT Abbreviation FROM drivers")
                existing_drivers_list = []
                if not existing_drivers_df.empty and 'Abbreviation' in existing_drivers_df.columns:
                    existing_drivers_list = existing_drivers_df['Abbreviation'].tolist()
                new_drivers = drivers_df[~drivers_df['Abbreviation'].isin(existing_drivers_list)].copy()
                db_drivers_schema = ['Abbreviation', 'DriverNumber', 'FullName', 'Nationality']
                for col in db_drivers_schema:
                    if col not in new_drivers.columns: new_drivers[col] = None
                if not new_drivers.empty:
                    database.save_data(new_drivers[db_drivers_schema], 'drivers')

            if not teams_df.empty:
                existing_teams_df = database.load_data("SELECT TeamName FROM teams")
                existing_teams_list = []
                if not existing_teams_df.empty and 'TeamName' in existing_teams_df.columns:
                     existing_teams_list = existing_teams_df['TeamName'].tolist()
                new_teams = teams_df[~teams_df['TeamName'].isin(existing_teams_list)].copy()
                db_teams_schema = ['TeamName', 'Nationality']
                for col in db_teams_schema:
                    if col not in new_teams.columns: new_teams[col] = None
                if not new_teams.empty:
                    database.save_data(new_teams[db_teams_schema], 'teams')
        except Exception as e:
            logger.error(f"Database interaction error for {year} R{round_number} {session_code}: {e}", exc_info=True)

    # If the event loop finishes, assume success for this event (no persistent rate limit)
    return True


# --- Main Function to Update Database ---
def update_database(years_list):
    """Fetches, processes, and stores F1 data for the specified years into the database."""
    logger.info(f"Starting database update process for years: {years_list}")
    year_iterator = sorted(years_list)
    # rate_limit_hit_globally = False # Removed - retry handles pauses

    # Get current time once for comparison (use UTC for consistency with FastF1)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    logger.info(f"Current UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")


    for year in year_iterator:
        # if rate_limit_hit_globally:
        #     logger.warning(f"Skipping year {year} due to previous rate limit error.")
        #     continue

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
                      logger.error(f"RATE LIMIT EXCEEDED after {schedule_load_attempts-1} retries fetching schedule for {year}: {e}. Skipping year.")
                      schedule = None
                      break
                 else:
                      logger.warning(f"Rate limit hit ({schedule_load_attempts}/{MAX_RATE_LIMIT_RETRIES}) fetching schedule for {year}. Sleeping {RATE_LIMIT_SLEEP_S}s...")
                      time.sleep(RATE_LIMIT_SLEEP_S)
                      logger.info(f"Retrying schedule fetch for {year}...")

            except Exception as e:
                 logger.error(f"Failed to fetch schedule for {year}: {e}", exc_info=True)
                 schedule = None
                 break

        if schedule is None or schedule.empty:
             logger.warning(f"No schedule data found or loaded for year {year}. Skipping year processing.")
             continue
        logger.info(f"Found {len(schedule)} events initially for {year}.")


        # <<< --- ADD DATE FILTERING FOR CURRENT YEAR --- >>>
        if year == now_utc.year:
            logger.info(f"Filtering schedule for current year ({year}) to include only past events...")
            try:
                # Ensure EventDate is timezone-aware UTC for comparison
                schedule['EventDate_dt'] = pd.to_datetime(schedule['EventDate'], utc=True, errors='coerce')
                # Keep only events where the date is in the past (or NaT if conversion failed)
                original_count = len(schedule)
                schedule = schedule[(schedule['EventDate_dt'] <= now_utc) | (schedule['EventDate_dt'].isnull())]
                filtered_count = len(schedule)
                if filtered_count < original_count:
                    logger.info(f"Filtered schedule for {year}: Kept {filtered_count}/{original_count} events occurring on or before {now_utc.date()}.")
                # Drop the temporary datetime column if no longer needed
                # schedule = schedule.drop(columns=['EventDate_dt'])
            except Exception as e:
                logger.error(f"Error during date filtering for {year}: {e}", exc_info=True)
                # Continue with unfiltered schedule if filtering fails
        # <<< --- END DATE FILTERING --- >>>


        if schedule.empty:
             logger.warning(f"No past events found in schedule for {year} up to {now_utc.date()}. Skipping year processing.")
             continue


        # 2. Save/Update Event Schedule Info (Keep simple try/except)
        # (This section remains the same as before)
        try:
            event_cols_from_schedule = ['RoundNumber', 'EventName', 'Country', 'Location', 'OfficialEventName', 'EventDate']
            available_cols = [col for col in event_cols_from_schedule if col in schedule.columns]
            schedule_to_save = schedule[available_cols].copy()
            schedule_to_save['Year'] = year
            final_event_cols_for_db = ['Year', 'RoundNumber', 'EventName', 'Country', 'Location', 'OfficialEventName', 'EventDate']
            for col in final_event_cols_for_db:
                if col not in schedule_to_save.columns: schedule_to_save[col] = None
            schedule_to_save = schedule_to_save[final_event_cols_for_db]
            if 'EventDate' in schedule_to_save.columns:
                # Use the original EventDate column for saving, format as string
                schedule_to_save['EventDate'] = pd.to_datetime(schedule_to_save['EventDate'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')


            existing_events_df = database.load_data(f"SELECT Year, RoundNumber FROM events WHERE Year = ?", params=(year,))
            if not existing_events_df.empty:
                existing_pairs = set(zip(existing_events_df['Year'].astype(int), existing_events_df['RoundNumber'].astype(int)))
                # Handle potential NaN in RoundNumber before converting to int
                schedule_to_save['RoundNumber'] = pd.to_numeric(schedule_to_save['RoundNumber'], errors='coerce').fillna(-1).astype(int)
                schedule_to_save_filtered = schedule_to_save[
                    ~schedule_to_save.apply(lambda row: (int(row['Year']), row['RoundNumber']) in existing_pairs, axis=1)
                ].copy()
            else:
                schedule_to_save_filtered = schedule_to_save.copy()
            if not schedule_to_save_filtered.empty:
                logger.info(f"Adding {len(schedule_to_save_filtered)} new events to schedule table for {year}.")
                database.save_data(schedule_to_save_filtered, 'events')
            else:
                logger.info(f"No new events to add to schedule table for {year}.")
        except Exception as e:
            logger.error(f"Non-critical error processing/saving schedule data for {year}: {e}", exc_info=True)


        # 3. Fetch Session Data for Each Event (Iterate over the potentially filtered schedule)
        if schedule is not None and not schedule.empty:
             logger.info(f"Fetching session data for {len(schedule)} relevant events in {year}...")
             event_iterator = tqdm(schedule.iterrows(), total=len(schedule), desc=f"Events {year}", unit="event", leave=True)
             stop_year_processing = False # Flag to break outer loop if needed
             for index, event_row in event_iterator:
                  try:
                       success_or_continue = fetch_and_store_event(year, event_row)
                       if not success_or_continue:
                           logger.error(f"Rate limit likely hit during event R{event_row.get('RoundNumber', 'N/A')}, stopping processing for year {year}.")
                           stop_year_processing = True
                           break
                  except Exception as inner_e:
                       logger.error(f"Unhandled error processing event R{event_row.get('RoundNumber', 'N/A')} ({event_row.get('EventName', 'N/A')}): {inner_e}", exc_info=True)

             if stop_year_processing: continue # Go to next year if flag is set

        else:
             logger.warning(f"Skipping session data fetch for {year} due to empty or missing schedule after filtering.")


    logger.info("===== Database update process finished. =====")