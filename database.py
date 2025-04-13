# /f1_predictor/database.py

import sqlite3
from sqlalchemy import create_engine, inspect, text
import pandas as pd
import config
import utils
import warnings
import sys
import os
import logging
logger = utils.get_logger(__name__)

# Create engine (Keep as is)
DB_ENGINE = None
try:
    db_path = config.DB_PATH
    logger.info(f"Attempting to create database engine for: {db_path}")
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir): os.makedirs(db_dir, exist_ok=True)
    DB_ENGINE = create_engine(f'sqlite:///{db_path}')
    logger.info(f"Database engine created successfully for: {db_path}")
except Exception as e:
    logger.error(f"CRITICAL: Failed to create database engine: {e}", exc_info=True)
    sys.exit(1)


def create_tables():
    """Creates necessary database tables and indexes if they don't exist."""
    if not DB_ENGINE: logger.error("DB engine invalid. Cannot create tables."); return

    # Table definitions (Keep as is)
    required_tables = {
        "events": """
            CREATE TABLE IF NOT EXISTS events (
                Year INTEGER NOT NULL, RoundNumber INTEGER NOT NULL, EventName TEXT,
                Country TEXT, Location TEXT, OfficialEventName TEXT, EventDate TEXT,
                PRIMARY KEY (Year, RoundNumber) ); """,
        "results": """
            CREATE TABLE IF NOT EXISTS results (
                Year INTEGER NOT NULL, RoundNumber INTEGER NOT NULL, SessionName TEXT NOT NULL,
                DriverNumber TEXT NOT NULL, Abbreviation TEXT, TeamName TEXT, GridPosition REAL,
                Position REAL, Points REAL, Status TEXT, Laps REAL, FastestLapTime REAL,
                Q1 REAL, Q2 REAL, Q3 REAL, PRIMARY KEY (Year, RoundNumber, SessionName, DriverNumber) ); """,
         "drivers": """
            CREATE TABLE IF NOT EXISTS drivers ( Abbreviation TEXT PRIMARY KEY NOT NULL, DriverNumber TEXT, FullName TEXT, Nationality TEXT ); """,
         "teams": """
            CREATE TABLE IF NOT EXISTS teams ( TeamName TEXT PRIMARY KEY NOT NULL, Nationality TEXT ); """,
         "laps": """
            CREATE TABLE IF NOT EXISTS laps (
                Year INTEGER NOT NULL, RoundNumber INTEGER NOT NULL, SessionName TEXT NOT NULL,
                DriverNumber TEXT NOT NULL, LapNumber INTEGER NOT NULL, LapTime REAL, Stint INTEGER,
                TyreLife REAL, Compound TEXT, IsAccurate INTEGER, -- Boolean as INTEGER (0 or 1)
                IsPitOutLap INTEGER, IsPitInLap INTEGER, Sector1Time REAL, Sector2Time REAL, Sector3Time REAL,
                PRIMARY KEY (Year, RoundNumber, SessionName, DriverNumber, LapNumber) ); """,
         "weather": """
            CREATE TABLE IF NOT EXISTS weather (
                Year INTEGER NOT NULL, RoundNumber INTEGER NOT NULL, SessionName TEXT NOT NULL,
                Time REAL NOT NULL, -- Seconds from session start or timestamp? Store offset for now.
                AirTemp REAL, TrackTemp REAL, Humidity REAL, Pressure REAL,
                WindSpeed REAL, WindDirection INTEGER, Rainfall INTEGER, -- Boolean as INTEGER
                PRIMARY KEY (Year, RoundNumber, SessionName, Time) ); """,
          "pit_stops": """
             CREATE TABLE IF NOT EXISTS pit_stops (
                 Year INTEGER NOT NULL, RoundNumber INTEGER NOT NULL, SessionName TEXT NOT NULL,
                 DriverNumber TEXT NOT NULL, StopNumber INTEGER NOT NULL, LapNumber INTEGER,
                 PitDuration REAL, -- Total time in pits
                 PRIMARY KEY (Year, RoundNumber, SessionName, DriverNumber, StopNumber) ); """
    }

    # --- NEW: Index definitions ---
    required_indexes = {
        "idx_results_main": "CREATE INDEX IF NOT EXISTS idx_results_main ON results (Year, RoundNumber, SessionName);",
        "idx_results_abbr": "CREATE INDEX IF NOT EXISTS idx_results_abbr ON results (Abbreviation);",
        "idx_events_loc": "CREATE INDEX IF NOT EXISTS idx_events_loc ON events (Location);",
        "idx_laps_main": "CREATE INDEX IF NOT EXISTS idx_laps_main ON laps (Year, RoundNumber, SessionName, DriverNumber);",
        "idx_weather_main": "CREATE INDEX IF NOT EXISTS idx_weather_main ON weather (Year, RoundNumber, SessionName);",
        "idx_pits_main": "CREATE INDEX IF NOT EXISTS idx_pits_main ON pit_stops (Year, RoundNumber, SessionName, DriverNumber);"
    }

    logger.info("Ensuring database tables and indexes exist...")
    try:
        with DB_ENGINE.connect() as connection:
            with connection.begin(): # Transaction
                # Create Tables
                for name, sql in required_tables.items():
                    logger.debug(f"Ensuring table: {name}")
                    connection.execute(text(sql))
                # Create Indexes
                logger.info("Creating/verifying indexes...")
                for name, sql in required_indexes.items():
                     logger.debug(f"Ensuring index: {name}")
                     connection.execute(text(sql))

            logger.info("Table and index creation/verification process completed.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed during table/index creation/verification: {e}", exc_info=True)
        sys.exit(1)

# --- save_data, load_data, check_data_exists, table_exists (Keep as is) ---
def save_data(df, table_name):
    if not DB_ENGINE: logger.error(f"DB engine invalid. Cannot save to '{table_name}'."); return
    if df is None or df.empty: logger.warning(f"Attempted to save empty DataFrame to '{table_name}'. Skipping."); return

    logger.debug(f"Attempting to save {len(df)} rows to '{table_name}'.")
    try:
        for col in df.select_dtypes(include=['boolean', 'bool']).columns:
             logger.debug(f"Converting boolean column '{col}' to integer for table '{table_name}'")
             df[col] = df[col].astype(int)

        df.to_sql(table_name, DB_ENGINE, if_exists='append', index=False, chunksize=1000)
        logger.info(f"Appended {len(df)} rows to '{table_name}'.")
    except Exception as e:
        if 'unique constraint failed' in str(e).lower(): logger.warning(f"Constraint violation saving to '{table_name}'. Data likely exists (or duplicate in batch).")
        else: logger.error(f"Error saving data to '{table_name}': {e}", exc_info=True)

def load_data(query, params=None):
    if not DB_ENGINE: logger.error("DB engine invalid. Cannot load data."); return pd.DataFrame()

    logger.debug(f"Executing query: {query[:200]}... with params: {params}")
    try:
        df = pd.read_sql(query, DB_ENGINE, params=params)
        # Reduce log noise for potentially large loads like laps
        log_level = logging.INFO if len(df) < 50000 else logging.DEBUG
        logger.log(log_level, f"Loaded {len(df)} rows using query: {query[:100]}...")
        return df
    except Exception as e:
        if "no such table" in str(e).lower(): logger.warning(f"Query failed because table does not exist: '{query[:100]}...': {e}")
        else: logger.error(f"Error executing query '{query[:100]}...': {e}", exc_info=True)
        return pd.DataFrame()

def check_data_exists(year, round_number, session_name):
    if not DB_ENGINE: logger.error("DB engine invalid. Cannot check data existence."); return False
    query = text(""" SELECT 1 FROM results WHERE Year = :year AND RoundNumber = :round AND SessionName = :session LIMIT 1 """)
    params = {"year": year, "round": round_number, "session": session_name}
    logger.debug(f"Checking existence with params: {params}")
    try:
        with DB_ENGINE.connect() as connection:
            result = connection.execute(query, params).scalar()
            exists = result is not None
            logger.debug(f"Existence check result for {params}: {exists}")
            return exists
    except Exception as e:
        if "no such table" in str(e).lower(): logger.warning(f"Table 'results' does not exist while checking for {params}. Assuming data does not exist.")
        else: logger.error(f"Error checking data existence for {params}: {e}", exc_info=True)
        return False

def table_exists(table_name):
    if not DB_ENGINE: logger.error("DB engine invalid. Cannot check table existence."); return False
    logger.debug(f"Checking if table '{table_name}' exists.")
    try:
        inspector = inspect(DB_ENGINE)
        exists = inspector.has_table(table_name)
        logger.debug(f"Table '{table_name}' exists: {exists}")
        return exists
    except Exception as e:
        logger.error(f"Error checking table existence for '{table_name}': {e}", exc_info=True)
        return False