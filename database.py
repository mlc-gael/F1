# /f1_predictor/database.py

import sqlite3
from sqlalchemy import create_engine, inspect, text
import pandas as pd
import config
import utils
import warnings
import sys
import os

# Get logger instance
logger = utils.get_logger(__name__)

# Create engine using path from config
DB_ENGINE = None
try:
    db_path = config.DB_PATH
    logger.info(f"Attempting to create database engine for: {db_path}")
    # Ensure directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Created database directory: {db_dir}")

    DB_ENGINE = create_engine(f'sqlite:///{db_path}')
    logger.info(f"Database engine created successfully for: {db_path}")
except Exception as e:
    logger.error(f"CRITICAL: Failed to create database engine for {config.DB_PATH}: {e}", exc_info=True)
    logger.error("Database operations will likely fail. Exiting.")
    sys.exit(1) # Exit if DB engine cannot be created


def table_exists(table_name):
    """Checks if a table exists in the database using SQLAlchemy inspect."""
    if not DB_ENGINE:
        logger.error("Cannot check table existence: Database engine not initialized.")
        return False
    try:
        inspector = inspect(DB_ENGINE)
        exists = inspector.has_table(table_name)
        logger.debug(f"Checked for table '{table_name}'. Exists: {exists}")
        return exists
    except Exception as e:
        logger.error(f"Error checking if table '{table_name}' exists: {e}", exc_info=True)
        return False


def create_tables():
    """Creates necessary database tables if they don't exist."""
    if not DB_ENGINE:
        logger.error("Database engine not initialized. Cannot create tables.")
        return

    # Use lowercase names for broader compatibility, ensure Primary Keys are defined
    # Use CREATE TABLE IF NOT EXISTS for safety
    required_tables = {
        "events": """
            CREATE TABLE IF NOT EXISTS events (
                Year INTEGER NOT NULL,
                RoundNumber INTEGER NOT NULL,
                EventName TEXT,
                Country TEXT,
                Location TEXT,
                OfficialEventName TEXT,
                EventDate TEXT, -- Storing as TEXT for simplicity
                PRIMARY KEY (Year, RoundNumber)
            );
        """,
        "results": """
            CREATE TABLE IF NOT EXISTS results (
                Year INTEGER NOT NULL,
                RoundNumber INTEGER NOT NULL,
                SessionName TEXT NOT NULL,
                DriverNumber TEXT NOT NULL,
                Abbreviation TEXT,
                TeamName TEXT,
                GridPosition REAL,
                Position REAL,
                Points REAL,
                Status TEXT,
                Laps REAL,
                FastestLapTime REAL,
                Q1 REAL,
                Q2 REAL,
                Q3 REAL,
                PRIMARY KEY (Year, RoundNumber, SessionName, DriverNumber)
            );
        """,
         "drivers": """
            CREATE TABLE IF NOT EXISTS drivers (
                Abbreviation TEXT PRIMARY KEY NOT NULL,
                DriverNumber TEXT,
                FullName TEXT,
                Nationality TEXT
            );
            """,
         "teams": """
            CREATE TABLE IF NOT EXISTS teams (
                TeamName TEXT PRIMARY KEY NOT NULL,
                Nationality TEXT
             );
         """
    }

    logger.info("Ensuring database tables exist...")
    try:
        with DB_ENGINE.connect() as connection:
            # Start a transaction
            with connection.begin():
                for name, sql in required_tables.items():
                    logger.debug(f"Executing CREATE TABLE IF NOT EXISTS for: {name}")
                    connection.execute(text(sql)) # Use text() wrapper
                    logger.debug(f"Table '{name}' ensured.")
            logger.info("Table creation/verification process completed successfully.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed during table creation/verification process: {e}", exc_info=True)
        logger.error("Exiting application because database tables could not be created/verified.")
        sys.exit(1)


def save_data(df, table_name):
    """Saves a DataFrame to the specified table using pandas.to_sql."""
    if not DB_ENGINE:
        logger.error(f"DB engine invalid. Cannot save to '{table_name}'.")
        return
    if df is None or df.empty:
        logger.warning(f"Attempted to save empty DataFrame to '{table_name}'. Skipping.")
        return

    logger.debug(f"Attempting to save {len(df)} rows to '{table_name}'.")
    try:
        # Ensure column types are suitable for DB before saving if needed
        # Example: Convert numpy specific types if they cause issues
        # for col in df.select_dtypes(include=['float64']).columns:
        #     df[col] = df[col].astype(float) # Ensure standard float
        # for col in df.select_dtypes(include=['int64']).columns:
        #      df[col] = df[col].astype(int) # Ensure standard int

        df.to_sql(table_name, DB_ENGINE, if_exists='append', index=False, chunksize=1000)
        logger.info(f"Appended {len(df)} rows to '{table_name}'.")
    except Exception as e:
        if 'UNIQUE constraint failed' in str(e).lower() or 'duplicate key value violates unique constraint' in str(e).lower():
             logger.warning(f"Constraint violation saving to '{table_name}'. Data likely exists.")
        else:
            logger.error(f"Error saving data to '{table_name}': {e}", exc_info=True)
            try: logger.debug(f"Sample DataFrame head:\n{df.head().to_string()}")
            except Exception: pass # Avoid errors during logging


def load_data(query, params=None):
    """Loads data from the database using a SQL query, optionally with parameters."""
    if not DB_ENGINE:
        logger.error("DB engine invalid. Cannot load data.")
        return pd.DataFrame()

    logger.debug(f"Executing query: {query[:200]}... with params: {params}")
    try:
        # Use pandas read_sql with parameters if provided
        df = pd.read_sql(query, DB_ENGINE, params=params)
        logger.info(f"Loaded {len(df)} rows using query: {query[:100]}...")
        return df
    except Exception as e:
        if "no such table" in str(e).lower():
             logger.error(f"Query failed because table does not exist: '{query[:100]}...': {e}")
        else:
             logger.error(f"Error executing query '{query[:100]}...': {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on ANY error


def check_data_exists(year, round_number, session_name):
    """Checks if data for a specific session exists using a COUNT or EXISTS query."""
    if not DB_ENGINE:
        logger.error("DB engine invalid. Cannot check data existence.")
        return False

    # Use SELECT 1 LIMIT 1 for efficiency on most DBs
    query = text("""
        SELECT 1 FROM results
        WHERE Year = :year AND RoundNumber = :round_number AND SessionName = :session_name
        LIMIT 1
    """)
    params = {"year": year, "round_number": round_number, "session_name": session_name}

    logger.debug(f"Checking existence with params: {params}")
    try:
        with DB_ENGINE.connect() as connection:
            result = connection.execute(query, params).scalar()
            exists = result is not None # If scalar returns 1 (or anything), it exists
            logger.debug(f"Existence check result for {params}: {exists} (Scalar: {result})")
            return exists
    except Exception as e:
        if "no such table" in str(e).lower():
             logger.warning(f"Table 'results' does not exist while checking data for {params}. Assuming data does not exist.")
             return False
        else:
             logger.error(f"Error checking data existence for {params}: {e}", exc_info=True)
             return False # Assume not exists on other errors