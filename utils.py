# /f1_predictor/utils.py

import logging
import config
import pandas as pd
import numpy as np
import os
import sys

def setup_logging():
    """Configures logging to file and console."""
    log_dir = os.path.dirname(config.LOG_FILE)
    try:
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        # Setup basic config
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.LOG_FILE, mode='a'), # Append mode
                logging.StreamHandler(sys.stdout) # Explicitly use stdout
            ],
            force=True # Override existing root logger config if any
        )
        # Silence overly verbose loggers
        logging.getLogger("fastf1").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        logging.info("Logging configured successfully.")

    except Exception as e:
        print(f"CRITICAL: Failed to configure logging: {e}", file=sys.stderr)
        # Optionally exit if logging is critical
        # sys.exit(1)


def get_logger(name):
    """Gets a logger instance configured by setup_logging."""
    return logging.getLogger(name)


def safe_to_numeric(series, fallback=None):
    """
    Safely converts a pandas Series to numeric.
    If fallback is provided (and not None), fills NaN values with it.
    If fallback is None, keeps NaN values resulting from coercion.
    """
    if series is None:
        return None

    # Convert to numeric, coercing errors to NaN
    num_series = pd.to_numeric(series, errors='coerce')

    # Only fill NaN if a fallback value (that is not None) is provided
    if fallback is not None:
        return num_series.fillna(fallback)
    else:
        # Otherwise, return the series with NaNs preserved
        return num_series

# Call setup logging once when the module is imported
# setup_logging() # Removed - Call explicitly in main.py after imports