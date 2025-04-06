# /f1_predictor/utils.py

import logging
import config # Now used for track coordinates maybe
import pandas as pd
import numpy as np
import os
import sys
import requests # For API calls
import json     # For parsing API response
import datetime # For forecast date handling

# --- Logging Setup ---
def setup_logging():
    """Configures logging to file and console."""
    log_dir = os.path.dirname(config.LOG_FILE)
    try:
        if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig( level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[ logging.FileHandler(config.LOG_FILE, mode='a'), logging.StreamHandler(sys.stdout) ], force=True )
        # Silence overly verbose loggers
        logging.getLogger("fastf1").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.info("Logging configured successfully.")
    except Exception as e:
        print(f"CRITICAL: Failed to configure logging: {e}", file=sys.stderr)

def get_logger(name):
    """Gets a logger instance configured by setup_logging."""
    return logging.getLogger(name)

logger = get_logger(__name__) # Logger for utils module

# --- Data Cleaning & Conversion ---
def safe_to_numeric(series, fallback=None):
    """Safely converts a pandas Series to numeric, handling fallback for NaNs."""
    if series is None: return None
    num_series = pd.to_numeric(series, errors='coerce')
    if fallback is not None: return num_series.fillna(fallback)
    else: return num_series

def parse_timedelta_to_seconds(time_obj):
    """Converts Pandas Timedelta to seconds (float) or returns np.nan."""
    if pd.isna(time_obj): return np.nan
    try:
        if isinstance(time_obj, (int, float)): return float(time_obj)
        if isinstance(time_obj, pd.Timedelta): return time_obj.total_seconds()
        return np.nan
    except (AttributeError, ValueError, TypeError): return np.nan


# --- Track Coordinates (Needs to be comprehensive!) ---
TRACK_COORDINATES = {
    # Location Name from Event Schedule -> (Latitude, Longitude)
    'Sakhir': (26.0325, 50.5104), 'Jeddah': (21.6319, 39.1044),
    'Melbourne': (-37.8497, 144.9680), 'Baku': (40.3725, 49.8533),
    'Miami': (25.9581, -80.2389), 'Imola': (44.3439, 11.7167),
    'Monaco': (43.7347, 7.4206), 'Catalunya': (41.5700, 2.2611),
    'Montréal': (45.5000, -73.5228), 'Spielberg': (47.2197, 14.7647),
    'Silverstone': (52.0783, -1.0169), 'Budapest': (47.5789, 19.2486),
    'Spa-Francorchamps': (50.4372, 5.9714), 'Zandvoort': (52.3888, 4.5409),
    'Monza': (45.6156, 9.2811), 'Marina Bay': (1.2914, 103.8640),
    'Suzuka': (34.8431, 136.5411), 'Lusail': (25.4900, 51.4542),
    'Austin': (30.1328, -97.6411), 'Mexico City': (19.4042, -99.0907),
    'São Paulo': (-23.7036, -46.6997), 'Las Vegas': (36.1146, -115.1728),
    'Yas Island': (24.4672, 54.6031), 'Shanghai': (31.3389, 121.2200),
    # Add more as needed...
}


# --- Weather Forecast Function (using Open-Meteo) ---
def get_weather_forecast(latitude, longitude):
    """
    Gets daily weather forecast for the race location using Open-Meteo API.

    Args:
        latitude (float): Latitude of the track.
        longitude (float): Longitude of the track.

    Returns:
        dict: Dictionary with forecast features (ForecastTemp, ForecastRainProb,
              ForecastWindSpeed) or None if fetch fails.
    """
    if latitude is None or longitude is None or np.isnan(latitude) or np.isnan(longitude):
         logger.warning("Missing or invalid latitude/longitude for forecast lookup.")
         return None

    logger.info(f"Fetching weather forecast for Lat: {latitude:.4f}, Lon: {longitude:.4f} using Open-Meteo...")

    # Open-Meteo API endpoint and parameters
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    # Ask for relevant daily variables
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'daily': 'temperature_2m_max,precipitation_probability_max,wind_speed_10m_max',
        'timezone': 'UTC',
        'forecast_days': 3 # Get a few days to cover potential race day variations
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=15) # Use a reasonable timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4XX, 5XX)
        data = response.json()
        logger.debug(f"Open-Meteo Raw Response: {json.dumps(data, indent=2)}") # Log raw response for debugging

        # --- Extract Data ---
        # Use the forecast for the *first* day provided by the API (index 0).
        # A more advanced approach could align this with the actual event date.
        if 'daily' in data and isinstance(data['daily'], dict) and 'time' in data['daily'] and len(data['daily']['time']) > 0:
            try:
                idx = 0 # Index for the first forecast day

                # Helper to safely get value from list or return NaN
                def get_metric(key, index, default=np.nan):
                    val_list = data.get('daily', {}).get(key)
                    if isinstance(val_list, list) and len(val_list) > index:
                        return val_list[index]
                    return default

                temp = get_metric('temperature_2m_max', idx)
                rain_prob = get_metric('precipitation_probability_max', idx)
                wind_speed = get_metric('wind_speed_10m_max', idx)

                # Convert rain probability from % to fraction (0-1)
                if rain_prob is not None and not np.isnan(rain_prob):
                    rain_prob = float(rain_prob) / 100.0
                else:
                     rain_prob = np.nan # Ensure it's NaN if None or already NaN

                # Create forecast dict, ensuring float type where applicable
                forecast = {
                    'ForecastTemp': float(temp) if temp is not None and not np.isnan(temp) else np.nan,
                    'ForecastRainProb': float(rain_prob) if rain_prob is not None and not np.isnan(rain_prob) else np.nan,
                    'ForecastWindSpeed': float(wind_speed) if wind_speed is not None and not np.isnan(wind_speed) else np.nan
                }

                # Filter out entries that are still NaN after processing
                final_forecast = {k: v for k, v in forecast.items() if not pd.isna(v)}

                if not final_forecast: # If all extractions failed or resulted in NaN
                     logger.warning("Open-Meteo response parsed, but failed to extract valid forecast metrics.")
                     return None

                logger.info(f"Successfully fetched and parsed Open-Meteo forecast: {final_forecast}")
                return final_forecast

            except (IndexError, KeyError, TypeError, ValueError) as e:
                 logger.error(f"Error parsing Open-Meteo daily data structure: {e}")
                 logger.debug(f"Received daily data structure: {data.get('daily')}")
                 return None
        else:
             logger.warning("Forecast data received from Open-Meteo, but 'daily' structure invalid or empty.")
             logger.debug(f"Received data: {data}")
             return None

    except requests.exceptions.Timeout:
         logger.error("Weather API request timed out.")
         return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
         logger.error(f"Failed to decode weather API response: {e}")
         return None
    except Exception as e:
         logger.error(f"Unexpected error during weather forecast fetch: {e}", exc_info=True)
         return None