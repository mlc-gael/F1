# /f1_predictor/config.py
import os

# --- General Paths ---
# Use absolute path for BASE_DIR to avoid issues when running from different locations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Default cache inside the project directory. Change if needed.
CACHE_DIR = os.path.join(BASE_DIR, 'fastf1_cache')
DB_PATH = os.path.join(BASE_DIR, 'f1_data.sqlite')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'f1_predictor_model.joblib')
LOG_FILE = os.path.join(BASE_DIR, 'f1_predictor.log')

# --- Data Parameters ---
# Years to fetch historical data FOR TRAINING
HISTORICAL_YEARS = [2022, 2023, 2024] # Adjust as needed
# The most recent COMPLETE year (or year up to which data is available for fetching)
# Used to determine the range for data loading if needed.
CURRENT_YEAR = 2025 # Adjust if needed 

# --- Prediction Parameters ---
TARGET_YEAR = 2025          # Year of the race to predict
TARGET_RACE_NAME = "Bahrain" # Name or round number of the race to predict

# --- Feature Engineering Parameters ---
N_LAST_RACES_FEATURES = 5    # Rolling window for driver features
N_LAST_RACES_TEAM = 10       # Rolling window for team features
WORST_EXPECTED_POS = 20     # Fallback value for missing/DNF positions
FILL_NA_VALUE = -999.0       # A distinct float value for numerical NaNs

# Track Characteristics (Expand this dictionary!)
TRACK_CHARACTERISTICS = {
    # From FastF1 event['Location'] - these might need verification/updates
    'Sakhir': 'Balanced', # Bahrain International Circuit
    'Jeddah': 'High Speed Street', # Jeddah Street Circuit
    'Melbourne': 'Street Circuit', # Albert Park Circuit
    'Baku': 'Street Circuit', # Baku City Circuit
    'Miami': 'High Speed Street', # Miami International Autodrome
    'Imola': 'Old School', # Autodromo Enzo e Dino Ferrari
    'Monaco': 'Street Circuit Low Speed', # Circuit de Monaco
    'Catalunya': 'Balanced High Deg', # Circuit de Barcelona-Catalunya
    'Montréal': 'Stop-Go', # Circuit Gilles Villeneuve (Note accents might differ)
    'Spielberg': 'Short Lap High Speed', # Red Bull Ring
    'Silverstone': 'High Speed', # Silverstone Circuit
    'Budapest': 'Technical High Downforce', # Hungaroring
    'Spa-Francorchamps': 'High Speed Elevation', # Spa-Francorchamps
    'Zandvoort': 'Old School Banked', # Zandvoort Circuit
    'Monza': 'Temple of Speed', # Autodromo Nazionale Monza
    'Marina Bay': 'Street Circuit High Downforce', # Marina Bay Street Circuit
    'Suzuka': 'Technical Figure Eight', # Suzuka International Racing Course
    'Lusail': 'High Speed Flowing', # Lusail International Circuit
    'Austin': 'Modern Mix', # Circuit of the Americas
    'Mexico City': 'High Altitude', # Autódromo Hermanos Rodríguez
    'São Paulo': 'Anti-Clockwise Elevation', # Interlagos Circuit / Autódromo José Carlos Pace
    'Las Vegas': 'High Speed Street', # Las Vegas Strip Circuit
    'Yas Island': 'Modern Mix', # Yas Marina Circuit
    # Add more as needed... Lookup Event['Location'] from fastf1 schedule
    'Shanghai': 'Modern Mix', # Shanghai International Circuit
    # Fallback for unknown tracks
    'Unknown': 'Unknown'
}

# --- Model Parameters ---
MODEL_TYPE = 'LightGBM' # Options: 'RandomForest', 'XGBoost', 'LightGBM'
# Parameters for RandomForest
RF_PARAMS = {'n_estimators': 150, 'max_depth': 12, 'min_samples_leaf': 3, 'random_state': 42, 'n_jobs': -1}
# Parameters for XGBoost
XGB_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'objective': 'reg:squarederror'} # Specify objective
# Parameters for LightGBM
LGBM_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': -1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'objective': 'regression_l1'} # Specify objective (e.g., MAE)


# --- Create directories if they don't exist ---
# This runs when config.py is imported. Make sure paths are defined above.
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Create log directory separately in utils to ensure logger is setup first
except OSError as e:
    print(f"Warning: Could not create directories: {e}")