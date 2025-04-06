# /f1_predictor/config.py
import os

# --- General Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'fastf1_cache')
DB_PATH = os.path.join(BASE_DIR, 'f1_data.sqlite')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'f1_predictor_model.joblib')
LOG_FILE = os.path.join(BASE_DIR, 'f1_predictor.log')

# --- Data Loading Flags ---
LOAD_LAPS = True # Set to True to load lap-by-lap data (SLOW, LARGE DB)
LOAD_WEATHER = True # Set to True to load weather data (Slower)
# Telemetry loading deferred (LOAD_TELEMETRY = False)

# --- Data Parameters ---
HISTORICAL_YEARS = [2021, 2022, 2023, 2024]
CURRENT_YEAR = 2025
TARGET_YEAR = 2025
TARGET_RACE_NAME = "Bahrain"

# --- Feature Engineering Parameters ---
N_LAST_RACES_FEATURES = 5
N_LAST_RACES_TEAM = 10
WORST_EXPECTED_POS = 20
FILL_NA_VALUE = -999.0
# Parameters for Lap/Stint Analysis
MIN_STINT_LAPS = 4 # Minimum laps for a stint to be considered for pace analysis
DEGRADATION_LAP_COUNT = 3 # Number of laps at start/end of stint to calculate simple degradation delta

# Track Characteristics Map (Location -> Type)
TRACK_CHARACTERISTICS = {
    'Sakhir': 'Balanced', 'Jeddah': 'High Speed Street', 'Melbourne': 'Street Circuit',
    'Baku': 'Street Circuit', 'Miami': 'High Speed Street', 'Imola': 'Old School',
    'Monaco': 'Street Circuit Low Speed', 'Catalunya': 'Balanced High Deg',
    'Montréal': 'Stop-Go', 'Spielberg': 'Short Lap High Speed', 'Silverstone': 'High Speed',
    'Budapest': 'Technical High Downforce', 'Spa-Francorchamps': 'High Speed Elevation',
    'Zandvoort': 'Old School Banked', 'Monza': 'Temple of Speed',
    'Marina Bay': 'Street Circuit High Downforce', 'Suzuka': 'Technical Figure Eight',
    'Lusail': 'High Speed Flowing', 'Austin': 'Modern Mix', 'Mexico City': 'High Altitude',
    'São Paulo': 'Anti-Clockwise Elevation', 'Las Vegas': 'High Speed Street',
    'Yas Island': 'Modern Mix', 'Shanghai': 'Modern Mix',
    'Unknown': 'Unknown'
}

# --- Model Parameters ---
MODEL_TYPE = 'RandomForest' # Options: 'RandomForest', 'XGBoost', 'LightGBM'
RF_PARAMS = {'n_estimators': 150, 'max_depth': 12, 'min_samples_leaf': 3, 'random_state': 42, 'n_jobs': -1}
XGB_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'objective': 'reg:squarederror'}
LGBM_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': -1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'objective': 'regression_l1'}


# --- Create directories ---
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
except OSError as e:
    print(f"Warning: Could not create directories: {e}")