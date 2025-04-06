# /f1_predictor/config.py
import os
import datetime # Added for dynamic year
# --- NEW: Added scipy for parameter distributions ---
from scipy.stats import randint, uniform

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
# Years for fetching historical data and training the model
HISTORICAL_YEARS = list(range(2021, datetime.datetime.now().year)) # e.g., [2021, 2022, 2023] if current year is 2024
# Current year for potentially updating ongoing season data
CURRENT_YEAR = datetime.datetime.now().year
# Default target year and race for prediction if not specified via args
TARGET_YEAR = datetime.datetime.now().year
TARGET_RACE_NAME = "Bahrain" # Or choose a sensible default

# --- Feature Engineering Parameters ---
# Number of past races for driver rolling performance features
N_LAST_RACES_FEATURES = 5
# Number of past races for team rolling performance features
N_LAST_RACES_TEAM = 10
# --- NEW: Window for rolling teammate comparison features ---
N_LAST_RACES_TEAMMATE_COMPARISON = 5
# Fallback grid/finish position for missing data or DNFs if needed during feature creation
WORST_EXPECTED_POS = 22 # Increased slightly
# General fill value for missing numeric data in features
FILL_NA_VALUE = -999.0
# Minimum laps for a stint to be considered valid for pace analysis
MIN_STINT_LAPS = 4
# Number of laps at start/end of stint to calculate simple degradation delta
DEGRADATION_LAP_COUNT = 3

# Track Characteristics Map (Location from Event Schedule -> General Type)
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
    'Portimão': 'Elevation Changes',
    'Unknown': 'Unknown'
}

# --- Model Parameters ---
MODEL_TYPE = 'XGBoost' # Options: 'RandomForest', 'XGBoost', 'LightGBM'

# --- Hyperparameter Tuning Parameters ---
# Set to True to enable RandomizedSearchCV, False to use fixed params below
ENABLE_TUNING = True
# Number of parameter settings to sample in RandomizedSearchCV
TUNING_N_ITER = 50 # Adjust based on desired search time (e.g., 30-100)
# Number of folds for TimeSeriesSplit during tuning and evaluation
CV_SPLITS = 5

# --- Fixed Model Parameters (Used if ENABLE_TUNING = False) ---
RF_PARAMS = {'n_estimators': 150, 'max_depth': 12, 'min_samples_leaf': 3, 'random_state': 42, 'n_jobs': -1}
XGB_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'objective': 'reg:squarederror'}
LGBM_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': -1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'objective': 'regression_l1', 'verbose': -1}

# --- Parameter Distributions for Tuning (Used if ENABLE_TUNING = True) ---
RF_PARAM_DIST = {
    'n_estimators': randint(100, 500),
    'max_depth': [None] + list(randint(5, 20).rvs(5)), # Sample a few depths + None
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 11),
    'max_features': ['sqrt', 'log2', None] + list(uniform(0.5, 0.5).rvs(3)) # Sample float ranges + fixed options
}
XGB_PARAM_DIST = {
    'n_estimators': randint(100, 700),
    'learning_rate': uniform(0.01, 0.2), # Search between 0.01 and 0.21
    'max_depth': randint(3, 10),
    'subsample': uniform(0.6, 0.4), # Search between 0.6 and 1.0
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': [0, 0.1, 0.5, 1],
    'reg_alpha': [0, 0.001, 0.01, 0.1],
    'reg_lambda': [1, 0.1, 0.01]
}
LGBM_PARAM_DIST = {
    'n_estimators': randint(100, 700),
    'learning_rate': uniform(0.01, 0.2),
    'num_leaves': randint(20, 60),
    'max_depth': [-1] + list(randint(5, 15).rvs(3)),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': [0, 0.001, 0.01, 0.1],
    'reg_lambda': [0, 0.01, 0.1, 1]
}


# --- Create directories ---
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir: os.makedirs(log_dir, exist_ok=True)
except OSError as e:
    print(f"Warning: Could not create directories: {e}")