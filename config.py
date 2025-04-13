# /f1_predictor/config.py
import os
import datetime
from scipy.stats import randint, uniform

# --- General Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'fastf1_cache')
DB_PATH = os.path.join(BASE_DIR, 'f1_data.sqlite')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'f1_predictor_model.joblib')
LOG_FILE = os.path.join(BASE_DIR, 'f1_predictor.log')
# --- NEW: Path for saving/loading engineered features (Feather format) ---
FEATURES_CACHE_PATH = os.path.join(BASE_DIR, 'engineered_features.feather')
# --- NEW: Path for saving model features and dtypes ---
MODEL_FEATURES_META_PATH = os.path.join(MODEL_DIR, 'model_features_meta.json')


# --- Data Loading Flags ---
LOAD_LAPS = True # Set to True to load lap-by-lap data (SLOW, LARGE DB)
LOAD_WEATHER = True # Set to True to load weather data (Slower)


# --- Data Parameters ---
# Define the range of years you *might* have data for
# Adjust start year if you have older data
ALL_AVAILABLE_YEARS = list(range(2021, datetime.datetime.now().year + 1))

# --- NEW: Define the Test Set Period ---
# Example: Use the single most recent COMPLETE year for testing
# Assumes the current year might be incomplete, so use year before current
# Adjust logic if needed (e.g., if current year is complete enough for testing)
CURRENT_YEAR = datetime.datetime.now().year
MOST_RECENT_FULL_YEAR = CURRENT_YEAR - 1

TEST_SET_YEARS = [MOST_RECENT_FULL_YEAR] if MOST_RECENT_FULL_YEAR in ALL_AVAILABLE_YEARS else []
if not TEST_SET_YEARS and ALL_AVAILABLE_YEARS: # Fallback if only 1 year of data exists
    TEST_SET_YEARS = [max(ALL_AVAILABLE_YEARS)]
    print(f"Warning: Only one year of data detected ({TEST_SET_YEARS[0]}). Using it as the test set. Training will likely fail.")

# Development years are all years *before* the test set start year
TEST_START_YEAR = min(TEST_SET_YEARS) if TEST_SET_YEARS else CURRENT_YEAR + 1 # Avoid errors if test set empty
DEV_YEARS = [y for y in ALL_AVAILABLE_YEARS if y < TEST_START_YEAR]

# Prediction Defaults (used if --predict is run without --year/--race)
TARGET_YEAR_DEFAULT = datetime.datetime.now().year
TARGET_RACE_NAME_DEFAULT = "Bahrain"


# --- Feature Engineering Parameters ---
N_LAST_RACES_FEATURES = 5
N_LAST_RACES_TEAM = 10
N_LAST_RACES_TEAMMATE_COMPARISON = 5
WORST_EXPECTED_POS = 22 # Used for initial DNF/Missing position handling
# Detailed fill values for NaNs encountered during prediction prep or final cleaning
FEATURE_FILL_DEFAULTS = {
    'default': -999.0, # Default fallback for unknown numeric features
    'Pts': 0.0,        # Points-related features
    'Points': 0.0,
    'Pos': float(WORST_EXPECTED_POS), # Position-related features
    'Position': float(WORST_EXPECTED_POS),
    'Rank': float(WORST_EXPECTED_POS + 5), # Standings fallback rank
    'Standing': float(WORST_EXPECTED_POS + 5),
    'Pace': 999.0,      # Pace features (higher is worse)
    'Deg': 5.0,         # Degradation fallback (high)
    'Temp': 20.0,       # Temperature fallback (Celsius)
    'RainProb': 0.1,    # Rain probability fallback (10%)
    'WindSpeed': 10.0,  # Wind speed fallback (generic km/h or m/s)
    'NumPits': 1.0,     # Pit stop fallback
    'Diff': 0.0,        # Difference features (e.g., teammate comparison)
    # OHE columns default to 0, handled separately in predict.py
}
# Grid Position Fallback Strategy during prediction if Quali/Official grid is missing
GRID_POS_FALLBACK_METHOD = 'RollingAvg' # Options: 'RollingAvg', 'MidPack', 'Worst'
GRID_POS_FALLBACK_MIDPACK_VALUE = 11.0 # Used if method is 'MidPack'

MIN_STINT_LAPS = 4 # Min laps for a valid stint in practice analysis
DEGRADATION_LAP_COUNT = 3 # Laps at start/end of stint for simple degradation calc


# Track Characteristics Map (Subjective, adjust as needed)
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
    'Unknown': 'Unknown' # Must have an Unknown fallback
}


# --- Model Parameters ---
MODEL_TYPE = 'XGBoost' # Options: 'RandomForest', 'XGBoost', 'LightGBM'


# --- Hyperparameter Tuning Parameters ---
ENABLE_TUNING = True # Set to True to enable RandomizedSearchCV
TUNING_N_ITER = 50 # Number of parameter settings to sample (adjust based on time)
CV_SPLITS = 5 # Number of folds for TimeSeriesSplit during tuning AND walk-forward eval
# Scorer to optimize during tuning. MAE is common. Rank metrics are harder to implement directly here.
TUNING_SCORER = 'neg_mean_absolute_error'


# --- Fixed Model Parameters (Used if ENABLE_TUNING = False or Tuning Fails) ---
RF_PARAMS = {'n_estimators': 150, 'max_depth': 12, 'min_samples_leaf': 3, 'random_state': 42, 'n_jobs': -1}
XGB_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'objective': 'reg:squarederror'}
LGBM_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': -1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'objective': 'regression_l1', 'verbose': -1}


# --- Parameter Distributions for Tuning (Used if ENABLE_TUNING = True) ---
# (Keep distributions as they were - these are examples, adjust ranges based on experience)
RF_PARAM_DIST = {
    'n_estimators': randint(100, 500),
    'max_depth': [None] + list(randint(5, 20).rvs(5)),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 11),
    'max_features': ['sqrt', 'log2', None] + list(uniform(0.5, 0.5).rvs(3))
}
XGB_PARAM_DIST = {
    'n_estimators': randint(100, 700),
    'learning_rate': uniform(0.01, 0.2), # Range: 0.01 to 0.21
    'max_depth': randint(3, 10),
    'subsample': uniform(0.6, 0.4), # Range: 0.6 to 1.0
    'colsample_bytree': uniform(0.6, 0.4), # Range: 0.6 to 1.0
    'gamma': [0, 0.1, 0.5, 1, 2], # Added more gamma options
    'reg_alpha': [0, 0.001, 0.01, 0.1, 0.5], # Added more alpha options
    'reg_lambda': [0, 0.1, 0.5, 1, 2] # Added more lambda options
}
LGBM_PARAM_DIST = {
    'n_estimators': randint(100, 700),
    'learning_rate': uniform(0.01, 0.2),
    'num_leaves': randint(20, 60),
    'max_depth': [-1] + list(randint(5, 15).rvs(3)),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': [0, 0.001, 0.01, 0.1, 0.5],
    'reg_lambda': [0, 0.01, 0.1, 0.5, 1]
}


# --- Create directories ---
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir: os.makedirs(log_dir, exist_ok=True)
    features_dir = os.path.dirname(FEATURES_CACHE_PATH)
    if features_dir and not os.path.exists(features_dir): os.makedirs(features_dir, exist_ok=True)
except OSError as e:
    print(f"Warning: Could not create directories: {e}")