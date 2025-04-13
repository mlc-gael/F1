# /f1_predictor/main.py

import sys
import argparse
import os
import pandas as pd
import numpy as np # Added for potential NaN checks

# --- Setup Logging First ---
try:
    import utils
    utils.setup_logging()
    logger = utils.get_logger(__name__)
    logger.info("--- Application Start ---")
except Exception as e:
    print(f"FATAL: Logging setup failed: {e}", file=sys.stderr)
    sys.exit(1)

# --- Import Other Modules ---
try:
    logger.info("Importing project modules...")
    import config
    import database
    import data_loader
    import feature_engineering
    import model as model_manager
    import predict # Keep predict import for its helpers like get_feature_fill_value
    # Import evaluation tools
    from sklearn.metrics import mean_absolute_error
    from scipy.stats import spearmanr, kendalltau
    logger.info("Project modules imported successfully.")
except ImportError as e:
    logger.error(f"FATAL: Failed to import a required module: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.error(f"FATAL: Unexpected error during module imports: {e}", exc_info=True)
    sys.exit(1)

# --- Feature Cache Dependency Check ---
try:
    import pyarrow
except ImportError:
    logger.warning("'pyarrow' not installed. Saving/loading features cache (Feather format) will fail.")


# --- Workflow Functions ---

def run_database_update():
    """Handles the database update process."""
    logger.info("--- Running Database Update Step ---")
    try:
        # Update data for all potentially relevant years
        years_to_update = config.ALL_AVAILABLE_YEARS
        logger.info(f"Targeting database update for years: {years_to_update}")
        data_loader.update_database(years_to_update)
        logger.info("--- Database Update Finished ---")
    except Exception as e:
        logger.error(f"Error during database update: {e}", exc_info=True)

def run_feature_generation(save=True):
    """Handles feature generation and optionally saves the full feature set."""
    logger.info("--- Running Feature Generation Step ---")
    try:
        df_features_all, feature_cols, target_col = feature_engineering.create_features()

        if df_features_all is None or df_features_all.empty:
            logger.error("Feature generation returned no data.")
            return None, [], None
        if not feature_cols or target_col is None:
             logger.error("Feature generation failed to define features/target.")
             return None, [], None

        logger.info(f"Feature generation successful. Shape: {df_features_all.shape}")

        if save:
            logger.info(f"Saving full feature set to {config.FEATURES_CACHE_PATH}...")
            try:
                df_features_all.reset_index(drop=True).to_feather(config.FEATURES_CACHE_PATH)
                logger.info("Full feature set saved successfully.")
            except NameError:
                 logger.error("Cannot save features: 'pyarrow' library not found. Install it (`pip install pyarrow`).")
            except Exception as e:
                logger.error(f"Error saving features to {config.FEATURES_CACHE_PATH}: {e}", exc_info=True)
                logger.warning("Proceeding without saved features cache.")

        return df_features_all, feature_cols, target_col

    except Exception as e:
        logger.error(f"Error during feature generation process: {e}", exc_info=True)
        return None, [], None

def evaluate_model_on_test_set(model, df_test, feature_cols, target_col):
    """Evaluates the trained model on the held-out test set."""
    logger.info(f"--- Evaluating Model on Test Set ({len(df_test)} samples) ---")

    if df_test is None or df_test.empty:
        logger.error("Cannot evaluate: Test DataFrame is empty.")
        return False # Indicate failure

    if model is None:
        logger.error("Cannot evaluate: Model object is None.")
        return False

    # --- Prepare Test Data ---
    try:
        X_test = df_test[feature_cols].copy() # Explicit copy
        y_test = df_test[target_col].copy()

        # Final check for NaNs/Infs in test features
        numeric_cols_test = X_test.select_dtypes(include=np.number).columns
        if not numeric_cols_test.empty:
            nan_mask = X_test[numeric_cols_test].isnull()
            if nan_mask.values.any():
                nan_cols = numeric_cols_test[nan_mask.any()].tolist()
                logger.warning(f"NaNs detected in TEST set features: {nan_cols}. Filling with defaults before prediction.")
                for col in nan_cols:
                    X_test[col].fillna(predict.get_feature_fill_value(col), inplace=True)

            inf_mask = np.isinf(X_test[numeric_cols_test])
            if inf_mask.values.any():
                 inf_cols = numeric_cols_test[inf_mask.any()].tolist()
                 logger.warning(f"Infs detected in TEST set features: {inf_cols}. Replacing.")
                 X_test[numeric_cols_test] = X_test[numeric_cols_test].replace([np.inf, -np.inf], config.FEATURE_FILL_DEFAULTS['default'] * 100)

        if y_test.isnull().any():
             logger.warning(f"NaNs found in TEST target ('{target_col}'). Evaluation metrics might be affected if not handled by metric function.")

    except KeyError as e:
         logger.error(f"KeyError preparing test data. Missing columns? {e}. Features expected: {feature_cols}. Available: {df_test.columns.tolist()}")
         return False
    except Exception as e:
         logger.error(f"Unexpected error preparing test data: {e}", exc_info=True)
         return False

    # --- Make Predictions ---
    try:
        test_preds = model.predict(X_test)
    except Exception as e:
        logger.error(f"Error predicting on test set: {e}", exc_info=True)
        return False

    # --- Calculate Metrics ---
    try:
        # Filter y_test and test_preds for valid pairs if y_test had NaNs originally
        valid_eval_indices = y_test.notna()
        y_test_valid = y_test[valid_eval_indices]
        test_preds_valid = test_preds[valid_eval_indices]

        if len(y_test_valid) == 0:
             logger.error("No valid target values in the test set remain for evaluation.")
             return False

        # MAE
        mae = mean_absolute_error(y_test_valid, test_preds_valid)
        logger.info(f"Test Set MAE: {mae:.4f}")

        # Rank Correlation (using helper from model module)
        # Create temporary DataFrame for ranking aligned with valid targets
        eval_df = df_test.loc[valid_eval_indices, ['Year', 'RoundNumber', 'Abbreviation']].copy()
        eval_df['TruePosition'] = y_test_valid
        eval_df['PredictedPosition_Raw'] = test_preds_valid
        test_spearman, test_kendall = model_manager.calculate_rank_correlation(eval_df, 'TruePosition', 'PredictedPosition_Raw')

        logger.info(f"Test Set Overall Rank Correlation:")
        logger.info(f"  Spearman's Rho: {test_spearman:.4f}") # P-value omitted for brevity
        logger.info(f"  Kendall's Tau:  {test_kendall:.4f}")

        # Optional: Per-race average rank correlation (more robust to outliers in single races)
        try:
            grouped_ranks = eval_df.groupby(['Year', 'RoundNumber'])
            race_corrs_spearman = grouped_ranks.apply(
                lambda g: model_manager.calculate_rank_correlation(g)[0] if len(g) > 1 else np.nan
            ).dropna()
            race_corrs_kendall = grouped_ranks.apply(
                 lambda g: model_manager.calculate_rank_correlation(g)[1] if len(g) > 1 else np.nan
            ).dropna()

            if not race_corrs_spearman.empty:
                 avg_spearman = race_corrs_spearman.mean()
                 std_spearman = race_corrs_spearman.std()
                 logger.info(f"Test Set Avg. Per-Race Spearman's Rho: {avg_spearman:.4f} (+/- {std_spearman:.4f})")
            if not race_corrs_kendall.empty:
                 avg_kendall = race_corrs_kendall.mean()
                 std_kendall = race_corrs_kendall.std()
                 logger.info(f"Test Set Avg. Per-Race Kendall's Tau:  {avg_kendall:.4f} (+/- {std_kendall:.4f})")
        except Exception as e_rank_avg:
             logger.warning(f"Could not calculate per-race average rank correlations: {e_rank_avg}")

    except Exception as e_metrics:
        logger.error(f"Failed to calculate evaluation metrics on test set: {e_metrics}", exc_info=True)
        return False # Indicate evaluation had issues

    logger.info("--- Test Set Evaluation Finished ---")
    return True # Indicate success


def run_training(force_train=False):
    """Handles model training on dev set and evaluates on test set."""
    logger.info("--- Running Model Training & Evaluation Step ---")
    model_exists = os.path.exists(config.MODEL_PATH)
    meta_exists = os.path.exists(config.MODEL_FEATURES_META_PATH)
    features_cache_exists = os.path.exists(config.FEATURES_CACHE_PATH)

    # --- Check if Training is Needed ---
    if not force_train and model_exists and meta_exists:
        logger.info(f"Skipping training: Model and metadata files already exist at {config.MODEL_DIR}.")
        if not features_cache_exists:
             logger.warning(f"Model exists, but features cache ({config.FEATURES_CACHE_PATH}) not found. Running feature generation...")
             run_feature_generation(save=True) # Generate cache if model is reused
        return # Skip training and evaluation

    if force_train and model_exists:
        logger.info("Forcing retraining even though model/meta files exist.")

    # --- 1. Generate/Load Features ---
    df_features_all = None
    feature_cols = None
    target_col = 'Position' # Standard target

    if not force_train and features_cache_exists:
        logger.info(f"Loading features from cache: {config.FEATURES_CACHE_PATH}")
        try:
            df_features_all = pd.read_feather(config.FEATURES_CACHE_PATH)
            if 'Year' not in df_features_all.columns or 'RoundNumber' not in df_features_all.columns:
                 raise ValueError("Loaded features cache is missing Year/RoundNumber.")
            logger.info(f"Features loaded successfully from cache ({len(df_features_all)} rows).")
            # Load feature cols from meta file if available
            if meta_exists:
                _, loaded_feature_cols, _ = model_manager.load_model_and_meta()
                if loaded_feature_cols:
                     feature_cols = loaded_feature_cols
                     logger.info("Feature list loaded from existing metadata.")
                else: logger.warning("Metadata exists but failed to load features. Will regenerate.")
            else: logger.warning("Feature cache exists but no metadata file found. Features will be defined during generation if forced.")

        except Exception as e:
            logger.warning(f"Failed to load features cache ({e}). Will regenerate.")
            df_features_all = None
            feature_cols = None # Reset feature cols if loading failed

    # Generate if not loaded or if forced
    if df_features_all is None or force_train:
        logger.info("Generating features (required or forced)...")
        df_features_all, feature_cols_gen, target_col_gen = run_feature_generation(save=True)
        if df_features_all is None:
            logger.error("Model training skipped: Feature generation failed.")
            return
        # Use newly generated feature list and target
        feature_cols = feature_cols_gen
        target_col = target_col_gen

    # Final check if we have features
    if df_features_all is None or not feature_cols or not target_col:
        logger.error("Could not obtain feature data or definition. Aborting training.")
        return

    # --- 2. Split Data: Development vs Test ---
    logger.info(f"Splitting data into Development ({config.DEV_YEARS}) and Test ({config.TEST_SET_YEARS}) sets.")
    if not config.TEST_SET_YEARS:
        logger.error("No TEST_SET_YEARS defined in config. Cannot perform train-test split evaluation.")
        return
    if not config.DEV_YEARS:
        logger.error("No DEV_YEARS defined in config (all years are in test set?). Cannot train.")
        return

    df_dev = df_features_all[df_features_all['Year'].isin(config.DEV_YEARS)].copy()
    df_test = df_features_all[df_features_all['Year'].isin(config.TEST_SET_YEARS)].copy()

    if df_dev.empty:
        logger.error(f"No data found for Development Years: {config.DEV_YEARS}. Cannot train.")
        return
    if df_test.empty:
        logger.warning(f"No data found for Test Years: {config.TEST_SET_YEARS}. Evaluation will be skipped.")

    logger.info(f"Development set size: {len(df_dev)}")
    logger.info(f"Test set size: {len(df_test)}")

    # --- 3. Train Model (includes internal Dev CV evaluation) ---
    final_trained_model = model_manager.train_model(df_dev, feature_cols, target_col)

    if final_trained_model is None:
        logger.error("Model training process failed.")
        return

    logger.info("--- Model Training on Development Set Finished Successfully ---")

    # --- 4. Evaluate Final Model on Test Set ---
    if not df_test.empty:
        evaluation_success = evaluate_model_on_test_set(final_trained_model, df_test, feature_cols, target_col)
        if not evaluation_success:
             logger.error("Evaluation on test set failed. Check logs.")
    else:
        logger.warning("Skipping final evaluation as Test Set is empty.")


def run_prediction(predict_year, predict_race_id):
    """Handles the prediction process for a target race."""
    logger.info("--- Running Prediction Step ---")
    logger.info(f"Targeting prediction for: Year {predict_year}, Race Identifier '{predict_race_id}'")

    # --- Check required files BEFORE prediction attempt ---
    model_files_exist = os.path.exists(config.MODEL_PATH) and os.path.exists(config.MODEL_FEATURES_META_PATH)
    features_cache_exists = os.path.exists(config.FEATURES_CACHE_PATH)

    if not model_files_exist:
         logger.error(f"Prediction failed: Model file ({config.MODEL_PATH}) or metadata ({config.MODEL_FEATURES_META_PATH}) not found. Run --train first.")
         return
    if not features_cache_exists:
         logger.error(f"Prediction failed: Features cache ({config.FEATURES_CACHE_PATH}) not found. Run --train or --generate-features first.")
         return

    try:
        predictions = predict.make_predictions(predict_year, predict_race_id)

        if predictions is not None and not predictions.empty:
            print("\n" + "="*40)
            print("      Predicted Race Results")
            print("="*40)
            # Get target race name for display if possible
            r_num, r_loc, r_name, r_date = predict.get_target_race_info(predict_year, predict_race_id)
            race_display_name = r_name if r_name else predict_race_id
            print(f"Race: {predict_year} {race_display_name}")
            print("-" * 40)
            # Adjust display format
            print(predictions.to_string(index=False, justify='center', float_format='%.2f'))
            print("="*40 + "\n")
            logger.info("--- Prediction Finished Successfully ---")
        elif predictions is not None and predictions.empty:
             logger.warning("Prediction process completed but resulted in an empty prediction set.")
             logger.warning("--- Prediction Finished (No Results) ---")
        else:
            # make_predictions logs errors internally
            logger.error("--- Prediction Finished (FAILED) ---")

    except Exception as e:
        logger.error(f"Error during prediction process: {e}", exc_info=True)
        logger.error("--- Prediction Finished (FAILED) ---")


def main():
    """Parses arguments and runs the main application logic."""
    parser = argparse.ArgumentParser(description="F1 Race Result Predictor")
    parser.add_argument("--update-db", action="store_true", help="Fetch latest data and update the database for all configured years.")
    parser.add_argument("--generate-features", action="store_true", help="Generate and save the features cache file from DB data.")
    parser.add_argument("--train", action="store_true", help="Train model on Development set & evaluate on Test set.")
    parser.add_argument("--predict", action="store_true", help="Predict results for a target race using the trained model.")
    parser.add_argument("--year", type=int, default=config.TARGET_YEAR_DEFAULT, help=f"Year for prediction (default: {config.TARGET_YEAR_DEFAULT}).")
    parser.add_argument("--race", default=config.TARGET_RACE_NAME_DEFAULT, help=f"Race Name or Round Number for prediction (default: {config.TARGET_RACE_NAME_DEFAULT}).")
    parser.add_argument("--force-train", action="store_true", help="Force feature generation and retraining, ignoring existing files.")
    parser.add_argument("--all", action="store_true", help="Run common workflow: update-db, train & evaluate, predict.")

    args = parser.parse_args()
    logger.info(f"Arguments received: {args}")

    # --- Ensure DB Tables Exist ---
    try:
        logger.info("Ensuring database tables exist before proceeding...")
        database.create_tables()
        logger.info("Database tables check/creation complete.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to ensure database tables exist: {e}", exc_info=True)
        logger.error("Exiting application.")
        sys.exit(1)

    # --- Workflow ---
    if args.all:
        logger.info("Running --all workflow: Update DB -> Train & Evaluate -> Predict")
        run_database_update()
        run_training(force_train=args.force_train) # Train now includes evaluation
        # Prediction uses the model trained on Dev set
        run_prediction(args.year, args.race)
    else:
        # Run individual steps if specified
        if args.update_db:
            run_database_update()

        if args.generate_features:
             run_feature_generation(save=True) # Explicit feature generation

        # Handle training & evaluation (triggered by --train or --force-train)
        if args.train or args.force_train:
            run_training(force_train=args.force_train)

        # Handle prediction
        if args.predict:
            run_prediction(args.year, args.race)

    # Check if any action was taken if --all wasn't used
    action_flags = [args.update_db, args.generate_features, args.train, args.predict]
    if not args.all and not any(action_flags):
        # Allow --force-train alone to trigger training
        if args.force_train:
             logger.info("Running training & evaluation due to --force-train flag.")
             run_training(force_train=True)
        else:
             logger.warning("No action specified. Use --update-db, --generate-features, --train, --predict, or --all.")
             parser.print_help()

    logger.info("--- Application End ---")

if __name__ == "__main__":
    main()