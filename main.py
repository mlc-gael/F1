# /f1_predictor/main.py

import sys
import argparse
import os # Import os for checking model file

# --- Set up logging FIRST ---
# Import utils and immediately configure logging
try:
    import utils
    utils.setup_logging() # Configure logging system
    logger = utils.get_logger(__name__) # Get logger for main module
    logger.info("--- Application Start ---")
except Exception as e:
    # Fallback print if logging setup failed catastrophically
    print(f"FATAL: Logging setup failed: {e}", file=sys.stderr)
    sys.exit(1)

# --- Import other modules AFTER logging is setup ---
try:
    logger.info("Importing project modules...")
    import config
    import database
    import data_loader
    import feature_engineering
    import model as model_manager # Use alias to avoid name clash
    import predict
    logger.info("Project modules imported successfully.")
except ImportError as e:
    logger.error(f"FATAL: Failed to import a required module: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.error(f"FATAL: Unexpected error during module imports: {e}", exc_info=True)
    sys.exit(1)


def run_database_update():
    """Handles the database update process."""
    logger.info("--- Running Database Update Step ---")
    try:
        # Determine years to update (historical + current)
        years_to_update = list(range(min(config.HISTORICAL_YEARS), config.CURRENT_YEAR + 1))
        logger.info(f"Targeting database update for years: {years_to_update}")
        data_loader.update_database(years_to_update)
        logger.info("--- Database Update Finished ---")
    except Exception as e:
        logger.error(f"Error during database update: {e}", exc_info=True)


def run_training(force_train=False):
    """Handles the model training process."""
    logger.info("--- Running Model Training Step ---")
    model_exists = os.path.exists(config.MODEL_PATH)

    if not force_train and model_exists:
        logger.info(f"Skipping training: Model already exists at {config.MODEL_PATH}. Use --force-train to override.")
        return

    if force_train and model_exists:
        logger.info("Forcing retraining even though model exists.")

    try:
        # 1. Generate Features
        logger.info("Generating features for training...")
        # Ensure feature engineering uses only configured historical years for training set
        # Note: create_features() currently loads all R/Q data - needs filtering if
        # you want to strictly separate train/test years within the function.
        # For simplicity here, we load all features then filter.
        df_features_all, feature_cols, target_col = feature_engineering.create_features()

        if df_features_all is None or df_features_all.empty:
            logger.error("Model training skipped: Feature generation returned no data.")
            return
        if not feature_cols or target_col is None:
             logger.error("Model training skipped: Feature generation failed to define features/target.")
             return

        # Filter data strictly to historical years for training
        logger.info(f"Filtering features to training years: {config.HISTORICAL_YEARS}")
        training_features = df_features_all[df_features_all['Year'].isin(config.HISTORICAL_YEARS)].copy()

        if training_features.empty:
            logger.error(f"No data found within specified historical years {config.HISTORICAL_YEARS} for training.")
            return

        logger.info(f"Proceeding with training using {len(training_features)} data points.")

        # 2. Train Model
        trained_model = model_manager.train_model(training_features, feature_cols, target_col)

        if trained_model is None:
            logger.error("Model training process failed.")
        else:
            logger.info("--- Model Training Finished Successfully ---")

    except Exception as e:
        logger.error(f"Error during model training process: {e}", exc_info=True)


def run_prediction(predict_year, predict_race_id):
    """Handles the prediction process for a target race."""
    logger.info("--- Running Prediction Step ---")
    logger.info(f"Targeting prediction for: Year {predict_year}, Race Identifier '{predict_race_id}'")

    try:
        predictions = predict.make_predictions(predict_year, predict_race_id)

        if predictions is not None and not predictions.empty:
            print("\n" + "="*30)
            print("   Predicted Race Results")
            print("="*30)
            print(f"Race: {predict_year} {predict_race_id}")
            print("-" * 30)
            print(predictions.to_string(index=False))

            print("="*30 + "\n")
            logger.info("--- Prediction Finished Successfully ---")
        elif predictions is not None and predictions.empty:
             logger.warning("Prediction process completed but resulted in an empty prediction set.")
             logger.warning("--- Prediction Finished (No Results) ---")
        else:
            logger.error("Prediction process failed.")
            logger.error("--- Prediction Finished (FAILED) ---")

    except Exception as e:
        logger.error(f"Error during prediction process: {e}", exc_info=True)
        logger.error("--- Prediction Finished (FAILED) ---")


def main():
    """Parses arguments and runs the main application logic."""
    parser = argparse.ArgumentParser(description="F1 Race Result Predictor")
    parser.add_argument("--update-db", action="store_true", help="Fetch latest data and update the database.")
    parser.add_argument("--train", action="store_true", help="Train the prediction model using historical data.")
    parser.add_argument("--predict", action="store_true", help="Predict results for the target race (defined in config).")
    parser.add_argument("--year", type=int, default=config.TARGET_YEAR, help=f"Year for prediction (default: {config.TARGET_YEAR}).")
    parser.add_argument("--race", default=config.TARGET_RACE_NAME, help=f"Race Name or Round Number for prediction (default: {config.TARGET_RACE_NAME}).")
    parser.add_argument("--force-train", action="store_true", help="Force retraining even if a model file exists.")
    parser.add_argument("--all", action="store_true", help="Run all steps: update-db, train, and predict.")

    args = parser.parse_args()
    logger.info(f"Arguments received: {args}")

    # --- Ensure DB Tables Exist ---
    # Moved here to run once at the start after imports
    try:
        logger.info("Ensuring database tables exist before proceeding...")
        database.create_tables()
        logger.info("Database tables check/creation complete.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to ensure database tables exist: {e}", exc_info=True)
        logger.error("Exiting application.")
        sys.exit(1)


    # --- Workflow based on arguments ---
    if args.all:
        logger.info("Running all steps: Update DB -> Train -> Predict")
        run_database_update()
        run_training(force_train=args.force_train) # Respect force_train with --all
        run_prediction(args.year, args.race)
    else:
        if args.update_db:
            run_database_update()

        if args.train or args.force_train:
             # Run training if --train or --force-train is specified
            run_training(force_train=args.force_train)

        if args.predict:
            # Check if model exists before predicting if training wasn't forced/run
            if not os.path.exists(config.MODEL_PATH):
                 logger.warning(f"Prediction requested, but model file not found at {config.MODEL_PATH}. Run with --train first or ensure model exists.")
            else:
                 run_prediction(args.year, args.race)

    if not (args.all or args.update_db or args.train or args.force_train or args.predict):
        logger.warning("No action specified. Use --update-db, --train, --predict, or --all.")
        parser.print_help()

    logger.info("--- Application End ---")


if __name__ == "__main__":
    main()