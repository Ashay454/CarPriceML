import os
import time
import argparse
import logging
from src.data_preprocessing import load_data, clean_data, prepare_features, feature_engineering
from src.model import train_model, save_model
from src.plot import plot_results, plot_exploratory_data_analysis

DATA_PATH = os.path.join('data', 'car_data.csv')
OUTPUT_DIR = 'outputs'
MODEL_NAME = 'car_price_model.pkl'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main(args):
    start_time = time.time()
    logging.info("Starting ML Pipeline...")

    if not os.path.exists(DATA_PATH):
        logging.error(f"Data file not found at {DATA_PATH}. Please check your 'data' folder.")
        return

    df = load_data(DATA_PATH)
    if df is None: return

    logging.info("Preprocessing data...")
    df_clean = clean_data(df)

    if not args.skip_eda:
        logging.info("Generating EDA plots...")
        df_eda = feature_engineering(df_clean.copy())
        plot_exploratory_data_analysis(df_eda, save_dir=OUTPUT_DIR)
    else:
        logging.info("Skipping EDA generation (--skip-eda flag used).")

    logging.info("Training model...")
    X, y = prepare_features(df_clean)
    model, X_test, y_test, predictions = train_model(X, y)

    save_model(model, filename=MODEL_NAME)

    logging.info("Generating evaluation metrics...")
    plot_results(y_test, predictions, save_dir=OUTPUT_DIR)

    elapsed = time.time() - start_time
    logging.info(f"Pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Car Price Prediction Model")
    parser.add_argument('--skip-eda', action='store_true', help="Skip generating EDA plots to save time")
    
    args = parser.parse_args()
    main(args)