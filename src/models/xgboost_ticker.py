# src/models/xgboost_baseline_per_ticker.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import argparse
from src.config import XGB_PARAMS, TICKERS # Import TICKERS as well

ALIGNED_RAW_TEXT_DAILY_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "aligned_raw_text_daily_history"
RESULTS_DIR_PER_TICKER = Path(__file__).resolve().parent.parent.parent / "results" / "xgboost_baseline_per_ticker"

def train_evaluate_xgboost_per_ticker(aligned_data_path, target_lag_days=1, history_lags_days=[1,2,3,5]):
    RESULTS_DIR_PER_TICKER.mkdir(parents=True, exist_ok=True)
    try:
        df_all_tickers = pd.read_csv(aligned_data_path)
    except FileNotFoundError:
        print(f"Aligned data file not found: {aligned_data_path}")
        return

    print(f"Loaded data with shape: {df_all_tickers.shape}")

    current_day_features = [
        'close_price_day_D', 'open_price_day_D', 'high_price_day_D',
        'low_price_day_D', 'volume_day_D', 'daily_return_Day_D'
    ]
    lagged_features = []
    for lag in history_lags_days:
        lagged_features.extend([
            f'close_price_day_D_minus_{lag}',
            f'daily_return_Day_D_minus_{lag}',
            f'volume_Day_D_minus_{lag}'
        ])
    feature_columns = current_day_features + lagged_features
    target_column = f'target_direction_D_plus_{target_lag_days}'

    overall_metrics = []

    for ticker in TICKERS: # Iterate through each ticker defined in config
        print(f"\n--- Processing and training for Ticker: {ticker} ---")
        df_ticker = df_all_tickers[df_all_tickers['ticker'] == ticker].copy()

        if df_ticker.empty:
            print(f"No data for ticker {ticker}. Skipping.")
            continue

        # Ensure all selected feature columns and target column exist
        missing_cols = [col for col in feature_columns if col not in df_ticker.columns]
        if missing_cols:
            print(f"Ticker {ticker}: Missing feature columns: {missing_cols}. Skipping.")
            continue
        if target_column not in df_ticker.columns:
            print(f"Ticker {ticker}: Target column '{target_column}' not found. Skipping.")
            continue

        df_ticker.dropna(subset=feature_columns + [target_column], inplace=True)
        if df_ticker.empty:
            print(f"Ticker {ticker}: DataFrame empty after dropping NaNs. Skipping.")
            continue

        print(f"Ticker {ticker}: Shape after NaN drop: {df_ticker.shape}")
        
        df_ticker['date_D'] = pd.to_datetime(df_ticker['date_D'])
        df_ticker_sorted = df_ticker.sort_values(by='date_D').reset_index(drop=True)

        X = df_ticker_sorted[feature_columns]
        y = df_ticker_sorted[target_column].astype(int)

        if X.empty or y.empty or len(y.unique()) < 2:
            print(f"Ticker {ticker}: Feature set X or target y is empty or has <2 classes. Skipping.")
            continue
            
        print(f"Ticker {ticker}: Value counts for target y:\n{y.value_counts(normalize=True, dropna=False)}")

        # Chronological Train/Test Split (80/20) for this ticker
        if len(df_ticker_sorted) < 10 : # Arbitrary small number, adjust as needed
            print(f"Ticker {ticker}: Not enough data points ({len(df_ticker_sorted)}) for a meaningful train/test split. Skipping.")
            continue

        train_size = int(0.8 * len(df_ticker_sorted))
        if train_size == 0 or train_size == len(df_ticker_sorted): # Handles cases where 80% is 0 or all data
            print(f"Ticker {ticker}: Train size is 0 or full dataset ({train_size} of {len(df_ticker_sorted)}). Cannot split. Skipping.")
            continue
            
        X_train_df = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_test_df = X.iloc[train_size:]
        y_test = y.iloc[train_size:]

        if X_train_df.empty or X_test_df.empty or len(y_train.unique()) < 2 or len(y_test) == 0:
            print(f"Ticker {ticker}: Train or test set empty or has <2 classes in train target after split. Skipping.")
            continue

        print(f"Ticker {ticker}: X_train shape: {X_train_df.shape}, y_train shape: {y_train.shape}")
        print(f"Ticker {ticker}: X_test shape: {X_test_df.shape}, y_test shape: {y_test.shape}")

        # XGBoost Model Training
        model_params = XGB_PARAMS.copy()
        if model_params.get('objective') == 'reg:squarederror':
            model_params['objective'] = 'binary:logistic'
            model_params['eval_metric'] = 'logloss'
        
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        if neg_count > 0 and pos_count > 0:
            scale_pos_weight_val = neg_count / pos_count
            model_params['scale_pos_weight'] = scale_pos_weight_val
        else: # If one class is missing in train, remove scale_pos_weight or set to 1
             if 'scale_pos_weight' in model_params: del model_params['scale_pos_weight']
        
        if 'n_estimators' not in model_params:
             model_params['n_estimators'] = 200
        
        # Add n_jobs parameter to control CPU cores (e.g., 2 cores)
        model_params['n_jobs'] = 16

        model = xgb.XGBClassifier(**model_params)
        print(f"Ticker {ticker}: Training XGBoost model...")
        try:
            model.fit(
                X_train_df, 
                y_train,
                eval_set=[(X_test_df, y_test)],
                verbose=False
            )
        except Exception as e:
            print(f"Ticker {ticker}: Error during model training: {e}. Skipping.")
            continue

        # Evaluation
        y_pred_test = model.predict(X_test_df)
        accuracy = accuracy_score(y_test, y_pred_test)
        report_dict = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred_test)

        print(f"Ticker {ticker}: Test Accuracy: {accuracy:.4f}")
        # print(f"Ticker {ticker}: Classification Report (Test Set):\n", classification_report(y_test, y_pred_test, zero_division=0))
        # print(f"Ticker {ticker}: Confusion Matrix (Test Set):\n", cm)

        overall_metrics.append({
            'ticker': ticker,
            'accuracy': accuracy,
            'precision_0': report_dict.get('0', {}).get('precision', 0),
            'recall_0': report_dict.get('0', {}).get('recall', 0),
            'f1_0': report_dict.get('0', {}).get('f1-score', 0),
            'precision_1': report_dict.get('1', {}).get('precision', 0),
            'recall_1': report_dict.get('1', {}).get('recall', 0),
            'f1_1': report_dict.get('1', {}).get('f1-score', 0),
            'support_0': report_dict.get('0', {}).get('support', 0),
            'support_1': report_dict.get('1', {}).get('support', 0),
            'cm_tn': cm[0,0] if cm.shape == (2,2) else 0,
            'cm_fp': cm[0,1] if cm.shape == (2,2) else 0,
            'cm_fn': cm[1,0] if cm.shape == (2,2) else 0,
            'cm_tp': cm[1,1] if cm.shape == (2,2) else 0,
        })

        # Optionally save individual ticker reports and feature importances
        # ... (code similar to the global model for saving reports)

    if not overall_metrics:
        print("No models were trained successfully for any ticker.")
        return

    # --- Aggregate and Save Overall Per-Ticker Metrics ---
    metrics_df = pd.DataFrame(overall_metrics)
    print("\n--- Overall Per-Ticker Performance ---")
    print(metrics_df)
    
    history_lags_str = "_".join(map(str, history_lags_days))
    summary_filename = f"per_ticker_summary_lag{target_lag_days}_hist{history_lags_str}.csv"
    metrics_df.to_csv(RESULTS_DIR_PER_TICKER / summary_filename, index=False)
    print(f"Per-ticker summary saved to {RESULTS_DIR_PER_TICKER / summary_filename}")

    # You could also calculate a macro-average or weighted-average performance across tickers
    avg_accuracy = metrics_df['accuracy'].mean()
    print(f"\nAverage accuracy across tickers: {avg_accuracy:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train per-ticker XGBoost baselines.")
    # ... (same argparse setup as before) ...
    parser.add_argument(
        '--target_lag_days', type=int, default=1,
        help="Target lag days for prediction."
    )
    parser.add_argument(
        '--history_lags', type=str, default="1,2,3,5",
        help="Comma-separated list of past days for historical stock features (e.g., '1,2,3')."
    )
    args = parser.parse_args()
    history_lags_list = [int(lag.strip()) for lag in args.history_lags.split(',')]
    
    history_lags_str_fn = "_".join(map(str, history_lags_list))
    input_filename = f"all_tickers_raw_text_aligned_daily_lag{args.target_lag_days}_hist{history_lags_str_fn}.csv"
    aligned_data_file_path = ALIGNED_RAW_TEXT_DAILY_DIR / input_filename

    train_evaluate_xgboost_per_ticker(
        aligned_data_file_path,
        target_lag_days=args.target_lag_days,
        history_lags_days=history_lags_list
    )