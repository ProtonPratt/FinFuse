# src/models/xgboost_baseline_daily.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import argparse
from src.config import XGB_PARAMS # Assuming config.py is in src/

ALIGNED_RAW_TEXT_DAILY_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "aligned_raw_text_daily_history"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "xgboost_baseline_daily_with_history"

def train_evaluate_xgboost_baseline_with_history(aligned_data_path, target_lag_days=1, history_lags_days=[1,2,3,5]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df = pd.read_csv(aligned_data_path)
    except FileNotFoundError:
        print(f"Aligned data file not found: {aligned_data_path}")
        return

    print(f"Loaded data with shape: {df.shape}")
    # df.dropna(inplace=True) # Drop general NaNs initially, specific feature NaNs later

    # --- Feature Selection (Current Day D + Lagged Historical Features) ---
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

    print(f"Attempting to use feature columns: {feature_columns}")

    # Ensure all selected feature columns and target column exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Missing one or more feature columns in the DataFrame: {missing_cols}")
        return
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found in the DataFrame.")
        return

    # Drop rows where *any* of the selected features or target are NaN
    df.dropna(subset=feature_columns + [target_column], inplace=True)
    if df.empty:
        print("DataFrame is empty after dropping NaNs from selected features/target.")
        return

    print(f"Shape after selecting features and dropping NaNs: {df.shape}")
    
    df['date_D'] = pd.to_datetime(df['date_D'])
    # Sort for chronological split
    df_sorted_globally = df.sort_values(by='date_D').reset_index(drop=True)

    X = df_sorted_globally[feature_columns]
    y = df_sorted_globally[target_column].astype(int)

    if X.empty or y.empty:
        print("Feature set X or target y is empty.")
        return
        
    print(f"Value counts for target variable y:\n{y.value_counts(normalize=True)}")
    if len(y.unique()) < 2:
        print("Target variable has fewer than 2 unique classes. Cannot train classifier.")
        return

    # --- Chronological Train/Test Split (80/20) ---
    train_size = int(0.8 * len(df_sorted_globally))
    X_train_df = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test_df = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    print(f"X_train shape: {X_train_df.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test_df.shape}, y_test shape: {y_test.shape}")

    if X_train_df.empty or X_test_df.empty or len(y_train.unique()) < 2 :
        print("Train or test set is empty or has insufficient classes after split.")
        return

    # --- XGBoost Model Training ---
    model_params = XGB_PARAMS.copy()
    if model_params.get('objective') == 'reg:squarederror':
        model_params['objective'] = 'binary:logistic'
        model_params['eval_metric'] = 'logloss'
    
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    if neg_count > 0 and pos_count > 0:
        scale_pos_weight_val = neg_count / pos_count
        model_params['scale_pos_weight'] = scale_pos_weight_val
        print(f"Using scale_pos_weight: {scale_pos_weight_val:.4f}")
    
    if 'n_estimators' not in model_params:
         model_params['n_estimators'] = 200

    model = xgb.XGBClassifier(**model_params)
    print("Training XGBoost model with historical features...")
    
    # For newer XGBoost versions, eval_metric is set in model parameters
    eval_set = [(X_test_df, y_test)]
    
    model.fit(
        X_train_df, 
        y_train,
        eval_set=eval_set,
        verbose=False
    )

    # --- Evaluation ---
    print("\nEvaluating model...")
    y_pred_test = model.predict(X_test_df)
    accuracy = accuracy_score(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)

    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report (Test Set):\n", report)
    print("\nConfusion Matrix (Test Set):\n", cm)
    
    history_lags_str = "_".join(map(str, history_lags_days))
    report_filename = f"xgboost_baseline_lag{target_lag_days}_hist{history_lags_str}_report.txt"
    importance_filename = f"xgboost_baseline_lag{target_lag_days}_hist{history_lags_str}_feature_importance.csv"

    with open(RESULTS_DIR / report_filename, "w") as f:
        f.write(f"XGBoost Baseline (with History) for Target Lag: {target_lag_days} days, History Lags: {history_lags_days}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    print(f"Results saved to {RESULTS_DIR / report_filename}")

    try:
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': X_train_df.columns, 'importance': importance}) # Use columns from X_train_df
        feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)
        print("\nFeature Importances:\n", feature_importance_df)
        feature_importance_df.to_csv(RESULTS_DIR / importance_filename, index=False)
    except Exception as e:
        print(f"Could not retrieve feature importances: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train XGBoost baseline with historical stock features.")
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

    train_evaluate_xgboost_baseline_with_history(
        aligned_data_file_path,
        target_lag_days=args.target_lag_days,
        history_lags_days=history_lags_list
    )