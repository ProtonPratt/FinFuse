# src/models/xgboost_baseline_daily.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split # We'll use this for a simple split after chronological sorting
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import argparse
# Assuming config.py is in src/
from src.config import XGB_PARAMS # Import your predefined XGBoost parameters

# Define input directory (where the output of raw_text_daily_aligner.py is)
ALIGNED_RAW_TEXT_DAILY_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "aligned_raw_text_daily"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "xgboost_baseline_daily"

print(f"Aligned data directory: {ALIGNED_RAW_TEXT_DAILY_DIR}")
print(f"Results directory: {RESULTS_DIR}")

def train_evaluate_xgboost_baseline(aligned_data_path, target_lag_days=1):
    """
    Trains and evaluates an XGBoost baseline model using only historical stock features.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(aligned_data_path)
    except FileNotFoundError:
        print(f"Aligned data file not found: {aligned_data_path}")
        return

    print(f"Loaded data with shape: {df.shape}")
    df.dropna(inplace=True) # General NaN drop, specific feature NaNs handled below
    if df.empty:
        print("DataFrame is empty after initial NaN drop.")
        return

    # --- Feature Selection (Only Numerical Stock Features for Day D) ---
    # These are features known at the end of Day D
    feature_columns = [
        'close_price_day_D',
        'open_price_day_D',
        'high_price_day_D',
        'low_price_day_D',
        'volume_day_D',
        'daily_return_Day_D' # Return observed on Day D itself
    ]
    target_column = f'target_direction_D_plus_{target_lag_days}'

    # Ensure all selected feature columns and target column exist
    if not all(col in df.columns for col in feature_columns):
        print(f"Missing one or more feature columns in the DataFrame. Required: {feature_columns}")
        return
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found in the DataFrame.")
        return

    # Drop rows where selected features or target are NaN
    df.dropna(subset=feature_columns + [target_column], inplace=True)
    if df.empty:
        print("DataFrame is empty after dropping NaNs from selected features/target.")
        return

    print(f"Shape after selecting features and dropping NaNs: {df.shape}")
    
    # Convert date_D to datetime for sorting
    df['date_D'] = pd.to_datetime(df['date_D'])
    df.sort_values(by=['ticker', 'date_D'], inplace=True) # Sort for chronological split per ticker

    X = df[feature_columns]
    y = df[target_column].astype(int) # Ensure target is integer for classification

    if X.empty or y.empty:
        print("Feature set X or target y is empty.")
        return
        
    print(f"Feature columns used: {feature_columns}")
    print(f"Target column: {target_column}")
    print(f"Value counts for target variable y:\n{y.value_counts(normalize=True)}")

    if len(y.unique()) < 2:
        print("Target variable has fewer than 2 unique classes. Cannot train classifier.")
        return

    # --- Chronological Train/Test Split (80/20) ---
    # For simplicity, we'll do a global chronological split after sorting by date.
    # A more robust approach might involve splitting per ticker then concatenating,
    # or using GroupShuffleSplit, but a global sort is a common first step.
    df_sorted_globally = df.sort_values(by='date_D').reset_index(drop=True)
    
    train_size = int(0.8 * len(df_sorted_globally))
    
    X_train_df = df_sorted_globally.iloc[:train_size][feature_columns]
    y_train = df_sorted_globally.iloc[:train_size][target_column].astype(int)
    
    X_test_df = df_sorted_globally.iloc[train_size:][feature_columns]
    y_test = df_sorted_globally.iloc[train_size:][target_column].astype(int)
    
    print(y.value_counts())

    # Convert to DMatrix for XGBoost efficiency if desired, or use scikit-learn API
    # For simplicity, let's use the scikit-learn API for XGBClassifier
    print(f"X_train shape: {X_train_df.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test_df.shape}, y_test shape: {y_test.shape}")

    if X_train_df.empty or X_test_df.empty:
        print("Train or test set is empty after split. Adjust data or split logic.")
        return
    if len(y_train.unique()) < 2:
        print("Training target variable has fewer than 2 unique classes after split.")
        return

    # --- XGBoost Model Training ---
    # Use parameters from config.py, ensure they are for classification
    model_params = XGB_PARAMS.copy()
    if model_params.get('objective') == 'reg:squarederror': # If params were for regression
        print("XGB_PARAMS objective is for regression, changing to 'binary:logistic' for classification.")
        model_params['objective'] = 'binary:logistic'
        model_params['eval_metric'] = 'logloss' # or 'auc', 'error'
    
    # Add early_stopping_rounds to model parameters
    early_stopping_rounds = XGB_PARAMS.get('early_stopping_rounds', 20)
    model = xgb.XGBClassifier(
        **model_params,
        early_stopping_rounds=early_stopping_rounds
    )

    print("Training XGBoost model...")
    model.fit(
        X_train_df, 
        y_train,
        eval_set=[(X_test_df, y_test)], # Evaluate on test set during training
        verbose=False # Set to True or a number to see training progress
    )

    # --- Evaluation ---
    print("\nEvaluating model...")
    y_pred_test = model.predict(X_test_df)
    y_pred_proba_test = model.predict_proba(X_test_df)[:, 1] # Probability of class 1

    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    print("\nClassification Report (Test Set):")
    report = classification_report(y_test, y_pred_test)
    print(report)

    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    # Save results
    with open(RESULTS_DIR / f"xgboost_baseline_lag{target_lag_days}_report.txt", "w") as f:
        f.write(f"XGBoost Baseline for Target Lag: {target_lag_days} days\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print(f"Results saved to {RESULTS_DIR / f'xgboost_baseline_lag{target_lag_days}_report.txt'}")

    # --- Feature Importance ---
    try:
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': feature_columns, 'importance': importance})
        feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)
        print("\nFeature Importances:")
        print(feature_importance_df)
        feature_importance_df.to_csv(RESULTS_DIR / f"xgboost_baseline_lag{target_lag_days}_feature_importance.csv", index=False)
    except Exception as e:
        print(f"Could not retrieve feature importances: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate an XGBoost baseline model on daily aligned data.")
    parser.add_argument(
        '--target_lag_days',
        type=int,
        default=1,
        help="The target lag days for which the aligned data was created (e.g., 1 for next day prediction)."
    )
    args = parser.parse_args()

    # Construct the input file path based on the target_lag_days
    input_filename = f"all_tickers_raw_text_aligned_daily_lag{args.target_lag_days}.csv"
    aligned_data_file_path = ALIGNED_RAW_TEXT_DAILY_DIR / input_filename

    # Make sure XGB_PARAMS from config are suitable for classification
    # Example adjustment if needed (though better to have it correct in config.py)
    # from src.config import XGB_PARAMS, XGB_NUM_BOOST_ROUND, XGB_EARLY_STOPPING_ROUNDS
    # if XGB_PARAMS['objective'] == 'reg:squarederror':
    #     XGB_PARAMS['objective'] = 'binary:logistic'
    #     XGB_PARAMS['eval_metric'] = 'logloss'

    train_evaluate_xgboost_baseline(aligned_data_file_path, target_lag_days=args.target_lag_days)