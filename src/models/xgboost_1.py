import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split # We'll do a chronological split manually
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import glob
import os

# --- Configuration ---
DATA_DIR = './' # Make sure this path is correct
FILE_PATTERN = '*_yahoo_data_0.csv'
TARGET_PRICE_COLUMN = 'Adj Close' # Use 'Adj Close' as it accounts for dividends/splits
# Define features to use. Exclude 'Date' and potentially redundant/future info.
# We will also drop 'Dividends' and 'Stock Splits' as they are often zero and
# their effect is captured in 'Adj Close'.
# 'Market Cap' and 'Dividend Yield' can be useful if they vary daily.
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
    'Daily Return', 'MA20', 'MA50', 'Volatility',
    'Market Cap', 'Dividend Yield'
]

def load_and_preprocess_data(data_dir, file_pattern):
    """Loads all stock CSVs, preprocesses, and creates target variable."""
    all_files = glob.glob(os.path.join(data_dir, file_pattern))
    if not all_files:
        print(f"No files found for pattern {file_pattern} in directory {data_dir}")
        return pd.DataFrame()

    list_of_dfs = []
    for filename in all_files:
        print(f"Processing file: {filename}")
        try:
            df = pd.read_csv(filename)
            # Extract ticker from filename (e.g., "AAPL" from "AAPL_yahoo_data_0.csv")
            ticker = os.path.basename(filename).split('_')[0]
            df['Ticker'] = ticker
            list_of_dfs.append(df)
        except Exception as e:
            print(f"Error reading or processing {filename}: {e}")
            continue

    if not list_of_dfs:
        print("No dataframes were loaded.")
        return pd.DataFrame()

    print(list_of_dfs[0].head())  # Show the first few rows of the first dataframe for debugging
    full_df = pd.concat(list_of_dfs, ignore_index=True)

    # --- Data Type Conversion and Basic Cleaning ---
    full_df['Date'] = pd.to_datetime(full_df['Date'])

    # Convert potential string NaNs or empty strings in numeric columns to actual NaNs
    # then to numeric.
    # The provided CSV sample has empty strings for some initial values.
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                    'Daily Return', 'MA20', 'MA50', 'Volatility',
                    'Market Cap', 'Dividend Yield']
    for col in numeric_cols:
        if col in full_df.columns:
            # Replace empty strings with NaN, then convert
            full_df[col] = full_df[col].replace('', np.nan)
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
        else:
            print(f"Warning: Column {col} not found in combined dataframe.")


    # Sort data chronologically for each stock (important for time series)
    full_df = full_df.sort_values(by=['Ticker', 'Date'])
    full_df = full_df.reset_index(drop=True)


    # --- Create Target Variable ---
    # Target: 1 if next day's TARGET_PRICE_COLUMN is higher, 0 otherwise.
    # We group by 'Ticker' to ensure shift doesn't cross between stocks.
    full_df['Next_Day_Adj_Close'] = full_df.groupby('Ticker')[TARGET_PRICE_COLUMN].shift(-1)
    full_df['Target'] = (full_df['Next_Day_Adj_Close'] > full_df[TARGET_PRICE_COLUMN]).astype(int)

    # --- Handle NaNs ---
    # NaNs can come from:
    # 1. Initial rows where MAs, Volatility, Daily Return are not calculable.
    # 2. Last row for each stock where 'Next_Day_Adj_Close' (and thus 'Target') is NaN.
    # 3. Any NaNs within the feature columns themselves.
    # We'll drop rows where any of our chosen features or the target is NaN.
    
    # Identify columns that will be used for features + target
    cols_to_check_for_nan = FEATURE_COLUMNS + ['Target']
    # Ensure all these columns actually exist in full_df before trying to dropna
    existing_cols_to_check = [col for col in cols_to_check_for_nan if col in full_df.columns]
    
    print(f"Shape before NaN drop: {full_df.shape}")
    full_df_cleaned = full_df.dropna(subset=existing_cols_to_check)
    print(f"Shape after NaN drop: {full_df_cleaned.shape}")

    if full_df_cleaned.empty:
        print("DataFrame is empty after NaN removal. Check data quality or feature availability.")
        return pd.DataFrame()

    return full_df_cleaned


def train_evaluate_xgboost(df, feature_cols, target_col='Target', test_size_ratio=0.2):
    """Trains XGBoost model and evaluates it."""
    if df.empty:
        print("Cannot train on an empty DataFrame.")
        return None, {}

    # Ensure all feature_cols exist in df
    actual_feature_cols = [col for col in feature_cols if col in df.columns]
    if len(actual_feature_cols) != len(feature_cols):
        missing = set(feature_cols) - set(actual_feature_cols)
        print(f"Warning: Missing feature columns: {missing}. Using available: {actual_feature_cols}")
    if not actual_feature_cols:
        print("No feature columns available for training.")
        return None, {}

    X = df[actual_feature_cols]
    y = df[target_col]

    # --- Chronological Train-Test Split ---
    # Crucial for time series: test set must be chronologically after train set.
    # We'll split the entire dataset. A more advanced approach might split per stock
    # or use walk-forward validation.
    split_index = int(len(X) * (1 - test_size_ratio))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if X_train.empty or X_test.empty:
        print("Train or Test set is empty after split. Adjust test_size_ratio or check data.")
        return None, {}

    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # --- XGBoost Model ---
    # Handle class imbalance if necessary (e.g. significantly more up days than down)
    # Calculate scale_pos_weight if classes are imbalanced
    # class_counts = y_train.value_counts()
    # scale_pos_weight = class_counts[0] / class_counts[1] if 1 in class_counts and class_counts[1] > 0 else 1

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss', # 'auc' or 'error' are also common
        use_label_encoder=False, # Recommended to set to False for newer XGBoost
        # scale_pos_weight=scale_pos_weight, # Uncomment if classes are imbalanced
        random_state=42,
        n_estimators=100, # Number of trees
        learning_rate=0.1,
        max_depth=3
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    # --- Predictions and Evaluation ---
    print("Making predictions on the test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate MDA (Mean Directional Accuracy)
    mda = np.mean(y_pred == y_test)
    
    # Calculate RMSE (Root Mean Square Error) for probabilities
    rmse = np.sqrt(np.mean((y_pred_proba - y_test) ** 2))

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Directional Accuracy (MDA): {mda:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

    # Feature importance
    try:
        feature_importances = pd.DataFrame({
            'feature': actual_feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFeature Importances:")
        print(feature_importances)
    except Exception as e:
        print(f"Could not get feature importances: {e}")


    results = {
        'accuracy': accuracy,
        'mda': mda,
        'rmse': rmse,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'feature_importances': feature_importances if 'feature_importances' in locals() else None
    }
    return model, results

# --- Main Execution ---
if __name__ == '__main__':
    # Create dummy stocks_cleaned directory and files if they don't exist for testing
    if not os.path.exists(DATA_DIR):
        # os.makedirs(DATA_DIR)
        print(f"Created dummy directory {DATA_DIR}. Populate it with your CSV files.")
        # # Example: create a dummy AAPL file
        # dummy_data = {
        #     'Date': pd.to_datetime(['2012-01-03 05:00:00', '2012-01-04 05:00:00', '2012-01-05 05:00:00', '2012-01-06 05:00:00', '2012-01-09 05:00:00']),
        #     'Open': [14.62, 14.64, 14.81, 14.95, 15.08],
        #     'High': [14.73, 14.81, 14.94, 15.09, 15.27],
        #     'Low': [14.60, 14.61, 14.73, 14.81, 15.04],
        #     'Close': [14.68, 14.76, 14.92, 15.06, 15.20],
        #     'Adj Close': [12.37, 12.44, 12.58, 12.69, 12.81],
        #     'Volume': [302220800, 260022000, 271269600, 310952000, 320880000],
        #     'Dividends': [0.0, 0.0, 0.0, 0.0, 0.0],
        #     'Stock Splits': [0.0, 0.0, 0.0, 0.0, 0.0],
        #     'Daily Return': [np.nan, 0.005374, 0.011101, 0.008923, 0.009296],
        #     'MA20': [np.nan, np.nan, 12.46, 12.51, 12.57], # Simplified MAs for dummy
        #     'MA50': [np.nan, np.nan, np.nan, 12.40, 12.45], # Simplified MAs for dummy
        #     'Volatility': [np.nan, np.nan, 0.008, 0.007, 0.006], # Simplified Vol for dummy
        #     'Market Cap': [2.8e12, 2.8e12, 2.8e12, 2.8e12, 2.8e12],
        #     'Dividend Yield': [0.53, 0.53, 0.53, 0.53, 0.53]
        # }
        # # Need more data for MAs to be non-NaN, extend dummy data
        # num_dummy_rows = 60 
        # start_date = pd.to_datetime('2012-01-03 05:00:00')
        # dummy_dates = [start_date + pd.Timedelta(days=i) for i in range(num_dummy_rows)]
        
        # ext_dummy_data = {'Date': dummy_dates}
        # base_price = 12.0
        # for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
        #     ext_dummy_data[col] = base_price + np.random.randn(num_dummy_rows).cumsum() * 0.1
        #     base_price = ext_dummy_data[col][-1] # for next iteration

        # ext_dummy_data['Volume'] = np.random.randint(1e7, 5e7, num_dummy_rows)
        # ext_dummy_data['Dividends'] = 0.0
        # ext_dummy_data['Stock Splits'] = 0.0
        
        # dummy_df = pd.DataFrame(ext_dummy_data)
        # dummy_df['Adj Close'] = dummy_df['Close'] # simplify for dummy
        # dummy_df['Daily Return'] = dummy_df['Adj Close'].pct_change()
        # dummy_df['MA20'] = dummy_df['Adj Close'].rolling(window=20).mean()
        # dummy_df['MA50'] = dummy_df['Adj Close'].rolling(window=50).mean()
        # dummy_df['Volatility'] = dummy_df['Daily Return'].rolling(window=20).std()
        # dummy_df['Market Cap'] = 2.8e12
        # dummy_df['Dividend Yield'] = 0.53
        
        # dummy_df.to_csv(os.path.join(DATA_DIR, "DUMMY_yahoo_data_0.csv"), index=False)
        # print(f"Created dummy file DUMMY_yahoo_data_0.csv with {len(dummy_df)} rows for testing.")
        # print("Please replace with your actual stock data files.")


    # --- Main workflow ---
    processed_data = load_and_preprocess_data(DATA_DIR, FILE_PATTERN)

    if not processed_data.empty:
        model, results = train_evaluate_xgboost(processed_data, FEATURE_COLUMNS)
        if model:
            print("\nModel training and evaluation complete.")
            # You can save the model using:
            # model.save_model("xgboost_stock_classifier.json")
            # loaded_model = xgb.XGBClassifier()
            # loaded_model.load_model("xgboost_stock_classifier.json")
    else:
        print("No data to train the model. Exiting.")