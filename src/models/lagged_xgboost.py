import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, mean_squared_error
from math import sqrt
# from scipy.stats import linregress # For trend calculation if desired
import glob
import os

# --- Configuration ---
DATA_DIR = './'
FILE_PATTERN = '*_yahoo_data_0.csv'
TARGET_PRICE_COLUMN = 'Adj Close'

# Define features that will be directly used or used as a base for window features
BASE_FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
    'Daily Return', 'MA20', 'MA50', 'Volatility',
    'Market Cap', 'Dividend Yield'
]

# Configuration for sliding window features
# Format: 'column_to_window': { 'windows': [size1, size2], 'aggs': ['mean', 'std', 'min', 'max']}
SLIDING_WINDOW_CONFIG = {
    'Adj Close': {
        'windows': [5, 10, 20],
        'aggs': ['mean', 'std', 'min', 'max'] # Could add 'median'
    },
    'Volume': {
        'windows': [5, 10],
        'aggs': ['mean', 'sum', 'std']
    },
    'Daily Return': {
        'windows': [3, 5, 10],
        'aggs': ['mean', 'std', 'sum'] # Sum of returns gives cumulative return over window
    }
}
# Configuration for simple lagged features (can be used alongside window features)
LAG_FEATURES_CONFIG = {
    'Adj Close': [1, 2],
    'Daily Return': [1],
}


def add_lagged_features(df, config):
    lagged_feature_names = []
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by=['Ticker', 'Date'])
    for col, lags in config.items():
        if col in df_copy.columns:
            for lag in lags:
                new_col_name = f'{col}_lag_{lag}'
                df_copy[new_col_name] = df_copy.groupby('Ticker')[col].shift(lag)
                lagged_feature_names.append(new_col_name)
        else:
            print(f"Warning: Lag Column '{col}' not found.")
    return df_copy, lagged_feature_names

def add_sliding_window_features(df, config):
    """Adds sliding window features to the DataFrame, grouped by Ticker."""
    window_feature_names = []
    # It's crucial that df_copy maintains an index that can be aligned with the results
    # of the groupby operations. Let's ensure it's sorted first.
    df_copy = df.sort_values(by=['Ticker', 'Date']).copy() # Sort and copy
    # If df already has a unique, monotonic index, resetting might not be necessary.
    # However, if it's coming from concatenation, resetting index is good practice.
    df_copy = df_copy.reset_index(drop=True) # Ensure a simple RangeIndex

    for col, params in config.items():
        if col in df_copy.columns:
            for window in params['windows']:
                for agg_func_name in params['aggs']:
                    new_col_name = f'{col}_win{window}_{agg_func_name}'
                    if agg_func_name == 'trend':
                        # Placeholder for trend
                        pass
                    else:
                        # Perform the rolling operation within each group
                        # The result of .agg() will have a MultiIndex (Ticker, original_index_within_group)
                        # The .shift(1) is applied per group due to the initial groupby
                        
                        # Calculate the rolling feature
                        # The .groupby('Ticker', group_keys=False) prevents adding the group keys as an index level
                        # to the result of .apply, making it easier to align.
                        # However, for rolling.agg.shift, the structure is usually okay.
                        
                        # The key is that the result of this chain must align with df_copy.index
                        # after the groupby operation.
                        
                        # Let's try to extract the values and re-assign if direct assignment fails
                        # This method explicitly aligns by the DataFrame's index after calculation
                        
                        # Step 1: Calculate the rolling feature per group
                        rolled_series = df_copy.groupby('Ticker')[col] \
                                             .rolling(window=window, min_periods=max(1, int(window*0.8))) \
                                             .agg(agg_func_name)
                        
                        # The result of rolling().agg() on a grouped object often has a MultiIndex
                        # (Ticker, original_index_of_date_within_ticker).
                        # We need to get it back to a single index aligned with df_copy.
                        
                        # Reset index to bring 'Ticker' and the original date index back as columns,
                        # then drop 'Ticker' level if it's there, and sort by original index.
                        # Then apply the shift per group.
                        
                        # A more robust way might be to compute, then merge back.
                        # Or, ensure the grouped shift works as expected.
                        
                        # Simplest attempt first: rely on pandas to align the shifted Series
                        # The .groupby('Ticker', group_keys=False) is important here for the .shift()
                        # to produce a Series that is more easily alignable with the original df_copy's index.
                        
                        # This calculation is done on a per-group basis.
                        # The result of agg will have a MultiIndex (Ticker, Date original index).
                        # We need to shift within the group and then align to df_copy's index.

                        # Attempt 1: More explicit index handling
                        # result_series = (df_copy.groupby('Ticker', group_keys=False)[col]
                        #                  .apply(lambda x: x.rolling(window=window, min_periods=max(1, int(window*0.8))).agg(agg_func_name).shift(1)))
                        # df_copy[new_col_name] = result_series
                        
                        # Attempt 2: Ensure result of groupby().rolling().agg() is flattened before shift
                        # The issue might be that .shift(1) is applied after .agg() which is on a multi-index.
                        # We need to shift *within* each group.
                        
                        # Create the new column first with NaNs
                        df_copy[new_col_name] = np.nan
                        
                        # Iterate over groups, calculate, and assign
                        # This is usually slower but more robust for complex index issues
                        for ticker, group_df in df_copy.groupby('Ticker'):
                            # Calculate rolling feature for this group
                            group_rolled_values = group_df[col].rolling(window=window, min_periods=max(1, int(window*0.8))).agg(agg_func_name).shift(1)
                            # Assign back to the corresponding part of df_copy
                            df_copy.loc[group_df.index, new_col_name] = group_rolled_values
                            
                    window_feature_names.append(new_col_name)
        else:
            print(f"Warning: Window Column '{col}' not found.")
    return df_copy, window_feature_names

def load_and_preprocess_data(data_dir, file_pattern, lag_config, window_config):
    """Loads, preprocesses, adds lags, window features, and creates target."""
    # ... (initial loading and numeric conversion same as before) ...
    all_files = glob.glob(os.path.join(data_dir, file_pattern))
    if not all_files:
        print(f"No files found for pattern {file_pattern} in directory {data_dir}")
        return pd.DataFrame(), []

    list_of_dfs = []
    for filename in all_files:
        print(f"Processing file: {filename}")
        try:
            df_temp = pd.read_csv(filename)
            ticker = os.path.basename(filename).split('_')[0]
            df_temp['Ticker'] = ticker
            list_of_dfs.append(df_temp)
        except Exception as e:
            print(f"Error reading or processing {filename}: {e}")
            continue

    if not list_of_dfs:
        print("No dataframes were loaded.")
        return pd.DataFrame(), []

    full_df = pd.concat(list_of_dfs, ignore_index=True)
    full_df['Date'] = pd.to_datetime(full_df['Date'])

    numeric_cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                               'Daily Return', 'MA20', 'MA50', 'Volatility',
                               'Market Cap', 'Dividend Yield'] # BASE_FEATURE_COLUMNS can also be used if only numeric
    for col in numeric_cols_to_convert:
        if col in full_df.columns:
            full_df[col] = full_df[col].replace('', np.nan) # Handle empty strings if any
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
        else:
            print(f"Warning: Column {col} for numeric conversion not found.")


    full_df = full_df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    # --- Add Lagged Features (Optional, can be used with window features) ---
    print("Adding lagged features...")
    full_df, added_lagged_names = add_lagged_features(full_df, lag_config)
    print(f"Added lagged features: {added_lagged_names}")

    # --- Add Sliding Window Features ---
    print("Adding sliding window features...")
    full_df, added_window_names = add_sliding_window_features(full_df, window_config)
    print(f"Added sliding window features: {added_window_names}")

    # Combine all feature names
    all_feature_columns = [col for col in BASE_FEATURE_COLUMNS if col in full_df.columns] + \
                          [col for col in added_lagged_names if col in full_df.columns] + \
                          [col for col in added_window_names if col in full_df.columns]
    # Remove duplicates if any column name ended up in multiple lists
    all_feature_columns = sorted(list(set(all_feature_columns)))


    # --- Create Target Variable ---
    full_df['Next_Day_Adj_Close'] = full_df.groupby('Ticker')[TARGET_PRICE_COLUMN].shift(-1)
    full_df['Target'] = (full_df['Next_Day_Adj_Close'] > full_df[TARGET_PRICE_COLUMN]).astype(int)

    # --- Handle NaNs ---
    cols_to_check_for_nan = [col for col in all_feature_columns if col in full_df.columns] + ['Target']
    print(f"Shape before NaN drop (after adding all features): {full_df.shape}")
    print(f"Number of potential features before NaN drop: {len(all_feature_columns)}")

    full_df_cleaned = full_df.dropna(subset=cols_to_check_for_nan)
    print(f"Shape after NaN drop: {full_df_cleaned.shape}")

    if full_df_cleaned.empty:
        print("DataFrame is empty after NaN removal. Check data, configs, or feature availability.")
        return pd.DataFrame(), []

    # Ensure all feature columns actually exist after processing and NaN drop
    final_feature_columns = [col for col in all_feature_columns if col in full_df_cleaned.columns]
    print(f"Number of final features for model: {len(final_feature_columns)}")

    return full_df_cleaned, final_feature_columns


def calculate_mda(y_true, y_pred):
    """Calculate Mean Directional Accuracy"""
    correct_directions = sum((y_true == 1) & (y_pred == 1)) + sum((y_true == 0) & (y_pred == 0))
    return correct_directions / len(y_true)

def train_evaluate_xgboost(df, feature_cols, target_col='Target', test_size_ratio=0.2):
    """Trains XGBoost model and evaluates it."""
    if df.empty or not feature_cols:
        print("Cannot train on an empty DataFrame or with no features.")
        return None, {}

    X = df[feature_cols]
    y = df[target_col]

    split_index = int(len(X) * (1 - test_size_ratio))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if X_train.empty or X_test.empty:
        print("Train or Test set is empty after split. Adjust test_size_ratio or check data.")
        return None, {}

    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")

    class_counts = y_train.value_counts()
    scale_pos_weight_val = 1
    if 0 in class_counts and 1 in class_counts and class_counts.get(1, 0) > 0 : # Ensure class 1 exists and is > 0
        scale_pos_weight_val = class_counts[0] / class_counts[1]
        print(f"Calculated scale_pos_weight: {scale_pos_weight_val:.2f} (class 0: {class_counts[0]}, class 1: {class_counts[1]})")
    else:
        print(f"Could not calculate scale_pos_weight. Class counts: {class_counts}")


    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss', # Consider 'auc' as well
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight_val,
        random_state=42,
        n_estimators=150, # May need more estimators with more features
        learning_rate=0.05, # May need smaller learning rate
        max_depth=4,       # May adjust depth
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=15, # Extended early stopping
    )

    print("Training XGBoost model...")
    # Use X_test, y_test for early stopping. For more robust validation, use a separate validation set.
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    print("Making predictions on the test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mda = calculate_mda(y_test.values, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError as e:
        print(f"Could not calculate ROC AUC: {e}. Test set might be too small or have only one class.")
        roc_auc = np.nan

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Mean Directional Accuracy: {mda:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

    feature_importances_df = pd.DataFrame()
    if hasattr(model, 'feature_importances_'):
        feature_importances_df = pd.DataFrame({
            'feature': X_train.columns, # Use columns from X_train
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFeature Importances (Top 20):")
        print(feature_importances_df.head(20))
    else:
        print("Could not retrieve feature importances.")


    results = {
        'accuracy': accuracy,
        'rmse': rmse,
        'mda': mda,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'feature_importances': feature_importances_df
    }
    return model, results

# --- Main Execution ---
if __name__ == '__main__':
    # ... (dummy data generation can remain similar, but ensure enough rows for max window + shift) ...
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created dummy directory {DATA_DIR}. Populate it with your CSV files.")
        num_dummy_rows = 150 # Increased for lags, MAs, and window features
        start_date = pd.to_datetime('2012-01-03 05:00:00')
        # ... (rest of dummy data generation logic from previous example)
        dummy_dates = [start_date + pd.Timedelta(days=i) for i in range(num_dummy_rows)]

        ext_dummy_data = {'Date': dummy_dates}
        base_price = 100.0
        price_volatility = 1.5 # std dev of daily price changes
        
        # Generate Open, Close first
        open_prices = []
        close_prices = []
        current_price = base_price
        for _ in range(num_dummy_rows):
            open_prices.append(current_price)
            change = np.random.normal(0, price_volatility)
            current_price += change
            current_price = max(current_price, 1.0) # Price floor
            close_prices.append(current_price)
        
        ext_dummy_data['Open'] = open_prices
        ext_dummy_data['Close'] = close_prices

        # Generate High and Low based on Open and Close
        ext_dummy_data['High'] = [max(o, c) + abs(np.random.normal(0, price_volatility * 0.3)) for o, c in zip(ext_dummy_data['Open'], ext_dummy_data['Close'])]
        ext_dummy_data['Low'] = [min(o, c) - abs(np.random.normal(0, price_volatility * 0.3)) for o, c in zip(ext_dummy_data['Open'], ext_dummy_data['Close'])]
        # Ensure Low <= Open/Close and High >= Open/Close
        ext_dummy_data['Low'] = [min(l, o, c) for l, o, c in zip(ext_dummy_data['Low'], ext_dummy_data['Open'], ext_dummy_data['Close'])]
        ext_dummy_data['High'] = [max(h, o, c) for h, o, c in zip(ext_dummy_data['High'], ext_dummy_data['Open'], ext_dummy_data['Close'])]


        dummy_df = pd.DataFrame(ext_dummy_data)
        dummy_df['Adj Close'] = dummy_df['Close'] # simplify
        dummy_df['Volume'] = np.random.randint(1e6, 5e7, num_dummy_rows)
        dummy_df['Dividends'] = 0.0
        dummy_df['Stock Splits'] = 0.0
        dummy_df['Daily Return'] = dummy_df['Adj Close'].pct_change()
        
        # For dummy data, use min_periods to get some values even with short history
        dummy_df['MA20'] = dummy_df['Adj Close'].rolling(window=20, min_periods=5).mean()
        dummy_df['MA50'] = dummy_df['Adj Close'].rolling(window=50, min_periods=10).mean()
        dummy_df['Volatility'] = dummy_df['Daily Return'].rolling(window=20, min_periods=5).std()
        
        dummy_df['Market Cap'] = 2.8e12 * (dummy_df['Adj Close']/dummy_df['Adj Close'].dropna().iloc[0] if not dummy_df['Adj Close'].dropna().empty else 1)
        dummy_df['Dividend Yield'] = 0.0053

        dummy_df.to_csv(os.path.join(DATA_DIR, "DUMMY_yahoo_data_0.csv"), index=False)
        print(f"Created dummy file DUMMY_yahoo_data_0.csv with {len(dummy_df)} rows for testing.")
        print("Please replace with your actual stock data files for meaningful results.")

    # --- Main workflow ---
    processed_data, final_feature_columns = load_and_preprocess_data(
        DATA_DIR,
        FILE_PATTERN,
        LAG_FEATURES_CONFIG,
        SLIDING_WINDOW_CONFIG
    )

    if not processed_data.empty and final_feature_columns:
        model, results = train_evaluate_xgboost(processed_data, final_feature_columns)
        if model:
            print("\nModel training and evaluation complete.")
            # model.save_model("xgboost_stock_classifier_with_windows.json")
    else:
        print("No data or features to train the model. Exiting.")