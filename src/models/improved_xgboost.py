# src/models/improved_xgboost_daily.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Assuming config.py is in src/
from src.config import XGB_PARAMS

# Define input directory
ALIGNED_RAW_TEXT_DAILY_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "aligned_raw_text_daily"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "improved_xgboost_daily"

def create_technical_features(df):
    """
    Create additional technical indicators and features including historical features
    """
    df = df.copy()
    
    # Sort by ticker and date for proper calculation
    df = df.sort_values(['ticker', 'date_D']).reset_index(drop=True)
    
    # === BASIC PRICE AND VOLUME FEATURES ===
    # Price-based features
    df['price_range'] = (df['high_price_day_D'] - df['low_price_day_D']) / df['close_price_day_D']
    df['body_size'] = abs(df['close_price_day_D'] - df['open_price_day_D']) / df['close_price_day_D']
    df['upper_shadow'] = (df['high_price_day_D'] - np.maximum(df['open_price_day_D'], df['close_price_day_D'])) / df['close_price_day_D']
    df['lower_shadow'] = (np.minimum(df['open_price_day_D'], df['close_price_day_D']) - df['low_price_day_D']) / df['close_price_day_D']
    
    # Volume features
    df['volume_price_trend'] = df['volume_day_D'] * df['daily_return_Day_D']
    df['volume_ma_ratio'] = df.groupby('ticker')['volume_day_D'].transform(lambda x: x / x.rolling(5, min_periods=1).mean())
    
    # === HISTORICAL FEATURES (Multiple time windows) ===
    for window in [3, 5, 10, 20]:
        # Price momentum and statistics
        df[f'return_{window}d'] = df.groupby('ticker')['daily_return_Day_D'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'volatility_{window}d'] = df.groupby('ticker')['daily_return_Day_D'].transform(lambda x: x.rolling(window, min_periods=1).std())
        df[f'return_sum_{window}d'] = df.groupby('ticker')['daily_return_Day_D'].transform(lambda x: x.rolling(window, min_periods=1).sum())
        
        # Price position relative to historical levels
        df[f'price_vs_high_{window}d'] = df.groupby('ticker').apply(
            lambda x: (x['close_price_day_D'] - x['high_price_day_D'].rolling(window, min_periods=1).max()) / x['close_price_day_D']
        ).reset_index(level=0, drop=True)
        
        df[f'price_vs_low_{window}d'] = df.groupby('ticker').apply(
            lambda x: (x['close_price_day_D'] - x['low_price_day_D'].rolling(window, min_periods=1).min()) / x['close_price_day_D']
        ).reset_index(level=0, drop=True)
        
        df[f'price_vs_mean_{window}d'] = df.groupby('ticker').apply(
            lambda x: (x['close_price_day_D'] - x['close_price_day_D'].rolling(window, min_periods=1).mean()) / x['close_price_day_D']
        ).reset_index(level=0, drop=True)
        
        # Volume patterns
        df[f'volume_trend_{window}d'] = df.groupby('ticker')['volume_day_D'].transform(lambda x: x / x.rolling(window, min_periods=1).mean())
        df[f'volume_volatility_{window}d'] = df.groupby('ticker')['volume_day_D'].transform(lambda x: x.rolling(window, min_periods=1).std() / x.rolling(window, min_periods=1).mean())
        
        # Price range patterns
        df[f'avg_range_{window}d'] = df.groupby('ticker').apply(
            lambda x: ((x['high_price_day_D'] - x['low_price_day_D']) / x['close_price_day_D']).rolling(window, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
    
    # === TECHNICAL INDICATORS ===
    # RSI-like indicator
    def calculate_rsi(returns, window=14):
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        avg_gain = gains.rolling(window, min_periods=1).mean()
        avg_loss = losses.rolling(window, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = df.groupby('ticker')['daily_return_Day_D'].transform(lambda x: calculate_rsi(x, 14))
    df['rsi_7'] = df.groupby('ticker')['daily_return_Day_D'].transform(lambda x: calculate_rsi(x, 7))
    
    # Moving Average Convergence Divergence (MACD)
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, min_periods=1).mean()
        ema_slow = prices.ewm(span=slow, min_periods=1).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, min_periods=1).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    macd_data = df.groupby('ticker')['close_price_day_D'].transform(lambda x: calculate_macd(x)[0])
    df['macd'] = macd_data
    df['macd_signal'] = df.groupby('ticker')['close_price_day_D'].transform(lambda x: calculate_macd(x)[1])
    df['macd_histogram'] = df.groupby('ticker')['close_price_day_D'].transform(lambda x: calculate_macd(x)[2])
    
    # Bollinger Bands
    for window in [10, 20]:
        rolling_mean = df.groupby('ticker')['close_price_day_D'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        rolling_std = df.groupby('ticker')['close_price_day_D'].transform(lambda x: x.rolling(window, min_periods=1).std())
        df[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
        df[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
        df[f'bb_position_{window}'] = (df['close_price_day_D'] - rolling_mean) / (rolling_std * 2)
        df[f'bb_width_{window}'] = (rolling_std * 4) / rolling_mean
    
    # === LAGGED FEATURES ===
    # Previous day features
    lag_features = ['daily_return_Day_D', 'volume_day_D', 'price_range', 'body_size']
    for feature in lag_features:
        for lag in [1, 2, 3]:
            df[f'{feature}_lag_{lag}'] = df.groupby('ticker')[feature].shift(lag)
    
    # === CROSS-SECTIONAL FEATURES ===
    # Daily rankings across all stocks
    df['return_rank'] = df.groupby('date_D')['daily_return_Day_D'].rank(pct=True)
    df['volume_rank'] = df.groupby('date_D')['volume_day_D'].rank(pct=True)
    df['volatility_rank'] = df.groupby('date_D')['volatility_5d'].rank(pct=True)
    
    # === MOMENTUM AND TREND FEATURES ===
    # Consecutive up/down days
    df['return_direction'] = np.where(df['daily_return_Day_D'] > 0, 1, -1)
    df['consecutive_direction'] = df.groupby('ticker')['return_direction'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
    )
    
    # Trend strength
    for window in [5, 10]:
        df[f'trend_strength_{window}d'] = df.groupby('ticker')['daily_return_Day_D'].transform(
            lambda x: x.rolling(window, min_periods=1).apply(lambda y: np.corrcoef(y, range(len(y)))[0, 1] if len(y) > 1 else 0)
        )
    
    return df

def train_evaluate_improved_xgboost(aligned_data_path, target_lag_days=1):
    """
    Trains and evaluates an improved XGBoost model with better features and handling
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(aligned_data_path)
    except FileNotFoundError:
        print(f"Aligned data file not found: {aligned_data_path}")
        return

    print(f"Loaded data with shape: {df.shape}")
    
    # Convert date column
    df['date_D'] = pd.to_datetime(df['date_D'])
    
    # Create technical features
    print("Creating technical features...")
    df = create_technical_features(df)
    
    # Basic feature columns (original)
    basic_features = [
        'close_price_day_D', 'open_price_day_D', 'high_price_day_D', 
        'low_price_day_D', 'volume_day_D', 'daily_return_Day_D'
    ]
    
    # Technical feature columns (expanded)
    technical_features = [
        'price_range', 'body_size', 'upper_shadow', 'lower_shadow',
        'volume_price_trend', 'volume_ma_ratio', 
        'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_histogram',
        'return_rank', 'volume_rank', 'volatility_rank',
        'return_direction', 'consecutive_direction'
    ]
    
    # Historical features (expanded)
    historical_features = []
    for window in [3, 5, 10, 20]:
        historical_features.extend([
            f'return_{window}d', f'volatility_{window}d', f'return_sum_{window}d',
            f'price_vs_high_{window}d', f'price_vs_low_{window}d', f'price_vs_mean_{window}d',
            f'volume_trend_{window}d', f'volume_volatility_{window}d', f'avg_range_{window}d'
        ])
    
    # Bollinger Bands features
    bollinger_features = []
    for window in [10, 20]:
        bollinger_features.extend([
            f'bb_upper_{window}', f'bb_lower_{window}', 
            f'bb_position_{window}', f'bb_width_{window}'
        ])
    
    # Lagged features
    lagged_features = []
    lag_base_features = ['daily_return_Day_D', 'volume_day_D', 'price_range', 'body_size']
    for feature in lag_base_features:
        for lag in [1, 2, 3]:
            lagged_features.append(f'{feature}_lag_{lag}')
    
    # Trend features
    trend_features = []
    for window in [5, 10]:
        trend_features.append(f'trend_strength_{window}d')
    
    # All feature columns
    feature_columns = basic_features + technical_features + historical_features
    target_column = f'target_direction_D_plus_{target_lag_days}'
    
    # Check for required columns
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Missing feature columns: {missing_cols}")
        feature_columns = [col for col in feature_columns if col in df.columns]
    
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found in the DataFrame.")
        return
    
    # Drop rows with NaN in selected features or target
    df_clean = df.dropna(subset=feature_columns + [target_column]).copy()
    
    if df_clean.empty:
        print("DataFrame is empty after dropping NaNs.")
        return
    
    print(f"Shape after feature engineering and cleaning: {df_clean.shape}")
    print(f"Number of features: {len(feature_columns)}")
    
    # Prepare features and target
    X = df_clean[feature_columns]
    y = df_clean[target_column].astype(int)
    
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Chronological split first
    df_clean_sorted = df_clean.sort_values('date_D').reset_index(drop=True)
    train_size = int(0.8 * len(df_clean_sorted))
    
    # Split the data
    X_train_raw = df_clean_sorted.iloc[:train_size][feature_columns]
    y_train = df_clean_sorted.iloc[:train_size][target_column].astype(int)
    X_test_raw = df_clean_sorted.iloc[train_size:][feature_columns]
    y_test = df_clean_sorted.iloc[train_size:][target_column].astype(int)
    
    # Feature scaling (fit on train, transform both)
    scaler = RobustScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train_raw), 
        columns=X_train_raw.columns,
        index=X_train_raw.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test_raw), 
        columns=X_test_raw.columns,
        index=X_test_raw.index
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")
    
    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Convert to sample weights
    sample_weights = np.array([class_weight_dict[label] for label in y_train])
    
    # Improved XGBoost parameters
    improved_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,  # Reduced to prevent overfitting
        'learning_rate': 0.05,  # Lower learning rate
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'min_child_weight': 5,  # Increased to prevent overfitting
        'reg_alpha': 1.0,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'scale_pos_weight': class_weights[1] / class_weights[0]  # Handle class imbalance
    }
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=500,
        early_stopping_rounds=50,
        **improved_params
    )
    
    print("Training improved XGBoost model...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        sample_weight_eval_set=[sample_weights, None],
        verbose=False
    )
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_proba_test)
    
    print(f"\nResults:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    print("\nClassification Report (Test Set):")
    report = classification_report(y_test, y_pred_test)
    print(report)
    
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 15 Most Important Features:")
    print(importance_df.head(15))
    
    # Save results
    results_text = f"""Improved XGBoost Results for Target Lag: {target_lag_days} days
Train Accuracy: {train_acc:.4f}
Test Accuracy: {test_acc:.4f}
Test AUC: {test_auc:.4f}

Classification Report:
{report}

Confusion Matrix:
{cm}

Class Weights Used: {class_weight_dict}
"""
    
    with open(RESULTS_DIR / f"improved_xgboost_lag{target_lag_days}_report.txt", "w") as f:
        f.write(results_text)
    
    importance_df.to_csv(RESULTS_DIR / f"improved_xgboost_lag{target_lag_days}_feature_importance.csv", index=False)
    
    # Time series cross-validation for more robust evaluation
    print("\nPerforming time series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for train_idx_cv, val_idx_cv in tscv.split(X_train):
        X_train_cv = X_train.iloc[train_idx_cv]
        y_train_cv = y_train.iloc[train_idx_cv]
        X_val_cv = X_train.iloc[val_idx_cv]
        y_val_cv = y_train.iloc[val_idx_cv]
        
        # Calculate sample weights for CV fold
        cv_weights = np.array([class_weight_dict[label] for label in y_train_cv])
        
        cv_model = xgb.XGBClassifier(n_estimators=200, **improved_params)
        cv_model.fit(X_train_cv, y_train_cv, sample_weight=cv_weights, verbose=False)
        cv_pred = cv_model.predict_proba(X_val_cv)[:, 1]
        cv_auc = roc_auc_score(y_val_cv, cv_pred)
        cv_scores.append(cv_auc)
    
    print(f"Cross-validation AUC scores: {cv_scores}")
    print(f"Mean CV AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")
    
    print(f"\nResults saved to {RESULTS_DIR}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train improved XGBoost model with better features")
    parser.add_argument('--target_lag_days', type=int, default=1, 
                        help="Target lag days for prediction")
    args = parser.parse_args()
    
    input_filename = f"all_tickers_raw_text_aligned_daily_lag{args.target_lag_days}.csv"
    aligned_data_file_path = ALIGNED_RAW_TEXT_DAILY_DIR / input_filename
    
    train_evaluate_improved_xgboost(aligned_data_file_path, target_lag_days=args.target_lag_days)