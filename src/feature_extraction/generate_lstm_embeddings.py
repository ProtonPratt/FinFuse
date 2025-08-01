# src/feature_extraction/generate_lstm_stock_embeddings.py
import pandas as pd
import numpy as np
import glob
import os
import torch
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Assuming rnns_stocks.py and its config are accessible or components are redefined/imported
# We need StockLSTM class and feature engineering functions/configs from it.
# For simplicity, let's copy/paste relevant parts if they are not easily importable
# Or, better, refactor rnns_stocks.py to make these components importable.

# --- START: Components from/inspired by rnns_stocks.py ---
# (You might need to adjust paths if these are in a different structure or refactor for imports)
# Assuming config.py and sentiment_analyzer.py are in src/
# This script is for stock features, so sentiment_analyzer isn't directly used here.

# Configuration from rnns_stocks.py (adjust if needed)
LSTM_DATA_DIR = '/ssd_scratch/pratyush.jena/AmbiguityAssault/dataset/stocks_cleaned/' # Used by LSTMs
LSTM_FILE_PATTERN = '*_yahoo_data_0.csv' # Used by LSTMs
LSTM_TARGET_PRICE_COLUMN = 'Adj Close' # Used by LSTMs for its own target, not directly for embeddings

LSTM_BASE_FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
    'Daily Return', 'MA20', 'MA50', 'Volatility',
    'Market Cap', 'Dividend Yield'
]
LSTM_LAG_FEATURES_CONFIG = {
    'Adj Close': [1, 2, 3, 5], 'Daily Return': [1, 2, 3], 'Volume': [1, 2]
}
LSTM_SLIDING_WINDOW_CONFIG = {
    'Adj Close': {'windows': [5, 10, 20], 'aggs': ['mean', 'std', 'min', 'max']},
    'Volume': {'windows': [5, 10], 'aggs': ['mean', 'sum', 'std']},
    'Daily Return': {'windows': [3, 5, 10], 'aggs': ['mean', 'std', 'sum']}
}
LSTM_HIDDEN_DIM = 128
LSTM_N_LAYERS = 2
# Input dim for LSTM will be calculated after feature engineering

# --- Feature Engineering Functions (copy from rnns_stocks.py or import) ---
def add_lagged_features_for_lstm(df, config): # Renamed to avoid conflict if imported
    lagged_feature_names = []
    # Important: LSTMs were trained ticker-wise, so process data per ticker
    # This function should expect a single-ticker DataFrame sorted by Date
    df_copy = df.copy() 
    for col, lags in config.items():
        if col in df_copy.columns:
            for lag in lags:
                new_col_name = f'{col}_lag_{lag}'
                df_copy[new_col_name] = df_copy[col].shift(lag) # Simple shift, assuming sorted by date
                lagged_feature_names.append(new_col_name)
    return df_copy, lagged_feature_names

def add_sliding_window_features_for_lstm(df, config): # Renamed
    window_feature_names = []
    df_copy = df.copy()
    for col, params in config.items():
        if col in df_copy.columns:
            for window in params['windows']:
                for agg_func_name in params['aggs']:
                    new_col_name = f'{col}_win{window}_{agg_func_name}'
                    # Rolling and then shifting by 1 to avoid using current day's value in window calc for predicting current day
                    rolled_series = df_copy[col].rolling(window=window, min_periods=max(1, int(window*0.8))).agg(agg_func_name)
                    df_copy[new_col_name] = rolled_series.shift(1)
                    window_feature_names.append(new_col_name)
    return df_copy, window_feature_names

# --- StockLSTM Model Definition (copy from rnns_stocks.py or import) ---
class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate) # This was after lstm_out in rnns_stocks, should be consistent
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state=None):
        lstm_out, (h_n, c_n) = self.lstm(x, hidden_state)
        # For embeddings, we're interested in lstm_out or h_n before the final FC layer
        # Using output of the last LSTM layer, last time step:
        out_last_step = lstm_out[:, -1, :] 
        # Or using the final hidden state of the last layer:
        # h_n_last_layer = h_n[-1, :, :] # h_n shape is (num_layers, batch, hidden_dim)
        
        # If we want the representation BEFORE the final classification linear layer of the LSTM
        # The original forward pass in rnns_stocks applies dropout then fc.
        # For embeddings, we often take the LSTM output directly.
        # Let's return h_n (final hidden states of all layers) and lstm_out (all hidden states of last layer)
        return lstm_out, (h_n, c_n) # fc is not used for embedding extraction here

# --- END: Components from rnns_stocks.py ---

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Where your trained LSTM .pth files are stored (from rnns_stocks.py)
TRAINED_LSTM_MODELS_DIR = '/ssd_scratch/pratyush.jena/AmbiguityAssault/trained_models/model_weights'
TRAINED_LSTM_MODELS_DIR = Path(TRAINED_LSTM_MODELS_DIR).resolve()
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "lstm_stock_embeddings"

def generate_embeddings():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_ticker_embeddings = []

    # Assuming LSTM models are saved like TICKER_lstm_model.pth in the root
    # e.g., AAPL_lstm_model.pth
    
    stock_files = glob.glob(os.path.join(LSTM_DATA_DIR, LSTM_FILE_PATTERN))

    for stock_filepath in stock_files:
        base_filename = os.path.basename(stock_filepath)
        try:
            # Try to infer ticker from filename robustly
            ticker_symbol_from_file = base_filename.split('_')[0].upper() # e.g., AAPL from AAPL_yahoo_data_0.csv
            if ticker_symbol_from_file not in TICKERS: # Use your global TICKERS list
                print(f"Skipping file {base_filename}, ticker {ticker_symbol_from_file} not in configured TICKERS.")
                continue
            print(f"\nProcessing Ticker: {ticker_symbol_from_file}")
        except IndexError:
            print(f"Could not infer ticker from filename: {base_filename}. Skipping.")
            continue

        # Load stock data for this ticker
        df_stock_raw = pd.read_csv(stock_filepath)
        df_stock_raw['Date'] = pd.to_datetime(df_stock_raw['Date'])
        df_stock_raw.sort_values(by='Date', inplace=True)
        
        # --- Apply THE SAME feature engineering as rnns_stocks.py ---
        # Basic numeric conversion and Daily Return
        df_processed = df_stock_raw.copy()
        for col in LSTM_BASE_FEATURE_COLUMNS:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Ensure 'Daily Return' exists if not in original CSV
        if 'Daily Return' not in df_processed.columns and 'Adj Close' in df_processed.columns:
             df_processed['Daily Return'] = df_processed['Adj Close'].pct_change()

        df_processed, added_lagged_names = add_lagged_features_for_lstm(df_processed, LSTM_LAG_FEATURES_CONFIG)
        df_processed, added_window_names = add_sliding_window_features_for_lstm(df_processed, LSTM_SLIDING_WINDOW_CONFIG)
        
        lstm_input_feature_cols = sorted(list(set(
            [col for col in LSTM_BASE_FEATURE_COLUMNS if col in df_processed.columns] +
            [col for col in added_lagged_names if col in df_processed.columns] +
            [col for col in added_window_names if col in df_processed.columns]
        )))
        
        # Drop NaNs produced by feature engineering (LSTMs were trained on cleaned data)
        df_featurized = df_processed.dropna(subset=lstm_input_feature_cols).copy()
        if df_featurized.empty:
            print(f"No data left for {ticker_symbol_from_file} after feature engineering and NaN drop.")
            continue

        # Scale features (fit scaler ONCE on this ticker's full featurized data,
        # as LSTMs were likely trained per-ticker with per-ticker scaling)
        # Or, if you used a global scaler for LSTMs, adapt this.
        # Assuming per-ticker scaling for LSTMs for now:
        scaler = StandardScaler()
        df_featurized.loc[:, lstm_input_feature_cols] = scaler.fit_transform(df_featurized[lstm_input_feature_cols])
        
        # Load pre-trained LSTM model for this ticker
        model_filename = f"{ticker_symbol_from_file}_lstm_model.pth" # Matching rnns_stocks.py save format
        model_path = TRAINED_LSTM_MODELS_DIR / model_filename
        if not model_path.exists():
            print(f"LSTM model not found for {ticker_symbol_from_file} at {model_path}. Skipping.")
            continue

        input_dim = len(lstm_input_feature_cols)
        # Output dim of LSTM was 1 for classification, dropout for inference is 0
        lstm_model = StockLSTM(input_dim, LSTM_HIDDEN_DIM, LSTM_N_LAYERS, 1, 0.0).to(DEVICE)
        try:
            lstm_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except Exception as e:
            print(f"Error loading model for {ticker_symbol_from_file}: {e}. Skipping.")
            continue
        lstm_model.eval()

        ticker_embeddings = []
        for index, row in tqdm(df_featurized.iterrows(), total=df_featurized.shape[0], desc=f"Extracting LSTM emb for {ticker_symbol_from_file}"):
            # Prepare input for LSTM: one day's features, seq_len=1
            features_np = row[lstm_input_feature_cols].values.astype(np.float32)
            features_tensor = torch.tensor(features_np).unsqueeze(0).unsqueeze(0).to(DEVICE) # [1, 1, num_features]

            with torch.no_grad():
                # We want the hidden state, not the final classification output
                _, (h_n, _) = lstm_model(features_tensor) 
                # h_n has shape (num_layers, batch_size=1, hidden_dim)
                # Take hidden state of the last layer
                embedding = h_n[-1].squeeze().cpu().numpy() # Shape (hidden_dim,)
            
            emb_dict = {'date_D': row['Date'].date(), 'ticker': ticker_symbol_from_file}
            for i, val in enumerate(embedding):
                emb_dict[f'lstm_emb_{i}'] = val
            ticker_embeddings.append(emb_dict)
        
        if ticker_embeddings:
            all_ticker_embeddings.extend(ticker_embeddings)

    if not all_ticker_embeddings:
        print("No LSTM embeddings were generated for any ticker.")
        return

    final_embeddings_df = pd.DataFrame(all_ticker_embeddings)
    output_csv_path = OUTPUT_DIR / "all_tickers_lstm_stock_embeddings.csv"
    final_embeddings_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved all LSTM stock embeddings to {output_csv_path}. Shape: {final_embeddings_df.shape}")
    if not final_embeddings_df.empty:
        print(final_embeddings_df.head())

if __name__ == '__main__':
    # Make sure TICKERS is defined, e.g. from your global config or hardcoded for this script
    # from src.config import TICKERS # If config is structured
    TICKERS = ["AAPL", "AMZN", "NVDA", "TSLA", "NKE"] # Matching your example
    generate_embeddings()