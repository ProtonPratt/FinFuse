import pandas as pd
import numpy as np
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import mean_squared_error
import math

TICKERS = ["AAPL", "AMZN", "NVDA", "TSLA", "NKE"] 

# --- Configuration (similar to XGBoost, with additions for PyTorch) ---
DATA_DIR = '/ssd_scratch/pratyush.jena/AmbiguityAssault/dataset/stocks_cleaned/'
FILE_PATTERN = '*_yahoo_data_0.csv'
TARGET_PRICE_COLUMN = 'Adj Close'

BASE_FEATURE_COLUMNS = [ # Keep these consistent
    'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
    'Daily Return', 'MA20', 'MA50', 'Volatility',
    'Market Cap', 'Dividend Yield'
]
LAG_FEATURES_CONFIG = {
    'Adj Close': [1, 2, 3, 5],
    'Daily Return': [1, 2, 3],
    'Volume': [1, 2]
}
SLIDING_WINDOW_CONFIG = {
    'Adj Close': {'windows': [5, 10, 20], 'aggs': ['mean', 'std', 'min', 'max']},
    'Volume': {'windows': [5, 10], 'aggs': ['mean', 'sum', 'std']},
    'Daily Return': {'windows': [3, 5, 10], 'aggs': ['mean', 'std', 'sum']}
}

# PyTorch / RNN specific
BATCH_SIZE = 64 # How many samples per batch to load
N_EPOCHS = 100    # Number of epochs to train for
LEARNING_RATE = 0.001
HIDDEN_DIM = 128 # Number of features in hidden state h
N_LSTM_LAYERS = 2 # Number of LSTM layers
DROPOUT_RATE = 0.25 # Dropout rate for regularization

# --- Feature Engineering Functions (add_lagged_features, add_sliding_window_features) ---
# --- (These are the same as in your previous XGBoost script, so I'll omit them for brevity) ---
# --- (Make sure they are defined in your script) ---
def add_lagged_features(df, config):
    lagged_feature_names = []
    df_copy = df.sort_values(by=['Ticker', 'Date']).copy().reset_index(drop=True)
    for col, lags in config.items():
        if col in df_copy.columns:
            for lag in lags:
                new_col_name = f'{col}_lag_{lag}'
                # Corrected lagged feature calculation: shift within group
                shifted_values = df_copy.groupby('Ticker', group_keys=False)[col].shift(lag)
                df_copy[new_col_name] = shifted_values
                lagged_feature_names.append(new_col_name)
        else:
            print(f"Warning: Lag Column '{col}' not found.")
    return df_copy, lagged_feature_names

def add_sliding_window_features(df, config):
    window_feature_names = []
    df_copy = df.sort_values(by=['Ticker', 'Date']).copy().reset_index(drop=True)
    for col, params in config.items():
        if col in df_copy.columns:
            for window in params['windows']:
                for agg_func_name in params['aggs']:
                    new_col_name = f'{col}_win{window}_{agg_func_name}'
                    if agg_func_name == 'trend': # Placeholder
                        pass
                    else:
                        rolled_agg = df_copy.groupby('Ticker')[col]\
                                          .rolling(window=window, min_periods=max(1, int(window*0.8)))\
                                          .agg(agg_func_name)
                        shifted_values = rolled_agg.groupby(level='Ticker', group_keys=False).shift(1)
                        df_copy[new_col_name] = shifted_values.reset_index(level='Ticker', drop=True)
                    window_feature_names.append(new_col_name)
        else:
            print(f"Warning: Window Column '{col}' not found.")
    return df_copy, window_feature_names
# --- End of Feature Engineering ---
def train_ticker_wise(data_dir, file_pattern, lag_config, window_config):
    """Train separate models for each ticker and evaluate their performance"""
    all_files = glob.glob(os.path.join(data_dir, file_pattern))
    results = {}
    
    for filename in all_files:
        ticker = filename
        print(f"\n{'='*50}")
        print(f"Training model for ticker: {ticker}")
        print(f"{'='*50}")
        
        # Load single ticker data
        df = pd.read_csv(filename)
        df['Ticker'] = ticker
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Process single ticker data
        df_processed = df.copy()
        for col in BASE_FEATURE_COLUMNS:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Add features
        df_processed, added_lagged_names = add_lagged_features(df_processed, lag_config)
        df_processed, added_window_names = add_sliding_window_features(df_processed, window_config)
        
        all_feature_columns = sorted(list(set(
            [col for col in BASE_FEATURE_COLUMNS if col in df_processed.columns] +
            [col for col in added_lagged_names if col in df_processed.columns] +
            [col for col in added_window_names if col in df_processed.columns]
        )))
        
        # Create target
        df_processed['Next_Day_Adj_Close'] = df_processed[TARGET_PRICE_COLUMN].shift(-1)
        df_processed['Target'] = (df_processed['Next_Day_Adj_Close'] > df_processed[TARGET_PRICE_COLUMN]).astype(float)
        
        # Clean data
        cols_to_check = [col for col in all_feature_columns if col in df_processed.columns] + ['Target']
        df_processed = df_processed.dropna(subset=cols_to_check).copy()
        
        if df_processed.empty:
            print(f"No valid data for {ticker}")
            continue
            
        # Train-test split
        split_index = int(len(df_processed) * 0.8)
        df_train = df_processed.iloc[:split_index].copy()
        df_test = df_processed.iloc[split_index:].copy()
        
        # Scale features
        scaler = StandardScaler()
        df_train.loc[:, all_feature_columns] = scaler.fit_transform(df_train[all_feature_columns])
        df_test.loc[:, all_feature_columns] = scaler.transform(df_test[all_feature_columns])
        
        # Create datasets and dataloaders
        train_dataset = StockDataset(df_train, all_feature_columns, 'Target')
        test_dataset = StockDataset(df_test, all_feature_columns, 'Target')
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = len(all_feature_columns)
        lstm_model = StockLSTM(input_dim, HIDDEN_DIM, N_LSTM_LAYERS, 1, DROPOUT_RATE).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
        
        # Train model
        print(f"\nTraining model for {ticker}...")
        lstm_model, best_metrics = train_model(lstm_model, train_loader, test_loader, 
                                             criterion, optimizer, device, N_EPOCHS, None)
        
        # save the model
        ticker_name = ticker.split('/')[-1].split('.')[0]  # Extract ticker name from filename
        ticker_name = ticker_name.split('_')[0]  # Handle cases like 'AAPL_yahoo_data_0.csv'
        model_save_path = f"{ticker_name}_lstm_model.pth"
        torch.save(lstm_model.state_dict(), model_save_path)
        
        # Evaluate final model
        print(f"\nFinal evaluation for {filename}...")
        eval_metrics = evaluate_model(lstm_model, test_loader, criterion, device)
        
        # Store results
        results[ticker] = {
            'best_epoch': best_metrics['epoch'],
            'accuracy': eval_metrics['accuracy'],
            'rmse': eval_metrics['rmse'],
            'mda': eval_metrics['mda'],
            'roc_auc': eval_metrics['roc_auc']
        }
        
    # Print summary of all results
    print("\n" + "="*80)
    print("Summary of Results for All Tickers")
    print("="*80)
    print(f"{'Ticker':<10} {'Accuracy':<10} {'RMSE':<10} {'MDA':<10} {'ROC AUC':<10}")
    print("-"*80)
    
    for ticker, metrics in results.items():
        print(f"{ticker:<10} {metrics['accuracy']:.4f}     {metrics['rmse']:.4f}     "
              f"{metrics['mda']:.4f}     {metrics['roc_auc']:.4f}")
    
    return results

def load_and_preprocess_data_for_rnn(data_dir, file_pattern, lag_config, window_config):
    all_files = glob.glob(os.path.join(data_dir, file_pattern))
    # ... (rest of initial file loading as before) ...
    list_of_dfs = []
    print("Loading files for RNN...")
    for filename in all_files:
        print(f"Reading file: {filename}") # Optional: for debugging
        try:
            # print(f"Processing file: {filename}") # Optional: for debugging
            df_temp = pd.read_csv(filename)
            ticker = os.path.basename(filename).split('_')[0]
            df_temp['Ticker'] = ticker
            list_of_dfs.append(df_temp)
        except Exception as e:
            print(f"Error reading or processing {filename}: {e}")
            continue
    if not list_of_dfs: return pd.DataFrame(), [], None, None

    full_df = pd.concat(list_of_dfs, ignore_index=True)
    full_df['Date'] = pd.to_datetime(full_df['Date'])

    numeric_cols_to_convert = BASE_FEATURE_COLUMNS[:] # Make a copy
    for col in numeric_cols_to_convert:
        if col in full_df.columns:
            full_df[col] = full_df[col].replace('', np.nan)
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    full_df = full_df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    print("Adding lagged features...")
    full_df, added_lagged_names = add_lagged_features(full_df, lag_config)
    print("Adding sliding window features...")
    full_df, added_window_names = add_sliding_window_features(full_df, window_config)

    all_feature_columns = sorted(list(set(
        [col for col in BASE_FEATURE_COLUMNS if col in full_df.columns] +
        [col for col in added_lagged_names if col in full_df.columns] +
        [col for col in added_window_names if col in full_df.columns]
    )))

    full_df['Next_Day_Adj_Close'] = full_df.groupby('Ticker')[TARGET_PRICE_COLUMN].shift(-1)
    full_df['Target'] = (full_df['Next_Day_Adj_Close'] > full_df[TARGET_PRICE_COLUMN]).astype(float) # Target as float for BCEWithLogitsLoss

    cols_to_check_for_nan = [col for col in all_feature_columns if col in full_df.columns] + ['Target']
    print(f"Shape before NaN drop: {full_df.shape}")
    full_df_cleaned = full_df.dropna(subset=cols_to_check_for_nan).copy() # Use .copy()
    print(f"Shape after NaN drop: {full_df_cleaned.shape}")

    if full_df_cleaned.empty: return pd.DataFrame(), [], None, None

    final_feature_columns = [col for col in all_feature_columns if col in full_df_cleaned.columns]
    if not final_feature_columns: return pd.DataFrame(), [], None, None

    # --- Train-Test Split (Chronological) ---
    # Important: Split BEFORE scaling, fit scaler ONLY on training data
    split_index = int(len(full_df_cleaned) * 0.8) # 80/20 split
    df_train = full_df_cleaned.iloc[:split_index].copy()
    df_test = full_df_cleaned.iloc[split_index:].copy()

    # --- Scaling ---
    scaler = StandardScaler()
    # Convert columns to float before scaling
    df_train[final_feature_columns] = df_train[final_feature_columns].astype(float)
    df_test[final_feature_columns] = df_test[final_feature_columns].astype(float)
    
    # Perform scaling
    df_train.loc[:, final_feature_columns] = scaler.fit_transform(df_train[final_feature_columns])
    df_test.loc[:, final_feature_columns] = scaler.transform(df_test[final_feature_columns])

    return df_train, df_test, final_feature_columns, scaler


# --- PyTorch Dataset ---
class StockDataset(Dataset):
    def __init__(self, dataframe, feature_cols, target_col):
        self.features = torch.tensor(dataframe[feature_cols].values, dtype=torch.float32)
        self.targets = torch.tensor(dataframe[target_col].values, dtype=torch.float32).unsqueeze(1) # Ensure target is [batch_size, 1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # For LSTMs, input shape is (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first=True
        # Here, we treat each day as a sequence of length 1.
        return self.features[idx].unsqueeze(0), self.targets[idx] # Add seq_len dim: [1, num_features]

# --- LSTM Model Definition ---
class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        # batch_first=True means input and output tensors are provided as (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)

        # Dropout layer for regularization after LSTM
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer to map LSTM output to our desired output_dim (1 for binary classification)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # No sigmoid here, as BCEWithLogitsLoss will handle it

    def forward(self, x, hidden_state=None):
        # x shape: (batch_size, seq_len=1, input_dim)
        # hidden_state can be passed in, or LSTM initializes it if None
        
        # LSTM out: (batch_size, seq_len, hidden_dim)
        # h_n: (num_layers, batch_size, hidden_dim) - final hidden state for each element in batch
        # c_n: (num_layers, batch_size, hidden_dim) - final cell state for each element in batch
        lstm_out, (h_n, c_n) = self.lstm(x, hidden_state)

        # We only need the output from the last time step of the sequence for classification
        # If seq_len > 1, use lstm_out[:, -1, :]
        # If seq_len = 1 (as in our current StockDataset), lstm_out[:, -1, :] is same as lstm_out.squeeze(1)
        out = lstm_out[:, -1, :] # Get the last time step's output: (batch_size, hidden_dim)
        
        out = self.dropout(out) # Apply dropout
        out = self.fc(out)      # (batch_size, output_dim)
        return out, (h_n.detach(), c_n.detach()) # Detach hidden state to prevent backprop through entire history if reused


# --- Training Function ---
def train_model(model, train_loader, test_loader, criterion, optimizer, device, n_epochs, df_train_tickers):
    model.train()
    best_test_acc = 0
    best_model_state = None
    best_metrics = None
    
    for epoch in range(n_epochs):
        # Training loop
        epoch_train_loss = 0
        train_targets, train_preds = [], []
        
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(features, None)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            train_targets.extend(targets.cpu().numpy())
            train_preds.extend(preds.cpu().numpy())
            
        # Evaluation loop
        model.eval()
        epoch_test_loss = 0
        test_targets, test_preds = [], []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)
                outputs, _ = model(features, None)
                loss = criterion(outputs, targets)
                epoch_test_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                test_targets.extend(targets.cpu().numpy())
                test_preds.extend(preds.cpu().numpy())
        
        # Calculate metrics
        train_acc = accuracy_score(np.array(train_targets).flatten(), np.array(train_preds).flatten())
        train_rmse = math.sqrt(mean_squared_error(np.array(train_targets).flatten(), np.array(train_preds).flatten()))
        
        test_acc = accuracy_score(np.array(test_targets).flatten(), np.array(test_preds).flatten())
        test_rmse = math.sqrt(mean_squared_error(np.array(test_targets).flatten(), np.array(test_preds).flatten()))
        test_mda = calculate_mda(np.array(test_targets).flatten(), np.array(test_preds).flatten())
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            best_metrics = {
                'epoch': epoch + 1,
                'test_acc': test_acc,
                'test_rmse': test_rmse,
                'test_mda': test_mda
            }
        
        print(f"Epoch [{epoch+1}/{n_epochs}]:")
        print(f"Train Loss: {epoch_train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Train RMSE: {train_rmse:.4f}")
        print(f"Test Loss: {epoch_test_loss/len(test_loader):.4f}, Test Acc: {test_acc:.4f}, Test RMSE: {test_rmse:.4f}")
        print("-" * 80)
        
        model.train()
    
    # Restore best model
    model.load_state_dict(best_model_state)
    print("\nBest Model Performance (Epoch {}):"
          .format(best_metrics['epoch']))
    print(f"Test Accuracy: {best_metrics['test_acc']:.4f}")
    print(f"Test RMSE: {best_metrics['test_rmse']:.4f}")
    print(f"Test MDA: {best_metrics['test_mda']:.4f}")
    
    return model, best_metrics

# Add this function before evaluate_model
def calculate_mda(y_true, y_pred):
    """Calculate Mean Directional Accuracy"""
    correct_directions = sum((np.array(y_true) == 1) & (np.array(y_pred) == 1)) + \
                        sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
    return correct_directions / len(y_true)

# --- Evaluation Function ---
def evaluate_model(model, test_loader, criterion, device):
    model.eval() # Set model to evaluation mode
    all_targets = []
    all_predictions = []
    all_probabilities = []
    test_loss = 0

    with torch.no_grad(): # Disable gradient calculations
        hidden_state = None # Reset hidden state for evaluation
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs, _ = model(features, hidden_state) # New hidden state not reused
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            probs = torch.sigmoid(outputs) # Get probabilities
            preds = (probs > 0.5).float()  # Get binary predictions

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets).flatten()
    all_predictions = np.array(all_predictions).flatten()
    
    avg_test_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    rmse = math.sqrt(mean_squared_error(all_targets, all_predictions))
    mda = calculate_mda(all_targets, all_predictions)
    
    try:
        roc_auc = roc_auc_score(all_targets, all_probabilities)
    except ValueError: # Handle cases like only one class in targets
        roc_auc = float('nan')
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    class_report = classification_report(all_targets, all_predictions, zero_division=0)

    print(f"\n--- LSTM Model Evaluation ---")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Mean Directional Accuracy: {mda:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    
    return {
        'accuracy': accuracy,
        'rmse': rmse,
        'mda': mda,
        'roc_auc': roc_auc,
        'test_loss': avg_test_loss
    }

# --- Main Execution ---
if __name__ == '__main__':
    # Load and preprocess data
    df_train, df_test, feature_cols, scaler = load_and_preprocess_data_for_rnn(
        DATA_DIR, FILE_PATTERN, LAG_FEATURES_CONFIG, SLIDING_WINDOW_CONFIG
    )

    if df_train.empty or not feature_cols:
        print("No data to train. Exiting.")
    else:
        print(f"Training data shape: {df_train.shape}, Test data shape: {df_test.shape}")
        print(f"Number of features: {len(feature_cols)}")

        train_dataset = StockDataset(df_train, feature_cols, 'Target')
        test_dataset = StockDataset(df_test, feature_cols, 'Target')

        # Note: For true sequential processing of stocks, shuffle=False would be needed,
        # and hidden states managed carefully. Here, shuffle=True is fine as we reset state per batch.
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        input_dim = len(feature_cols)
        output_dim = 1 # Binary classification

        lstm_model = StockLSTM(input_dim, HIDDEN_DIM, N_LSTM_LAYERS, output_dim, DROPOUT_RATE).to(device)
        
        # --- Class Weighting for Imbalance (Optional but good) ---
        # Calculate positive class weight for BCEWithLogitsLoss
        # targets_train = df_train['Target'].values
        # neg_count = np.sum(targets_train == 0)
        # pos_count = np.sum(targets_train == 1)
        # # pos_weight = total_samples / (2 * num_positive_samples) or neg_samples / pos_samples
        # if pos_count > 0:
        #     pos_weight_val = neg_count / pos_count
        #     pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
        #     print(f"Positive class weight for BCEWithLogitsLoss: {pos_weight_val:.2f}")
        # else:
        #     pos_weight = None
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion = nn.BCEWithLogitsLoss() # Simpler version without class weighting for now

        optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)

        # print("Starting LSTM training...")
        # # Pass df_train['Ticker'] if you implement more advanced stateful LSTM
        # lstm_model, best_metrics = train_model(lstm_model, train_loader, test_loader, criterion, optimizer, device, N_EPOCHS, None)

        # print("Evaluating LSTM model...")
        # evaluate_model(lstm_model, test_loader, criterion, device)

        # You can save the model:
        # torch.save(lstm_model.state_dict(), "lstm_stock_model_stocks.pth")
        # To load:
        # model = StockLSTM(...)
        # model.load_state_dict(torch.load("lstm_stock_model.pth"))
        # model.to(device)
        
        results = train_ticker_wise(DATA_DIR, FILE_PATTERN, LAG_FEATURES_CONFIG, SLIDING_WINDOW_CONFIG)