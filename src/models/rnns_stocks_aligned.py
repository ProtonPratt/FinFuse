# src/models/lstm_baseline_from_aligned.py
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, mean_squared_error
from pathlib import Path
import argparse
import math

# --- Configuration (can be imported from src.config or defined here) ---
# MLP_... params are not directly used, but LSTM params are
# Assuming config.py might have some of these, or define them:
BATCH_SIZE = 64
N_EPOCHS = 50 # Adjusted from 100 for potentially faster baseline training
LEARNING_RATE = 0.001
LSTM_HIDDEN_DIM_BASELINE = 64 # Can be smaller for a baseline
LSTM_N_LAYERS_BASELINE = 1 # Simpler LSTM
LSTM_DROPOUT_RATE_BASELINE = 0.2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- PyTorch Dataset (Simplified for pre-selected features) ---
class AlignedStockDataset(Dataset):
    def __init__(self, features_np, targets_np):
        # Features are already selected and scaled numerical stock features
        self.features = torch.tensor(features_np, dtype=torch.float32)
        self.targets = torch.tensor(targets_np, dtype=torch.long) # For CrossEntropyLoss

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # LSTM expects input as (batch, seq_len, input_size) or (seq_len, batch, input_size)
        # Here, each day's features are treated as a sequence of length 1
        return self.features[idx].unsqueeze(0), self.targets[idx] # Add seq_len dim: [1, num_features]

# --- LSTM Model Definition (Same as StockLSTM from previous scripts) ---
class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(BaselineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim) # output_dim should be num_classes (e.g., 2)

    def forward(self, x, hidden_state=None):
        lstm_out, (h_n, c_n) = self.lstm(x, hidden_state)
        out_last_step = lstm_out[:, -1, :] # Get the last time step's output
        out_dropout = self.dropout(out_last_step)
        out_final = self.fc(out_dropout)
        return out_final # No need to return hidden_state for simple training loop

# --- Training Function (adapted from previous MLP/Fused scripts) ---
def train_baseline_lstm(model, train_loader, val_loader, criterion, optimizer, device, n_epochs):
    print(f"Starting LSTM baseline training on {device} for {n_epochs} epochs...")
    best_val_accuracy = 0.0
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features) # LSTM forward pass
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted_train == targets).sum().item()

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted_val = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted_val == targets).sum().item()
        
        train_accuracy = correct_train / total_train if total_train > 0 else 0
        val_accuracy = correct_val / total_val if total_val > 0 else 0
        print(f"Epoch {epoch+1}/{n_epochs} => Train Loss: {train_loss/len(train_loader):.4f}, TA: {train_accuracy:.4f} | Val Loss: {val_loss/len(val_loader):.4f}, VA: {val_accuracy:.4f}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"  New best VA: {best_val_accuracy:.4f}")
            
    if best_model_state: model.load_state_dict(best_model_state)
    return model

# --- Evaluation Function (adapted from previous MLP/Fused scripts) ---
def evaluate_baseline_lstm(model, test_loader, criterion, device, model_name_str="LSTM_Stock_Only_Baseline"):
    model.eval()
    test_loss = 0.0
    all_targets_np, all_predictions_np, all_probabilities_class1_np = [], [], []
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_targets_np.extend(targets.cpu().numpy())
            all_predictions_np.extend(predicted.cpu().numpy())
            all_probabilities_class1_np.extend(probs[:, 1].cpu().numpy())

    all_targets_np, all_predictions_np = np.array(all_targets_np), np.array(all_predictions_np)
    accuracy = accuracy_score(all_targets_np, all_predictions_np)
    mda = accuracy # For binary 0/1, MDA is accuracy
    rmse_proba = math.sqrt(mean_squared_error(all_targets_np, np.array(all_probabilities_class1_np)))
    rmse_class = math.sqrt(mean_squared_error(all_targets_np, all_predictions_np))
    report = classification_report(all_targets_np, all_predictions_np, zero_division=0)
    cm = confusion_matrix(all_targets_np, all_predictions_np)
    roc_auc = roc_auc_score(all_targets_np, np.array(all_probabilities_class1_np)) if len(np.unique(all_targets_np)) > 1 else float('nan')

    print(f"\n--- Evaluation for {model_name_str} ---")
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    print(f"Accuracy (MDA): {accuracy:.4f}")
    print(f"RMSE (Probabilities vs True 0/1): {rmse_proba:.4f}")
    print(f"RMSE (Predicted Class vs True 0/1): {rmse_class:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    return {'accuracy': accuracy, 'mda': mda, 'rmse_proba': rmse_proba, 'rmse_class': rmse_class, 'roc_auc': roc_auc, 'report': report, 'cm': cm}

# --- Main Execution ---
ALIGNED_DATA_INPUT_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "aligned_raw_text_daily"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "lstm_stock_only_baseline"

def run_lstm_stock_baseline(aligned_data_path, target_lag_days=1, history_lags_list=[1,2,3,5]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df = pd.read_csv(aligned_data_path)
    except FileNotFoundError:
        print(f"Aligned data file not found: {aligned_data_path}")
        return

    print(f"Loaded aligned data with shape: {df.shape}")

    # --- Define Stock Feature Columns (current day + historical lags) ---
    current_day_features = [
        'close_price_day_D', 'open_price_day_D', 'high_price_day_D',
        'low_price_day_D', 'volume_day_D', 'daily_return_Day_D'
    ]
    lagged_features = []
    for lag in history_lags_list:
        lagged_features.extend([
            f'close_price_day_D_minus_{lag}',
            f'daily_return_Day_D_minus_{lag}',
            f'volume_Day_D_minus_{lag}'
        ])
    all_stock_feature_cols = current_day_features + lagged_features
    target_column = f'target_direction_D_plus_{target_lag_days}'

    # Ensure columns exist and drop NaNs
    missing_cols = [col for col in all_stock_feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing stock feature columns: {missing_cols}")
        return
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found.")
        return
    
    df.dropna(subset=all_stock_feature_cols + [target_column], inplace=True)
    if df.empty:
        print("DataFrame is empty after dropping NaNs from stock features/target.")
        return
    
    print(f"Data shape after NaN drop for features: {df.shape}")
    y_series = df[target_column].astype(int)
    if len(y_series.unique()) < 2:
        print("Target variable has fewer than 2 unique classes.")
        return

    # --- Chronological Train/Val/Test Split ---
    df['date_D'] = pd.to_datetime(df['date_D'])
    df_sorted = df.sort_values(by='date_D').reset_index(drop=True)
    
    train_size_ratio, val_size_ratio = 0.7, 0.15
    train_end_idx = int(train_size_ratio * len(df_sorted))
    val_end_idx = train_end_idx + int(val_size_ratio * len(df_sorted))

    df_train = df_sorted.iloc[:train_end_idx]
    df_val = df_sorted.iloc[train_end_idx:val_end_idx]
    df_test = df_sorted.iloc[val_end_idx:]

    # Extract features and target
    X_train_raw = df_train[all_stock_feature_cols].values.astype(np.float32)
    y_train_np = df_train[target_column].values.astype(int)
    X_val_raw = df_val[all_stock_feature_cols].values.astype(np.float32)
    y_val_np = df_val[target_column].values.astype(int)
    X_test_raw = df_test[all_stock_feature_cols].values.astype(np.float32)
    y_test_np = df_test[target_column].values.astype(int)

    # Scale Stock Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Create Datasets and DataLoaders
    train_dataset = AlignedStockDataset(X_train_scaled, y_train_np)
    val_dataset = AlignedStockDataset(X_val_scaled, y_val_np)
    test_dataset = AlignedStockDataset(X_test_scaled, y_test_np)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train_scaled.shape[1]
    output_dim = len(np.unique(y_train_np)) # Should be 2
    if output_dim < 2:
        print("Not enough classes in target for training after split.")
        return

    model = BaselineLSTM(input_dim, LSTM_HIDDEN_DIM_BASELINE, LSTM_N_LAYERS_BASELINE, output_dim, LSTM_DROPOUT_RATE_BASELINE).to(DEVICE)
    criterion = nn.CrossEntropyLoss() # Suitable for multi-class output (even if binary)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training LSTM Stock-Only Baseline (Input Dim: {input_dim})...")
    trained_model = train_baseline_lstm(model, train_loader, val_loader, criterion, optimizer, DEVICE, N_EPOCHS)
    
    model_name_str = f"LSTM_StockOnly_Lag{target_lag_days}_Hist{'_'.join(map(str,history_lags_list))}"
    eval_metrics = evaluate_baseline_lstm(trained_model, test_loader, criterion, DEVICE, model_name_str=model_name_str)
    
    # Save results
    results_df = pd.DataFrame([{'model_config': model_name_str, **eval_metrics}])
    hist_str_fn = "_".join(map(str, history_lags_list))
    results_filename = f"lstm_stock_only_baseline_lag{target_lag_days}_hist{hist_str_fn}_summary.csv"
    results_df.to_csv(RESULTS_DIR / results_filename, index=False)
    print(f"\nSaved LSTM baseline results to {RESULTS_DIR / results_filename}")
    print(results_df[['model_config', 'accuracy', 'mda', 'rmse_proba', 'rmse_class', 'roc_auc']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LSTM baseline using stock features from aligned data.")
    parser.add_argument('--target_lag_days', type=int, default=1)
    parser.add_argument('--history_lags', type=str, default="1,2,3,5", help="Comma-separated history lags (e.g., 1,2,3).")
    args = parser.parse_args()

    history_lags_list_arg = [int(lag.strip()) for lag in args.history_lags.split(',')]
    history_lags_str_fn = "_".join(map(str, history_lags_list_arg))

    # Construct input filename based on args
    input_filename = f"all_tickers_raw_text_aligned_daily_lag{args.target_lag_days}_hist{history_lags_str_fn}.csv"
    aligned_data_file_path = ALIGNED_DATA_INPUT_DIR / input_filename
    
    print(f"Using aligned data from: {aligned_data_file_path}")

    run_lstm_stock_baseline(
        aligned_data_file_path,
        target_lag_days=args.target_lag_days,
        history_lags_list=history_lags_list_arg
    )