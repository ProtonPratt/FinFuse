# src/models/fused_classifier_daily.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, mean_squared_error
from pathlib import Path
import argparse
import math # For sqrt in RMSE

# Assuming config.py is in src/
from src.config import (
    TICKERS, SENTIMENT_MODEL_NAME, DEVICE, # SENTIMENT_MODEL_NAME might be overridden locally
    MLP_EPOCHS, MLP_BATCH_SIZE, MLP_LEARNING_RATE, MLP_HIDDEN_UNITS, MLP_DROPOUT_RATE
)
from src.sentiment_analyzer import SentimentAnalyzer # Your existing class

# For SentenceTransformer embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers library not found. Some embedding models will not be available.")

# Define input directory (where the output of raw_text_daily_aligner.py is)
ALIGNED_RAW_TEXT_DAILY_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "aligned_raw_text_daily"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "fused_classifier_daily"
MODELS_CACHE_DIR_FUSED = Path(__file__).resolve().parent.parent.parent / "models_cache_fused"

# --- PyTorch Dataset for Fused Features ---
class FusedFeatureDataset(Dataset):
    def __init__(self, stock_features_np, text_features_np, targets_np):
        # Concatenate stock and text features
        fused_features = np.concatenate((stock_features_np, text_features_np), axis=1)
        self.features = torch.tensor(fused_features, dtype=torch.float32)
        self.targets = torch.tensor(targets_np, dtype=torch.long) # For CrossEntropyLoss (0 or 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# --- Simple MLP Model (same as in text_only_classifier_daily.py) ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim=2, dropout_rate=0.3):
        super(SimpleMLP, self).__init__()
        layers = []
        current_dim = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(current_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = units
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- Mean Directional Accuracy (MDA) ---
def calculate_mda(y_true, y_pred):
    """ Simple accuracy for directional prediction (0 or 1) """
    return accuracy_score(y_true, y_pred)

# --- Training and Evaluation Functions (similar to text_only, adapted slightly for clarity) ---
def train_fused_model(model, train_loader, val_loader, criterion, optimizer, device, n_epochs):
    print(f"Starting training on {device} for {n_epochs} epochs...")
    best_val_accuracy = 0.0
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted_train == targets).sum().item()

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted_val = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted_val == targets).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        
        print(f"Epoch {epoch+1}/{n_epochs} => "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"  New best validation accuracy: {best_val_accuracy:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

def evaluate_fused_model(model, test_loader, criterion, device, model_name_str="FusedMLP"):
    model.eval()
    test_loss = 0.0
    all_targets_np = []
    all_predictions_np = []
    all_probabilities_class1_np = [] # For ROC AUC

    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features) # [batch_size, num_classes]
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1) # [batch_size, num_classes]
            _, predicted = torch.max(outputs.data, 1) # [batch_size]

            all_targets_np.extend(targets.cpu().numpy())
            all_predictions_np.extend(predicted.cpu().numpy())
            all_probabilities_class1_np.extend(probs[:, 1].cpu().numpy()) # Prob of class 1

    avg_test_loss = test_loss / len(test_loader)
    
    # Ensure they are numpy arrays for sklearn metrics
    all_targets_np = np.array(all_targets_np)
    all_predictions_np = np.array(all_predictions_np)
    all_probabilities_class1_np = np.array(all_probabilities_class1_np)

    accuracy = accuracy_score(all_targets_np, all_predictions_np)
    mda = calculate_mda(all_targets_np, all_predictions_np) # MDA is same as accuracy for binary 0/1
    
    # RMSE of predicted probabilities vs true labels (0 or 1)
    # This is a bit unusual for classification but sometimes reported.
    # More common to report RMSE if predicting continuous returns.
    # Here, it's RMSE of the probability of class 1 vs actual 0/1.
    rmse_proba = math.sqrt(mean_squared_error(all_targets_np, all_probabilities_class1_np))
    
    # RMSE of predicted classes (0/1) vs true labels (0/1)
    rmse_class = math.sqrt(mean_squared_error(all_targets_np, all_predictions_np))


    report = classification_report(all_targets_np, all_predictions_np, zero_division=0)
    cm = confusion_matrix(all_targets_np, all_predictions_np)
    try:
        roc_auc = roc_auc_score(all_targets_np, all_probabilities_class1_np)
    except ValueError:
        roc_auc = float('nan')

    print(f"\n--- Evaluation for {model_name_str} ---")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Accuracy (MDA): {accuracy:.4f}") # MDA is accuracy here
    print(f"RMSE (Probabilities vs True 0/1): {rmse_proba:.4f}")
    print(f"RMSE (Predicted Class vs True 0/1): {rmse_class:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    return {'accuracy': accuracy, 'mda': mda, 'rmse_proba': rmse_proba, 'rmse_class': rmse_class, 'roc_auc': roc_auc, 'report': report, 'cm': cm}

# --- Main Function ---
def run_fused_classification(aligned_data_path, target_lag_days=1, history_lags_days_str="1_2_3_5", history_lags_list=[1,2,3,5]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_CACHE_DIR_FUSED.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(aligned_data_path)
    except FileNotFoundError:
        print(f"Aligned data file not found: {aligned_data_path}")
        return

    print(f"Loaded data with shape: {df.shape}")

    # --- Target Variable ---
    target_column = f'target_direction_D_plus_{target_lag_days}'
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found.")
        return
    
    # --- Stock Feature Columns ---
    current_day_stock_features = [
        'close_price_day_D', 'open_price_day_D', 'high_price_day_D',
        'low_price_day_D', 'volume_day_D', 'daily_return_Day_D'
    ]
    lagged_stock_features = []
    for lag in history_lags_list:
        lagged_stock_features.extend([
            f'close_price_day_D_minus_{lag}',
            f'daily_return_Day_D_minus_{lag}',
            f'volume_Day_D_minus_{lag}'
        ])
    all_stock_feature_cols = current_day_stock_features + lagged_stock_features
    
    # Ensure all stock feature columns exist
    missing_stock_cols = [col for col in all_stock_feature_cols if col not in df.columns]
    if missing_stock_cols:
        print(f"Missing stock feature columns: {missing_stock_cols}")
        return

    # Drop rows with NaN target, NaN news_text, or NaN in any selected stock feature
    df.dropna(subset=[target_column, 'aggregated_news_text_day_D'] + all_stock_feature_cols, inplace=True)
    df = df[df['aggregated_news_text_day_D'].str.strip() != ""].copy()

    if df.empty:
        print("DataFrame is empty after dropping NaNs.")
        return

    y_series = df[target_column].astype(int)
    if len(y_series.unique()) < 2:
        print("Target variable has fewer than 2 unique classes.")
        return

    # --- Chronological Train/Test Split ---
    df['date_D'] = pd.to_datetime(df['date_D'])
    df_sorted = df.sort_values(by='date_D').reset_index(drop=True)
    
    train_size_ratio, val_size_ratio = 0.7, 0.15
    train_end_idx = int(train_size_ratio * len(df_sorted))
    val_end_idx = train_end_idx + int(val_size_ratio * len(df_sorted))

    df_train_raw = df_sorted.iloc[:train_end_idx]
    df_val_raw = df_sorted.iloc[train_end_idx:val_end_idx]
    df_test_raw = df_sorted.iloc[val_end_idx:]

    y_train_np = df_train_raw[target_column].values.astype(int)
    y_val_np = df_val_raw[target_column].values.astype(int)
    y_test_np = df_test_raw[target_column].values.astype(int)

    # Extract and Scale Stock Features
    X_train_stock_raw = df_train_raw[all_stock_feature_cols].values.astype(np.float32)
    X_val_stock_raw = df_val_raw[all_stock_feature_cols].values.astype(np.float32)
    X_test_stock_raw = df_test_raw[all_stock_feature_cols].values.astype(np.float32)

    stock_scaler = StandardScaler()
    X_train_stock_scaled = stock_scaler.fit_transform(X_train_stock_raw)
    X_val_stock_scaled = stock_scaler.transform(X_val_stock_raw)
    X_test_stock_scaled = stock_scaler.transform(X_test_stock_raw)
    
    # --- Define Text Feature Extractors ---
    text_feature_extractors = {}

    # 1. FinBERT Sentiment
    print(f"Initializing FinBERT Sentiment Analyzer...")
    finbert_analyzer = SentimentAnalyzer(model_name="ProsusAI/finbert", device=DEVICE)
    def get_finbert_sentiment_features(texts):
        sentiment_data = finbert_analyzer.get_sentiment_scores(texts)
        return np.array([[d.get('positive',0), d.get('negative',0), d.get('neutral',0)] for d in sentiment_data], dtype=np.float32)
    text_feature_extractors['FinBERT_Sentiment'] = {
        'extractor_fn': get_finbert_sentiment_features, 'feature_dim': 3
    }

    # 2. FinBERT-Tone Sentiment
    print("Initializing FinBERT-Tone Analyzer...")
    finbert_tone_analyzer = SentimentAnalyzer(model_name="yiyanghkust/finbert-tone", device=DEVICE)
    def get_finbert_tone_features(texts):
        sentiment_data = finbert_tone_analyzer.get_sentiment_scores(texts)
        return np.array([[d.get('positive',0), d.get('negative',0), d.get('neutral',0)] for d in sentiment_data], dtype=np.float32)
    text_feature_extractors['FinBERT_Tone'] = {
        'extractor_fn': get_finbert_tone_features, 'feature_dim': 3
    }

    # 3. FinBERT2 Sentiment
    print("Initializing FinBERT2 Analyzer...")
    finbert2_analyzer = SentimentAnalyzer(model_name="Narsil/finbert2", device=DEVICE)
    def get_finbert2_features(texts):
        sentiment_data = finbert2_analyzer.get_sentiment_scores(texts)
        return np.array([[d.get('positive',0), d.get('negative',0), d.get('neutral',0)] for d in sentiment_data], dtype=np.float32)
    text_feature_extractors['FinBERT2'] = {
        'extractor_fn': get_finbert2_features, 'feature_dim': 3
    }

    # 4. RoBERTa Sentiment
    print("Initializing RoBERTa Sentiment Analyzer...")
    roberta_analyzer = SentimentAnalyzer(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", device=DEVICE)
    def get_roberta_features(texts):
        sentiment_data = roberta_analyzer.get_sentiment_scores(texts)
        return np.array([[d.get('positive',0), d.get('negative',0), d.get('neutral',0)] for d in sentiment_data], dtype=np.float32)
    text_feature_extractors['RoBERTa_Sentiment'] = {
        'extractor_fn': get_roberta_features, 'feature_dim': 3
    }

    # 5. BERT Base
    print("Initializing BERT Base Analyzer...")
    bert_analyzer = SentimentAnalyzer(model_name="bert-base-uncased", device=DEVICE)
    def get_bert_features(texts):
        sentiment_data = bert_analyzer.get_sentiment_scores(texts)
        return np.array([[d.get('positive',0), d.get('negative',0), d.get('neutral',0)] for d in sentiment_data], dtype=np.float32)
    text_feature_extractors['BERT_Base'] = {
        'extractor_fn': get_bert_features, 'feature_dim': 3
    }

    # 6. SentenceTransformer Embeddings (if available)
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        st_model_name = 'all-MiniLM-L6-v2'
        print(f"Initializing SentenceTransformer ({st_model_name})...")
        st_model = SentenceTransformer(st_model_name, device=DEVICE, cache_folder=str(MODELS_CACHE_DIR_FUSED))
        def get_st_embeddings(texts):
            return st_model.encode(texts, show_progress_bar=True, batch_size=128).astype(np.float32)
        text_feature_extractors[f'ST_{st_model_name.replace("/", "_")}'] = {
            'extractor_fn': get_st_embeddings, 'feature_dim': st_model.get_sentence_embedding_dimension()
        }

    # Add memory cleanup function
    def cleanup_gpu_memory():
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    overall_results = []

    for extractor_name, config in text_feature_extractors.items():
        print(f"\n--- Processing and Training Fused Model with: {extractor_name} ---")
        extractor_fn = config['extractor_fn']
        
        print("Extracting text features for TRAIN set...")
        X_train_text_raw = extractor_fn(df_train_raw['aggregated_news_text_day_D'].tolist())
        print("Extracting text features for VALIDATION set...")
        X_val_text_raw = extractor_fn(df_val_raw['aggregated_news_text_day_D'].tolist())
        print("Extracting text features for TEST set...")
        X_test_text_raw = extractor_fn(df_test_raw['aggregated_news_text_day_D'].tolist())

        # Scale Text Features (especially dense embeddings)
        text_scaler = StandardScaler()
        X_train_text_scaled = text_scaler.fit_transform(X_train_text_raw)
        X_val_text_scaled = text_scaler.transform(X_val_text_raw)
        X_test_text_scaled = text_scaler.transform(X_test_text_raw)

        # Create Datasets and DataLoaders
        train_dataset = FusedFeatureDataset(X_train_stock_scaled, X_train_text_scaled, y_train_np)
        val_dataset = FusedFeatureDataset(X_val_stock_scaled, X_val_text_scaled, y_val_np)
        test_dataset = FusedFeatureDataset(X_test_stock_scaled, X_test_text_scaled, y_test_np)

        train_loader = DataLoader(train_dataset, batch_size=MLP_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=MLP_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=MLP_BATCH_SIZE, shuffle=False)

        input_dim_stock = X_train_stock_scaled.shape[1]
        input_dim_text = X_train_text_scaled.shape[1]
        fused_input_dim = input_dim_stock + input_dim_text
        
        output_dim = len(np.unique(y_train_np)) # Should be 2 for binary
        if output_dim < 2:
            print(f"Skipping {extractor_name}, not enough classes in target.")
            continue
            
        model = SimpleMLP(fused_input_dim, MLP_HIDDEN_UNITS, output_dim, MLP_DROPOUT_RATE).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE)

        fused_model_name_str = f"Fused_Stock_And_{extractor_name}"
        trained_model = train_fused_model(model, train_loader, val_loader, criterion, optimizer, DEVICE, MLP_EPOCHS)
        eval_metrics = evaluate_fused_model(trained_model, test_loader, criterion, DEVICE, model_name_str=fused_model_name_str)
        overall_results.append({'model_config': fused_model_name_str, **eval_metrics})
        
        # Clean up GPU memory
        del trained_model, train_loader, val_loader, test_loader
        cleanup_gpu_memory()

    print("\n--- Overall Fused Model Comparison ---")
    results_df = pd.DataFrame(overall_results)
    print(results_df[['model_config', 'accuracy', 'mda', 'rmse_proba', 'rmse_class', 'roc_auc']])
    results_df.to_csv(RESULTS_DIR / f"fused_models_summary_lag{target_lag_days}_hist{history_lags_days_str}.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MLP classifiers using fused stock and text features.")
    parser.add_argument('--target_lag_days', type=int, default=1)
    parser.add_argument('--history_lags', type=str, default="1,2,3,5", help="Comma-separated history lags, e.g., 1,2,3")
    args = parser.parse_args()
    
    history_lags_list_arg = [int(lag.strip()) for lag in args.history_lags.split(',')]
    history_lags_str_arg = "_".join(map(str, history_lags_list_arg)) # For filename consistency

    input_filename = f"all_tickers_raw_text_aligned_daily_lag{args.target_lag_days}_hist{history_lags_str_arg}.csv"
    aligned_data_file_path = ALIGNED_RAW_TEXT_DAILY_DIR / input_filename

    run_fused_classification(
        aligned_data_file_path,
        target_lag_days=args.target_lag_days,
        history_lags_days_str=history_lags_str_arg,
        history_lags_list=history_lags_list_arg
    )