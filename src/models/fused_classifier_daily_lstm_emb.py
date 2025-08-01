# src/models/fused_classifier_daily_with_lstm_emb.py
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
import math

# Assuming config.py is in src/
from src.config import (
    SENTIMENT_MODEL_NAME, DEVICE, # SENTIMENT_MODEL_NAME might be overridden locally
    MLP_EPOCHS, MLP_BATCH_SIZE, MLP_LEARNING_RATE, MLP_HIDDEN_UNITS, MLP_DROPOUT_RATE
)
from src.sentiment_analyzer import SentimentAnalyzer

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# --- START: Copied from previous fused_classifier_daily.py (Dataset, MLP, train, evaluate) ---
# FusedFeatureDataset, SimpleMLP, calculate_mda, train_fused_model, evaluate_fused_model
# (These class/function definitions are identical to the previous fused_classifier_daily.py)
class FusedFeatureDataset(Dataset): # Takes already scaled stock_emb and scaled text_emb
    def __init__(self, stock_embeddings_np, text_features_np, targets_np):
        fused_features = np.concatenate((stock_embeddings_np, text_features_np), axis=1)
        self.features = torch.tensor(fused_features, dtype=torch.float32)
        self.targets = torch.tensor(targets_np, dtype=torch.long)

    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.targets[idx]

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim=2, dropout_rate=0.0):
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
    def forward(self, x): return self.network(x)

def calculate_mda(y_true, y_pred): return accuracy_score(y_true, y_pred)

def train_fused_model(model, train_loader, val_loader, criterion, optimizer, device, n_epochs):
    # ... (identical to previous fused_classifier_daily.py) ...
    print(f"Starting training on {device} for {n_epochs} epochs...")
    best_val_accuracy = 0.0
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
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


def evaluate_fused_model(model, test_loader, criterion, device, model_name_str="FusedMLP_LSTM_Text"):
    # ... (identical to previous fused_classifier_daily.py, ensure RMSE and MDA are calculated and returned) ...
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
    mda = calculate_mda(all_targets_np, all_predictions_np)
    rmse_proba = math.sqrt(mean_squared_error(all_targets_np, np.array(all_probabilities_class1_np)))
    rmse_class = math.sqrt(mean_squared_error(all_targets_np, all_predictions_np))
    report = classification_report(all_targets_np, all_predictions_np, zero_division=0)
    cm = confusion_matrix(all_targets_np, all_predictions_np)
    roc_auc = roc_auc_score(all_targets_np, np.array(all_probabilities_class1_np)) if len(np.unique(all_targets_np)) > 1 else float('nan')

    print(f"\n--- Eval: {model_name_str} ---")
    print(f"Acc (MDA): {accuracy:.4f}, RMSE(proba): {rmse_proba:.4f}, RMSE(class): {rmse_class:.4f}, ROC AUC: {roc_auc:.4f}")
    print("Report:\n", report); print("CM:\n", cm)
    return {'accuracy': accuracy, 'mda': mda, 'rmse_proba': rmse_proba, 'rmse_class': rmse_class, 'roc_auc': roc_auc, 'report': report, 'cm': cm}

# --- END: Copied ---

# Define input directories
ALIGNED_RAW_TEXT_DAILY_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "aligned_raw_text_daily"
LSTM_EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "lstm_stock_embeddings"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "fused_lstm_text_classifier_daily"
MODELS_CACHE_DIR_FUSED_LSTM = Path(__file__).resolve().parent.parent.parent / "models_cache_fused_lstm"


def run_fused_lstm_text_classification(
    aligned_raw_text_data_path, # Path to output of raw_text_daily_aligner.py
    lstm_stock_embeddings_path, # Path to output of generate_lstm_stock_embeddings.py
    target_lag_days=1,
    history_lags_days_str="1_2_3_5" # Used for naming consistency if raw_text_aligner used it
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_CACHE_DIR_FUSED_LSTM.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Loading aligned raw text data from: {aligned_raw_text_data_path}")
        print(f"Loading LSTM stock embeddings from: {lstm_stock_embeddings_path}")
        df_raw_aligned = pd.read_csv(aligned_raw_text_data_path)
        df_lstm_emb = pd.read_csv(lstm_stock_embeddings_path)
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        return

    print(f"Loaded raw aligned data (news+target): {df_raw_aligned.shape}")
    print(f"Loaded LSTM stock embeddings: {df_lstm_emb.shape}")

    # Prepare for merge: convert dates to datetime objects
    df_raw_aligned['date_D'] = pd.to_datetime(df_raw_aligned['date_D'])
    df_lstm_emb['date_D'] = pd.to_datetime(df_lstm_emb['date_D'])

    # Merge raw aligned data (containing news text and target) with LSTM embeddings
    df = pd.merge(df_raw_aligned[['date_D', 'ticker', 'aggregated_news_text_day_D', f'target_direction_D_plus_{target_lag_days}']],
                  df_lstm_emb,
                  on=['date_D', 'ticker'],
                  how='inner') # Use inner join to keep only rows with both news, target, and LSTM embedding

    print(f"Shape after merging news/target with LSTM embeddings: {df.shape}")

    target_column = f'target_direction_D_plus_{target_lag_days}'
    lstm_emb_cols = [col for col in df.columns if col.startswith('lstm_emb_')]
    
    df.dropna(subset=[target_column, 'aggregated_news_text_day_D'] + lstm_emb_cols, inplace=True)
    df = df[df['aggregated_news_text_day_D'].str.strip() != ""].copy()

    if df.empty:
        print("DataFrame is empty after critical NaNs drop (target, news, or LSTM embeddings).")
        return
    y_series = df[target_column].astype(int)
    if len(y_series.unique()) < 2:
        print("Target variable has fewer than 2 unique classes after processing.")
        return

    # --- Chronological Train/Val/Test Split ---
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

    # Extract and Scale LSTM Embeddings (now our "stock features")
    X_train_stock_emb_raw = df_train_raw[lstm_emb_cols].values.astype(np.float32)
    X_val_stock_emb_raw = df_val_raw[lstm_emb_cols].values.astype(np.float32)
    X_test_stock_emb_raw = df_test_raw[lstm_emb_cols].values.astype(np.float32)

    stock_emb_scaler = StandardScaler()
    X_train_stock_emb_scaled = stock_emb_scaler.fit_transform(X_train_stock_emb_raw)
    X_val_stock_emb_scaled = stock_emb_scaler.transform(X_val_stock_emb_raw)
    X_test_stock_emb_scaled = stock_emb_scaler.transform(X_test_stock_emb_raw)
    
    # --- Text Feature Extractors ---
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
        st_model = SentenceTransformer(st_model_name, device=DEVICE, cache_folder=str(MODELS_CACHE_DIR_FUSED_LSTM))
        def get_st_embeddings(texts):
            return st_model.encode(texts, show_progress_bar=True, batch_size=128).astype(np.float32)
        text_feature_extractors[f'ST_{st_model_name.replace("/", "_")}'] = {
            'extractor_fn': get_st_embeddings, 'feature_dim': st_model.get_sentence_embedding_dimension()
        }

    # Optional: Free up memory after each model run
    def cleanup_gpu_memory():
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    overall_results = []
    for extractor_name, config_text in text_feature_extractors.items():
        print(f"\n--- Fusing LSTM Embeddings with Text Features from: {extractor_name} ---")
        extractor_fn = config_text['extractor_fn']
        
        X_train_text_raw = extractor_fn(df_train_raw['aggregated_news_text_day_D'].tolist())
        X_val_text_raw = extractor_fn(df_val_raw['aggregated_news_text_day_D'].tolist())
        X_test_text_raw = extractor_fn(df_test_raw['aggregated_news_text_day_D'].tolist())

        text_scaler = StandardScaler()
        X_train_text_scaled = text_scaler.fit_transform(X_train_text_raw)
        X_val_text_scaled = text_scaler.transform(X_val_text_raw)
        X_test_text_scaled = text_scaler.transform(X_test_text_raw)

        train_dataset = FusedFeatureDataset(X_train_stock_emb_scaled, X_train_text_scaled, y_train_np)
        val_dataset = FusedFeatureDataset(X_val_stock_emb_scaled, X_val_text_scaled, y_val_np)
        test_dataset = FusedFeatureDataset(X_test_stock_emb_scaled, X_test_text_scaled, y_test_np)

        train_loader = DataLoader(train_dataset, batch_size=MLP_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=MLP_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=MLP_BATCH_SIZE, shuffle=False)

        fused_input_dim = X_train_stock_emb_scaled.shape[1] + X_train_text_scaled.shape[1]
        output_dim = len(np.unique(y_train_np))
        if output_dim < 2: continue

        model = SimpleMLP(fused_input_dim, MLP_HIDDEN_UNITS, output_dim, MLP_DROPOUT_RATE).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE)

        fused_model_name_str = f"Fused_LSTMemb_And_{extractor_name}"
        trained_model = train_fused_model(model, train_loader, val_loader, criterion, optimizer, DEVICE, MLP_EPOCHS)
        eval_metrics = evaluate_fused_model(trained_model, test_loader, criterion, DEVICE, model_name_str=fused_model_name_str)
        overall_results.append({'model_config': fused_model_name_str, **eval_metrics})
        
        # Clean up GPU memory
        del trained_model, train_loader, val_loader, test_loader
        cleanup_gpu_memory()

    print("\n--- Overall Fused (LSTM Emb + Text) Model Comparison ---")
    results_df = pd.DataFrame(overall_results)
    if not results_df.empty:
        print(results_df[['model_config', 'accuracy', 'mda', 'rmse_proba', 'rmse_class', 'roc_auc']])
        results_df.to_csv(RESULTS_DIR / f"fused_lstm_text_models_encoders_summary_lag{target_lag_days}_hist{history_lags_days_str}.csv", index=False)
    else:
        print("No results to display.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MLP with Fused LSTM Stock Embeddings and Text Features.")
    parser.add_argument('--target_lag_days', type=int, default=1)
    parser.add_argument('--history_lags_str', type=str, default="1_2_3_5", help="For naming consistency with aligned raw text file.")
    # Path to the output of raw_text_daily_aligner.py
    parser.add_argument('--aligned_raw_text_path', type=str, 
                        default=f"all_tickers_raw_text_aligned_daily_lag1_hist1_2_3_5.csv") 
    # Path to the output of generate_lstm_stock_embeddings.py
    parser.add_argument('--lstm_embeddings_path', type=str, 
                        default="all_tickers_lstm_stock_embeddings.csv")
    args = parser.parse_args()

    # Construct full paths
    raw_text_file_path = ALIGNED_RAW_TEXT_DAILY_DIR / args.aligned_raw_text_path
    lstm_emb_file_path = LSTM_EMBEDDINGS_DIR / args.lstm_embeddings_path
    
    print(f"Using aligned raw text data from: {raw_text_file_path}")
    print(f"Using LSTM stock embeddings from: {lstm_emb_file_path}")

    run_fused_lstm_text_classification(
        raw_text_file_path,
        lstm_emb_file_path,
        target_lag_days=args.target_lag_days,
        history_lags_days_str=args.history_lags_str
    )