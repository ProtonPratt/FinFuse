# src/models/text_only_classifier_daily.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from pathlib import Path
import argparse

# Assuming config.py is in src/
from src.config import (
    TICKERS, SENTIMENT_MODEL_NAME, DEVICE,
    MLP_EPOCHS, MLP_BATCH_SIZE, MLP_LEARNING_RATE, MLP_HIDDEN_UNITS, MLP_DROPOUT_RATE
)
from src.sentiment_analyzer import SentimentAnalyzer # Your existing class

# For SentenceTransformer embeddings (if you want to compare)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers library not found. Some embedding models will not be available.")


# Define input directory (where the output of raw_text_daily_aligner.py is)
ALIGNED_RAW_TEXT_DAILY_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "aligned_raw_text_daily"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "text_only_classifier_daily"
MODELS_CACHE_DIR_TEXT_ONLY = Path(__file__).resolve().parent.parent.parent / "models_cache_text_only" # Separate cache

print(f"Aligned data directory: {ALIGNED_RAW_TEXT_DAILY_DIR}")

# --- PyTorch Dataset for Text Features ---
class TextFeatureDataset(Dataset):
    def __init__(self, text_features_np, targets_np):
        self.features = torch.tensor(text_features_np, dtype=torch.float32)
        self.targets = torch.tensor(targets_np, dtype=torch.long) # For CrossEntropyLoss (0 or 1)
        # If using BCEWithLogitsLoss, targets should be float32 and unsqueezed:
        # self.targets = torch.tensor(targets_np, dtype=torch.float32).unsqueeze(1)


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# --- Simple MLP Model ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim=2, dropout_rate=0.3): # output_dim=2 for CrossEntropyLoss
        super(SimpleMLP, self).__init__()
        layers = []
        current_dim = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(current_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = units
        layers.append(nn.Linear(current_dim, output_dim))
        # No final sigmoid/softmax if using CrossEntropyLoss or BCEWithLogitsLoss
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- Training and Evaluation Functions ---
def train_mlp_model(model, train_loader, val_loader, criterion, optimizer, device, n_epochs):
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
            
            # For CrossEntropyLoss, outputs are logits [batch_size, num_classes]
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted_train == targets).sum().item()

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_targets = []
        all_val_preds = []
        all_val_probs = []

        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # For CrossEntropyLoss
                probs_val = torch.softmax(outputs, dim=1) # Get probabilities
                _, predicted_val = torch.max(outputs.data, 1)
                
                total_val += targets.size(0)
                correct_val += (predicted_val == targets).sum().item()
                all_val_targets.extend(targets.cpu().numpy())
                all_val_preds.extend(predicted_val.cpu().numpy())
                all_val_probs.extend(probs_val.cpu().numpy())


        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        
        print(f"Epoch {epoch+1}/{n_epochs} => "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy() # Deep copy
            print(f"  New best validation accuracy: {best_val_accuracy:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

def evaluate_mlp_model(model, test_loader, criterion, device, model_name_str="MLP"):
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probabilities_class1 = [] # For ROC AUC

    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities_class1.extend(probs[:, 1].cpu().numpy()) # Prob of class 1

    avg_test_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    report = classification_report(all_targets, all_predictions, zero_division=0)
    cm = confusion_matrix(all_targets, all_predictions)
    
    try:
        roc_auc = roc_auc_score(all_targets, all_probabilities_class1)
    except ValueError: # Handle cases like only one class in targets
        roc_auc = float('nan')


    print(f"\n--- Evaluation for {model_name_str} ---")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    return {'accuracy': accuracy, 'roc_auc': roc_auc, 'report': report, 'cm': cm}

# --- Main Function ---
def run_text_only_classification(aligned_data_path, target_lag_days=1, history_lags_days_str="1_2_3_5"):
    """
    Trains and evaluates MLP classifiers using only text features from various models.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_CACHE_DIR_TEXT_ONLY.mkdir(parents=True, exist_ok=True) # For SentimentAnalyzer cache

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
    
    # Drop rows with NaN target or NaN aggregated_news_text
    df.dropna(subset=[target_column, 'aggregated_news_text_day_D'], inplace=True)
    df = df[df['aggregated_news_text_day_D'].str.strip() != ""].copy() # Ensure news text is not empty

    if df.empty:
        print("DataFrame is empty after dropping NaNs from target/news_text.")
        return

    y_series = df[target_column].astype(int)
    if len(y_series.unique()) < 2:
        print("Target variable has fewer than 2 unique classes. Cannot train.")
        return

    # --- Chronological Train/Test Split (applied to indices, then to features) ---
    df['date_D'] = pd.to_datetime(df['date_D'])
    df_sorted = df.sort_values(by='date_D').reset_index(drop=True)
    
    train_size_ratio = 0.7 # Using 70% for train, 15% for val, 15% for test
    val_size_ratio = 0.15

    train_end_idx = int(train_size_ratio * len(df_sorted))
    val_end_idx = train_end_idx + int(val_size_ratio * len(df_sorted))

    df_train_raw = df_sorted.iloc[:train_end_idx]
    df_val_raw = df_sorted.iloc[train_end_idx:val_end_idx]
    df_test_raw = df_sorted.iloc[val_end_idx:]

    y_train = df_train_raw[target_column].values.astype(int)
    y_val = df_val_raw[target_column].values.astype(int)
    y_test = df_test_raw[target_column].values.astype(int)

    # --- Define Text Feature Extractors ---
    text_feature_extractors = {}

    # 1. FinBERT Sentiment (using your SentimentAnalyzer)
    print(f"Initializing FinBERT Sentiment Analyzer...")
    finbert_analyzer = SentimentAnalyzer(model_name="ProsusAI/finbert", device=DEVICE)
    def get_finbert_sentiment_features(texts):
        sentiment_data = finbert_analyzer.get_sentiment_scores(texts)
        return np.array([[d.get('positive',0), d.get('negative',0), d.get('neutral',0)] for d in sentiment_data])
    text_feature_extractors['FinBERT_Sentiment'] = {
        'extractor_fn': get_finbert_sentiment_features, 'feature_dim': 3
    }

    # 2. FinBERT-Tone Sentiment
    print("Initializing FinBERT-Tone Analyzer...")
    finbert_tone_analyzer = SentimentAnalyzer(model_name="yiyanghkust/finbert-tone", device=DEVICE)
    def get_finbert_tone_features(texts):
        sentiment_data = finbert_tone_analyzer.get_sentiment_scores(texts)
        return np.array([[d.get('positive',0), d.get('negative',0), d.get('neutral',0)] for d in sentiment_data])
    text_feature_extractors['FinBERT_Tone'] = {
        'extractor_fn': get_finbert_tone_features, 'feature_dim': 3
    }

    # 3. FinBERT2 Sentiment
    print("Initializing FinBERT2 Analyzer...")
    finbert2_analyzer = SentimentAnalyzer(model_name="Narsil/finbert2", device=DEVICE)
    def get_finbert2_features(texts):
        sentiment_data = finbert2_analyzer.get_sentiment_scores(texts)
        return np.array([[d.get('positive',0), d.get('negative',0), d.get('neutral',0)] for d in sentiment_data])
    text_feature_extractors['FinBERT2'] = {
        'extractor_fn': get_finbert2_features, 'feature_dim': 3
    }

    # 4. RoBERTa Sentiment
    print("Initializing RoBERTa Sentiment Analyzer...")
    roberta_analyzer = SentimentAnalyzer(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", device=DEVICE)
    def get_roberta_features(texts):
        sentiment_data = roberta_analyzer.get_sentiment_scores(texts)
        return np.array([[d.get('positive',0), d.get('negative',0), d.get('neutral',0)] for d in sentiment_data])
    text_feature_extractors['RoBERTa_Sentiment'] = {
        'extractor_fn': get_roberta_features, 'feature_dim': 3
    }

    # 5. BERT Base
    print("Initializing BERT Base Analyzer...")
    bert_analyzer = SentimentAnalyzer(model_name="bert-base-uncased", device=DEVICE)
    def get_bert_features(texts):
        sentiment_data = bert_analyzer.get_sentiment_scores(texts)
        return np.array([[d.get('positive',0), d.get('negative',0), d.get('neutral',0)] for d in sentiment_data])
    text_feature_extractors['BERT_Base'] = {
        'extractor_fn': get_bert_features, 'feature_dim': 3
    }

    # # 6. BERT Large (optional - more resource intensive)
    # print("Initializing BERT Large Analyzer...")
    # bert_large_analyzer = SentimentAnalyzer(model_name="bert-large-uncased", device=DEVICE)
    # def get_bert_large_features(texts):
    #     sentiment_data = bert_large_analyzer.get_sentiment_scores(texts)
    #     return np.array([[d.get('positive',0), d.get('negative',0), d.get('neutral',0)] for d in sentiment_data])
    # text_feature_extractors['BERT_Large'] = {
    #     'extractor_fn': get_bert_large_features, 'feature_dim': 3
    # }

    # 7. SentenceTransformer Embeddings (if available)
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        st_model_name = 'all-MiniLM-L6-v2'
        print(f"Initializing SentenceTransformer ({st_model_name})...")
        st_model = SentenceTransformer(st_model_name, device=DEVICE, cache_folder=str(MODELS_CACHE_DIR_TEXT_ONLY))
        def get_st_embeddings(texts):
            return st_model.encode(texts, show_progress_bar=True, batch_size=128)
        text_feature_extractors[f'ST_{st_model_name.replace("/", "_")}'] = {
            'extractor_fn': get_st_embeddings, 'feature_dim': st_model.get_sentence_embedding_dimension()
        }

    overall_results = []

    # --- Iterate through each text feature type ---
    for extractor_name, config in text_feature_extractors.items():
        print(f"\n--- Processing and Training with: {extractor_name} ---")
        extractor_fn = config['extractor_fn']
        
        print("Extracting text features for TRAIN set...")
        X_train_text_raw = extractor_fn(df_train_raw['aggregated_news_text_day_D'].tolist())
        print("Extracting text features for VALIDATION set...")
        X_val_text_raw = extractor_fn(df_val_raw['aggregated_news_text_day_D'].tolist())
        print("Extracting text features for TEST set...")
        X_test_text_raw = extractor_fn(df_test_raw['aggregated_news_text_day_D'].tolist())

        # Scaling (optional for sentiment scores 0-1, recommended for dense embeddings)
        scaler = StandardScaler()
        X_train_text_scaled = scaler.fit_transform(X_train_text_raw)
        X_val_text_scaled = scaler.transform(X_val_text_raw)
        X_test_text_scaled = scaler.transform(X_test_text_raw)

        # Create Datasets and DataLoaders
        train_dataset = TextFeatureDataset(X_train_text_scaled, y_train)
        val_dataset = TextFeatureDataset(X_val_text_scaled, y_val)
        test_dataset = TextFeatureDataset(X_test_text_scaled, y_test)

        train_loader = DataLoader(train_dataset, batch_size=MLP_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=MLP_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=MLP_BATCH_SIZE, shuffle=False)

        # Initialize and train MLP
        input_dim = config['feature_dim']
        # output_dim = 1 if using BCEWithLogitsLoss else len(np.unique(y_train)) # For CrossEntropy, num_classes
        output_dim = len(np.unique(y_train)) # Assuming y_train contains 0 and 1, so 2 classes
        if output_dim < 2:
            print(f"Skipping {extractor_name}, not enough classes in target for training.")
            continue

        model = SimpleMLP(input_dim, MLP_HIDDEN_UNITS, output_dim, MLP_DROPOUT_RATE).to(DEVICE)
        
        # Use CrossEntropyLoss for multi-class (even if binary, it works)
        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE)

        trained_model = train_mlp_model(model, train_loader, val_loader, criterion, optimizer, DEVICE, MLP_EPOCHS)
        
        eval_metrics = evaluate_mlp_model(trained_model, test_loader, criterion, DEVICE, model_name_str=extractor_name)
        overall_results.append({'model': extractor_name, **eval_metrics})

        # Save model (optional)
        torch.save(trained_model.state_dict(), RESULTS_DIR / f"mlp_{extractor_name}_lag{target_lag_days}_hist{history_lags_days_str}.pth")

    # --- Print and Save Summary ---
    print("\n--- Overall Text-Only Model Comparison ---")
    results_df = pd.DataFrame(overall_results)
    print(results_df[['model', 'accuracy', 'roc_auc']])
    results_df.to_csv(RESULTS_DIR / f"text_only_models_summary_lag{target_lag_days}_hist{history_lags_days_str}.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MLP classifiers using only text features from daily news.")
    parser.add_argument(
        '--target_lag_days', type=int, default=1,
        help="Target lag days for prediction (must match the aligned data file)."
    )
    parser.add_argument(
        '--history_lags_str', type=str, default="1_2_3_5", # Match default from aligner's output
        help="String representing history lags used in data alignment (e.g., '1_2_3_5')."
    )
    args = parser.parse_args()

    input_filename = f"all_tickers_raw_text_aligned_daily_lag{args.target_lag_days}_hist{args.history_lags_str}.csv"
    aligned_data_file_path = ALIGNED_RAW_TEXT_DAILY_DIR / input_filename

    run_text_only_classification(
        aligned_data_file_path,
        target_lag_days=args.target_lag_days,
        history_lags_days_str=args.history_lags_str
    )