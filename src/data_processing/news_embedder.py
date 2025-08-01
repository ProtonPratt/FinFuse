# src/data_processing/news_embedder.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# Assuming SentenceTransformer is installed: pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# Assuming your SentimentAnalyzer class and config are accessible
from src.sentiment_analyzer import SentimentAnalyzer # If you put it in src/
from src.config import SENTIMENT_MODEL_NAME, DEVICE # Or define directly for now


ALIGNED_RAW_TEXT_DAILY_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "aligned_raw_text_daily"
EMBEDDED_NEWS_DAILY_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "embedded_news_daily"

# Define text feature extraction configurations
TEXT_EXTRACTORS_CONFIG = {
    "finbert_sentiment": {
        "type": "sentiment",
        "model_name": "ProsusAI/finbert", # Make sure this is a sentiment model
        "output_cols_prefix": "finbert_sent"
    },
    "minilm_embedding": {
        "type": "embedding",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "output_cols_prefix": "minilm_emb"
    },
    # Add more here, e.g., for bert-base-uncased embeddings
    "bert_base_embedding": {
         "type": "embedding",
         "model_name": "sentence-transformers/bert-base-nli-mean-tokens", # A bert-base fine-tuned for sentence embeddings
         "output_cols_prefix": "bert_base_emb"
    }
}

def add_text_features(input_filepath, text_column='aggregated_news_text_day_D'):
    """
    Loads aligned data, extracts text features using configured models, and saves the enriched data.
    """
    EMBEDDED_NEWS_DAILY_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"Input file not found: {input_filepath}")
        return

    print(f"Loaded data for embedding: {input_filepath}, shape: {df.shape}")
    if text_column not in df.columns:
        print(f"Text column '{text_column}' not found. Available columns: {df.columns.tolist()}")
        return

    # Ensure text column is string and handle NaNs by filling with empty string
    df[text_column] = df[text_column].astype(str).fillna('')
    texts_to_process = df[text_column].tolist()

    df_enriched = df.copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for extractor_name, config in TEXT_EXTRACTORS_CONFIG.items():
        print(f"Extracting features using: {extractor_name} ({config['model_name']})...")
        if config['type'] == "sentiment":
            analyzer = SimpleSentimentAnalyzer(model_name=config['model_name'], device=device)
            # Batch processing for sentiment analyzer
            batch_size = 32  # Adjust as needed
            all_sentiment_scores = []
            for i in tqdm(range(0, len(texts_to_process), batch_size), desc=f"Sent. Analysis {extractor_name}"):
                batch_texts = texts_to_process[i:i + batch_size]
                all_sentiment_scores.extend(analyzer.get_sentiment_scores_batch(batch_texts))
            
            sentiment_df = pd.DataFrame(all_sentiment_scores)
            # Standardize column names (e.g., positive, negative, neutral)
            # The pipeline might return labels with different casings or orders.
            # Common labels for FinBERT are 'positive', 'negative', 'neutral'.
            expected_labels = ['positive', 'negative', 'neutral']
            for label in expected_labels:
                if label in sentiment_df.columns:
                    df_enriched[f"{config['output_cols_prefix']}_{label}"] = sentiment_df[label]
                else: # Handle if a label is missing, e.g. some models only output pos/neg
                    df_enriched[f"{config['output_cols_prefix']}_{label}"] = 0.0


        elif config['type'] == "embedding":
            model_st = SentenceTransformer(config['model_name'], device=device)
            embeddings = model_st.encode(texts_to_process, show_progress_bar=True, batch_size=128)
            embedding_dim = embeddings.shape[1]
            for i in range(embedding_dim):
                df_enriched[f"{config['output_cols_prefix']}_{i}"] = embeddings[:, i]
        
        print(f"Done with {extractor_name}.")

    # Construct output filename
    input_basename = Path(input_filepath).stem
    output_filename = f"{input_basename}_with_text_features.csv"
    output_path = EMBEDDED_NEWS_DAILY_DIR / output_filename
    df_enriched.to_csv(output_path, index=False)
    print(f"Saved enriched data with text features to: {output_path}. Shape: {df_enriched.shape}")
    print(df_enriched.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add text embeddings/sentiment scores to aligned daily data.")
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help="Path to the input CSV file from raw_text_daily_aligner.py (e.g., all_tickers_raw_text_aligned_daily_lag1_hist1_2_3_5.csv)"
    )
    args = parser.parse_args()

    input_full_path = ALIGNED_RAW_TEXT_DAILY_DIR / args.input_file
    add_text_features(input_full_path)