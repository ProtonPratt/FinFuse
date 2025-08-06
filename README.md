# Stock-Price-Movement-Prediction-from-news-and-stock-data
This repository contains the code and results for a project on stock price movement prediction using news and stock data.

## Project Overview

This project investigates the predictability of daily stock price movements by combining historical stock data with financial news analysis. We systematically evaluate various machine learning and deep learning models, from simple baselines to more complex fusion architectures, to understand the predictive power of each modality (stock data, news text) and their combination.

The core of the project revolves around three main experimental setups:
1.  **Stock-Only Models**: Baselines using only historical stock data (OHLCV, returns, etc.) to predict future price movements.
2.  **Text-Only Models**: Models that use only the textual information from financial news to make predictions.
3.  **Fused Models**: Models that combine both stock and text data to make a final prediction.

## Architecture

The project's architecture is designed as a modular pipeline that flows from data collection and alignment to feature engineering, modeling, and finally, evaluation. This structure allows for easy experimentation with different components and models.

```
+---------------------+      +-----------------------+      +----------------------+
|   Data Ingestion    |----->|  Feature Engineering  |----->|       Modeling       |
| & Alignment         |      |                       |      |                      |
+---------------------+      +-----------------------+      +----------------------+
| - Stock Data (YFinance) |      | - Stock Features:     |      | - Stock-Only Models  |
| - News Data (AlphaVantage)|      |   - Raw Historical    |      |   - XGBoost          |
| - Align by Date/Ticker  |      |   - LSTM Embeddings   |      |   - LSTM             |
+---------------------+      | - Text Features:      |      |                      |
                             |   - BERT (Sentiment)  |      | - Text-Only Models   |
                             |   - FinBERT (Tone)    |      |   - MLP              |
                             |   - RoBERTa (Sentiment)|      |                      |
                             |   - Dense Embeddings  |      | - Fused Models       |
                             +-----------------------+      |   - MLP (Early Fusion)|
                                                            +----------------------+
                                                                       |
                                                                       v
+---------------------+
|      Evaluation     |
+---------------------+
| - Accuracy (MDA)    |
| - ROC AUC           |
| - F1-Score          |
+---------------------+
```

### 1. Data Ingestion and Alignment

The first stage of the pipeline involves collecting and aligning the data from different sources:

*   **Stock Data**: Daily Open, High, Low, Close, and Volume (OHLCV) data is sourced from **Yahoo Finance**.
*   **News Data**: Financial news headlines and summaries are collected from **Alpha Vantage**.
*   **Alignment**: The stock and news data are aligned by **ticker and date**. For each trading day, all news articles published on that day are aggregated into a single document. The prediction target is the binary direction of the stock's closing price on the next trading day.

### 2. Feature Engineering

Once the data is aligned, we extract features from both the stock and text data:

#### Stock Features

Two types of stock features are engineered:

1.  **Raw Historical Features**: These include the current day's OHLCV, daily return, and lagged values (1, 2, 3, and 5 days prior) of closing price, daily return, and volume. These features are used in the XGBoost baseline and one of the fusion models.
2.  **LSTM Stock Embeddings**: To capture the temporal dynamics of the stock data, a pre-trained LSTM model is used to generate embeddings. The final hidden state of the LSTM for a given day is used as a learned representation of the stock's history up to that point.

#### Text Features

We use a variety of pre-trained BERT-based models to transform the aggregated daily news text into numerical features:

*   **Domain-Specific Models**:
    *   `ProsusAI/finbert` ("FinBERT Sentiment"): For sentiment analysis.
    *   `yiyanghkust/finbert-tone` ("FinBERT Tone"): For financial tone analysis.
    *   `Narsil/finbert2` ("FinBERT2"): A larger, more powerful version of FinBERT.
*   **General Models**:
    *   `cardiffnlp/twitter-roberta-base-sentiment-latest` ("RoBERTa Sentiment"): A robust sentiment analysis model.
    *   `bert-base-uncased` ("BERT Base"): A general-purpose BERT model.
    *   `all-MiniLM-L6-v2`: A sentence-transformer model for generating dense semantic embeddings.

### 3. Modeling

We explore three categories of models to assess the predictive power of each modality:

1.  **Stock-Only Baselines**:
    *   **XGBoost**: A powerful gradient boosting model trained on the raw historical stock features.
    *   **LSTM**: A Long Short-Term Memory network trained on the aligned historical stock features to establish a deep learning baseline.

2.  **Text-Only Models**:
    *   **MLP**: A Multi-Layer Perceptron (MLP) is trained on the text features extracted from the various BERT models. This helps to assess the predictive power of news text in isolation.

3.  **Fused Models**:
    *   **Early Fusion MLP**: We combine the stock and text features by concatenating them and feeding them into an MLP. We experiment with two types of fusion:
        1.  **Raw Stock + Text**: Fusing the raw historical stock features with the text features.
        2.  **LSTM Embeddings + Text**: Fusing the LSTM-generated stock embeddings with the text features.

### 4. Training and Evaluation

All models are trained and evaluated using a chronological 80/20 train/test split (or 70/15/15 for MLPs). Numerical features are standardized using `StandardScaler`. The performance of each model is assessed using the following metrics:

*   **Accuracy (Mean Directional Accuracy - MDA)**
*   **ROC AUC**
*   **F1-Score (Macro)**
*   **Confusion Matrix**

## Experimental Results

The following tables summarize the performance of the different models. All results are for predicting the stock direction for the next trading day (`target_lag_days=1`).

### Baselines: Stock Features Only

| Model Configuration | Accuracy (MDA) | ROC AUC | F1 (Macro) |
| :--- | :--- | :--- | :--- |
| XGBoost (Eng. Hist. Stock Feat.) | 0.5386 | 0.5177 | 0.51 |
| LSTM (Aligned Hist. Stock Feat.) | 0.5023 | 0.5077 | 0.50 |

### Text-Only Models (MLP)

| Model (Text Feature) | Accuracy (MDA) | ROC AUC | F1 (Macro) |
| :--- | :--- | :--- | :--- |
| FinBERT Sentiment | 0.4914 | 0.5336 | 0.44 |
| FinBERT Tone | 0.4963 | 0.5000 | 0.33 |
| FinBERT2 | 0.4988 | 0.5248 | 0.47 |
| RoBERTa Sentiment | 0.5012 | 0.4776 | 0.49 |
| BERT Base | 0.4963 | 0.5000 | 0.33 |
| ST_all-MiniLM-L6-v2 | 0.5110 | 0.5183 | 0.51 |

### Early Fusion: Raw Historical Stock Features + Text Features (MLP)

| Fused Model (Raw Stock + Text Type) | Accuracy (MDA) | ROC AUC | F1 (Macro) |
| :--- | :--- | :--- | :--- |
| Raw Stock + FinBERT Sentiment | 0.4597 | 0.4761 | 0.45 |
| Raw Stock + FinBERT Tone | 0.5257 | 0.5230 | 0.51 |
| Raw Stock + FinBERT2 | 0.4768 | 0.4822 | 0.48 |
| Raw Stock + RoBERTa Sentiment | 0.4768 | 0.4754 | 0.46 |
| Raw Stock + BERT Base | 0.5208 | 0.5092 | 0.51 |
| Raw Stock + ST_all-MiniLM-L6-v2 | 0.4914 | 0.5133 | 0.49 |

### Early Fusion: LSTM Stock Embeddings + Text Features (MLP)

| Fused Model (LSTM Emb. + Text Type) | Accuracy (MDA) | ROC AUC | F1 (Macro) |
| :--- | :--- | :--- | :--- |
| LSTM Emb + FinBERT Sentiment | 0.5355 | 0.5340 | 0.53 |
| LSTM Emb + FinBERT Tone | 0.5061 | 0.5159 | 0.49 |
| LSTM Emb + FinBERT2 | 0.5355 | 0.5461 | 0.53 |
| LSTM Emb + RoBERTa Sentiment | 0.5110 | 0.5021 | 0.51 |
| **LSTM Emb + BERT Base** | **0.5477** | **0.5323** | **0.55** |
| LSTM Emb + ST_all-MiniLM-L6-v2 | 0.5012 | 0.5354 | 0.50 |

## Overall Conclusions

The key takeaways from this project are:

*   **Difficulty of the Task**: Predicting daily stock direction is extremely challenging, with most models performing only slightly better than random chance.
*   **Value of Fusion**: The best-performing model was the one that fused **LSTM stock embeddings** with **BERT Base text features**. This suggests that combining learned representations from both modalities is the most effective approach.
*   **Limitations of Daily Data**: The daily resolution of the data likely dilutes the signal from news, as any immediate market reactions are averaged out.

## Future Work

Future work could explore the following directions:

*   **Intra-day Data**: Using higher-frequency data (e.g., 1-minute or 5-minute intervals) to better capture the immediate impact of news.
*   **Advanced Fusion Models**: Experimenting with more sophisticated fusion mechanisms, such as attention or multi-modal transformers.
*   **Richer Text Representations**: Fine-tuning language models on financial news or using larger language models (LLMs) for feature extraction.

## Project Implementation

The project is structured as follows:

*   **`data/`**: Contains raw and processed data.
*   **`data_analysis/`**: Contains EDA notebooks and outputs.
*   **`dataset/`**: Contains various processed and aligned datasets.
*   **`noteboooks/`**: Contains notebooks for feature engineering and model development.
*   **`src/`**: Contains the main source code for data processing, feature extraction, model training, and evaluation.
*   **`results/`**: Contains model performance results and reports.
*   **`trained_models/`**: Contains saved model weights and scalers.

### Key Scripts:

*   **`src/data_processing/`**: Scripts for cleaning and processing raw data.
*   **`src/feature_extraction/`**: Scripts for extracting features from stock and text data.
*   **`src/model_pipes/`**: Scripts defining the model training and evaluation pipelines.
*   **`src/train_eval.py`**: Main script for running training and evaluation experiments.
*   **`src/config.py`**: Configuration file for experiments.

### To Reproduce:

1.  Install dependencies from `requirements.txt`.
2.  Run the data processing and feature engineering scripts in `noteboooks/`.
3.  Run the training and evaluation scripts using `src/train_eval.py`.
